import streamlit as st
import cv2
import os
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import math
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from openai import OpenAI
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Jump Analyzer IA", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("Jump Analyzer — IA Coach, Técnica y Predicción")
st.markdown("""
Procesa un video, detecta saltos, analiza tu técnica con IA y te da recomendaciones profesionales.

Incluye:
- **Tiempo de vuelo**
- **Altura del salto (fórmula física)**
- **Altura por desplazamiento**
- **Análisis técnico con ángulos**
- **Evaluación inteligente (IA Coach GPT)**
- **Predicción de tu progreso**
""")

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("Ajustes")
    user_name = st.text_input("Nombre (opcional)")
    user_height_cm = st.number_input("Tu altura (cm)", min_value=100, max_value=250, value=175)
    detection_confidence = st.slider("Confianza Pose", 0.1, 0.99, 0.5)
    show_ai = st.checkbox("Activar IA Coach", True)
    show_technique = st.checkbox("Analizar técnica", True)
    show_predict = st.checkbox("Predecir progreso", True)
    show_plots = st.checkbox("Mostrar gráficos", True)
    process_button = st.button("Procesar video")

st.info("Sugerencia: video de perfil y cuerpo completo.")

col1, col2 = st.columns([1, 3])
with col1:
    upload = st.file_uploader("Sube video", type=["mp4", "mov", "avi", "mkv"])

def save_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read())
    return tfile.name

# ------------------ VIDEO ANALYSIS ------------------
def analyze_video(path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    data = {
        'frame': [], 'ankle_left_y': [], 'ankle_right_y': [],
        'hip_y': [], 'knee_angle': [], 'hip_angle': [], 'head_y': []
    }

    pbar = st.progress(0)
    frame_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_i += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        h, w, _ = frame.shape

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            def y(id): return lm[id].y * h if lm[id].visibility > 0.5 else None
            def xy(id): return (lm[id].x * w, lm[id].y * h) if lm[id].visibility > 0.5 else None

            la = y(27); ra = y(28)
            lh = y(23); rh = y(24)
            nose = y(0)

            hip_y = (lh + rh) / 2 if lh and rh else None

            # Angles
            hip_point = xy(24)
            knee_point = xy(26)
            ankle_point = xy(28)

            def angle(a, b, c):
                if None in [a, b, c]:
                    return None
                ba = np.array(a) - np.array(b)
                bc = np.array(c) - np.array(b)
                cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

            knee_angle = angle(hip_point, knee_point, ankle_point)

            # Hip angle (flexión)
            shoulder = xy(12)
            hip_angle = angle(shoulder, hip_point, knee_point)

        else:
            la = ra = hip_y = knee_angle = hip_angle = nose = None

        data['frame'].append(frame_i)
        data['ankle_left_y'].append(la)
        data['ankle_right_y'].append(ra)
        data['hip_y'].append(hip_y)
        data['knee_angle'].append(knee_angle)
        data['hip_angle'].append(hip_angle)
        data['head_y'].append(nose)

        pbar.progress(min(frame_i / frame_count, 1))

    cap.release()
    pose.close()
    pbar.empty()

    return pd.DataFrame(data), fps

# ------------------ JUMP DETECTION ------------------
def detect_jump_frames(df, fps):
    ankles = df[['ankle_left_y', 'ankle_right_y']].mean(axis=1)
    baseline = ankles.dropna().head(40).median()
    delta = 0.04 * baseline

    off_ground = (df['ankle_left_y'] < baseline - delta) & \
                 (df['ankle_right_y'] < baseline - delta)

    idx = np.where(off_ground.fillna(False))[0]
    if len(idx) == 0:
        return None

    groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    jump = max(groups, key=len)

    t0, t1 = jump[0], jump[-1]
    T = (t1 - t0 + 1) / fps

    return t0, t1, T, baseline

def flight_height(T): return 9.81 * T * T / 8

def pixel_to_cm(px, px_height, cm_height):
    return (px / px_height) * cm_height

# ------------------ MAIN ------------------
if upload:
    video_path = save_uploaded_file(upload)
    st.success("Video cargado.")

    if process_button:
        st.info("Procesando…")
        df, fps = analyze_video(video_path)

        # Body height in pixels
        px_h = (df['head_y'] - df[['ankle_left_y', 'ankle_right_y']].min(axis=1)).dropna().median()

        res = detect_jump_frames(df, fps)
        if not res:
            st.error("No se detectó un salto claro.")
        else:
            t0, t1, T, baseline = res
            h_cm = flight_height(T) * 100

            hip = df["hip_y"]
            base = hip[:t0].dropna().median()
            apex = hip[t0:t1+1].dropna().min()
            disp = base - apex

            real_cm = pixel_to_cm(disp, px_h, user_height_cm)

            st.header("Resultados")
            st.write(f"Tiempo de vuelo: **{T:.3f} s**")
            st.write(f"Altura por fórmula física: **{h_cm:.1f} cm**")
            st.write(f"Altura por cadera: **{real_cm:.1f} cm**")

            # ------------------ GRÁFICO ------------------
            if show_plots:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df["frame"], df["hip_y"])
                ax.axvline(t0, color="green")
                ax.axvline(t1, color="red")
                ax.invert_yaxis()
                st.pyplot(fig)

            # ------------------ IA COACH ------------------
            if show_ai:
                prompt = f"""
                Actúa como un entrenador profesional de salto vertical.

                Datos del salto:
                - Tiempo de vuelo: {T:.3f}s
                - Altura física: {h_cm:.1f} cm
                - Altura por cadera: {real_cm:.1f} cm
                - Altura del usuario: {user_height_cm} cm
                - Ángulo de rodilla mínimo: {df['knee_angle'].min()}
                - Ángulo de cadera mínimo: {df['hip_angle'].min()}
                - Simetría tobillos: diferencia promedio = {float(df['ankle_left_y'].median() - df['ankle_right_y'].median()):.2f}px

                Evalúa:
                - técnica del salto
                - profundidad del squat
                - explosividad
                - estabilidad lateral
                - simetría
                - recomendaciones avanzadas
                """

                with st.spinner("Generando análisis IA…"):
                    ai = client.chat.completions.create(
                        model="gpt-4.1",
                        messages=[{"role": "user", "content": prompt}]
                    )

                st.subheader("🧠 IA Coach")
                st.write(ai.choices[0].message["content"])

            # ------------------ PREDICCIÓN ------------------
            if show_predict:
                if "jump_history.csv" in os.listdir():
                    old = pd.read_csv("jump_history.csv")
                else:
                    old = pd.DataFrame(columns=["jump"])

                new = pd.DataFrame({"jump": [real_cm]})
                total = pd.concat([old, new], ignore_index=True)
                total.to_csv("jump_history.csv", index=False)

                if len(total) > 2:
                    X = np.arange(len(total)).reshape(-1, 1)
                    y = total["jump"].values
                    model = LinearRegression().fit(X, y)
                    future = model.predict([[len(total)]])[0]

                    st.subheader("📈 Predicción")
                    st.write(f"Tu próximo salto estimado: **{future:.1f} cm**")

                else:
                    st.info("Aún no hay suficientes saltos para predecir.")

        os.unlink(video_path)
else:
    st.warning("Sube un video para comenzar.")
