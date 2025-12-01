# Streamlit Jump Analyzer - app.py
# Requisitos: streamlit, opencv-python, mediapipe, numpy, pandas, matplotlib
# Instalar: pip install streamlit opencv-python mediapipe numpy pandas matplotlib

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import os
import math
from matplotlib import pyplot as plt

st.set_page_config(page_title="Jump Analyzer", layout="wide")

st.title("Jump Analyzer — Medición de saltos (Streamlit + MediaPipe)")
st.markdown("""
Esta app procesa un video (o webcam) y detecta saltos usando MediaPipe Pose.
Calcula el *tiempo de vuelo* (T) y estima la *altura del salto* con la fórmula física

\[ h = g * T^2 / 8 \]

Para una medición en centímetros la app pide tu altura real (cm) y la usa para convertir píxeles a cm.
""")

# Sidebar: opciones
with st.sidebar:
    st.header("Ajustes")
    user_name = st.text_input("Nombre (opcional)")
    user_height_cm = st.number_input("Tu altura real (cm) — usada para escala", min_value=100, max_value=250, value=175)
    detection_confidence = st.slider("Confianza mínima de detección (pose) ", 0.1, 0.99, 0.5)
    method_choice = st.selectbox("Método de estimación", ["Tiempo de vuelo (recomendado)", "Desplazamiento vertical de cadera"])
    show_plots = st.checkbox("Mostrar gráfico de trayectoria", value=True)
    process_button = st.button("Procesar video")

st.info("Sugerencia: sube un video de perfil donde se vea todo el cuerpo. Si no hay video, puedes usar la webcam (menos preciso en algunas máquinas).")

# Input: video upload or webcam
col1, col2 = st.columns([1,3])
with col1:
    st.subheader("Fuente")
    upload = st.file_uploader("Sube video (mp4, mov, avi)", type=["mp4","mov","avi","mkv"])
    use_webcam = st.checkbox("Usar webcam (si tu navegador lo permite)")

# helper: convert uploaded to temp file
def save_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read())
    tfile.flush()
    return tfile.name

# Pose detector init
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Processing function

def analyze_video(path, fps_override=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS)>0 else (fps_override if fps_override else 30)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT)>0 else None

    pose = mp_pose.Pose(min_detection_confidence=detection_confidence, min_tracking_confidence=0.5)

    ankle_left_y = []
    ankle_right_y = []
    hip_y = []
    head_y = []
    frame_idxs = []
    landmarks_frames = []

    pbar = st.progress(0)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        h, w, _ = frame.shape
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # indices from MediaPipe: 27=left_ankle, 28=right_ankle, 23=left_hip,24=right_hip,0=nose as proxy head
            la = lm[27]
            ra = lm[28]
            lh = lm[23]
            rh = lm[24]
            nose = lm[0]
            # y are normalized [0..1], convert to pixels (y increases downward)
            ankle_left_y.append(la.y * h if la.visibility>0.1 else None)
            ankle_right_y.append(ra.y * h if ra.visibility>0.1 else None)
            hip_y.append(((lh.y+rh.y)/2.0) * h if (lh.visibility>0.1 or rh.visibility>0.1) else None)
            head_y.append(nose.y * h if nose.visibility>0.1 else None)
            landmarks_frames.append(results.pose_landmarks)
        else:
            ankle_left_y.append(None)
            ankle_right_y.append(None)
            hip_y.append(None)
            head_y.append(None)
            landmarks_frames.append(None)

        frame_idxs.append(i-1)
        if frame_count:
            pbar.progress(min(1.0, i/frame_count))
        else:
            pbar.progress(min(1.0, i/300))

    cap.release()
    pose.close()
    pbar.empty()

    df = pd.DataFrame({
        'frame': frame_idxs,
        'ankle_left_y': ankle_left_y,
        'ankle_right_y': ankle_right_y,
        'hip_y': hip_y,
        'head_y': head_y,
    })
    return df, fps, landmarks_frames


# Detect baseline and flight frames

def detect_jump_frames(df, fps):
    # compute ankle median baseline using first 30 frames with valid ankles
    ankles_mean = df[['ankle_left_y','ankle_right_y']].mean(axis=1)
    valid_initial = ankles_mean.dropna().head(30)
    if len(valid_initial) < 5:
        st.warning("No se detectaron suficientes frames iniciales válidos para calcular baseline. Asegúrate de que el video muestre a la persona de pie al inicio.")
        baseline = ankles_mean.median()
    else:
        baseline = valid_initial.median()

    # threshold in pixels
    delta_px = 0.03 * baseline if baseline else 20

    # consider "off ground" when both ankles are at least delta_px above baseline (i.e., smaller y)
    off_ground = (df['ankle_left_y'] < (baseline - delta_px)) & (df['ankle_right_y'] < (baseline - delta_px))

    # find contiguous segments where off_ground == True
    off_idx = np.where(off_ground.fillna(False).to_numpy())[0]
    if len(off_idx) == 0:
        return None, None, baseline

    # pick the longest continuous block as the jump
    groups = np.split(off_idx, np.where(np.diff(off_idx) != 1)[0]+1)
    longest = max(groups, key=len)
    takeoff_frame = int(longest[0])
    landing_frame = int(longest[-1])

    flight_frames = landing_frame - takeoff_frame + 1
    flight_time = flight_frames / fps

    return takeoff_frame, landing_frame, flight_time, baseline


# Height from flight time (physics) and from pixel displacement

def flight_time_to_height(T):
    g = 9.81
    return g * (T**2) / 8.0


def pixel_disp_to_cm(pixel_disp, pixel_person_height, real_person_height_cm):
    # scale factor: cm per pixel
    scale = real_person_height_cm / pixel_person_height
    return pixel_disp * scale


# Main run
if (upload is not None) or use_webcam:
    if upload is not None:
        video_path = save_uploaded_file(upload)
        st.success(f"Video guardado temporalmente: {os.path.basename(video_path)}")
    else:
        video_path = None

    if process_button:
        if use_webcam and video_path is None:
            st.warning("La opción webcam se intentará usar si el navegador lo permite — pero en Streamlit ventana puede no soportar captura de cámara en todos los hosts. Si falla, sube un video.")

        if video_path is None and not use_webcam:
            st.error("Sube un video o activa la webcam.")
        else:
            if use_webcam:
                # try capture from default camera
                tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tmpname = tmpfile.name
                tmpfile.close()
                st.info("Grabando 6 segundos desde la webcam... mantente en perfil. (Si tu instancia no admite webcam, falla)")
                cam = cv2.VideoCapture(0)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(tmpname, fourcc, 20.0, (640,480))
                frames_to_record = 6 * 20
                for i in range(frames_to_record):
                    ret, frame = cam.read()
                    if not ret:
                        break
                    out.write(frame)
                cam.release(); out.release()
                video_path = tmpname

            st.info("Analizando video — esto puede tardar según la duración y el CPU")
            df, fps, landmarks = analyze_video(video_path)

            # estimate pixel person height as head to ankle median
            pixel_person_height = (df['head_y'] - df[['ankle_left_y','ankle_right_y']].min(axis=1)).dropna().median()
            if np.isnan(pixel_person_height) or pixel_person_height<=0:
                st.warning("No se pudo estimar la altura en píxeles. Asegúrate de que el video muestre la persona completa.")
                pixel_person_height = None

            detect_res = detect_jump_frames(df, fps)
            if detect_res is None or detect_res[0] is None:
                st.error("No se detectó un salto claro. Prueba con un video más claro o toma el perfil completo.")
            else:
                takeoff_frame, landing_frame, flight_time, baseline = detect_res
                h_flight_m = flight_time_to_height(flight_time)
                h_flight_cm = h_flight_m * 100.0

                # hip displacement method
                # baseline hip y = median of hip y before takeoff
                hip_series = df['hip_y']
                hip_baseline = hip_series[:takeoff_frame].dropna().median() if takeoff_frame>0 else hip_series.dropna().median()
                hip_apex = hip_series[takeoff_frame:landing_frame+1].dropna().min()
                hip_disp_px = hip_baseline - hip_apex if (not math.isnan(hip_baseline) and not math.isnan(hip_apex)) else None
                hip_disp_cm = None
                if pixel_person_height:
                    hip_disp_cm = pixel_disp_to_cm(hip_disp_px, pixel_person_height, user_height_cm) if hip_disp_px is not None else None

                st.subheader("Resultados")
                st.write(f"FPS detectado: {fps:.1f}")
                st.write(f"Frames: takeoff={takeoff_frame}, landing={landing_frame}, frames en vuelo={landing_frame-takeoff_frame+1}")
                st.write(f"Tiempo de vuelo (T): {flight_time:.3f} s")
                st.write(f"Altura estimada por tiempo de vuelo: {h_flight_cm:.2f} cm")

                if hip_disp_cm is not None:
                    st.write(f"Desplazamiento vertical de cadera (estimado): {hip_disp_cm:.2f} cm")

                # show sample frame with pose
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, takeoff_frame-5))
                ret, sample = cap.read()
                if ret:
                    sample_rgb = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
                    # draw pose for the frame nearest to apex
                    apex_frame = int((takeoff_frame+landing_frame)//2)
                    if landmarks[apex_frame] is not None:
                        mp_drawing.draw_landmarks(sample_rgb, landmarks[apex_frame], mp_pose.POSE_CONNECTIONS)
                    st.image(sample_rgb, caption="Frame de referencia (con landmarks)")
                cap.release()

                # plot hip y and ankles
                if show_plots:
                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.plot(df['frame'], df['hip_y'], label='hip_y')
                    ax.plot(df['frame'], df['ankle_left_y'], label='ankle_left_y', alpha=0.7)
                    ax.plot(df['frame'], df['ankle_right_y'], label='ankle_right_y', alpha=0.7)
                    ax.invert_yaxis()
                    ax.axvline(takeoff_frame, color='green', linestyle='--', label='takeoff')
                    ax.axvline(landing_frame, color='red', linestyle='--', label='landing')
                    ax.set_xlabel('Frame')
                    ax.set_ylabel('Pixels (y)')
                    ax.legend()
                    st.pyplot(fig)

                # prepare downloadable CSV
                df['off_ground'] = ((df['ankle_left_y'] < (baseline - 0.03*baseline)) & (df['ankle_right_y'] < (baseline - 0.03*baseline)))
                csv = df.to_csv(index=False)
                st.download_button("Descargar datos (CSV)", data=csv, file_name=f"jump_data_{user_name or 'user'}.csv")

                st.success("Análisis completado.")

else:
    st.warning("Sube un video o activa la webcam en la barra lateral para comenzar.")

st.markdown("---")
st.caption("Notas: La estimación por tiempo de vuelo es más fiable que la conversión de pixeles si el video permite detectar con claridad la separación del suelo. Para mayor precisión añade una referencia de escala (por ejemplo, una regla visible) o usa una cámara con FPS conocido.")
