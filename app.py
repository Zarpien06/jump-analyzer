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

st.title("Jump Analyzer — Medición de saltos")
st.markdown("""
Esta app procesa un video y detecta saltos.
Calcula el *tiempo de vuelo* (T) y estima la *altura del salto* con:

\\[ h = g * T^2 / 8 \\]

Para convertir a centímetros se pide tu altura real.
""")

# Sidebar
with st.sidebar:
    st.header("Ajustes")
    user_name = st.text_input("Nombre (opcional)")
    user_height_cm = st.number_input("Tu altura real (cm)", min_value=100, max_value=250, value=175)
    detection_confidence = st.slider("Confianza Pose", 0.1, 0.99, 0.5)
    show_plots = st.checkbox("Mostrar gráfico", value=True)
    process_button = st.button("Procesar video")

st.info("Sugerencia: video de perfil y cuerpo completo.")

# Entrada de video
col1, col2 = st.columns([1, 3])
with col1:
    upload = st.file_uploader("Sube video", type=["mp4", "mov", "avi", "mkv"])

def save_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read())
    return tfile.name

# Análisis del video usando MediaPipe
def analyze_video(path, fps_override=None):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else (fps_override or 30)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ankle_left_y = []
    ankle_right_y = []
    hip_y = []
    head_y = []
    frame_idxs = []

    pbar = st.progress(0)

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        i += 1

        # Convertir BGR a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        h, w, _ = frame.shape

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # MediaPipe landmarks indices:
            # 27 = left_ankle, 28 = right_ankle
            # 23 = left_hip, 24 = right_hip
            # 0 = nose
            
            la = landmarks[27].y * h if landmarks[27].visibility > 0.5 else None
            ra = landmarks[28].y * h if landmarks[28].visibility > 0.5 else None
            lh = landmarks[23].y * h if landmarks[23].visibility > 0.5 else None
            rh = landmarks[24].y * h if landmarks[24].visibility > 0.5 else None
            nose = landmarks[0].y * h if landmarks[0].visibility > 0.5 else None
            
            ankle_left_y.append(la)
            ankle_right_y.append(ra)
            
            if lh is not None and rh is not None:
                hip_y.append((lh + rh) / 2)
            else:
                hip_y.append(None)
            
            head_y.append(nose)
        else:
            ankle_left_y.append(None)
            ankle_right_y.append(None)
            hip_y.append(None)
            head_y.append(None)
        
        frame_idxs.append(i)

        if frame_count:
            pbar.progress(min(1.0, i / frame_count))
        else:
            pbar.progress(min(1.0, i / 300))

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

    return df, fps

# Detect jump frames
def detect_jump_frames(df, fps):
    ankles_mean = df[['ankle_left_y', 'ankle_right_y']].mean(axis=1)
    baseline = ankles_mean.dropna().head(30).median()

    delta_px = 0.03 * baseline

    off_ground = (df['ankle_left_y'] < (baseline - delta_px)) & (df['ankle_right_y'] < (baseline - delta_px))

    idx = np.where(off_ground.fillna(False))[0]
    if len(idx) == 0:
        return None

    groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    longest = max(groups, key=len)

    t0, t1 = longest[0], longest[-1]
    flight_time = (t1 - t0 + 1) / fps

    return t0, t1, flight_time, baseline

def flight_time_to_height(T):
    return 9.81 * T * T / 8

def pixel_disp_to_cm(px, px_person_height, real_cm):
    return (px / px_person_height) * real_cm

# MAIN
if upload is not None:
    video_path = save_uploaded_file(upload)
    st.success("Video cargado.")

    if process_button:
        st.info("Procesando…")
        df, fps = analyze_video(video_path)

        # Estimar altura en pixeles
        pixel_person_height = (df['head_y'] - df[['ankle_left_y', 'ankle_right_y']].min(axis=1)).dropna().median()

        detect_res = detect_jump_frames(df, fps)
        if detect_res is None:
            st.error("No se detectó salto.")
        else:
            t0, t1, T, baseline = detect_res
            h_cm = flight_time_to_height(T) * 100

            st.subheader("Resultados")
            st.write(f"FPS: {fps:.1f}")
            st.write(f"Tiempo de vuelo: {T:.3f} s")
            st.write(f"Altura estimada: **{h_cm:.2f} cm**")

            # Pixel displacement method
            hip = df["hip_y"]
            hip_base = hip[:t0].dropna().median()
            hip_apex = hip[t0:t1+1].dropna().min()
            hip_disp = hip_base - hip_apex

            if pixel_person_height and pixel_person_height > 0:
                hip_cm = pixel_disp_to_cm(hip_disp, pixel_person_height, user_height_cm)
                st.write(f"Altura por desplazamiento de cadera: {hip_cm:.2f} cm")

            # Graficar
            if show_plots:
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(df['frame'], df['hip_y'], label='Cadera', linewidth=2)
                ax.plot(df['frame'], df['ankle_left_y'], label='Tobillo Izq', alpha=0.7)
                ax.plot(df['frame'], df['ankle_right_y'], label='Tobillo Der', alpha=0.7)
                ax.axvline(t0, color='green', linestyle='--', label='Inicio salto')
                ax.axvline(t1, color='red', linestyle='--', label='Fin salto')
                ax.invert_yaxis()
                ax.legend()
                ax.set_xlabel('Frame')
                ax.set_ylabel('Posición Y (pixeles)')
                ax.set_title('Trayectoria del salto')
                st.pyplot(fig)

            csv = df.to_csv(index=False)
            st.download_button("Descargar CSV", csv, "jump_data.csv")
            
        # Limpiar archivo temporal
        try:
            os.unlink(video_path)
        except:
            pass

else:
    st.warning("Sube un video para comenzar.")

st.markdown("---")
st.caption("Funciona con MediaPipe — Compatible con Python 3.13")
