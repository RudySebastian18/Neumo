# app.py - NeumoAudio Streamlit app (Prototipo)
# Ejecutar: streamlit run app.py

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="NeumoAudio", layout="centered")

# -------------------------
# Config / Parameters
# -------------------------
SR = 16000                # target sample rate
N_MELS = 128              # Mel bands
DURATION = 6.0            # seconds (for padding/truncation)
MODEL_PATH_H5 = "models/model.h5"           # ruta por defecto para .h5
MODEL_PATH_SAVED = "models/saved_model"     # ruta por defecto para SavedModel dir

CLASS_NAMES = ["Normal", "Crackles", "Wheezes", "Mixed"]  # adapta a tu dataset

# -------------------------
# Utility functions
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Intenta cargar modelo .h5 o SavedModel. Devuelve None si no existe."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH_H5)
        st.sidebar.success("Modelo cargado desde: " + MODEL_PATH_H5)
        return model
    except Exception:
        try:
            model = tf.keras.models.load_model(MODEL_PATH_SAVED)
            st.sidebar.success("Modelo cargado desde: " + MODEL_PATH_SAVED)
            return model
        except Exception:
            st.sidebar.warning("No se encontró un modelo en 'models/'. Usando modo DEMO (sin modelo).")
            return None

def load_audio(wav_bytes):
    # librosa accepts file-like objects: use BytesIO
    y, sr = librosa.load(io.BytesIO(wav_bytes), sr=None)
    return y, sr

def preprocess_audio(y, sr, target_sr=SR, duration=DURATION):
    # resample
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    # mono
    if y.ndim > 1:
        y = librosa.to_mono(y)
    # pad / truncate to fixed duration
    max_len = int(target_sr * duration)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode="constant")
    else:
        y = y[:max_len]
    return y, sr

def make_mel_spectrogram(y, sr, n_mels=N_MELS):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def plot_spectrogram(S_db):
    fig, ax = plt.subplots(figsize=(6, 3))
    img = librosa.display.specshow(S_db, x_axis="time", y_axis="mel", sr=SR, ax=ax)
    ax.set(title="Espectrograma Mel (dB)")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    return fig

def model_input_from_spectrogram(S_db):
    """Convierte espectrograma a la entrada esperada por la red.
    Ajusta según arquitectura: ejemplo: (H, W, 1) normalizado."""
    x = S_db.astype(np.float32)
    # normalizar (mín-max) — modifica según tu pipeline de entrenamiento
    x = (x - x.min()) / (x.max() - x.min() + 1e-9)
    x = np.expand_dims(x, axis=-1)      # (H, W, 1)
    x = np.expand_dims(x, axis=0)       # (1, H, W, 1)
    return x

def predict(model, S_db):
    if model is None:
        # Demo: simple heuristic por energía en bandas (solo para demo)
        avg = S_db.mean()
        if avg < -30:
            return {"Normal": 0.7, "Crackles": 0.1, "Wheezes": 0.1, "Mixed": 0.1}
        else:
            return {"Normal": 0.1, "Crackles": 0.4, "Wheezes": 0.4, "Mixed": 0.1}
    x = model_input_from_spectrogram(S_db)
    probs = model.predict(x)[0]
    # si tu modelo devuelve logits, aplica softmax:
    if probs.sum() != 1.0:
        probs = tf.nn.softmax(probs).numpy()
    return {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}

# -------------------------
# App UI
# -------------------------
st.title("NeumoAudio — Cribado de enfermedades respiratorias")
st.markdown(
    """
    **NeumoAudio** es un prototipo open-source para clasificación automática de sonidos pulmonares.
    Suba un archivo WAV (estetoscopio digital) y obtenga un resultado de riesgo.
    """
)

with st.sidebar:
    st.header("Configuración")
    st.markdown("Asegúrate de colocar tu modelo en `models/model.h5` o `models/saved_model/`")
    model = load_model()
    st.markdown("---")
    st.caption("Proyecto: NeumoAudio — Prototipo para pasantías Prociencia")

uploaded_file = st.file_uploader("Sube un archivo WAV (estetoscopio)", type=["wav", "WAV", "wave"])

col1, col2 = st.columns([1, 1])

if uploaded_file is not None:
    wav_bytes = uploaded_file.read()
    st.sidebar.write(f"Archivo: {uploaded_file.name} ({len(wav_bytes)//1024} KB)")
    try:
        y, sr = load_audio(wav_bytes)
    except Exception as e:
        st.error("No se pudo leer el archivo WAV. Verifica el formato.")
        st.exception(e)
        st.stop()

    y_proc, sr_proc = preprocess_audio(y, sr)
    S_db = make_mel_spectrogram(y_proc, sr_proc)

    with col1:
        st.subheader("Espectrograma")
        fig = plot_spectrogram(S_db)
        st.pyplot(fig)

    with col2:
        st.subheader("Audio")
        st.audio(wav_bytes)
        st.write(f"Duración (s): {len(y)/sr:.2f} / Procesado: {DURATION}s, SR={SR}")

    # Botón de inferencia
    if st.button("Analizar (Clasificar)"):
        with st.spinner("Realizando inferencia..."):
            try:
                result = predict(model, S_db)
                # ordena por prob desc
                sorted_res = sorted(result.items(), key=lambda x: x[1], reverse=True)
                st.success("Inferencia completada")
                st.table(
                    {"Clase": [r[0] for r in sorted_res], "Probabilidad": [f"{r[1]*100:.1f} %" for r in sorted_res]}
                )
                top_class, top_prob = sorted_res[0]
                # Semáforo simple
                if top_class == "Normal":
                    color = "🟢"
                elif top_class == "Crackles" and top_prob > 0.6:
                    color = "🔴"
                elif top_class in ["Wheezes", "Mixed"] and top_prob > 0.5:
                    color = "🔴"
                else:
                    color = "🟠"
                st.markdown(f"### Resultado principal: {color} **{top_class}** ({top_prob*100:.1f}%)")
                st.info("Este sistema es una herramienta de apoyo — no reemplaza el diagnóstico clínico.")
            except Exception as e:
                st.error("Error durante la inferencia.")
                st.exception(e)
else:
    st.info("Sube un archivo WAV para comenzar. Puedes usar muestras de ICBHI o grabaciones de tu equipo.")

st.markdown("---")
st.caption("NeumoAudio — prototipo con software libre. Implementación offline con Docker recomendada.")
