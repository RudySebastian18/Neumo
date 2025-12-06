import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import io

st.set_page_config(page_title="NeumoAudio – Prototipo", layout="centered")

st.title("🔬 NeumoAudio – Prototipo de Clasificación de Sonidos Pulmonares")
st.write("Sube un archivo de audio en formato **WAV** para generar su espectrograma y obtener una predicción preliminar.")

# ------------------------------
# Cargar modelo (Placeholder)
# ------------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("modelo_cnn.h5")
        return model
    except:
        return None  # No hay modelo todavía

model = load_model()

# ------------------------------
# Función para convertir WAV → espectrograma
# ------------------------------
def audio_to_melspectrogram(file_bytes):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000)

    # Generar Mel-Spectrograma
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


# ------------------------------
# Subida del audio
# ------------------------------
uploaded_file = st.file_uploader("📁 Cargar archivo WAV", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    file_bytes = uploaded_file.read()

    # Convertir a Mel-Spectrograma
    mel_spec = audio_to_melspectrogram(file_bytes)

    st.subheader("📊 Espectrograma de Mel")
    fig, ax = plt.subplots(figsize=(6, 4))
    librosa.display.specshow(mel_spec, sr=16000, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title("Mel-Spectrograma")
    st.pyplot(fig)

    # ------------------------------
    # Predicción del modelo (si existe)
    # ------------------------------
    st.subheader("🤖 Predicción del Modelo")

    if model is None:
        st.warning("⚠ No se encontró un modelo entrenado (modelo_cnn.h5). Se usará una predicción simulada.")

        # PREDICCIÓN FICTICIA
        simulated_probs = {
            "Normal": np.random.uniform(0.4, 0.9),
            "Sibilancias": np.random.uniform(0.1, 0.7),
            "Crepitantes": np.random.uniform(0.1, 0.7)
        }

        predicted_label = max(simulated_probs, key=simulated_probs.get)

        st.metric("Resultado", predicted_label)
        st.json(simulated_probs)

    else:
        # Preparar input para el modelo real
        input_data = np.expand_dims(mel_spec, axis=(0, -1))

        preds = model.predict(input_data)[0]

        clases = ["Normal", "Sibilancias", "Crepitantes"]
        resultado = clases[np.argmax(preds)]

        st.metric("Resultado", resultado)
        st.json({clases[i]: float(preds[i]) for i in range(len(clases))})

