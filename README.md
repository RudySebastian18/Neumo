# Neumo
El presente proyecto propone el desarrollo de una plataforma de inteligencia artificial y software libre para el cribado automático de enfermedades respiratorias. A partir de la entrada de grabaciones de audio pulmonar en formato WAV 
# NeumoAudio - Prototipo
Prototipo para cribado automático de sonidos pulmonares usando software libre.

## Estructura
- app.py                # Streamlit app
- models/               # colocar model.h5 o carpeta saved_model/
- requirements.txt

## Uso local (desarrollo)
1. Crear entorno e instalar:
   pip install -r requirements.txt
2. Ejecutar:
   streamlit run app.py

## Docker (ejemplo)
Construir imagen y ejecutar (ver Dockerfile ejemplo en la carpeta `docker/`).

## Nota
Este prototipo es una herramienta de apoyo. La validación clínica y aprobación regulatoria son requeridas antes de cualquier uso médico.
