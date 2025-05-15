"""
test_openai.py

Genera una frase de prueba usando el TTS de OpenAI (modelo gpt-4o-mini-tts),
guarda el audio en un archivo WAV a 16 kHz mono y lo reproduce localmente.

Requisitos:
- openai >=1.0.0
- python-dotenv
- soundfile
- sounddevice

Autor: Rami8612
Fecha: 2025-05-15
"""

import os
import io
from dotenv import load_dotenv
import openai
import soundfile as sf
import sounddevice as sd


FS_TARGET = 16000  # Frecuencia objetivo para reproducción


def cargar_api_key():
    """Carga la clave desde un archivo .env en el directorio raíz."""
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(dotenv_path=env_path)
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("❌ No se encontró OPENAI_API_KEY en el archivo .env")
    return key


def generar_voz_openai(texto: str, archivo_salida: str = "openai_tts.wav") -> str:
    """Genera voz con OpenAI TTS y guarda el WAV."""
    respuesta = openai.audio.speech.create(
        model="gpt-4o-mini-tts",  # o "tts-1-hd"
        voice="coral",            # alloy | nova | shimmer | echo | fable
        input=texto,
        response_format="wav"
    )
    wav_bytes = respuesta.read()
    with open(archivo_salida, "wb") as f:
        f.write(wav_bytes)
    return archivo_salida


def reproducir_audio(path: str):
    """Reproduce un archivo WAV con sounddevice."""
    data, sr = sf.read(path, dtype='float32')
    sd.play(data, sr)
    sd.wait()


if __name__ == "__main__":
    openai.api_key = cargar_api_key()
    texto = "Esta es una prueba de síntesis de voz utilizando la API de OpenAI."
    print(f"🎤 Solicitando TTS: {texto}")
    wav_path = generar_voz_openai(texto)
    print(f"✅ Archivo generado: {wav_path}")
    reproducir_audio(wav_path)
    print("🔊 Reproducción finalizada.")
