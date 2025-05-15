"""
test_tts_windows.py

Convierte texto a voz en español usando pyttsx3 (motor SAPI5 de Windows),
guarda el resultado en un archivo WAV a 16 kHz mono y lo reproduce.

Requisitos:
- pyttsx3
- soundfile
- sounddevice

Autor: Rami8612
Fecha: 2025-05-15
"""

import os
import tempfile
import pyttsx3
import soundfile as sf
import sounddevice as sd
from scipy.signal import resample


FS_TARGET = 16000  # Frecuencia objetivo para salida (Whisper-friendly)


def generar_voz_windows(texto: str, archivo_salida: str = "tts_windows.wav") -> str:
    """Genera un archivo de voz usando pyttsx3 en español."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 170)

    # Buscar voz en español
    for v in engine.getProperty('voices'):
        if 'spanish' in v.name.lower():
            engine.setProperty('voice', v.id)
            break

    # Generar archivo temporal
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    engine.save_to_file(texto, temp_path)
    engine.runAndWait()

    # Leer y convertir a 16 kHz mono
    data, sr = sf.read(temp_path, dtype='float32')
    os.remove(temp_path)
    if sr != FS_TARGET:
        data = resample(data, int(len(data) * FS_TARGET / sr))
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Guardar archivo final
    sf.write(archivo_salida, data, FS_TARGET)
    return archivo_salida


def reproducir_audio(path: str):
    """Reproduce un archivo WAV con sounddevice."""
    data, sr = sf.read(path, dtype='float32')
    sd.play(data, sr)
    sd.wait()


if __name__ == "__main__":
    texto = "Esta es una prueba de voz generada localmente con pyttsx3 en español."
    print(f"🎤 Generando voz: {texto}")
    wav_path = generar_voz_windows(texto)
    print(f"✅ Archivo generado: {wav_path}")
    reproducir_audio(wav_path)
    print("🔊 Reproducción finalizada.")
