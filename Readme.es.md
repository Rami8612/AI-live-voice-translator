# 🧠 AI Live Voice Translator

Traducción de voz en tiempo real con modelos de Inteligencia Artificial. Este proyecto permite transcribir, traducir y reproducir audio en vivo usando dos modos:

* **Pro (API)**: Usa la API de OpenAI para transcripción, traducción y TTS (voz Coral).
* **Offline**: Funciona 100% local usando `faster-whisper`, `transformers` y `pyttsx3`.

---

## 📁 Estructura del repositorio

```
AI-live-voice-translator/
├── live_translate_pro.py            # Traducción + voz Coral (OpenAI)
├── live_translate_offline.py       # Modo Offline: STT + traducción + voz local
├── requirements.txt                # Dependencias principales
├── .env                            # (no incluido) Clave OPENAI_API_KEY
├── utils/                          # Herramientas y tests
│   ├── detect_devices.py           # Lista entradas/salidas de audio disponibles
│   ├── test_openai.py              # Prueba TTS con OpenAI (voz Coral)
│   └── test_tts_windows.py         # Prueba TTS local (pyttsx3)
```

---

## 🧪 Requisitos

* Python 3.10 o superior
* Sistema operativo: **Windows** (modo TTS local requiere SAPI5)

### 🔌 Instalación de dependencias

```bash
pip install -r requirements.txt
```

---

## 🚀 Ejecución

### 1. Modo Pro (requiere API de OpenAI)

```bash
python live_translate_pro.py --device 0 --out_device 1 --threshold 0.01 --silence 0.5
```

### 2. Modo Offline (100% local)

```bash
python live_translate_offline.py --device 0 --out_device 1 --threshold 0.01 --silence 0.5
```

### 3. Ver entradas y salidas disponibles

```bash
python utils/detect_devices.py
```

---

## 💡 Archivos de prueba

* `utils/test_openai.py` → Prueba generación de voz Coral (OpenAI)
* `utils/test_tts_windows.py` → Prueba voz local Windows (pyttsx3)

---

## 📌 Características principales

* 🎤 Captura de audio en vivo (loopback)
* 🧠 Transcripción con Whisper (API o local)
* 🌐 Traducción en tiempo real (GPT-4o o HuggingFace)
* 🔊 Reproducción secuencial sin superposición

---

## 👤 Autor

**Rami8612**

---

## ⚖️ Licencia

Uso libre con fines educativos y personales. Se agradece la mención si compartes o modificas.

---

🔄 Proyecto en evolución. Se agradecen sugerencias, mejoras o estrellas en GitHub :)
