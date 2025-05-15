# 🧠 AI Live Voice Translator

Real-time voice translation using Artificial Intelligence models. This project enables live audio **transcription, translation, and playback**, offering two modes:

* **Pro (API)**: Uses OpenAI API for transcription, translation, and TTS (Coral voice).
* **Offline**: 100% local execution using `faster-whisper`, `transformers`, and `pyttsx3`.

---

## 📁 Repository Structure

```
AI-live-voice-translator/
├── live_translate_pro.py            # Translation + Coral voice (OpenAI)
├── live_translate_offline.py       # Offline mode: STT + translation + local TTS
├── requirements.txt                # Main dependencies
├── .env                            # (not included) Your OPENAI_API_KEY
├── utils/                          # Tools and tests
│   ├── detect_devices.py           # Lists available audio inputs/outputs
│   ├── test_openai.py              # Test OpenAI Coral voice
│   └── test_tts_windows.py         # Test local voice (pyttsx3)
```

---

## 🧪 Requirements

* Python 3.10 or later
* OS: **Windows** (local TTS requires SAPI5)

### 🔌 Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Pro Mode (requires OpenAI API)

```bash
python live_translate_pro.py --device 0 --out_device 1 --threshold 0.01 --silence 0.5
```

### 2. Offline Mode (100% local)

```bash
python live_translate_offline.py --device 0 --out_device 1 --threshold 0.01 --silence 0.5
```

### 3. List available devices

```bash
python utils/detect_devices.py
```

---

## 💡 Test Scripts

* `utils/test_openai.py` → Tests OpenAI Coral voice
* `utils/test_tts_windows.py` → Tests local Windows voice (pyttsx3)

---

## 📌 Key Features

* 🎤 Real-time audio capture (loopback)
* 🧠 Transcription via Whisper (API or local)
* 🌐 Real-time translation (GPT-4o or HuggingFace)
* 🔊 Sequential playback without overlap

---

## 👤 Author

**Rami8612**

---

## ⚖️ License

Free to use for educational and personal purposes. Attribution is appreciated if you share or modify it.

---

🔄 This project is evolving. Suggestions, improvements, and stars on GitHub are welcome!

👉 Also available in [Spanish](README.es.md)
