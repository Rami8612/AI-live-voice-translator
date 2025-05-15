"""
live_translate_offline.py — Traducción en vivo 100% offline
------------------------------------------------------------

Este script permite capturar audio del sistema, transcribirlo localmente con Whisper
(faster-whisper), traducirlo al español con un modelo Hugging Face (opus-mt-en-es) 
y convertirlo a voz usando el motor TTS de Windows (pyttsx3), todo de forma secuencial 
y completamente offline.

Características:
- Captura audio por loopback usando soundcard.
- Segmenta frases mediante VAD simple por RMS y silencio.
- Transcribe con Whisper en CPU (formato int8 para mayor eficiencia).
- Traduce con transformer local de HuggingFace usando PyTorch.
- Sintetiza la voz en español con pyttsx3 y reproduce por altavoz seleccionado.
- Permite elegir dispositivo de entrada y salida mediante CLI.

Uso:
    python live_translate_offline.py --device 0 --out_device 1 --threshold 0.01 --silence 0.5

Requisitos:
    - Python 3.10+
    - soundcard, faster-whisper, transformers, pyttsx3, scipy, torch, colorama, soundfile

Autor:
    Rami8612

Repositorio:
    https://github.com/Rami8612/
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import tempfile
import argparse
import queue
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import soundcard as sc
import soundfile as sf
from colorama import init, Fore, Style
from scipy.signal import resample
import pythoncom
import pyttsx3
from faster_whisper import WhisperModel
from transformers import pipeline
import torch

# Configuración global
init(autoreset=True)
FS = 16000
CHANNELS = 2
FRAME_MS = 100

# Carga modelo Whisper local
print("Cargando modelo Whisper local (CPU)...", flush=True)
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# Carga pipeline de traducción
print("Cargando pipeline de traducción en→es (PyTorch)...", flush=True)
use_cuda = torch.cuda.is_available()
translator = pipeline(
    "translation_en_to_es",
    model="Helsinki-NLP/opus-mt-en-es",
    framework="pt",
    device=0 if use_cuda else -1
)

# Argumentos CLI
parser = argparse.ArgumentParser('Live Translate 100% Offline')
parser.add_argument('--device',     type=int,   default=1,    help='Índice loopback IN')
parser.add_argument('--out_name',   type=str,   default='',   help='Substring altavoz OUT')
parser.add_argument('--out_device', type=int,   default=-1,   help='Índice altavoz OUT si no usa --out_name')
parser.add_argument('--threshold',  type=float, default=0.01, help='Umbral RMS para VAD')
parser.add_argument('--silence',    type=float, default=0.5,  help='Silencio [s] para segmentar')
args = parser.parse_args()
THRESH = args.threshold
SILENCE_SEC = args.silence

# Selección de dispositivos
mics = sc.all_microphones(include_loopback=True)
if not (0 <= args.device < len(mics)):
    raise RuntimeError('Índice device IN fuera de rango')
mic = mics[args.device]
speakers = sc.all_speakers()
if args.out_name:
    spk = next((s for s in speakers if args.out_name.lower() in s.name.lower()), None)
    if not spk:
        raise RuntimeError(f"Altavoz conteniendo '{args.out_name}' no encontrado")
else:
    spk = speakers[args.out_device] if args.out_device >= 0 else sc.default_speaker()
print(Fore.CYAN + f"IN  : {mic.name}\nOUT : {spk.name}" + Style.RESET_ALL)
print(f"RMS ≥ {THRESH}, silencio ≥ {SILENCE_SEC}s — segmentando frases\n")

# VAD setup
FRAME_SAMPLES = int(FS * FRAME_MS / 1000)
SILENCE_FRAMES = int(SILENCE_SEC * 1000 / FRAME_MS)

# Colas
audio_q = queue.Queue()
text_q = queue.Queue()
play_q = queue.Queue()
last_line = ''

# Hilo TTS local
def tts_worker():
    pythoncom.CoInitialize()
    engine = pyttsx3.init(driverName='sapi5')
    engine.setProperty('rate', 170)
    for v in engine.getProperty('voices'):
        if 'spanish' in v.name.lower():
            engine.setProperty('voice', v.id)
            break
    while True:
        text = text_q.get()
        if text is None:
            break
        fd, path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        engine.save_to_file(text, path)
        engine.runAndWait()
        data, sr = sf.read(path, dtype='float32')
        os.remove(path)
        if sr != FS:
            data = resample(data, int(len(data) * FS / sr))
        if data.ndim > 1:
            data = data.mean(axis=1)
        play_q.put(data)
        text_q.task_done()
threading.Thread(target=tts_worker, daemon=True).start()

# Hilo reproductor secuencial

def player_worker():
    while True:
        arr = play_q.get()
        if arr is None:
            break
        if arr.ndim == 1:
            arr = arr[:, None]
        with spk.player(samplerate=FS, channels=arr.shape[1]) as player:
            player.play(arr)
        play_q.task_done()
threading.Thread(target=player_worker, daemon=True).start()

# Productor con VAD

def producer_vad():
    buffer = []
    silent = 0
    while True:
        with mic.recorder(samplerate=FS, channels=CHANNELS) as rec:
            frame = rec.record(FRAME_SAMPLES)
        mono = frame.mean(axis=1)
        rms = np.sqrt(np.mean(mono * mono))
        if rms >= THRESH:
            buffer.append(frame)
            silent = 0
        else:
            if buffer:
                buffer.append(frame)
                silent += 1
                if silent >= SILENCE_FRAMES:
                    segment = np.vstack(buffer)
                    audio_q.put(segment)
                    buffer = []
                    silent = 0

# Procesador de segmentos

def process_segment(seg: np.ndarray):
    global last_line
    mono = seg.mean(axis=1)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, mono, FS, format='WAV', subtype='PCM_16')
        path = tmp.name
    try:
        segments, _ = whisper_model.transcribe(path, beam_size=5)
        text_en = ' '.join([s.text for s in segments]).strip()
        if not text_en:
            return
        out = translator(text_en)
        text_es = out[0]['translation_text'].strip()
        if text_es and text_es != last_line:
            last_line = text_es
            print(Fore.GREEN + text_es + Style.RESET_ALL)
            text_q.put(text_es)
    except Exception as e:
        print('[ERR]', e)
    finally:
        os.remove(path)

# Consumidor

def consumer():
    with ThreadPoolExecutor(max_workers=2) as pool:
        while True:
            seg = audio_q.get()
            if seg is None:
                break
            pool.submit(process_segment, seg)

# Main
if __name__ == '__main__':
    threading.Thread(target=producer_vad, daemon=True).start()
    consumer()
    text_q.join()
    play_q.join()
    text_q.put(None)
    play_q.put(None)
    print('✅ Sesión finalizada')
