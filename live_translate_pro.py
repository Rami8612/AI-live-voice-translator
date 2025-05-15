"""
live_translate_pro.py — Traducción en vivo + TTS Coral con segmentación basada en silencio (VAD)
----------------------------------------------------------------------------------
• Captura audio loopback.
• Detecta voz y silencio por RMS en frames de 100 ms.
• Segmenta en frases al detectar silencio prolongado.
• Transcribe con Whisper → Traduce con GPT-4o-mini (streaming).
• Convierte cada frase traducida a voz "coral" (OpenAI TTS) y la encola.
• Reproduce secuencialmente las frases en tu altavoz elegido.
• CLI: --device IN, --out_name SUBSTRING o --out_device INDEX,
       --threshold RMS, --silence SILENCE_SEC.

Comando de uso:
    python live_translate_pro.py --device 0 --out_device 1 --threshold 0.01 --silence 0.5

Autor: Rami8612
"""

import os, io, tempfile, argparse, queue, threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import soundcard as sc
import soundfile as sf
from dotenv import load_dotenv
import openai
from colorama import init, Fore, Style
from scipy.signal import resample

# Configuración
FS = 16000           # Frecuencia de muestreo
CHANNELS = 2         # Audio estéreo
FRAME_MS = 100       # Tamaño de frame en ms para VAD
init(autoreset=True)

# Carga clave OpenAI
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise RuntimeError('OPENAI_API_KEY no definida en .env o en entorno')

# CLI
parser = argparse.ArgumentParser('Traducción en vivo con VAD y TTS Coral')
parser.add_argument('--device',     type=int,   default=1,    help='Índice loopback (IN)')
parser.add_argument('--out_name',   type=str,   default='',   help='Nombre (substring) de altavoz (OUT)')
parser.add_argument('--out_device', type=int,   default=-1,   help='Índice altavoz (OUT) si no usa --out_name')
parser.add_argument('--threshold',  type=float, default=0.01, help='Umbral RMS para voz (0–1)')
parser.add_argument('--silence',    type=float, default=0.5,  help='Duración de silencio [s] para cortar')
args = parser.parse_args()
THRESH = args.threshold
SILENCE_SEC = args.silence

# Dispositivos
mics = sc.all_microphones(include_loopback=True)
if not (0 <= args.device < len(mics)):
    raise RuntimeError('Índice de dispositivo de entrada fuera de rango')
mic = mics[args.device]
# Salida
speakers = sc.all_speakers()
if args.out_name:
    spk = next((s for s in speakers if args.out_name.lower() in s.name.lower()), None)
    if not spk:
        raise RuntimeError(f"Altavoz que contiene '{args.out_name}' no encontrado")
else:
    spk = speakers[args.out_device] if args.out_device >= 0 else sc.default_speaker()

print(Fore.CYAN + f"IN : {mic.name}\nOUT: {spk.name}" + Style.RESET_ALL)
print(f"RMS ≥ {THRESH}, silencio ≥ {SILENCE_SEC}s → segmentando frases\n")

# Colas
audio_q = queue.Queue()
play_q = queue.Queue()
last_line = ''

# VAD y segmentación
FRAME_SAMPLES = int(FS * FRAME_MS / 1000)
SILENCE_FRAMES = int(SILENCE_SEC * 1000 / FRAME_MS)

def producer_vad():
    """Segmenta audio en frases usando VAD por RMS y silencio."""
    buffer = []
    silent_count = 0
    try:
        while True:
            with mic.recorder(samplerate=FS, channels=CHANNELS) as rec:
                frame = rec.record(FRAME_SAMPLES)
            mono = frame.mean(axis=1)
            rms = np.sqrt(np.mean(mono**2))
            if rms >= THRESH:
                buffer.append(frame)
                silent_count = 0
            else:
                if buffer:
                    buffer.append(frame)
                    silent_count += 1
                    if silent_count >= SILENCE_FRAMES:
                        segment = np.vstack(buffer)
                        audio_q.put(segment)
                        buffer = []
                        silent_count = 0
    except KeyboardInterrupt:
        audio_q.put(None)

def playback_worker():
    while True:
        arr = play_q.get()
        if arr is None:
            break
        if arr.ndim == 1:
            arr = arr[:, None]
        with spk.player(samplerate=FS, channels=arr.shape[1]) as player:
            player.play(arr)
        play_q.task_done()

threading.Thread(target=playback_worker, daemon=True).start()

def tts_coral(text: str) -> np.ndarray:
    rsp = openai.audio.speech.create(
        model='gpt-4o-mini-tts', voice='coral', input=text, response_format='wav'
    )
    data, sr = sf.read(io.BytesIO(rsp.read()), dtype='float32')
    if sr != FS:
        data = resample(data, int(len(data) * FS / sr))
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data

def process_segment(seg: np.ndarray):
    global last_line
    mono = seg.mean(axis=1)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, mono, FS, format='WAV', subtype='PCM_16')
        path = tmp.name
    try:
        with open(path, 'rb') as f:
            txt = openai.audio.transcriptions.create(
                model='whisper-1', file=f, response_format='text'
            ).strip()
        if not txt:
            return
        tokens = []
        for part in openai.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role':'system','content':'Eres traductor ING→ESP. Devuelve solo la traducción.'},
                {'role':'user','content':txt}
            ], stream=True
        ):
            c = part.choices[0].delta.content or ''
            tokens.append(c)
        line = ''.join(tokens).strip()
        if line and line != last_line:
            last_line = line
            print(Fore.GREEN + line + Style.RESET_ALL)
            play_q.put(tts_coral(line))
    except Exception as e:
        print('[ERR]', e)
    finally:
        os.remove(path)

def consumer():
    with ThreadPoolExecutor(max_workers=2) as pool:
        while True:
            seg = audio_q.get()
            if seg is None:
                break
            pool.submit(process_segment, seg)

if __name__ == '__main__':
    threading.Thread(target=producer_vad, daemon=True).start()
    consumer()
    play_q.join()
    play_q.put(None)
    print('✅ Sesión finalizada')
