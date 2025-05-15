"""
detect_devices.py — Explorador de dispositivos de audio (v2)
# --------------------------------------------------------------------------------
# • Lista todos los dispositivos de entrada (micrófonos + loopbacks) y salida (altavoces).
# • Marca el predeterminado con ★.
# • Identifica los loopbacks con [L] (útiles para capturar audio del sistema).
# • Usa colorama para resaltar visualmente.
"""

import importlib.util, subprocess, sys
from textwrap import shorten
from colorama import init, Fore, Style

# Asegurar que soundcard esté instalado
if importlib.util.find_spec("soundcard") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "soundcard"])
import soundcard as sc

init(autoreset=True)

def fmt(name, width=45):
    """Acorta el nombre si es largo y lo alinea."""
    return shorten(name, width=width, placeholder="…").ljust(width)

# Micrófonos y loopbacks
def_mic = sc.default_microphone()
print(Fore.CYAN + "\nMicrófonos (incluye loopback)" + "\n" + "-"*55 + Style.RESET_ALL)
for i, mic in enumerate(sc.all_microphones(include_loopback=True)):
    star = "★" if mic.name == def_mic.name else " "
    loop = "[L]" if mic.isloopback else "   "
    print(f"[{i:2}] {star} {loop} {fmt(mic.name)}")

# Altavoces / salidas
def_sp = sc.default_speaker()
print(Fore.CYAN + "\nAltavoces / salidas" + "\n" + "-"*55 + Style.RESET_ALL)
for i, sp in enumerate(sc.all_speakers()):
    star = "★" if sp.name == def_sp.name else " "
    print(f"[{i:2}] {star}    {fmt(sp.name)}")

print("\n★ = predeterminado   [L] = loopback\n")
