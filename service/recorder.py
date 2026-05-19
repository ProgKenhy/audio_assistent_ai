import os

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd

SAMPLE_RATE = 16000
DURATION = 2
RECORDS_PER_COMMAND = 30

commands = ["go", "stop", "left", "right", "noise"]

for cmd in commands:
    os.makedirs(f"dataset/{cmd}", exist_ok=True)


def record_audio() -> np.ndarray:
    print("Говори...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio


for cmd in commands:
    print(f"\n=== Команда: {cmd} ===")

    for i in range(1, RECORDS_PER_COMMAND + 1):
        input("Нажми Enter и скажи команду...")

        audio = record_audio()
        path = f"dataset/{cmd}/{i:03d}.wav"
        wav.write(path, SAMPLE_RATE, audio)

        print(f"Сохранено: {path}")
