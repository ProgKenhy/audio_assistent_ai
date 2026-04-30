import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os

SAMPLE_RATE = 16000
DURATION = 2  # секунды

commands = ["go", "stop", "left", "right", "noise"]

# создаём папки
for cmd in commands:
    os.makedirs(f"dataset/{cmd}", exist_ok=True)


def record_audio():
    print("Говори...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype='float32')
    sd.wait()
    return audio


for cmd in commands:
    print(f"\n=== Команда: {cmd} ===")
    
    for i in range(60, 101):   # 30 записей на команду
        input("Нажми Enter и скажи команду...")
        
        audio = record_audio()
        
        path = f"dataset/{cmd}/{i}.wav"
        wav.write(path, SAMPLE_RATE, audio)
        
        print(f"Сохранено: {path}")