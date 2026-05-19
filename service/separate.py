import os
import sys

import librosa
import soundfile as sf

SRC_DIR = "dataset/_background_noise"
OUT_DIR = "dataset/noise"
SAMPLE_RATE = 16000


def main():
    if not os.path.isdir(SRC_DIR):
        print(f"Папка не найдена: {SRC_DIR}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    for fname in sorted(os.listdir(SRC_DIR)):
        if not fname.endswith(".wav"):
            continue
        audio, _ = librosa.load(os.path.join(SRC_DIR, fname), sr=SAMPLE_RATE)
        base = fname[:-4]
        for i, start in enumerate(range(0, len(audio) - SAMPLE_RATE, SAMPLE_RATE)):
            chunk = audio[start:start + SAMPLE_RATE]
            sf.write(os.path.join(OUT_DIR, f"{base}_{i}.wav"), chunk, SAMPLE_RATE)
        print(f"Обработан: {fname}")


if __name__ == "__main__":
    main()
