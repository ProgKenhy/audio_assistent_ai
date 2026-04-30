import librosa, soundfile as sf, os
os.makedirs('dataset/noise', exist_ok=True)
for fname in os.listdir('dataset/_background_noise'):
    if not fname.endswith('.wav'): continue
    audio, sr = librosa.load(f'dataset/_background_noise/{fname}', sr=16000)
    for i, start in enumerate(range(0, len(audio) - sr, sr)):
        chunk = audio[start:start+sr]
        sf.write(f'dataset/noise/{fname[:-4]}_{i}.wav', chunk, sr)