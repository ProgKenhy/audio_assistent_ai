import os
import pickle
import queue
import threading
import time

import numpy as np
import librosa
import torch
import sounddevice as sd
import webrtcvad

from .config import Config
from .model  import MobileNet1D
from .voice_adapter import VoiceAdapter


class AudioEngine:
    """
    Аудио движок голосового ассистента.

    Поток _worker непрерывно:
      1. Слушает микрофон через sounddevice callback
      2. Детектирует речь через WebRTC VAD (конечный автомат)
      3. Прогоняет буфер через sliding window + TTA
      4. Корректирует вероятности через kNN VoiceAdapter
      5. Кладёт результат в command_q

    Формат элементов command_q:
        (label: str, conf: float, probs: np.ndarray, accepted: bool)
    """

    def __init__(self):
        self.device  = torch.device(Config.DEVICE)
        self.model   = MobileNet1D(
            in_channels=3 * Config.N_MFCC,
            num_classes=len(Config.LABELS),
        ).to(self.device)
        self.scaler  = None
        self.adapter = VoiceAdapter(self.model)

        adapter_path = Config.ADAPTER_PATH
        if os.path.exists(adapter_path):
            self.adapter.load(adapter_path)
            print("✓ VoiceAdapter загружен")

        self.audio_q   = queue.Queue()
        self.command_q = queue.Queue()
        self.mic_level = 0.0   # обновляется в VAD, читается рендером для шкалы

        self._vad = webrtcvad.Vad(Config.WEBRTC_AGGRESSIVENESS)
        self._load_assets()

    # ── Загрузка весов и scaler ───────────────────────────────────────────

    def _load_assets(self):
        if os.path.exists(Config.MODEL_PATH):
            state = torch.load(Config.MODEL_PATH, map_location=self.device)
            if MobileNet1D.is_compatible(state):
                self.model.load_state_dict(state)
                print(f"✓ MobileNet1D загружена  ({Config.MODEL_PATH})")
            else:
                print("⚠  Веса от старой AudioCNN — переобучи модель с MobileNet1D")
            self.model.eval()
        else:
            print(f"⚠  {Config.MODEL_PATH} не найден — модель не загружена")

        if os.path.exists(Config.SCALER_PATH):
            with open(Config.SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
            print("✓ Scaler загружен")
        else:
            print(f"⚠  {Config.SCALER_PATH} не найден — нормализация отключена")

        print(
            f"● Устройство: {self.device} | "
            f"WebRTC VAD (агресс={Config.WEBRTC_AGGRESSIVENESS}) | "
            f"SW шаг={Config.SLIDE_STEP} | TTA×{Config.TTA_RUNS} | kNN-адаптер"
        )

    # ── Извлечение признаков ──────────────────────────────────────────────

    def _mfcc_full(self, audio: np.ndarray) -> np.ndarray:
        """
        Считает MFCC + delta + delta² для всего буфера.
        Возвращает (120, T) без выравнивания — для sliding window.
        """
        a    = audio.flatten().astype(np.float32)
        peak = np.max(np.abs(a))
        if peak > 1e-6:
            a /= peak
        mfcc   = librosa.feature.mfcc(y=a, sr=Config.SAMPLE_RATE,
                                       n_mfcc=Config.N_MFCC, hop_length=160)
        delta  = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        return np.concatenate([mfcc, delta, delta2], axis=0)   # (120, T)

    # ── Один инференс ─────────────────────────────────────────────────────

    def _infer(self, feat_flat: np.ndarray) -> np.ndarray:
        """feat_flat: уже нормализованный вектор (12000,). Возвращает softmax (5,)."""
        f = feat_flat.reshape(1, 1, 3 * Config.N_MFCC, Config.MAX_FRAMES)
        x = torch.tensor(f, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return torch.softmax(self.model(x), dim=1)[0].cpu().numpy()

    # ── Sliding window + TTA ──────────────────────────────────────────────

    def _predict(self, audio: np.ndarray) -> tuple[int, float, np.ndarray]:
        """
        Надёжное предсказание: sliding window по полному буферу × TTA сдвиги.

        Sliding window решает проблему "команда попала в край буфера" —
        модель видит её в нескольких позициях, берётся усреднённая вероятность.

        TTA (±2 фрейма случайный сдвиг) даёт +3–5% точности без переобучения.
        """
        full   = self._mfcc_full(audio)   # (120, T_full)
        T_full = full.shape[1]

        # Собираем все окна
        windows: list[np.ndarray] = []
        if T_full <= Config.MAX_FRAMES:
            windows.append(np.pad(full, ((0, 0), (0, Config.MAX_FRAMES - T_full))))
        else:
            for s in range(0, T_full - Config.MAX_FRAMES + 1, Config.SLIDE_STEP):
                windows.append(full[:, s : s + Config.MAX_FRAMES])
            windows.append(full[:, T_full - Config.MAX_FRAMES :])   # хвост

        # TTA × sliding window → усредняем вероятности
        all_probs: list[np.ndarray] = []
        for win in windows:
            for _ in range(Config.TTA_RUNS):
                shift = np.random.randint(-2, 3)
                if shift > 0:
                    w = np.pad(win, ((0, 0), (shift, 0)))[:, : Config.MAX_FRAMES]
                elif shift < 0:
                    w = np.pad(win, ((0, 0), (0, -shift)))[:, -Config.MAX_FRAMES :]
                else:
                    w = win.copy()
                if self.scaler:
                    w = self.scaler.transform(w.reshape(1, -1))
                all_probs.append(self._infer(w))

        avg  = np.mean(all_probs, axis=0)
        pred = int(avg.argmax())
        conf = float(avg[pred])
        return pred, conf, avg

    # ── WebRTC VAD ────────────────────────────────────────────────────────

    def _is_speech(self, chunk: np.ndarray) -> bool:
        """
        Определяет наличие речи в chunk через WebRTC VAD.

        Chunk (50 мс) делится на 5 фреймов по 10 мс.
        Решение мажоритарным голосованием: >50% фреймов = речь.
        Обновляет mic_level для визуализации шкалы.
        """
        flat = chunk.flatten()
        self.mic_level = float(np.sqrt(np.mean(flat ** 2)))

        pcm = (flat * 32767).astype(np.int16)
        n   = len(pcm) // Config.WEBRTC_FRAME_SAMPS
        if n == 0:
            return False

        votes = 0
        for i in range(n):
            frame = pcm[i * Config.WEBRTC_FRAME_SAMPS :
                        (i + 1) * Config.WEBRTC_FRAME_SAMPS].tobytes()
            try:
                if self._vad.is_speech(frame, Config.SAMPLE_RATE):
                    votes += 1
            except Exception:
                pass
        return votes > n * 0.5

    # ── VAD FSM ───────────────────────────────────────────────────────────

    def _collect_speech(self) -> np.ndarray | None:
        """
        Конечный автомат: WAIT → SPEAKING → SILENCE.

        Возвращает аудио-буфер когда фраза завершена,
        или None если звук слишком короткий (шорох/кашель).
        """
        ring_buf:   list[np.ndarray] = []
        speech_buf: list[np.ndarray] = []
        silent_cnt = 0
        speech_cnt = 0
        in_speech  = False

        while True:
            chunk = self.audio_q.get()
            sp    = self._is_speech(chunk)

            if not in_speech:
                # Кольцевой буфер предконтекста (pre-roll)
                ring_buf.append(chunk)
                if len(ring_buf) > 2:
                    ring_buf.pop(0)
                if sp:
                    in_speech  = True
                    speech_buf = ring_buf.copy()
                    speech_cnt = 1
            else:
                speech_buf.append(chunk)
                speech_cnt += 1
                if sp:
                    silent_cnt = 0
                else:
                    silent_cnt += 1
                    if silent_cnt >= Config.SILENCE_LIMIT:
                        if speech_cnt < Config.MIN_SPEECH_CHUNKS:
                            return None   # слишком короткий — шорох
                        return np.concatenate(speech_buf, axis=0)
                # Защита от зависания при постоянном шуме
                if speech_cnt >= Config.MAX_SPEECH_CHUNKS:
                    return np.concatenate(speech_buf, axis=0)

    def _flush(self):
        """Очищает накопившиеся чанки — вызывается после обработки команды."""
        while not self.audio_q.empty():
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break

    # ── Рабочий поток ─────────────────────────────────────────────────────

    def _worker(self):
        with sd.InputStream(
            samplerate=Config.SAMPLE_RATE,
            channels=1,
            blocksize=Config.BLOCK_SIZE,
            callback=lambda d, f, t, s: self.audio_q.put(d.copy()),
        ):
            while True:
                audio = self._collect_speech()
                if audio is None:
                    continue

                pred, conf, raw_probs = self._predict(audio)

                # kNN-коррекция поверх CNN
                probs = self.adapter.correct(raw_probs)
                pred  = int(probs.argmax())
                conf  = float(probs[pred])

                label     = Config.COMMANDS[pred]
                threshold = Config.COMMAND_THRESHOLDS[label]
                accepted  = label != "noise" and conf >= threshold

                self.command_q.put((label, conf, probs.copy(), accepted))

                if accepted:
                    self.adapter.add(pred, conf)

                    # сохраняем периодически
                    if self.adapter.n_samples % 10 == 0:
                        self.adapter.save(Config.ADAPTER_PATH)

                self._flush()
                time.sleep(Config.COOLDOWN_SEC)
                self._flush()

    def start(self):
        threading.Thread(target=self._worker, daemon=True).start()
