import logging
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
from .model import MobileNet1D
from .multi_user_adapter import MultiUserVoiceAdapter, VoiceAdapter

logger = logging.getLogger(__name__)


class AudioEngine:

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.model = MobileNet1D(
            in_channels=3 * Config.N_MFCC,
            num_classes=len(Config.LABELS),
        ).to(self.device)
        self.scaler = None
        self._last_embedding: np.ndarray | None = None

        self.multi_adapter = MultiUserVoiceAdapter(save_dir=Config.USERS_DIR)
        self.current_speaker: str | None = None

        self.audio_q: queue.Queue = queue.Queue()
        self.command_q: queue.Queue = queue.Queue(maxsize=Config.COMMAND_QUEUE_MAX)
        self.mic_level = 0.0

        self._vad = webrtcvad.Vad(Config.WEBRTC_AGGRESSIVENESS)
        self._running = False
        self._worker_thread: threading.Thread | None = None
        self._stream: sd.InputStream | None = None

        self.model.head[-1].register_forward_pre_hook(self._capture_embedding)
        self._load_assets()

    def _capture_embedding(self, module, inputs):
        self._last_embedding = inputs[0].detach().cpu().numpy()[0]

    def set_current_user(self, user_id: str):
        if user_id in self.multi_adapter.adapters:
            self.current_speaker = user_id
            self.multi_adapter.current_user = user_id
            samples = len(self.multi_adapter.adapters[user_id].embeddings)
            logger.info("Пользователь %s (%d образцов)", user_id, samples)
        else:
            self.multi_adapter.adapters[user_id] = VoiceAdapter()
            self.current_speaker = user_id
            self.multi_adapter.current_user = user_id
            logger.info("Создан профиль: %s", user_id)

    def _load_assets(self):
        if os.path.exists(Config.MODEL_PATH):
            try:
                state = torch.load(
                    Config.MODEL_PATH,
                    map_location=self.device,
                    weights_only=True,
                )
            except TypeError:
                state = torch.load(Config.MODEL_PATH, map_location=self.device)
            if MobileNet1D.is_compatible(state):
                self.model.load_state_dict(state)
                logger.info("MobileNet1D загружена: %s", Config.MODEL_PATH)
            else:
                logger.warning("Веса от старой AudioCNN — пропуск загрузки")
            self.model.eval()
        else:
            logger.warning("Модель не найдена: %s", Config.MODEL_PATH)

        if os.path.exists(Config.SCALER_PATH):
            with open(Config.SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
            logger.info("Scaler загружен")
        else:
            logger.warning("Scaler не найден: %s", Config.SCALER_PATH)

        users = self.multi_adapter.get_user_list()
        logger.info(
            "Устройство: %s | WebRTC VAD=%s | SW=%s TTA=%s",
            self.device,
            Config.WEBRTC_AGGRESSIVENESS,
            Config.SLIDE_STEP,
            Config.TTA_RUNS,
        )
        if users:
            logger.info("Пользователи: %s", users)

    def _mfcc_full(self, audio: np.ndarray) -> np.ndarray:
        a = audio.flatten().astype(np.float32)
        peak = np.max(np.abs(a))
        if peak > 1e-6:
            a /= peak
        mfcc = librosa.feature.mfcc(
            y=a, sr=Config.SAMPLE_RATE, n_mfcc=Config.N_MFCC, hop_length=160)
        delta = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        return np.concatenate([mfcc, delta, delta2], axis=0)

    def _infer(self, feat_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = feat_flat.reshape(1, 1, 3 * Config.N_MFCC, Config.MAX_FRAMES)
        x = torch.tensor(f, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            out = self.model(x)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()

        if self._last_embedding is None:
            embedding = np.zeros(Config.EMBEDDING_DIM, dtype=np.float32)
        else:
            embedding = self._last_embedding.copy()

        return probs, embedding

    def _predict_with_embedding(self, audio: np.ndarray):
        full = self._mfcc_full(audio)
        T_full = full.shape[1]

        windows = []
        if T_full <= Config.MAX_FRAMES:
            windows.append(
                np.pad(full, ((0, 0), (0, Config.MAX_FRAMES - T_full))))
        else:
            for s in range(0, T_full - Config.MAX_FRAMES + 1, Config.SLIDE_STEP):
                windows.append(full[:, s:s + Config.MAX_FRAMES])
            windows.append(full[:, T_full - Config.MAX_FRAMES:])

        all_probs = []
        all_embeddings = []

        for win in windows:
            for _ in range(Config.TTA_RUNS):
                shift = np.random.randint(-2, 3)
                if shift > 0:
                    w = np.pad(win, ((0, 0), (shift, 0)))[:, :Config.MAX_FRAMES]
                elif shift < 0:
                    w = np.pad(win, ((0, 0), (0, -shift)))[:, -Config.MAX_FRAMES:]
                else:
                    w = win.copy()
                if self.scaler:
                    w = self.scaler.transform(w.reshape(1, -1))
                probs, emb = self._infer(w)
                all_probs.append(probs)
                all_embeddings.append(emb)

        avg_probs = np.mean(all_probs, axis=0)
        avg_embedding = np.mean(all_embeddings, axis=0)
        pred = int(avg_probs.argmax())
        conf = float(avg_probs[pred])
        return pred, conf, avg_probs, avg_embedding

    def _is_speech(self, chunk: np.ndarray) -> bool:
        flat = chunk.flatten()
        self.mic_level = float(np.sqrt(np.mean(flat ** 2)))

        pcm = (flat * 32767).astype(np.int16)
        n = len(pcm) // Config.WEBRTC_FRAME_SAMPS
        if n == 0:
            return False

        votes = 0
        for i in range(n):
            frame = pcm[
                i * Config.WEBRTC_FRAME_SAMPS:(i + 1) * Config.WEBRTC_FRAME_SAMPS
            ].tobytes()
            try:
                if self._vad.is_speech(frame, Config.SAMPLE_RATE):
                    votes += 1
            except Exception:
                pass
        return votes > n * 0.5

    def _collect_speech(self):
        ring_buf = []
        speech_buf = []
        silent_cnt = 0
        speech_cnt = 0
        in_speech = False

        while self._running:
            try:
                chunk = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            sp = self._is_speech(chunk)

            if not in_speech:
                ring_buf.append(chunk)
                if len(ring_buf) > 2:
                    ring_buf.pop(0)
                if sp:
                    in_speech = True
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
                            return None
                        return np.concatenate(speech_buf, axis=0)
                if speech_cnt >= Config.MAX_SPEECH_CHUNKS:
                    return np.concatenate(speech_buf, axis=0)
        return None

    def _flush(self):
        while not self.audio_q.empty():
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break

    def _enqueue_command(self, item):
        try:
            self.command_q.put_nowait(item)
        except queue.Full:
            try:
                self.command_q.get_nowait()
            except queue.Empty:
                pass
            self.command_q.put_nowait(item)

    def _worker(self):
        with sd.InputStream(
                samplerate=Config.SAMPLE_RATE,
                channels=1,
                blocksize=Config.BLOCK_SIZE,
                callback=lambda d, f, t, s: self.audio_q.put(d.copy()),
        ) as stream:
            self._stream = stream
            while self._running:
                audio = self._collect_speech()
                if audio is None:
                    continue

                pred, conf, raw_probs, embedding = self._predict_with_embedding(audio)

                speaker = self.current_speaker
                if speaker and speaker in self.multi_adapter.adapters:
                    probs = self.multi_adapter.correct_for_user(
                        speaker, raw_probs, embedding)
                else:
                    probs = raw_probs
                    speaker = None

                pred = int(probs.argmax())
                conf = float(probs[pred])

                label = Config.COMMANDS[pred]
                threshold = Config.COMMAND_THRESHOLDS[label]
                accepted = label != "noise" and conf >= threshold

                if accepted and speaker:
                    adapter = self.multi_adapter.adapters[speaker]
                    adapter.set_features(embedding)
                    adapter.add(pred)
                    if adapter.n_samples % 10 == 0:
                        self.multi_adapter.save_user(speaker)
                        logger.debug(
                            "[%s] %d/%d образцов",
                            speaker, adapter.n_samples, adapter.max_samples)

                self._enqueue_command(
                    (label, conf, probs.copy(), accepted, speaker))

                self._flush()
                time.sleep(Config.COOLDOWN_SEC)
                self._flush()

        self._stream = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker, daemon=True, name="AudioEngine")
        self._worker_thread.start()

    def stop(self):
        self._running = False
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
        if self.current_speaker:
            self.multi_adapter.save_user(self.current_speaker)
