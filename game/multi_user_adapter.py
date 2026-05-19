import logging
import os
import pickle

import numpy as np

from .config import Config

logger = logging.getLogger(__name__)

_PICKLE_KEYS = frozenset({"embeddings", "max_samples", "k", "alpha"})


class VoiceAdapter:
    def __init__(self, max_samples: int = 50, k: int = 5, alpha: float = 0.3):
        self.max_samples = max_samples
        self.k = k
        self.alpha = alpha
        self.embeddings: list[tuple[np.ndarray, int]] = []
        self._features: np.ndarray | None = None

    def set_features(self, features: np.ndarray):
        self._features = np.asarray(features, dtype=np.float32)

    def add(self, label_idx: int):
        if self._features is None:
            return
        self.embeddings.append((self._features.copy(), int(label_idx)))
        if len(self.embeddings) > self.max_samples:
            self.embeddings.pop(0)

    def correct(self, cnn_probs: np.ndarray) -> np.ndarray:
        n_classes = len(Config.COMMANDS)
        if len(self.embeddings) < self.k or self._features is None:
            return cnn_probs

        embs = np.array([e for e, _ in self.embeddings])
        labels = np.array([l for _, l in self.embeddings])
        dists = np.linalg.norm(embs - self._features, axis=1)

        top_k = np.argsort(dists)[:self.k]
        knn_probs = np.zeros(n_classes)
        for idx in top_k:
            knn_probs[labels[idx]] += 1.0
        knn_probs /= self.k

        return (1 - self.alpha) * cnn_probs + self.alpha * knn_probs

    @property
    def n_samples(self) -> int:
        return len(self.embeddings)


def _validate_user_data(data: dict) -> bool:
    if not isinstance(data, dict) or not _PICKLE_KEYS.issubset(data.keys()):
        return False
    for emb, label in data["embeddings"]:
        if not isinstance(emb, np.ndarray) or emb.ndim != 1:
            return False
        if not isinstance(label, (int, np.integer)):
            return False
    return True


class MultiUserVoiceAdapter:
    def __init__(self, save_dir: str | None = None):
        self.save_dir = save_dir or Config.USERS_DIR
        self.adapters: dict[str, VoiceAdapter] = {}
        self.current_user: str | None = None

        os.makedirs(self.save_dir, exist_ok=True)
        self._load_all_users()

    def correct_for_user(
        self,
        user_id: str,
        cnn_probs: np.ndarray,
        embedding: np.ndarray | None = None,
    ) -> np.ndarray:
        self.current_user = user_id
        if user_id not in self.adapters:
            self.adapters[user_id] = VoiceAdapter()
        if embedding is not None:
            self.adapters[user_id].set_features(embedding)
        return self.adapters[user_id].correct(cnn_probs)

    def save_user(self, user_id: str):
        if user_id not in self.adapters:
            return

        adapter = self.adapters[user_id]
        filepath = os.path.join(self.save_dir, f"{user_id}.pkl")

        try:
            with open(filepath, "wb") as f:
                pickle.dump({
                    "embeddings": adapter.embeddings,
                    "max_samples": adapter.max_samples,
                    "k": adapter.k,
                    "alpha": adapter.alpha,
                }, f)
        except OSError as e:
            logger.error("Ошибка сохранения %s: %s", user_id, e)

    def _load_all_users(self):
        if not os.path.isdir(self.save_dir):
            return

        for filename in os.listdir(self.save_dir):
            if not filename.endswith(".pkl"):
                continue
            user_id = filename[:-4]
            filepath = os.path.join(self.save_dir, filename)

            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                if not _validate_user_data(data):
                    logger.warning("Некорректный профиль %s — пропуск", user_id)
                    continue

                adapter = VoiceAdapter(
                    max_samples=data["max_samples"],
                    k=data["k"],
                    alpha=data["alpha"],
                )
                adapter.embeddings = data["embeddings"]
                self.adapters[user_id] = adapter
                logger.info(
                    "Загружен %s (%d образцов)",
                    user_id, len(adapter.embeddings))

            except (OSError, pickle.UnpicklingError, KeyError) as e:
                logger.warning("Ошибка загрузки %s: %s", user_id, e)

    def get_user_list(self) -> list[str]:
        return list(self.adapters.keys())

    def get_stats(self) -> dict[str, dict]:
        return {
            user_id: {"samples": len(adapter.embeddings)}
            for user_id, adapter in self.adapters.items()
        }

    def get_current_user(self) -> str | None:
        return self.current_user

    def set_current_user(self, user_id: str):
        if user_id in self.adapters:
            self.current_user = user_id
        else:
            logger.warning("Пользователь %s не найден", user_id)
