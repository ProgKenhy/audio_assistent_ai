import os
import pickle
import numpy as np


class VoiceAdapter:
    """
    kNN адаптер поверх CNN.
    Сохраняет голосовые эмбеддинги пользователя и корректирует предсказания.
    """

    def __init__(self, model=None, max_samples: int = 50, k: int = 5, alpha: float = 0.3):
        self.model = model
        self.max_samples = max_samples
        self.k = k
        self.alpha = alpha

        self.embeddings = []  # (embedding, label)
        self._features = None

        # если модель передана — вешаем hook
        if model is not None:
            model.head[-1].register_forward_hook(self._hook)

    # ── Hook для извлечения эмбеддингов ─────────────────────────────

    def _hook(self, module, inp, out):
        self._features = inp[0].detach().cpu().numpy()[0]

    # ── Добавление нового примера ───────────────────────────────────

    def add(self, label_idx: int, conf: float):
        if self._features is None:
            return

        # только хорошие примеры
        if conf < 0.69:
            return

        if len(self.embeddings) >= self.max_samples:
            return

        self.embeddings.append((self._features.copy(), label_idx))

    # ── Коррекция CNN вероятностей ───────────────────────────────────

    def correct(self, cnn_probs: np.ndarray) -> np.ndarray:
        if len(self.embeddings) < self.k or self._features is None:
            return cnn_probs

        embs = np.array([e for e, _ in self.embeddings])
        labels = np.array([l for _, l in self.embeddings])

        dists = np.linalg.norm(embs - self._features, axis=1)
        top_k = np.argsort(dists)[:self.k]

        knn = np.zeros(len(cnn_probs))
        for i in top_k:
            knn[labels[i]] += 1.0

        knn /= self.k

        return (1 - self.alpha) * cnn_probs + self.alpha * knn

    # ── Персистентность ─────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "embeddings": self.embeddings,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        if not os.path.exists(path):
            return

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.embeddings = data.get("embeddings", [])

    # ── статус ──────────────────────────────────────────────────────

    @property
    def n_samples(self):
        return len(self.embeddings)