import numpy as np
import pickle
import os
from collections import defaultdict


class VoiceAdapter:
    def __init__(self, max_samples: int = 50, k: int = 5, alpha: float = 0.3):
        self.max_samples = max_samples
        self.k = k
        self.alpha = alpha
        self.embeddings = []
        self._features = None

    def set_features(self, features):
        self._features = features

    def add(self, label_idx: int):
        if self._features is None:
            return
        self.embeddings.append((self._features.copy(), label_idx))
        if len(self.embeddings) > self.max_samples:
            self.embeddings.pop(0)

    def correct(self, cnn_probs: np.ndarray) -> np.ndarray:
        if len(self.embeddings) < self.k or self._features is None:
            return cnn_probs

        embs = np.array([e for e, _ in self.embeddings])
        labels = np.array([l for _, l in self.embeddings])
        dists = np.linalg.norm(embs - self._features, axis=1)

        top_k = np.argsort(dists)[:self.k]
        knn_probs = np.zeros(len(cnn_probs))
        for idx in top_k:
            knn_probs[labels[idx]] += 1.0
        knn_probs /= self.k

        return (1 - self.alpha) * cnn_probs + self.alpha * knn_probs
    
    @property
    def n_samples(self):
        return len(self.embeddings)


class MultiUserVoiceAdapter:
    def __init__(self, save_dir: str = "users/"):
        self.save_dir = save_dir
        self.adapters = {}
        self.user_embeddings = defaultdict(list)
        self.current_user = None
        self._current_embedding = None

        os.makedirs(save_dir, exist_ok=True)
        self._load_all_users()

    def set_current_embedding(self, embedding: np.ndarray):
        self._current_embedding = embedding
        if self.current_user and self.current_user in self.adapters:
            self.adapters[self.current_user].set_features(embedding)

    def identify_speaker(self, embedding: np.ndarray) -> str:
        if not self.user_embeddings:
            return None

        best_user = None
        best_score = 0

        for user_id, embeddings in self.user_embeddings.items():
            if len(embeddings) < 5:
                continue

            distances = [np.linalg.norm(embedding - e) for e in embeddings]
            mean_dist = np.mean(distances)
            score = 1.0 / (1.0 + mean_dist)

            if score > best_score:
                best_score = score
                best_user = user_id

        return best_user if best_score > 0.6 else None

    def add_user_sample(self, user_id: str, embedding: np.ndarray, label_idx: int):
        if user_id not in self.adapters:
            self.adapters[user_id] = VoiceAdapter()
            self.user_embeddings[user_id] = []
            print(f"Создан новый пользователь: {user_id}")

        adapter = self.adapters[user_id]
        adapter.set_features(embedding)
        adapter.add(label_idx)

        self.user_embeddings[user_id].append(embedding.copy())

        if len(self.user_embeddings[user_id]) > 30:
            self.user_embeddings[user_id].pop(0)

        self._save_user(user_id)

    def correct(self, cnn_probs: np.ndarray, embedding: np.ndarray = None) -> np.ndarray:
        if embedding is not None:
            self._current_embedding = embedding
            user = self.identify_speaker(embedding)
            if user:
                self.current_user = user
                if user in self.adapters:
                    self.adapters[user].set_features(embedding)
                    return self.adapters[user].correct(cnn_probs)

        if self.current_user and self.current_user in self.adapters:
            return self.adapters[self.current_user].correct(cnn_probs)

        return cnn_probs

    def correct_for_user(self, user_id: str, cnn_probs: np.ndarray, embedding: np.ndarray = None) -> np.ndarray:
        self.current_user = user_id
        if user_id in self.adapters:
            if embedding is not None:
                self.adapters[user_id].set_features(embedding)
            return self.adapters[user_id].correct(cnn_probs)
        return cnn_probs

    def _save_user(self, user_id: str):
        if user_id not in self.adapters:
            return

        adapter = self.adapters[user_id]
        filepath = os.path.join(self.save_dir, f"{user_id}.pkl")

        try:
            with open(filepath, "wb") as f:
                pickle.dump({
                    'embeddings': adapter.embeddings,
                    'max_samples': adapter.max_samples,
                    'k': adapter.k,
                    'alpha': adapter.alpha
                }, f)
        except Exception as e:
            print(f"Ошибка сохранения {user_id}: {e}")

    def _load_all_users(self):
        if not os.path.exists(self.save_dir):
            return

        for filename in os.listdir(self.save_dir):
            if filename.endswith(".pkl"):
                user_id = filename[:-4]
                filepath = os.path.join(self.save_dir, filename)

                try:
                    with open(filepath, "rb") as f:
                        data = pickle.load(f)

                    adapter = VoiceAdapter(
                        max_samples=data['max_samples'],
                        k=data['k'],
                        alpha=data['alpha']
                    )
                    adapter.embeddings = data['embeddings']
                    self.adapters[user_id] = adapter

                    self.user_embeddings[user_id] = []
                    for emb, _ in adapter.embeddings:
                        self.user_embeddings[user_id].append(emb)

                    print(f"Загружен пользователь: {user_id} ({len(adapter.embeddings)} образцов)")

                except Exception as e:
                    print(f"Ошибка загрузки {user_id}: {e}")

    def get_user_list(self) -> list:
        return list(self.adapters.keys())

    def get_stats(self) -> dict:
        stats = {}
        for user_id, adapter in self.adapters.items():
            stats[user_id] = {
                'samples': len(adapter.embeddings),
                'identified': len(self.user_embeddings.get(user_id, []))
            }
        return stats

    def get_current_user(self) -> str:
        return self.current_user

    def set_current_user(self, user_id: str):
        if user_id in self.adapters:
            self.current_user = user_id
            print(f"Текущий пользователь: {user_id}")
        else:
            print(f"Пользователь {user_id} не найден")