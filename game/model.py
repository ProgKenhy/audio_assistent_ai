import numpy as np
import torch
import torch.nn as nn

from .config import Config


# ─────────────────────────────────────────────────────────────────────────────
# MobileNet1D
#
# Заменяет старую AudioCNN. Работает по временной оси MFCC (1D вместо 2D).
#
# Ключевые блоки:
#   SEBlock      — Squeeze-and-Excitation: учится какие каналы важны
#   DSConvBlock  — Depthwise Separable Conv + BN + SE + residual
#                  в 5–8× меньше вычислений чем обычная Conv
#
# Входной тензор: (B, 1, 120, 100) — тот же формат что у AudioCNN,
# squeeze(1) разворачивает его в (B, 120, 100) для 1D-свёрток.
# ─────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: взвешивает каналы по их важности."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return x * self.fc(x).unsqueeze(-1)


class DSConvBlock(nn.Module):
    """Depthwise Separable Convolution + BatchNorm + ReLU + SE + residual."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw  = nn.Conv1d(in_ch, in_ch, 3, stride=stride,
                             padding=1, groups=in_ch, bias=False)
        self.pw  = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm1d(out_ch)
        self.se  = SEBlock(out_ch)
        self.act = nn.ReLU()
        self.skip = (
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False)
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn(self.pw(self.dw(x))))
        return self.se(out) + self.skip(x)


class MobileNet1D(nn.Module):
    def __init__(self, in_channels: int = 120, num_classes: int = 5):
        super().__init__()
        # in_channels = 3 * N_MFCC = 120  (mfcc + delta + delta2)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            DSConvBlock(64,  128, stride=2),
            DSConvBlock(128, 128),
            DSConvBlock(128, 256, stride=2),
            DSConvBlock(256, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)                          # (B,1,120,100) → (B,120,100)
        return self.head(self.blocks(self.stem(x)))

    @staticmethod
    def is_compatible(state_dict: dict) -> bool:
        """Проверяет что веса от MobileNet1D, а не от старой AudioCNN."""
        return any("stem" in k or "blocks" in k for k in state_dict)


# ─────────────────────────────────────────────────────────────────────────────
# VoiceAdapter — kNN персонализация поверх CNN
#
# Собирает 256-мерные эмбеддинги (вход последнего Linear) через forward hook.
# При каждом предсказании смешивает вероятности CNN с kNN по голосу пользователя.
# Адаптируется автоматически — начинает помогать после ~10 уверенных команд.
# ─────────────────────────────────────────────────────────────────────────────

class VoiceAdapter:
    def __init__(self, model: MobileNet1D, max_samples: int = 50,
                 k: int = 5, alpha: float = 0.3):
        self.max_samples = max_samples
        self.k           = k
        self.alpha       = alpha          # вес kNN (0 = только CNN, 1 = только kNN)
        self.embeddings: list[tuple[np.ndarray, int]] = []
        self._features: np.ndarray | None = None

        # Hook захватывает вход последнего Linear = 256-мерный эмбеддинг
        model.head[-1].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._features = inp[0].detach().cpu().numpy()[0]

    def add(self, label_idx: int):
        """Запомнить текущий эмбеддинг с меткой команды."""
        if self._features is None:
            return
        self.embeddings.append((self._features.copy(), label_idx))
        if len(self.embeddings) > self.max_samples:
            self.embeddings.pop(0)   # FIFO

    def correct(self, cnn_probs: np.ndarray) -> np.ndarray:
        """Скорректировать вероятности CNN через kNN. Если данных мало — вернуть как есть."""
        if len(self.embeddings) < self.k or self._features is None:
            return cnn_probs

        embs   = np.array([e for e, _ in self.embeddings])
        labels = np.array([l for _, l in self.embeddings])
        dists  = np.linalg.norm(embs - self._features, axis=1)
        top_k  = np.argsort(dists)[:self.k]

        knn = np.zeros(len(Config.COMMANDS))
        for i in top_k:
            knn[labels[i]] += 1.0
        knn /= self.k

        return (1 - self.alpha) * cnn_probs + self.alpha * knn

    @property
    def n_samples(self) -> int:
        return len(self.embeddings)
