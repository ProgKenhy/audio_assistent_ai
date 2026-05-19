import os
import sys


class Config:
    # ── Аудио ────────────────────────────────────────────────────────────
    SAMPLE_RATE = 16000
    N_MFCC      = 40
    MAX_FRAMES  = 100
    BLOCK_SIZE  = 800          # 50 мс/чанк — нужно для webrtcvad

    # WebRTC VAD
    WEBRTC_AGGRESSIVENESS = 3  # 0-3; поднять до 3 если много ложных срабатываний
    WEBRTC_FRAME_MS       = 10
    WEBRTC_FRAME_SAMPS    = SAMPLE_RATE * WEBRTC_FRAME_MS // 1000  # 160

    # VAD конечный автомат
    MIN_SPEECH_CHUNKS = 4
    SILENCE_LIMIT     = 8
    MAX_SPEECH_CHUNKS = 25
    COOLDOWN_SEC      = 0.4

    # Sliding window + TTA
    SLIDE_STEP = 20
    TTA_RUNS   = 3

    # Команды
    LABELS   = {"go": 0, "stop": 1, "left": 2, "right": 3, "noise": 4}
    COMMANDS = ["go", "stop", "left", "right", "noise"]

    VAD_THRESHOLD = 0.69

    COMMAND_THRESHOLDS = {
        "go":    0.60,
        "stop":  0.69,
        "left":  0.69,
        "right": 0.69,
        "noise": 1.00,
    }

    # ── Игра ─────────────────────────────────────────────────────────────
    GRID_SIZE   = 20
    BASE_FPS    = 4
    MAX_FPS     = 8
    HISTORY_LEN = 7
    USERS_DIR   = "users"
    SCORES_DIR  = "game/scores"
    COMMAND_QUEUE_MAX = 32

    # ── Модель ───────────────────────────────────────────────────────────
    if hasattr(sys, "_MEIPASS"):
        BASE_DIR = sys._MEIPASS
    else:
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    DEVICE      = "cpu"
    MODEL_PATH  = os.path.join(BASE_DIR, "models/best_model.pt")
    SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")
    EMBEDDING_DIM = 256

    @classmethod
    def score_file(cls, user_id: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
        return os.path.join(cls.SCORES_DIR, f"{safe}.txt")

    # ── Палитра ──────────────────────────────────────────────────────────
    BG          = (15,  15,  18)
    FIELD_BG    = (22,  22,  26)
    PANEL_BG    = (28,  28,  33)
    GRID        = (35,  35,  42)
    SNAKE_BODY  = (46,  204, 113)
    SNAKE_HEAD  = (39,  174, 96)
    FOOD        = (231, 76,  60)
    TEXT_MAIN   = (236, 240, 241)
    TEXT_MUTED  = (127, 140, 141)
    ACCENT_CMD  = (52,  152, 219)
    ACCENT_WARN = (241, 196, 15)
    ACCENT_OK   = (46,  204, 113)
    ACCENT_LOW  = (241, 196, 15)
    ACCENT_NOISE= (99,  110, 114)
