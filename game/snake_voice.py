import queue
import threading
import pickle
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import librosa
import sounddevice as sd
import pygame

# ─────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────
class Config:
    SAMPLE_RATE = 16000
    N_MFCC = 40
    MAX_FRAMES = 100
    BLOCK_SIZE = 800    # 50 мс на чанк — стабильная единица для VAD
    VAD_THRESHOLD = 0.01
    CONFIDENCE_THRESH = 0.7
    COMMAND_THRESHOLDS = {
    "go": 0.55,
    "stop": 0.8,
    "left": 0.58,
    "right": 0.66,
    "noise": 1.0
}
    
    LABELS = {"go": 0, "stop": 1, "left": 2, "right": 3, "noise": 4}
    COMMANDS = ["go", "stop", "left", "right", "noise"]
    
    GRID_SIZE = 20         
    BASE_FPS = 4           
    MAX_FPS = 8           
    SCORE_FILE = "game/best_score.txt"
    
    BG = (15, 15, 18)
    FIELD_BG = (22, 22, 26)
    PANEL_BG = (28, 28, 33)
    GRID = (35, 35, 42)
    SNAKE_BODY = (46, 204, 113)
    SNAKE_HEAD = (39, 174, 96)
    FOOD = (231, 76, 60)
    TEXT_MAIN = (236, 240, 241)
    TEXT_MUTED = (127, 140, 141)
    ACCENT_CMD = (52, 152, 219)
    ACCENT_WARN = (241, 196, 15)

# ─────────────────────────────────────────────
# МОДЕЛЬ
# ─────────────────────────────────────────────
class AudioCNN(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.pool(self.conv(x)))

# ─────────────────────────────────────────────
# ОБРАБОТКА ГОЛОСА (С ЛОГАМИ)
# ─────────────────────────────────────────────
class AudioEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AudioCNN().to(self.device)
        self.scaler = None
        self.audio_q = queue.Queue()
        self.command_q = queue.Queue()
        self.mic_level = 0.0
        self._load_assets()

    def _load_assets(self):
        try:
            if os.path.exists("models/best_model.pt"):
                self.model.load_state_dict(torch.load("models/best_model.pt", map_location=self.device))
                self.model.eval()
                print("✓ Модель весов загружена")
            if os.path.exists("models/scaler.pkl"):
                with open("models/scaler.pkl", "rb") as f: self.scaler = pickle.load(f)
                print("✓ Scaler загружен")
        except Exception as e:
            print(f"⚠ Ошибка ресурсов: {e}")

    def extract_features(self, audio):
        audio = audio.flatten().astype(np.float32)
        peak = np.max(np.abs(audio))
        if peak > 1e-6: audio /= peak
        mfcc = librosa.feature.mfcc(y=audio, sr=Config.SAMPLE_RATE, n_mfcc=Config.N_MFCC, hop_length=160)
        delta = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feat = np.concatenate([mfcc, delta, delta2], axis=0)
        T = feat.shape[1]
        return feat[:, :Config.MAX_FRAMES] if T >= Config.MAX_FRAMES else np.pad(feat, ((0,0),(0, Config.MAX_FRAMES-T)))

    def _mic_callback(self, indata, frames, time, status):
        self.audio_q.put(indata.copy())
        self.mic_level = float(np.sqrt(np.mean(indata**2)))

    def _worker(self):
        print(f"● Слушаю микрофон (VAD threshold: {Config.VAD_THRESHOLD})...")
        with sd.InputStream(samplerate=Config.SAMPLE_RATE, channels=1, blocksize=Config.BLOCK_SIZE, callback=self._mic_callback):
            while True:
                buf = []
                # Ожидание начала речи
                while True:
                    chunk = self.audio_q.get()
                    if self.mic_level > Config.VAD_THRESHOLD:
                        buf.append(chunk)
                        break
                
                # Запись фразы (примерно 0.8 - 1 сек)
                for _ in range(12): 
                    buf.append(self.audio_q.get())
                
                audio = np.concatenate(buf, axis=0)
                feat = self.extract_features(audio)
                
                if self.scaler: 
                    feat = self.scaler.transform(feat.reshape(1, -1)).reshape(1, 1, *feat.shape)
                else: 
                    feat = feat[np.newaxis, np.newaxis, :, :]
                
                x = torch.tensor(feat, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    output = self.model(x)
                    probs = torch.softmax(output, dim=1)[0]
                    pred = probs.argmax().item()
                    conf = probs[pred].item()
                    cmd_name = Config.COMMANDS[pred]


                    if conf >= Config.COMMAND_THRESHOLDS.get(cmd_name, 0.7) and cmd_name != "noise":
                        # Очищаем очередь, если там застряли старые команды (для мгновенной реакции)
                        while not self.command_q.empty(): self.command_q.get()
                        self.command_q.put((cmd_name, conf))

    def start(self):
        threading.Thread(target=self._worker, daemon=True).start()

# ─────────────────────────────────────────────
# ЛОГИКА ИГРЫ
# ─────────────────────────────────────────────
class Snake:
    def __init__(self, size):
        self.size = size
        self.reset()
    def reset(self):
        mid = self.size // 2
        self.body = [(mid, mid), (mid-1, mid), (mid-2, mid)]
        self.dir = (1, 0)
        self.grow = False
    def turn(self, cmd):
        dx, dy = self.dir
        if cmd == "left": self.dir = (dy, -dx)
        elif cmd == "right": self.dir = (-dy, dx)
    def step(self):
        hx, hy = self.body[0]
        dx, dy = self.dir
        new = ((hx + dx) % self.size, (hy + dy) % self.size)
        if new in self.body[:-1]: return False
        self.body.insert(0, new)
        if self.grow: self.grow = False
        else: self.body.pop()
        return True

# ─────────────────────────────────────────────
# ПРИЛОЖЕНИЕ
# ─────────────────────────────────────────────
class App:
    def __init__(self):
        pygame.init()
        self.info = pygame.display.Info()
        self.screen = pygame.display.set_mode((self.info.current_w, self.info.current_h), pygame.FULLSCREEN)
        
        game_zone_w = int(self.info.current_w * 0.75)
        self.side_w = self.info.current_w - game_zone_w
        self.cell = min(game_zone_w // (Config.GRID_SIZE + 4), self.info.current_h // (Config.GRID_SIZE + 4))
        
        self.field_rect = pygame.Rect(
            (game_zone_w - Config.GRID_SIZE * self.cell) // 2,
            (self.info.current_h - Config.GRID_SIZE * self.cell) // 2,
            Config.GRID_SIZE * self.cell, Config.GRID_SIZE * self.cell
        )

        self.audio = AudioEngine()
        self.audio.start()
        self.snake = Snake(Config.GRID_SIZE)
        self.food = self._get_food()
        self.score, self.best = 0, self._load_best()
        self.paused = False
        self.over = False
        self.last_cmd = "---"
        self.last_conf = 0.0
        
        self.font_l = pygame.font.SysFont("dejavusans", 50, bold=True)
        self.font_m = pygame.font.SysFont("dejavusans", 24)
        self.clock = pygame.time.Clock()

    def _get_food(self):
        while True:
            p = (random.randint(0, Config.GRID_SIZE-1), random.randint(0, Config.GRID_SIZE-1))
            if p not in self.snake.body: return p

    def _load_best(self):
        try: return int(open(Config.SCORE_FILE).read())
        except: return 0

    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()
                if e.type == pygame.KEYDOWN and e.key == pygame.K_r and self.over:
                    self.snake.reset(); self.over = False; self.score = 0

            # Получение команд
            while not self.audio.command_q.empty():
                c, conf = self.audio.command_q.get_nowait()
                self.last_cmd = c.upper()
                self.last_conf = conf
                
                if c == "stop": self.paused = True
                elif c == "go": 
                    if self.over: self.snake.reset(); self.over = False; self.score = 0
                    self.paused = False
                elif c in ["left", "right"] and not self.paused: 
                    self.snake.turn(c)

            if not self.paused and not self.over:
                if not self.snake.step(): self.over = True
                elif self.snake.body[0] == self.food:
                    self.snake.grow = True; self.score += 1; self.food = self._get_food()
                    if self.score > self.best:
                        self.best = self.score
                        with open(Config.SCORE_FILE, "w") as f: f.write(str(self.best))

            self.draw()
            self.clock.tick(min(Config.BASE_FPS + self.score // 3, Config.MAX_FPS))

    def draw(self):
        self.screen.fill(Config.BG)
        pygame.draw.rect(self.screen, Config.FIELD_BG, self.field_rect)
        
        # Сетка
        for i in range(Config.GRID_SIZE + 1):
            d = i * self.cell
            pygame.draw.line(self.screen, Config.GRID, (self.field_rect.x + d, self.field_rect.y), (self.field_rect.x + d, self.field_rect.bottom))
            pygame.draw.line(self.screen, Config.GRID, (self.field_rect.x, self.field_rect.y + d), (self.field_rect.right, self.field_rect.y + d))

        # Еда и Змейка
        pygame.draw.circle(self.screen, Config.FOOD, (self.field_rect.x + self.food[0]*self.cell + self.cell//2, self.field_rect.y + self.food[1]*self.cell + self.cell//2), self.cell//2 - 3)
        for i, (x, y) in enumerate(self.snake.body):
            color = Config.SNAKE_HEAD if i == 0 else Config.SNAKE_BODY
            pygame.draw.rect(self.screen, color, (self.field_rect.x + x*self.cell + 1, self.field_rect.y + y*self.cell + 1, self.cell - 2, self.cell - 2), border_radius=4)

        # Сайдбар
        px = self.info.current_w - self.side_w + 40
        self.screen.blit(self.font_l.render("AI SNAKE", True, Config.TEXT_MAIN), (px, 50))
        self.screen.blit(self.font_m.render(f"Счёт: {self.score}", True, Config.TEXT_MAIN), (px, 140))
        self.screen.blit(self.font_m.render(f"Рекорд: {self.best}", True, Config.TEXT_MUTED), (px, 180))
        
        # Инфо о предсказании на экране
        threshold = Config.COMMAND_THRESHOLDS.get(self.last_cmd.lower(), Config.CONFIDENCE_THRESH)
        if self.last_conf > threshold:
            cmd_color = Config.ACCENT_CMD      # уверенно
        elif self.last_conf > threshold * 0.7:
            cmd_color = Config.ACCENT_WARN     # сомнительно
        else:
            cmd_color = Config.TEXT_MUTED      # шум
        self.screen.blit(self.font_m.render(f"Голос: {self.last_cmd}", True, cmd_color), (px, 260))
        self.screen.blit(self.font_m.render(f"Conf: {self.last_conf:.2f}", True, Config.TEXT_MUTED), (px, 300))
        
        # Визуализация громкости
        vol_h = int(self.audio.mic_level * 1000)
        pygame.draw.rect(self.screen, (50, 50, 50), (px, 360, 150, 10))
        pygame.draw.rect(self.screen, Config.ACCENT_CMD, (px, 360, min(vol_h, 150), 10))

        if self.paused:
            s = self.font_l.render("ПАУЗА", True, Config.ACCENT_WARN)
            self.screen.blit(s, (self.field_rect.centerx - s.get_width()//2, self.field_rect.centery))
        if self.over:
            s = self.font_l.render("GAME OVER", True, Config.FOOD)
            self.screen.blit(s, (self.field_rect.centerx - s.get_width()//2, self.field_rect.centery))

        pygame.display.flip()

if __name__ == "__main__":
    App().run()