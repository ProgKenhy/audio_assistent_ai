import logging
import os
import random
import collections
import sys
import time

import numpy as np
import pygame

from .config import Config
from .audio import AudioEngine
from .snake import Snake
from .multi_user_adapter import MultiUserVoiceAdapter

logger = logging.getLogger(__name__)


class App:
    def __init__(self):
        pygame.init()

        self.user_id = self._select_user()

        self.info = pygame.display.Info()
        self.screen = pygame.display.set_mode(
            (self.info.current_w, self.info.current_h), pygame.FULLSCREEN)
        pygame.display.set_caption("AI Snake")

        game_zone_w = int(self.info.current_w * 0.75)
        self.side_w = self.info.current_w - game_zone_w
        self.cell = min(
            game_zone_w // (Config.GRID_SIZE + 4),
            self.info.current_h // (Config.GRID_SIZE + 4))
        self.field_rect = pygame.Rect(
            (game_zone_w - Config.GRID_SIZE * self.cell) // 2,
            (self.info.current_h - Config.GRID_SIZE * self.cell) // 2,
            Config.GRID_SIZE * self.cell,
            Config.GRID_SIZE * self.cell)

        self.audio = AudioEngine()
        self.audio.set_current_user(self.user_id)
        self.audio.start()

        self.snake = Snake(Config.GRID_SIZE)
        self.food = self._get_food()
        self.score = 0
        self.best = self._load_best()
        self.paused = False
        self.over = False

        self._history = collections.deque(maxlen=Config.HISTORY_LEN)

        self.font_l = pygame.font.SysFont("dejavusans", 50, bold=True)
        self.font_m = pygame.font.SysFont("dejavusans", 22)
        self.font_s = pygame.font.SysFont("dejavusans", 16)
        self.font_xs = pygame.font.SysFont("dejavusans", 14)
        self.clock = pygame.time.Clock()

    def _select_user(self) -> str:
        print("\n" + "=" * 50)
        print("AI SNAKE - ВЫБОР ПОЛЬЗОВАТЕЛЯ")
        print("=" * 50)

        adapter = MultiUserVoiceAdapter(save_dir=Config.USERS_DIR)
        users = adapter.get_user_list()

        if users:
            print("\nСуществующие пользователи:")
            for i, user in enumerate(users, 1):
                samples = adapter.get_stats().get(user, {}).get("samples", 0)
                print(f"  {i}. {user} ({samples} образцов)")
            print(f"  {len(users) + 1}. Создать нового пользователя")

            choice = input("\nВыберите номер: ").strip()
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(users):
                    user_id = users[choice_num - 1]
                    print(f"\nВыбран пользователь: {user_id}")
                    return user_id
                if choice_num == len(users) + 1:
                    return self._create_new_user()
            except ValueError:
                pass

        return self._create_new_user()

    def _create_new_user(self) -> str:
        print("\nСОЗДАНИЕ НОВОГО ПОЛЬЗОВАТЕЛЯ")
        user_id = input("Введите ваше имя: ").strip()
        if not user_id:
            user_id = f"user_{int(time.time())}"
        print(f"Создан пользователь: {user_id}")
        return user_id

    def _get_food(self) -> tuple[int, int]:
        while True:
            p = (random.randint(0, Config.GRID_SIZE - 1),
                 random.randint(0, Config.GRID_SIZE - 1))
            if p not in self.snake.body:
                return p

    def _load_best(self) -> int:
        path = Config.score_file(self.user_id)
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    return int(f.read().strip())
            except (OSError, ValueError):
                pass
        legacy = os.path.join(os.path.dirname(__file__), "best_score.txt")
        if os.path.isfile(legacy):
            try:
                with open(legacy) as f:
                    return int(f.read().strip())
            except (OSError, ValueError):
                pass
        return 0

    def _save_best(self):
        os.makedirs(Config.SCORES_DIR, exist_ok=True)
        with open(Config.score_file(self.user_id), "w") as f:
            f.write(str(self.best))

    def _quit(self):
        self.audio.stop()
        pygame.quit()
        sys.exit()

    def run(self):
        try:
            while True:
                self._handle_events()
                self._handle_commands()
                self._step_game()
                self.draw()
                self.clock.tick(
                    min(Config.BASE_FPS + self.score // 3, Config.MAX_FPS))
        finally:
            self.audio.stop()

    def _handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self._quit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self._quit()
                if e.key == pygame.K_r and self.over:
                    self._restart()

    def _handle_commands(self):
        while not self.audio.command_q.empty():
            data = self.audio.command_q.get_nowait()
            if len(data) == 5:
                label, conf, probs, accepted, speaker = data
            else:
                label, conf, probs, accepted = data
                speaker = None

            self._history.appendleft((label, conf, probs, accepted, speaker))

            if not accepted:
                continue

            if label == "stop":
                self.paused = True
            elif label == "go":
                if self.over:
                    self._restart()
                self.paused = False
            elif label in ("left", "right") and not self.paused:
                self.snake.turn(label)

    def _step_game(self):
        if self.paused or self.over:
            return
        if not self.snake.step():
            self.over = True
            return
        if self.snake.body[0] == self.food:
            self.snake.grow = True
            self.score += 1
            self.food = self._get_food()
            if self.score > self.best:
                self.best = self.score
                self._save_best()

    def _restart(self):
        self.snake.reset()
        self.food = self._get_food()
        self.score = 0
        self.over = False
        self.paused = False

    def draw(self):
        self.screen.fill(Config.BG)
        self._draw_field()
        self._draw_overlays()
        self._draw_sidebar()
        pygame.display.flip()

    def _draw_field(self):
        pygame.draw.rect(self.screen, Config.FIELD_BG, self.field_rect)

        for i in range(Config.GRID_SIZE + 1):
            d = i * self.cell
            pygame.draw.line(self.screen, Config.GRID,
                             (self.field_rect.x + d, self.field_rect.y),
                             (self.field_rect.x + d, self.field_rect.bottom))
            pygame.draw.line(self.screen, Config.GRID,
                             (self.field_rect.x, self.field_rect.y + d),
                             (self.field_rect.right, self.field_rect.y + d))

        pygame.draw.circle(
            self.screen, Config.FOOD,
            (self.field_rect.x + self.food[0] * self.cell + self.cell // 2,
             self.field_rect.y + self.food[1] * self.cell + self.cell // 2),
            self.cell // 2 - 3)

        for i, (x, y) in enumerate(self.snake.body):
            color = Config.SNAKE_HEAD if i == 0 else Config.SNAKE_BODY
            pygame.draw.rect(
                self.screen, color,
                (self.field_rect.x + x * self.cell + 1,
                 self.field_rect.y + y * self.cell + 1,
                 self.cell - 2, self.cell - 2),
                border_radius=4)

    def _draw_overlays(self):
        if self.paused:
            s = self.font_l.render("ПАУЗА", True, Config.ACCENT_WARN)
            self.screen.blit(s, (self.field_rect.centerx - s.get_width() // 2,
                                 self.field_rect.centery))
        if self.over:
            s = self.font_l.render("GAME OVER", True, Config.FOOD)
            self.screen.blit(s, (self.field_rect.centerx - s.get_width() // 2,
                                 self.field_rect.centery))

    def _draw_sidebar(self):
        px = self.info.current_w - self.side_w + 30
        sw = self.side_w - 50
        y = 50

        self.screen.blit(
            self.font_l.render("AI SNAKE", True, Config.TEXT_MAIN), (px, y))
        y += 68

        self.screen.blit(
            self.font_m.render(f"Счёт: {self.score}", True, Config.TEXT_MAIN),
            (px, y))
        y += 30
        self.screen.blit(
            self.font_m.render(f"Рекорд: {self.best}", True, Config.TEXT_MUTED),
            (px, y))
        y += 44

        self._hline(px, y, sw)
        y += 14

        if self._history:
            lbl, conf, probs, acc, speaker = self._history[0]
            row_col = self._row_color(lbl, acc)

            if speaker:
                speaker_surf = self.font_s.render(
                    f"[{speaker}]", True, Config.ACCENT_OK)
                self.screen.blit(speaker_surf, (px, y))
                y += 18

            self.screen.blit(
                self.font_s.render("Голос:", True, Config.TEXT_MUTED), (px, y))
            self.screen.blit(
                self.font_l.render(lbl.upper(), True, row_col), (px + 78, y - 6))
            y += 46

            thr = Config.COMMAND_THRESHOLDS.get(lbl, 0.7)
            self.screen.blit(
                self.font_s.render(
                    f"conf {conf:.0%}   порог {thr:.0%}", True, Config.TEXT_MUTED),
                (px, y))
            y += 24

            self._draw_prob_bars(px, y, sw, probs, acc, height=36, show_labels=True)
            y += 58
        else:
            self.screen.blit(
                self.font_m.render("Говорите...", True, Config.TEXT_MUTED), (px, y))
            y += 58

        self._hline(px, y, sw)
        y += 14

        self.screen.blit(
            self.font_s.render("МИК", True, Config.TEXT_MUTED), (px, y))
        y += 18
        vol_fill = min(int(self.audio.mic_level * 1800), sw)
        vol_col = (Config.ACCENT_OK
                   if self.audio.mic_level > Config.VAD_THRESHOLD
                   else (55, 55, 68))
        pygame.draw.rect(self.screen, (40, 40, 52), (px, y, sw, 8), border_radius=4)
        if vol_fill > 0:
            pygame.draw.rect(self.screen, vol_col, (px, y, vol_fill, 8), border_radius=4)
        y += 20

        current_user = self.audio.current_speaker
        if current_user and current_user in self.audio.multi_adapter.adapters:
            adapter_obj = self.audio.multi_adapter.adapters[current_user]
            n = adapter_obj.n_samples
            mx = adapter_obj.max_samples
        else:
            n = 0
            mx = 50

        self.screen.blit(
            self.font_xs.render(
                f"kNN адаптер: {n}/{mx}", True, Config.TEXT_MUTED), (px, y))
        y += 16
        pygame.draw.rect(self.screen, (40, 40, 52), (px, y, sw, 5), border_radius=3)
        if n:
            pygame.draw.rect(self.screen, Config.ACCENT_CMD,
                             (px, y, int(sw * n / mx), 5), border_radius=3)
        y += 20

        self._hline(px, y, sw)
        y += 12

        self.screen.blit(
            self.font_s.render("ИСТОРИЯ", True, Config.TEXT_MUTED), (px, y))
        y += 20

        for label, conf, probs, accepted, speaker in self._history:
            if y + 22 > self.info.current_h - 14:
                break
            row_col = self._row_color(label, accepted)

            self.screen.blit(
                self.font_s.render(f"{label.upper():<5}", True, row_col), (px, y))

            self.screen.blit(
                self.font_xs.render(f"{conf:.0%}", True, Config.TEXT_MUTED),
                (px + 52, y + 2))

            seg_w = max(1, (sw - 96) // len(Config.COMMANDS))
            bx = px + 90
            win_i = int(probs.argmax())
            for i, p in enumerate(probs):
                bc = (row_col if (i == win_i and accepted)
                      else Config.ACCENT_LOW if i == win_i
                      else (52, 52, 62))
                bh = max(1, int(p * 16))
                pygame.draw.rect(self.screen, bc,
                                 (bx + i * (seg_w + 2), y + 16 - bh, seg_w, bh),
                                 border_radius=1)
            y += 22

    def _hline(self, x: int, y: int, w: int):
        pygame.draw.line(self.screen, Config.GRID, (x, y), (x + w, y))

    def _row_color(self, label: str, accepted: bool) -> tuple[int, int, int]:
        if accepted:
            return Config.ACCENT_OK
        if label == "noise":
            return Config.ACCENT_NOISE
        return Config.ACCENT_LOW

    def _draw_prob_bars(self, px: int, y: int, sw: int,
                        probs: np.ndarray, accepted: bool,
                        height: int = 36, show_labels: bool = True):
        n = len(Config.COMMANDS)
        cell_w = sw // n
        win_i = int(probs.argmax())

        for i, (cmd, p) in enumerate(zip(Config.COMMANDS, probs)):
            bx = px + i * cell_w
            is_w = (i == win_i)
            bc = (Config.ACCENT_OK if (is_w and accepted)
                  else Config.ACCENT_LOW if is_w
                  else (55, 55, 68))
            bh = max(2, int(p * height))

            pygame.draw.rect(self.screen, (35, 35, 45),
                             (bx, y, cell_w - 2, height), border_radius=3)
            pygame.draw.rect(self.screen, bc,
                             (bx, y + height - bh, cell_w - 2, bh),
                             border_radius=3)

            if show_labels:
                pct = self.font_xs.render(f"{p:.0%}", True, bc)
                self.screen.blit(pct, (
                    bx + (cell_w - 2 - pct.get_width()) // 2,
                    y + height - bh - 14))
                lbl = self.font_xs.render(cmd, True, Config.TEXT_MUTED)
                self.screen.blit(lbl, (
                    bx + (cell_w - 2 - lbl.get_width()) // 2,
                    y + height + 2))
