# 🐍 Snake.md — Техническая документация игры

---

## 1. Общая архитектура игры

Игра — это не просто “змейка”, а интерактивная оболочка для голосового управления.

Важно понимать:

👉 змейка здесь — это демонстрационный интерфейс, а не основная цель проекта
👉 основная задача — реакция на голос в реальном времени

### Архитектура:

```
AudioEngine (голос)
        ↓
command_q (очередь команд)
        ↓
App (игра)
        ↓
Snake (логика)
        ↓
Renderer (pygame)
```

### Почему именно так?

Такое разделение даёт:

* независимость аудио и игры
* отсутствие лагов UI
* масштабируемость (можно заменить змейку на любое приложение)

---

## 2. Главный цикл игры (Game Loop)

Класс `App` — это центр управления.

### Основной цикл:

```python
while True:
    self._handle_events()
    self._handle_commands()
    self._step_game()
    self.draw()
    self.clock.tick(...)
```

### Разбор по шагам

#### 1. _handle_events()

Обрабатывает:

* закрытие окна
* ESC → выход
* R → рестарт

👉 стандартный pygame event loop

---

#### 2. _handle_commands() ⚡ КЛЮЧЕВОЙ МОМЕНТ

```python
while not self.audio.command_q.empty():
    label, conf, probs, accepted = ...
```

Здесь игра получает команды из AudioEngine

### Почему очередь?

Потому что:

* аудио работает в другом потоке
* нельзя блокировать основной поток игры

👉 очередь = безопасная синхронизация потоков

### Логика применения команды:

```python
if not accepted:
    continue
```

👉 отсеиваем шум

Далее:

```python
if label == "stop":
    self.paused = True
elif label == "go":
    self.paused = False
elif label in ("left", "right"):
    self.snake.turn(label)
```

### Важно ⚠️

Команда применяется только если `accepted=True`

👉 игра НЕ доверяет модели полностью
👉 используется фильтрация через threshold

---

## 3. Логика змейки (Snake)

### 3.1 Представление змейки

```python
self.body = [(x1,y1), (x2,y2), ...]
```

👉 список координат

Почему список?

* легко добавлять голову (`insert(0)`)
* легко удалять хвост (`pop()`)

---

### 3.2 Направление

```python
self.dir = (dx, dy)
```

| Направление | Вектор  |
| ----------- | ------- |
| вправо      | (1, 0)  |
| влево       | (-1, 0) |
| вверх       | (0, -1) |
| вниз        | (0, 1)  |

👉 векторная модель движения

---

### 3.3 Поворот (математически красиво 🔥)

```python
if cmd == "left":
    self.dir = (dy, -dx)
elif cmd == "right":
    self.dir = (-dy, dx)
```

👉 поворот вектора на 90°

---

## 4. Движение змейки

### step()

```python
new = ((hx + dx) % size, (hy + dy) % size)
```

👉 телепорт через стены

### Почему `% size`?

* упрощает логику
* делает управление forgiving

---

### Проверка столкновения

```python
if new in self.body[:-1]:
    return False
```

👉 столкновение с собой = game over

---

### Рост

```python
if self.grow:
    self.grow = False
else:
    self.body.pop()
```

👉 либо растём, либо двигаемся

---

## 5. Еда (Food)

### Генерация

```python
while True:
    p = random position
    if p not in snake.body:
        return p
```

👉 еда не появляется внутри змейки

---

## 6. Скорость игры

```python
self.clock.tick(min(BASE_FPS + score//3, MAX_FPS))
```

👉 сложность растёт со временем
👉 есть потолок скорости

---

## 7. Пауза и Game Over

### Пауза

```python
if self.paused:
    skip update
```

### Game Over

```python
if not self.snake.step():
    self.over = True
```

---

## 8. Система истории команд 📊

```python
deque(maxlen=7)
```

### Почему deque?

* быстрые операции
* авто-ограничение размера

### Что хранится:

```python
(label, conf, probs, accepted)
```

---

## 9. Визуализация (pygame)

### 9.1 UI структура

```
| игровое поле | панель |
```

---

### 9.2 Игровое поле

* змейка → rect
* еда → circle
* сетка → lines

---

### 9.3 Панель

* последняя команда
* confidence
* probability bars 📊

👉 видно поведение модели

---

### 9.4 Микрофон 🎤

```python
self.audio.mic_level
```

👉 уровень сигнала

---

### 9.5 kNN адаптация

```
kNN: 12/50
```

👉 сколько примеров собрано

---

## 10. Почему игра стабильна

* асинхронность
* очередь команд
* threshold фильтрация
* простая физика
* forgiving дизайн

---

## 11. Ограничения

❌ нет режимов игры

---

## 12. Потенциал расширения 🚀

* разные игры (shooter, platformer)
* дообучение модели
* мультиплеер
* новые команды

---

## 13. Сборка ⚙️

```bash
pyinstaller --noconfirm --clean \
--add-data "models:models" \
--hidden-import sounddevice \
--hidden-import librosa \
--hidden-import sklearn \
run.py
```

---

## 14. Итог 🎯

Игра — это:

👉 интерфейс для AI
👉 система тестирования аудиомодели
👉 real-time демонстрация ML
