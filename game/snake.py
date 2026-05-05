class Snake:
    def __init__(self, size: int):
        self.size = size
        self.reset()

    def reset(self):
        mid       = self.size // 2
        self.body = [(mid, mid), (mid - 1, mid), (mid - 2, mid)]
        self.dir  = (1, 0)
        self.grow = False

    def turn(self, cmd: str):
        dx, dy = self.dir
        if cmd == "left":
            self.dir = (dy, -dx)
        elif cmd == "right":
            self.dir = (-dy, dx)

    def step(self) -> bool:
        hx, hy = self.body[0]
        dx, dy = self.dir
        new    = ((hx + dx) % self.size, (hy + dy) % self.size)
        if new in self.body[:-1]:
            return False
        self.body.insert(0, new)
        if self.grow:
            self.grow = False
        else:
            self.body.pop()
        return True