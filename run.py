import logging

from game.app import App

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

if __name__ == "__main__":
    App().run()
