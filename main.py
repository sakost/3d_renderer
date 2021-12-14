from __future__ import annotations

import pygame as pg

from engine import Renderer


def download_bunny():
    import requests

    with open("resources/bunny.obj", "w") as file:
        for content in requests.get(
                "https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj"
                ).iter_content():
            file.write(content.decode())


def main():
    app = Renderer(
        (1600, 900),
        enable_mouse_control=True,
    )
    app.start()


if __name__ == '__main__':
    pg.init()
    main()
    pg.quit()
