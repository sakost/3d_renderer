from __future__ import annotations

import pygame as pg

from . import object_3d as object_3d_module
from . import renderer as renderer_module


class Axis(object_3d_module.Object3D):
    def __init__(self, render: renderer_module.Renderer):
        vertexes = [[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
        faces = [[0, 1], [0, 2], [0, 3]]
        colors = [pg.Color('red'), pg.Color('green'), pg.Color('blue')]

        super().__init__(
            render,
            vertexes,
            faces,
            [],
            [],
            colors=colors,
            draw_polygon=False,
            draw_wireframe=True,
            draw_vertexes=False,
            label="XYZ",
        )
