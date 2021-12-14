import math
import numpy as np


class Projection:
    def __init__(self, render):
        NEAR = render.camera.near_plane
        FAR = render.camera.far_plane
        RIGHT = math.tan(render.camera.h_fov / 2)
        LEFT = -RIGHT
        TOP = math.tan(render.camera.v_fov / 2)
        BOTTOM = -TOP

        m00 = 2 / (RIGHT - LEFT)
        m11 = 2 / (TOP - BOTTOM)
        m22 = (FAR + NEAR) / (FAR - NEAR)
        m32 = -2 * NEAR * FAR / (FAR - NEAR)
        self._projection_matrix = np.array([
            [m00, 0, 0, 0],
            [0, m11, 0, 0],
            [0, 0, m22, 1],
            [0, 0, m32, 0]
        ], dtype=float)

        HW, HH = render.H_WIDTH, render.H_HEIGHT
        self._to_screen_matrix = np.array([
            [HW, 0, 0, 0],
            [0, -HH, 0, 0],
            [0, 0, 1, 0],
            [HW, HH, 0, 1]
        ], dtype=float)

    def to_ndc_matrix(self, matrix: np.ndarray):
        matrix = matrix @ self._projection_matrix
        matrix /= matrix[:, -1].reshape(-1, 1)
        return matrix

    def to_screen_matrix(self, matrix: np.ndarray):
        return (matrix @ self._to_screen_matrix)[:, :2]
