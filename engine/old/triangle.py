from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt

from .point import Point
from .color import MAX_COLOR_VALUE, MIN_COLOR_VALUE


class TrianglePolygon:
    DEFAULT_COLOR = np.array((MAX_COLOR_VALUE, MAX_COLOR_VALUE, MAX_COLOR_VALUE))

    __slots__ = ("color", "_norm_color", "_norm", "data")

    def __init__(self, p1: Point, p2: Point, p3: Point, color: Optional[npt.NDArray] = None):
        if color is None:
            color = self.DEFAULT_COLOR.copy()

        self.color = color

        self._norm_color = None
        self._norm = None

        self.data = np.array([p1.data, p2.data, p3.data])

    def clear_cache(self):
        self._norm_color = None
        self._norm = None

    @classmethod
    def from_points(cls, p1: Point, p2: Point, p3: Point) -> TrianglePolygon:
        return cls(p1=p1, p2=p2, p3=p3)

    @property
    def norm_color(self):
        if self._norm_color is None:
            self._norm_color = self.color / MAX_COLOR_VALUE
        return self._norm_color

    @property
    def p1(self):
        return self.data[0]

    @property
    def p2(self):
        return self.data[1]

    @property
    def p3(self):
        return self.data[2]

    @property
    def norm(self):
        if self._norm is None:
            u = self.p2 - self.p1
            v = self.p3 - self.p1

            self._norm = np.cross(u, v)
            self._norm /= np.linalg.norm(self._norm)

        return self._norm

    def shade(self, val: float) -> npt.NDArray:
        return np.clip(self.color * val, MIN_COLOR_VALUE, MAX_COLOR_VALUE).astype(dtype=int)
