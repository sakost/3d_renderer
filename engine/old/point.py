from __future__ import annotations

import numpy as np
import numpy.typing as npt


class Point:
    DIM = 3

    def __init__(self, x: float, y: float, z: float):
        self.data = np.array([float(x), float(y), float(z)], dtype=np.float64)

    @property
    def x(self):
        return self.data[0]

    @property
    def y(self):
        return self.data[1]

    @property
    def z(self):
        return self.data[2]

    @property
    def xy(self):
        return self.data[[0, 1]]

    @property
    def yz(self):
        return self.data[[1, 2]]

    @property
    def xz(self):
        return self.data[[0, 2]]

    @property
    def yx(self):
        return self.data[[1, 0]]

    @property
    def zx(self):
        return self.data[[2, 0]]

    @property
    def zy(self):
        return self.data[[2, 1]]

    @classmethod
    def from_xyz(cls, x: float, y: float, z: float) -> Point:
        return cls(x=x, y=y, z=z)

    @classmethod
    def from_point(cls, point: Point) -> Point:
        return cls(x=point.x, y=point.y, z=point.z)

    @classmethod
    def from_numpy(cls, data: npt.NDArray) -> Point:
        return cls(x=data[0], y=data[1], z=data[2])
