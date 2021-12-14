from __future__ import annotations

from copy import deepcopy
from typing import TextIO

import numpy as np

from .triangle import TrianglePolygon
from .point import Point


class Mesh:
    def __init__(self, triangles: list[TrianglePolygon], copy_list: bool = True):
        if copy_list:
            triangles = deepcopy(triangles)

        self.triangles = triangles

    @classmethod
    def from_file_obj(cls, obj: TextIO) -> Mesh:
        vert_data = []
        triangle_indices = []

        for line in obj:
            tokens = line.split()
            if tokens[0] == "v":
                vert_data.append(np.array([float(val) for val in tokens[1:3+1]], dtype=float))
            elif tokens[0] == "f":
                line_indices = np.array([
                    int(token.split("/", maxsplit=1)[0])-1 for token in tokens[1:]
                ], dtype=int)
                triangle_indices.append(line_indices)

        mesh_data = []
        for idx in triangle_indices:
            triangle = TrianglePolygon(
                Point.from_numpy(vert_data[idx[0]]),
                Point.from_numpy(vert_data[idx[1]]),
                Point.from_numpy(vert_data[idx[2]]),
            )

            mesh_data.append(triangle)

        return cls(mesh_data, copy_list=False)
