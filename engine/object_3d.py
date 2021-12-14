from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pygame as pg

from . import matrix_functions
from . import renderer as renderer_module


class Object3D:
    DEFAULT_COLOR = pg.Color("white")
    SHADOW_COLOR_COEFFICIENT = 0.5

    def __init__(
        self,
        renderer: renderer_module.Renderer,
        vertices: list[list[float]],
        faces: list[list[int]],
        faces_norms: list[int],
        vertices_norms: list[list[float]],
        light_coords: tuple[float, float, float] = (-1, -1, -6),
        colors: Optional[Union[list[pg.Color], pg.Color]] = None,
        font_labels: Optional[pg.font.Font] = None,
        label: Optional[str] = None,
        is_moving_object: bool = False,
        draw_vertexes: bool = False,
        draw_wireframe: bool = False,
        draw_polygon: bool = True,
        draw_norms: bool = False,
        use_roberts_algorithm: bool = False,
    ):
        self.renderer = renderer

        self.vertices = np.array(vertices)
        self.faces = np.array([np.array(face, dtype=int) for face in faces], dtype=object)

        if not vertices_norms and use_roberts_algorithm:
            vertices_norms, faces_norms = self.roberts_algorithm(self.faces, self.vertices)

        self.vertices_norms = np.array(vertices_norms)
        self.faces_norms = np.array(faces_norms, dtype=int)

        self.light_coords = np.array(light_coords, dtype=int)

        self.translate([0.0001, 0.0001, 0.0001])

        if font_labels is None:
            font_labels = pg.font.SysFont('Arial', 30, bold=True)
        self.font = font_labels

        if isinstance(colors, list):
            self.color_faces = list(zip(colors, self.faces))
        else:
            color = self.DEFAULT_COLOR
            if isinstance(colors, pg.Color):
                color = colors

            self.color_faces = [(color, face) for face in self.faces]

        (
            self.draw_wireframe_flag,
            self.draw_polygon_flag,
            self.movement_flag,
            self.draw_vertexes_flag,
            self.draw_norms_flag,
        ) = (
            draw_wireframe,
            draw_polygon,
            is_moving_object,
            draw_vertexes,
            draw_norms,
        )

        if label is None:
            label = ''
        self.label = label

    def draw(self):
        self.screen_projection(False)
        if self.movement_flag:
            self.movement()

    def movement(self):
        pass

    def calculate_screen_indices(self, matrix: np.ndarray, orthographic: bool = False):
        world_coords = self.renderer.camera.from_world_coords(matrix.astype(np.float64))
        if orthographic:
            world_coords = world_coords.copy()
            world_coords[:, 2] = self.renderer.camera.near_plane + 0.001
        ndc_coords = self.renderer.projection.to_ndc_matrix(world_coords)
        indices = np.all(~((ndc_coords > 2) | (ndc_coords < -2)), axis=-1)
        return self.renderer.projection.to_screen_matrix(ndc_coords), indices

    def screen_projection(self, orthographic: bool = False):
        vertices, vertices_indices = self.calculate_screen_indices(self.vertices, orthographic=orthographic)

        for index, (color, face) in enumerate(self.color_faces):
            face = face.astype(int)

            polygon_indices = vertices_indices[face]
            if not np.all(polygon_indices):
                continue

            polygon = vertices[face]

            if self.draw_polygon_flag:
                self.draw_face(color, index, polygon, orthographic=orthographic)

                if self.draw_norms_flag:
                    self.draw_norm(index, orthographic=orthographic)

            if self.draw_wireframe_flag:
                pg.draw.polygon(self.renderer.screen, color, polygon, 1)

            if self.label:
                text = self.font.render(self.label[index], True, pg.Color('white'))
                self.renderer.screen.blit(text, polygon[-1])

        if self.draw_vertexes_flag:
            for vertex in vertices:
                pg.draw.circle(self.renderer.screen, pg.Color('white'), vertex, 2)

    def draw_face(self, color, face_index, polygon, orthographic: bool = False):
        face_indices = self.faces[face_index].astype(int)

        face_norm_idx = int(self.faces_norms[face_index])
        norm = self.vertices_norms[face_norm_idx].astype(float)[:-1]

        # cast to unit vectors
        norm /= np.linalg.norm(norm)
        light_direction = self.light_coords / np.linalg.norm(self.light_coords)

        # must divided by (|norm| * |light_direction|) but they are already unit vectors
        cos_theta_light = np.dot(norm, light_direction)

        means_3d = self.vertices[face_indices].mean(
            axis=-2,
            keepdims=True,
            dtype=np.float64,
        ).reshape(4)

        forward = (self.renderer.camera.position - means_3d)[:-1]
        forward /= np.linalg.norm(forward)

        # must divided by (|norm| * |forward|) but they are already unit vectors
        cos_theta = np.dot(norm, forward)
        theta = np.arccos(cos_theta)

        if theta < np.pi / 2:
            color = np.array(color, dtype=np.float)
            color[:-1] = np.clip(
                color[:-1] - color[:-1] * cos_theta_light * type(self).SHADOW_COLOR_COEFFICIENT, 0,
                255
                )
            color = color.astype(np.uint8)

            pg.draw.polygon(self.renderer.screen, pg.Color(list(color)), polygon, 0)

    def draw_norm(self, face_index, orthographic: bool = False):
        face_indices = self.faces[face_index].astype(int)

        means_3d = self.vertices[face_indices].mean(
            axis=-2,
            keepdims=True,
            dtype=np.float64,
        )

        norm_3d = self.vertices_norms[self.faces_norms[face_index]].astype(np.float64).reshape(
            (1, 4)
            )

        norm_3d[:, :-1] /= np.linalg.norm(norm_3d[:, :-1])

        norm_3d[:, :-1] += means_3d[:, :-1]

        means, means_indices = self.calculate_screen_indices(means_3d, orthographic=orthographic)
        norm, norm_index = self.calculate_screen_indices(norm_3d, orthographic=orthographic)

        if np.all(means_indices) & np.all(norm_index):
            pg.draw.line(self.renderer.screen, pg.Color("white"), means[0], norm[0])


    def translate(self, pos):
        translate_matrix = matrix_functions.translate(pos)
        self.vertices = self.vertices @ translate_matrix

    def scale(self, scale_to):
        scale_matrix = matrix_functions.scale(scale_to)
        self.vertices = self.vertices @ scale_matrix
        if self.vertices_norms.size > 0:
            self.vertices_norms = self.vertices_norms @ scale_matrix

    def rotate_x(self, angle):
        rotate_x_matrix = matrix_functions.rotate_x(angle)
        self.vertices = self.vertices @ rotate_x_matrix
        if self.vertices_norms.size > 0:
            self.vertices_norms = self.vertices_norms @ rotate_x_matrix

    def rotate_y(self, angle):
        rotate_y_matrix = matrix_functions.rotate_y(angle)
        self.vertices = self.vertices @ rotate_y_matrix
        if self.vertices_norms.size > 0:
            self.vertices_norms = self.vertices_norms @ rotate_y_matrix

    def rotate_z(self, angle):
        rotate_z_matrix = matrix_functions.rotate_z(angle)
        self.vertices = self.vertices @ rotate_z_matrix
        if self.vertices_norms.size > 0:
            self.vertices_norms = self.vertices_norms @ rotate_z_matrix

    def rotate(self, angle: float, vector: np.ndarray):
        rotate_matrix = matrix_functions.rotate(angle, vector)
        self.vertices = self.vertices @ rotate_matrix
        if self.vertices_norms.size > 0:
            self.vertices_norms = self.vertices_norms @ rotate_matrix

    @staticmethod
    def roberts_algorithm(faces: np.ndarray, vertices: np.ndarray) -> tuple[list[list[float]], list[int]]:
        faces_norms: list[int] = []
        vertices_norms: list[list[float]] = []

        W = vertices.mean(axis=-2)

        for idx, face in enumerate(faces):
            face = np.array(face).astype(int)
            face_vertices = vertices[face]

            if face_vertices.shape[0] < 3:
                raise ValueError("face must be consist of at least 3 vertices")

            vec1 = face_vertices[1] - face_vertices[0]
            vec2 = face_vertices[2] - face_vertices[0]

            A, B, C = np.cross(vec1[:-1], vec2[:-1])
            D = -(A * face_vertices[0][0] + B * face_vertices[0][1] + C * face_vertices[0][2])

            sign = -(A * W[0] + B * W[1] + C * W[2] + D)
            sign /= abs(sign)

            A *= sign
            B *= sign
            C *= sign
            D *= sign

            vertices_norms.append([A, B, C, 1])
            faces_norms.append(idx)

        return vertices_norms, faces_norms


class Cube(Object3D):
    def __init__(self, renderer: renderer_module.Renderer, **kwargs):
        vertices = [
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 1],
        ]
        faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 4, 5, 1],
            [2, 3, 7, 6],
            [1, 2, 6, 5],
            [0, 3, 7, 4],
        ]
        faces_norms = [0, 1, 2, 3, 4, 5]
        vertices_norms = [
            [0, 0, -1, 1],
            [0, 0, 1, 1],
            [-1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, -1, 0, 1],
        ]

        super().__init__(renderer, vertices, faces, faces_norms, vertices_norms, **kwargs)


class Wedge(Object3D):
    def __init__(self, renderer: renderer_module.Renderer, **kwargs):
        vertices = [
            [0, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            [0, 0, 1, 1],

            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ]

        faces = [
            [0, 1, 2, 3],
            [2, 3, 4, 5],
            [0, 4, 5, 1],
            [0, 4, 3],
            [1, 5, 2],
        ]

        vertices_norms, faces_norms = self.roberts_algorithm(
            np.array(faces, dtype=object),
            np.array(vertices, dtype=np.float64),
        )

        super(Wedge, self).__init__(
            renderer,
            vertices=vertices,
            faces=faces,
            faces_norms=faces_norms,
            vertices_norms=vertices_norms,
            draw_wireframe=False,
            draw_polygon=True,
            draw_norms=True,
        )
