import pygame as pg
import numpy as np

from . import matrix_functions


class Camera:
    def __init__(self, renderer, position):
        self.renderer = renderer
        self.position = np.array([*position, 1.0])
        self.forward = np.array([0, 0, 1, 1])
        self.up = np.array([0, 1, 0, 1])
        self.right = np.array([1, 0, 0, 1])
        self.h_fov = np.pi / 3
        self.v_fov = self.h_fov * (renderer.HEIGHT / renderer.WIDTH)
        self.near_plane = 0.1
        self.far_plane = 100
        self.moving_speed = 0.04
        self.keyboard_rotation_speed = 0.015
        self.mouse_rotation_speed = 0.7

    def keyboard_control(self):
        key = pg.key.get_pressed()
        if key[pg.K_a]:
            # [obj.translate((-self.moving_speed, 0, 0)) for obj in self.renderer.objects]
            self.position -= self.right * self.moving_speed
        if key[pg.K_d]:
            # [obj.translate((self.moving_speed, 0, 0)) for obj in self.renderer.objects]
            self.position += self.right * self.moving_speed
        if key[pg.K_w]:
            # [obj.translate((0, self.moving_speed, 0)) for obj in self.renderer.objects]
            self.position += self.forward * self.moving_speed
        if key[pg.K_s]:
            # [obj.translate((0, -self.moving_speed, 0)) for obj in self.renderer.objects]
            self.position -= self.forward * self.moving_speed
        if key[pg.K_q]:
            # [obj.translate((0, 0, self.moving_speed)) for obj in self.renderer.objects]
            self.position += self.up * self.moving_speed
        if key[pg.K_e]:
            # [obj.translate((0, 0, -self.moving_speed)) for obj in self.renderer.objects]
            self.position -= self.up * self.moving_speed

        if key[pg.K_LEFT]:
            # [obj.rotate_z(-self.keyboard_rotation_speed) for obj in self.renderer.objects]
            self.camera_yaw(-self.keyboard_rotation_speed)
        if key[pg.K_RIGHT]:
            # [obj.rotate_z(self.keyboard_rotation_speed) for obj in self.renderer.objects]
            self.camera_yaw(self.keyboard_rotation_speed)
        if key[pg.K_UP]:
            # [obj.rotate_x(-self.keyboard_rotation_speed) for obj in self.renderer.objects]
            self.camera_pitch(-self.keyboard_rotation_speed)
        if key[pg.K_DOWN]:
            # [obj.rotate_x(self.keyboard_rotation_speed) for obj in self.renderer.objects]
            self.camera_pitch(self.keyboard_rotation_speed)
        if key[pg.K_LEFTBRACKET]:
            # [obj.rotate_y(self.keyboard_rotation_speed) for obj in self.renderer.objects]
            self.camera_roll(self.keyboard_rotation_speed)
        if key[pg.K_RIGHTBRACKET]:
            # [obj.rotate_z(-self.keyboard_rotation_speed) for obj in self.renderer.objects]
            self.camera_roll(-self.keyboard_rotation_speed)

        if key[pg.K_q]:
            self.renderer.stop()

    def mouse_control(self):
        if pg.mouse.get_focused():
            # mouse movements
            diff_width, diff_height = pg.mouse.get_pos()
            diff_width -= self.renderer.H_WIDTH
            diff_height -= self.renderer.H_HEIGHT
            diff_width /= self.renderer.H_WIDTH
            diff_height /= self.renderer.H_HEIGHT

            pg.mouse.set_pos((self.renderer.H_WIDTH, self.renderer.H_HEIGHT))

            self.camera_yaw(self.mouse_rotation_speed * diff_width)
            self.camera_pitch(self.mouse_rotation_speed * diff_height)

    def mouse_wheel(self, event):
        self.position += self.forward * self.moving_speed * event.y

    def camera_yaw(self, angle):
        rotate = matrix_functions.rotate(angle, self.up[:-1])
        self.rotate_all(rotate)

    def camera_pitch(self, angle):
        rotate = matrix_functions.rotate(angle, self.right[:-1])
        self.rotate_all(rotate)

    def camera_roll(self, angle: float):
        rotate = matrix_functions.rotate(angle, self.forward[:-1])
        self.rotate_all(rotate)

    def rotate_all(self, rotate_matrix: np.ndarray):
        self.forward = self.forward @ rotate_matrix
        self.right = self.right @ rotate_matrix
        self.up = self.up @ rotate_matrix

    @property
    def translate_matrix(self):
        x, y, z, w = self.position
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [-x, -y, -z, 1]
        ], dtype=float)

    @property
    def rotate_matrix(self):
        rx, ry, rz, w = self.right
        fx, fy, fz, w = self.forward
        ux, uy, uz, w = self.up
        return np.array([
            [rx, ux, fx, 0],
            [ry, uy, fy, 0],
            [rz, uz, fz, 0],
            [0, 0, 0, 1]
        ], dtype=float)

    @property
    def camera_matrix(self):
        return self.translate_matrix @ self.rotate_matrix

    def from_world_coords(self, matrix: np.ndarray):
        return matrix @ self.camera_matrix

    def to_world_coords(self, matrix: np.ndarray):
        return matrix @ np.linalg.inv(self.camera_matrix)
