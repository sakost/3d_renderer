from typing import TextIO

import pygame as pg
import numpy as np

from . import camera as camera_module
from . import projection as projection_module
from . import object_3d as object_3d_module
from . import axis as axis_module


class Renderer:
    def __init__(
        self,
        res: tuple[int, int] = (1600, 900),
        enable_mouse_control: bool = False,
        enable_keyboard_control: bool = True,
    ):
        self.RES = self.WIDTH, self.HEIGHT = res
        self.H_WIDTH, self.H_HEIGHT = self.WIDTH // 2, self.HEIGHT // 2
        self.FPS = 60
        self.screen = pg.display.set_mode(self.RES)
        self.clock = pg.time.Clock()

        self.projection = None
        self.camera = None
        self.objects: list[object_3d_module.Object3D] = []

        self._running = False
        self.mouse_control_flag = enable_mouse_control
        self.keyboard_control_flag = enable_keyboard_control

        self.create_objects()

        if self.mouse_control_flag:
            pg.mouse.set_visible(False)

    def create_objects(self):
        self.camera = camera_module.Camera(self, [-0.5, 1, -4])
        self.projection = projection_module.Projection(self)

        axis = axis_module.Axis(self)
        axis.translate([0.7, 0.9, 0.7])

        world_axis = axis_module.Axis(self)
        world_axis.movement_flag = False
        world_axis.translate([0.0001, 0.0001, 0.0001])

        # self.objects.append(axis)
        # self.objects.append(world_axis)

        # with open("resources/bunny.obj", "r") as f:
        #     obj = self.get_object_from_file(f)
        #
        # obj.scale(7)
        # #
        # self.objects.append(obj)

        wedge = object_3d_module.Wedge(self, light_coords=(10, 10, 10))

        wedge.rotate_y(np.pi / 4)
        wedge.translate([0.2, 0.4, 0.2])

        self.objects.append(wedge)

        # cube = object_3d_module.Cube(self, light_coords=(-10, -10, -10))
        # cube.rotate_y(-np.pi / 4)
        # cube.translate([0.2, 0.4, 0.2])

        # self.objects.append(cube)

    def get_object_from_file(self, file_obj: TextIO, use_robert_algorithm: bool = True):
        vertices, faces, faces_norms, vertices_norms = [], [], [], []
        for line in file_obj:
            if line.startswith('v '):
                vertices.append([float(i) for i in line.split()[1:]] + [1])
            elif line.startswith('vn'):
                vertices_norms.append([float(i) for i in line.split()[1:]] + [1])
            elif line.startswith('f '):
                face_vertices_raw = line.split()[1:]
                face = []
                norm = None
                for face_raw in face_vertices_raw:
                    face_raw_split = face_raw.split('/')
                    if len(face_raw_split) < 3:
                        vertex_num = int(face_raw_split[0]) - 1
                    else:
                        vertex_num, norm_vertex = map(lambda x: x-1, map(int, [face_raw_split[0], face_raw_split[2]]))
                        if norm is None:
                            norm = norm_vertex

                    face.append(vertex_num)

                faces.append(face)
                if norm is not None:
                    faces_norms.append(norm)
        if not vertices_norms and use_robert_algorithm:
            vertices_norms, faces_norms = object_3d_module.Object3D.roberts_algorithm(np.array(faces, dtype=object), np.array(vertices, dtype=np.float64))

        return object_3d_module.Object3D(self, vertices, faces, faces_norms, vertices_norms)

    def draw(self):
        self.screen.fill(pg.Color('darkslategray'))
        [obj.draw() for obj in self.objects]

    def loop(self):
        while self._running:
            self.draw()

            if self.keyboard_control_flag:
                self.camera.keyboard_control()

            if self.mouse_control_flag:
                self.camera.mouse_control()

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.stop()
                    break
                elif event.type == pg.MOUSEWHEEL and self.mouse_control_flag:
                    self.camera.mouse_wheel(event)

            pg.display.set_caption(str(self.clock.get_fps()))
            pg.display.flip()
            self.clock.tick(self.FPS)

    def stop(self):
        self._running = False

    def start(self):
        self._running = True
        self.loop()
