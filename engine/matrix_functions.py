import numpy as np


def translate(pos: tuple[float, float, float]):
    tx, ty, tz = pos
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [tx, ty, tz, 1],
    ])


def rotate(angle: float, vector: np.ndarray):
    vector = vector.reshape(3).astype(float)
    vector /= np.linalg.norm(vector)

    W = np.array([
        [0, -vector[2], vector[1], 0],
        [vector[2], 0, -vector[0], 0],
        [-vector[1], vector[0], 0, 0],
        [0, 0, 0, 1],
    ])
    return (np.eye(W.shape[0]) + np.sin(angle) * W + (2 * np.sin(angle/2) ** 2) * W @ W).T


def rotate_x(angle: float):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle), np.sin(angle), 0],
        [0, -np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1],
    ])


def rotate_y(angle: float):
    return np.array([
        [np.cos(angle), 0, -np.sin(angle), 0],
        [0, 1, 0, 0],
        [np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1],
    ])


def rotate_z(angle: float):
    return np.array([
        [np.cos(angle), np.sin(angle), 0, 0],
        [-np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])


def scale(coefficient: float):
    return np.array([
        [coefficient, 0, 0, 0],
        [0, coefficient, 0, 0],
        [0, 0, coefficient, 0],
        [0, 0, 0, 1],
    ])
