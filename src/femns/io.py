"""Escrita dos resultados da simulacao em VTK."""

import os

import meshio
import numpy as np


def write_vtk(path: str, points: np.ndarray, cells, point_data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meshio.write_points_cells(path, points, cells, point_data=point_data)
