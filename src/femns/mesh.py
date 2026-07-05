"""Leitura de malha e conectividade (elemento MINI, vizinhanca de nos/elementos)."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import meshio
import numpy as np


@dataclass
class Mesh:
    raw: meshio.Mesh
    X: np.ndarray
    Y: np.ndarray
    IEN: np.ndarray
    IENbound: np.ndarray
    IENboundElem: list
    boundNames: list
    npoints: int
    ne: int


def read_mesh(path: str) -> Mesh:
    """Le uma malha .msh (triangulos, Nr. 1) com arestas de contorno marcadas (Nr. 0)."""
    raw = meshio.read(path)

    X = raw.points[:, 0]
    Y = raw.points[:, 1]
    IEN = raw.cells[1].data
    IENbound = raw.cells[0].data
    IENboundTypeElem = list(raw.cell_data["gmsh:physical"][0] - 1)
    boundNames = list(raw.field_data.keys())
    IENboundElem = [boundNames[elem] for elem in IENboundTypeElem]

    return Mesh(
        raw=raw,
        X=X,
        Y=Y,
        IEN=IEN,
        IENbound=IENbound,
        IENboundElem=IENboundElem,
        boundNames=boundNames,
        npoints=len(X),
        ne=IEN.shape[0],
    )


def elem_mini(IEN: np.ndarray, X: np.ndarray, Y: np.ndarray):
    """Adiciona o no de centroide de cada elemento (bolha do elemento MINI).

    Retorna a IEN com uma coluna extra apontando para o centroide, e X, Y
    com as coordenadas dos centroides ao final.
    """
    npoints = len(X)
    ne = IEN.shape[0]

    IEN_new = np.hstack((IEN, np.zeros((ne, 1), dtype=IEN.dtype)))

    v1, v2, v3 = IEN[:, 0], IEN[:, 1], IEN[:, 2]

    xc = (X[v1] + X[v2] + X[v3]) / 3.0
    yc = (Y[v1] + Y[v2] + Y[v3]) / 3.0

    X_new = np.concatenate([X, xc])
    Y_new = np.concatenate([Y, yc])

    IEN_new[:, 3] = np.arange(npoints, npoints + ne)

    return IEN_new, X_new, Y_new


def montar_EToE(IEN: np.ndarray):
    """Monta a matriz de elementos vizinhos por face.

    Retorna:
        EToE: array (n_elementos, n_faces) onde EToE[e, f] e o elemento
              vizinho na face f do elemento e, ou -1 se f e contorno.
        faces_locais: definicao das faces locais (pares de indices de no).
    """
    n_elem, n_nos_elem = IEN.shape
    n_faces = n_nos_elem  # T3 -> 3 faces, Q4 -> 4 faces

    faces_locais = [(i, (i + 1) % n_nos_elem) for i in range(n_nos_elem)]

    face_para_elem = defaultdict(list)
    for e, elem in enumerate(IEN):
        for f, (a, b) in enumerate(faces_locais):
            face = tuple(sorted([elem[a], elem[b]]))
            face_para_elem[face].append((e, f))

    EToE = np.full((n_elem, n_faces), -1, dtype=int)

    for face, elems in face_para_elem.items():
        if len(elems) == 2:  # face interna
            (e1, f1), (e2, f2) = elems
            EToE[e1, f1] = e2
            EToE[e2, f2] = e1

    return EToE, faces_locais


def montar_NToN(IEN: np.ndarray, n_nos: int):
    """Monta, para cada no, a lista ordenada de nos vizinhos (compartilham elemento)."""
    vizinhos = [set() for _ in range(n_nos)]
    for elem in IEN:
        for a, b in combinations(elem, 2):
            vizinhos[a].add(b)
            vizinhos[b].add(a)

    NToN = np.empty(n_nos, dtype=object)
    for i, v in enumerate(vizinhos):
        NToN[i] = np.array(sorted(v), dtype=int)

    return NToN
