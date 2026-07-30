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
    IENpoint: np.ndarray
    IENpointElem: list
    boundNames: list
    npoints: int
    ne: int


def _blocos_do_tipo(raw: meshio.Mesh, tipo: str) -> list[int]:
    """Indices (na lista de cell blocks do meshio) dos blocos de um `tipo` (ex. 'triangle').

    Malhas do Gmsh podem ter blocos em qualquer ordem/quantidade -- ex.
    `lid.msh` tem um bloco extra `vertex` (1 no so, ponto de referencia de
    pressao para fixar o "zero" numa cavidade tampada sem saida fisica de
    fluido) antes do bloco `line`, o que quebrava a leitura por indice fixo
    (`raw.cells[0]`/`raw.cells[1]`) usada anteriormente.
    """
    return [i for i, cb in enumerate(raw.cells) if cb.type == tipo]


def _nomes_fisicos(raw: meshio.Mesh, boundNames: list, indices_blocos: list[int]) -> list:
    """Nome do physical group de cada celula dos blocos em `indices_blocos`, concatenados."""
    nomes = []
    for i in indices_blocos:
        tags = raw.cell_data["gmsh:physical"][i] - 1
        nomes.extend(boundNames[tag] for tag in tags)
    return nomes


def read_mesh(path: str) -> Mesh:
    """Le uma malha .msh (triangulos + arestas/pontos de contorno marcados por physical group)."""
    return _mesh_from_raw(meshio.read(path))


def _mesh_from_raw(raw: meshio.Mesh) -> Mesh:
    """Constroi o `Mesh` a partir de um `meshio.Mesh` ja lido -- separado de `read_mesh` pra
    poder testar a selecao de blocos por tipo sem precisar escrever/ler um `.msh` de verdade.

    Blocos de celula sao identificados pelo tipo (`triangle`, `line`,
    `vertex`), nao por posicao -- `IENpoint`/`IENpointElem` (do bloco
    `vertex`, opcional) cobrem contornos definidos por um unico no, como o
    ponto de referencia de pressao de `lid.msh`; ficam vazios se a malha
    nao tiver esse bloco (caso de `poiseuille.msh`/`degrau.msh`).
    """
    X = raw.points[:, 0]
    Y = raw.points[:, 1]
    boundNames = list(raw.field_data.keys())

    idx_tri = _blocos_do_tipo(raw, "triangle")
    IEN = np.concatenate([raw.cells[i].data for i in idx_tri], axis=0)

    idx_line = _blocos_do_tipo(raw, "line")
    IENbound = np.concatenate([raw.cells[i].data for i in idx_line], axis=0) if idx_line else np.empty((0, 2), dtype=int)
    IENboundElem = _nomes_fisicos(raw, boundNames, idx_line)

    idx_vertex = _blocos_do_tipo(raw, "vertex")
    IENpoint = np.concatenate([raw.cells[i].data for i in idx_vertex], axis=0).reshape(-1) if idx_vertex else np.empty(0, dtype=int)
    IENpointElem = _nomes_fisicos(raw, boundNames, idx_vertex)

    return Mesh(
        raw=raw,
        X=X,
        Y=Y,
        IEN=IEN,
        IENbound=IENbound,
        IENboundElem=IENboundElem,
        IENpoint=IENpoint,
        IENpointElem=IENpointElem,
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


def montar_node_to_elem(IEN: np.ndarray, npoints: int):
    """Monta, para cada no de vertice, o indice de um elemento que o contem.

    So cobre os `npoints` nos "reais" (vertices, sem o centroide/bolha
    do elemento MINI) -- e um chute inicial barato para busca por
    caminhada em malha (ex.: localizacao de ponto em metodos
    semi-Lagrangeanos), nao precisa ser o unico elemento vizinho.
    """
    node_to_elem = np.full(npoints, -1, dtype=int)
    for e, elem in enumerate(IEN):
        for a in elem:
            if node_to_elem[a] == -1:
                node_to_elem[a] = e

    return node_to_elem
