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


def elem_tri6(IEN: np.ndarray, X: np.ndarray, Y: np.ndarray):
    """Adiciona os nos de aresta (ponto medio) de cada elemento, gerando a
    conectividade do elemento Tri6 (velocidade P2, pressao P1 -- Taylor-Hood).

    Convencao de nos de aresta (Zienkiewicz vol. 1, cap. 8): v4 = aresta
    (v1,v2), v5 = aresta (v2,v3), v6 = aresta (v3,v1). Arestas compartilhadas
    entre elementos vizinhos reaproveitam o mesmo no -- nao duplicam, ao
    contrario do centroide do MINI (que e proprio de cada elemento).

    Retorna a IEN com 3 colunas extras (nos de aresta), X, Y com as
    coordenadas dos pontos medios ao final, e `edge_para_no`: dict
    (no_a, no_b) -> indice do no de aresta (chave ordenada, no_a<no_b) --
    usado para estender `IENbound` com o no de aresta de cada segmento de
    contorno (ver `estende_IENbound_tri6`).
    """
    npoints = len(X)
    ne = IEN.shape[0]

    IEN_new = np.hstack((IEN, np.zeros((ne, 3), dtype=IEN.dtype)))

    arestas_locais = [(0, 1), (1, 2), (2, 0)]  # v4, v5, v6
    edge_para_no = {}
    xs, ys = [], []
    proximo_no = npoints

    for e, elem in enumerate(IEN):
        for f, (a, b) in enumerate(arestas_locais):
            va, vb = int(elem[a]), int(elem[b])
            chave = (va, vb) if va < vb else (vb, va)
            no = edge_para_no.get(chave)
            if no is None:
                no = proximo_no
                edge_para_no[chave] = no
                xs.append((X[va] + X[vb]) / 2.0)
                ys.append((Y[va] + Y[vb]) / 2.0)
                proximo_no += 1
            IEN_new[e, 3 + f] = no

    X_new = np.concatenate([X, np.array(xs, dtype=X.dtype)])
    Y_new = np.concatenate([Y, np.array(ys, dtype=Y.dtype)])

    return IEN_new, X_new, Y_new, edge_para_no


def estende_IENbound_tri6(IENbound: np.ndarray, edge_para_no: dict) -> np.ndarray:
    """Insere o no de aresta (ponto medio) entre os dois vertices de cada
    segmento de contorno, para a condicao de contorno tambem valer sobre o
    grau de liberdade de velocidade do meio da aresta (Tri6).

    `edge_para_no` vem de `elem_tri6` -- todo segmento de `IENbound` e
    tambem uma aresta de algum triangulo do dominio, entao ja tem entrada
    no dict.

    Retorna um array (n_segmentos, 3): [v1, no_aresta, v2].
    """
    novo = np.empty((IENbound.shape[0], 3), dtype=IENbound.dtype)
    for i, (a, b) in enumerate(IENbound):
        va, vb = int(a), int(b)
        chave = (va, vb) if va < vb else (vb, va)
        novo[i, 0] = a
        novo[i, 1] = edge_para_no[chave]
        novo[i, 2] = b
    return novo


def area_com_sinal(IEN, X, Y):
    """Area orientada de cada elemento (positiva se os vertices estao anti-horarios).

    Existe para detectar **inversao de elemento** quando a malha se move
    (`moving_mesh.aplicar_oscilacao`). Isso importa porque um elemento
    invertido nao gera erro nenhum sozinho: `assembly.assemble_mini` toma
    `abs()` da area, mas os coeficientes `bi, ci` sao calculados sem
    `abs`, entao um elemento de orientacao trocada entra no sistema com
    `Gx`/`Gy` de sinal errado -- silenciosamente. A convencao do projeto
    e' `IEN[e, 0:3]` sempre anti-horario (ver `semi_lagrangian_tri`).

    Funciona com `torch` ou `numpy`.
    """
    v1, v2, v3 = IEN[:, 0], IEN[:, 1], IEN[:, 2]
    return 0.5 * (X[v1] * (Y[v2] - Y[v3]) + X[v2] * (Y[v3] - Y[v1]) + X[v3] * (Y[v1] - Y[v2]))


def atualiza_nos_extras(IEN, X, Y):
    """Reposiciona os nos extras do elemento a partir dos vertices, apos a malha se mover.

    Os nos acrescentados por `elem_mini`/`elem_tri6` sao definidos
    geometricamente pelos vertices do elemento (centroide, ponto medio de
    aresta). Quando os vertices se deslocam -- malha movel/ALE, ver
    `moving_mesh.aplicar_oscilacao` -- eles precisam ser recalculados, ou o
    elemento deixa de ser o que `assembly` assume que ele e' (o centroide
    do MINI, por exemplo, sairia de dentro do triangulo).

    Decide pelo numero de colunas de `IEN` (4 = MINI, 6 = Tri6). Funciona
    com `torch` ou `numpy` (so indexacao e media). Modifica `X`, `Y` no
    lugar e os devolve.
    """
    v1, v2, v3 = IEN[:, 0], IEN[:, 1], IEN[:, 2]

    if IEN.shape[1] == 4:  # MINI: centroide
        X[IEN[:, 3]] = (X[v1] + X[v2] + X[v3]) / 3.0
        Y[IEN[:, 3]] = (Y[v1] + Y[v2] + Y[v3]) / 3.0
    elif IEN.shape[1] == 6:  # Tri6: pontos medios das arestas (v4=v1v2, v5=v2v3, v6=v3v1)
        for col, (a, b) in enumerate(((v1, v2), (v2, v3), (v3, v1)), start=3):
            X[IEN[:, col]] = (X[a] + X[b]) / 2.0
            Y[IEN[:, col]] = (Y[a] + Y[b]) / 2.0
    else:
        raise ValueError(f"IEN com {IEN.shape[1]} colunas: esperado 4 (MINI) ou 6 (Tri6)")

    return X, Y


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


def montar_edge_to_elem(IEN: np.ndarray, npoints: int, n_extra: int):
    """Monta, para cada no de aresta do elemento Tri6, o indice de um elemento que o contem.

    Analogo a `montar_node_to_elem`, mas para os nos extras de aresta do
    Tri6 (colunas 3:6 de `IEN`, ja' estendida por `elem_tri6`) em vez dos
    vertices -- chute inicial para a busca por caminhada semi-Lagrangeana
    quando o pe da caracteristica parte de um no de aresta
    (`semi_lagrangian.calculo_sl(elemento="tri6")`).

    Ao contrario do centroide do MINI, o no de aresta e' compartilhado
    por ate 2 elementos (`elem_tri6` deduplica), sem um deles ser "o"
    dono natural -- por isso essa semente precisa ser construida
    varrendo a malha, e nao inferida por indice como
    `np.arange(ne)` faz para o centroide (`elem_start_extra` default de
    `calculo_sl`, valido so' para o MINI). Qualquer um dos elementos que
    compartilham a aresta serve como semente: a caminhada corrige a
    partir dai.

    Retorna um array de tamanho `n_extra`, indexado por `no - npoints`.
    """
    edge_to_elem = np.full(n_extra, -1, dtype=int)
    for e, elem in enumerate(IEN):
        for a in elem[3:6]:
            idx = a - npoints
            if edge_to_elem[idx] == -1:
                edge_to_elem[idx] = e

    return edge_to_elem
