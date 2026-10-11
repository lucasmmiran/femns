"""Leitura de malha e conectividade (elemento MINI, vizinhanca de nos/elementos)."""

from dataclasses import dataclass
from itertools import combinations

import meshio
import numpy as np

from femns.io import blocos_do_tipo, conectividade


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


DIMENSAO_DO_BLOCO = {"vertex": 0, "line": 1, "triangle": 2}


def nome_por_tag(raw: meshio.Mesh) -> dict:
    """`{(dimensao, tag): nome}` dos physical groups do gmsh.

    O nome sai do numero do grupo (`field_data[nome] = [tag, dim]`), e nao da
    posicao dele em `field_data`: o gmsh nao garante tags 1..N na ordem de
    declaracao (ex.: `inner_wall` com tag 9 declarado depois de `surface`, tag
    8), e um mesmo numero pode existir em dimensoes diferentes.
    """
    return {(int(dim), int(tag)): nome for nome, (tag, dim) in raw.field_data.items()}


def _nomes_fisicos(raw: meshio.Mesh, indices_blocos: list[int]) -> list:
    """Nome do physical group de cada celula dos blocos em `indices_blocos`, concatenados."""
    por_tag = nome_por_tag(raw)
    nomes = []
    for i in indices_blocos:
        dim = DIMENSAO_DO_BLOCO[raw.cells[i].type]
        nomes.extend(por_tag[(dim, int(tag))] for tag in raw.cell_data["gmsh:physical"][i])
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

    IEN = conectividade(raw.cells, "triangle")

    IENbound = conectividade(raw.cells, "line", obrigatorio=False)
    IENboundElem = _nomes_fisicos(raw, blocos_do_tipo(raw.cells, "line"))

    IENpoint = conectividade(raw.cells, "vertex", obrigatorio=False).reshape(-1)
    IENpointElem = _nomes_fisicos(raw, blocos_do_tipo(raw.cells, "vertex"))

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


@dataclass
class Arestas:
    """Topologia de arestas de uma malha de triangulos (so os 3 vertices).

    Arestas numeradas em ordem de **primeira aparicao** varrendo `IEN`
    elemento a elemento, face a face -- a mesma ordem em que o antigo laco de
    `elem_tri6` criava os nos de aresta, para a numeracao do Tri6 nao mudar.
    Face local `f` de um elemento liga os vertices locais `f` e `(f+1)%3`
    (`(0,1), (1,2), (2,0)`): e' a mesma convencao de `montar_EToE` e da coluna
    `3+f` do Tri6, agora por construcao (vem daqui).

    - `vertices` (na, 2): os dois vertices de cada aresta, `va < vb`.
    - `id` (ne, 3): numero da aresta da face `f` do elemento `e`.
    - `elementos`, `faces` (na, 2): os (ate) 2 elementos que compartilham a
      aresta e a face local dela em cada um; `-1` na 2a coluna para aresta de
      contorno.
    """

    vertices: np.ndarray
    id: np.ndarray
    elementos: np.ndarray
    faces: np.ndarray
    _chaves: np.ndarray  # chave `va*base + vb` de cada aresta, ordenada
    _ordem: np.ndarray   # numero da aresta de cada chave ordenada
    _base: int

    def indice(self, va: np.ndarray, vb: np.ndarray) -> np.ndarray:
        """Numero da aresta de cada par de vertices (em qualquer ordem), vetorizado."""
        va, vb = np.asarray(va), np.asarray(vb)
        chave = np.minimum(va, vb).astype(np.int64) * self._base + np.maximum(va, vb)
        pos = np.searchsorted(self._chaves, chave)
        pos = np.minimum(pos, len(self._chaves) - 1)
        if not np.array_equal(self._chaves[pos], chave):
            raise KeyError("par de vertices que nao e' aresta da malha")
        return self._ordem[pos]


def topologia_arestas(IEN: np.ndarray) -> Arestas:
    """Monta a topologia de arestas (`Arestas`) dos 3 primeiros nos de `IEN`, vetorizado."""
    IEN3 = np.asarray(IEN)[:, :3].astype(np.int64)
    ne = IEN3.shape[0]
    base = int(IEN3.max()) + 1

    a = IEN3                       # vertice local f
    b = IEN3[:, [1, 2, 0]]         # vertice local (f+1)%3
    lo, hi = np.minimum(a, b).ravel(), np.maximum(a, b).ravel()
    chave = lo * base + hi         # um valor por (elemento, face), em ordem de varredura

    chaves, primeiro, inv = np.unique(chave, return_index=True, return_inverse=True)
    na = len(chaves)
    ordem_aparicao = np.argsort(primeiro, kind="stable")   # arestas na ordem da 1a aparicao
    numero = np.empty(na, dtype=np.int64)
    numero[ordem_aparicao] = np.arange(na)                  # chave ordenada -> numero da aresta
    aresta_do_slot = numero[inv]                            # (ne*3,)

    vertices = np.stack([lo[primeiro], hi[primeiro]], axis=1)[ordem_aparicao]

    # elementos/faces de cada aresta: agrupa os slots (e, f) por aresta
    slots = np.argsort(aresta_do_slot, kind="stable")       # slots em ordem crescente de (e, f) dentro da aresta
    contagem = np.bincount(aresta_do_slot, minlength=na)
    inicio = np.concatenate([[0], np.cumsum(contagem)[:-1]])
    elementos = np.full((na, 2), -1, dtype=np.int64)
    faces = np.full((na, 2), -1, dtype=np.int64)
    s1 = slots[inicio]
    elementos[:, 0], faces[:, 0] = s1 // 3, s1 % 3
    dupla = contagem == 2         # aresta interna (contagem > 2 seria malha nao-manifold: fica sem vizinho)
    s2 = slots[inicio[dupla] + 1]
    elementos[dupla, 1], faces[dupla, 1] = s2 // 3, s2 % 3

    return Arestas(vertices=vertices, id=aresta_do_slot.reshape(ne, 3), elementos=elementos,
                   faces=faces, _chaves=chaves, _ordem=numero, _base=base)


def elem_tri6(IEN: np.ndarray, X: np.ndarray, Y: np.ndarray):
    """Adiciona os nos de aresta (ponto medio) de cada elemento, gerando a
    conectividade do elemento Tri6 (velocidade P2, pressao P1 -- Taylor-Hood).

    Convencao de nos de aresta (Zienkiewicz vol. 1, cap. 8): v4 = aresta
    (v1,v2), v5 = aresta (v2,v3), v6 = aresta (v3,v1). Arestas compartilhadas
    entre elementos vizinhos reaproveitam o mesmo no -- nao duplicam, ao
    contrario do centroide do MINI (que e proprio de cada elemento). O no da
    aresta `k` de `topologia_arestas` e' `npoints + k`.

    Retorna a IEN com 3 colunas extras (nos de aresta), X, Y com as
    coordenadas dos pontos medios ao final, e a `Arestas` da malha (usada por
    `estende_IENbound_tri6` e para achar o no de uma aresta pelos vertices).
    """
    npoints = len(X)
    arestas = topologia_arestas(IEN)

    IEN_new = np.hstack((IEN, (npoints + arestas.id).astype(IEN.dtype)))

    va, vb = arestas.vertices[:, 0], arestas.vertices[:, 1]
    X_new = np.concatenate([X, (X[va] + X[vb]) / 2.0])
    Y_new = np.concatenate([Y, (Y[va] + Y[vb]) / 2.0])

    return IEN_new, X_new, Y_new, arestas


def estende_IENbound_tri6(IENbound: np.ndarray, arestas: Arestas, npoints: int) -> np.ndarray:
    """Insere o no de aresta (ponto medio) entre os dois vertices de cada
    segmento de contorno, para a condicao de contorno tambem valer sobre o
    grau de liberdade de velocidade do meio da aresta (Tri6).

    `arestas` vem de `elem_tri6` -- todo segmento de `IENbound` e tambem uma
    aresta de algum triangulo do dominio.

    Retorna um array (n_segmentos, 3): [v1, no_aresta, v2].
    """
    no_meio = npoints + arestas.indice(IENbound[:, 0], IENbound[:, 1])
    return np.column_stack([IENbound[:, 0], no_meio, IENbound[:, 1]]).astype(IENbound.dtype)


def area_com_sinal(IEN, X, Y):
    """Area orientada de cada elemento (positiva se os vertices estao anti-horarios).

    Existe para detectar **inversao de elemento** quando a malha se move
    (`moving_mesh.aplicar_oscilacao`). Isso importa porque um elemento
    invertido nao gera erro nenhum sozinho: `elemento.ElementoTriangular.geometria` toma
    `abs()` da area, mas os coeficientes `bi, ci` sao calculados sem
    `abs`, entao um elemento de orientacao trocada entra no sistema com
    `Gx`/`Gy` de sinal errado -- silenciosamente. A convencao do projeto
    e' `IEN[e, 0:3]` sempre anti-horario (a busca por caminhada de
    `semi_lagrangian.calculo_sl` tambem assume isso).

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


def montar_EToE(IEN: np.ndarray) -> np.ndarray:
    """Vizinhos por face: `EToE[e, f]` e' o elemento do outro lado da face `f`
    do elemento `e`, ou -1 se `f` e' contorno. Face `f` liga os vertices locais
    `f` e `(f+1)%3` (ver `topologia_arestas`). Usa so as 3 primeiras colunas.
    """
    ar = topologia_arestas(IEN)
    EToE = np.full((np.asarray(IEN).shape[0], 3), -1, dtype=int)
    interna = ar.elementos[:, 1] != -1
    e1, f1 = ar.elementos[interna, 0], ar.faces[interna, 0]
    e2, f2 = ar.elementos[interna, 1], ar.faces[interna, 1]
    EToE[e1, f1] = e2
    EToE[e2, f2] = e1
    return EToE


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


def montar_no_para_elem(IEN: np.ndarray, nnodes: int) -> np.ndarray:
    """Para cada no (vertices e nos extras), o primeiro elemento que o contem.

    Chute inicial da busca por caminhada do semi-Lagrangeano
    (`semi_lagrangian.calculo_sl`): o pe da caracteristica de um no costuma
    cair perto dele. Vale para qualquer `IEN` (so vertices, MINI ou Tri6):
    o centroide do MINI so aparece no proprio elemento, e o no de aresta do
    Tri6 pega o primeiro dos (ate) 2 elementos que o compartilham -- qualquer
    um serve, a caminhada corrige. No que nao aparece em `IEN` fica -1.
    """
    IEN = np.asarray(IEN)
    nos = IEN.ravel()
    elems = np.repeat(np.arange(IEN.shape[0]), IEN.shape[1])
    unicos, primeiro = np.unique(nos, return_index=True)
    semente = np.full(nnodes, -1, dtype=int)
    semente[unicos] = elems[primeiro]
    return semente
