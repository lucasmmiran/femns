import meshio
import numpy as np
import pytest

from femns.mesh import (
    _mesh_from_raw,
    elem_mini,
    elem_tri6,
    estende_IENbound_tri6,
    montar_EToE,
    montar_no_para_elem,
    montar_NToN,
    topologia_arestas,
)
from geometrias import quadrado_dois_triangulos


def test_montar_NToN():
    X, Y, IEN = quadrado_dois_triangulos()
    NToN = montar_NToN(IEN, len(X))

    assert set(NToN[0]) == {1, 2, 3}
    assert set(NToN[1]) == {0, 2}
    assert set(NToN[2]) == {0, 1, 3}
    assert set(NToN[3]) == {0, 2}


def test_topologia_arestas_numera_por_primeira_aparicao():
    """Varrendo IEN elemento a elemento, face a face: elemento 0 da as
    arestas (0,1), (1,2), (0,2); elemento 1 reencontra (0,2) e acrescenta
    (2,3), (0,3). E' a ordem em que os nos de aresta do Tri6 sao criados."""
    _, _, IEN = quadrado_dois_triangulos()
    ar = topologia_arestas(IEN)

    assert ar.vertices.tolist() == [[0, 1], [1, 2], [0, 2], [2, 3], [0, 3]]
    assert ar.id.tolist() == [[0, 1, 2], [2, 3, 4]]
    # diagonal (0,2): face 2 do elemento 0 e face 0 do elemento 1
    assert ar.elementos[2].tolist() == [0, 1]
    assert ar.faces[2].tolist() == [2, 0]
    # demais arestas sao de contorno
    for k in (0, 1, 3, 4):
        assert ar.elementos[k, 1] == -1 and ar.faces[k, 1] == -1


def test_topologia_arestas_indice_em_qualquer_ordem():
    _, _, IEN = quadrado_dois_triangulos()
    ar = topologia_arestas(IEN)

    assert ar.indice(np.array([2, 0, 3]), np.array([0, 2, 2])).tolist() == [2, 2, 3]
    with pytest.raises(KeyError):
        ar.indice(np.array([1]), np.array([3]))  # 1-3 nao e' aresta


def test_montar_EToE_face_compartilhada():
    _, _, IEN = quadrado_dois_triangulos()
    EToE = montar_EToE(IEN)

    # Elementos 0 e 1 compartilham a aresta (0, 2): face local 2 do elemento 0
    # ((2,0) -> nos (2,0)) e face local 0 do elemento 1 ((0,2) -> nos (0,2)).
    assert EToE[0, 2] == 1
    assert EToE[1, 0] == 0

    # As demais faces sao de contorno (-1)
    assert EToE[0, 0] == -1
    assert EToE[0, 1] == -1
    assert EToE[1, 1] == -1
    assert EToE[1, 2] == -1


def test_montar_no_para_elem_vertices():
    _, _, IEN = quadrado_dois_triangulos()
    semente = montar_no_para_elem(IEN, nnodes=4)

    # No 0 e 2 pertencem aos dois elementos (0 e 1); o primeiro elemento
    # encontrado ao varrer IEN e o elemento 0 para ambos.
    assert semente[0] == 0
    assert semente[2] == 0

    # No 1 so pertence ao elemento 0, no 3 so ao elemento 1.
    assert semente[1] == 0
    assert semente[3] == 1


def test_montar_no_para_elem_mini_e_tri6():
    """A mesma funcao da a semente dos nos extras: o centroide do MINI e' do
    proprio elemento; o no de aresta do Tri6 fica com o primeiro elemento que
    o contem (a diagonal, compartilhada, fica com o elemento 0)."""
    X, Y, IEN0 = quadrado_dois_triangulos()
    npoints = 4

    IENm, Xm, _ = elem_mini(IEN0, X, Y)
    assert montar_no_para_elem(IENm, len(Xm)).tolist() == [0, 0, 0, 1, 0, 1]

    IEN6, X6, _, ar = elem_tri6(IEN0, X, Y)
    semente = montar_no_para_elem(IEN6, len(X6))
    assert semente.shape == (9,)
    no = {tuple(v): npoints + k for k, v in enumerate(ar.vertices.tolist())}
    assert semente[no[(0, 1)]] == 0 and semente[no[(1, 2)]] == 0
    assert semente[no[(2, 3)]] == 1 and semente[no[(0, 3)]] == 1
    assert semente[no[(0, 2)]] == 0


def test_elem_mini_adiciona_centroide():
    X = np.array([0.0, 1.0, 0.0])
    Y = np.array([0.0, 0.0, 1.0])
    IEN = np.array([[0, 1, 2]])

    IEN_new, X_new, Y_new = elem_mini(IEN, X, Y)

    assert IEN_new.shape == (1, 4)
    assert IEN_new[0, 3] == 3
    assert X_new[3] == np.mean(X)
    assert Y_new[3] == np.mean(Y)


def test_elem_tri6_deduplica_no_de_aresta_compartilhada():
    """Os dois triangulos do quadrado compartilham a aresta (0,2) -- o no
    de aresta correspondente deve ser o MESMO nos dois elementos, nao um
    novo no por elemento (diferente do centroide do MINI)."""
    X, Y, IEN = quadrado_dois_triangulos()

    IEN_new, X_new, Y_new, arestas = elem_tri6(IEN, X, Y)

    assert IEN_new.shape == (2, 6)
    # elemento 0: v4=aresta(0,1), v5=aresta(1,2), v6=aresta(2,0)
    # elemento 1: v4=aresta(0,2), v5=aresta(2,3), v6=aresta(3,0)
    # aresta(2,0) do elemento 0 e aresta(0,2) do elemento 1 sao a mesma aresta.
    assert IEN_new[0, 5] == IEN_new[1, 3]

    # 4 vertices + 5 arestas distintas (das 6 "pontas" de aresta, 1 par se repete)
    assert X_new.shape == (9,)
    assert Y_new.shape == (9,)
    assert len(arestas.vertices) == 5

    no_diagonal = 4 + arestas.indice(0, 2)
    assert X_new[no_diagonal] == 0.5
    assert Y_new[no_diagonal] == 0.5
    assert IEN_new[0, 5] == no_diagonal


def test_estende_IENbound_tri6_insere_no_de_aresta():
    X, Y, IEN = quadrado_dois_triangulos()
    _, _, _, arestas = elem_tri6(IEN, X, Y)

    # contorno externo do quadrado (nao inclui a diagonal (0,2), que e interna)
    IENbound = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    IENbound_novo = estende_IENbound_tri6(IENbound, arestas, npoints=4)

    assert IENbound_novo.shape == (4, 3)
    for i, (a, b) in enumerate(IENbound):
        assert IENbound_novo[i, 0] == a
        assert IENbound_novo[i, 1] == 4 + arestas.indice(a, b)
        assert IENbound_novo[i, 2] == b


def test_mesh_from_raw_acha_blocos_por_tipo_independente_da_ordem():
    """lid.msh tem um bloco 'vertex' (ponto de referencia de pressao, cavidade
    tampada sem saida fisica de fluido) ANTES do bloco 'line' -- a leitura
    por indice fixo (raw.cells[0]/[1]) pegava o bloco errado. Este teste
    monta um `meshio.Mesh` sintetico com a mesma ordem de blocos (vertex,
    line, triangle) pra travar a selecao por tipo, sem precisar escrever/ler
    um `.msh` de verdade (evita depender dos requisitos do writer do gmsh
    pra `point_data`, irrelevantes pra malhas de verdade exportadas do Gmsh).
    """
    points = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ])
    cells = [
        ("vertex", np.array([[0]])),
        ("line", np.array([[0, 1], [1, 2], [2, 3], [3, 0]])),
        ("triangle", np.array([[0, 1, 2], [0, 2, 3]])),
    ]
    cell_data = {
        "gmsh:physical": [
            np.array([1]),
            np.array([2, 2, 2, 2]),
            np.array([3, 3]),
        ]
    }
    # field_data: nome -> [tag fisico, dimensao] (0=ponto, 1=linha, 2=superficie)
    field_data = {"outlet": np.array([1, 0]), "wall": np.array([2, 1]), "surface": np.array([3, 2])}
    raw = meshio.Mesh(points=points, cells=cells, cell_data=cell_data, field_data=field_data)

    mesh = _mesh_from_raw(raw)

    assert mesh.IEN.shape == (2, 3)
    assert mesh.IENbound.shape == (4, 2)
    assert mesh.IENboundElem == ["wall"] * 4
    assert list(mesh.IENpoint) == [0]
    assert mesh.IENpointElem == ["outlet"]
    assert mesh.npoints == 4
    assert mesh.ne == 2


def test_mesh_from_raw_sem_bloco_vertex_fica_com_ienpoint_vazio():
    """poiseuille.msh/degrau.msh nao tem bloco vertex -- IENpoint deve ficar vazio, nao quebrar."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    cells = [
        ("line", np.array([[0, 1], [1, 2], [2, 0]])),
        ("triangle", np.array([[0, 1, 2]])),
    ]
    cell_data = {"gmsh:physical": [np.array([1, 1, 1]), np.array([2])]}
    field_data = {"wall": np.array([1, 1]), "surface": np.array([2, 2])}
    raw = meshio.Mesh(points=points, cells=cells, cell_data=cell_data, field_data=field_data)

    mesh = _mesh_from_raw(raw)

    assert mesh.IENpoint.size == 0
    assert mesh.IENpointElem == []


def test_mesh_from_raw_nomeia_pelo_tag_e_nao_pela_ordem_do_field_data():
    """Tags fora da ordem de declaracao (ex.: `inner_wall` = 9 depois de `surface` = 8 em
    valve_10g_15k.msh) eram lidos por posicao e viravam o nome errado, em silencio."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    cells = [
        ("line", np.array([[0, 1], [1, 2], [2, 0]])),
        ("triangle", np.array([[0, 1, 2]])),
    ]
    cell_data = {"gmsh:physical": [np.array([1, 9, 1]), np.array([8])]}
    field_data = {"wall": np.array([1, 1]), "surface": np.array([8, 2]), "inner_wall": np.array([9, 1])}
    raw = meshio.Mesh(points=points, cells=cells, cell_data=cell_data, field_data=field_data)

    assert _mesh_from_raw(raw).IENboundElem == ["wall", "inner_wall", "wall"]
