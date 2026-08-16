import meshio
import numpy as np

from femns.mesh import _mesh_from_raw, elem_mini, elem_tri6, estende_IENbound_tri6, montar_EToE, montar_node_to_elem, montar_NToN


def quadrado_dois_triangulos():
    """Quadrado unitario dividido em 2 triangulos: 0=(0,0) 1=(1,0) 2=(1,1) 3=(0,1)."""
    X = np.array([0.0, 1.0, 1.0, 0.0])
    Y = np.array([0.0, 0.0, 1.0, 1.0])
    IEN = np.array([[0, 1, 2], [0, 2, 3]])
    return X, Y, IEN


def test_montar_NToN():
    X, Y, IEN = quadrado_dois_triangulos()
    NToN = montar_NToN(IEN, len(X))

    assert set(NToN[0]) == {1, 2, 3}
    assert set(NToN[1]) == {0, 2}
    assert set(NToN[2]) == {0, 1, 3}
    assert set(NToN[3]) == {0, 2}


def test_montar_EToE_face_compartilhada():
    _, _, IEN = quadrado_dois_triangulos()
    EToE, faces_locais = montar_EToE(IEN)

    # Elementos 0 e 1 compartilham a aresta (0, 2): face local 2 do elemento 0
    # ((2,0) -> nos (2,0)) e face local 0 do elemento 1 ((0,2) -> nos (0,2)).
    assert EToE[0, 2] == 1
    assert EToE[1, 0] == 0

    # As demais faces sao de contorno (-1)
    assert EToE[0, 0] == -1
    assert EToE[0, 1] == -1
    assert EToE[1, 1] == -1
    assert EToE[1, 2] == -1


def test_montar_node_to_elem():
    _, _, IEN = quadrado_dois_triangulos()
    node_to_elem = montar_node_to_elem(IEN, npoints=4)

    # No 0 e 2 pertencem aos dois elementos (0 e 1); o primeiro elemento
    # encontrado ao varrer IEN e o elemento 0 para ambos.
    assert node_to_elem[0] == 0
    assert node_to_elem[2] == 0

    # No 1 so pertence ao elemento 0, no 3 so ao elemento 1.
    assert node_to_elem[1] == 0
    assert node_to_elem[3] == 1


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

    IEN_new, X_new, Y_new, edge_para_no = elem_tri6(IEN, X, Y)

    assert IEN_new.shape == (2, 6)
    # elemento 0: v4=aresta(0,1), v5=aresta(1,2), v6=aresta(2,0)
    # elemento 1: v4=aresta(0,2), v5=aresta(2,3), v6=aresta(3,0)
    # aresta(2,0) do elemento 0 e aresta(0,2) do elemento 1 sao a mesma aresta.
    assert IEN_new[0, 5] == IEN_new[1, 3]

    # 4 vertices + 5 arestas distintas (das 6 "pontas" de aresta, 1 par se repete)
    assert X_new.shape == (9,)
    assert Y_new.shape == (9,)
    assert len(edge_para_no) == 5

    no_diagonal = edge_para_no[(0, 2)]
    assert X_new[no_diagonal] == 0.5
    assert Y_new[no_diagonal] == 0.5
    assert IEN_new[0, 5] == no_diagonal


def test_estende_IENbound_tri6_insere_no_de_aresta():
    X, Y, IEN = quadrado_dois_triangulos()
    _, _, _, edge_para_no = elem_tri6(IEN, X, Y)

    # contorno externo do quadrado (nao inclui a diagonal (0,2), que e interna)
    IENbound = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    IENbound_novo = estende_IENbound_tri6(IENbound, edge_para_no)

    assert IENbound_novo.shape == (4, 3)
    for i, (a, b) in enumerate(IENbound):
        chave = (min(a, b), max(a, b))
        assert IENbound_novo[i, 0] == a
        assert IENbound_novo[i, 1] == edge_para_no[chave]
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
