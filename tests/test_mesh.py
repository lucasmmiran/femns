import numpy as np

from femns.mesh import elem_mini, montar_EToE, montar_NToN


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


def test_elem_mini_adiciona_centroide():
    X = np.array([0.0, 1.0, 0.0])
    Y = np.array([0.0, 0.0, 1.0])
    IEN = np.array([[0, 1, 2]])

    IEN_new, X_new, Y_new = elem_mini(IEN, X, Y)

    assert IEN_new.shape == (1, 4)
    assert IEN_new[0, 3] == 3
    assert X_new[3] == np.mean(X)
    assert Y_new[3] == np.mean(Y)
