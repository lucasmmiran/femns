import torch

from femns.assembly import assemble_mini, assemble_tri6


def test_assemble_mini_triangulo_referencia():
    """Triangulo retangulo (0,0),(1,0),(0,1), area=0.5, checado contra valores analiticos."""
    X = torch.tensor([0.0, 1.0, 0.0, 1.0 / 3.0], dtype=torch.float64)
    Y = torch.tensor([0.0, 0.0, 1.0, 1.0 / 3.0], dtype=torch.float64)
    IEN = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    K, M, Gx, Gy, Gvx, Gvy = assemble_mini(X, Y, IEN, ne=1, npoints=3)

    K_dense = K.to_dense()
    M_dense = M.to_dense()

    # Bloco P1 padrao (sem a correcao da bolha) para este triangulo e conhecido:
    # [[1,-0.5,-0.5],[-0.5,0.5,0],[-0.5,0,0.5]]. A correcao da bolha soma
    # 0.9 (= 9/10*(zx+zy), com zx=zy=0.5) a cada entrada do bloco de vertices.
    K_esperado = torch.tensor([
        [1.9, 0.4, 0.4, -2.7],
        [0.4, 1.4, 0.9, -2.7],
        [0.4, 0.9, 1.4, -2.7],
        [-2.7, -2.7, -2.7, 8.1],
    ], dtype=torch.float64)

    assert torch.allclose(K_dense, K_esperado, atol=1e-10)

    # M = (area/840) * [[83,13,13,45]x3, [45,45,45,243]], area=0.5
    M_esperado = (0.5 / 840.0) * torch.tensor([
        [83, 13, 13, 45],
        [13, 83, 13, 45],
        [13, 13, 83, 45],
        [45, 45, 45, 243],
    ], dtype=torch.float64)

    assert torch.allclose(M_dense, M_esperado, atol=1e-10)

    # Matrizes de rigidez/massa devem ser simetricas
    assert torch.allclose(K_dense, K_dense.T, atol=1e-10)
    assert torch.allclose(M_dense, M_dense.T, atol=1e-10)

    assert Gx.shape == (4, 3)
    assert Gy.shape == (4, 3)
    assert Gvx.shape == (4, 4)
    assert Gvy.shape == (4, 4)


def test_assemble_tri6_triangulo_referencia():
    """Triangulo retangulo (0,0),(1,0),(0,1), area=0.5, nos de aresta nos pontos
    medios (v4=aresta(v1,v2), v5=aresta(v2,v3), v6=aresta(v3,v1)). Valores
    esperados obtidos por integracao simbolica exata (sympy) das funcoes de
    forma P2 (velocidade) e P1 (pressao) e conferidos contra a saida da funcao.
    """
    X = torch.tensor([0.0, 1.0, 0.0, 0.5, 0.5, 0.0], dtype=torch.float64)
    Y = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.5, 0.5], dtype=torch.float64)
    IEN = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)

    K, M, Gx, Gy, Gvx, Gvy = assemble_tri6(X, Y, IEN, npoints=3, nnodes=6)

    K_esperado = (1.0 / 6.0) * torch.tensor([
        [6, 1, 1, -4, 0, -4],
        [1, 3, 0, -4, 0, 0],
        [1, 0, 3, 0, 0, -4],
        [-4, -4, 0, 16, -8, 0],
        [0, 0, 0, -8, 16, -8],
        [-4, 0, -4, 0, -8, 16],
    ], dtype=torch.float64)
    assert torch.allclose(K.to_dense(), K_esperado, atol=1e-10)

    M_esperado = (0.5 / 180.0) * torch.tensor([
        [6, -1, -1, 0, -4, 0],
        [-1, 6, -1, 0, 0, -4],
        [-1, -1, 6, -4, 0, 0],
        [0, 0, -4, 32, 16, 16],
        [-4, 0, 0, 16, 32, 16],
        [0, -4, 0, 16, 16, 32],
    ], dtype=torch.float64)
    assert torch.allclose(M.to_dense(), M_esperado, atol=1e-10)

    # Rigidez/massa devem ser simetricas; K deve zerar campo de velocidade constante.
    assert torch.allclose(K.to_dense(), K.to_dense().T, atol=1e-10)
    assert torch.allclose(M.to_dense(), M.to_dense().T, atol=1e-10)
    assert torch.allclose(K.to_dense().sum(dim=1), torch.zeros(6, dtype=torch.float64), atol=1e-10)

    # Gx/Gy: gradiente P2 COMPLETO, INT(phi_j dN_i/dx) -- `gxele`/`gyele` da
    # referencia, nao a variante "slip" (ver docstring de assemble_tri6).
    Gx_esperado = (1.0 / 6.0) * torch.tensor([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
        [1, -1, 0],
        [1, 1, 2],
        [-1, -1, -2],
    ], dtype=torch.float64)
    assert torch.allclose(Gx.to_dense(), Gx_esperado, atol=1e-10)

    Gy_esperado = (1.0 / 6.0) * torch.tensor([
        [-1, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
        [-1, -2, -1],
        [1, 2, 1],
        [1, 0, -1],
    ], dtype=torch.float64)
    assert torch.allclose(Gy.to_dense(), Gy_esperado, atol=1e-10)

    # Uma pressao constante nao pode gerar forca liquida: soma de cada coluna = 0.
    assert torch.allclose(Gx.to_dense().sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(Gy.to_dense().sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-10)

    assert Gvx.shape == (6, 6)
    assert Gvy.shape == (6, 6)
