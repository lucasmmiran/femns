import torch

from femns.assembly import assemble_mini


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
