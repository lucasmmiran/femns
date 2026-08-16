import torch

from femns.assembly import assemble_mini, assemble_tri6
from femns.solver import build_system_matrix


def blocos_de_acoplamento(A: torch.Tensor, n: int, m: int):
    """Extrai do sistema global os blocos B (acoplamento de pressao no momento em x)
    e C (equacao de continuidade, coeficientes de vx)."""
    A_d = A.to_dense()
    B = A_d[:n, 2 * n:2 * n + m]
    C = A_d[2 * n:2 * n + m, :n]
    return B, C


def test_sistema_de_sela_e_simetrico_c_igual_b_transposto():
    """Regressao: um sistema de sela [A B; C 0] so e bem posto se C = B^T.

    Quebrar isso (ex.: montar Gx numa forma fraca e o operador de divergencia
    em outra, que diferem por integracao por partes -- logo por sinal) faz o
    BiCGSTAB deixar de convergir e a simulacao divergir; ja aconteceu com a
    variante "slip" do Tri6, ver CLAUDE.local.md (2026-08-15).
    """
    n, m = 2, 1
    K = torch.eye(n, dtype=torch.float64).to_sparse_coo()
    M = torch.eye(n, dtype=torch.float64).to_sparse_coo()
    Gx = torch.tensor([[2.0], [3.0]], dtype=torch.float64).to_sparse_coo()
    Gy = torch.tensor([[5.0], [7.0]], dtype=torch.float64).to_sparse_coo()

    A = build_system_matrix(dt=1.0, Re=1.0, K=K, M=M, Gx=Gx, Gy=Gy, beta=1.0)
    B, C = blocos_de_acoplamento(A, n, m)

    assert torch.allclose(B, -Gx.to_dense(), atol=1e-12)  # momento entra como -beta*Gx
    assert torch.allclose(C, B.T, atol=1e-12)


def triangulo_de_referencia_tri6():
    X = torch.tensor([0.0, 1.0, 0.0, 0.5, 0.5, 0.0], dtype=torch.float64)
    Y = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.5, 0.5], dtype=torch.float64)
    IEN = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    return X, Y, IEN


def test_tri6_e_mini_produzem_sistema_simetrico():
    """Os dois elementos montam Gx/Gy na mesma forma fraca, entao os dois geram
    um sistema de sela simetrico -- o Tri6 nao e um caso especial no solver."""
    X6, Y6, IEN6 = triangulo_de_referencia_tri6()
    K, M, Gx, Gy, _, _ = assemble_tri6(X6, Y6, IEN6, npoints=3, nnodes=6)
    A = build_system_matrix(0.1, 1.0, K, M, Gx, Gy, beta=1.0)
    B, C = blocos_de_acoplamento(A, n=6, m=3)
    assert torch.allclose(C, B.T, atol=1e-12)

    Xm = torch.tensor([0.0, 1.0, 0.0, 1.0 / 3.0], dtype=torch.float64)
    Ym = torch.tensor([0.0, 0.0, 1.0, 1.0 / 3.0], dtype=torch.float64)
    IENm = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    K, M, Gx, Gy, _, _ = assemble_mini(Xm, Ym, IENm, ne=1, npoints=3)
    A = build_system_matrix(0.1, 1.0, K, M, Gx, Gy, beta=1.0)
    B, C = blocos_de_acoplamento(A, n=4, m=3)
    assert torch.allclose(C, B.T, atol=1e-12)
