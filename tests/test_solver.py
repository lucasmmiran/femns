import torch

from femns.elemento import MINI, TRI6
from femns.solver import build_system_matrix, time_step
from geometrias import triangulo_mini_referencia, triangulo_tri6_referencia


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


def test_tri6_e_mini_produzem_sistema_simetrico():
    """Os dois elementos montam Gx/Gy na mesma forma fraca, entao os dois geram
    um sistema de sela simetrico -- o Tri6 nao e um caso especial no solver."""
    X6, Y6, IEN6 = triangulo_tri6_referencia()
    K, M, Gx, Gy = TRI6.montar(X6, Y6, IEN6, npoints=3, nnodes=6)
    A = build_system_matrix(0.1, 1.0, K, M, Gx, Gy, beta=1.0)
    B, C = blocos_de_acoplamento(A, n=6, m=3)
    assert torch.allclose(C, B.T, atol=1e-12)

    Xm, Ym, IENm = triangulo_mini_referencia()
    K, M, Gx, Gy = MINI.montar(Xm, Ym, IENm, npoints=3, nnodes=4)
    A = build_system_matrix(0.1, 1.0, K, M, Gx, Gy, beta=1.0)
    B, C = blocos_de_acoplamento(A, n=4, m=3)
    assert torch.allclose(C, B.T, atol=1e-12)


def test_time_step_lado_direito_e_m_v_estrela_sobre_dt():
    """O lado direito do momento e' `M v* / dt` (adveccao semi-Lagrangeana).

    Com `A = I` e `M = I` o sistema devolve o proprio lado direito: `v = v*/dt`
    e `p = 0`. Os nos com Dirichlet saem com o valor prescrito.
    """
    n, m = 4, 2  # nos de velocidade, nos de pressao
    N = 2 * n + m
    A = torch.eye(N, dtype=torch.float64).to_sparse_csr()
    M = torch.eye(n, dtype=torch.float64).to_sparse_csr()
    vx_star = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    vy_star = torch.tensor([-1.0, 0.5, 0.0, 2.0], dtype=torch.float64)
    dt = 0.5

    vx_cc = torch.zeros(n, dtype=torch.float64)
    vx_cc[0] = 9.0
    pts_vx = torch.tensor([0])
    vazio = torch.tensor([], dtype=torch.long)

    vx, vy, p, _, _ = time_step(
        A, M, vx_star, vy_star, dt,
        vx_cc, torch.zeros(n, dtype=torch.float64), torch.zeros(m, dtype=torch.float64),
        pts_vx, vazio, vazio, npoints=m, ne=n - m)

    assert torch.allclose(vx[1:], vx_star[1:] / dt)
    assert vx[0] == 9.0
    assert torch.allclose(vy, vy_star / dt)
    assert torch.allclose(p, torch.zeros(m, dtype=torch.float64))

