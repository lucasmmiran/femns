import torch

from femns.assembly import assemble_mini, assemble_tri6
from femns.solver import build_system_matrix, time_step


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


def test_time_step_ale_recai_no_euleriano_com_malha_parada():
    """Com w = 0 a correcao ALE (v - w) tem que dar exatamente o mesmo
    resultado de nao passar w nenhum -- garante que ligar malha movel nao
    muda o caminho Euleriano por acidente.
    """
    torch.manual_seed(0)
    npoints, ne = 4, 2
    n = npoints + ne

    M = torch.eye(n, dtype=torch.float64).to_sparse_csr()
    Gvx = torch.eye(n, dtype=torch.float64).to_sparse_csr()
    Gvy = (2.0 * torch.eye(n, dtype=torch.float64)).to_sparse_csr()
    A = torch.eye(2 * n + npoints, dtype=torch.float64).to_sparse_csr()

    vx = torch.rand(n, dtype=torch.float64)
    vy = torch.rand(n, dtype=torch.float64)
    vazio_v = torch.zeros(n, dtype=torch.float64)
    vazio_p = torch.zeros(npoints, dtype=torch.float64)
    idx_vazio = torch.zeros(0, dtype=torch.long)

    args = (A, M, Gvx, Gvy, vx, vy, 0.1, vazio_v, vazio_v, vazio_p,
            idx_vazio, idx_vazio, idx_vazio, npoints, ne)

    sem_w = time_step(*args)
    com_w_zero = time_step(*args, wx=torch.zeros(n, dtype=torch.float64),
                            wy=torch.zeros(n, dtype=torch.float64))

    assert torch.equal(sem_w[0], com_w_zero[0])
    assert torch.equal(sem_w[1], com_w_zero[1])


def test_time_step_ale_usa_velocidade_relativa():
    """Passar w tem que ser equivalente a rodar sem w com o campo convectivo
    ja deslocado: e' a definicao de usar (v - w) como velocidade convectiva.
    Aqui `w = vx, vy` zera a velocidade relativa, entao o termo convectivo
    some e o RHS vira so o de massa.
    """
    npoints, ne = 4, 2
    n = npoints + ne

    M = torch.eye(n, dtype=torch.float64).to_sparse_csr()
    Gvx = torch.eye(n, dtype=torch.float64).to_sparse_csr()
    Gvy = (2.0 * torch.eye(n, dtype=torch.float64)).to_sparse_csr()
    A = torch.eye(2 * n + npoints, dtype=torch.float64).to_sparse_csr()

    vx = torch.rand(n, dtype=torch.float64) + 1.0
    vy = torch.rand(n, dtype=torch.float64) + 1.0
    zero = torch.zeros(n, dtype=torch.float64)
    vazio_p = torch.zeros(npoints, dtype=torch.float64)
    idx_vazio = torch.zeros(0, dtype=torch.long)
    dt = 0.1

    # w = v -> velocidade relativa nula -> sem termo convectivo
    vx_ale, vy_ale, *_ = time_step(
        A, M, Gvx, Gvy, vx, vy, dt, zero, zero, vazio_p,
        idx_vazio, idx_vazio, idx_vazio, npoints, ne, wx=vx, wy=vy)

    # A = I, entao a solucao e' o proprio RHS: so (1/dt) M v
    assert torch.allclose(vx_ale, (1 / dt) * vx)
    assert torch.allclose(vy_ale, (1 / dt) * vy)
