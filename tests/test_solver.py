import pytest
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
    """O lado direito do momento e' `M v*/dt` (adveccao semi-Lagrangeana).

    Com `A = I` e `M = I` o sistema devolve o proprio lado direito: `v = v*/dt`
    e `p = 0`. Os graus com Dirichlet saem com o valor prescrito (`rhs_cc`).
    """
    from femns.reassembly import Sistema

    n, m = 4, 2  # nos de velocidade, nos de pressao
    N = 2 * n + m
    eh_cc = torch.zeros(N, dtype=torch.bool)
    eh_cc[0] = True
    rhs_cc = torch.zeros(N, dtype=torch.float64)
    rhs_cc[0] = 9.0
    sistema = Sistema(A=torch.eye(N, dtype=torch.float64).to_sparse_csr(),
                      M=torch.eye(n, dtype=torch.float64).to_sparse_csr(),
                      beta=1.0, rhs_cc=rhs_cc, eh_cc=eh_cc)
    vx_star = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    vy_star = torch.tensor([-1.0, 0.5, 0.0, 2.0], dtype=torch.float64)
    dt = 0.5

    for metodo in ("bicgstab", "minres"):
        vx, vy, p, _, info = time_step(sistema, vx_star, vy_star, dt, npoints=m, ne=n - m, metodo=metodo)

        assert torch.allclose(vx[1:], vx_star[1:] / dt)
        assert vx[0] == 9.0
        assert torch.allclose(vy, vy_star / dt)
        assert torch.allclose(p, torch.zeros(m, dtype=torch.float64))
        assert info["residual_verdadeiro"] < 1e-12


@pytest.mark.parametrize("metodo, precondicionador, subsolver", [
    ("bicgstab", "nenhum", None), ("minres", "nenhum", None),
    ("bicgstab", "bloco_triangular", "chebyshev"), ("bicgstab", "bloco_triangular", "lu"),
    ("minres", "bloco_diagonal", "chebyshev"), ("minres", "bloco_diagonal", "lu"),
    ("bicgstab", "bloco_triangular", "fsai"), ("minres", "bloco_diagonal", "fsai"),
])
def test_metodos_e_precondicionadores_batem_com_a_solucao_direta(metodo, precondicionador, subsolver):
    """Caso real pequeno (Poiseuille, MINI, malha do projeto): cada combinacao
    converge e da a mesma solucao do solver direto, **na pressao tambem**."""
    import numpy as np

    from femns.boundary import assign_boundary_names, build_boundary_conditions
    from femns.elemento import MINI
    from femns.mesh import read_mesh
    from femns.precondicionadores import montar
    from femns.reassembly import Remontador

    malha = read_mesh("meshes/poiseuille.msh")
    IEN, X, Y, IENb = MINI.adiciona_nos(malha.IEN, malha.X, malha.Y, malha.IENbound)
    npo, nn = malha.npoints, len(X)
    cond = {"inlet": {"vx": 1.0, "vy": 0.0}, "bottom": {"vx": 0.0, "vy": 0.0},
            "top": {"vx": 0.0, "vy": 0.0}, "outlet": {"p": 0.0}}
    nomes = assign_boundary_names(IENb, malha.IENboundElem, npo, ["outlet", "inlet", "top", "bottom"], nnodes=nn)
    vxc, vyc, pc, vxp, vyp, pp = build_boundary_conditions(IENb, nomes, cond, npo, "cpu", nnodes=nn, X=X, Y=Y,
                                                          IENboundElem=malha.IENboundElem)
    dt, Re = 0.01, 1.0
    R = Remontador(dt, Re, torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(IEN), MINI, npo, nn - npo,
                   vxp, vyp, pp, valores_cc=(vxc, vyc, pc), precisa_blocos=True)
    sis = R.sistema_inicial
    v_star = torch.from_numpy(np.where(np.arange(nn) < npo, 0.5, 0.4))
    precond = montar(precondicionador, sis, dt, Re, subsolver=subsolver or "chebyshev")

    vx, vy, p, _, info = time_step(sis, v_star, 0 * v_star, dt, npo, nn - npo, metodo=metodo,
                                   precond=precond, tol=1e-10, max_iter=5000, verificar_direto=True)

    assert info["residual_verdadeiro"] <= 1e-10
    assert info["erro_direto_v"] < 1e-7
    assert info["erro_direto_p"] < 1e-7
