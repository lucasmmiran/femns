import torch

from femns.krylov import bicgstab


def test_bicgstab_sistema_pequeno_nao_simetrico():
    A_dense = torch.tensor([
        [4.0, 1.0, 0.0],
        [2.0, 3.0, 1.0],
        [0.0, 1.0, 2.0],
    ], dtype=torch.float64)
    b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    x_ref = torch.linalg.solve(A_dense, b.unsqueeze(1)).squeeze(1)

    A_csr = A_dense.to_sparse_csr()
    x, info = bicgstab(A_csr, b, tol=1e-10, max_iter=100)

    assert info["residual"] <= 1e-8
    assert torch.allclose(x, x_ref, atol=1e-6)


def test_bicgstab_matriz_diagonalmente_dominante_maior():
    torch.manual_seed(0)
    n = 60
    A_dense = torch.rand(n, n, dtype=torch.float64) * 0.1
    A_dense += torch.diag(torch.full((n,), 10.0, dtype=torch.float64))  # dominancia diagonal
    b = torch.rand(n, dtype=torch.float64)

    x_ref = torch.linalg.solve(A_dense, b.unsqueeze(1)).squeeze(1)

    A_csr = A_dense.to_sparse_csr()
    x, info = bicgstab(A_csr, b, tol=1e-10, max_iter=500)

    assert info["residual"] <= 1e-8
    assert torch.allclose(x, x_ref, atol=1e-6)


def test_bicgstab_warm_start_reduz_iteracoes():
    torch.manual_seed(1)
    n = 60
    A_dense = torch.rand(n, n, dtype=torch.float64) * 0.1
    A_dense += torch.diag(torch.full((n,), 10.0, dtype=torch.float64))
    A_csr = A_dense.to_sparse_csr()

    b1 = torch.rand(n, dtype=torch.float64)
    x1, _ = bicgstab(A_csr, b1, tol=1e-10, max_iter=500)

    # segundo lado direito bem proximo do primeiro (como entre passos de tempo)
    b2 = b1 + 1e-4 * torch.rand(n, dtype=torch.float64)
    _, info_cold = bicgstab(A_csr, b2, tol=1e-10, max_iter=500)
    _, info_warm = bicgstab(A_csr, b2, x0=x1, tol=1e-10, max_iter=500)

    assert info_warm["iters"] <= info_cold["iters"]


def test_bicgstab_lado_direito_nulo():
    A_dense = torch.eye(4, dtype=torch.float64)
    b = torch.zeros(4, dtype=torch.float64)

    x, info = bicgstab(A_dense.to_sparse_csr(), b)

    assert torch.allclose(x, torch.zeros(4, dtype=torch.float64))
    assert info["iters"] == 0


def _sela_aleatoria(n=30, m=10, seed=0):
    """Sistema de ponto de sela simetrico [F B; B^T 0] com F SPD e B de posto cheio."""
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(n, n, generator=g, dtype=torch.float64)
    F = R @ R.T + n * torch.eye(n, dtype=torch.float64)
    B = torch.randn(n, m, generator=g, dtype=torch.float64)
    A = torch.zeros(n + m, n + m, dtype=torch.float64)
    A[:n, :n], A[:n, n:], A[n:, :n] = F, B, B.T
    b = torch.randn(n + m, generator=g, dtype=torch.float64)
    return A, F, B, b


def test_minres_resolve_sistema_de_sela_sem_precondicionador():
    from femns.krylov import minres

    A, _, _, b = _sela_aleatoria()
    x, info = minres(A.to_sparse_csr(), b, tol=1e-10, max_iter=500)

    assert info["residual"] <= 1e-10
    assert torch.allclose(x, torch.linalg.solve(A, b), atol=1e-8)


def test_minres_com_precondicionador_de_bloco_exato_converge_em_poucas_iteracoes():
    """Com P = diag(F, S) exato, o operador pre-condicionado tem 3 autovalores
    distintos (Murphy, Golub & Wathen 2000): o MINRES converge em 3 iteracoes."""
    from femns.krylov import minres

    A, F, B, b = _sela_aleatoria()
    n = F.shape[0]
    S = B.T @ torch.linalg.solve(F, B)
    Finv, Sinv = torch.linalg.inv(F), torch.linalg.inv(S)

    def precond(v):
        return torch.cat([Finv @ v[:n], Sinv @ v[n:]])

    x, info = minres(A.to_sparse_csr(), b, tol=1e-10, max_iter=50, precond=precond)

    assert info["iters"] <= 4
    assert torch.allclose(x, torch.linalg.solve(A, b), atol=1e-8)


def test_minres_usa_chute_inicial_e_para_no_residuo_verdadeiro():
    from femns.krylov import minres

    A, _, _, b = _sela_aleatoria(seed=1)
    exato = torch.linalg.solve(A, b)
    _, info = minres(A.to_sparse_csr(), b, x0=exato.clone(), tol=1e-8)
    assert info["iters"] == 0

    x, info = minres(A.to_sparse_csr(), b, tol=1e-6, max_iter=500)
    verdadeiro = (torch.linalg.norm(b - A @ x) / torch.linalg.norm(b)).item()
    assert verdadeiro <= 1e-6 and abs(verdadeiro - info["residual"]) < 1e-14


def _spd_aleatoria(n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(n, n, generator=g, dtype=torch.float64)
    return R @ R.T + 0.5 * n * torch.eye(n, dtype=torch.float64)


def test_limites_jacobi_cobrem_o_espectro():
    from femns.krylov import limites_jacobi

    A = _spd_aleatoria()
    d = torch.diag(A)
    exatos = torch.linalg.eigvalsh(torch.diag(d ** -0.5) @ A @ torch.diag(d ** -0.5))
    lmin, lmax = limites_jacobi(A.to_sparse_csr(), d, passos=40)

    assert lmax >= exatos[-1].item()          # limite superior (o Chebyshev exige)
    assert lmin >= exatos[0].item() - 1e-10   # Ritz minimo nao fica abaixo do verdadeiro
    assert lmin <= 1.5 * exatos[0].item()


def test_chebyshev_jacobi_reduz_o_erro_e_e_simetrico():
    from femns.krylov import chebyshev_jacobi, limites_jacobi

    A = _spd_aleatoria()
    d = torch.diag(A)
    lmin, lmax = limites_jacobi(A.to_sparse_csr(), d, passos=40)
    aplicar = chebyshev_jacobi(A.to_sparse_csr(), d, lmin, lmax, passos=8)

    g = torch.Generator().manual_seed(3)
    b = torch.randn(40, generator=g, dtype=torch.float64)
    exato = torch.linalg.solve(A, b)
    assert torch.linalg.norm(aplicar(b) - exato) < 0.2 * torch.linalg.norm(exato)

    # simetria do operador: u . P(v) == v . P(u)
    u = torch.randn(40, generator=g, dtype=torch.float64)
    v = torch.randn(40, generator=g, dtype=torch.float64)
    assert abs(torch.dot(u, aplicar(v)) - torch.dot(v, aplicar(u))) < 1e-10
    # e linear: P(u + 2v) == P(u) + 2 P(v)
    assert torch.allclose(aplicar(u + 2 * v), aplicar(u) + 2 * aplicar(v), atol=1e-12)


def test_fsai_padrao_completo_e_a_inversa_exata():
    """Com A densa o padrao triangular inferior e' completo e a FSAI e' exata:
    G^T G = A^-1 (G e' o fator de Cholesky da inversa)."""
    from femns.krylov import fsai

    A = _spd_aleatoria(n=12)
    G, Gt = fsai(A.to_sparse_csr())
    Gd = G.to_dense()
    assert torch.allclose(Gd, torch.tril(Gd))                       # triangular inferior
    assert torch.allclose(torch.diag(Gd @ A @ Gd.T), torch.ones(12, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(Gt.to_dense() @ Gd, torch.linalg.inv(A), atol=1e-10)


def test_fsai_laplaciano_esparso_melhora_o_condicionamento_sobre_jacobi():
    """Laplaciano 2D esparso (padrao incompleto): G A G^T tem diagonal 1 e
    condicionamento menor que o de D^-1/2 A D^-1/2."""
    from femns.krylov import fsai

    m = 12
    T = 2 * torch.eye(m, dtype=torch.float64) - torch.diag(torch.ones(m - 1, dtype=torch.float64), 1) \
        - torch.diag(torch.ones(m - 1, dtype=torch.float64), -1)
    eye = torch.eye(m, dtype=torch.float64)
    A = torch.kron(T, eye) + torch.kron(eye, T)
    G, _ = fsai(A.to_sparse_csr())
    Gd = G.to_dense()
    GAGt = Gd @ A @ Gd.T
    assert torch.allclose(torch.diag(GAGt), torch.ones(m * m, dtype=torch.float64), atol=1e-12)
    kappa_fsai = torch.linalg.cond(GAGt).item()
    d = torch.diag(A) ** -0.5
    kappa_jac = torch.linalg.cond(d[:, None] * A * d[None, :]).item()
    assert kappa_fsai < 0.7 * kappa_jac


def test_chebyshev_fsai_aproxima_a_inversa_e_e_simetrico():
    from femns.krylov import chebyshev_fsai

    A = _spd_aleatoria(n=40)
    aplicar = chebyshev_fsai(A.to_sparse_csr(), passos=4)
    g = torch.Generator().manual_seed(4)
    b = torch.randn(40, generator=g, dtype=torch.float64)
    exato = torch.linalg.solve(A, b)
    assert torch.linalg.norm(aplicar(b) - exato) < 0.05 * torch.linalg.norm(exato)
    u = torch.randn(40, generator=g, dtype=torch.float64)
    v = torch.randn(40, generator=g, dtype=torch.float64)
    assert abs(torch.dot(u, aplicar(v)) - torch.dot(v, aplicar(u))) < 1e-10
