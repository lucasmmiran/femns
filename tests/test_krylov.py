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
