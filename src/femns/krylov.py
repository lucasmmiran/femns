"""Solver iterativo esparso (BiCGSTAB), agnostico de device (CPU, CUDA, ROCm).

Nao depende de nenhuma biblioteca de solver esparso do fabricante da GPU
(cuDSS, rocSOLVER, ...) - so usa produto matriz-vetor esparso e algebra
densa de vetor, ambos suportados pelo torch em qualquer backend.
"""

import torch


def _csr_diagonal(A: torch.Tensor) -> torch.Tensor:
    """Extrai a diagonal de uma matriz esparsa CSR."""
    A = A.to_sparse_csr()
    n = A.shape[0]
    crow = A.crow_indices()
    col = A.col_indices()
    val = A.values()

    counts = crow[1:] - crow[:-1]
    row_idx = torch.repeat_interleave(torch.arange(n, device=A.device), counts)

    diag = torch.zeros(n, dtype=A.dtype, device=A.device)
    mask = row_idx == col
    diag[row_idx[mask]] = val[mask]
    return diag


def jacobi_preconditioner(A: torch.Tensor):
    """Pre-condicionador de Jacobi (diagonal): precond(v) = v / diag(A).

    Graus de liberdade com diagonal nula (ex.: linhas de pressao sem
    condicao de contorno no sistema de sela de Navier-Stokes) ficam sem
    escala (fator 1), ja que dividir por zero ali nao faz sentido.
    """
    diag = _csr_diagonal(A)
    inv_diag = torch.where(diag != 0, 1.0 / diag, torch.ones_like(diag))
    return lambda v: inv_diag * v


def bicgstab(A: torch.Tensor, b: torch.Tensor, x0: torch.Tensor = None,
             tol: float = 1e-8, max_iter: int = 2000, precond=None):
    """Resolve A x = b pelo metodo BiCGSTAB pre-condicionado (right-preconditioning).

    A: tensor esparso quadrado (n, n), formato CSR ou COO.
    b: tensor denso (n,).
    x0: chute inicial (n,); usa vetor nulo se omitido. Um bom chute inicial
        (ex.: a solucao do passo de tempo anterior) reduz bastante o numero
        de iteracoes necessarias.
    precond: funcao v -> K^-1 v (ex.: `jacobi_preconditioner(A)`); por
        padrao nao pre-condiciona (identidade). Sistemas de sela como o de
        Navier-Stokes tipicamente nao convergem sem pre-condicionador.

    Retorna (x, info), onde info = {"iters": int, "residual": float} com a
    norma residual relativa ||b - A x|| / ||b|| alcancada.
    """
    device, dtype = b.device, b.dtype
    n = b.shape[0]

    if precond is None:
        def precond(v):
            return v

    def matvec(v):
        return torch.mm(A, v.unsqueeze(1)).squeeze(1)

    x = x0.clone() if x0 is not None else torch.zeros(n, dtype=dtype, device=device)

    b_norm = torch.linalg.norm(b)
    if b_norm == 0:
        return torch.zeros(n, dtype=dtype, device=device), {"iters": 0, "residual": 0.0}

    r = b - matvec(x)
    r_hat = r.clone()

    rho = alpha = omega = torch.ones((), dtype=dtype, device=device)
    v = torch.zeros(n, dtype=dtype, device=device)
    p = torch.zeros(n, dtype=dtype, device=device)

    residual = (torch.linalg.norm(r) / b_norm).item()
    iters = 0

    if residual > tol:
        for iters in range(1, max_iter + 1):
            rho_new = torch.dot(r_hat, r)
            if rho_new == 0:
                break  # breakdown: r_hat ortogonal ao residuo atual

            beta = (rho_new / rho) * (alpha / omega)
            p = r + beta * (p - omega * v)
            y = precond(p)
            v = matvec(y)

            r_hat_v = torch.dot(r_hat, v)
            if r_hat_v == 0:
                break  # breakdown
            alpha = rho_new / r_hat_v

            s = r - alpha * v
            s_norm = (torch.linalg.norm(s) / b_norm).item()
            if s_norm <= tol:
                x = x + alpha * y
                residual = s_norm
                break

            z = precond(s)
            t = matvec(z)
            t_dot_t = torch.dot(t, t)
            if t_dot_t == 0:
                x = x + alpha * y
                residual = s_norm
                break
            omega = torch.dot(t, s) / t_dot_t

            x = x + alpha * y + omega * z
            r = s - omega * t
            rho = rho_new

            residual = (torch.linalg.norm(r) / b_norm).item()
            if residual <= tol:
                break

    return x, {"iters": iters, "residual": residual}
