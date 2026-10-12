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


def _norma_residuo(A, b, x, b_norm) -> float:
    return (torch.linalg.norm(b - torch.mm(A, x.unsqueeze(1)).squeeze(1)) / b_norm).item()


def minres(A: torch.Tensor, b: torch.Tensor, x0: torch.Tensor = None,
           tol: float = 1e-8, max_iter: int = 2000, precond=None):
    """Resolve A x = b pelo MINRES pre-condicionado, para `A` simetrica (indefinida).

    Uma multiplicacao matriz-vetor e uma aplicacao do pre-condicionador por
    iteracao, recorrencia curta (memoria constante). `precond` (v -> P^-1 v)
    tem de ser **simetrico e definido positivo** e **fixo** (o mesmo operador
    em toda iteracao); `None` = identidade.

    O MINRES minimiza o residuo na norma `P^-1`, e a estimativa barata que ele
    carrega (`|eta|`) e' dessa norma, nao de `||b - A x||`. Para parar pelo
    mesmo criterio do `bicgstab` (residuo relativo verdadeiro, euclidiano),
    quando a estimativa cai abaixo de `tol` o residuo verdadeiro e' calculado
    (uma multiplicacao a mais) e a iteracao so termina se ele tambem estiver
    abaixo de `tol`.

    Algoritmo: C. C. Paige, M. A. Saunders (1975), SIAM J. Numer. Anal.
    12(4):617-629, na forma pre-condicionada de H. C. Elman, D. J. Silvester,
    A. J. Wathen (2014), Finite Elements and Fast Iterative Solvers, 2a ed.,
    OUP, cap. 4.

    Retorna (x, info) como o `bicgstab`; `info["residual"]` e' o residuo
    relativo verdadeiro final.
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

    v = b - matvec(x)
    residual = (torch.linalg.norm(v) / b_norm).item()
    if residual <= tol:
        return x, {"iters": 0, "residual": residual}

    z = precond(v)
    gamma = torch.sqrt(torch.dot(z, v))
    eta0 = gamma.item()
    eta = gamma
    v_old = torch.zeros_like(v)
    w = torch.zeros_like(v)
    w_old = torch.zeros_like(v)
    gamma_old = torch.ones((), dtype=dtype, device=device)
    c = c_old = torch.ones((), dtype=dtype, device=device)
    s = s_old = torch.zeros((), dtype=dtype, device=device)

    iters = 0
    for iters in range(1, max_iter + 1):
        z = z / gamma
        Az = matvec(z)
        delta = torch.dot(Az, z)
        v_new = Az - (delta / gamma) * v - (gamma / gamma_old) * v_old
        z_new = precond(v_new)
        gamma_new = torch.sqrt(torch.clamp(torch.dot(z_new, v_new), min=0.0))

        alpha0 = c * delta - c_old * s * gamma
        alpha1 = torch.sqrt(alpha0 * alpha0 + gamma_new * gamma_new)
        alpha2 = s * delta + c_old * c * gamma
        alpha3 = s_old * gamma
        c_new = alpha0 / alpha1
        s_new = gamma_new / alpha1

        w_new = (z - alpha3 * w_old - alpha2 * w) / alpha1
        x = x + c_new * eta * w_new
        eta = -s_new * eta

        v_old, v, z = v, v_new, z_new
        gamma_old, gamma = gamma, gamma_new
        c_old, c, s_old, s = c, c_new, s, s_new
        w_old, w = w, w_new

        if abs(eta.item()) / eta0 <= tol or gamma_new.item() == 0.0:
            residual = _norma_residuo(A, b, x, b_norm)
            if residual <= tol or gamma_new.item() == 0.0:
                break
    else:
        residual = _norma_residuo(A, b, x, b_norm)

    return x, {"iters": iters, "residual": residual}


def _mv(A, v):
    return torch.mm(A, v.unsqueeze(1)).squeeze(1)


def _lanczos_extremos(op, n: int, dtype, device, passos: int = 30, seed: int = 0):
    """Valores de Ritz extremos `(min, max)` do operador simetrico `op` (Lanczos, semente fixa)."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, generator=g, dtype=dtype).to(device)
    q = q / torch.linalg.norm(q)
    q_old = torch.zeros_like(q)
    alfas, betas = [], []
    beta = torch.zeros((), dtype=dtype, device=device)
    for _ in range(min(passos, n)):
        w = op(q) - beta * q_old
        alfa = torch.dot(w, q)
        w = w - alfa * q
        beta = torch.linalg.norm(w)
        alfas.append(alfa)
        betas.append(beta)
        if beta.item() == 0.0:
            break
        q_old, q = q, w / beta
    T = torch.diag(torch.stack(alfas))
    if len(alfas) > 1:
        fora = torch.stack(betas[:len(alfas) - 1])
        T = T + torch.diag(fora, 1) + torch.diag(fora, -1)
    ritz = torch.linalg.eigvalsh(T)
    return ritz[0].item(), ritz[-1].item()


def limites_jacobi(A: torch.Tensor, diag: torch.Tensor, passos: int = 30, seed: int = 0):
    """Estimativa `(lambda_min, lambda_max)` dos autovalores de `D^-1 A` (A SPD, D = diag(A)) por Lanczos.

    `passos` iteracoes de Lanczos em `D^-1/2 A D^-1/2` (simetrica), a partir de
    um vetor aleatorio de semente fixa (reprodutivel). Os valores de Ritz
    extremos convergem rapido; o maximo e' aumentado em 10% porque o
    Chebyshev precisa de um limite **superior** (com `lambda_max` subestimado
    ele amplifica as componentes acima do intervalo e o pre-condicionador
    deixa de ser definido positivo). O minimo de Ritz superestima o verdadeiro,
    o que so torna o Chebyshev mais lento nas componentes mais baixas.
    Ref.: Y. Saad (2003), Iterative Methods for Sparse Linear Systems, 2a ed.,
    SIAM, cap. 6.
    """
    d_inv_raiz = 1.0 / torch.sqrt(diag)
    lmin, lmax = _lanczos_extremos(lambda q: d_inv_raiz * _mv(A, d_inv_raiz * q),
                                   diag.shape[0], diag.dtype, diag.device, passos, seed)
    return lmin, 1.1 * lmax


def chebyshev(A: torch.Tensor, M_inv, lambda_min: float, lambda_max: float, passos: int):
    """Aproximacao de `A^-1` (A SPD) por `passos` passos de Chebyshev pre-condicionado por `M_inv`, partindo de zero.

    `M_inv` (simetrico, definido positivo) e' o pre-condicionador da iteracao
    (`r -> D^-1 r` no Jacobi; `r -> G^T G r` na FSAI); `[lambda_min,
    lambda_max]` deve cobrir o espectro de `M_inv A` por cima. Devolve a funcao
    `r -> z ~ A^-1 r`: um polinomio fixo em `M_inv A` vezes `M_inv`, portanto
    **linear, fixo e simetrico** -- serve dentro de pre-condicionadores do
    BiCGSTAB e do MINRES. Custa `passos - 1` multiplicacoes por `A` e `passos`
    aplicacoes de `M_inv`.

    Refs.: G. H. Golub, R. S. Varga (1961), Numer. Math. 3:147-168;
    Y. Saad (2003), Iterative Methods for Sparse Linear Systems, 2a ed., SIAM,
    alg. 12.1.
    """
    theta = 0.5 * (lambda_max + lambda_min)
    delta = 0.5 * (lambda_max - lambda_min)
    sigma = theta / delta

    def aplicar(r):
        x = torch.zeros_like(r)
        rho = 1.0 / sigma
        d = M_inv(r) / theta
        res = r
        for k in range(passos):
            x = x + d
            if k == passos - 1:
                break
            res = res - _mv(A, d)
            rho_novo = 1.0 / (2.0 * sigma - rho)
            d = rho_novo * rho * d + (2.0 * rho_novo / delta) * M_inv(res)
            rho = rho_novo
        return x

    return aplicar


def chebyshev_jacobi(A: torch.Tensor, diag: torch.Tensor, lambda_min: float, lambda_max: float, passos: int):
    """`chebyshev` com Jacobi (`M_inv = D^-1`). Ref. adicional: A. J. Wathen, T. Rees (2009), ETNA 34:125-135."""
    d_inv = 1.0 / diag
    return chebyshev(A, lambda r: d_inv * r, lambda_min, lambda_max, passos)


def fsai(A: torch.Tensor):
    """Inversa aproximada esparsa fatorada (FSAI) de `A` (SPD, CSR): `G` triangular inferior com `G^T G ~ A^-1`.

    Padrao de `G` = parte triangular inferior do padrao de `A`. Para cada linha
    `i`, com `P_i` as colunas `j <= i` da linha, resolve o sistema denso pequeno
    `A[P_i, P_i] y = e_i` e faz `G[i, P_i] = y / sqrt(y_i)`; assim `G A G^T`
    tem diagonal 1. Todas as linhas sao independentes: os sistemas sao
    resolvidos **em lote**, na GPU (`torch.linalg.solve`), sem nenhuma
    fatoracao sequencial -- e a aplicacao `G^T (G r)` e' so 2 produtos
    matriz-vetor, sem substituicao triangular.

    Ref.: L. Yu. Kolotilina, A. Yu. Yeremin (1993), "Factorized sparse
    approximate inverse preconditionings I. Theory", SIAM J. Matrix Anal.
    Appl. 14(1):45-58; E. Chow (2000), SIAM J. Sci. Comput. 21(5):1804-1822.

    Retorna `(G, Gt)` em CSR.
    """
    A = A.to_sparse_csr()
    dev, dtype = A.device, A.dtype
    n = A.shape[0]
    crow, col, val = A.crow_indices().to(torch.int64), A.col_indices().to(torch.int64), A.values()
    linha = torch.repeat_interleave(torch.arange(n, device=dev), crow[1:] - crow[:-1])

    # padrao triangular inferior, cada linha em ordem crescente de coluna (o ultimo e' a diagonal)
    inf = col <= linha
    li, lc = linha[inf], col[inf]
    cont = torch.bincount(li, minlength=n)
    k = int(cont.max().item())
    inicio = torch.zeros(n, dtype=torch.int64, device=dev)
    inicio[1:] = torch.cumsum(cont, 0)[:-1]
    pos = torch.arange(li.numel(), device=dev) - inicio[li]
    # padding a esquerda: a diagonal fica sempre na ultima posicao (k-1)
    desloc = k - cont[li]
    P = torch.full((n, k), -1, dtype=torch.int64, device=dev)
    P[li, pos + desloc] = lc

    # submatrizes densas A[P_i, P_i]: busca de cada par (a, b) nas chaves ordenadas do CSR
    chave = linha * n + col
    valido = P >= 0
    Pa = P.clamp(min=0)
    pares = Pa.unsqueeze(2) * n + Pa.unsqueeze(1)                    # (n, k, k)
    idx = torch.searchsorted(chave, pares.reshape(-1)).clamp(max=chave.numel() - 1)
    achou = (chave[idx] == pares.reshape(-1)).reshape(n, k, k)
    sub = torch.where(achou, val[idx].reshape(n, k, k), torch.zeros((), dtype=dtype, device=dev))
    ambos = valido.unsqueeze(2) & valido.unsqueeze(1)
    sub = torch.where(ambos, sub, torch.zeros((), dtype=dtype, device=dev))
    sub = sub + torch.diag_embed((~valido).to(dtype))                 # posicoes de padding: identidade

    rhs = torch.zeros(n, k, dtype=dtype, device=dev)
    rhs[:, k - 1] = 1.0
    y = torch.linalg.solve(sub, rhs)
    y = y / torch.sqrt(y[:, k - 1:k])

    G = torch.sparse_coo_tensor(torch.stack([li, lc]), y[li, pos + desloc], size=(n, n)).coalesce()
    return G.to_sparse_csr(), G.transpose(0, 1).coalesce().to_sparse_csr()


def chebyshev_fsai(A: torch.Tensor, passos: int, lanczos: int = 30):
    """`chebyshev` pre-condicionado pela FSAI (`M_inv = G^T G`), com os limites de `G A G^T` por Lanczos."""
    G, Gt = fsai(A)

    def M_inv(r):
        return _mv(Gt, _mv(G, r))

    lmin, lmax = _lanczos_extremos(lambda q: _mv(G, _mv(A, _mv(Gt, q))), A.shape[0], A.dtype, A.device, lanczos)
    return chebyshev(A, M_inv, lmin, 1.1 * lmax, passos)
