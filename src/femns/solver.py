"""Montagem do sistema global (Navier-Stokes, Euler implicito) e passo de tempo."""

import torch

from femns.krylov import bicgstab, minres


def pressure_scale_factor(K: torch.Tensor, M: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor,
                           dt: float, Re: float) -> float:
    """Estima o fator de escala `beta` que equilibra o bloco de velocidade
    (M/dt + K/Re) com o bloco de acoplamento de pressao (Gx, Gy).

    O termo 1/dt no bloco de velocidade deixa esse bloco ordens de grandeza
    maior que o de acoplamento de pressao, o que deixa o sistema de sela
    (saddle-point) mal condicionado para o BiCGSTAB — sem essa escala ele
    nao converge para a solucao correta (ver README/discussao no historico
    do projeto). Aplicar `beta` nas colunas/linhas de pressao (ver
    `build_system_matrix`) resolve isso; a pressao resolvida fica em uma
    variavel escalada `p_tilde = p / beta`, convertida de volta em `time_step`.
    """
    vel_block = ((1 / dt) * M + (1 / Re) * K).coalesce()
    vel_norm = torch.linalg.vector_norm(vel_block.values())
    press_norm = torch.sqrt(
        torch.linalg.vector_norm(Gx.coalesce().values()) ** 2
        + torch.linalg.vector_norm(Gy.coalesce().values()) ** 2
    )
    return (vel_norm / press_norm).item()


def build_system_matrix(dt: float, Re: float, K: torch.Tensor, M: torch.Tensor,
                         Gx: torch.Tensor, Gy: torch.Tensor, beta: float = 1.0):
    """Monta a matriz de bloco global A do sistema (vx, vy, p_tilde), sem condicoes de contorno.

        [ M/dt + K/Re,  0          , -beta*Gx ] [vx]
        [ 0           , M/dt + K/Re, -beta*Gy ] [vy]   =  b
        [ beta*Dx     , beta*Dy    ,  0       ] [p_tilde]

    onde Dx = -Gx^T e Dy = -Gy^T (restricao de incompressibilidade), e
    `beta` e o fator de `pressure_scale_factor` (1.0 = sem escala). A
    pressao fisica e p = beta * p_tilde.

    Os dois elementos (`elemento.Mini`, `elemento.Tri6`) montam Gx/Gy na
    mesma forma fraca, `INT(phi_j dN_i/dx)`, entao o bloco de continuidade
    sai como -Gx^T nos dois casos e o sistema de sela fica simetrico
    (C = B^T). Essa simetria nao e cosmetica: alimentar aqui um Gx montado
    na outra forma fraca (derivada na pressao, `INT(N_i dphi_j/dx)`, que
    difere desta por integracao por partes -- logo por sinal) quebra a
    relacao adjunta e o BiCGSTAB deixa de convergir (ver CLAUDE.local.md,
    2026-08-15).
    """
    n = K.shape[0]  # npoints + ne
    m = Gx.shape[1]  # npoints

    Gx_s, Gy_s = beta * Gx, beta * Gy
    Dx, Dy = -torch.transpose(Gx_s, 0, 1), -torch.transpose(Gy_s, 0, 1)

    zero_n_n = torch.sparse_coo_tensor(size=(n, n), dtype=K.dtype, device=K.device)
    zero_m_m = torch.sparse_coo_tensor(size=(m, m), dtype=K.dtype, device=K.device)

    a_sup = torch.cat([(1 / dt) * M + (1 / Re) * K, zero_n_n, -Gx_s], dim=1)
    a_mid = torch.cat([zero_n_n, (1 / dt) * M + (1 / Re) * K, -Gy_s], dim=1)
    a_inf = torch.cat([Dx, Dy, zero_m_m], dim=1)

    return torch.cat([a_sup, a_mid, a_inf], dim=0).coalesce()


def solucao_direta(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solucao do sistema por fatoracao direta (scipy, CPU) -- so para validacao.

    E' a referencia contra a qual o resultado dos metodos iterativos e'
    conferido (pressao inclusive: um residuo pequeno nao garante o nivel da
    pressao, ver plano 12/10 §1.3).
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    A = A.to_sparse_csr()
    A_sp = sp.csr_matrix((A.values().cpu().numpy(), A.col_indices().cpu().numpy(),
                          A.crow_indices().cpu().numpy()), shape=A.shape)
    x = spsolve(A_sp.tocsc(), b.cpu().numpy())
    return torch.from_numpy(np.asarray(x)).to(device=b.device, dtype=b.dtype)


def time_step(sistema, vx_star: torch.Tensor, vy_star: torch.Tensor, dt: float,
              npoints: int, ne: int, x0: torch.Tensor = None, metodo: str = "bicgstab",
              precond=None, tol: float = 1e-8, max_iter: int = 2000, verificar_direto: bool = False):
    """Avanca um passo de tempo (Euler implicito).

    `sistema` e' o `reassembly.Sistema` da geometria deste passo (matriz com as
    condicoes de Dirichlet eliminadas de forma simetrica, e a parcela delas no
    lado direito). Resolve com `metodo` (`"bicgstab"` ou `"minres"`, ver
    `femns.krylov`) e o pre-condicionador `precond` (`femns.precondicionadores`;
    `None` = nenhum), reaproveitando `x0` (solucao do passo anterior, em
    `p~ = p/beta`) como chute inicial.

    Adveccao semi-Lagrangeana: `vx_star`/`vy_star` sao a velocidade do passo
    anterior ja interpolada no pe da caracteristica (`femns.semi_lagrangian`),
    e entram so no lado direito, `M v*/dt` -- sem restricao de CFL. A matriz
    `A` nao tem termo advectivo. Com malha movel (ALE) a correcao tambem e'
    toda do semi-Lagrangeano (geometrica, ver `calculo_sl`); aqui nada muda.
    (A v1.0 tinha tambem a adveccao Euleriana explicita, removida na v2.0.)

    Retorna (vx, vy, p, x_next, info): os vetores fisicos com as condicoes de
    contorno aplicadas, `x_next` (vx, vy, p_tilde) para usar como `x0` no
    proximo passo, e `info` com `iters`, `residual` (o que o metodo reporta) e
    `residual_verdadeiro` (`||b - A x|| / ||b||`, mesmo criterio para os dois
    metodos). Com `verificar_direto=True`, tambem `erro_direto_v` e
    `erro_direto_p`: diferenca maxima relativa contra a solucao direta
    (`solucao_direta`), na velocidade e na pressao, e `tempo_direto` (o custo
    dessa checagem, para ser descontado de medidas de tempo).
    """
    M = sistema.M
    n_vel = npoints + ne
    b = torch.cat([(1 / dt) * torch.mm(M, vx_star.unsqueeze(1)).squeeze(1),
                   (1 / dt) * torch.mm(M, vy_star.unsqueeze(1)).squeeze(1),
                   torch.zeros(npoints, dtype=vx_star.dtype, device=vx_star.device)])
    # linhas livres: b menos o que as colunas eliminadas levariam; prescritas: o valor prescrito
    b = torch.where(sistema.eh_cc, sistema.rhs_cc, b + sistema.rhs_cc)

    if metodo == "minres":
        x, info = minres(sistema.A, b, x0=x0, tol=tol, max_iter=max_iter, precond=precond)
    elif metodo == "bicgstab":
        x, info = bicgstab(sistema.A, b, x0=x0, tol=tol, max_iter=max_iter, precond=precond)
    else:
        raise ValueError(f"metodo desconhecido: {metodo!r} (use 'bicgstab' ou 'minres')")

    b_norm = torch.linalg.norm(b)
    info["residual_verdadeiro"] = (
        torch.linalg.norm(b - torch.mm(sistema.A, x.unsqueeze(1)).squeeze(1)) / b_norm).item() if b_norm > 0 else 0.0

    if verificar_direto:
        from timeit import default_timer as timer
        t0 = timer()
        x_dir = solucao_direta(sistema.A, b)
        info["erro_direto_v"] = ((x[:2 * n_vel] - x_dir[:2 * n_vel]).abs().max()
                                 / x_dir[:2 * n_vel].abs().max().clamp_min(1e-300)).item()
        info["erro_direto_p"] = ((x[2 * n_vel:] - x_dir[2 * n_vel:]).abs().max()
                                 / x_dir[2 * n_vel:].abs().max().clamp_min(1e-300)).item()
        info["tempo_direto"] = timer() - t0  # quem mede tempo de passo desconta isto

    x = torch.where(sistema.eh_cc, sistema.rhs_cc, x)  # valores prescritos exatos
    vx = x[:n_vel]
    vy = x[n_vel:2 * n_vel]
    p = sistema.beta * x[2 * n_vel:]

    return vx, vy, p, x, info
