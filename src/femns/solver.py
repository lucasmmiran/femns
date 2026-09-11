"""Montagem do sistema global (Navier-Stokes, Euler implicito) e passo de tempo."""

import torch

from femns.krylov import bicgstab


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

    Os dois assemblers (`assemble_mini`, `assemble_tri6`) montam Gx/Gy na
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


def time_step(A: torch.Tensor, M: torch.Tensor, Gvx: torch.Tensor, Gvy: torch.Tensor,
              vx: torch.Tensor, vy: torch.Tensor, dt: float,
              vx_cc: torch.Tensor, vy_cc: torch.Tensor, p_cc: torch.Tensor,
              vx_cc_pts: torch.Tensor, vy_cc_pts: torch.Tensor, p_cc_pts: torch.Tensor,
              npoints: int, ne: int, beta: float = 1.0, x0: torch.Tensor = None,
              vx_star: torch.Tensor = None, vy_star: torch.Tensor = None,
              wx: torch.Tensor = None, wy: torch.Tensor = None,
              tol: float = 1e-8, max_iter: int = 2000):
    """Avanca um passo de tempo (Euler implicito).

    Resolve o sistema esparso com BiCGSTAB (ver `femns.krylov`), reaproveitando
    `x0` (solucao do passo anterior, com p ja convertida para p_tilde) como
    chute inicial - reduz bastante o numero de iteracoes entre passos.

    Adveccao: se `vx_star`/`vy_star` forem passados (velocidade ja
    interpolada no pe da caracteristica, ver `femns.semi_lagrangian`),
    usa esse termo direto no RHS -- sem restricao de CFL. Caso
    contrario (padrao), usa a adveccao Euleriana (`advection: eulerian` no
    config): o termo convectivo linearizado na velocidade do passo
    anterior, montado a partir de `vx, vy` e das matrizes `Gvx, Gvy`.
    A matriz `A` e identica nos dois casos -- ela nunca teve termo
    advectivo (ver docs/semi_lagrangian_strategy.pdf).

    Malha movel (ALE): se `wx, wy` (velocidade da malha, ver
    `moving_mesh.velocidade_malha`) forem passados, a adveccao Euleriana
    usa a velocidade **relativa** `v - w` como velocidade convectiva, que
    e' a forma ALE de `(v.grad)v` -- num referencial que se move, o que
    transporta e' o quanto o fluido anda *em relacao a malha*. Com
    `w = 0` (malha parada) recai exatamente no caso Euleriano. Nao tem
    efeito no caminho semi-Lagrangeano: la a correcao de malha movel e'
    geometrica (interpolar na malha do passo anterior, ver
    `femns.semi_lagrangian.calculo_sl`), nao um termo a subtrair aqui.

    Retorna (vx, vy, p, x_next, info): os vetores fisicos com as condicoes de
    contorno aplicadas, `x_next` (vx, vy, p_tilde) para usar como `x0` no
    proximo passo, e `info` com iteracoes/residuo do BiCGSTAB.
    """
    if vx_star is not None:
        b_sup = (1 / dt) * torch.mm(M, vx_star.unsqueeze(1))
        b_mid = (1 / dt) * torch.mm(M, vy_star.unsqueeze(1))
    else:
        # Velocidade convectiva: v (Euleriano) ou v - w (ALE, malha movel).
        cx = vx if wx is None else vx - wx
        cy = vy if wy is None else vy - wy

        # O termo convectivo linearizado e' `vg @ w`, com
        # `vg = diag(cx) Gvx + diag(cy) Gvy`. Aplicado a um vetor:
        #     vg @ w == cx (.) (Gvx @ w) + cy (.) (Gvy @ w)
        # -- calculado direto, sem materializar `diag(cx)` densa n x n
        # (era O(n^2) de VRAM por passo; ver plano de 2026-09-07). Resultado
        # algebricamente identico ao `torch.mm(vg, w)` anterior.
        cx1, cy1 = cx.unsqueeze(1), cy.unsqueeze(1)

        def _vg_mv(w1):
            return cx1 * torch.mm(Gvx, w1) + cy1 * torch.mm(Gvy, w1)

        b_sup = (1 / dt) * torch.mm(M, vx.unsqueeze(1)) - _vg_mv(vx.unsqueeze(1))
        b_mid = (1 / dt) * torch.mm(M, vy.unsqueeze(1)) - _vg_mv(vy.unsqueeze(1))

    b_inf = torch.zeros(npoints, dtype=vx.dtype, device=vx.device)

    b_sup, b_mid = b_sup.squeeze(1), b_mid.squeeze(1)
    b_sup[vx_cc_pts] = vx_cc[vx_cc_pts]
    b_mid[vy_cc_pts] = vy_cc[vy_cc_pts]
    b_inf[p_cc_pts] = p_cc[p_cc_pts] / beta  # BC atua sobre p_tilde = p/beta

    b = torch.cat([b_sup, b_mid, b_inf], dim=0)

    x, info = bicgstab(A, b, x0=x0, tol=tol, max_iter=max_iter)

    vx = x[:npoints + ne]
    vy = x[npoints + ne:2 * (npoints + ne)]
    p_tilde = x[2 * (npoints + ne):]
    p = beta * p_tilde

    vx[vx_cc_pts] = vx_cc[vx_cc_pts]
    vy[vy_cc_pts] = vy_cc[vy_cc_pts]
    p[p_cc_pts] = p_cc[p_cc_pts]

    x_next = torch.cat([vx, vy, p / beta])

    return vx, vy, p, x_next, info
