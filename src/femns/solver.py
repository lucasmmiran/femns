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
              tol: float = 1e-8, max_iter: int = 2000):
    """Avanca um passo de tempo (Euler implicito).

    Resolve o sistema esparso com BiCGSTAB (ver `femns.krylov`), reaproveitando
    `x0` (solucao do passo anterior, com p ja convertida para p_tilde) como
    chute inicial - reduz bastante o numero de iteracoes entre passos.

    Adveccao: se `vx_star`/`vy_star` forem passados (velocidade ja
    interpolada no pe da caracteristica, ver `femns.semi_lagrangian`),
    usa esse termo direto no RHS -- sem restricao de CFL. Caso
    contrario (padrao), usa a adveccao linearizada explicita de hoje,
    montada a partir de `vx, vy` e das matrizes convectivas `Gvx, Gvy`.
    A matriz `A` e identica nos dois casos -- ela nunca teve termo
    advectivo (ver docs/semi_lagrangian_strategy.pdf).

    Retorna (vx, vy, p, x_next, info): os vetores fisicos com as condicoes de
    contorno aplicadas, `x_next` (vx, vy, p_tilde) para usar como `x0` no
    proximo passo, e `info` com iteracoes/residuo do BiCGSTAB.
    """
    if vx_star is not None:
        b_sup = (1 / dt) * torch.mm(M, vx_star.unsqueeze(1))
        b_mid = (1 / dt) * torch.mm(M, vy_star.unsqueeze(1))
    else:
        vx_diag = torch.diag(vx).to_sparse_csr()
        vy_diag = torch.diag(vy).to_sparse_csr()

        vg = torch.mm(vx_diag, Gvx) + torch.mm(vy_diag, Gvy)

        b_sup = (1 / dt) * torch.mm(M, vx.unsqueeze(1)) - torch.mm(vg, vx.unsqueeze(1))
        b_mid = (1 / dt) * torch.mm(M, vy.unsqueeze(1)) - torch.mm(vg, vy.unsqueeze(1))

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
