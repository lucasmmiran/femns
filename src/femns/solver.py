"""Montagem do sistema global (Navier-Stokes, Euler implicito) e passo de tempo."""

import torch

from femns.boundary import apply_boundary_conditions


def build_system_matrix(dt: float, Re: float, K: torch.Tensor, M: torch.Tensor,
                         Gx: torch.Tensor, Gy: torch.Tensor):
    """Monta a matriz de bloco global A do sistema (vx, vy, p), sem condicoes de contorno.

        [ M/dt + K/Re,  0          , -Gx ] [vx]
        [ 0           , M/dt + K/Re, -Gy ] [vy]   =  b
        [ Dx          , Dy         ,  0  ] [p ]

    onde Dx = -Gx^T e Dy = -Gy^T (restricao de incompressibilidade).
    """
    n = K.shape[0]  # npoints + ne
    m = Gx.shape[1]  # npoints

    Dx, Dy = -torch.transpose(Gx, 0, 1), -torch.transpose(Gy, 0, 1)

    zero_n_n = torch.sparse_coo_tensor(size=(n, n), dtype=K.dtype, device=K.device)
    zero_m_m = torch.sparse_coo_tensor(size=(m, m), dtype=K.dtype, device=K.device)

    a_sup = torch.cat([(1 / dt) * M + (1 / Re) * K, zero_n_n, -Gx], dim=1)
    a_mid = torch.cat([zero_n_n, (1 / dt) * M + (1 / Re) * K, -Gy], dim=1)
    a_inf = torch.cat([Dx, Dy, zero_m_m], dim=1)

    return torch.cat([a_sup, a_mid, a_inf], dim=0).coalesce()


def time_step(A: torch.Tensor, M: torch.Tensor, Gvx: torch.Tensor, Gvy: torch.Tensor,
              vx: torch.Tensor, vy: torch.Tensor, dt: float,
              vx_cc: torch.Tensor, vy_cc: torch.Tensor, p_cc: torch.Tensor,
              vx_cc_pts: torch.Tensor, vy_cc_pts: torch.Tensor, p_cc_pts: torch.Tensor,
              npoints: int, ne: int):
    """Avanca um passo de tempo (Euler implicito, adveccao linearizada em vx/vy atuais).

    Retorna os novos vetores (vx, vy, p) ja com as condicoes de contorno aplicadas.
    """
    vx_diag = torch.diag(vx).to_sparse_csr()
    vy_diag = torch.diag(vy).to_sparse_csr()

    vg = torch.mm(vx_diag, Gvx) + torch.mm(vy_diag, Gvy)

    b_sup = (1 / dt) * torch.mm(M, vx.unsqueeze(1)) - torch.mm(vg, vx.unsqueeze(1))
    b_mid = (1 / dt) * torch.mm(M, vy.unsqueeze(1)) - torch.mm(vg, vy.unsqueeze(1))
    b_inf = torch.zeros(npoints, dtype=vx.dtype, device=vx.device)

    b_sup, b_mid = b_sup.squeeze(1), b_mid.squeeze(1)
    b_sup[vx_cc_pts] = vx_cc[vx_cc_pts]
    b_mid[vy_cc_pts] = vy_cc[vy_cc_pts]
    b_inf[p_cc_pts] = p_cc[p_cc_pts]

    b = torch.cat([b_sup, b_mid, b_inf], dim=0)

    # TODO(perf): resolvido denso pois o solve esparso direto (scipy/torch) nao
    # convergiu de forma estavel nos testes iniciais; revisitar para malhas maiores.
    x = torch.linalg.solve(A.to_dense(), b.unsqueeze(1))

    vx = x[:npoints + ne].squeeze(1)
    vy = x[npoints + ne:2 * (npoints + ne)].squeeze(1)
    p = x[2 * (npoints + ne):].squeeze(1)

    vx[vx_cc_pts] = vx_cc[vx_cc_pts]
    vy[vy_cc_pts] = vy_cc[vy_cc_pts]
    p[p_cc_pts] = p_cc[p_cc_pts]

    return vx, vy, p
