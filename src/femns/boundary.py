"""Definicao e aplicacao de condicoes de contorno a partir da malha e do config."""

import numpy as np
import torch


def assign_boundary_names(IENbound: np.ndarray, IENboundElem: list, npoints: int, priority: list[str],
                           IENpoint: np.ndarray = None, IENpointElem: list = None):
    """Atribui a cada no de contorno o nome do contorno ao qual ele pertence.

    `priority` lista os nomes de contorno da prioridade mais baixa para a
    mais alta: quando um no pertence a mais de um contorno (ex.: quinas da
    malha), prevalece o ultimo nome da lista que o contem.

    `IENpoint`/`IENpointElem` (opcionais, de `mesh.read_mesh`) cobrem
    contornos definidos por um unico no (ex.: ponto de referencia de
    pressao numa cavidade tampada, sem saida fisica de fluido) -- tratados
    dentro da mesma ordem de prioridade que as arestas.
    """
    ccName = [None for _ in range(npoints)]
    IENpoint = IENpoint if IENpoint is not None else np.empty(0, dtype=int)
    IENpointElem = IENpointElem if IENpointElem is not None else []

    for nome in priority:
        for a, b in IENbound[np.array(IENboundElem) == nome]:
            ccName[a] = nome
            ccName[b] = nome
        for a in IENpoint[np.array(IENpointElem) == nome]:
            ccName[a] = nome

    return ccName


def build_boundary_conditions(IENbound: np.ndarray, ccName: list, conditions: dict, npoints: int, device,
                               IENpoint: np.ndarray = None):
    """Monta os vetores de condicao de contorno (valores e indices) a partir do config.

    `conditions` mapeia nome do contorno -> {"vx": v, "vy": v, "p": v}; um
    componente so e restringido (Dirichlet) nos contornos que o listam.
    `IENpoint` (opcional) acrescenta nos marcados por contorno de ponto
    unico (ver `assign_boundary_names`) ao conjunto de nos considerado.

    Retorno: vx_cc, vy_cc, p_cc (tensores com o valor da condicao em cada no)
    e vx_cc_pts, vy_cc_pts, p_cc_pts (indices dos nos restringidos por componente).
    """
    IENpoint = IENpoint if IENpoint is not None else np.empty(0, dtype=int)
    cc = np.unique(np.concatenate([IENbound.reshape(IENbound.size), IENpoint]))
    ccName_arr = np.array(ccName, dtype=object)

    vx_cc = torch.zeros(npoints, dtype=torch.float64, device=device)
    vy_cc = torch.zeros(npoints, dtype=torch.float64, device=device)
    p_cc = torch.zeros(npoints, dtype=torch.float64, device=device)

    vx_idx, vy_idx, p_idx = [], [], []

    for nome, valores in conditions.items():
        idx = cc[ccName_arr[cc] == nome]
        if "vx" in valores:
            vx_cc[idx] = valores["vx"]
            vx_idx.append(idx)
        if "vy" in valores:
            vy_cc[idx] = valores["vy"]
            vy_idx.append(idx)
        if "p" in valores:
            p_cc[idx] = valores["p"]
            p_idx.append(idx)

    vx_cc_pts = np.concatenate(vx_idx) if vx_idx else np.array([], dtype=int)
    vy_cc_pts = np.concatenate(vy_idx) if vy_idx else np.array([], dtype=int)
    p_cc_pts = np.concatenate(p_idx) if p_idx else np.array([], dtype=int)

    vx_cc_pts = torch.from_numpy(vx_cc_pts).to(device)
    vy_cc_pts = torch.from_numpy(vy_cc_pts).to(device)
    p_cc_pts = torch.from_numpy(p_cc_pts).to(device)

    return vx_cc, vy_cc, p_cc, vx_cc_pts, vy_cc_pts, p_cc_pts


def apply_boundary_conditions(A: torch.Tensor, vx_cc_pts, vy_cc_pts, p_cc_pts, npoints: int, ne: int):
    """Zera as linhas de A referentes as condicoes de contorno e coloca 1 na diagonal."""
    rows_vx = vx_cc_pts
    rows_vy = vy_cc_pts + npoints + ne
    rows_p = p_cc_pts + 2 * (npoints + ne)
    rows_bc = torch.cat([rows_vx, rows_vy, rows_p])

    A = A.coalesce()
    mask = ~torch.isin(A.indices()[0], rows_bc)
    new_indices = A.indices()[:, mask]
    new_values = A.values()[mask]

    diag_indices = torch.stack([rows_bc, rows_bc])
    diag_values = torch.ones_like(rows_bc, dtype=A.dtype, device=A.device)

    final_indices = torch.cat([new_indices, diag_indices], dim=1)
    final_values = torch.cat([new_values, diag_values])

    return torch.sparse_coo_tensor(final_indices, final_values, size=A.shape, device=A.device, dtype=A.dtype)
