"""Definicao e aplicacao de condicoes de contorno a partir da malha e do config."""

import numpy as np
import torch


def assign_boundary_names(IENbound: np.ndarray, IENboundElem: list, npoints: int, priority: list[str],
                           IENpoint: np.ndarray = None, IENpointElem: list = None, nnodes: int = None):
    """Atribui a cada no de contorno o nome do contorno ao qual ele pertence.

    `priority` lista os nomes de contorno da prioridade mais baixa para a
    mais alta: quando um no pertence a mais de um contorno (ex.: quinas da
    malha), prevalece o ultimo nome da lista que o contem.

    `IENpoint`/`IENpointElem` (opcionais, de `mesh.read_mesh`) cobrem
    contornos definidos por um unico no (ex.: ponto de referencia de
    pressao numa cavidade tampada, sem saida fisica de fluido) -- tratados
    dentro da mesma ordem de prioridade que as arestas.

    `IENbound` pode ter qualquer numero de colunas por segmento (2 para
    elementos so com vertices, ex. MINI; 3 com no de aresta no meio, ex.
    Tri6 -- ver `mesh.estende_IENbound_tri6`). `nnodes` (opcional, default
    `npoints`) e o total de nos de velocidade -- precisa ser maior que
    `npoints` quando o contorno inclui nos que nao sao vertice (arestas
    do Tri6).
    """
    nnodes = nnodes if nnodes is not None else npoints
    ccName = [None for _ in range(nnodes)]
    IENpoint = IENpoint if IENpoint is not None else np.empty(0, dtype=int)
    IENpointElem = IENpointElem if IENpointElem is not None else []

    for nome in priority:
        for segmento in IENbound[np.array(IENboundElem) == nome]:
            for a in segmento:
                ccName[a] = nome
        for a in IENpoint[np.array(IENpointElem) == nome]:
            ccName[a] = nome

    return ccName


def build_boundary_conditions(IENbound: np.ndarray, ccName: list, conditions: dict, npoints: int, device,
                               IENpoint: np.ndarray = None, nnodes: int = None):
    """Monta os vetores de condicao de contorno (valores e indices) a partir do config.

    `conditions` mapeia nome do contorno -> {"vx": v, "vy": v, "p": v}; um
    componente so e restringido (Dirichlet) nos contornos que o listam.
    `IENpoint` (opcional) acrescenta nos marcados por contorno de ponto
    unico (ver `assign_boundary_names`) ao conjunto de nos considerado.
    `nnodes` (opcional, default `npoints`) e o total de nos de velocidade
    (ver `assign_boundary_names`) -- `vx_cc`/`vy_cc` sao alocados nesse
    tamanho, `p_cc` sempre em `npoints` (pressao so tem grau de liberdade
    nos vertices).

    Retorno: vx_cc, vy_cc, p_cc (tensores com o valor da condicao em cada no)
    e vx_cc_pts, vy_cc_pts, p_cc_pts (indices dos nos restringidos por componente).
    """
    nnodes = nnodes if nnodes is not None else npoints
    IENpoint = IENpoint if IENpoint is not None else np.empty(0, dtype=int)
    cc = np.unique(np.concatenate([IENbound.reshape(IENbound.size), IENpoint]))
    ccName_arr = np.array(ccName, dtype=object)

    vx_cc = torch.zeros(nnodes, dtype=torch.float64, device=device)
    vy_cc = torch.zeros(nnodes, dtype=torch.float64, device=device)
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
            # pressao so tem grau de liberdade nos vertices (P1): descarta
            # nos de aresta do Tri6 que porventura estejam em `idx`.
            idx_p = idx[idx < npoints]
            p_cc[idx_p] = valores["p"]
            p_idx.append(idx_p)

    vx_cc_pts = np.concatenate(vx_idx) if vx_idx else np.array([], dtype=int)
    vy_cc_pts = np.concatenate(vy_idx) if vy_idx else np.array([], dtype=int)
    p_cc_pts = np.concatenate(p_idx) if p_idx else np.array([], dtype=int)

    vx_cc_pts = torch.from_numpy(vx_cc_pts).to(device)
    vy_cc_pts = torch.from_numpy(vy_cc_pts).to(device)
    p_cc_pts = torch.from_numpy(p_cc_pts).to(device)

    return vx_cc, vy_cc, p_cc, vx_cc_pts, vy_cc_pts, p_cc_pts


def apply_boundary_conditions(A: torch.Tensor, vx_cc_pts, vy_cc_pts, p_cc_pts, npoints: int, ne: int):
    """Zera as linhas de A referentes as condicoes de contorno e coloca 1 na diagonal.

    `npoints + ne` deve ser o tamanho do bloco de velocidade (linhas/colunas
    de vx e de vy em `A`, ver `solver.build_system_matrix`). Para o MINI
    isso e `npoints + numero_de_elementos` (1 bolha por elemento); para o
    Tri6, `ne` deve ser `nnodes - npoints` (nos de aresta, nao ha relacao
    direta com o numero de elementos).
    """
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
