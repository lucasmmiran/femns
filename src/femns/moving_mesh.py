"""Deslocamento senoidal de um no da malha (experimento de malha movel/ALE)."""

import numpy as np
import torch


def nova_velocidade(vx: torch.Tensor, vy: torch.Tensor, dist: torch.Tensor, pts_viz):
    """Interpola a velocidade de um no deslocado a partir de seus vizinhos (IDW)."""
    device = dist.device
    beta = 2

    vx_viz = vx[pts_viz].to(device)
    vy_viz = vy[pts_viz].to(device)

    pesos = 1.0 / dist**beta
    pesos_norm = pesos / pesos.sum()

    return torch.dot(pesos_norm, vx_viz), torch.dot(pesos_norm, vy_viz)


def mover_ponto(ponto: int, X, Y, vx: torch.Tensor, vy: torch.Tensor, x0: float, y0: float,
                NToN, t: float, amplitude_factor: float = 0.1, omega: float = np.pi * 200):
    """Move um no da malha em uma trajetoria senoidal: pos_{m+1} = pos_0 + A*[cos, sin](omega*t).

    A amplitude A e proporcional a distancia ao vizinho mais proximo, para
    evitar que o no cruze elementos vizinhos.
    """
    device = X.device
    pts_viz = NToN[ponto]

    coo_ponto = torch.tensor([[x0, y0]]).to(device)
    coo_viz = torch.stack([X[pts_viz], Y[pts_viz]], dim=1).to(device)
    dist = torch.norm(coo_viz - coo_ponto, dim=1).to(device)

    amplitude = amplitude_factor * min(dist)

    X[ponto] = x0 + amplitude * np.cos(omega * t)
    Y[ponto] = y0 + amplitude * np.sin(omega * t)

    vx[ponto], vy[ponto] = nova_velocidade(vx, vy, dist, pts_viz)

    return X, Y, vx, vy
