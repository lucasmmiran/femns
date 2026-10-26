"""Geometria e lei temporal da valvula esfera fechando por rotacao.

Geometria: tubo reto de semialtura `H`; esfera de raio `r = sqrt(H^2 + (L/2)^2)`
centrada na origem com um furo passante de largura `2H` e corda `L`. O angulo
`theta` e' medido a partir da valvula **totalmente aberta** (furo alinhado com o
tubo), cresce no sentido anti-horario e fecha em `theta = 2*asen(H/r)`.

A classificacao e a cinematica do contorno (esfera rigida, arcos deslizando,
resto fixo) ficam em `femns.geometria_movel`, generica: o bloco antigo
`valvula_rotacao` do config e' convertido para `geometria_movel` por
`geometria_movel.de_valvula_rotacao`. Aqui fica o que e' so da valvula: as
grandezas geometricas, a conferencia da corda contra `H`, `L`, `theta0` e a
lei temporal (rampa em cosseno), usada por todos os corpos.

Modulo em numpy puro, sem `torch`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TOL_GEOMETRIA = 1e-6


# =============================================================================
# 1. Geometria
# =============================================================================
@dataclass
class GeometriaValvula:
    """Grandezas da valvula a partir da semialtura do tubo `H` e da corda `L`."""

    H: float = 0.5
    L: float = 2.0

    def __post_init__(self):
        self.r = float(np.hypot(self.H, self.L / 2))
        self.phi1 = float(np.arcsin(self.H / self.r))
        self.theta_fechamento = 2 * self.phi1

    def fresta(self, theta: float) -> float:
        """Menor abertura de passagem (sem filetes): ponta da corda ate a junta oposta."""
        return 2 * self.r * np.sin(self.phi1 - theta / 2)


def rotacao(theta: float, x, y):
    """`R(theta) (x, y)` em torno da origem (eixo da valvula)."""
    c, s = np.cos(theta), np.sin(theta)
    return c * x - s * y, s * x + c * y


# =============================================================================
# 2. Conferencia da malha
# =============================================================================
def conferir_corda(X, Y, IENbound, nomes, H: float, L: float, theta0: float,
                   grupos_corda=("valve_up", "valve_down")) -> None:
    """Confere que a corda, desrotacionada de `theta0` (rad), cai em `y = +-H` com `|x| <= L/2`.

    Pega `H`, `L` ou `theta0` errados (ou o sentido de rotacao trocado) antes de
    deformar a malha em silencio. `X, Y` sao os vertices na malha gerada.
    """
    if theta0 <= 0:
        raise ValueError("theta0 precisa ser > 0: em theta = 0 os arcos tem comprimento nulo.")
    sel = np.isin(np.asarray(nomes), list(grupos_corda))
    idx = np.unique(np.asarray(IENbound)[sel][:, :2])
    if len(idx) == 0:
        raise ValueError(f"nenhum no nos grupos da corda {list(grupos_corda)}")
    xi, yi = rotacao(-theta0, np.asarray(X)[idx], np.asarray(Y)[idx])
    erro_y = np.abs(np.abs(yi) - H).max()
    if erro_y > TOL_GEOMETRIA or np.abs(xi).max() > L / 2 + TOL_GEOMETRIA:
        raise ValueError(
            f"a corda desrotacionada de theta0={np.degrees(theta0):.4f} graus nao cai em "
            f"y = +-H={H} com |x| <= L/2={L / 2} (desvio em y: {erro_y:.2e}, max |x|: "
            f"{np.abs(xi).max():.4f}). Confira H, L, theta0 e o sentido de rotacao.")


# =============================================================================
# 3. Lei temporal
# =============================================================================
def theta_em(t: float, th_ini: float, th_fim: float, duracao: float, atraso: float = 0.0) -> float:
    """Rampa `(1 - cos(pi*u))/2`, `u = (t - atraso)/duracao`: sai e chega com derivada nula."""
    u = min(max((t - atraso) / duracao, 0.0), 1.0)
    return th_ini + (th_fim - th_ini) * 0.5 * (1.0 - np.cos(np.pi * u))


def omega_em(t: float, th_ini: float, th_fim: float, duracao: float, atraso: float = 0.0) -> float:
    """`dtheta/dt` analitico de `theta_em`."""
    t = t - atraso
    if t < 0.0 or t > duracao:
        return 0.0
    return (th_fim - th_ini) * 0.5 * np.pi / duracao * np.sin(np.pi * t / duracao)
