"""Dente na parede de um canal crescendo para dentro do escoamento (malha deslizante, IEN fixa).

O dente e' um retangulo preso no piso `y = y_piso` (grupo de contorno
`grupo`, faces laterais + topo). Ele cresce como um **pistao rigido** que sai
do piso: a altura vai de `h0` (a da malha) a `h(t)` e

- o **topo** sobe rigido, `y = y_piso + h`;
- os nos das **laterais** deslizam na vertical mantendo a fracao da altura
  que ocupavam na malha inicial, `y = y_piso + eta*h` com
  `eta = (y0 - y_piso)/h0` (como os arcos da valvula, `valvula_rotacao`);
- as quinas com o piso e todo o resto do contorno ficam parados.

O interior segue a extensao harmonica do deslocamento do contorno com
rigidez por area (`movimento_malha.extensao_harmonica`, `chi`), a mesma
configuracao E0 usada na valvula.

A velocidade do fluido nas paredes do dente e' a do pistao, `(0, dh/dt)` em
todos os nos do grupo (quem fica nas quinas com o piso e' decidido por
`boundary.priority`) -- nas laterais ela difere da velocidade dos nos da
malha, que deslizam a `eta*dh/dt`.

So vertices; os nos extras do elemento seguem por `mesh.atualiza_nos_extras`.
numpy/scipy, sem torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femns.movimento_malha import area_com_sinal, extensao_harmonica, qualidade, tamanho_local

TOL_GEOMETRIA = 1e-9


@dataclass
class ContornoDente:
    y_piso: float
    h0: float                  # altura do dente na malha
    x_esq: float
    x_dir: float
    y_teto: float              # parede oposta (limite para o crescimento)
    contorno: np.ndarray       # mascara bool: todo vertice de contorno
    idx_movel: np.ndarray      # vertices do dente acima do piso (topo + laterais)
    eta: np.ndarray            # (y0 - y_piso)/h0 de cada vertice movel; 1 no topo
    idx_lateral: np.ndarray    # so os das laterais (0 < eta < 1)


def preparar(X, Y, IENbound, nomes, grupo: str = "dente") -> ContornoDente:
    """Acha o dente pelo grupo de contorno e confere que ele e' um retangulo apoiado no piso.

    `X, Y` sao so os vertices. O piso e' o menor `y` do grupo, o topo o maior;
    as laterais tem que ser verticais (`x` igual ao das quinas).
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    IENbound = np.asarray(IENbound)[:, :2]
    sel = np.asarray(nomes) == grupo
    if not sel.any():
        raise ValueError(f"grupo de contorno {grupo!r} nao existe na malha")
    nos = np.unique(IENbound[sel])
    y_piso, y_topo = Y[nos].min(), Y[nos].max()
    h0 = y_topo - y_piso
    if h0 <= 0:
        raise ValueError(f"dente {grupo!r} com altura nula")
    x_esq, x_dir = X[nos].min(), X[nos].max()
    no_piso = np.abs(Y[nos] - y_piso) <= TOL_GEOMETRIA
    no_topo = np.abs(Y[nos] - y_topo) <= TOL_GEOMETRIA
    lateral = ~no_piso & ~no_topo
    if (~(np.isclose(X[nos][lateral], x_esq, atol=TOL_GEOMETRIA)
          | np.isclose(X[nos][lateral], x_dir, atol=TOL_GEOMETRIA))).any():
        raise ValueError(f"dente {grupo!r}: as laterais nao sao verticais em x = {x_esq:.4g} e {x_dir:.4g}")
    if no_piso.sum() != 2:
        raise ValueError(f"dente {grupo!r}: esperava 2 vertices no piso (as quinas), achei {no_piso.sum()}")

    contorno = np.zeros(len(X), bool)
    contorno[np.unique(IENbound)] = True
    movel = nos[~no_piso]
    eta = (Y[movel] - y_piso) / h0
    # parede oposta: o contorno acima do dente
    acima = np.flatnonzero(contorno & (X >= x_esq) & (X <= x_dir) & (Y > y_topo + TOL_GEOMETRIA))
    y_teto = Y[acima].min() if len(acima) else np.inf
    return ContornoDente(y_piso=float(y_piso), h0=float(h0), x_esq=float(x_esq), x_dir=float(x_dir),
                         y_teto=float(y_teto), contorno=contorno, idx_movel=movel, eta=eta,
                         idx_lateral=movel[eta < 1 - TOL_GEOMETRIA])


class MovimentoDente:
    """Estado da malha (so vertices) e o passo `altura_atual -> altura_nova`.

    Mesmo papel de `movimento_malha.MovimentoValvula`: `X`, `Y` (copiados)
    sao a geometria atual; `passo` atualiza no lugar e devolve o diagnostico.
    """

    def __init__(self, contorno: ContornoDente, IEN, X, Y, chi: float = 2.0, desloc_max_h: float = 0.05):
        self.c = contorno
        self.IEN = np.asarray(IEN)[:, :3]
        self.X = np.asarray(X, float).copy()
        self.Y = np.asarray(Y, float).copy()
        self.n = len(self.X)
        self.chi = chi
        self.desloc_max_h = desloc_max_h
        self.altura = contorno.h0
        self.h0 = tamanho_local(self.IEN, self.X, self.Y, self.n)
        if qualidade(self.IEN, self.X, self.Y).min() <= 0:
            raise ValueError("malha inicial com elemento invertido (ou orientacao horaria)")

    def passo(self, altura_nova: float) -> dict:
        """Leva o dente de `self.altura` a `altura_nova`, em subpassos se o topo andar
        mais que `desloc_max_h` vezes o menor `h` dos vertices moveis."""
        if self.c.y_piso + altura_nova >= self.c.y_teto:
            raise ValueError(f"altura {altura_nova:.4g}: o topo do dente encosta na parede oposta "
                             f"(y = {self.c.y_teto:.4g})")
        h_min = self.h0[self.c.idx_movel].min()
        n_sub = max(1, int(np.ceil(abs(altura_nova - self.altura) / (self.desloc_max_h * h_min))))
        for h in np.linspace(self.altura, altura_nova, n_sub + 1)[1:]:
            info = self._passo_simples(float(h))
        info["subpassos"] = n_sub
        return info

    def _passo_simples(self, altura_nova: float) -> dict:
        c = self.c
        dY = np.zeros(self.n)
        dY[c.idx_movel] = c.y_piso + c.eta * altura_nova - self.Y[c.idx_movel]
        ex, ey = extensao_harmonica(self.IEN, self.X, self.Y, c.contorno, np.zeros(self.n), dY, self.chi)
        self.X += ex
        self.Y += ey
        self.Y[c.idx_movel] = c.y_piso + c.eta * altura_nova  # exato, sem acumular arredondamento
        self.altura = float(altura_nova)
        q = qualidade(self.IEN, self.X, self.Y)
        A = area_com_sinal(self.IEN, self.X, self.Y)
        return dict(altura=self.altura, q_min=float(q.min()), q_med=float(q.mean()),
                    invertidos=int((A <= 0).sum()), area_min=float(A.min()), area_total=float(A.sum()))


def velocidade_pistao(ccName, grupo: str, v: float, nnodes: int):
    """`(ux, uy)` do pistao: `(0, v)` em todo no (extras inclusive) cujo nome de contorno e' `grupo`."""
    m = np.array([ccName[i] == grupo for i in range(nnodes)])
    return np.zeros(nnodes), np.where(m, v, 0.0)
