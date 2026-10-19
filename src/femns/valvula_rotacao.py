"""Cinematica do contorno da valvula esfera fechando por rotacao (malha deslizante, IEN fixa).

Geometria: tubo reto de semialtura `H`; esfera de raio `r = sqrt(H^2 + (L/2)^2)`
centrada na origem com um furo passante de largura `2H` e corda `L`. O angulo
`theta` e' medido a partir da valvula **totalmente aberta** (furo alinhado com o
tubo), cresce no sentido anti-horario e fecha em `theta = 2*asen(H/r)`.

Cada vertice de contorno cai em uma de tres classes (`preparar`):

- **rigido**: ponto material da esfera -- a corda (`grupos_corda`) e os
  filetes da quina esfera/corda, se a malha tiver (vertices dos `grupos_arco`
  *dentro* do circulo, `|p| < r`). Rotacao rigida exata a partir da posicao de
  referencia: `p = R(theta - theta0) p0` (nao incremental).
- **arco**: vertices dos `grupos_arco` *sobre* o circulo `|p| = r`, entre a
  ponta fixa (junta com o tubo, ou ponto de tangencia de um filete fixo) e a
  ponta movel (ponta da corda, ou tangencia de um filete rigido). So a
  posicao angular e' livre: o no fica no circulo por construcao. Ver
  `prever_arcos` / `suavizar_arcos` (plano 12/10, `plano_movimento_malha_1210.md` §3.4).
- **fixo**: o resto (tubo, entrada, saida, juntas e os filetes da carcaca,
  `|p| > r`).

As malhas antigas (`valve_10g_15k.msh`, quinas vivas) e as com filetes
(`valvula_10g.msh`) caem no mesmo esquema: o que muda e' so quem sao as pontas
de cada arco.

Modulo em numpy puro, sem `torch`: e' geometria de malha, nao entra no sistema
de Navier-Stokes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
# 2. Classificacao do contorno
# =============================================================================
@dataclass
class Arco:
    """Um arco de parede: vertices em ordem, da ponta fixa `nos[0]` a ponta movel `nos[-1]`.

    `nos[1:-1]` deslizam. `phi0` sao os angulos de referencia (desenrolados a
    partir de `nos[0]`); `lam` os pesos 1D da malha de referencia,
    `phi0[i] = lam[i]*phi0[i-1] + (1-lam[i])*phi0[i+1]` (ver `suavizar_arcos`).
    """

    grupo: str
    nos: np.ndarray
    phi0: np.ndarray
    lam: np.ndarray = field(init=False)

    def __post_init__(self):
        d_esq = self.phi0[1:-1] - self.phi0[:-2]
        d_dir = self.phi0[2:] - self.phi0[1:-1]
        self.lam = d_dir / (d_esq + d_dir)


@dataclass
class ContornoValvula:
    geo: GeometriaValvula
    theta0: float              # angulo em que a malha foi gerada (rad)
    X0: np.ndarray             # posicao de referencia dos vertices (theta0)
    Y0: np.ndarray
    contorno: np.ndarray       # mascara bool: todo vertice de contorno
    idx_rigido: np.ndarray     # pontos materiais da esfera
    arcos: list                # lista de `Arco`

    @property
    def idx_arco(self) -> np.ndarray:
        """Vertices que deslizam sobre os arcos (sem as pontas)."""
        return np.concatenate([a.nos[1:-1] for a in self.arcos]) if self.arcos else np.empty(0, int)

    @property
    def idx_movel(self) -> np.ndarray:
        """Todos os vertices de contorno que se movem."""
        return np.concatenate([self.idx_rigido, self.idx_arco])


def _nos_dos_grupos(IENbound, nomes, grupos) -> np.ndarray:
    sel = np.isin(np.asarray(nomes), list(grupos))
    return np.unique(np.asarray(IENbound)[sel][:, :2])


def _vizinhos_contorno(IENbound, n: int) -> list:
    viz = [[] for _ in range(n)]
    for a, b in np.asarray(IENbound)[:, :2]:
        viz[a].append(b)
        viz[b].append(a)
    return viz


def _caminhos(nos: np.ndarray, viz: list) -> list:
    """Componentes conexas (caminhos) do subgrafo de contorno induzido por `nos`, em ordem."""
    em = set(int(i) for i in nos)
    grau = {i: sum(j in em for j in viz[i]) for i in em}
    vistos, caminhos = set(), []
    for ponta in sorted(i for i in em if grau[i] < 2):
        if ponta in vistos:
            continue
        caminho, anterior = [ponta], None
        while True:
            prox = [j for j in viz[caminho[-1]] if j in em and j != anterior]
            if not prox:
                break
            anterior = caminho[-1]
            caminho.append(prox[0])
        vistos.update(caminho)
        caminhos.append(np.array(caminho, dtype=int))
    if vistos != em:
        raise ValueError("nos de arco formando um ciclo fechado no contorno")
    return caminhos


def preparar(IEN, X, Y, IENbound, nomes, H: float, L: float, theta0: float,
             grupos_corda=("valve_up", "valve_down"),
             grupos_arco=("outter_wall", "inner_wall")) -> ContornoValvula:
    """Classifica os vertices de contorno e guarda a referencia de cada um.

    `X, Y` sao so os vertices (`npoints`). `theta0` (rad) e' o angulo em que a
    malha foi gerada. Confere a geometria contra `H`, `L`, `theta0`: valor
    errado (ou sentido de rotacao trocado) levanta erro aqui, em vez de
    deformar a malha em silencio.
    """
    geo = GeometriaValvula(H=H, L=L)
    X = np.asarray(X, float).copy()
    Y = np.asarray(Y, float).copy()
    n = len(X)
    if theta0 <= 0:
        raise ValueError("theta0 precisa ser > 0: em theta = 0 os arcos tem comprimento nulo.")
    if theta0 >= geo.theta_fechamento:
        raise ValueError(f"theta0 = {np.degrees(theta0):.2f} graus >= fechamento "
                         f"{np.degrees(geo.theta_fechamento):.2f} graus.")

    contorno = np.zeros(n, bool)
    contorno[np.unique(np.asarray(IENbound)[:, :2])] = True

    # --- corda: desfaz a rotacao de theta0 -> y = +-H, |x| <= L/2
    idx_corda = _nos_dos_grupos(IENbound, nomes, grupos_corda)
    if len(idx_corda) == 0:
        raise ValueError(f"nenhum no nos grupos da corda {list(grupos_corda)}")
    xi, yi = rotacao(-theta0, X[idx_corda], Y[idx_corda])
    erro_y = np.abs(np.abs(yi) - H).max()
    if erro_y > TOL_GEOMETRIA or np.abs(xi).max() > L / 2 + TOL_GEOMETRIA:
        raise ValueError(
            f"a corda desrotacionada de theta0={np.degrees(theta0):.4f} graus nao cai em "
            f"y = +-H={H} com |x| <= L/2={L / 2} (desvio em y: {erro_y:.2e}, max |x|: "
            f"{np.abs(xi).max():.4f}). Confira H, L, theta0 e o sentido de rotacao.")

    # --- grupos de arco: sobre o circulo (desliza), dentro (filete rigido), fora (filete fixo)
    em_grupo_arco = _nos_dos_grupos(IENbound, nomes, grupos_arco)
    em_grupo_arco = em_grupo_arco[~np.isin(em_grupo_arco, idx_corda)]
    rho = np.hypot(X[em_grupo_arco], Y[em_grupo_arco])
    no_circulo = em_grupo_arco[np.abs(rho - geo.r) <= TOL_GEOMETRIA]
    filete_rigido = em_grupo_arco[rho < geo.r - TOL_GEOMETRIA]
    if len(no_circulo) == 0:
        raise ValueError(f"nenhum no dos grupos {list(grupos_arco)} sobre o circulo r={geo.r:.6f}; "
                         f"confira H e L.")

    rigido = np.zeros(n, bool)
    rigido[idx_corda] = True
    rigido[filete_rigido] = True

    # --- cadeias de arco: cada caminho sobre o circulo, com as pontas
    viz = _vizinhos_contorno(IENbound, n)
    grupo_do_no = {}
    for (a, b), g in zip(np.asarray(IENbound)[:, :2], nomes):
        if g in grupos_arco:
            grupo_do_no.setdefault(int(a), g)
            grupo_do_no.setdefault(int(b), g)
    arcos = []
    for cam in _caminhos(no_circulo, viz):
        # ancora de cada ponta: o vizinho de contorno fora do caminho
        anc = []
        for ponta, interno in ((cam[0], cam[1] if len(cam) > 1 else None),
                               (cam[-1], cam[-2] if len(cam) > 1 else None)):
            fora = [j for j in viz[ponta] if j != interno and j not in set(cam.tolist())]
            if len(fora) != 1:
                raise ValueError(f"arco com ponta {ponta} sem vizinho unico fora do arco")
            anc.append(fora[0])
        lado_rigido = [rigido[a] for a in anc]
        if lado_rigido.count(True) != 1:
            raise ValueError(f"arco {cam[:3]}...: precisa de exatamente uma ponta ligada a parte "
                             f"rigida (corda ou filete da esfera); achei {lado_rigido.count(True)}")
        if lado_rigido[0]:
            cam, anc = cam[::-1], anc[::-1]
        # ponta movel: a propria ancora, se for da corda (quina viva -- ela esta
        # no circulo); se for filete da esfera, o ponto de tangencia (cam[-1])
        # e' ele mesmo um ponto material da esfera.
        if anc[1] in set(idx_corda.tolist()):
            cam = np.append(cam, anc[1])
        else:
            rigido[cam[-1]] = True
        ang = np.arctan2(Y[cam], X[cam])
        phi = ang[0] + np.unwrap(ang - ang[0])
        if not (np.all(np.diff(phi) > 0) or np.all(np.diff(phi) < 0)):
            raise ValueError(f"arco do grupo {grupo_do_no.get(int(cam[1]))}: angulos nao monotonos")
        if len(cam) >= 3:
            arcos.append(Arco(grupo=grupo_do_no.get(int(cam[1]), "?"), nos=cam, phi0=phi))

    # ponta movel anda +theta (anti-horario): confere o sentido de cada arco
    for a in arcos:
        if abs(a.phi0[-1] - a.phi0[0]) > np.pi:
            raise ValueError("arco com mais de 180 graus: classificacao errada")

    return ContornoValvula(geo=geo, theta0=float(theta0), X0=X, Y0=Y, contorno=contorno,
                           idx_rigido=np.flatnonzero(rigido), arcos=arcos)


# =============================================================================
# 3. Cinematica do contorno
# =============================================================================
def posicoes_rigidas(c: ContornoValvula, theta: float):
    """`(idx, x, y)` dos pontos materiais da esfera no angulo `theta` (rotacao exata da referencia)."""
    idx = c.idx_rigido
    x, y = rotacao(theta - c.theta0, c.X0[idx], c.Y0[idx])
    return idx, x, y


def _phi_atual(a: Arco, X, Y) -> np.ndarray:
    ang = np.arctan2(Y[a.nos], X[a.nos])
    return a.phi0 + np.angle(np.exp(1j * (ang - a.phi0)))  # perto da referencia, sem salto de 2*pi


def prever_arcos(c: ContornoValvula, X, Y, theta_atual: float, theta_novo: float) -> None:
    """Preditor dos arcos (no lugar): estica a distribuicao atual entre as pontas.

    Ponta fixa parada, ponta movel `phi_B + (theta_novo - theta0)`; cada no
    interno mantem a fracao do arco que ocupa agora -- `eta_i^n * dTheta_arco`
    do plano. Com a suavizacao desligada isso e' `phi = phi_A + eta0*(phi_B - phi_A)`.
    """
    for a in c.arcos:
        phi = _phi_atual(a, X, Y)
        phiA = a.phi0[0]
        phiB = a.phi0[-1] + (theta_atual - c.theta0)
        phiB_novo = a.phi0[-1] + (theta_novo - c.theta0)
        eta = (phi[1:-1] - phiA) / (phiB - phiA)
        phi_novo = phiA + eta * (phiB_novo - phiA)
        X[a.nos[1:-1]] = c.geo.r * np.cos(phi_novo)
        Y[a.nos[1:-1]] = c.geo.r * np.sin(phi_novo)


def suavizar_arcos(c: ContornoValvula, X, Y, a2: float, varreduras: int, pesos: str = "referencia") -> None:
    """Corretor 1D dos arcos (no lugar), Jacobi em coordenada angular.

    `phi_i <- phi_i + a2*(lam_i phi_{i-1} + (1-lam_i) phi_{i+1} - phi_i)`.
    `pesos="referencia"`: `lam` da malha inicial -- o ponto fixo e' a
    distribuicao proporcional, a mesma do preditor (o corretor so desfaz o que
    outra coisa tiver desarrumado). `"uniforme"`: `lam = 1/2`, ponto fixo de
    espacamento igual (desfaz a gradacao do gmsh perto dos filetes).
    """
    if a2 <= 0 or varreduras <= 0:
        return
    for a in c.arcos:
        lam = a.lam if pesos == "referencia" else np.full(len(a.nos) - 2, 0.5)
        phi = _phi_atual(a, X, Y)
        for _ in range(varreduras):
            alvo = lam * phi[:-2] + (1 - lam) * phi[2:]
            phi[1:-1] += a2 * (alvo - phi[1:-1])
        X[a.nos[1:-1]] = c.geo.r * np.cos(phi[1:-1])
        Y[a.nos[1:-1]] = c.geo.r * np.sin(phi[1:-1])


def mascara_esfera(c: ContornoValvula) -> np.ndarray:
    """Vertices que sao pontos da superficie da esfera: os rigidos e todos os
    vertices de arco (pontas inclusive -- a ponta fixa e' o ponto de contato com
    a carcaca, sobre o circulo). Filetes da carcaca (`|p| > r`) ficam de fora."""
    m = np.zeros(len(c.X0), bool)
    m[c.idx_rigido] = True
    for a in c.arcos:
        m[a.nos] = True
    return m


def velocidade_esfera(c: ContornoValvula, X, Y, omega: float):
    """`omega x p = omega*(-y, x)` nos vertices da esfera (`mascara_esfera`), zero no resto.

    E' a velocidade do ponto material da esfera que esta em `p` -- diferente da
    velocidade do no da malha nos arcos, onde o no desliza (`eta*omega x p`).
    """
    m = mascara_esfera(c)
    return np.where(m, -omega * np.asarray(Y), 0.0), np.where(m, omega * np.asarray(X), 0.0)


# =============================================================================
# 4. Lei temporal
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
