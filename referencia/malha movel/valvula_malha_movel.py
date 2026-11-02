"""
valvula_malha_movel.py
======================

Malha de triângulos móvel para uma válvula esfera 2D que gira de um ângulo θ.

Geometria (mesma notação do caderno):
    - Tubo reto com paredes em y = +H e y = -H, de x = -Lp (entrada) a x = +Lp (saída).
    - Esfera de raio r centrada na origem, com r^2 = H^2 + (L1/2)^2.
    - Furo da esfera com paredes de comprimento L1 (cordas φ2–φ1 em cima e φ3–φ4 embaixo).
    - φ1 = asen(H/r), φ2 = π - φ1, φ3 = π + φ1, φ4 = 2π - φ1.
    - A esfera gira de θ no sentido anti-horário: x = xi cosθ - yi senθ, y = xi senθ + yi cosθ.
    - Cada arco exposto mede S = r·θ. A válvula fecha totalmente em θ = 2·φ1.

Cada parede (de cima e de baixo) é UM caminho contínuo, da entrada até a saída:

    cima : tubo (-Lp,H)→P(φ2) | arco φ2→φ2+θ | corda P(φ2+θ)→P(φ1+θ) | arco φ1+θ→φ1 | tubo P(φ1)→(Lp,H)
    baixo: tubo (-Lp,-H)→P(φ3)| arco φ3→φ3+θ | corda P(φ3+θ)→P(φ4+θ) | arco φ4+θ→φ4 | tubo P(φ4)→(Lp,-H)

    com P(ang) = r·(cos ang, sen ang). O trecho em contato com a válvula mede L1 + 2S.

Algoritmo por passo de θ:
    1. Contorno: cada nó de parede tem uma coordenada de arco de referência s0 (em θ=0).
       Um mapa linear por partes s0 → s(θ) diz onde ele fica no caminho novo.
       Os nós da corda giram rígidos (quinas do furo exatas); os nós de uma faixa
       "buffer" de comprimento b no tubo esticam e escorregam para cima dos arcos.
    2. Interior: nós dentro do núcleo (|p| < fator_nucleo·r) giram rígidos; o resto
       resolve um laplaciano do deslocamento (incremental), com o contorno como Dirichlet.
    3. Checagem: área com sinal (triângulo invertido), nós fora do domínio e qualidade q.
    4. Se falhar: retriangulação de Delaunay dos MESMOS nós + suavização laplaciana
       protegida. O número de nós nunca muda; só a conectividade.

Dependências: numpy, scipy, matplotlib.  (Opcional: pacote `triangle` para
triangulação restrita; se não existir, usa Delaunay + filtro de centróide.)

Uso rápido:
    python valvula_malha_movel.py              # roda o exemplo e salva figuras
    python valvula_malha_movel.py --help

Autor: gerado com Claude para o projeto de CFD da válvula esfera.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import Delaunay
from matplotlib.path import Path

try:  # triangulação restrita opcional
    import triangle as _tr  # type: ignore
except Exception:  # pragma: no cover
    _tr = None


# =============================================================================
# 1. Geometria
# =============================================================================
@dataclass
class Geometria:
    H: float    # meia-altura do tubo
    L1: float   # comprimento da parede do furo (corda)
    Lp: float   # meio-comprimento do domínio (entrada em -Lp, saída em +Lp)

    def __post_init__(self):
        self.r = float(np.hypot(self.H, self.L1 / 2))
        f1 = float(np.arcsin(self.H / self.r))
        self.phi = {1: f1, 2: np.pi - f1, 3: np.pi + f1, 4: 2 * np.pi - f1}
        self.theta_fechamento = 2 * f1

    def P(self, ang):
        """Ponto do círculo no ângulo ang."""
        return self.r * np.array([np.cos(ang), np.sin(ang)])

    def S(self, th):
        """Arco exposto S = r·θ."""
        return self.r * th

    # ---- peças de cada parede, ordenadas da entrada para a saída ----
    def pecas(self, parede: str, th: float):
        H, Lp, p = self.H, self.Lp, self.phi
        if parede == "cima":
            a, b, y = 2, 1, H      # arco em φ2 (esquerda) e φ1 (direita)
        elif parede == "baixo":
            a, b, y = 3, 4, -H     # arco em φ3 (esquerda) e φ4 (direita)
        else:
            raise ValueError(parede)
        return [
            ("seg", np.array([-Lp, y]), self.P(p[a])),        # tubo de entrada
            ("arc", p[a], p[a] + th),                         # arco esquerdo (S)
            ("seg", self.P(p[a] + th), self.P(p[b] + th)),    # corda do furo (L1)
            ("arc", p[b] + th, p[b]),                         # arco direito (S)
            ("seg", self.P(p[b]), np.array([Lp, y])),         # tubo de saída
        ]

    def comprimento(self, peca):
        if peca[0] == "arc":
            return self.r * abs(peca[2] - peca[1])
        return float(np.linalg.norm(peca[2] - peca[1]))

    def avaliar(self, pecas, s):
        """Coordenadas (x, y) para cada comprimento de arco s ao longo do caminho."""
        L = np.array([self.comprimento(pc) for pc in pecas])
        borda = np.r_[0.0, np.cumsum(L)]
        s = np.clip(np.asarray(s, float), 0.0, borda[-1])
        k = np.clip(np.searchsorted(borda, s, side="right") - 1, 0, len(pecas) - 1)
        out = np.empty((len(s), 2))
        for i, (si, ki) in enumerate(zip(s, k)):
            pc = pecas[ki]
            u = (si - borda[ki]) / L[ki] if L[ki] > 0 else 0.0
            if pc[0] == "arc":
                g = pc[1] + u * (pc[2] - pc[1])
                out[i] = self.r * np.array([np.cos(g), np.sin(g)])
            else:
                out[i] = (1 - u) * pc[1] + u * pc[2]
        return out, borda

    # ---- teste exato de pertinência ao domínio fluido ----
    def dentro(self, xy, th, tol=0.0):
        """Domínio = (tubo fora da esfera) ∪ (furo girado dentro da esfera)."""
        x, y = xy[:, 0], xy[:, 1]
        rho = np.hypot(x, y)
        tubo = (np.abs(y) < self.H - tol) & (rho >= self.r)
        yl = -x * np.sin(th) + y * np.cos(th)      # coordenada no referencial da esfera
        furo = (rho < self.r - tol) & (np.abs(yl) < self.H - tol)
        return (tubo | furo) & (np.abs(x) < self.Lp - tol)


# =============================================================================
# 2. Malha inicial (θ = 0): retângulo estruturado dividido em triângulos
# =============================================================================
@dataclass
class Malha:
    xy: np.ndarray                 # (N, 2) posições atuais
    tri: np.ndarray                # (M, 3) conectividade
    xy0: np.ndarray                # posições em θ = 0 (referência)
    parede: dict                   # {"cima": idx ordenados, "baixo": idx ordenados}
    fixos: np.ndarray              # entrada/saída (não se mexem)
    contorno: np.ndarray           # todos os nós de contorno
    interior: np.ndarray           # nós internos
    s0: dict = field(default_factory=dict)  # coordenada de arco de referência por parede


def malha_inicial(geo: Geometria, h: float) -> Malha:
    H, L1, Lp = geo.H, geo.L1, geo.Lp
    m = max(1, round(L1 / (2 * h)))
    hx = L1 / (2 * m)                          # garante nós exatamente em x = ±L1/2
    nx_meio = round(Lp / hx)
    geo.Lp = Lp = nx_meio * hx                 # ajusta Lp para caber na grade
    ny = max(2, round(2 * H / h))
    xs = np.linspace(-Lp, Lp, 2 * nx_meio + 1)
    ys = np.linspace(-H, H, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    xy = np.c_[X.ravel(), Y.ravel()]
    nX, nY = len(xs), len(ys)
    idx = lambda i, j: i * nY + j

    tri = []
    for i in range(nX - 1):
        for j in range(nY - 1):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            if (i + j) % 2 == 0:
                tri += [(a, b, c), (a, c, d)]
            else:
                tri += [(a, b, d), (b, c, d)]
    tri = np.array(tri)

    cima = np.array([idx(i, nY - 1) for i in range(nX)])
    baixo = np.array([idx(i, 0) for i in range(nX)])
    entrada = np.array([idx(0, j) for j in range(1, nY - 1)])
    saida = np.array([idx(nX - 1, j) for j in range(1, nY - 1)])
    contorno = np.unique(np.r_[cima, baixo, entrada, saida])
    interior = np.setdiff1d(np.arange(len(xy)), contorno)

    M = Malha(xy=xy.copy(), tri=tri, xy0=xy.copy(),
              parede={"cima": cima, "baixo": baixo},
              fixos=np.r_[entrada, saida], contorno=contorno, interior=interior)
    for w in ("cima", "baixo"):
        M.s0[w] = xy[M.parede[w], 0] + Lp       # s0 = x + Lp em θ = 0
    return M


# =============================================================================
# 3. Contorno: mapa s0 -> s(θ) e posição dos nós de parede
# =============================================================================
def mapa_arco(geo: Geometria, s0, th, buffer):
    """Mapa linear por partes do comprimento de arco de referência para o atual.

    Nós de referência        -> caminho em θ
      0                      -> 0                       (entrada, fixo)
      A = Lp - L1/2 - b      -> A                       (início do buffer esquerdo)
      B = Lp - L1/2          -> B + S                   (quina do furo, gira rígida)
      C = Lp + L1/2          -> C + S                   (quina do furo, gira rígida)
      D = Lp + L1/2 + b      -> D + 2S                  (fim do buffer direito)
      2Lp                    -> 2Lp + 2S                (saída, fixo)

    Entre B e C: s = s0 + S  -> a corda gira rígida (é a fórmula da rotação).
    Entre A e B, e entre C e D: esticamento (b + S)/b -> nós escorregam do tubo para o arco.
    Fora do buffer: deslocamento só no parâmetro; a posição física não muda.
    """
    Lp, L1, S = geo.Lp, geo.L1, geo.S(th)
    b = min(buffer, Lp - L1 / 2)
    k0 = [0.0, Lp - L1 / 2 - b, Lp - L1 / 2, Lp + L1 / 2, Lp + L1 / 2 + b, 2 * Lp]
    k1 = [0.0, k0[1], k0[2] + S, k0[3] + S, k0[4] + 2 * S, 2 * Lp + 2 * S]
    return np.interp(s0, k0, k1)


def nos_parede(geo: Geometria, M: Malha, parede: str, th: float,
               buffer: float, encaixar_juncoes=True):
    """Posições dos nós de uma parede em θ (não incremental: só depende de θ)."""
    s = mapa_arco(geo, M.s0[parede], th, buffer)
    pcs = geo.pecas(parede, th)
    _, borda = geo.avaliar(pcs, [0.0])
    S = geo.S(th)
    if encaixar_juncoes and S > 1e-12:
        # junções tubo/arco em s = borda[1] (esquerda) e s = borda[4] (direita).
        # Não existem nós lá de forma contínua (o arco nasce com comprimento zero),
        # então o nó mais próximo é "encaixado" quando está a menos de meio espaçamento.
        quinas = {int(np.argmin(np.abs(M.s0[parede] - (geo.Lp - geo.L1 / 2)))),
                  int(np.argmin(np.abs(M.s0[parede] - (geo.Lp + geo.L1 / 2))))}
        for J in (borda[1], borda[4]):
            k = int(np.argmin(np.abs(s - J)))
            if k in quinas or k == 0 or k == len(s) - 1:
                continue
            ds = min(s[k] - s[k - 1], s[k + 1] - s[k])
            if abs(s[k] - J) < 0.5 * ds:
                s[k] = J
    xy, _ = geo.avaliar(pcs, s)
    return xy


def aplicar_contorno(geo, M, th, buffer, encaixar_juncoes=True):
    novo = M.xy.copy()
    for w in ("cima", "baixo"):
        novo[M.parede[w]] = nos_parede(geo, M, w, th, buffer, encaixar_juncoes)
    return novo


# =============================================================================
# 4. Qualidade
# =============================================================================
def area_com_sinal(xy, tri):
    a, b, c = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def qualidade(xy, tri):
    """q = 4√3·A / (l1²+l2²+l3²): 1 = equilátero, 0 = degenerado, <0 = invertido."""
    A = area_com_sinal(xy, tri)
    a, b, c = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    l2 = (np.sum((a - b) ** 2, 1) + np.sum((b - c) ** 2, 1) + np.sum((c - a) ** 2, 1))
    return 4 * np.sqrt(3) * A / np.maximum(l2, 1e-300)


def orientar(xy, tri):
    tri = tri.copy()
    neg = area_com_sinal(xy, tri) < 0
    tri[neg] = tri[neg][:, [0, 2, 1]]
    return tri


def arestas(tri):
    e = np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]
    e.sort(axis=1)
    return np.unique(e, axis=0)


# =============================================================================
# 5. Movimento do interior (laplaciano do deslocamento)
# =============================================================================
def laplaciano(xy, tri, n):
    e = arestas(tri)
    w = 1.0 / np.maximum(np.linalg.norm(xy[e[:, 0]] - xy[e[:, 1]], axis=1), 1e-12)
    i, j = e[:, 0], e[:, 1]
    W = sp.coo_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    D = sp.diags(np.asarray(W.sum(1)).ravel())
    return (D - W).tocsr()


def rot(xy, dth):
    c, s = np.cos(dth), np.sin(dth)
    return np.c_[c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]]


def mover_interior(geo, M, xy_cont_novo, dth, fator_nucleo):
    """Deslocamento incremental: Dirichlet no contorno e no núcleo rígido; laplaciano no resto."""
    n = len(M.xy)
    d = np.zeros((n, 2))
    d[M.contorno] = xy_cont_novo[M.contorno] - M.xy[M.contorno]

    rho = np.hypot(M.xy[:, 0], M.xy[:, 1])
    nucleo = np.intersect1d(M.interior, np.where(rho < fator_nucleo * geo.r)[0])
    d[nucleo] = rot(M.xy[nucleo], dth) - M.xy[nucleo]   # eq. do interior da válvula

    conhecidos = np.r_[M.contorno, nucleo]
    livres = np.setdiff1d(np.arange(n), conhecidos)
    if len(livres):
        K = laplaciano(M.xy, M.tri, n)
        Kff = K[livres][:, livres]
        Kfc = K[livres][:, conhecidos]
        for c in range(2):
            d[livres, c] = spla.spsolve(Kff.tocsc(), -Kfc @ d[conhecidos, c])
    return M.xy + d, nucleo


# =============================================================================
# 6. Retriangulação + suavização protegida
# =============================================================================
def anel_contorno(M: Malha, xy):
    """Índices do contorno em ordem anti-horária: baixo → saída → cima (invertida) → entrada."""
    ent = M.fixos[xy[M.fixos, 0] < 0]
    sai = M.fixos[xy[M.fixos, 0] > 0]
    ent = ent[np.argsort(-xy[ent, 1])]          # entrada: de cima para baixo
    sai = sai[np.argsort(xy[sai, 1])]           # saída: de baixo para cima
    return np.r_[M.parede["baixo"], sai, M.parede["cima"][::-1], ent]


def retriangular(M: Malha, xy, usar_triangle=True):
    """Nova conectividade para os MESMOS nós.

    - Com o pacote `triangle`: Delaunay restrita, as arestas de contorno são respeitadas
      por construção e nenhum nó novo é criado (opção 'pY').
    - Sem ele: Delaunay comum + descarte dos triângulos com centróide fora do polígono.
      Funciona bem enquanto o espaçamento no contorno for <= espaçamento interno e
      nenhuma passagem do domínio for mais estreita que ~1 elemento.
    """
    anel = anel_contorno(M, xy)
    if usar_triangle and _tr is not None:
        n = len(anel)
        seg = np.c_[np.arange(n), np.r_[np.arange(1, n), 0]]
        mapa = np.r_[anel, M.interior]
        t = _tr.triangulate({"vertices": xy[mapa], "segments": seg}, "pY")
        if len(t["vertices"]) != len(mapa):
            raise RuntimeError("triangle inseriu nós novos; verifique o contorno")
        tri = mapa[t["triangles"]]
    else:
        tri = Delaunay(xy).simplices
        tri = tri[Path(xy[anel]).contains_points(xy[tri].mean(1))]
    return orientar(xy, tri)


def arestas_contorno_faltando(M: Malha, tri, xy):
    """Quantas arestas do contorno não aparecem na triangulação (deve ser 0)."""
    anel = anel_contorno(M, xy)
    e = np.sort(np.c_[anel, np.roll(anel, -1)], axis=1)
    existentes = {tuple(a) for a in arestas(tri)}
    return sum(tuple(a) not in existentes for a in e)


def folga(pts, anel_xy, k=6):
    """Distância ao contorno com sinal: > 0 dentro do polígono, < 0 fora.

    Usa KD-tree: só testa os segmentos que tocam os k vértices de contorno mais
    próximos (suficiente com espaçamento de contorno aproximadamente uniforme).
    Pontos longe (> 2 segmentos) são tratados como dentro, com folga aproximada.
    """
    from scipy.spatial import cKDTree
    n = len(anel_xy)
    a_all, b_all = anel_xy, np.roll(anel_xy, -1, 0)
    smax = np.max(np.linalg.norm(b_all - a_all, axis=1))
    dv, iv = cKDTree(anel_xy).query(pts, k=k)
    out = dv[:, 0].copy()
    perto = dv[:, 0] < 2 * smax
    if perto.any():
        p = pts[perto][:, None, :]
        seg = np.c_[iv[perto], (iv[perto] - 1) % n]            # segmento i→i+1 e i-1→i
        a, b = a_all[seg], b_all[seg]
        ab = b - a
        t = np.clip(np.sum((p - a) * ab, 2) / np.maximum(np.sum(ab ** 2, 2), 1e-300), 0, 1)
        d = np.sqrt(np.min(np.sum((p - a - t[..., None] * ab) ** 2, 2), 1))
        out[perto] = np.where(Path(anel_xy).contains_points(pts[perto]), d, -d)
    return out


def suavizar(M, xy, tri, moveis, tol, iters=8, relax=0.5):
    """Laplaciano 'protegido' (vetorizado, tipo Jacobi).

    Cada nó móvel vai relax·(média dos vizinhos − posição). O movimento é desfeito
    nos nós que (a) saem do domínio ou (b) pertencem a um triângulo cuja qualidade
    ficou pior que antes e abaixo de 0.5. Repete `iters` vezes.
    """
    n = len(xy)
    e = arestas(tri)
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(n, n)).tocsr()
    grau = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    movel = np.zeros(n, bool)
    movel[moveis] = True
    anel_xy = xy[anel_contorno(M, xy)]
    for _ in range(iters):
        q0 = qualidade(xy, tri)
        media = (A @ xy) / grau[:, None]
        cand = xy.copy()
        cand[movel] += relax * (media[movel] - xy[movel])
        f0 = folga(xy[moveis], anel_xy)
        f1 = folga(cand[moveis], anel_xy)
        ruim_c = (f1 < tol) & (f1 < f0)
        cand[moveis[ruim_c]] = xy[moveis[ruim_c]]
        for _rev in range(5):                       # desfaz onde piorou
            q1 = qualidade(cand, tri)
            ruim = (q1 < q0) & (q1 < 0.5)
            if not ruim.any():
                break
            nos = np.unique(tri[ruim])
            nos = nos[movel[nos]]
            if len(nos) == 0:
                break
            cand[nos] = xy[nos]
        xy = cand
    return xy


# =============================================================================
# 7. Laço principal
# =============================================================================
@dataclass
class Config:
    h: float = 0.05               # tamanho de elemento
    buffer: float = 1.0           # faixa do tubo que alimenta cada arco (b)
    fator_nucleo: float = 0.75    # |p| < fator·r gira rígido
    q_min: float = 0.30           # abaixo disso retriangula
    folga_min: float = 0.25       # nó interno fica a >= folga_min·h do contorno
    encaixar_juncoes: bool = False  # True = nó exato na junção tubo/arco (custa qualidade)
    iters_suav: int = 20
    usar_triangle: bool = True    # usa o pacote `triangle` se estiver instalado


def passo(geo, M: Malha, th_ant, th, cfg: Config):
    """Avança a malha de θ_ant para θ. Retorna dict com diagnóstico."""
    alvo = aplicar_contorno(geo, M, th, cfg.buffer, cfg.encaixar_juncoes)
    xy_novo, nucleo = mover_interior(geo, M, alvo, th - th_ant, cfg.fator_nucleo)

    # nós internos fora do domínio (ou colados na parede): recua o passo
    tol = cfg.folga_min * cfg.h
    anel_xy = xy_novo[anel_contorno(M, xy_novo)]
    fora = M.interior[folga(xy_novo[M.interior], anel_xy) < tol]
    for i in fora:
        d = xy_novo[i] - M.xy[i]
        for t in (0.5, 0.25, 0.125, 0.0):
            cand = M.xy[i] + t * d
            if folga(cand[None], anel_xy)[0] >= tol:
                xy_novo[i] = cand
                break
        else:   # nem a posição antiga serve (a parede passou por cima): vai à média dos vizinhos
            viz = np.unique(M.tri[np.any(M.tri == i, 1)])
            xy_novo[i] = xy_novo[viz[viz != i]].mean(0)

    q = qualidade(xy_novo, M.tri)
    retri = False
    if q.min() < cfg.q_min or len(fora):
        tri = retriangular(M, xy_novo, cfg.usar_triangle)
        moveis = np.setdiff1d(M.interior, nucleo)
        xy_novo = suavizar(M, xy_novo, tri, moveis, tol, cfg.iters_suav)
        tri = retriangular(M, xy_novo, cfg.usar_triangle)   # conectividade final
        M.tri = tri
        q = qualidade(xy_novo, tri)
        retri = True
    M.xy = xy_novo

    ainda_fora = int((folga(M.xy[M.interior], M.xy[anel_contorno(M, M.xy)]) <= 0).sum())
    faltando = arestas_contorno_faltando(M, M.tri, M.xy) if retri else 0
    return dict(theta=th, q_min=float(q.min()), q_med=float(q.mean()),
                invertidos=int((q <= 0).sum()), retriangulou=retri,
                nos_fora=ainda_fora, arestas_faltando=faltando, n_tri=len(M.tri))


def simular(geo, cfg: Config, th_max, n_passos, callback=None):
    M = malha_inicial(geo, cfg.h)
    hist = []
    ths = np.linspace(0.0, th_max, n_passos + 1)
    for k in range(1, len(ths)):
        info = passo(geo, M, ths[k - 1], ths[k], cfg)
        hist.append(info)
        if callback:
            callback(k, M, info)
    return M, hist


# =============================================================================
# 8. Exportação e figuras
# =============================================================================
def exportar(M: Malha, arquivo):
    """Salva nós, triângulos e marcadores de contorno (npz)."""
    np.savez(arquivo, xy=M.xy, tri=M.tri, parede_cima=M.parede["cima"],
             parede_baixo=M.parede["baixo"], fixos=M.fixos)


def desenhar(ax, geo, M, th, titulo=""):
    q = qualidade(M.xy, M.tri)
    tp = ax.tripcolor(M.xy[:, 0], M.xy[:, 1], M.tri, facecolors=q,
                      cmap="viridis", vmin=0, vmax=1, edgecolors="k", linewidth=0.15)
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(geo.r * np.cos(t), geo.r * np.sin(t), color="0.6", lw=0.6, ls="--")
    for w, cor in (("cima", "tab:red"), ("baixo", "tab:orange")):
        p = M.xy[M.parede[w]]
        ax.plot(p[:, 0], p[:, 1], "-", color=cor, lw=1.2)
    for k in (1, 2, 3, 4):
        for ang in (geo.phi[k], geo.phi[k] + th):
            P = geo.P(ang)
            ax.plot(*P, "o", ms=2.5, color="tab:red")
    ax.set_aspect("equal")
    ax.set_xlim(-2.2 * geo.r, 2.2 * geo.r)
    ax.set_ylim(-1.15 * geo.r, 1.15 * geo.r)
    ax.set_title(titulo, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return tp


def main():
    ap = argparse.ArgumentParser(description="Malha móvel da válvula esfera 2D")
    ap.add_argument("--H", type=float, default=0.5)
    ap.add_argument("--L1", type=float, default=1.4)
    ap.add_argument("--Lp", type=float, default=3.0)
    ap.add_argument("--h", type=float, default=0.05)
    ap.add_argument("--frac", type=float, default=0.9,
                    help="fração do ângulo de fechamento total (2·φ1)")
    ap.add_argument("--passos", type=int, default=120)
    ap.add_argument("--buffer", type=float, default=1.0)
    ap.add_argument("--nucleo", type=float, default=0.75)
    ap.add_argument("--qmin", type=float, default=0.30)
    ap.add_argument("--encaixar", action="store_true",
                    help="encaixa um nó exatamente em cada junção tubo/arco")
    ap.add_argument("--sem-triangle", action="store_true",
                    help="força Delaunay + filtro mesmo com o pacote triangle instalado")
    ap.add_argument("--exportar", action="store_true", help="salva um .npz por passo")
    ap.add_argument("--saida", default="valvula")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geo = Geometria(a.H, a.L1, a.Lp)
    cfg = Config(h=a.h, buffer=a.buffer, fator_nucleo=a.nucleo, q_min=a.qmin,
                 usar_triangle=not a.sem_triangle, encaixar_juncoes=a.encaixar)
    th_max = a.frac * geo.theta_fechamento
    print(f"r = {geo.r:.4f}  φ1 = {np.degrees(geo.phi[1]):.2f}°  "
          f"fechamento = {np.degrees(geo.theta_fechamento):.2f}°  θmax = {np.degrees(th_max):.2f}°")

    alvos = np.degrees(th_max) * np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    fotos = {}

    def cb(k, M, info):
        th = np.degrees(info["theta"])
        for alvo in alvos:
            if alvo not in fotos and th >= alvo - 1e-9:
                fotos[alvo] = (M.xy.copy(), M.tri.copy(), info["theta"])
        if a.exportar:
            exportar(M, f"{a.saida}_{k:04d}.npz")

    M0 = malha_inicial(Geometria(a.H, a.L1, a.Lp), a.h)
    fotos[0.0] = (M0.xy.copy(), M0.tri.copy(), 0.0)
    M, hist = simular(geo, cfg, th_max, a.passos, cb)

    n_retri = sum(h["retriangulou"] for h in hist)
    pior = min(h["q_min"] for h in hist)
    inv = max(h["invertidos"] for h in hist)
    fora = max(h["nos_fora"] for h in hist)
    falt = max(h["arestas_faltando"] for h in hist)
    modo = "triangle (restrita)" if (_tr is not None and cfg.usar_triangle) else "Delaunay + filtro"
    print(f"retriangulação: {modo}   máx. arestas de contorno faltando = {falt}")
    if falt:
        print("  AVISO: há arestas de contorno perdidas. Instale o pacote `triangle` "
              "(pip install triangle) para triangulação restrita.")
    print(f"nós = {len(M.xy)}  passos = {len(hist)}  retriangulações = {n_retri}")
    print(f"pior q_min = {pior:.3f}  máx. invertidos = {inv}  máx. nós fora = {fora}")

    fig, axs = plt.subplots(len(fotos), 1, figsize=(7, 2.1 * len(fotos)))
    for ax, alvo in zip(axs, sorted(fotos)):
        xy, tri, th = fotos[alvo]
        M.xy, M.tri = xy, tri
        tp = desenhar(ax, geo, M, th, f"θ = {np.degrees(th):.1f}°   "
                      f"q_min = {qualidade(xy, tri).min():.2f}")
    fig.colorbar(tp, ax=axs, shrink=0.6, label="qualidade q")
    fig.savefig(f"{a.saida}_malhas.png", dpi=170, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(7, 3))
    th = [np.degrees(h["theta"]) for h in hist]
    ax.plot(th, [h["q_min"] for h in hist], label="q mínimo")
    ax.plot(th, [h["q_med"] for h in hist], label="q médio")
    rt = [t for t, h in zip(th, hist) if h["retriangulou"]]
    ax.plot(rt, [cfg.q_min] * len(rt), "|", color="tab:red", ms=10, label="retriangulação")
    ax.axhline(cfg.q_min, color="tab:red", lw=0.6, ls=":")
    ax.set_xlabel("θ [graus]"); ax.set_ylabel("q"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.savefig(f"{a.saida}_qualidade.png", dpi=170, bbox_inches="tight")
    print(f"figuras: {a.saida}_malhas.png, {a.saida}_qualidade.png")


if __name__ == "__main__":
    main()
