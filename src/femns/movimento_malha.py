"""Movimento de malha por soma ponderada de deslocamentos (plano 12/10, `docs/planos/plano_movimento_malha_1210.md`).

Cada vertice anda, por passo de tempo,

    dx_i = a1*dx_rot + a2*dx_suav + a3*dx_ext,

organizado como preditor-corretor (plano §5):

1. **contorno**, exato: pontos materiais da esfera por rotacao rigida, arcos
   esticados entre as pontas, resto parado (`femns.valvula_rotacao`);
2. **preditor do interior**: o termo de rotacao (`rotacao`), que arrasta os
   nos com a esfera -- mistura radial `psi(rho)` ou interpolacao pelo inverso
   da distancia dos deslocamentos do contorno (Witteveen & Bijl 2009) --,
   mais `a3` vezes a extensao harmonica do que o termo de rotacao nao
   cobriu no contorno (Laplaciano P1 com rigidez por area, `chi`; Johnson &
   Tezduyar 1994);
3. **corretor**: `varreduras` de Jacobi de suavizacao de coordenadas,
   `x_i <- x_i + a2*(alvo_i - x_i)`, com o alvo

   - `valor_medio`: `sum_j lam_ij x_j`, `lam` = coordenadas de valor medio
     da malha **inicial** (Floater 2003) -- a malha do gmsh e' ponto fixo, a
     gradacao fica;
   - `uniforme`: centroide dos vizinhos (Laplaciano; Herrmann 1976, Field 1988);
   - `odt`: media dos circuncentros da estrela ponderada pela area
     (Chen & Holst 2011);

   com guarda por elemento: um no so anda se nenhum triangulo da estrela
   inverte nem piora abaixo de `q_guarda` (Freitag & Ollivier-Gooch 1997;
   Canann, Tristano & Staten 1998). Os arcos tem o corretor 1D proprio
   (`a2_arco`).

So vertices; os nos extras do elemento (Tri6, MINI) seguem por
`mesh.atualiza_nos_extras`. numpy/scipy, sem torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import femns.valvula_rotacao as vr

ROTACOES = ("nenhuma", "radial", "idw")
PESOS = ("valor_medio", "uniforme", "odt")
PESOS_ARCO = ("referencia", "uniforme")


# =============================================================================
# 1. Opcoes (bloco `valvula_rotacao.movimento` do config)
# =============================================================================
@dataclass(frozen=True)
class OpcoesMovimento:
    rotacao: str = "radial"
    rho1: float = 1.0          # em unidades de r: psi = 1 para rho <= rho1*r
    rho2: float = 1.3          # psi = 0 para rho >= rho2*r
    idw_p: float = 3.0
    pesos: str = "valor_medio"
    a2: float = 0.5
    varreduras: int = 3
    a2_arco: float = 0.0
    pesos_arco: str = "referencia"
    a3: float = 0.0
    chi: float = 2.0
    q_guarda: float = 0.5
    # 0,05: com 0,2 h por passo o corretor sozinho (sem rotacao nem extensao) nao
    # acompanha a ponta da corda e a primeira camada inverte (medido em valvula_10g)
    desloc_max_h: float = 0.05

    @property
    def descricao(self) -> str:
        rot = {"nenhuma": "sem rotacao no interior",
               "radial": f"rotacao radial psi (rho1={self.rho1} r, rho2={self.rho2} r)",
               "idw": f"rotacao por IDW (p={self.idw_p})"}[self.rotacao]
        return (f"{rot}; suavizacao {self.pesos} a2={self.a2} x {self.varreduras} varreduras "
                f"(q_guarda={self.q_guarda}); arcos a2={self.a2_arco} ({self.pesos_arco}); "
                f"extensao harmonica a3={self.a3} (chi={self.chi})")


# Padrao do bloco omitido: so a extensao harmonica com chi = 2 -- o
# comportamento da valvula antes deste plano (caso de controle E0).
CONTROLE = OpcoesMovimento(rotacao="nenhuma", a2=0.0, varreduras=0, a3=1.0)


def opcoes_movimento(cfg: dict | None) -> OpcoesMovimento:
    """Valida o bloco `movimento` e devolve as opcoes. Omitido = `CONTROLE`."""
    if cfg is None:
        return CONTROLE
    if not isinstance(cfg, dict):
        raise ValueError(f"valvula_rotacao.movimento precisa ser um mapa (recebido: {cfg!r})")
    chaves = {"rotacao", "suavizacao", "extensao", "desloc_max_h"}
    if set(cfg) - chaves:
        raise ValueError(f"valvula_rotacao.movimento: chave(s) desconhecida(s) {sorted(set(cfg) - chaves)} "
                         f"(use {', '.join(sorted(chaves))})")

    def bloco(nome, permitidas):
        b = cfg.get(nome) or {}
        if not isinstance(b, dict) or set(b) - permitidas:
            raise ValueError(f"valvula_rotacao.movimento.{nome}: use um mapa com {sorted(permitidas)} "
                             f"(recebido: {b!r})")
        return b

    rot = bloco("rotacao", {"tipo", "rho1", "rho2", "p"})
    suav = bloco("suavizacao", {"pesos", "a2", "varreduras", "a2_arco", "pesos_arco", "q_guarda"})
    ext = bloco("extensao", {"a3", "chi"})
    d = OpcoesMovimento()
    o = OpcoesMovimento(
        rotacao=rot.get("tipo", "radial" if rot else "nenhuma"),
        rho1=float(rot.get("rho1", d.rho1)), rho2=float(rot.get("rho2", d.rho2)), idw_p=float(rot.get("p", d.idw_p)),
        pesos=suav.get("pesos", d.pesos), a2=float(suav.get("a2", d.a2 if suav else 0.0)),
        varreduras=int(suav.get("varreduras", d.varreduras if suav else 0)),
        a2_arco=float(suav.get("a2_arco", d.a2_arco)), pesos_arco=suav.get("pesos_arco", d.pesos_arco),
        q_guarda=float(suav.get("q_guarda", d.q_guarda)),
        a3=float(ext.get("a3", 1.0 if ext else 0.0)), chi=float(ext.get("chi", d.chi)),
        desloc_max_h=float(cfg.get("desloc_max_h", d.desloc_max_h)),
    )
    if o.rotacao not in ROTACOES:
        raise ValueError(f"movimento.rotacao.tipo desconhecido: {o.rotacao!r} (use {', '.join(ROTACOES)})")
    if o.pesos not in PESOS:
        raise ValueError(f"movimento.suavizacao.pesos desconhecido: {o.pesos!r} (use {', '.join(PESOS)})")
    if o.pesos_arco not in PESOS_ARCO:
        raise ValueError(f"movimento.suavizacao.pesos_arco desconhecido: {o.pesos_arco!r} "
                         f"(use {', '.join(PESOS_ARCO)})")
    if o.rotacao == "radial" and not 0 <= o.rho1 < o.rho2:
        raise ValueError(f"movimento.rotacao: precisa 0 <= rho1 < rho2 (recebido {o.rho1}, {o.rho2})")
    for nome, v, lo, hi in (("suavizacao.a2", o.a2, 0, 1), ("suavizacao.a2_arco", o.a2_arco, 0, 1),
                            ("extensao.a3", o.a3, 0, 1), ("suavizacao.q_guarda", o.q_guarda, 0, 1)):
        if not lo <= v <= hi:
            raise ValueError(f"movimento.{nome} precisa estar em [{lo}, {hi}] (recebido {v})")
    if o.varreduras < 0 or o.desloc_max_h <= 0 or o.chi < 0:
        raise ValueError("movimento: varreduras >= 0, desloc_max_h > 0 e chi >= 0")
    return o


# =============================================================================
# 2. Geometria por elemento
# =============================================================================
def area_com_sinal(IEN, X, Y) -> np.ndarray:
    a, b, c = IEN[:, 0], IEN[:, 1], IEN[:, 2]
    return 0.5 * ((X[b] - X[a]) * (Y[c] - Y[a]) - (Y[b] - Y[a]) * (X[c] - X[a]))


def qualidade(IEN, X, Y) -> np.ndarray:
    """`q = 4*sqrt(3)*A / (l1^2+l2^2+l3^2)`: 1 equilatero, 0 degenerado, <0 invertido (Knupp 2001)."""
    A = area_com_sinal(IEN, X, Y)
    a, b, c = IEN[:, 0], IEN[:, 1], IEN[:, 2]
    l2 = ((X[a] - X[b]) ** 2 + (Y[a] - Y[b]) ** 2 + (X[b] - X[c]) ** 2
          + (Y[b] - Y[c]) ** 2 + (X[c] - X[a]) ** 2 + (Y[c] - Y[a]) ** 2)
    return 4 * np.sqrt(3) * A / np.maximum(l2, 1e-300)


def tamanho_local(IEN, X, Y, n: int) -> np.ndarray:
    """`h` por vertice: media do comprimento das arestas que chegam nele."""
    soma, cont = np.zeros(n), np.zeros(n)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        ia, ib = IEN[:, a], IEN[:, b]
        L = np.hypot(X[ia] - X[ib], Y[ia] - Y[ib])
        for i in (ia, ib):
            np.add.at(soma, i, L)
            np.add.at(cont, i, 1)
    return soma / np.maximum(cont, 1)


# =============================================================================
# 3. Pesos da suavizacao
# =============================================================================
def pesos_valor_medio(IEN, X, Y, n: int) -> sp.csr_matrix:
    """Coordenadas de valor medio (Floater 2003), linhas somando 1.

    `mu_ij = (tan(a-/2) + tan(a+/2)) / |x_j - x_i|`, com `a-`, `a+` os angulos
    em `i` dos dois triangulos da aresta `ij`. Positivas em qualquer
    triangulacao valida e exatas para funcoes lineares: `sum_j lam_ij x_j = x_i`
    na malha em que foram calculadas (e em qualquer imagem afim dela).
    """
    linhas, colunas, vals = [], [], []
    for i_, j_, k_ in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        i, j, k = IEN[:, i_], IEN[:, j_], IEN[:, k_]
        ex, ey = X[j] - X[i], Y[j] - Y[i]
        fx, fy = X[k] - X[i], Y[k] - Y[i]
        lj, lk = np.hypot(ex, ey), np.hypot(fx, fy)
        ang = np.arctan2(np.abs(ex * fy - ey * fx), ex * fx + ey * fy)  # angulo em i
        t = np.tan(ang / 2)
        linhas += [i, i]
        colunas += [j, k]
        vals += [t / lj, t / lk]
    W = sp.csr_matrix((np.concatenate(vals), (np.concatenate(linhas), np.concatenate(colunas))), shape=(n, n))
    soma = np.asarray(W.sum(axis=1)).ravel()
    return sp.diags(1.0 / np.where(soma > 0, soma, 1.0)) @ W


def pesos_uniformes(IEN, n: int) -> sp.csr_matrix:
    """`1/|N_i|` para cada vizinho de aresta (Laplaciano de malha, 'umbrella')."""
    r = np.concatenate([IEN[:, a] for a, b in ((0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2))])
    c = np.concatenate([IEN[:, b] for a, b in ((0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2))])
    A = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    A.data[:] = 1.0  # aresta interna aparece em dois triangulos
    soma = np.asarray(A.sum(axis=1)).ravel()
    return sp.diags(1.0 / np.where(soma > 0, soma, 1.0)) @ A


def alvo_odt(IEN, X, Y, n: int):
    """`x_i* = sum |T| c_T / sum |T|` sobre a estrela de `i`, `c_T` o circuncentro (Chen & Holst 2011)."""
    a, b, c = IEN[:, 0], IEN[:, 1], IEN[:, 2]
    ax, ay, bx, by, cx, cy = X[a], Y[a], X[b], Y[b], X[c], Y[c]
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    a2, b2, c2 = ax**2 + ay**2, bx**2 + by**2, cx**2 + cy**2
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    A = np.abs(d) / 4
    sx, sy, sa = np.zeros(n), np.zeros(n), np.zeros(n)
    for v in (a, b, c):
        np.add.at(sx, v, A * ux)
        np.add.at(sy, v, A * uy)
        np.add.at(sa, v, A)
    sa = np.where(sa > 0, sa, 1.0)
    return sx / sa, sy / sa


# =============================================================================
# 4. Termos do preditor
# =============================================================================
def mistura_radial(rho, rho1: float, rho2: float) -> np.ndarray:
    """`psi`: 1 ate `rho1`, cosseno ate 0 em `rho2`."""
    u = np.clip((rho - rho1) / (rho2 - rho1), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * u))


def laplaciano_p1(IEN, X, Y, n: int, chi: float = 0.0) -> sp.csr_matrix:
    """Rigidez P1 `K_ij = INT grad(N_i).grad(N_j)`; `chi > 0` multiplica cada elemento
    por `(A_max/A_e)**chi` (Johnson & Tezduyar 1994): elementos pequenos ficam rigidos."""
    vi, vj, vk = IEN[:, 0], IEN[:, 1], IEN[:, 2]
    b = np.stack([Y[vj] - Y[vk], Y[vk] - Y[vi], Y[vi] - Y[vj]], axis=1)
    c = np.stack([X[vk] - X[vj], X[vi] - X[vk], X[vj] - X[vi]], axis=1)
    area = np.abs(area_com_sinal(IEN, X, Y))
    k = (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]) / (4.0 * area)[:, None, None]
    if chi > 0:
        k = k * ((area.max() / area) ** chi)[:, None, None]
    return sp.csr_matrix((k.ravel(), (np.repeat(IEN, 3, axis=1).ravel(), np.tile(IEN, (1, 3)).ravel())),
                         shape=(n, n))


# =============================================================================
# 5. O movimento
# =============================================================================
class MovimentoValvula:
    """Estado da malha (so vertices) e o passo `theta_atual -> theta_novo`.

    `IEN` so com os 3 vertices. `X`, `Y` (copiados) sao a geometria atual;
    `passo` atualiza no lugar e devolve o diagnostico.
    """

    def __init__(self, contorno: vr.ContornoValvula, IEN, X, Y, opcoes: OpcoesMovimento):
        self.c = contorno
        self.IEN = np.asarray(IEN)[:, :3]
        self.X = np.asarray(X, float).copy()
        self.Y = np.asarray(Y, float).copy()
        self.n = len(self.X)
        self.o = opcoes
        self.theta = contorno.theta0
        self.interior = np.flatnonzero(~contorno.contorno)
        self.h0 = tamanho_local(self.IEN, self.X, self.Y, self.n)
        self.q0 = qualidade(self.IEN, self.X, self.Y)
        if self.q0.min() <= 0:
            raise ValueError("malha inicial com elemento invertido (ou orientacao horaria)")

        if opcoes.pesos == "valor_medio":
            self.W = pesos_valor_medio(self.IEN, self.X, self.Y, self.n)[self.interior]
        elif opcoes.pesos == "uniforme":
            self.W = pesos_uniformes(self.IEN, self.n)[self.interior]
        else:
            self.W = None

        # elementos de cada vertice interior (para a guarda): CSR elemento x no
        ne = len(self.IEN)
        self._E = sp.csr_matrix((np.ones(3 * ne), (np.repeat(np.arange(ne), 3), self.IEN.ravel())),
                                shape=(ne, self.n))

        self._idw = None
        if opcoes.rotacao == "idw":
            b = np.flatnonzero(contorno.contorno)
            d = np.hypot(self.X[self.interior, None] - self.X[None, b], self.Y[self.interior, None] - self.Y[None, b])
            w = 1.0 / np.maximum(d, 1e-300) ** opcoes.idw_p
            self._idw = (b, w / w.sum(axis=1, keepdims=True))

    # --- termos ---------------------------------------------------------
    def _rotacao_interior(self, dth: float, dX: np.ndarray, dY: np.ndarray):
        """Deslocamento de rotacao do interior e o que ele 'cobre' no contorno."""
        o, ii = self.o, self.interior
        rx, ry = np.zeros(self.n), np.zeros(self.n)
        if o.rotacao == "radial":
            rho = np.hypot(self.X, self.Y)
            psi = mistura_radial(rho, o.rho1 * self.c.geo.r, o.rho2 * self.c.geo.r)
            xr, yr = vr.rotacao(dth, self.X, self.Y)
            rx, ry = psi * (xr - self.X), psi * (yr - self.Y)
        elif o.rotacao == "idw":
            b, w = self._idw
            rx[ii], ry[ii] = w @ dX[b], w @ dY[b]
            rx[b], ry[b] = dX[b], dY[b]  # interpolante exato no contorno
        return rx, ry

    def _extensao(self, bx: np.ndarray, by: np.ndarray):
        """Extensao harmonica (K e = 0 no interior) do resto `b` no contorno."""
        K = laplaciano_p1(self.IEN, self.X, self.Y, self.n, self.o.chi)
        ii, B = self.interior, np.flatnonzero(self.c.contorno)
        lu = spla.splu(K[ii][:, ii].tocsc())
        KIB = K[ii][:, B]
        ex, ey = np.zeros(self.n), np.zeros(self.n)
        ex[ii] = lu.solve(-(KIB @ bx[B]))
        ey[ii] = lu.solve(-(KIB @ by[B]))
        return ex, ey

    def _alvo(self):
        ii = self.interior
        if self.W is not None:
            return self.W @ self.X, self.W @ self.Y
        ax, ay = alvo_odt(self.IEN, self.X, self.Y, self.n)
        return ax[ii], ay[ii]

    def _corrigir(self) -> int:
        """`varreduras` de Jacobi com guarda; devolve quantos movimentos foram recusados."""
        o, ii = self.o, self.interior
        recusados = 0
        for _ in range(o.varreduras):
            ax, ay = self._alvo()
            dx, dy = o.a2 * (ax - self.X[ii]), o.a2 * (ay - self.Y[ii])
            q_antes = qualidade(self.IEN, self.X, self.Y)
            X0, Y0 = self.X[ii].copy(), self.Y[ii].copy()
            fator = np.ones(len(ii))
            for _ in range(5):  # 1, 1/2, 1/4, 1/8, depois 0
                self.X[ii], self.Y[ii] = X0 + fator * dx, Y0 + fator * dy
                q = qualidade(self.IEN, self.X, self.Y)
                ruim = (q <= 0) | ((q < q_antes) & (q < o.q_guarda))
                if not ruim.any():
                    break
                nos_ruins = np.flatnonzero(np.asarray(self._E[ruim].sum(axis=0)).ravel() > 0)
                local = np.isin(ii, nos_ruins)
                fator[local] *= 0.5
            else:
                local = np.isin(ii, nos_ruins)
                fator[local] = 0.0
                self.X[ii], self.Y[ii] = X0 + fator * dx, Y0 + fator * dy
            recusados += int((fator < 1).sum())
        return recusados

    # --- passo ------------------------------------------------------------
    def passo(self, theta_novo: float) -> dict:
        """Leva a malha de `self.theta` a `theta_novo`, em subpassos se preciso.

        O contorno mais rapido (ponta da corda) anda `r*|dtheta|`; se isso passar
        de `desloc_max_h` vezes o menor `h` dos vertices moveis, o passo e'
        dividido (plano §5) -- o corretor so acompanha deslocamentos pequenos.
        """
        movel = self.c.idx_movel
        h_min = self.h0[movel].min() if len(movel) else np.inf
        n_sub = max(1, int(np.ceil(self.c.geo.r * abs(theta_novo - self.theta) / (self.o.desloc_max_h * h_min))))
        info, recusados = None, 0
        for th in np.linspace(self.theta, theta_novo, n_sub + 1)[1:]:
            info = self._passo_simples(float(th))
            recusados += info["recusados"]
        info["recusados"], info["subpassos"] = recusados, n_sub
        return info

    def _passo_simples(self, theta_novo: float) -> dict:
        o, c = self.o, self.c
        Xn, Yn = self.X.copy(), self.Y.copy()
        dth = theta_novo - self.theta

        # 1. contorno exato (rigido, depois arcos -- o preditor do arco usa theta)
        idx, xr, yr = vr.posicoes_rigidas(c, theta_novo)
        Xc, Yc = self.X.copy(), self.Y.copy()
        Xc[idx], Yc[idx] = xr, yr
        vr.prever_arcos(c, Xc, Yc, self.theta, theta_novo)
        vr.suavizar_arcos(c, Xc, Yc, o.a2_arco, o.varreduras, o.pesos_arco)
        dX, dY = Xc - self.X, Yc - self.Y
        cont = c.contorno

        # 2. preditor do interior: rotacao + a3 * extensao do resto no contorno
        rx, ry = self._rotacao_interior(dth, dX, dY)
        if o.a3 > 0:
            ex, ey = self._extensao(np.where(cont, dX - rx, 0.0), np.where(cont, dY - ry, 0.0))
            rx, ry = rx + o.a3 * ex, ry + o.a3 * ey
        ii = self.interior
        self.X[ii] += rx[ii]
        self.Y[ii] += ry[ii]
        self.X[cont], self.Y[cont] = Xc[cont], Yc[cont]

        # 3. corretor
        recusados = self._corrigir() if o.a2 > 0 and o.varreduras > 0 else 0
        self.theta = float(theta_novo)

        q = qualidade(self.IEN, self.X, self.Y)
        A = area_com_sinal(self.IEN, self.X, self.Y)
        desloc = np.hypot(self.X - Xn, self.Y - Yn)
        return dict(theta_deg=float(np.degrees(theta_novo)), q_min=float(q.min()), q_med=float(q.mean()),
                    invertidos=int((A <= 0).sum()), area_min=float(A.min()), area_total=float(A.sum()),
                    desloc_max_h=float((desloc / self.h0).max()), recusados=recusados)

    def gradacao(self, raio: float) -> float:
        """Media de `h/h0` nos vertices com `|p| < raio` (1 = gradacao da malha inicial mantida)."""
        h = tamanho_local(self.IEN, self.X, self.Y, self.n)
        sel = np.hypot(self.X, self.Y) < raio
        return float((h[sel] / self.h0[sel]).mean())
