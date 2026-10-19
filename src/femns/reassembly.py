"""Montagem do sistema global, com remontagem barata quando so as coordenadas mudam.

`Remontador` e' o unico caminho de montagem da simulacao: monta o sistema
inicial (`sistema_inicial`) e, com `mesh_motion`, remonta a cada passo
(`remontar`). Com malha movel a **topologia** e' invariante -- so os *valores* das
matrizes de elemento mudam. `Remontador` captura, na primeira montagem, o
padrao de esparsidade e a permutacao do `coalesce` (o mapa "contribuicao de
elemento -> posicao no vetor de valores CSR"); nos passos seguintes so
recomputa os valores de elemento (`elemento.montar`, barato) e os espalha nos
padroes CSR fixos, pulando o `torch.cat` dos 9 blocos, o `.coalesce()` grande
e o `.to_sparse_csr()` -- o grosso do custo da remontagem.

As condicoes de Dirichlet sao eliminadas de forma **simetrica** (linha e
coluna; ver `boundary.apply_boundary_conditions`), entao `A` e' simetrica e a
contribuicao dos valores prescritos vai para o lado direito
(`Sistema.rhs_cc`). O resultado e' algebricamente identico ao caminho
`build_system_matrix` + `apply_boundary_conditions` + `parcela_cc`
(conferido em `Remontador.__init__` contra essa montagem de referencia).
"""

from dataclasses import dataclass

import torch

from femns.solver import pressure_scale_factor


@dataclass
class Sistema:
    """O sistema linear de um passo de tempo, numa geometria.

    - `A` (CSR, `N x N`): matriz de bloco com as CCs eliminadas
      simetricamente (simetrica, indefinida).
    - `M` (CSR): massa de velocidade, sem CC (lado direito `M v*/dt`).
    - `beta`: equilibracao da pressao (`p~ = p / beta`).
    - `rhs_cc` (`N`): parcela das CCs no lado direito -- nas linhas livres,
      `-(A_completa @ g)`; nas prescritas, o valor prescrito `g` (em `p~`
      para a pressao).
    - `eh_cc` (`N`, bool): graus de liberdade com Dirichlet.
    - `Fx`, `Fy` (CSR, `nn x nn`): bloco de velocidade `M/dt + K/Re` com as
      CCs de `vx`/`vy` eliminadas; `Mp`, `Lp` (CSR, `np x np`): massa e
      laplaciano P1 da pressao com as CCs de `p` eliminadas. So existem se o
      `Remontador` foi criado com `precisa_blocos=True` (pre-condicionadores
      de bloco, `femns.precondicionadores`); senao sao `None`.
    """

    A: torch.Tensor
    M: torch.Tensor
    beta: float
    rhs_cc: torch.Tensor
    eh_cc: torch.Tensor
    Fx: torch.Tensor = None
    Fy: torch.Tensor = None
    Mp: torch.Tensor = None
    Lp: torch.Tensor = None
    lp_zero: torch.Tensor = None  # (np,) bool: Dirichlet so do Lp (saida livre), ver Remontador


class _MapaCoalesce:
    """Mapeia os valores brutos de uma matriz COO (na ordem de construcao de
    `ElementoTriangular.montar`, fixa dado o IEN) para os valores coalescidos, sem re-ordenar
    a cada passo. `urow`/`ucol` sao as coordenadas dos slots coalescidos.

    A soma dos duplicados usa `torch.sparse_coo_tensor(...).coalesce()` (soma
    segmentada 1D), **nao** `scatter_add_` -- este ultimo e' nao-deterministico
    na GPU (soma atomica em ordem arbitraria), o que perturbaria `A` em ~1 ULP
    a cada chamada e, com um BiCGSTAB que nao converge (Re alto), estouraria
    para diferencas de ~1e-2 na solucao truncada.
    """

    def __init__(self, indices_brutos: torch.Tensor, n_cols: int):
        flat = indices_brutos[0].to(torch.int64) * n_cols + indices_brutos[1].to(torch.int64)
        uflat, inv = torch.unique(flat, sorted=True, return_inverse=True)
        self._inv = inv.unsqueeze(0)
        self.nnz = uflat.numel()
        self.urow = torch.div(uflat, n_cols, rounding_mode="floor")
        self.ucol = uflat - self.urow * n_cols

    def merge(self, valores_brutos: torch.Tensor) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self._inv, valores_brutos, size=(self.nnz,)).coalesce().values()


def _row_ptr(rows_ordenadas: torch.Tensor, n_linhas: int) -> torch.Tensor:
    """crow (indptr) CSR a partir das linhas ja ordenadas de cada nao-zero."""
    contagem = torch.bincount(rows_ordenadas, minlength=n_linhas)
    crow = torch.zeros(n_linhas + 1, dtype=torch.int64, device=rows_ordenadas.device)
    torch.cumsum(contagem, dim=0, out=crow[1:])
    return crow


def _soma_por_indice(indices: torch.Tensor, valores: torch.Tensor, n: int) -> torch.Tensor:
    """Vetor denso de tamanho `n` com `valores` somados por `indices` (deterministico, ver `_MapaCoalesce`)."""
    if indices.numel() == 0:
        return torch.zeros(n, dtype=valores.dtype, device=valores.device)
    return torch.sparse_coo_tensor(indices.unsqueeze(0), valores, size=(n,)).coalesce().to_dense()


class _Padrao:
    """Padrao CSR fixo de uma matriz `n x n` montada por blocos, com eliminacao simetrica de Dirichlet.

    `blocos` e' uma lista `(nome, linhas, colunas)` com as coordenadas globais
    de cada slot coalescido do bloco. Ficam no CSR as entradas com linha **e**
    coluna livres, mais a diagonal 1 de cada grau de liberdade prescrito
    (`eh_cc`). As entradas com linha livre e coluna prescrita sao as que
    vao para o lado direito (`levantamento`).
    """

    def __init__(self, blocos, n: int, eh_cc: torch.Tensor):
        self.n = n
        dev = eh_cc.device
        cc = torch.nonzero(eh_cc, as_tuple=False).squeeze(1)

        linhas = torch.cat([r for _, r, _ in blocos])
        colunas = torch.cat([c for _, _, c in blocos])
        mantem = ~eh_cc[linhas] & ~eh_cc[colunas]
        final_r = torch.cat([linhas[mantem], cc])
        final_c = torch.cat([colunas[mantem], cc])
        eh_diag_cc = torch.cat([torch.zeros(int(mantem.sum()), dtype=torch.bool, device=dev),
                                torch.ones(cc.numel(), dtype=torch.bool, device=dev)])

        ordem = torch.argsort(final_r.to(torch.int64) * n + final_c.to(torch.int64))
        csr_r = final_r[ordem].to(torch.int64)
        self.col = final_c[ordem].to(torch.int64)
        self.crow = _row_ptr(csr_r, n)
        self.nnz = self.col.numel()
        self.cc_slots = torch.nonzero(eh_diag_cc[ordem], as_tuple=False).squeeze(1)

        # destino de cada slot de bloco no vetor de valores CSR (os que saem
        # pela eliminacao vao para o "lixo" no indice nnz) e as entradas que
        # formam o levantamento (linha livre, coluna prescrita)
        csr_flat = csr_r * n + self.col
        self.destinos, self.levantamento_idx = {}, {}
        for nome, r, c in blocos:
            r, c = r.to(torch.int64), c.to(torch.int64)
            bflat = r * n + c
            pos = torch.searchsorted(csr_flat, bflat)
            acerta = ~eh_cc[r] & ~eh_cc[c] & (csr_flat[pos.clamp(max=self.nnz - 1)] == bflat)
            self.destinos[nome] = torch.where(acerta, pos, torch.full_like(pos, self.nnz))
            lev = ~eh_cc[r] & eh_cc[c]
            self.levantamento_idx[nome] = (torch.nonzero(lev, as_tuple=False).squeeze(1), r[lev], c[lev])

    def csr(self, val_por_bloco: dict) -> torch.Tensor:
        """Matriz CSR com os valores de cada bloco (vetor de valores novo a cada chamada)."""
        dtype = next(iter(val_por_bloco.values())).dtype
        vals = torch.zeros(self.nnz + 1, dtype=dtype, device=self.col.device)
        for nome, destino in self.destinos.items():
            vals.scatter_(0, destino, val_por_bloco[nome])
        vals[self.cc_slots] = 1.0
        return torch.sparse_csr_tensor(self.crow, self.col, vals[:self.nnz], size=(self.n, self.n))

    def levantamento(self, val_por_bloco: dict, g: torch.Tensor) -> torch.Tensor:
        """`(A_completa @ g)` nas linhas livres -- o que as colunas eliminadas levariam."""
        linhas, valores = [], []
        for nome, (idx, r, c) in self.levantamento_idx.items():
            if idx.numel():
                linhas.append(r)
                valores.append(val_por_bloco[nome][idx] * g[c])
        if not linhas:
            return torch.zeros(self.n, dtype=g.dtype, device=g.device)
        return _soma_por_indice(torch.cat(linhas), torch.cat(valores), self.n)


class Remontador:
    """Monta o `Sistema` -- uma vez na criacao (`sistema_inicial`) e de novo a
    cada passo de malha movel (`remontar`), so recomputando valores.

    `dt` e `Re` sao fixos ao longo da simulacao. `beta` (equilibracao da
    pressao, `solver.pressure_scale_factor`) e' calculado na geometria inicial
    se nao for dado, e pode variar por passo (ver `remontar`).

    `valores_cc` = `(vx_cc, vy_cc, p_cc)` sao os valores prescritos
    (`boundary.build_boundary_conditions`); sem eles, valem zero.
    `precisa_blocos=True` monta tambem `Fx`, `Fy`, `Mp`, `Lp` (ver `Sistema`).
    `nos_saida_livre` (vertices de contornos sem velocidade prescrita, isto
    e', saidas com a condicao natural) recebem Dirichlet **so no `Lp`** do
    pre-condicionador: sem pressao prescrita em lugar nenhum, o `Lp` de
    Neumann puro e' singular. E' a pratica para os operadores de pressao dos
    pre-condicionadores de Schur (Elman, Silvester & Wathen 2014, cap. 9).
    Nao muda o sistema resolvido. Se nao houver nem pressao prescrita nem
    saida livre, o primeiro vertice e' fixado no `Lp`.

    `verificar=True` (padrao) confere a montagem inicial contra o caminho de
    referencia `build_system_matrix` + `apply_boundary_conditions` +
    `parcela_cc` -- e' a garantia de que o mapa de valores esta certo; nao
    desligue em simulacao.
    """

    def __init__(self, dt, Re, X, Y, IEN, elemento, npoints, n_extra,
                 vx_cc_pts, vy_cc_pts, p_cc_pts, beta=None, valores_cc=None,
                 precisa_blocos=False, nos_saida_livre=None, verificar=True):
        self.dt, self.Re = float(dt), float(Re)
        self.IEN = IEN
        self.elemento = elemento  # femns.elemento.MINI ou TRI6
        self.npoints = npoints
        self.nnodes = npoints + n_extra
        self.precisa_blocos = precisa_blocos
        nn, npo = self.nnodes, npoints
        self.N = 2 * nn + npo
        dev = X.device

        K, M, Gx, Gy = elemento.montar(X, Y, IEN, npo, nn)
        if beta is None:
            beta = pressure_scale_factor(K, M, Gx, Gy, dt, Re)

        # K, M compartilham o padrao (nn x nn); Gx, Gy compartilham (nn x npo)
        self._m_nn = _MapaCoalesce(K._indices(), nn)
        self._m_n3 = _MapaCoalesce(Gx._indices(), npo)

        # graus de liberdade com Dirichlet e valores prescritos (g, em p~ para a pressao)
        self.eh_cc = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.eh_cc[vx_cc_pts] = True
        self.eh_cc[vy_cc_pts.to(torch.int64) + nn] = True
        self.eh_cc[p_cc_pts.to(torch.int64) + 2 * nn] = True
        self._g_vel = torch.zeros(self.N, dtype=X.dtype, device=dev)  # vx, vy prescritos; pressao a parte (depende de beta)
        self._g_p = torch.zeros(self.N, dtype=X.dtype, device=dev)
        self._vx_pts = vx_cc_pts.to(torch.int64)
        self._vy_pts = vy_cc_pts.to(torch.int64)
        if valores_cc is not None:
            vx_cc, vy_cc, p_cc = valores_cc
            self._g_vel[vx_cc_pts] = vx_cc[vx_cc_pts]
            self._g_vel[vy_cc_pts.to(torch.int64) + nn] = vy_cc[vy_cc_pts]
            self._g_p[p_cc_pts.to(torch.int64) + 2 * nn] = p_cc[p_cc_pts]

        # --- padrao de A: 6 blocos disjuntos (linha, coluna) globais ---
        mn, m3 = self._m_nn, self._m_n3
        self._pA = _Padrao([
            ("vx", mn.urow, mn.ucol),
            ("vy", mn.urow + nn, mn.ucol + nn),
            ("gx", m3.urow, m3.ucol + 2 * nn),
            ("gy", m3.urow + nn, m3.ucol + 2 * nn),
            ("dx", m3.ucol + 2 * nn, m3.urow),
            ("dy", m3.ucol + 2 * nn, m3.urow + nn),
        ], self.N, self.eh_cc)

        # --- M como CSR (padrao proprio = m_nn, sem CC) ---
        self._m_crow = _row_ptr(self._m_nn.urow, nn)
        self._m_col = self._m_nn.ucol.to(torch.int64)

        # --- blocos para os pre-condicionadores ---
        if precisa_blocos:
            self._pFx = _Padrao([("F", mn.urow, mn.ucol)], nn, self.eh_cc[:nn])
            self._pFy = _Padrao([("F", mn.urow, mn.ucol)], nn, self.eh_cc[nn:2 * nn])
            Mp, _ = elemento.matrizes_pressao(X, Y, IEN, npo)
            self._m_pp = _MapaCoalesce(Mp._indices(), npo)
            self._pP = _Padrao([("P", self._m_pp.urow, self._m_pp.ucol)], npo, self.eh_cc[2 * nn:])
            eh_lp = self.eh_cc[2 * nn:].clone()
            if nos_saida_livre is not None and len(nos_saida_livre):
                eh_lp[torch.as_tensor(nos_saida_livre, dtype=torch.int64, device=dev)] = True
            if not eh_lp.any():
                eh_lp[0] = True
            self._lp_zero = eh_lp & ~self.eh_cc[2 * nn:]
            self._pL = _Padrao([("P", self._m_pp.urow, self._m_pp.ucol)], npo, eh_lp)

        self.erro_verificacao = None
        if verificar:
            self.erro_verificacao = self._verificar(K, M, Gx, Gy, beta, vx_cc_pts, vy_cc_pts, p_cc_pts, n_extra)

        # O `Sistema` na geometria de criacao.
        self.sistema_inicial = self._saida(X, Y, K, M, Gx, Gy, beta)

    def _verificar(self, K, M, Gx, Gy, beta, vx_cc_pts, vy_cc_pts, p_cc_pts, n_extra) -> float:
        from femns.boundary import apply_boundary_conditions, parcela_cc
        from femns.solver import build_system_matrix

        A_completa = build_system_matrix(self.dt, self.Re, K, M, Gx, Gy, beta)
        A_ref = apply_boundary_conditions(A_completa, vx_cc_pts, vy_cc_pts, p_cc_pts, self.npoints, n_extra)
        A_ref = A_ref.to_sparse_csr()
        if not (torch.equal(A_ref.crow_indices().to(torch.int64), self._pA.crow)
                and torch.equal(A_ref.col_indices().to(torch.int64), self._pA.col)):
            raise AssertionError("Remontador: padrao de esparsidade nao bate com a montagem de referencia")

        val = self._valores_blocos(K, M, Gx, Gy, beta)
        A_vals = self._pA.csr(val).values()
        escala = A_ref.values().abs().max().clamp_min(1e-30)
        erro_A = ((A_vals - A_ref.values()).abs().max() / escala).item()

        g = self._g(beta)
        rhs_ref = parcela_cc(A_completa, g, self.eh_cc)
        rhs = self._rhs_cc(val, g)
        escala_rhs = rhs_ref.abs().max().clamp_min(1e-30)
        erro_rhs = ((rhs - rhs_ref).abs().max() / escala_rhs).item()

        erro = max(erro_A, erro_rhs)
        if erro > 1e-10:
            raise AssertionError(
                f"Remontador: valores divergem da montagem de referencia "
                f"(erro relativo maximo: A {erro_A:.2e}, lado direito das CCs {erro_rhs:.2e})")
        return erro

    def _valores_blocos(self, K, M, Gx, Gy, beta) -> dict:
        Kc = self._m_nn.merge(K._values())
        Mc = self._m_nn.merge(M._values())
        Gxc = self._m_n3.merge(Gx._values())
        Gyc = self._m_n3.merge(Gy._values())
        vel = Mc / self.dt + Kc / self.Re
        return {
            "vx": vel, "vy": vel,
            "gx": -beta * Gxc, "gy": -beta * Gyc,
            "dx": -beta * Gxc, "dy": -beta * Gyc,
            "_M": Mc, "_gx": Gxc, "_gy": Gyc,
        }

    def _g(self, beta) -> torch.Tensor:
        """Valores prescritos em todos os graus de liberdade (pressao em p~ = p/beta)."""
        return self._g_vel + self._g_p / beta

    def _rhs_cc(self, val, g) -> torch.Tensor:
        return torch.where(self.eh_cc, g, -self._pA.levantamento(val, g))

    def definir_valores_cc(self, vx_cc: torch.Tensor, vy_cc: torch.Tensor) -> None:
        """Troca os valores prescritos de velocidade (mesmos nos com Dirichlet).

        Para paredes moveis, cuja velocidade muda a cada passo: vale a partir
        da proxima `remontar` (o lado direito das CCs, `rhs_cc`, e' refeito la).
        """
        self._g_vel[self._vx_pts] = vx_cc[self._vx_pts]
        self._g_vel[self._vy_pts + self.nnodes] = vy_cc[self._vy_pts]

    def remontar(self, X, Y, beta=None) -> Sistema:
        """Remonta na geometria `(X, Y)`. `beta=None` recalcula a equilibracao
        (`pressure_scale_factor`) nesta geometria; um valor fixa a escala."""
        K, M, Gx, Gy = self.elemento.montar(X, Y, self.IEN, self.npoints, self.nnodes)
        if beta is None:
            beta = pressure_scale_factor(K, M, Gx, Gy, self.dt, self.Re)
        return self._saida(X, Y, K, M, Gx, Gy, beta)

    def _saida(self, X, Y, K, M, Gx, Gy, beta) -> Sistema:
        val = self._valores_blocos(K, M, Gx, Gy, beta)
        A = self._pA.csr(val)
        M_csr = torch.sparse_csr_tensor(self._m_crow, self._m_col, val["_M"], size=(self.nnodes, self.nnodes))
        sistema = Sistema(A=A, M=M_csr, beta=beta, rhs_cc=self._rhs_cc(val, self._g(beta)), eh_cc=self.eh_cc)

        if self.precisa_blocos:
            sistema.Fx = self._pFx.csr({"F": val["vx"]})
            sistema.Fy = self._pFy.csr({"F": val["vy"]})
            Mp, Lp = self.elemento.matrizes_pressao(X, Y, self.IEN, self.npoints)
            sistema.Mp = self._pP.csr({"P": self._m_pp.merge(Mp._values())})
            sistema.Lp = self._pL.csr({"P": self._m_pp.merge(Lp._values())})
            sistema.lp_zero = self._lp_zero

        # Gx/Gy coalescidos da geometria atual, so para `divergencia` (o
        # operador D so e' montado se alguem pedir a divergencia).
        self._gx, self._gy = val["_gx"], val["_gy"]
        self._D = None
        return sistema

    def divergencia(self, vx, vy):
        """Divergencia discreta `D_x vx + D_y vy` da velocidade, na geometria da ultima montagem.

        `D = -G^T` **sem** o fator `beta` -- e' a restricao fisica de
        incompressibilidade; `beta` e' so escala numerica do solver (ver
        `solver.build_system_matrix`, que usa a mesma convencao no bloco de
        continuidade de `A`). Usada como metrica (planilha de benchmark),
        nunca entra no sistema resolvido. Os operadores sao montados na
        primeira chamada depois de cada montagem e reaproveitados.
        """
        if self._D is None:
            indices = torch.stack([self._m_n3.urow, self._m_n3.ucol])
            tamanho = (self.nnodes, self.npoints)
            self._D = tuple(
                (-torch.transpose(torch.sparse_coo_tensor(indices, g, size=tamanho).coalesce(), 0, 1))
                .coalesce().to_sparse_csr()
                for g in (self._gx, self._gy))
        Dx, Dy = self._D
        return torch.mm(Dx, vx.unsqueeze(1)).squeeze(1) + torch.mm(Dy, vy.unsqueeze(1)).squeeze(1)
