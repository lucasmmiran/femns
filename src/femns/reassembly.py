"""Remontagem barata do sistema quando so as coordenadas da malha mudam.

Com `mesh_motion` a malha se move a cada passo e o sistema `A` precisa ser
remontado junto. Mas a **topologia** e' invariante -- so os *valores* das
matrizes de elemento mudam. `Remontador` captura, na primeira montagem, o
padrao de esparsidade e a permutacao do `coalesce` (o mapa "contribuicao de
elemento -> posicao no buffer de valores CSR"); nos passos seguintes so
recomputa os valores de elemento (`assemble_*`, barato) e faz `scatter_add`
nos buffers CSR fixos, pulando o `torch.cat` dos 9 blocos, o `.coalesce()`
grande e o `.to_sparse_csr()` -- o grosso do custo da remontagem.

O resultado e' algebricamente identico ao caminho
`build_system_matrix` + `apply_boundary_conditions` + `.to_sparse_csr()`
(conferido em `Remontador.__init__` contra a montagem de referencia).
"""

import torch

from femns.assembly import assemble_mini, assemble_tri6
from femns.solver import pressure_scale_factor


class _MapaCoalesce:
    """Mapeia os valores brutos de uma matriz COO (na ordem de construcao de
    `assemble_*`, fixa dado o IEN) para os valores coalescidos, sem re-ordenar
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


def _assemble(element, X, Y, IEN, ne, npoints, nnodes):
    if element == "tri6":
        return assemble_tri6(X, Y, IEN, npoints, nnodes)
    return assemble_mini(X, Y, IEN, ne, npoints)


def _row_ptr(rows_ordenadas: torch.Tensor, n_linhas: int) -> torch.Tensor:
    """crow (indptr) CSR a partir das linhas ja ordenadas de cada nao-zero."""
    contagem = torch.bincount(rows_ordenadas, minlength=n_linhas)
    crow = torch.zeros(n_linhas + 1, dtype=torch.int64, device=rows_ordenadas.device)
    torch.cumsum(contagem, dim=0, out=crow[1:])
    return crow


class Remontador:
    """Remonta `A` (CSR), `M` (CSR) e, se `precisa_gv`, `Gvx`/`Gvy` (CSR) a
    cada passo de malha movel, so recomputando valores.

    `dt` e `Re` sao fixos ao longo da simulacao. `beta` pode variar por passo
    (equilibracao recalculada -- ver `remontar`).
    """

    def __init__(self, dt, Re, X, Y, IEN, element, npoints, n_extra,
                 vx_cc_pts, vy_cc_pts, p_cc_pts, beta_ref, precisa_gv=False,
                 verificar=True):
        self.dt, self.Re = float(dt), float(Re)
        self.IEN = IEN
        self.element = element
        self.npoints = npoints
        self.ne = IEN.shape[0]
        self.nnodes = npoints + n_extra
        self.precisa_gv = precisa_gv
        nn, npo = self.nnodes, npoints
        self.N = 2 * nn + npo
        dev = X.device
        dtype = X.dtype

        K, M, Gx, Gy, Gvx, Gvy = _assemble(element, X, Y, IEN, self.ne, npo, nn)

        # K, M, Gvx, Gvy compartilham o padrao (nn x nn); Gx, Gy compartilham (nn x npo)
        self._m_nn = _MapaCoalesce(K._indices(), nn)
        self._m_n3 = _MapaCoalesce(Gx._indices(), npo)

        # --- padrao de A: 6 blocos disjuntos (linha, coluna) globais ---
        # (bloco, mapa, offset_linha, offset_coluna, transpor)
        blocos = [
            ("vx", self._m_nn, 0, 0, False),
            ("vy", self._m_nn, nn, nn, False),
            ("gx", self._m_n3, 0, 2 * nn, False),
            ("gy", self._m_n3, nn, 2 * nn, False),
            ("dx", self._m_n3, 2 * nn, 0, True),
            ("dy", self._m_n3, 2 * nn, nn, True),
        ]

        is_bc_row = torch.zeros(self.N, dtype=torch.bool, device=dev)
        is_bc_row[vx_cc_pts] = True
        is_bc_row[vy_cc_pts.to(torch.int64) + nn] = True
        is_bc_row[p_cc_pts.to(torch.int64) + 2 * nn] = True
        bc_rows = torch.unique(torch.nonzero(is_bc_row, as_tuple=False).squeeze(1))

        rows_pre, cols_pre = [], []
        self._blocos = []
        for nome, mapa, off_r, off_c, transp in blocos:
            if transp:
                r = mapa.ucol + off_r
                c = mapa.urow + off_c
            else:
                r = mapa.urow + off_r
                c = mapa.ucol + off_c
            self._blocos.append((nome, mapa, r, c))
            rows_pre.append(r)
            cols_pre.append(c)
        rows_pre = torch.cat(rows_pre)
        cols_pre = torch.cat(cols_pre)

        # --- aplica condicao de contorno: dropa linhas BC, poe diagonal 1 ---
        mantem = ~is_bc_row[rows_pre]
        final_rows = torch.cat([rows_pre[mantem], bc_rows])
        final_cols = torch.cat([cols_pre[mantem], bc_rows])
        eh_diag_bc = torch.cat([
            torch.zeros(int(mantem.sum()), dtype=torch.bool, device=dev),
            torch.ones(bc_rows.numel(), dtype=torch.bool, device=dev),
        ])

        ordem = torch.argsort(final_rows.to(torch.int64) * self.N + final_cols.to(torch.int64))
        csr_rows = final_rows[ordem]
        self._csr_col = final_cols[ordem].to(torch.int64)
        self._csr_crow = _row_ptr(csr_rows, self.N)
        self._nnz_A = self._csr_col.numel()
        self._bc_slots = torch.nonzero(eh_diag_bc[ordem], as_tuple=False).squeeze(1)

        # posicao no buffer CSR de cada contribuicao de bloco que sobrevive a BC
        # (mapa slot-do-bloco -> slot CSR); os que caem em linha BC vao pro
        # "lixo" no indice self._nnz_A.
        csr_flat = csr_rows.to(torch.int64) * self.N + self._csr_col
        for i, (nome, mapa, r, c) in enumerate(self._blocos):
            bflat = r.to(torch.int64) * self.N + c.to(torch.int64)
            pos = torch.searchsorted(csr_flat, bflat)
            pos_cl = pos.clamp(max=self._nnz_A - 1)
            acerta = (~is_bc_row[r]) & (csr_flat[pos_cl] == bflat)
            destino = torch.where(acerta, pos, torch.full_like(pos, self._nnz_A))
            self._blocos[i] = (nome, mapa, destino)

        # --- M como CSR (padrao proprio = m_nn) ---
        self._m_crow = _row_ptr(self._m_nn.urow, nn)
        self._m_col = self._m_nn.ucol.to(torch.int64)

        # buffer reutilizado entre passos
        self._buf = torch.zeros(self._nnz_A + 1, dtype=dtype, device=dev)

        self.erro_verificacao = None
        if verificar:
            from femns.boundary import apply_boundary_conditions
            from femns.solver import build_system_matrix
            A_ref = build_system_matrix(dt, Re, K, M, Gx, Gy, beta_ref)
            A_ref = apply_boundary_conditions(A_ref, vx_cc_pts, vy_cc_pts, p_cc_pts, npo, n_extra)
            A_ref = A_ref.to_sparse_csr()
            if not (torch.equal(A_ref.crow_indices().to(torch.int64), self._csr_crow)
                    and torch.equal(A_ref.col_indices().to(torch.int64), self._csr_col)):
                raise AssertionError(
                    "Remontador: padrao de esparsidade nao bate com a montagem de referencia")
            A_vals, _ = self._valores_A(K, M, Gx, Gy, beta_ref)
            denom = A_ref.values().abs().max().clamp_min(1e-30)
            self.erro_verificacao = ((A_vals - A_ref.values()).abs().max() / denom).item()
            if self.erro_verificacao > 1e-10:
                raise AssertionError(
                    f"Remontador: valores divergem da montagem de referencia "
                    f"(erro relativo maximo {self.erro_verificacao:.2e})")

    def _valores_A(self, K, M, Gx, Gy, beta):
        Kc = self._m_nn.merge(K._values())
        Mc = self._m_nn.merge(M._values())
        Gxc = self._m_n3.merge(Gx._values())
        Gyc = self._m_n3.merge(Gy._values())
        vel = Mc / self.dt + Kc / self.Re

        val_por_bloco = {
            "vx": vel, "vy": vel,
            "gx": -beta * Gxc, "gy": -beta * Gyc,
            "dx": -beta * Gxc, "dy": -beta * Gyc,
        }
        self._buf.zero_()
        for nome, _mapa, destino in self._blocos:
            self._buf.scatter_(0, destino, val_por_bloco[nome])
        self._buf[self._bc_slots] = 1.0
        return self._buf[:self._nnz_A], Mc

    def remontar(self, X, Y, beta=None):
        """Remonta na geometria `(X, Y)`. `beta=None` recalcula a equilibracao
        (`pressure_scale_factor`) nesta geometria; um valor fixa a escala.

        Retorna `(A_csr, M_csr, Gvx_csr, Gvy_csr, Gx_coo, Gy_coo, beta)` --
        mesma ordem de `run_simulation.montar_sistema`. `Gvx/Gvy` sao `None`
        quando `precisa_gv=False` (caminho semi-Lagrangeano).
        """
        K, M, Gx, Gy, Gvx, Gvy = _assemble(
            self.element, X, Y, self.IEN, self.ne, self.npoints, self.nnodes)
        if beta is None:
            beta = pressure_scale_factor(K, M, Gx, Gy, self.dt, self.Re)

        A_vals, Mc = self._valores_A(K, M, Gx, Gy, beta)
        A_csr = torch.sparse_csr_tensor(self._csr_crow, self._csr_col, A_vals,
                                        size=(self.N, self.N))
        M_csr = torch.sparse_csr_tensor(self._m_crow, self._m_col, Mc,
                                        size=(self.nnodes, self.nnodes))

        gvx_csr = gvy_csr = None
        if self.precisa_gv:
            gvx_csr = torch.sparse_csr_tensor(
                self._m_crow, self._m_col, self._m_nn.merge(Gvx._values()),
                size=(self.nnodes, self.nnodes))
            gvy_csr = torch.sparse_csr_tensor(
                self._m_crow, self._m_col, self._m_nn.merge(Gvy._values()),
                size=(self.nnodes, self.nnodes))

        gx_coo = torch.sparse_coo_tensor(
            torch.stack([self._m_n3.urow, self._m_n3.ucol]), self._m_n3.merge(Gx._values()),
            size=(self.nnodes, self.npoints)).coalesce()
        gy_coo = torch.sparse_coo_tensor(
            torch.stack([self._m_n3.urow, self._m_n3.ucol]), self._m_n3.merge(Gy._values()),
            size=(self.nnodes, self.npoints)).coalesce()

        return A_csr, M_csr, gvx_csr, gvy_csr, gx_coo, gy_coo, beta
