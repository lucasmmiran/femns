"""Tipos de elemento finito (MINI e Tri6) como classes.

`ElementoTriangular` reune o que so depende dos 3 vertices do triangulo
(geometria, indices das matrizes globais, montagem) e cada subclasse define
o que e' proprio do elemento: os nos extras, as matrizes locais e as funcoes
de forma (interpolacao do semi-Lagrangeano).

Uma instancia representa o **tipo** de elemento, nao um triangulo: todos os
metodos recebem e devolvem arrays com **todos** os elementos da malha de uma
vez, entao a vetorizacao (e a execucao na GPU) e' a mesma de antes. A escolha
e' feita uma vez, no inicio da simulacao (`elemento_por_nome`), e o objeto e'
passado adiante -- em vez de cada modulo testar `element == "tri6"`.
"""

from typing import NamedTuple

import numpy as np
import torch

from femns.mesh import elem_mini, elem_tri6, estende_IENbound_tri6


class Geometria(NamedTuple):
    """Area e coeficientes geometricos de cada elemento (vetores de tamanho `ne`).

    `bi = y_j - y_k`, `ci = x_k - x_j` (e ciclicamente): `dλ_i/dx = bi/(2A)`,
    `dλ_i/dy = ci/(2A)`. A area e' tomada em modulo -- ver `mesh.area_com_sinal`
    para detectar elemento invertido.
    """

    area: torch.Tensor
    bi: torch.Tensor
    bj: torch.Tensor
    bk: torch.Tensor
    ci: torch.Tensor
    cj: torch.Tensor
    ck: torch.Tensor


class ElementoTriangular:
    """Base: tudo que so depende dos 3 vertices. Ver as subclasses `Mini` e `Tri6`."""

    nome: str
    n_nos: int  # nos de velocidade por elemento (colunas de IEN)

    def __repr__(self):
        return f"{type(self).__name__}()"

    # -- geometria e montagem (torch) --------------------------------------

    @staticmethod
    def geometria(X: torch.Tensor, Y: torch.Tensor, IEN: torch.Tensor) -> Geometria:
        """Area e `b`, `c` de todos os elementos, a partir dos 3 primeiros nos de `IEN`."""
        device = X.device

        vi = IEN[:, 0].to(device)
        vj = IEN[:, 1].to(device)
        vk = IEN[:, 2].to(device)

        xi, yi = X[vi], Y[vi]
        xj, yj = X[vj], Y[vj]
        xz, yz = X[vk], Y[vk]

        area = 0.5 * torch.abs(
            xi * (yj - yz) +
            xj * (yz - yi) +
            xz * (yi - yj)
        )

        bi = Y[vj] - Y[vk]
        bj = Y[vk] - Y[vi]
        bk = Y[vi] - Y[vj]

        ci = X[vk] - X[vj]
        cj = X[vi] - X[vk]
        ck = X[vj] - X[vi]

        return Geometria(area, bi, bj, bk, ci, cj, ck)

    @staticmethod
    def indices_coo(IEN: torch.Tensor, n_linhas: int, n_colunas: int) -> torch.Tensor:
        """Indices (linha, coluna) globais de todas as entradas das matrizes locais `n_linhas x n_colunas`.

        Para um elemento com nos `[a, b, c]` e matriz 3x3:
        `repeat_interleave` da as linhas `a a a b b b c c c` e `repeat` as
        colunas `a b c a b c a b c` -- os 9 pares na mesma ordem em que a
        matriz local e' achatada (linha por linha). As colunas sao sempre os
        primeiros `n_colunas` nos (3 = vertices, para os blocos de pressao).
        """
        linhas = IEN[:, :n_linhas].repeat_interleave(n_colunas, dim=1)
        colunas = IEN[:, :n_colunas].repeat(1, n_linhas)
        return torch.stack([linhas, colunas], dim=0).reshape(2, -1)

    def montar(self, X: torch.Tensor, Y: torch.Tensor, IEN: torch.Tensor, npoints: int, nnodes: int):
        """Monta as matrizes globais esparsas (COO, nao coalescidas) do elemento.

        - X, Y: coordenadas de todos os nos (vertices seguidos dos nos extras).
        - IEN: conectividade com `n_nos` colunas (ver `adiciona_nos`).
        - npoints: numero de vertices (graus de liberdade de pressao, P1).
        - nnodes: numero total de nos de velocidade.

        Retorna `K, M, Gx, Gy`, no device de X/Y. `Gx`/`Gy` tem a
        forma fraca `INT(phi_j dN_i/dx)` nos dois elementos, o que deixa o
        sistema de sela simetrico (ver `solver.build_system_matrix`).
        """
        device = X.device
        k, m, gx, gy = self.valores_locais(self.geometria(X, Y, IEN))

        idx_nn = self.indices_coo(IEN, self.n_nos, self.n_nos)
        idx_n3 = self.indices_coo(IEN, self.n_nos, 3)

        def esparsa(indices, valores, n_colunas):
            return torch.sparse_coo_tensor(indices, valores, size=(nnodes, n_colunas), dtype=torch.float64).to(device)

        return (esparsa(idx_nn, k, nnodes), esparsa(idx_nn, m, nnodes),
                esparsa(idx_n3, gx, npoints), esparsa(idx_n3, gy, npoints))

    def matrizes_pressao(self, X: torch.Tensor, Y: torch.Tensor, IEN: torch.Tensor, npoints: int):
        """Massa `Mp` e laplaciano `Lp` P1 nos vertices (espaco da pressao), COO nao coalescidas.

        Iguais para MINI e Tri6 (a pressao e' P1 nos dois): so usam os 3
        vertices de cada elemento. Por elemento,

            Mp_e = (A/12) [[2, 1, 1], [1, 2, 1], [1, 1, 2]]
            Lp_e = (b b^T + c c^T) / (4A)

        Nao entram no sistema resolvido: sao os ingredientes do complemento de
        Schur de Cahouet-Chabard usado pelos pre-condicionadores de bloco
        (`femns.precondicionadores`).
        """
        device = X.device
        geo = self.geometria(X, Y, IEN)
        b = torch.stack([geo.bi, geo.bj, geo.bk], dim=1)
        c = torch.stack([geo.ci, geo.cj, geo.ck], dim=1)

        base_mp = torch.tensor([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]], dtype=X.dtype, device=device)
        mp = (geo.area / 12.0)[:, None, None] * base_mp
        lp = (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]) / (4.0 * geo.area)[:, None, None]

        idx = self.indices_coo(IEN, 3, 3)
        tamanho = (npoints, npoints)
        return (torch.sparse_coo_tensor(idx, mp.flatten(), size=tamanho, dtype=torch.float64).to(device),
                torch.sparse_coo_tensor(idx, lp.flatten(), size=tamanho, dtype=torch.float64).to(device))

    def valores_locais(self, geo: Geometria):
        """Valores das matrizes locais de todos os elementos, ja achatados:
        `K, M` com `n_nos**2` entradas por elemento e `Gx, Gy` com `3*n_nos`.
        Ordem de retorno: `K, M, Gx, Gy`."""
        raise NotImplementedError

    # -- malha (numpy) -----------------------------------------------------

    def adiciona_nos(self, IEN: np.ndarray, X: np.ndarray, Y: np.ndarray, IENbound: np.ndarray):
        """Acrescenta os nos extras do elemento. Retorna `(IEN, X, Y, IENbound)` estendidos."""
        raise NotImplementedError

    # -- semi-Lagrangeano (numpy) ------------------------------------------

    @staticmethod
    def recorta_lambdas(li, lj, lk):
        """Recorta as coordenadas baricentricas para [0, 1] e renormaliza (soma 1).

        Absorve os pequenos negativos que a tolerancia da busca por elemento
        deixa passar (`semi_lagrangian.Localizador.localizar`). A mesma
        politica vale para MINI e Tri6 -- no Tri6 ela foi herdada por analogia,
        nao re-derivada para a base quadratica (ver `docs/lambda.md`).
        Funciona com escalares e arrays numpy.
        """
        li, lj, lk = np.clip(li, 0.0, 1.0), np.clip(lj, 0.0, 1.0), np.clip(lk, 0.0, 1.0)
        soma = li + lj + lk
        return li / soma, lj / soma, lk / soma

    def funcoes_forma(self, li, lj, lk) -> tuple:
        """Valores das `n_nos` funcoes de forma nas coordenadas baricentricas
        dadas, na ordem das colunas de `IEN`."""
        raise NotImplementedError

    def interpolar(self, li, lj, lk, nos: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Valor do campo nodal `v` nos pontos de coordenadas baricentricas
        `(li, lj, lk)` (recortadas, ver `recorta_lambdas`), cada um no
        elemento cujos nos sao a linha de `nos` (`IEN[elem]`, `n_nos` colunas).

        Soma `N_0 v_0 + N_1 v_1 + ...` da esquerda para a direita, na ordem
        das colunas de `IEN`.
        """
        N = self.funcoes_forma(*self.recorta_lambdas(li, lj, lk))
        total = N[0] * v[nos[:, 0]]
        for a in range(1, self.n_nos):
            total = total + N[a] * v[nos[:, a]]
        return total

    def valores_na_aresta(self, IEN: np.ndarray, e: np.ndarray, f: np.ndarray,
                           ib1: np.ndarray, ib2: np.ndarray, w1: np.ndarray, w2: np.ndarray,
                           v: np.ndarray) -> np.ndarray:
        """Valor do campo `v` no ponto `w1*B1 + w2*B2` da aresta `f` do elemento
        `e` (vertices `ib1`, `ib2`) -- ver `semi_lagrangian._fallback_interceptacao`."""
        raise NotImplementedError


class Mini(ElementoTriangular):
    """Elemento MINI: velocidade P1 + bolha (4 nos, o 4o e' o centroide), pressao P1.

    Base nodal: `N_i = λ_i - 9 λ1 λ2 λ3`, `N_b = 27 λ1 λ2 λ3` (o DOF do
    centroide e' a velocidade fisica ali, nao um coeficiente modal).
    """

    nome = "mini"
    n_nos = 4

    def valores_locais(self, geo: Geometria):
        area, bi, bj, bk, ci, cj, ck = geo

        zx = (1.0 / (4.0 * area)) * (bj**2 + bj * bk + bk**2)
        zy = (1.0 / (4.0 * area)) * (cj**2 + cj * ck + ck**2)

        kx_elem_values = torch.stack([
            (1.0 / (4.0 * area)) * bi * bi + (9 / 10) * zx, (1.0 / (4.0 * area)) * bi * bj + (9 / 10) * zx, (1.0 / (4.0 * area)) * bi * bk + (9 / 10) * zx, (-27 / 10) * zx,
            (1.0 / (4.0 * area)) * bj * bi + (9 / 10) * zx, (1.0 / (4.0 * area)) * bj * bj + (9 / 10) * zx, (1.0 / (4.0 * area)) * bj * bk + (9 / 10) * zx, (-27 / 10) * zx,
            (1.0 / (4.0 * area)) * bk * bi + (9 / 10) * zx, (1.0 / (4.0 * area)) * bk * bj + (9 / 10) * zx, (1.0 / (4.0 * area)) * bk * bk + (9 / 10) * zx, (-27 / 10) * zx,
            (-27 / 10) * zx, (-27 / 10) * zx, (-27 / 10) * zx, (81 / 10) * zx], dim=1)

        ky_elem_values = torch.stack([
            (1.0 / (4.0 * area)) * ci * ci + (9 / 10) * zy, (1.0 / (4.0 * area)) * ci * cj + (9 / 10) * zy, (1.0 / (4.0 * area)) * ci * ck + (9 / 10) * zy, (-27 / 10) * zy,
            (1.0 / (4.0 * area)) * cj * ci + (9 / 10) * zy, (1.0 / (4.0 * area)) * cj * cj + (9 / 10) * zy, (1.0 / (4.0 * area)) * cj * ck + (9 / 10) * zy, (-27 / 10) * zy,
            (1.0 / (4.0 * area)) * ck * ci + (9 / 10) * zy, (1.0 / (4.0 * area)) * ck * cj + (9 / 10) * zy, (1.0 / (4.0 * area)) * ck * ck + (9 / 10) * zy, (-27 / 10) * zy,
            (-27 / 10) * zy, (-27 / 10) * zy, (-27 / 10) * zy, (81 / 10) * zy], dim=1)

        gx_elem_values = torch.stack([
            (1.0 / 6.0) * ((9 / 20) * bi + bi), (1.0 / 6.0) * ((9 / 20) * bj + bi), (1.0 / 6.0) * ((9 / 20) * bk + bi),
            (1.0 / 6.0) * ((9 / 20) * bi + bj), (1.0 / 6.0) * ((9 / 20) * bj + bj), (1.0 / 6.0) * ((9 / 20) * bk + bj),
            (1.0 / 6.0) * ((9 / 20) * bi + bk), (1.0 / 6.0) * ((9 / 20) * bj + bk), (1.0 / 6.0) * ((9 / 20) * bk + bk),
            -(9 / 40) * bi, -(9 / 40) * bj, -(9 / 40) * bk], dim=1)

        gy_elem_values = torch.stack([
            (1.0 / 6.0) * ((9 / 20) * ci + ci), (1.0 / 6.0) * ((9 / 20) * cj + ci), (1.0 / 6.0) * ((9 / 20) * ck + ci),
            (1.0 / 6.0) * ((9 / 20) * ci + cj), (1.0 / 6.0) * ((9 / 20) * cj + cj), (1.0 / 6.0) * ((9 / 20) * ck + cj),
            (1.0 / 6.0) * ((9 / 20) * ci + ck), (1.0 / 6.0) * ((9 / 20) * cj + ck), (1.0 / 6.0) * ((9 / 20) * ck + ck),
            -(9 / 40) * ci, -(9 / 40) * cj, -(9 / 40) * ck], dim=1)



        m_elem_values = (area / 840.0).unsqueeze(1) * torch.tensor(
            [83, 13, 13, 45,
             13, 83, 13, 45,
             13, 13, 83, 45,
             45, 45, 45, 243],
            dtype=area.dtype, device=area.device).unsqueeze(0)

        K_values = (kx_elem_values + ky_elem_values).flatten()

        return K_values, m_elem_values.flatten(), gx_elem_values.flatten(), gy_elem_values.flatten()

    def adiciona_nos(self, IEN, X, Y, IENbound):
        # a bolha e' sempre interior: o contorno nao ganha no
        IEN_novo, X_novo, Y_novo = elem_mini(IEN, X, Y)
        return IEN_novo, X_novo, Y_novo, IENbound

    def funcoes_forma(self, li, lj, lk):
        """`N_i = λ_i - 9 λ1 λ2 λ3` (vertices) e `N_b = 27 λ1 λ2 λ3` (centroide).

        Base nodal: o DOF do centroide e' a velocidade fisica ali. Verificado
        batendo a matriz de rigidez, entrada a entrada, contra
        `tests/test_assembly.py::test_assemble_mini_triangulo_referencia` --
        a base "P1 + bolha crua", sem a correcao `9 λ1 λ2 λ3`, NAO bate.
        """
        corr = 9.0 * li * lj * lk
        return (li - corr), (lj - corr), (lk - corr), (27.0 * li * lj * lk)

    def valores_na_aresta(self, IEN, e, f, ib1, ib2, w1, w2, v):
        # sobre a aresta a bolha se anula: o campo MINI la' e' P1 (exato)
        return w1 * v[ib1] + w2 * v[ib2]


class Tri6(ElementoTriangular):
    """Elemento Tri6 (Taylor-Hood): velocidade P2 (3 vertices + 3 nos de aresta), pressao P1.

    Convencao de nos de aresta (Zienkiewicz vol. 1, cap. 8): coluna 3 =
    aresta (v1,v2), 4 = (v2,v3), 5 = (v3,v1). A variante "slip" de Gx da
    referencia (`gxslipele`) NAO e' usada: ela quebra a simetria C = B^T e o
    BiCGSTAB deixa de convergir (medido em 2026-08-15, ver CLAUDE.local.md).
    """

    nome = "tri6"
    n_nos = 6

    def valores_locais(self, geo: Geometria):
        area, bi, bj, bk, ci, cj, ck = geo

        m_elem_values = (area / 180.0).unsqueeze(1)*torch.tensor([ 6.0,-1.0,-1.0, 0.0,-4.0, 0.0,
                                                                  -1.0, 6.0,-1.0, 0.0, 0.0,-4.0,
                                                                  -1.0,-1.0, 6.0,-4.0, 0.0, 0.0,
                                                                   0.0, 0.0,-4.0,32.0,16.0,16.0,
                                                                  -4.0, 0.0, 0.0,16.0,32.0,16.0,
                                                                   0.0,-4.0, 0.0,16.0,16.0,32.0],
                                                                dtype=area.dtype, device=area.device).unsqueeze(0)

        m_area = (1.0 / (12.0 * area))

        kx_elem_values = torch.stack([
            m_area * 3.0 * bi * bi, m_area * -bi * bj, m_area * -bi * bk, m_area * 4 * bi * bj, torch.zeros_like(bi), m_area * 4 * bi * bk,
            m_area * - bi * bj, m_area * 3 * bj * bj, m_area * - bj * bk, m_area * 4 * bi * bj, m_area * 4 * bj * bk, torch.zeros_like(bi),
            m_area * - bi * bk, m_area * - bj * bk, m_area * 3 * bk * bk, torch.zeros_like(bi), m_area * 4 * bj * bk, m_area * 4 * bi * bk,
            m_area * 4 * bi * bj, m_area * 4 * bi * bj, torch.zeros_like(bi), m_area * 8 * (bi * bi + bi * bj + bj *bj), 
                                                             m_area * 4 * (bj * bj + bj * bk + bi * bj + 2 * bi * bk),
                                                             m_area * 4 * (bi * bi + bi * bk + bi * bj + 2 * bj * bk),
            torch.zeros_like(bi), m_area * 4 * bj * bk, m_area * 4 * bj * bk, m_area * 4 * (bi * bj + 2 * bi * bk + bj * bj + bj * bk),
                                                             m_area * 8 * (bj * bj + bj * bk + bk * bk),
                                                             m_area * 4 * (2 * bi * bj + bi * bk + bj * bk + bk * bk),
            m_area * 4 * bi * bk, torch.zeros_like(bi), m_area * 4 * bi * bk, m_area * 4 * (bi * bi + bi * bj + bi * bk + 2 * bj * bk),
                                                             m_area * 4 * (2 * bi * bj + bi * bk + bj * bk + bk * bk),
                                                             m_area * 8 * (bi * bi + bi * bk + bk * bk)], dim=1)

        ky_elem_values = torch.stack([
            m_area * 3 * ci * ci, m_area * -ci * cj, m_area * -ci * ck, m_area * 4 * ci * cj, torch.zeros_like(bi), m_area * 4 * ci * ck,
            m_area * -ci * cj, m_area * 3 * cj * cj, m_area * -cj * ck, m_area * 4 * ci * cj, m_area * 4 * cj * ck, torch.zeros_like(bi),
            m_area * -ci * ck, m_area * -cj * ck, m_area * 3 * ck * ck, torch.zeros_like(bi), m_area * 4 * cj * ck, m_area * 4 * ci * ck,
            m_area * 4 * ci * cj, m_area * 4 * ci * cj, torch.zeros_like(bi), m_area * 8 * (ci * ci + ci * cj + cj *cj),
                                                             m_area * 4 * (cj * cj + cj * ck + ci * cj + 2 * ci * ck),
                                                             m_area * 4 * (ci * ci + ci * ck + ci * cj + 2 * cj * ck),
            torch.zeros_like(bi), m_area * 4 * cj * ck, m_area * 4 * cj * ck, m_area * 4 * (ci * cj + 2 * ci * ck + cj * cj + cj * ck),
                                                             m_area * 8 * (cj * cj + cj * ck + ck * ck),
                                                             m_area * 4 * (2 * ci * cj + ci * ck + cj * ck + ck * ck),
            m_area * 4 * ci * ck, torch.zeros_like(bi), m_area * 4 * ci * ck, m_area * 4 * (ci * ci + ci * cj + ci * ck + 2 * cj * ck),
                                                             m_area * 4 * (2 * ci * cj + ci * ck + cj * ck + ck * ck),
                                                             m_area * 8 * (ci * ci + ci * ck + ck * ck)], dim=1)

        # Gradiente P2 completo, INT(phi_j dN_i/dx) (`gxele`/`gyele` da referencia).
        zb, zc = torch.zeros_like(bi), torch.zeros_like(ci)
        gx_elem_values = torch.stack([
            (1.0 / 6.0) * bi,             zb,                           zb,
            zb,                           (1.0 / 6.0) * bj,             zb,
            zb,                           zb,                           (1.0 / 6.0) * bk,
            (1.0 / 6.0) * (bi + 2 * bj),  (1.0 / 6.0) * (2 * bi + bj),  (1.0 / 6.0) * (bi + bj),
            (1.0 / 6.0) * (bj + bk),      (1.0 / 6.0) * (bj + 2 * bk),  (1.0 / 6.0) * (2 * bj + bk),
            (1.0 / 6.0) * (bi + 2 * bk),  (1.0 / 6.0) * (bi + bk),      (1.0 / 6.0) * (2 * bi + bk)], dim=1)

        gy_elem_values = torch.stack([
            (1.0 / 6.0) * ci,             zc,                           zc,
            zc,                           (1.0 / 6.0) * cj,             zc,
            zc,                           zc,                           (1.0 / 6.0) * ck,
            (1.0 / 6.0) * (ci + 2 * cj),  (1.0 / 6.0) * (2 * ci + cj),  (1.0 / 6.0) * (ci + cj),
            (1.0 / 6.0) * (cj + ck),      (1.0 / 6.0) * (cj + 2 * ck),  (1.0 / 6.0) * (2 * cj + ck),
            (1.0 / 6.0) * (ci + 2 * ck),  (1.0 / 6.0) * (ci + ck),      (1.0 / 6.0) * (2 * ci + ck)], dim=1)



        K_values = (kx_elem_values + ky_elem_values).flatten()

        return K_values, m_elem_values.flatten(), gx_elem_values.flatten(), gy_elem_values.flatten()

    def adiciona_nos(self, IEN, X, Y, IENbound):
        # o no de aresta tambem existe no contorno: IENbound ganha a coluna do meio
        IEN_novo, X_novo, Y_novo, arestas = elem_tri6(IEN, X, Y)
        return IEN_novo, X_novo, Y_novo, estende_IENbound_tri6(IENbound, arestas, len(X))

    def funcoes_forma(self, li, lj, lk):
        """P2 padrao (Zienkiewicz vol. 1, cap. 8): `N_i = λ_i (2 λ_i - 1)` nos
        vertices e `N_ij = 4 λ_i λ_j` nos nos de aresta (colunas 3, 4, 5 =
        arestas (i,j), (j,k), (k,i))."""
        Ni, Nj, Nk = li * (2.0 * li - 1.0), lj * (2.0 * lj - 1.0), lk * (2.0 * lk - 1.0)
        Nij, Njk, Nki = 4.0 * li * lj, 4.0 * lj * lk, 4.0 * lk * li
        return Ni, Nj, Nk, Nij, Njk, Nki

    def valores_na_aresta(self, IEN, e, f, ib1, ib2, w1, w2, v):
        # sobre a aresta so 3 funcoes de forma nao se anulam: os 2 vertices e o
        # no de aresta -- os 3 polinomios de Lagrange quadraticos em s (exato)
        ibm = IEN[e, 3 + f]
        s = w2  # w1*B1 + w2*B2 == (1-s)*B1 + s*B2
        peso1, peso2, pesom = (1.0 - s) * (1.0 - 2.0 * s), s * (2.0 * s - 1.0), 4.0 * s * (1.0 - s)
        return peso1 * v[ib1] + peso2 * v[ib2] + pesom * v[ibm]


MINI = Mini()
TRI6 = Tri6()

_POR_NOME = {MINI.nome: MINI, TRI6.nome: TRI6}


def elemento_por_nome(nome: str) -> ElementoTriangular:
    """Instancia do elemento pelo nome do config (`simulation.element`)."""
    try:
        return _POR_NOME[nome]
    except KeyError:
        raise ValueError(f"simulation.element desconhecido: {nome!r} (use {' ou '.join(map(repr, _POR_NOME))})") from None
