"""Adveccao semi-Lagrangeana por matriz de interpolacao esparsa (metodologia de CLAUDE_sl.md).

Alternativa a `femns.semi_lagrangian` (busca+interpolacao ponto a ponto em
numpy): aqui o operador de interpolacao no pe da caracteristica e
materializado como matriz esparsa `conv` (`self.conv`, torch), aplicada a
qualquer campo por `phi_d = conv @ phi` -- a mesma matriz e reaproveitada
entre `vx` e `vy` num mesmo passo, amortizando a busca por elemento.

Hierarquia de classes (ver CLAUDE_sl.md, secao "Estrutura do codigo"):
`SemiLagrangianTri` concentra todo o algoritmo (precomputo geometrico,
busca por caminhada vetorizada, interceptacao de contorno, montagem de
`conv`); cada subclasse concreta so declara a geometria do seu elemento
(`NEN`, `EDGE_LOCAL`, `_shape`, `_edge_shape`).

Tratamento de contorno: quando o pe da caracteristica sai do dominio, o
trajeto (posicao atual do no -> ponto advectado) e interceptado com a
aresta de saida e a linha resultante e interpolada entre os 2 nos de canto
dessa aresta -- **diferente** do fallback de Dirichlet usado em
`femns.semi_lagrangian`. Como `conv` opera sobre os valores *atuais* de
`vx, vy` (ja com Dirichlet aplicado pelo solver a cada passo), essa
interceptacao recupera o valor de contorno correto sem precisar de
`ccName`/`conditions` aqui.
"""

from abc import ABC, abstractmethod

import torch

from femns.semi_lagrangian import backtrace


class SemiLagrangianTri(ABC):
    """Base do metodo: precomputo geometrico + busca vetorizada + montagem de `conv`.

    `IEN` e as coordenadas `X, Y` sao os arrays completos pos-`mesh.elem_mini`
    (4a coluna = no de centroide/bolha); `EToE` vem de `mesh.montar_EToE`
    sobre a malha original (so vertices, 3 faces por elemento).
    `node_to_elem` (`mesh.montar_node_to_elem`) e o chute inicial de busca
    para os nos de vertice; nos de centroide usam a si mesmos como chute
    (mesmo padrao de `femns.semi_lagrangian.calculo_sl`).

    Espera tudo em `torch`, ja no device de destino -- ao contrario do
    modulo numpy, aqui nao ha round-trip CPU<->GPU por passo (CLAUDE_sl.md:
    "com device=cuda a busca inteira roda na GPU sem transferencias").
    """

    NEN: int
    """Numero de nos por elemento (colunas de IEN usadas no caso interior)."""

    EDGE_LOCAL: tuple
    """Por face f (0, 1, 2 -- convencao de `mesh.montar_EToE`), os indices
    locais dos nos de canto dessa aresta, na ordem `(no em s=1, no em s=0)`
    -- a interceptacao devolve o peso do no em s=1 diretamente como `s`.
    So os 2 primeiros nos de cada face sao usados na interceptacao (nos
    internos de aresta, se um elemento de ordem maior precisar, entram via
    `_edge_shape` mas nao mudam o calculo geometrico de `s`)."""

    def __init__(self, IEN: torch.Tensor, EToE: torch.Tensor, X: torch.Tensor, Y: torch.Tensor,
                 node_to_elem: torch.Tensor, npoints: int, ne: int):
        device = X.device
        self.IEN = torch.as_tensor(IEN, dtype=torch.long, device=device)
        self.EToE = torch.as_tensor(EToE, dtype=torch.long, device=device)
        self.X = X
        self.Y = Y
        self.npoints = npoints
        self.ne = ne
        self.conv = None  # montada em compute()
        self.status = None

        self._edge_local = torch.tensor(self.EDGE_LOCAL, dtype=torch.long, device=device)

        node_to_elem_t = torch.as_tensor(node_to_elem, dtype=torch.long, device=device)
        centroide_start = torch.arange(ne, dtype=torch.long, device=device)
        self._elem_start = torch.cat([node_to_elem_t, centroide_start])

        self._precompute_barycentric_coeffs()

    def _precompute_barycentric_coeffs(self):
        """Coeficientes afins `a_k, b_k, c_k, det` por elemento: `l_k(x,y) = (a_k + b_k*x + c_k*y) / det`.

        `b_k, c_k` sao os mesmos de `assembly.assemble_mini` (mesma
        convencao de sinal); `a_k` (termo constante) nao e usado la mas e
        precomputado aqui para avaliar `l_k` num ponto qualquer sem
        recalcular area a cada consulta -- CLAUDE_sl.md proibe formula de
        area (shoelace) inline por chamada, so precomputo por malha.
        IEN[e, 0:3] sempre em ordem anti-horaria (convencao do projeto),
        entao `det` (= 2*area) sai positivo direto, sem `abs`.
        """
        vi, vj, vk = self.IEN[:, 0], self.IEN[:, 1], self.IEN[:, 2]
        xi, yi = self.X[vi], self.Y[vi]
        xj, yj = self.X[vj], self.Y[vj]
        xk, yk = self.X[vk], self.Y[vk]

        bi, bj, bk = yj - yk, yk - yi, yi - yj
        ci, cj, ck = xk - xj, xi - xk, xj - xi
        ai = xj * yk - xk * yj
        aj = xk * yi - xi * yk
        ak = xi * yj - xj * yi
        det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)

        self._a = torch.stack([ai, aj, ak], dim=1)
        self._b = torch.stack([bi, bj, bk], dim=1)
        self._c = torch.stack([ci, cj, ck], dim=1)
        self._det = det

    def _barycentric(self, xd: torch.Tensor, yd: torch.Tensor, elem: torch.Tensor):
        """Avalia l1, l2, l3 de (xd, yd) em `elem` via os coeficientes afins pre-computados."""
        a, b, c = self._a[elem], self._b[elem], self._c[elem]
        det = self._det[elem]
        lam = (a + b * xd.unsqueeze(1) + c * yd.unsqueeze(1)) / det.unsqueeze(1)
        return lam[:, 0], lam[:, 1], lam[:, 2]

    def _boundary_intercept(self, col_buf: torch.Tensor, val_buf: torch.Tensor, idx: torch.Tensor,
                             elem: torch.Tensor, face: torch.Tensor, xd: torch.Tensor, yd: torch.Tensor,
                             det_eps: float = 1e-12) -> int:
        """Interceptacao do trajeto (posicao atual do no -> pe da caracteristica) com a aresta de saida.

        Formula equivalente a `computeIntercept` do codigo de referencia do
        professor: resolve a intersecao entre o segmento (R1=no, R2=pe da
        caracteristica) e a reta que liga os 2 nos de canto da aresta
        (B1=no em s=1, B2=no em s=0); `s = det_x/det` e o peso de B1.
        Trajetoria paralela a aresta (`det ~= 0`) e o caso degenerado citado
        em CLAUDE_sl.md ("Armadilhas conhecidas") -- cai em linha identidade,
        igual ao no que nao converge em `max_iter`. `s` e sempre limitado a
        [0,1] (peso negativo tambem viraria bug silencioso).

        Escreve direto nas colunas 0/1 de `col_buf`/`val_buf` (linha
        identidade so usa a coluna 0) -- ver `_locate_and_assemble` sobre por
        que isso dispensa `coalesce()`. Devolve o numero de nos resolvidos
        por interceptacao (nao degenerados) em `idx`.
        """
        local = self._edge_local[face]
        ib1 = self.IEN[elem, local[:, 0]]
        ib2 = self.IEN[elem, local[:, 1]]

        R1X, R1Y = self.X[idx], self.Y[idx]
        R2X, R2Y = xd, yd
        B1X, B1Y = self.X[ib1], self.Y[ib1]
        B2X, B2Y = self.X[ib2], self.Y[ib2]

        a1, b1, c1 = B1X - B2X, R1X - R2X, R1X - B2X
        a2, b2, c2 = B1Y - B2Y, R1Y - R2Y, R1Y - B2Y
        det = a1 * b2 - a2 * b1
        detx = c1 * b2 - c2 * b1

        degenerado = torch.abs(det) < det_eps
        det_seguro = torch.where(degenerado, torch.ones_like(det), det)
        s = torch.clamp(detx / det_seguro, 0.0, 1.0)

        idx_ok = idx[~degenerado]
        if idx_ok.numel() > 0:
            w1, w2 = self._edge_shape(s[~degenerado])
            col_buf[idx_ok, 0] = ib1[~degenerado]
            col_buf[idx_ok, 1] = ib2[~degenerado]
            val_buf[idx_ok, 0] = w1
            val_buf[idx_ok, 1] = w2

        idx_bad = idx[degenerado]
        if idx_bad.numel() > 0:
            col_buf[idx_bad, 0] = idx_bad
            val_buf[idx_bad, 0] = 1.0

        return int(idx_ok.numel())

    def _locate_and_assemble(self, xd: torch.Tensor, yd: torch.Tensor, elem_start: torch.Tensor,
                              tol: float, max_iter: int):
        """Busca por caminhada vetorizada ("frente de onda") que monta `conv` (CSR) direto.

        Mesma convencao de `femns.semi_lagrangian.localizar_pontos_partida`
        (vertice de baricentrica mais negativa `m` escapa pela face
        `(m+1) % 3` -- nunca `EToE[e, m]` direto, essa e a convencao
        incompativel do codigo do professor). Cada iteracao do laco externo
        da um passo de caminhada simultaneo em todos os nos ainda ativos via
        mascara booleana; o laco roda no maximo `max_iter` vezes, nao
        `len(xd)`.

        Em vez de acumular triplets `(row,col,val)` em ordem de descoberta e
        ordenar/deduplicar no final (`coalesce()`, ~37% do custo por passo
        medido no perfil -- trabalho puro desperdicado, ja que nunca ha
        `(row,col)` duplicado por construcao: cada linha e escrita uma unica
        vez, com colunas sempre distintas dentro dela), escreve direto num
        buffer denso `(n, NEN)` -- cada linha grava na sua propria posicao
        (scatter). `-1` marca coluna nao usada (fronteira usa so 2 das
        `NEN` colunas, identidade so 1). O "flatten" mascarado desse buffer
        no final ja sai ordenado por linha de graca, permitindo montar o
        CSR (`crow_indices` a partir da contagem por linha) sem nenhum sort.
        """
        device = self.X.device
        n = xd.shape[0]
        elem = elem_start.clone()
        done = torch.zeros(n, dtype=torch.bool, device=device)

        col_buf = torch.full((n, self.NEN), -1, dtype=torch.long, device=device)
        val_buf = torch.zeros((n, self.NEN), dtype=self.X.dtype, device=device)

        status = {"dentro": 0, "fronteira": 0, "nao_convergido": 0}

        for _ in range(max_iter):
            ativos = torch.where(~done)[0]
            if ativos.numel() == 0:
                break

            l1, l2, l3 = self._barycentric(xd[ativos], yd[ativos], elem[ativos])
            lam = torch.stack([l1, l2, l3], dim=1)
            dentro = lam.min(dim=1).values >= -tol

            idx_dentro = ativos[dentro]
            if idx_dentro.numel() > 0:
                l1d, l2d, l3d = self._clamp_barycentric(l1[dentro], l2[dentro], l3[dentro])
                col_buf[idx_dentro] = self.IEN[elem[idx_dentro]][:, :self.NEN]
                val_buf[idx_dentro] = torch.stack(self._shape(l1d, l2d, l3d), dim=1)
                done[idx_dentro] = True
                status["dentro"] += int(idx_dentro.numel())

            idx_fora = ativos[~dentro]
            if idx_fora.numel() == 0:
                continue

            m = lam[~dentro].argmin(dim=1)
            face = (m + 1) % 3
            proximo = self.EToE[elem[idx_fora], face]

            saiu = proximo == -1
            idx_saiu = idx_fora[saiu]
            idx_anda = idx_fora[~saiu]

            if idx_saiu.numel() > 0:
                n_ok = self._boundary_intercept(
                    col_buf, val_buf, idx_saiu, elem[idx_saiu], face[saiu], xd[idx_saiu], yd[idx_saiu])
                done[idx_saiu] = True
                status["fronteira"] += n_ok
                status["nao_convergido"] += int(idx_saiu.numel()) - n_ok

            if idx_anda.numel() > 0:
                elem[idx_anda] = proximo[~saiu]

        idx_resto = torch.where(~done)[0]
        if idx_resto.numel() > 0:
            col_buf[idx_resto, 0] = idx_resto
            val_buf[idx_resto, 0] = 1.0
            status["nao_convergido"] += int(idx_resto.numel())

        mask = col_buf != -1
        crow_indices = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            mask.sum(dim=1).cumsum(0),
        ])
        col_indices = col_buf[mask]
        values = val_buf[mask]
        return crow_indices, col_indices, values, status

    @staticmethod
    def _clamp_barycentric(l1: torch.Tensor, l2: torch.Tensor, l3: torch.Tensor):
        """Recorta para [0,1] e renormaliza -- absorve negativos residuais da tolerancia de busca.

        Mesmo tratamento do `interpolar_mini` de `femns.semi_lagrangian`
        (nao e especifico de elemento, entao vive na base).
        """
        l1, l2, l3 = torch.clamp(l1, 0.0, 1.0), torch.clamp(l2, 0.0, 1.0), torch.clamp(l3, 0.0, 1.0)
        soma = l1 + l2 + l3
        return l1 / soma, l2 / soma, l3 / soma

    def compute(self, vx: torch.Tensor, vy: torch.Tensor, dt: float, tol: float = 1e-9, max_iter: int = 50):
        """Avanca um passo semi-Lagrangeano: backtrace + monta `conv` + aplica a vx e vy.

        `conv` e reconstruida a cada chamada (depende da velocidade via o
        backtrace) e guardada em `self.conv`/`self.status` para diagnostico
        -- mas e montada so uma vez por passo e reaproveitada nas duas
        aplicacoes (`vx`, `vy`), como pede CLAUDE_sl.md.
        """
        xd, yd = backtrace(self.X, self.Y, vx, vy, dt)
        crow_indices, col_indices, values, status = self._locate_and_assemble(xd, yd, self._elem_start, tol, max_iter)

        n = self.npoints + self.ne
        # CSR direto a partir do buffer ja ordenado por linha (ver
        # _locate_and_assemble) -- sem passar por COO/coalesce().
        self.conv = torch.sparse_csr_tensor(
            crow_indices, col_indices, values, size=(n, n), dtype=self.X.dtype, device=self.X.device)
        self.status = status

        # torch.mm, mesma convencao de solver.py/krylov.py.
        vx_star = torch.mm(self.conv, vx.unsqueeze(1)).squeeze(1)
        vy_star = torch.mm(self.conv, vy.unsqueeze(1)).squeeze(1)
        return vx_star, vy_star

    @abstractmethod
    def _shape(self, l1: torch.Tensor, l2: torch.Tensor, l3: torch.Tensor) -> tuple:
        """Funcoes de forma no interior do elemento, a partir das baricentricas. Devolve `NEN` tensores."""

    @abstractmethod
    def _edge_shape(self, s: torch.Tensor) -> tuple:
        """Funcoes de forma de linha na fronteira, a partir de `s` (peso do no em s=1). Devolve 2 tensores."""


class SemiLagrangianMini(SemiLagrangianTri):
    """Elemento MINI (P1 + bolha, base nodal) -- o unico elemento montado pelo projeto hoje.

    Mesma base de `interpolar_mini` em `femns.semi_lagrangian`
    (`Ni = li - 9*li*lj*lk`, `Nb = 27*l1*l2*l3`), so vetorizada em torch.
    """

    NEN = 4
    EDGE_LOCAL = ((0, 1), (1, 2), (0, 2))

    def _shape(self, l1: torch.Tensor, l2: torch.Tensor, l3: torch.Tensor) -> tuple:
        corr = 9.0 * l1 * l2 * l3
        return l1 - corr, l2 - corr, l3 - corr, 27.0 * l1 * l2 * l3

    def _edge_shape(self, s: torch.Tensor) -> tuple:
        return s, 1.0 - s
