"""Pre-condicionadores do sistema de ponto de sela de cada passo de tempo.

O sistema (`reassembly.Sistema`) e' `[F B; B^T 0]`, com `F = M/dt + K/Re`
(um bloco por componente de velocidade) e `B = -beta G`. Como a adveccao e'
semi-Lagrangeana e fica toda no lado direito, a matriz nao tem termo
convectivo em nenhum Reynolds: e' um **Stokes generalizado**. O complemento de
Schur `S = B^T F^-1 B` e' entao bem aproximado pela formula de
Cahouet-Chabard,

    S^-1 ~ beta^-2 ( (1/Re) Mp^-1 + (1/dt) Lp^-1 ),

com `Mp` e `Lp` a massa e o laplaciano P1 da pressao (J. Cahouet,
J.-P. Chabard (1988), Int. J. Numer. Methods Fluids 8(8):869-895;
K.-A. Mardal, R. Winther (2004), Numer. Math. 98(2):305-327 -- robusto em
`h`, `dt` e `1/Re`).

Opcoes (`simulation.solver.precondicionador`):

- `nenhum` -- identidade.
- `jacobi` -- diagonal de `A` (`krylov.jacobi_preconditioner`).
- `bloco_triangular` -- `P = [F^ B; 0 -S^]`, para o BiCGSTAB (plano 12/10,
  opcao B2). Com blocos exatos o operador pre-condicionado tem 2 autovalores
  distintos (I. C. F. Ipsen (2001), SIAM J. Sci. Comput. 23(3):1050-1051).
- `bloco_diagonal` -- `P = diag(F^, S^)`, simetrico e definido positivo, para
  o MINRES (opcao M1). Com blocos exatos, 3 autovalores (M. F. Murphy,
  G. H. Golub, A. J. Wathen (2000), SIAM J. Sci. Comput. 21(6):1969-1972;
  D. Silvester, A. Wathen (1994), SIAM J. Numer. Anal. 31(5):1352-1367).

`F^-1`, `Mp^-1` e `Lp^-1` (os sub-solvers, `simulation.solver.subsolver`):

- `chebyshev` -- Chebyshev com Jacobi com numero fixo de passos
  (`krylov.chebyshev_jacobi`), na GPU. Linear, fixo e simetrico, como o
  BiCGSTAB e o MINRES exigem. Fraco para `F` dominado por difusao e para
  `Lp` (laplaciano): medido em 12/10 no Poiseuille, 80 passos ainda deixam
  ~90% do erro em `Lp^-1`.
- `fsai` -- Chebyshev pre-condicionado por uma inversa aproximada esparsa
  fatorada (`krylov.fsai`, Kolotilina & Yeremin 1993) em `F` e `Lp`, inteiro
  na GPU (a FSAI e' construida resolvendo, em lote, um sistema denso pequeno
  por linha). `Mp` fica com Chebyshev-Jacobi. Passos em `chebyshev:`.
- `lu` -- fatoracao LU esparsa exata de cada bloco (`scipy.sparse.linalg.splu`,
  na CPU), refeita a cada montagem. Os vetores vao e voltam da GPU a cada
  aplicacao (tamanho de um bloco, barato). Exato, portanto tambem linear e
  fixo. Nao escala para malhas muito grandes (o preenchimento da fatoracao
  cresce), mas nas malhas do projeto as 4 fatoracoes custam dezenas de ms.

Os graus de liberdade com Dirichlet (linhas e colunas identidade em `A`)
passam direto: `z = r` neles.
"""

from dataclasses import dataclass

import torch

from femns.krylov import _csr_diagonal, chebyshev_fsai, chebyshev_jacobi, jacobi_preconditioner, limites_jacobi

NOMES = ("nenhum", "jacobi", "bloco_triangular", "bloco_diagonal")
SUBSOLVERS = ("chebyshev", "fsai", "lu")
DE_BLOCO = ("bloco_triangular", "bloco_diagonal")

# Limites exatos do espectro de D^-1 Mp para P1 em triangulos (A. J. Wathen
# (1987), IMA J. Numer. Anal. 7(4):449-457): [1/2, 2]. Valem tambem depois da
# eliminacao das CCs (submatriz principal; a linha identidade da autovalor 1).
_LIMITES_MASSA_P1 = (0.5, 2.0)


@dataclass(frozen=True)
class PassosChebyshev:
    """Passos de Chebyshev de cada bloco aproximado (`simulation.solver.chebyshev`)."""

    velocidade: int = 40
    massa: int = 5
    laplaciano: int = 80


def _mv(A, v):
    return torch.mm(A, v.unsqueeze(1)).squeeze(1)


def _lu(A_csr):
    """`r -> A^-1 r` exato por LU esparsa (scipy, CPU); o vetor volta no device/dtype de `r`."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu

    A = A_csr.to_sparse_csr()
    fator = splu(sp.csr_matrix((A.values().cpu().numpy(), A.col_indices().cpu().numpy(),
                                A.crow_indices().cpu().numpy()), shape=A.shape).tocsc())

    def aplicar(r):
        return torch.from_numpy(fator.solve(r.cpu().numpy())).to(device=r.device, dtype=r.dtype)
    return aplicar


class _Blocos:
    """As aproximacoes `F^-1` (por componente) e `S^-1` de um `Sistema`."""

    def __init__(self, sistema, dt: float, Re: float, passos: PassosChebyshev, subsolver: str = "chebyshev"):
        self.nn = sistema.Fx.shape[0]
        self.np = sistema.Mp.shape[0]
        self.eh_cc = sistema.eh_cc
        self.eh_cc_p = sistema.eh_cc[2 * self.nn:]

        if subsolver == "lu":
            self.Fx_inv, self.Fy_inv = _lu(sistema.Fx), _lu(sistema.Fy)
            self.Mp_inv, self.Lp_inv = _lu(sistema.Mp), _lu(sistema.Lp)
        elif subsolver == "fsai":
            self.Fx_inv = chebyshev_fsai(sistema.Fx, passos.velocidade)
            self.Fy_inv = chebyshev_fsai(sistema.Fy, passos.velocidade)
            dmp = _csr_diagonal(sistema.Mp)
            self.Mp_inv = chebyshev_jacobi(sistema.Mp, dmp, *_LIMITES_MASSA_P1, passos.massa)
            self.Lp_inv = chebyshev_fsai(sistema.Lp, passos.laplaciano)
        elif subsolver == "chebyshev":
            dx = _csr_diagonal(sistema.Fx)
            dy = _csr_diagonal(sistema.Fy)
            lmin_f, lmax_f = limites_jacobi(sistema.Fx, dx)  # Fx e Fy so diferem nas CCs: mesmos limites
            self.Fx_inv = chebyshev_jacobi(sistema.Fx, dx, lmin_f, lmax_f, passos.velocidade)
            self.Fy_inv = chebyshev_jacobi(sistema.Fy, dy, lmin_f, lmax_f, passos.velocidade)

            dmp = _csr_diagonal(sistema.Mp)
            self.Mp_inv = chebyshev_jacobi(sistema.Mp, dmp, *_LIMITES_MASSA_P1, passos.massa)
            dlp = _csr_diagonal(sistema.Lp)
            self.Lp_inv = chebyshev_jacobi(sistema.Lp, dlp, *limites_jacobi(sistema.Lp, dlp), passos.laplaciano)
        else:
            raise ValueError(f"subsolver desconhecido: {subsolver!r} (use {' ou '.join(SUBSOLVERS)})")

        if sistema.lp_zero is not None and bool(sistema.lp_zero.any()):
            # Dirichlet homogeneo do Lp na saida livre: entrada e saida zeradas
            # nesses nos (o operador continua simetrico e semidefinido; somado
            # ao termo de massa, definido positivo).
            mantem = (~sistema.lp_zero).to(sistema.Mp.dtype)
            lp_inv = self.Lp_inv
            self.Lp_inv = lambda r: mantem * lp_inv(mantem * r)

        self.coef_massa = 1.0 / (Re * sistema.beta ** 2)
        self.coef_lapl = 1.0 / (dt * sistema.beta ** 2)

    def S_inv(self, rp):
        """`S^-1 rp` de Cahouet-Chabard nos graus de pressao livres; identidade nos prescritos."""
        zp = self.coef_massa * self.Mp_inv(rp) + self.coef_lapl * self.Lp_inv(rp)
        return torch.where(self.eh_cc_p, rp, zp)

    def F_inv(self, ru):
        return torch.cat([self.Fx_inv(ru[:self.nn]), self.Fy_inv(ru[self.nn:])])


def montar(nome: str, sistema, dt: float, Re: float, passos: PassosChebyshev = PassosChebyshev(),
           subsolver: str = "chebyshev"):
    """Pre-condicionador `r -> z ~ A^-1 r` para o `Sistema`, ou `None` (`nenhum`).

    Os de bloco exigem um `Sistema` com `Fx`, `Fy`, `Mp`, `Lp` (Remontador com
    `precisa_blocos=True`). Monta de novo a cada montagem do sistema (malha
    movel): os limites espectrais do Chebyshev dependem da geometria.
    """
    if nome == "nenhum":
        return None
    if nome == "jacobi":
        return jacobi_preconditioner(sistema.A)
    if nome not in DE_BLOCO:
        raise ValueError(f"pre-condicionador desconhecido: {nome!r} (use {', '.join(NOMES)})")
    if sistema.Fx is None:
        raise ValueError(f"pre-condicionador {nome!r} precisa dos blocos do sistema (Remontador com precisa_blocos=True)")

    blocos = _Blocos(sistema, dt, Re, passos, subsolver)
    nn2 = 2 * blocos.nn
    eh_cc = sistema.eh_cc
    A = sistema.A

    if nome == "bloco_diagonal":
        def aplicar(r):
            z = torch.cat([blocos.F_inv(r[:nn2]), blocos.S_inv(r[nn2:])])
            return torch.where(eh_cc, r, z)
        return aplicar

    # bloco_triangular: P = [F^ B; 0 -S^]  ->  z_p = -S^-1 r_p ;  z_u = F^-1 (r_u - B z_p)
    def aplicar(r):
        zp = -blocos.S_inv(r[nn2:])
        zp = torch.where(blocos.eh_cc_p, r[nn2:], zp)
        Bz = _mv(A, torch.cat([torch.zeros(nn2, dtype=r.dtype, device=r.device), zp]))[:nn2]
        z = torch.cat([blocos.F_inv(r[:nn2] - Bz), zp])
        return torch.where(eh_cc, r, z)
    return aplicar
