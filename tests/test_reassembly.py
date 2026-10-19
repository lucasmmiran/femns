import numpy as np
import pytest
import torch

from femns.boundary import apply_boundary_conditions
from femns.elemento import elemento_por_nome
from femns.mesh import elem_mini, elem_tri6
from femns.reassembly import Remontador
from femns.solver import build_system_matrix, pressure_scale_factor


def malha_quadrada(nx):
    """Grade nx*nx de vertices em [0,1]^2, cada celula -> 2 triangulos."""
    xs = np.linspace(0.0, 1.0, nx)
    X, Y = np.meshgrid(xs, xs)
    X, Y = X.ravel(), Y.ravel()
    idx = np.arange(nx * nx).reshape(nx, nx)
    tris = []
    for j in range(nx - 1):
        for i in range(nx - 1):
            a, b, c, d = idx[j, i], idx[j, i + 1], idx[j + 1, i], idx[j + 1, i + 1]
            tris += [[a, b, d], [a, d, c]]
    return X.astype(np.float64), Y.astype(np.float64), np.array(tris, dtype=np.int64)


def _monta_caso(element, nx=4, dt=0.05, Re=800.0):
    X_np, Y_np, IEN_np = malha_quadrada(nx)
    npoints, ne = nx * nx, IEN_np.shape[0]
    if element == "tri6":
        IEN2, X2, Y2, _ = elem_tri6(IEN_np, X_np, Y_np)
        nnodes = X2.shape[0]
    else:
        IEN2, X2, Y2 = elem_mini(IEN_np, X_np, Y_np)
        nnodes = npoints + ne
    n_extra = nnodes - npoints
    dev = torch.device("cpu")
    IEN = torch.from_numpy(IEN2).to(dev)
    X, Y = torch.from_numpy(X2).to(dev), torch.from_numpy(Y2).to(dev)

    # borda esquerda (x=0) restringe vx/vy; borda direita (x=1) fixa p
    esq = torch.from_numpy(np.where(X2 < 1e-9)[0])
    dir_ = torch.from_numpy(np.where((X2 > 1 - 1e-9) & (np.arange(nnodes) < npoints))[0])
    return dict(dt=dt, Re=Re, X=X, Y=Y, IEN=IEN, element=elemento_por_nome(element), npoints=npoints,
                ne=ne, nnodes=nnodes, n_extra=n_extra, vxp=esq, vyp=esq, pp=dir_)


def _referencia(c, X, Y, beta):
    K, M, Gx, Gy = c["element"].montar(X, Y, c["IEN"], c["npoints"], c["nnodes"])
    A = build_system_matrix(c["dt"], c["Re"], K, M, Gx, Gy, beta)
    A = apply_boundary_conditions(A, c["vxp"], c["vyp"], c["pp"], c["npoints"], c["n_extra"])
    return A.to_sparse_csr(), M.coalesce().to_sparse_csr()


@pytest.mark.parametrize("element", ["mini", "tri6"])
def test_remontador_reproduz_montagem_de_referencia(element):
    c = _monta_caso(element)
    A_ref, _ = _referencia(c, c["X"], c["Y"], 1.0)
    beta = pressure_scale_factor(*_assemble_kmgg(c, c["X"], c["Y"]), c["dt"], c["Re"])
    R = Remontador(c["dt"], c["Re"], c["X"], c["Y"], c["IEN"], c["element"], c["npoints"],
                   c["n_extra"], c["vxp"], c["vyp"], c["pp"], beta)
    # a verificacao interna ja roda em __init__; confirma a tolerancia
    assert R.erro_verificacao is not None and R.erro_verificacao < 1e-12

    # padrao CSR identico ao da montagem de referencia
    A_ref_b, _ = _referencia(c, c["X"], c["Y"], beta)
    assert torch.equal(A_ref_b.crow_indices().to(torch.int64), R._pA.crow)
    assert torch.equal(A_ref_b.col_indices().to(torch.int64), R._pA.col)


def _assemble_kmgg(c, X, Y):
    K, M, Gx, Gy = c["element"].montar(X, Y, c["IEN"], c["npoints"], c["nnodes"])
    return K, M, Gx, Gy


@pytest.mark.parametrize("element", ["mini", "tri6"])
def test_remontar_em_geometria_movida_bate_com_montagem_do_zero(element):
    c = _monta_caso(element)
    beta = pressure_scale_factor(*_assemble_kmgg(c, c["X"], c["Y"]), c["dt"], c["Re"])
    R = Remontador(c["dt"], c["Re"], c["X"], c["Y"], c["IEN"], c["element"], c["npoints"],
                   c["n_extra"], c["vxp"], c["vyp"], c["pp"], beta)

    torch.manual_seed(0)
    dX = 0.01 * torch.randn_like(c["X"])
    dY = 0.01 * torch.randn_like(c["Y"])
    # nao mexe nos nos de contorno (mantem a geometria da borda)
    borda = torch.cat([c["vxp"], c["pp"]]).unique()
    dX[borda] = 0.0
    dY[borda] = 0.0
    Xm, Ym = c["X"] + dX, c["Y"] + dY

    A_ref, M_ref = _referencia(c, Xm, Ym, beta)
    sis = R.remontar(Xm, Ym, beta=beta)

    assert torch.allclose(sis.A.values(), A_ref.values(), atol=1e-12, rtol=0)
    assert torch.allclose(sis.M.values(), M_ref.values(), atol=1e-12, rtol=0)


@pytest.mark.parametrize("element", ["mini", "tri6"])
def test_remontar_e_deterministico(element):
    c = _monta_caso(element)
    beta = pressure_scale_factor(*_assemble_kmgg(c, c["X"], c["Y"]), c["dt"], c["Re"])
    R = Remontador(c["dt"], c["Re"], c["X"], c["Y"], c["IEN"], c["element"], c["npoints"],
                   c["n_extra"], c["vxp"], c["vyp"], c["pp"], beta)
    Xm = c["X"] + 0.01 * torch.sin(3 * c["X"])
    s1 = R.remontar(Xm, c["Y"], beta=beta)
    s2 = R.remontar(Xm, c["Y"], beta=beta)
    assert torch.equal(s1.A.values(), s2.A.values())
    assert torch.equal(s1.rhs_cc, s2.rhs_cc)


def test_remontar_recalcula_beta_quando_none():
    c = _monta_caso("mini")
    K, M, Gx, Gy = _assemble_kmgg(c, c["X"], c["Y"])
    beta0 = pressure_scale_factor(K, M, Gx, Gy, c["dt"], c["Re"])
    R = Remontador(c["dt"], c["Re"], c["X"], c["Y"], c["IEN"], c["element"], c["npoints"],
                   c["n_extra"], c["vxp"], c["vyp"], c["pp"], beta0)
    Xm = c["X"] + 0.02 * torch.sin(5 * c["X"])
    Km, Mm, Gxm, Gym = _assemble_kmgg(c, Xm, c["Y"])
    beta_ref = pressure_scale_factor(Km, Mm, Gxm, Gym, c["dt"], c["Re"])
    assert abs(R.remontar(Xm, c["Y"], beta=None).beta - beta_ref) < 1e-10


@pytest.mark.parametrize("element", ["mini", "tri6"])
def test_divergencia_bate_com_menos_g_transposto_e_acompanha_a_geometria(element):
    """`divergencia` = `-(Gx^T vx + Gy^T vy)`, sem beta, na geometria da ultima
    montagem -- depois de `remontar` tem que usar o G da malha nova."""
    c = _monta_caso(element)
    R = Remontador(c["dt"], c["Re"], c["X"], c["Y"], c["IEN"], c["element"], c["npoints"],
                   c["n_extra"], c["vxp"], c["vyp"], c["pp"])
    g = torch.Generator().manual_seed(0)
    vx = torch.rand(c["nnodes"], dtype=torch.float64, generator=g)
    vy = torch.rand(c["nnodes"], dtype=torch.float64, generator=g)

    def esperado(X, Y):
        _, _, Gx, Gy = _assemble_kmgg(c, X, Y)
        return -(Gx.to_dense().T @ vx + Gy.to_dense().T @ vy)

    assert torch.allclose(R.divergencia(vx, vy), esperado(c["X"], c["Y"]), atol=1e-12)

    Xm = c["X"] + 0.01 * torch.sin(7.0 * c["Y"])
    R.remontar(Xm, c["Y"])
    assert torch.allclose(R.divergencia(vx, vy), esperado(Xm, c["Y"]), atol=1e-12)


def _valores_cc(c):
    """Valores prescritos nao nulos e distintos por no (para o levantamento ser exercitado de verdade)."""
    nn, npo = c["nnodes"], c["npoints"]
    vx_cc = 1.0 + 0.5 * c["Y"]
    vy_cc = -0.3 + 0.2 * c["X"]
    p_cc = 2.0 + c["Y"][:npo]
    return vx_cc[:nn], vy_cc[:nn], p_cc


@pytest.mark.parametrize("element", ["mini", "tri6"])
def test_condicoes_de_contorno_simetricas(element):
    """A eliminacao e' simetrica (A == A^T), os graus prescritos ficam como
    identidade desacoplada, e o lado direito `rhs_cc` e' exatamente o que as
    colunas eliminadas levariam: resolvendo A x = b + rhs_cc (linhas livres)
    da a mesma solucao da matriz completa com x[cc] = g."""
    c = _monta_caso(element)
    vals = _valores_cc(c)
    R = Remontador(c["dt"], c["Re"], c["X"], c["Y"], c["IEN"], c["element"], c["npoints"],
                   c["n_extra"], c["vxp"], c["vyp"], c["pp"], valores_cc=vals)
    sis = R.sistema_inicial
    A = sis.A.to_dense()
    assert torch.equal(A, A.T)

    cc = sis.eh_cc
    assert torch.all(A[cc][:, cc] == torch.eye(int(cc.sum()), dtype=A.dtype))
    assert torch.all(A[cc][:, ~cc] == 0)

    # solucao com a matriz completa (sem eliminacao), impondo x[cc] = g por substituicao
    K, M, Gx, Gy = _assemble_kmgg(c, c["X"], c["Y"])
    A_full = build_system_matrix(c["dt"], c["Re"], K, M, Gx, Gy, sis.beta).to_dense()
    g = torch.where(cc, sis.rhs_cc, torch.zeros_like(sis.rhs_cc))
    b = torch.rand(A.shape[0], dtype=A.dtype, generator=torch.Generator().manual_seed(1))
    livres = ~cc
    x_full = g.clone()
    x_full[livres] = torch.linalg.solve(A_full[livres][:, livres], (b - A_full @ g)[livres])

    x = torch.linalg.solve(A, torch.where(cc, sis.rhs_cc, b + sis.rhs_cc))
    assert torch.allclose(x, x_full, atol=1e-9)


@pytest.mark.parametrize("element", ["mini", "tri6"])
def test_definir_valores_cc_equivale_a_remontador_novo(element):
    """Parede movel: trocar a velocidade prescrita e remontar da' o mesmo sistema
    (A e rhs_cc) que um Remontador criado do zero com os valores novos."""
    c = _monta_caso(element)
    vx0, vy0, p0 = _valores_cc(c)
    R = Remontador(c["dt"], c["Re"], c["X"], c["Y"], c["IEN"], c["element"], c["npoints"],
                   c["n_extra"], c["vxp"], c["vyp"], c["pp"], beta=1.3, valores_cc=(vx0, vy0, p0))
    vx1, vy1 = 0.7 * vx0 - 0.2 * c["X"][: c["nnodes"]], vy0 + c["Y"][: c["nnodes"]]
    R.definir_valores_cc(vx1, vy1)
    sis = R.remontar(c["X"], c["Y"], beta=1.3)
    R2 = Remontador(c["dt"], c["Re"], c["X"], c["Y"], c["IEN"], c["element"], c["npoints"],
                    c["n_extra"], c["vxp"], c["vyp"], c["pp"], beta=1.3, valores_cc=(vx1, vy1, p0))
    assert torch.allclose(sis.rhs_cc, R2.sistema_inicial.rhs_cc, atol=1e-13)
    assert not torch.allclose(sis.rhs_cc, R.sistema_inicial.rhs_cc)


def test_blocos_dos_precondicionadores():
    """Com `precisa_blocos=True`: Fx/Fy sao o bloco de velocidade de A com as
    CCs do componente eliminadas; Mp/Lp sao SPD com as CCs de pressao
    eliminadas, e Lp tem soma de linha nula nos nos interiores sem CC."""
    c = _monta_caso("mini")
    R = Remontador(c["dt"], c["Re"], c["X"], c["Y"], c["IEN"], c["element"], c["npoints"],
                   c["n_extra"], c["vxp"], c["vyp"], c["pp"], precisa_blocos=True)
    sis = R.sistema_inicial
    nn = c["nnodes"]
    A = sis.A.to_dense()
    assert torch.allclose(sis.Fx.to_dense(), A[:nn, :nn], atol=1e-12)
    assert torch.allclose(sis.Fy.to_dense(), A[nn:2 * nn, nn:2 * nn], atol=1e-12)

    for P in (sis.Mp.to_dense(), sis.Lp.to_dense()):
        assert torch.allclose(P, P.T, atol=1e-14)
        assert torch.linalg.eigvalsh(P).min() > 0

    # Mp sem CC soma a area do dominio (quadrado unitario)
    R2 = Remontador(c["dt"], c["Re"], c["X"], c["Y"], c["IEN"], c["element"], c["npoints"],
                    c["n_extra"], c["vxp"][:0], c["vyp"][:0], c["pp"][:0], precisa_blocos=True)
    assert abs(R2.sistema_inicial.Mp.to_dense().sum().item() - 1.0) < 1e-12
    _, Lp_bruto = c["element"].matrizes_pressao(c["X"], c["Y"], c["IEN"], c["npoints"])
    assert torch.allclose(Lp_bruto.to_dense().sum(dim=1),
                          torch.zeros(c["npoints"], dtype=torch.float64), atol=1e-12)



def test_lp_sem_pressao_prescrita_fica_definido_positivo_com_a_saida_livre():
    """Sem nenhuma pressao prescrita (saida com condicao natural), o Lp de Neumann
    puro seria singular. Com Dirichlet so no Lp nos nos da saida livre ele e'
    SPD; Mp e o sistema A nao mudam (o pre-condicionador e' que usa o Lp)."""
    c = _monta_caso("mini")
    saida = c["pp"]  # borda direita: aqui sem pressao prescrita, como saida livre
    sem_p = c["pp"][:0]
    R = Remontador(c["dt"], c["Re"], c["X"], c["Y"], c["IEN"], c["element"], c["npoints"],
                   c["n_extra"], c["vxp"], c["vyp"], sem_p, precisa_blocos=True, nos_saida_livre=saida)
    sis = R.sistema_inicial
    assert torch.linalg.eigvalsh(sis.Lp.to_dense()).min() > 0
    assert int(sis.lp_zero.sum()) == len(saida)

    # sem saida livre nem pressao prescrita: fixa um vertice so para o Lp nao ser singular
    R2 = Remontador(c["dt"], c["Re"], c["X"], c["Y"], c["IEN"], c["element"], c["npoints"],
                    c["n_extra"], c["vxp"], c["vyp"], sem_p, precisa_blocos=True)
    assert torch.linalg.eigvalsh(R2.sistema_inicial.Lp.to_dense()).min() > 0
    assert int(R2.sistema_inicial.lp_zero.sum()) == 1
