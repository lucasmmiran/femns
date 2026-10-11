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
    assert torch.equal(A_ref_b.crow_indices().to(torch.int64), R._csr_crow)
    assert torch.equal(A_ref_b.col_indices().to(torch.int64), R._csr_col)


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
    A_re, M_re, *_ = R.remontar(Xm, Ym, beta=beta)

    assert torch.allclose(A_re.values(), A_ref.values(), atol=1e-12, rtol=0)
    assert torch.allclose(M_re.values(), M_ref.values(), atol=1e-12, rtol=0)


@pytest.mark.parametrize("element", ["mini", "tri6"])
def test_remontar_e_deterministico(element):
    c = _monta_caso(element)
    beta = pressure_scale_factor(*_assemble_kmgg(c, c["X"], c["Y"]), c["dt"], c["Re"])
    R = Remontador(c["dt"], c["Re"], c["X"], c["Y"], c["IEN"], c["element"], c["npoints"],
                   c["n_extra"], c["vxp"], c["vyp"], c["pp"], beta)
    Xm = c["X"] + 0.01 * torch.sin(3 * c["X"])
    v1 = R.remontar(Xm, c["Y"], beta=beta)[0].values().clone()
    v2 = R.remontar(Xm, c["Y"], beta=beta)[0].values().clone()
    assert torch.equal(v1, v2)


def test_remontar_recalcula_beta_quando_none():
    c = _monta_caso("mini")
    K, M, Gx, Gy = _assemble_kmgg(c, c["X"], c["Y"])
    beta0 = pressure_scale_factor(K, M, Gx, Gy, c["dt"], c["Re"])
    R = Remontador(c["dt"], c["Re"], c["X"], c["Y"], c["IEN"], c["element"], c["npoints"],
                   c["n_extra"], c["vxp"], c["vyp"], c["pp"], beta0)
    Xm = c["X"] + 0.02 * torch.sin(5 * c["X"])
    Km, Mm, Gxm, Gym = _assemble_kmgg(c, Xm, c["Y"])
    beta_ref = pressure_scale_factor(Km, Mm, Gxm, Gym, c["dt"], c["Re"])
    *_, beta_re = R.remontar(Xm, c["Y"], beta=None)
    assert abs(beta_re - beta_ref) < 1e-10


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
