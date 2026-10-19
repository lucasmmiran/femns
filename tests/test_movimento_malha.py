"""Movimento de malha por soma ponderada de deslocamentos (femns.movimento_malha)."""

import os

import numpy as np
import pytest
import scipy.sparse.linalg as spla

import femns.movimento_malha as mm
import femns.valvula_rotacao as vr
from femns.mesh import read_mesh

MALHA = os.path.join(os.path.dirname(__file__), "..", "meshes", "valvula_10g.msh")
TH0 = np.radians(10.0)


@pytest.fixture(scope="module")
def malha():
    return read_mesh(MALHA)


@pytest.fixture(scope="module")
def contorno(malha):
    return vr.preparar(malha.IEN, malha.X, malha.Y, malha.IENbound, malha.IENboundElem, H=0.5, L=2.0, theta0=TH0)


def test_valor_medio_reproduz_a_malha_e_movimento_afim(malha, contorno):
    W = mm.pesos_valor_medio(malha.IEN, malha.X, malha.Y, malha.npoints)
    ii = np.flatnonzero(~contorno.contorno)
    assert W.data.min() > 0
    np.testing.assert_allclose(np.asarray(W.sum(axis=1)).ravel()[ii], 1.0, atol=1e-14)
    np.testing.assert_allclose((W @ malha.X)[ii], malha.X[ii], atol=1e-13)
    np.testing.assert_allclose((W @ malha.Y)[ii], malha.Y[ii], atol=1e-13)
    # rotacao rigida + translacao + cisalhamento: o termo de suavizacao e' zero
    xr, yr = vr.rotacao(0.4, malha.X, malha.Y)
    xa, ya = 1.3 * xr + 0.2 * yr + 0.7, yr - 0.1
    np.testing.assert_allclose((W @ xa)[ii], xa[ii], atol=1e-12)
    np.testing.assert_allclose((W @ ya)[ii], ya[ii], atol=1e-12)


def test_pesos_uniformes_sao_o_centroide():
    IEN = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    X = np.array([0.0, 1.0, 1.0, 0.0, 0.3])
    Y = np.array([0.0, 0.0, 1.0, 1.0, 0.6])
    W = mm.pesos_uniformes(IEN, 5)
    assert (W @ X)[4] == pytest.approx(0.5) and (W @ Y)[4] == pytest.approx(0.5)


def test_odt_num_quadrado_simetrico_e_ponto_fixo():
    IEN = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    X = np.array([0.0, 1.0, 1.0, 0.0, 0.5])
    Y = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
    ax, ay = mm.alvo_odt(IEN, X, Y, 5)
    assert (ax[4], ay[4]) == pytest.approx((0.5, 0.5))


@pytest.mark.parametrize("bloco", [
    {"suavizacao": {"pesos": "valor_medio", "a2": 0.5, "varreduras": 3}},
    {"rotacao": {"tipo": "radial", "rho1": 0.5, "rho2": 1.0},
     "suavizacao": {"pesos": "valor_medio", "a2": 0.5, "varreduras": 3, "a2_arco": 0.5, "pesos_arco": "uniforme"}},
    {"rotacao": {"tipo": "idw"}, "suavizacao": {"pesos": "uniforme", "a2": 0.5, "varreduras": 2}},
    {"extensao": {"a3": 1.0}},
])
def test_invariantes_do_passo(malha, contorno, bloco):
    mv = mm.MovimentoValvula(contorno, malha.IEN, malha.X, malha.Y, mm.opcoes_movimento(bloco))
    area0 = mm.area_com_sinal(mv.IEN, mv.X, mv.Y).sum()
    fixos = contorno.contorno.copy()
    fixos[contorno.idx_movel] = False
    for th in np.radians(np.linspace(10.1, 12.0, 20)):  # subpassos automaticos (desloc_max_h)
        info = mv.passo(th)
    assert info["invertidos"] == 0
    np.testing.assert_array_equal(mv.X[fixos], malha.X[fixos])          # tubo e juntas nunca andam
    np.testing.assert_array_equal(mv.Y[fixos], malha.Y[fixos])
    arco = contorno.idx_arco
    np.testing.assert_allclose(np.hypot(mv.X[arco], mv.Y[arco]), contorno.geo.r, atol=1e-14)
    idx, xr, yr = vr.posicoes_rigidas(contorno, np.radians(12.0))
    np.testing.assert_allclose(mv.X[idx], xr, atol=1e-14)               # esfera: rotacao rigida exata
    # a area do dominio so muda pela corda poligonal dos arcos (que esticam)
    assert info["area_total"] == pytest.approx(area0, abs=1e-4)


def test_extensao_e_harmonica(malha, contorno):
    """a3 = 1, sem rotacao nem suavizacao: o deslocamento do interior resolve K d = 0."""
    mv = mm.MovimentoValvula(contorno, malha.IEN, malha.X, malha.Y, mm.CONTROLE)
    X0, Y0 = mv.X.copy(), mv.Y.copy()
    K = mm.laplaciano_p1(mv.IEN, X0, Y0, mv.n, chi=2.0)
    assert mv.passo(np.radians(10.01))["subpassos"] == 1
    d = mv.X - X0
    ii = mv.interior
    np.testing.assert_allclose((K @ d)[ii], 0.0, atol=1e-10 * np.abs(K).max())


def test_guarda_recusa_movimento_que_inverte():
    """Estrela nao convexa: o centroide dos vizinhos fica fora do nucleo e inverteria um triangulo."""
    anel = np.array([[-1, -1], [1, -1], [1, 1], [0.9, 1], [0.8, 1], [0.0, -0.3], [-1, 1]], float)
    n = len(anel)
    X = np.r_[anel[:, 0], 0.051]
    Y = np.r_[anel[:, 1], -0.38]
    IEN = np.array([[i, (i + 1) % n, n] for i in range(n)])
    assert (mm.area_com_sinal(IEN, X, Y) > 0).all()
    W = mm.pesos_uniformes(IEN, n + 1)
    Xc, Yc = X.copy(), Y.copy()
    Xc[n], Yc[n] = (W @ X)[n], (W @ Y)[n]
    assert (mm.area_com_sinal(IEN, Xc, Yc) <= 0).any()  # sem guarda, inverte

    c = vr.ContornoValvula(geo=vr.GeometriaValvula(), theta0=0.1, X0=X, Y0=Y,
                           contorno=np.arange(n + 1) < n, idx_rigido=np.empty(0, int), arcos=[])
    mv = mm.MovimentoValvula(c, IEN, X, Y, mm.OpcoesMovimento(rotacao="nenhuma", pesos="uniforme",
                                                            a2=1.0, varreduras=1))
    recusados = mv._corrigir()
    assert recusados == 1
    assert (mm.area_com_sinal(IEN, mv.X, mv.Y) > 0).all()


def test_subpassos_quando_o_contorno_anda_muito(malha, contorno):
    mv = mm.MovimentoValvula(contorno, malha.IEN, malha.X, malha.Y, mm.CONTROLE)
    info = mv.passo(np.radians(11.0))
    assert info["subpassos"] > 1 and info["invertidos"] == 0


def test_opcoes_movimento():
    assert mm.opcoes_movimento(None) == mm.CONTROLE
    o = mm.opcoes_movimento({"rotacao": {"tipo": "idw", "p": 2}, "suavizacao": {"a2": 0.3}})
    assert (o.rotacao, o.idw_p, o.pesos, o.a2, o.varreduras, o.a3) == ("idw", 2.0, "valor_medio", 0.3, 3, 0.0)
    for ruim in ({"rotacao": {"tipo": "x"}}, {"suavizacao": {"pesos": "x"}}, {"suavizacao": {"a2": 2}},
                 {"rotacao": {"tipo": "radial", "rho1": 1.0, "rho2": 0.5}}, {"outra": 1},
                 {"extensao": {"a3": -1}}, {"suavizacao": {"pesos_arco": "x"}}):
        with pytest.raises(ValueError):
            mm.opcoes_movimento(ruim)


def test_laplaciano_p1_soma_de_linha_nula(malha):
    K = mm.laplaciano_p1(malha.IEN, malha.X, malha.Y, malha.npoints, chi=1.0)
    np.testing.assert_allclose(np.asarray(K.sum(axis=1)).ravel(), 0.0, atol=1e-9 * abs(K).max())
    assert spla.norm(K - K.T) < 1e-10 * spla.norm(K)
