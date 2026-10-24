"""Dente crescendo na parede do canal (femns.dente_crescendo)."""

import os

import numpy as np
import pytest

import femns.dente_crescendo as dc
from femns.mesh import read_mesh
from femns.movimento_malha import laplaciano_p1

MALHA = os.path.join(os.path.dirname(__file__), "..", "meshes", "canal_dente.msh")


@pytest.fixture(scope="module")
def malha():
    return read_mesh(MALHA)


@pytest.fixture(scope="module")
def contorno(malha):
    return dc.preparar(malha.X, malha.Y, malha.IENbound, malha.IENboundElem)


def test_preparar_acha_o_dente(contorno):
    assert contorno.y_piso == pytest.approx(-0.5)
    assert contorno.h0 == pytest.approx(0.2)
    assert (contorno.x_esq, contorno.x_dir) == pytest.approx((-0.05, 0.05))
    assert contorno.y_teto == pytest.approx(0.5)
    assert np.all((contorno.eta > 0) & (contorno.eta <= 1))
    assert len(contorno.idx_lateral) < len(contorno.idx_movel)


def test_grupo_inexistente_levanta_erro(malha):
    with pytest.raises(ValueError, match="nao existe"):
        dc.preparar(malha.X, malha.Y, malha.IENbound, malha.IENboundElem, grupo="valve_up")


def test_passo_mantem_contorno_e_area(malha, contorno):
    mv = dc.MovimentoDente(contorno, malha.IEN, malha.X, malha.Y)
    for h in np.linspace(0.2, 0.6, 41)[1:]:
        info = mv.passo(h)
    assert info["invertidos"] == 0 and info["q_min"] > 0.3
    # area do fluido = canal - dente
    assert info["area_total"] == pytest.approx(7.0 - 0.1 * 0.6, abs=1e-10)
    # dente: topo em y_piso + h, laterais proporcionais, x parado
    m = contorno.idx_movel
    np.testing.assert_allclose(mv.Y[m], -0.5 + contorno.eta * 0.6, atol=1e-14)
    np.testing.assert_array_equal(mv.X[m], malha.X[m])
    # resto do contorno parado
    fixo = np.flatnonzero(contorno.contorno)
    fixo = fixo[~np.isin(fixo, m)]
    np.testing.assert_array_equal(mv.X[fixo], malha.X[fixo])
    np.testing.assert_array_equal(mv.Y[fixo], malha.Y[fixo])


def test_passo_simples_e_harmonico(malha, contorno):
    mv = dc.MovimentoDente(contorno, malha.IEN, malha.X, malha.Y)
    X0, Y0 = mv.X.copy(), mv.Y.copy()
    K = laplaciano_p1(mv.IEN, X0, Y0, mv.n, mv.chi)
    mv._passo_simples(0.21)
    ii = np.flatnonzero(~contorno.contorno)
    for d in (mv.X - X0, mv.Y - Y0):
        np.testing.assert_allclose((K @ d)[ii], 0.0, atol=1e-10 * np.abs(K).max())


def test_topo_nao_pode_encostar_na_parede(malha, contorno):
    mv = dc.MovimentoDente(contorno, malha.IEN, malha.X, malha.Y)
    with pytest.raises(ValueError, match="parede oposta"):
        mv.passo(1.0)


def test_velocidade_pistao():
    ux, uy = dc.velocidade_pistao(["dente", "bottom", None, "dente"], "dente", 0.3, 4)
    np.testing.assert_array_equal(ux, 0.0)
    np.testing.assert_array_equal(uy, [0.3, 0.0, 0.0, 0.3])
