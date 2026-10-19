"""Cinematica do contorno da valvula esfera fechando por rotacao (femns.valvula_rotacao)."""

import os

import numpy as np
import pytest

import femns.valvula_rotacao as vr
from femns.mesh import read_mesh

MALHA = os.path.join(os.path.dirname(__file__), "..", "meshes", "valvula_10g.msh")
TH0 = np.radians(10.0)


@pytest.fixture(scope="module")
def malha():
    return read_mesh(MALHA)


def _contorno(m, th0=TH0, H=0.5, L=2.0):
    return vr.preparar(m.IEN, m.X, m.Y, m.IENbound, m.IENboundElem, H=H, L=L, theta0=th0)


def test_geometria():
    g = vr.GeometriaValvula(H=0.5, L=2.0)
    assert g.r == pytest.approx(np.sqrt(0.5**2 + 1.0**2))
    assert np.degrees(g.theta_fechamento) == pytest.approx(2 * np.degrees(np.arctan(0.5)))
    assert g.fresta(0.0) == pytest.approx(1.0)
    assert g.fresta(g.theta_fechamento) == pytest.approx(0.0, abs=1e-14)


def test_classificacao_da_malha_com_filetes(malha):
    c = _contorno(malha)
    nomes = np.asarray(malha.IENboundElem)
    # 4 arcos: dois da esfera (outter_wall) e dois da carcaca (inner_wall)
    assert sorted(a.grupo for a in c.arcos) == ["inner_wall", "inner_wall", "outter_wall", "outter_wall"]
    for a in c.arcos:
        # pontas e nos internos sobre o circulo; a ponta movel e' ponto material da esfera
        np.testing.assert_allclose(np.hypot(malha.X[a.nos], malha.Y[a.nos]), c.geo.r, atol=1e-12)
        assert a.nos[-1] in c.idx_rigido and a.nos[0] not in c.idx_rigido
        assert np.all((a.lam > 0) & (a.lam < 1))
    # corda inteira rigida; filetes da carcaca (|p| > r) fixos
    corda = np.unique(malha.IENbound[np.isin(nomes, ["valve_up", "valve_down"])])
    assert np.isin(corda, c.idx_rigido).all()
    parede = np.unique(malha.IENbound[np.isin(nomes, ["outter_wall", "inner_wall"])])
    fora = parede[np.hypot(malha.X[parede], malha.Y[parede]) > c.geo.r + 1e-6]
    assert len(fora) > 0 and not np.isin(fora, c.idx_movel).any()


def test_referencia_reproduz_a_malha(malha):
    c = _contorno(malha)
    idx, x, y = vr.posicoes_rigidas(c, TH0)
    np.testing.assert_allclose(x, malha.X[idx], atol=1e-14)
    np.testing.assert_allclose(y, malha.Y[idx], atol=1e-14)
    X, Y = malha.X.copy(), malha.Y.copy()
    vr.prever_arcos(c, X, Y, TH0, TH0)
    np.testing.assert_allclose(X, malha.X, atol=1e-14)
    np.testing.assert_allclose(Y, malha.Y, atol=1e-14)


@pytest.mark.parametrize("H, L, th0", [(0.5, 1.5, TH0), (0.5, 2.0, -TH0), (0.5, 2.0, np.radians(12.0))])
def test_geometria_errada_levanta_erro(malha, H, L, th0):
    with pytest.raises(ValueError):
        _contorno(malha, th0=th0, H=H, L=L)


def test_arcos_esticam_entre_as_pontas(malha):
    c = _contorno(malha)
    X, Y = malha.X.copy(), malha.Y.copy()
    th = np.radians(20.0)
    idx, xr, yr = vr.posicoes_rigidas(c, th)
    X[idx], Y[idx] = xr, yr
    vr.prever_arcos(c, X, Y, TH0, th)
    for a in c.arcos:
        phi = np.unwrap(np.arctan2(Y[a.nos], X[a.nos]))
        assert phi[0] == pytest.approx(a.phi0[0], abs=1e-14)                  # ponta fixa parada
        assert phi[-1] - a.phi0[-1] == pytest.approx(th - TH0, abs=1e-12)     # ponta movel gira com a esfera
        eta0 = (a.phi0 - a.phi0[0]) / (a.phi0[-1] - a.phi0[0])
        np.testing.assert_allclose((phi - phi[0]) / (phi[-1] - phi[0]), eta0, atol=1e-12)  # proporcional
        np.testing.assert_allclose(np.hypot(X[a.nos], Y[a.nos]), c.geo.r, atol=1e-14)
    # com os pesos da referencia o corretor 1D nao mexe na distribuicao proporcional
    X2, Y2 = X.copy(), Y.copy()
    vr.suavizar_arcos(c, X2, Y2, a2=0.5, varreduras=5, pesos="referencia")
    np.testing.assert_allclose(X2, X, atol=1e-13)
    # com pesos uniformes ela tende ao espacamento igual
    vr.suavizar_arcos(c, X2, Y2, a2=1.0, varreduras=2000, pesos="uniforme")
    for a in c.arcos:
        dphi = np.diff(np.unwrap(np.arctan2(Y2[a.nos], X2[a.nos])))
        np.testing.assert_allclose(dphi, dphi.mean(), rtol=1e-6)


def test_malha_antiga_de_quinas_vivas():
    caminho = os.path.join(os.path.dirname(__file__), "..", "meshes", "valve_10g_15k.msh")
    if not os.path.exists(caminho):
        pytest.skip("malha valve_10g_15k.msh ausente")
    m = read_mesh(caminho)
    c = vr.preparar(m.IEN, m.X, m.Y, m.IENbound, m.IENboundElem, H=0.5, L=1.5, theta0=TH0)
    # 4 arcos de 17 nos: junta fixa + 15 deslizando + ponta da corda (que e' da corda)
    assert [len(a.nos) for a in c.arcos] == [17] * 4
    assert len(c.idx_arco) == 60


def test_velocidade_da_esfera(malha):
    c = _contorno(malha)
    om = 0.7
    ux, uy = vr.velocidade_esfera(c, malha.X, malha.Y, om)
    m = vr.mascara_esfera(c)
    np.testing.assert_allclose(ux[m], -om * malha.Y[m])
    np.testing.assert_allclose(uy[m], om * malha.X[m])
    assert not ux[~m].any() and not uy[~m].any()
    # filetes da carcaca (fora do circulo) ficam parados; arcos inteiros sao da esfera
    parede = np.unique(malha.IENbound[np.isin(np.asarray(malha.IENboundElem), ["outter_wall"])])
    fora = parede[np.hypot(malha.X[parede], malha.Y[parede]) > c.geo.r + 1e-6]
    assert len(fora) > 0 and not m[fora].any()
    assert m[c.idx_arco].all() and m[c.idx_rigido].all()


def test_lei_temporal():
    a, b, T, d = 0.1, 0.3, 4.0, 1.0
    assert vr.theta_em(0.5, a, b, T, d) == a
    assert vr.theta_em(9.0, a, b, T, d) == b
    for t in (1.3, 2.0, 4.1):
        eps = 1e-6
        fd = (vr.theta_em(t + eps, a, b, T, d) - vr.theta_em(t - eps, a, b, T, d)) / (2 * eps)
        assert vr.omega_em(t, a, b, T, d) == pytest.approx(fd, rel=1e-6)
    assert vr.omega_em(d, a, b, T, d) == 0.0 and vr.omega_em(d + T, a, b, T, d) == pytest.approx(0.0, abs=1e-15)
