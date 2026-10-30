"""Geometria movel: corpos, classificacao do contorno e velocidade das paredes (femns.geometria_movel)."""

import os

import numpy as np
import pytest

import femns.geometria_movel as gm
import femns.valvula_rotacao as vr
from femns.boundary import assign_boundary_names
from femns.elemento import elemento_por_nome
from femns.mesh import atualiza_nos_extras, read_mesh

DIR = os.path.join(os.path.dirname(__file__), "..", "meshes")
TH0 = np.radians(10.0)
R = float(np.hypot(0.5, 1.0))

VALVULA = {
    "corpos": {"esfera": {"tipo": "rotacao", "centro": [0.0, 0.0], "malha": 10.0, "inicio": 10.0, "fim": 20.0,
                          "duracao": 8.0}},
    "contornos": {"valve_up": {"tipo": "rigido", "corpo": "esfera"},
                  "valve_down": {"tipo": "rigido", "corpo": "esfera"},
                  "outter_wall": {"tipo": "deslizante", "corpo": "esfera"},
                  "inner_wall": {"tipo": "deslizante", "corpo": "esfera", "velocidade_fluido": "condicao"},
                  "upper": "fixo"},
}
DENTE = {
    "corpos": {"pistao": {"tipo": "translacao", "direcao": [0, 2], "inicio": 0.0, "fim": 0.4, "duracao": 8.0}},
    "contornos": {"dente": {"tipo": "deslizante", "corpo": "pistao"}},
}


@pytest.fixture(scope="module")
def valvula():
    m = read_mesh(os.path.join(DIR, "valvula_10g.msh"))
    return m, gm.preparar(gm.ler_config(VALVULA), m.X, m.Y, m.IENbound, m.IENboundElem)


@pytest.fixture(scope="module")
def dente():
    m = read_mesh(os.path.join(DIR, "canal_dente.msh"))
    return m, gm.preparar(gm.ler_config(DENTE), m.X, m.Y, m.IENbound, m.IENboundElem)


# --- config -------------------------------------------------------------------
def test_ler_config():
    c = gm.ler_config(VALVULA)
    esfera = c.corpos["esfera"]
    assert isinstance(esfera, gm.Rotacao) and esfera.malha == pytest.approx(TH0)
    assert isinstance(c.contornos["valve_up"], gm.Rigido) and c.contornos["valve_up"].velocidade_fluido == "malha"
    assert isinstance(c.contornos["outter_wall"], gm.Deslizante)
    assert c.contornos["outter_wall"].velocidade_fluido == "corpo"
    assert "upper" not in c.contornos  # fixo
    d = gm.ler_config(DENTE).corpos["pistao"]
    assert d.direcao == pytest.approx((0.0, 1.0)) and d.malha == 0.0   # normalizada; malha = inicio


@pytest.mark.parametrize("ruim", [
    {"corpos": {}, "contornos": {"a": {"tipo": "rigido", "corpo": "x"}}},
    {"corpos": {"c": {"tipo": "giro", "inicio": 0, "fim": 1}}, "contornos": {}},
    {"corpos": {"c": {"tipo": "rotacao", "fim": 1, "duracao": 1}}, "contornos": {"a": {"tipo": "rigido", "corpo": "c"}}},
    {"corpos": {"c": {"tipo": "rotacao", "inicio": 0, "fim": 1, "duracao": 1}},
     "contornos": {"a": {"tipo": "rigido", "corpo": "outro"}}},
    {"corpos": {"c": {"tipo": "rotacao", "inicio": 0, "fim": 1, "duracao": 1}},
     "contornos": {"a": {"tipo": "deslizante", "corpo": "c", "velocidade_fluido": "x"}}},
    {"corpos": {"c": {"tipo": "translacao", "inicio": 0, "fim": 1, "duracao": 1}},
     "contornos": {"a": {"tipo": "deslizante", "corpo": "c", "raio": 1.0}}},
    {"corpos": {"c": {"tipo": "translacao", "inicio": 0, "fim": 1, "duracao": 1},
                "d": {"tipo": "translacao", "inicio": 0, "fim": 1, "duracao": 1}},
     "contornos": {"a": {"tipo": "rigido", "corpo": "c"}}},          # corpo d sem contorno
    {"corpos": {"c": {"tipo": "translacao", "inicio": 0, "fim": 1}}, "contornos": {"a": {"tipo": "rigido", "corpo": "c"}}},
])
def test_ler_config_rejeita(ruim):
    with pytest.raises(ValueError):
        gm.ler_config(ruim)


def test_duracao_padrao():
    cfg = {"corpos": {"c": {"tipo": "translacao", "inicio": 0, "fim": 1}}, "contornos": {"a": {"tipo": "rigido", "corpo": "c"}}}
    assert gm.ler_config(cfg, duracao_padrao=3.0).corpos["c"].duracao == 3.0


def test_bloco_do_config_converte_os_antigos():
    vr_cfg = {"enabled": True, "theta0": 10.0, "theta_fim": 20.0, "velocidade_malha": ["valve_up", "valve_down"],
              "velocidade_esfera": ["outter_wall"]}
    b = gm.bloco_do_config({"valvula_rotacao": vr_cfg})
    c = gm.ler_config(b, duracao_padrao=1.0)
    assert c.contornos["inner_wall"].velocidade_fluido == "condicao"
    assert c.contornos["outter_wall"].raio == pytest.approx(R)
    with pytest.raises(ValueError, match="fecha"):
        gm.bloco_do_config({"valvula_rotacao": dict(vr_cfg, theta_fim=60.0)})
    d = gm.ler_config(gm.bloco_do_config({"dente_crescendo": {"enabled": True, "altura_inicio": 0.2, "altura_fim": 0.6}}),
                      duracao_padrao=1.0)
    assert d.corpos["pistao"].malha == 0.2 and isinstance(d.contornos["dente"], gm.Deslizante)
    assert gm.bloco_do_config({}) is None
    with pytest.raises(ValueError, match="um so"):
        gm.bloco_do_config({"valvula_rotacao": vr_cfg, "geometria_movel": {"enabled": True}})


def test_corpos():
    r = gm.Rotacao(nome="r", inicio=0.0, fim=1.0, duracao=1.0, centro=(1.0, 2.0))
    x, y = r.mover(np.array([2.0]), np.array([2.0]), np.pi / 2)
    assert (x[0], y[0]) == pytest.approx((1.0, 3.0))
    ux, uy = r.velocidade(np.array([2.0]), np.array([2.0]), 0.5)
    assert (ux[0], uy[0]) == pytest.approx((0.0, 0.5))
    t = gm.Translacao(nome="t", inicio=0.0, fim=1.0, duracao=1.0, direcao=(3.0, 4.0))
    x, y = t.mover(np.array([0.0]), np.array([0.0]), 5.0)
    assert (x[0], y[0]) == pytest.approx((3.0, 4.0))
    assert t.valor(0.5) == pytest.approx(0.5) and t.valor(2.0) == 1.0


# --- valvula: deslizante de rotacao --------------------------------------------
def test_classificacao_da_valvula(valvula):
    m, g = valvula
    nomes = np.asarray(m.IENboundElem)
    # 4 arcos: dois da esfera (outter_wall) e dois da carcaca (inner_wall)
    assert sorted(a.grupo for a in g.arcos) == ["inner_wall", "inner_wall", "outter_wall", "outter_wall"]
    rig = g.rigido["esfera"]
    for a in g.arcos:
        np.testing.assert_allclose(np.hypot(m.X[a.nos], m.Y[a.nos]), R, atol=1e-12)
        assert a.raio == pytest.approx(R, abs=1e-6)           # raio inferido da malha
        assert a.nos[-1] in rig and a.nos[0] not in rig        # ponta movel e' ponto do corpo
        assert np.all((a.lam > 0) & (a.lam < 1))
    corda = np.unique(m.IENbound[np.isin(nomes, ["valve_up", "valve_down"])])
    assert np.isin(corda, rig).all()
    parede = np.unique(m.IENbound[np.isin(nomes, ["outter_wall", "inner_wall"])])
    fora = parede[np.hypot(m.X[parede], m.Y[parede]) > R + 1e-6]
    assert len(fora) > 0 and not np.isin(fora, g.idx_movel).any()   # filetes da carcaca parados
    assert not g.superficie["esfera"][fora].any()


def test_valvula_no_estado_da_malha_reproduz_a_malha(valvula):
    m, g = valvula
    Xc, Yc = g.contorno_em(m.X, m.Y, g.estado_malha(), g.estado_malha())
    np.testing.assert_allclose(Xc, m.X, atol=1e-14)
    np.testing.assert_allclose(Yc, m.Y, atol=1e-14)


def test_arcos_esticam_entre_as_pontas(valvula):
    m, g = valvula
    th = np.radians(20.0)
    X, Y = g.contorno_em(m.X, m.Y, {"esfera": TH0}, {"esfera": th})
    idx = g.rigido["esfera"]
    xr, yr = vr.rotacao(th - TH0, m.X[idx], m.Y[idx])
    np.testing.assert_allclose(X[idx], xr, atol=1e-14)
    for a in g.arcos:
        phi = np.unwrap(np.arctan2(Y[a.nos], X[a.nos]))
        assert phi[0] == pytest.approx(a.phi0[0], abs=1e-14)                  # ponta parada
        assert phi[-1] - a.phi0[-1] == pytest.approx(th - TH0, abs=1e-12)     # ponta movel gira com o corpo
        eta0 = (a.phi0 - a.phi0[0]) / (a.phi0[-1] - a.phi0[0])
        np.testing.assert_allclose((phi - phi[0]) / (phi[-1] - phi[0]), eta0, atol=1e-12)
        np.testing.assert_allclose(np.hypot(X[a.nos], Y[a.nos]), a.raio, atol=1e-14)
    X2, Y2 = X.copy(), Y.copy()
    g.suavizar_arcos(X2, Y2, a2=0.5, varreduras=5, pesos="referencia")
    np.testing.assert_allclose(X2, X, atol=1e-13)
    g.suavizar_arcos(X2, Y2, a2=1.0, varreduras=2000, pesos="uniforme")
    for a in g.arcos:
        dphi = np.diff(np.unwrap(np.arctan2(Y2[a.nos], X2[a.nos])))
        np.testing.assert_allclose(dphi, dphi.mean(), rtol=1e-6)


def test_raio_errado_levanta_erro():
    m = read_mesh(os.path.join(DIR, "valvula_10g.msh"))
    cfg = gm.ler_config({**VALVULA, "contornos": {**VALVULA["contornos"],
                                                  "outter_wall": {"tipo": "deslizante", "corpo": "esfera", "raio": 1.3}}})
    with pytest.raises(ValueError, match="circulo"):
        gm.preparar(cfg, m.X, m.Y, m.IENbound, m.IENboundElem)


def test_grupo_inexistente_levanta_erro(dente):
    m, _ = dente
    cfg = gm.ler_config({**DENTE, "contornos": {"valve_up": {"tipo": "rigido", "corpo": "pistao"}}})
    with pytest.raises(ValueError, match="nao existem"):
        gm.preparar(cfg, m.X, m.Y, m.IENbound, m.IENboundElem)


@pytest.mark.parametrize("H, L, th0", [(0.5, 1.5, TH0), (0.5, 2.0, -TH0), (0.5, 2.0, np.radians(12.0))])
def test_conferir_corda(H, L, th0):
    m = read_mesh(os.path.join(DIR, "valvula_10g.msh"))
    vr.conferir_corda(m.X, m.Y, m.IENbound, m.IENboundElem, 0.5, 2.0, TH0)
    with pytest.raises(ValueError):
        vr.conferir_corda(m.X, m.Y, m.IENbound, m.IENboundElem, H, L, th0)


def test_malha_antiga_de_quinas_vivas():
    caminho = os.path.join(DIR, "valve_10g_15k.msh")
    if not os.path.exists(caminho):
        pytest.skip("malha valve_10g_15k.msh ausente")
    m = read_mesh(caminho)
    b = gm.de_valvula_rotacao({"theta0": 10.0, "theta_fim": 20.0, "L": 1.5})
    g = gm.preparar(gm.ler_config(b, duracao_padrao=1.0), m.X, m.Y, m.IENbound, m.IENboundElem)
    # 4 arcos de 17 nos: junta fixa + 15 deslizando + ponta da corda (que e' da corda)
    assert [len(a.nos) for a in g.arcos] == [17] * 4
    assert len(g.idx_arco) == 60


# --- dente: deslizante de translacao -------------------------------------------
def test_classificacao_do_dente(dente):
    m, g = dente
    nos = np.unique(m.IENbound[np.asarray(m.IENboundElem) == "dente"])
    topo = nos[np.isclose(m.Y[nos], -0.3)]
    piso = nos[np.isclose(m.Y[nos], -0.5)]
    assert len(piso) == 2 and set(topo) <= set(g.rigido["pistao"])       # topo rigido
    assert not np.isin(piso, g.idx_movel).any()                           # base parada
    (rt,) = g.retas
    np.testing.assert_allclose(rt.eta, (m.Y[rt.nos] + 0.5) / 0.2, atol=1e-12)
    assert g.superficie["pistao"][nos].all()


def test_dente_cresce(dente):
    m, g = dente
    X, Y = g.contorno_em(m.X, m.Y, {"pistao": 0.0}, {"pistao": 0.4})
    m_ = g.idx_movel
    np.testing.assert_array_equal(X[m_], m.X[m_])
    rt = g.retas[0]
    np.testing.assert_allclose(Y[rt.nos], -0.5 + rt.eta * 0.6, atol=1e-14)
    np.testing.assert_allclose(Y[g.rigido["pistao"]], -0.3 + 0.4 + 0 * X[g.rigido["pistao"]], atol=1e-14)
    fixo = np.flatnonzero(g.contorno)
    fixo = fixo[~np.isin(fixo, m_)]
    np.testing.assert_array_equal(Y[fixo], m.Y[fixo])
    assert g.desloc_max({"pistao": 0.0}, {"pistao": 0.4}) == pytest.approx(0.4)


# --- velocidade que o fluido ve ------------------------------------------------
def _paredes(m, g, prioridade):
    el = elemento_por_nome("tri6")
    IEN, X, Y, IENb = el.adiciona_nos(m.IEN, m.X, m.Y, m.IENbound)
    cc = assign_boundary_names(IENb, m.IENboundElem, m.npoints, prioridade, m.IENpoint, m.IENpointElem, nnodes=len(X))
    return cc, gm.ParedesMoveis(g, cc, len(X), lambda a, b: atualiza_nos_extras(IEN, a, b))


def test_velocidade_das_paredes_da_valvula(valvula):
    m, g = valvula
    cc, p = _paredes(m, g, ["outlet", "inlet", "upper", "bottom", "inner_wall", "outter_wall", "valve_up", "valve_down"])
    assert not any(cc[i] == "inner_wall" for i in p.idx)                  # condicao: fica com o config
    dt, th = 1e-3, TH0 + 1e-4
    X, Y = g.contorno_em(m.X, m.Y, {"esfera": TH0}, {"esfera": th})
    ux, uy = p.velocidades(m.X, m.Y, X, Y, {"esfera": TH0}, {"esfera": th}, dt)
    corda = np.array([cc[i] in ("valve_up", "valve_down") for i in p.idx])
    vert = (p.idx < m.npoints)
    i = p.idx[corda & vert]
    np.testing.assert_allclose(ux[corda & vert], (X[i] - m.X[i]) / dt)     # malha: diferenca finita
    esf = np.array([cc[i] == "outter_wall" for i in p.idx]) & vert
    i = p.idx[esf]
    sup = g.superficie["esfera"][i]
    om = (th - TH0) / dt
    np.testing.assert_allclose(ux[esf][sup], -om * Y[i][sup])              # corpo: omega x p
    np.testing.assert_allclose(uy[esf][sup], om * X[i][sup])
    assert not ux[esf][~sup].any()                                          # filete da carcaca parado


def test_velocidade_do_pistao_e_uniforme(dente):
    m, g = dente
    cc, p = _paredes(m, g, ["outlet", "inlet", "top", "dente", "bottom"])
    X, Y = g.contorno_em(m.X, m.Y, {"pistao": 0.0}, {"pistao": 0.01})
    ux, uy = p.velocidades(m.X, m.Y, X, Y, {"pistao": 0.0}, {"pistao": 0.01}, 0.1)
    assert len(p.idx) > 0 and all(cc[i] == "dente" for i in p.idx)
    np.testing.assert_allclose(uy, 0.1, rtol=1e-12)
    np.testing.assert_allclose(ux, 0.0, atol=1e-15)


def test_lei_temporal():
    a, b, T, d = 0.1, 0.3, 4.0, 1.0
    assert vr.theta_em(0.5, a, b, T, d) == a
    assert vr.theta_em(9.0, a, b, T, d) == b
    for t in (1.3, 2.0, 4.1):
        eps = 1e-6
        fd = (vr.theta_em(t + eps, a, b, T, d) - vr.theta_em(t - eps, a, b, T, d)) / (2 * eps)
        assert vr.omega_em(t, a, b, T, d) == pytest.approx(fd, rel=1e-6)


def test_geometria_da_valvula():
    g = vr.GeometriaValvula(H=0.5, L=2.0)
    assert g.r == pytest.approx(R)
    assert np.degrees(g.theta_fechamento) == pytest.approx(2 * np.degrees(np.arctan(0.5)))
    assert g.fresta(0.0) == pytest.approx(1.0)


def test_corpo_com_etapas():
    cfg = gm.ler_config({"corpos": {"e": {"tipo": "rotacao", "inicio": 10, "malha": 17, "atraso": 1.0,
                                          "etapas": [{"fim": 5, "duracao": 1.0}, {"fim": 40, "duracao": 7.0},
                                                     {"fim": 5, "duracao": 7.0}]}},
                         "contornos": {"a": {"tipo": "rigido", "corpo": "e"}}})
    c = cfg.corpos["e"]
    graus = lambda t: float(np.degrees(c.valor(t)))  # noqa: E731
    assert c.fim == pytest.approx(np.radians(5)) and c.duracao == 15.0 and c.malha == pytest.approx(np.radians(17))
    for t, esperado in ((0.0, 10), (1.0, 10), (2.0, 5), (9.0, 40), (16.0, 5), (99.0, 5)):
        assert graus(t) == pytest.approx(esperado, abs=1e-12)
    assert graus(1.5) == pytest.approx(7.5)                     # meio da 1a rampa
    assert graus(5.5) == pytest.approx(22.5)                    # meio da 2a
    eps = 1e-6                                                  # velocidade nula nas trocas de sentido
    for tt in (2.0, 9.0):
        assert abs(c.valor(tt + eps) - c.valor(tt - eps)) / (2 * eps) < 1e-4
    with pytest.raises(ValueError, match="OU"):
        gm.ler_config({"corpos": {"e": {"tipo": "rotacao", "inicio": 10, "fim": 5, "etapas": [{"fim": 5, "duracao": 1}]}},
                       "contornos": {"a": {"tipo": "rigido", "corpo": "e"}}})
