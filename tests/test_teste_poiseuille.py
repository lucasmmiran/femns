"""Malha e metricas do teste de regressao do Poiseuille (scripts/teste_poiseuille.py)."""

import importlib.util
import os

import numpy as np

from femns.mesh import read_mesh

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script(nome):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(RAIZ, "scripts", f"{nome}.py"))
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_malha_gravada_e_a_do_gerador(tmp_path):
    """`meshes/Poiseuille_test.msh` e' reproduzida byte a byte pelo gerador (a referencia depende dela)."""
    gerador = _script("gerar_malha_poiseuille_test")
    destino = tmp_path / "malha.msh"
    gerador.escrever_msh(str(destino), *gerador.gerar(140, 20))
    with open(os.path.join(RAIZ, "meshes", "Poiseuille_test.msh"), "rb") as f:
        assert destino.read_bytes() == f.read()


def test_malha_contornos_e_orientacao():
    malha = read_mesh(os.path.join(RAIZ, "meshes", "Poiseuille_test.msh"))
    assert set(malha.IENboundElem) == {"inlet", "outlet", "upper", "down"}
    X, Y, IEN = malha.X, malha.Y, malha.IEN
    a, b, c = IEN[:, 0], IEN[:, 1], IEN[:, 2]
    area = 0.5 * ((X[b] - X[a]) * (Y[c] - Y[a]) - (X[c] - X[a]) * (Y[b] - Y[a]))
    assert (area > 0).all()  # todos anti-horarios
    assert np.isclose(area.sum(), 7.0)
    nomes = np.array(malha.IENboundElem)
    for nome, coord, valor in (("inlet", X, 0.0), ("outlet", X, 7.0), ("down", Y, 0.0), ("upper", Y, 1.0)):
        assert np.allclose(coord[malha.IENbound[nomes == nome]], valor)


def test_erros_exatos_zeram_na_solucao_exata():
    teste = _script("teste_poiseuille")
    rng = np.random.default_rng(0)
    X, Y = rng.uniform(0, 7, 500), rng.uniform(0, 1, 500)
    re_ = 100.0
    erros = teste.erros_exatos(X, Y, 6 * Y * (1 - Y), 0 * Y, -12 / re_ * (X - 7), re_)
    assert erros["vx_max"] < 1e-14 and erros["vy_max"] == 0.0
    assert erros["dpdx_erro_rel"] < 1e-10


def test_definir_chave_aninhada():
    teste = _script("teste_poiseuille")
    cfg = {"simulation": {"dt": 0.01}}
    teste._definir(cfg, "simulation.solver.chebyshev.velocidade", "3")
    teste._definir(cfg, "simulation.dt", "0.02")
    assert cfg == {"simulation": {"dt": 0.02, "solver": {"chebyshev": {"velocidade": 3}}}}
