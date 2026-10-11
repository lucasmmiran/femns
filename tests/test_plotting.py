import os

import numpy as np
from matplotlib.tri import Triangulation

from femns.io import write_vtk
from femns.plotting import ler_solucao, plot_paineis, plot_solucoes


def quadrado_dois_triangulos():
    """Mesmo quadrado unitario de tests/test_mesh.py, em formato de ponto/celula VTK."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    cells = [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))]
    return points, cells


def campos(escala=1.0):
    return {
        "vx": escala * np.array([0.0, 1.0, 1.0, 0.0]),
        "vy": escala * np.array([0.0, 0.0, 0.5, 0.5]),
        "p": escala * np.array([1.0, 0.5, 0.0, 0.5]),
    }


def triang():
    return Triangulation([0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0], np.array([[0, 1, 2], [0, 2, 3]]))


def test_plot_paineis_um_painel_gera_png_nao_vazio(tmp_path):
    path = tmp_path / "campo.png"
    plot_paineis([triang()], [np.array([0.0, 1.0, 2.0, 1.0])], [None], "Teste", str(path))
    assert path.exists() and path.stat().st_size > 0


def test_plot_paineis_dois_paineis_gera_png_nao_vazio(tmp_path):
    path = tmp_path / "comparacao.png"
    plot_paineis([triang(), triang()], [np.array([0.0, 1.0, 2.0, 1.0]), np.array([0.0, 2.0, 4.0, 2.0])],
                 ["A", "B"], "Teste", str(path))
    assert path.exists() and path.stat().st_size > 0


def test_plot_solucoes_uma_solucao_um_png_por_variavel_mais_a_magnitude(tmp_path):
    points, cells = quadrado_dois_triangulos()

    caminhos = plot_solucoes([(points, cells, campos())], [None], str(tmp_path))

    nomes = {os.path.basename(c) for c in caminhos}
    assert nomes == {"vx.png", "vy.png", "p.png", "velocidade_magnitude.png"}
    assert all(os.path.getsize(c) > 0 for c in caminhos)


def test_plot_solucoes_pula_variavel_ausente(tmp_path):
    points, cells = quadrado_dois_triangulos()

    caminhos = plot_solucoes([(points, cells, {"p": np.array([1.0, 0.5, 0.0, 0.5])})], [None], str(tmp_path))

    assert {os.path.basename(c) for c in caminhos} == {"p.png"}  # sem vx/vy, nao ha magnitude


def test_plot_solucoes_duas_solucoes_usa_sufixo_de_comparacao(tmp_path):
    points, cells = quadrado_dois_triangulos()

    caminhos = plot_solucoes([(points, cells, campos()), (points, cells, campos(2.0))], ["mini", "tri6"], str(tmp_path))

    nomes = {os.path.basename(c) for c in caminhos}
    assert nomes == {"vx_comparacao.png", "vy_comparacao.png", "p_comparacao.png", "velocidade_magnitude_comparacao.png"}
    assert all(os.path.getsize(c) > 0 for c in caminhos)


def test_ler_solucao_le_vtk_escrito_por_write_vtk(tmp_path):
    points, cells = quadrado_dois_triangulos()
    vtk_path = tmp_path / "solucao-1.vtk"
    write_vtk(str(vtk_path), points, cells, campos())

    caminhos = plot_solucoes([ler_solucao(str(vtk_path))], [None], str(tmp_path / "imagens"))

    assert len(caminhos) == 4 and all(os.path.exists(c) for c in caminhos)
