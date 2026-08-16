import os

import numpy as np
from matplotlib.tri import Triangulation

from femns.io import write_vtk
from femns.plotting import (
    plot_campo,
    plot_campo_comparacao,
    plot_comparacao_solucao,
    plot_comparacao_ultimas_solucoes,
    plot_solucao,
    plot_ultima_solucao,
    ultimo_vtk,
)


def quadrado_dois_triangulos():
    """Mesmo quadrado unitario de tests/test_mesh.py, em formato de ponto/celula VTK."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    cells = [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))]
    return points, cells


def test_plot_campo_gera_png_nao_vazio(tmp_path):
    triang = Triangulation([0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0], np.array([[0, 1, 2], [0, 2, 3]]))
    path = tmp_path / "campo.png"

    plot_campo(triang, np.array([0.0, 1.0, 2.0, 1.0]), "Teste", str(path))

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_solucao_gera_um_png_por_variavel_mais_a_magnitude(tmp_path):
    points, cells = quadrado_dois_triangulos()
    point_data = {
        "vx": np.array([0.0, 1.0, 1.0, 0.0]),
        "vy": np.array([0.0, 0.0, 0.5, 0.5]),
        "p": np.array([1.0, 0.5, 0.0, 0.5]),
    }

    caminhos = plot_solucao(points, cells, point_data, str(tmp_path))

    nomes = {os.path.basename(c) for c in caminhos}
    assert nomes == {"vx.png", "vy.png", "p.png", "velocidade_magnitude.png"}
    for caminho in caminhos:
        assert os.path.exists(caminho)
        assert os.path.getsize(caminho) > 0


def test_plot_solucao_pula_variavel_ausente(tmp_path):
    points, cells = quadrado_dois_triangulos()
    point_data = {"p": np.array([1.0, 0.5, 0.0, 0.5])}  # sem vx/vy

    caminhos = plot_solucao(points, cells, point_data, str(tmp_path))

    nomes = {os.path.basename(c) for c in caminhos}
    assert nomes == {"p.png"}  # sem vx/vy, nao da pra calcular magnitude


def test_plot_ultima_solucao_le_vtk_escrito_por_write_vtk(tmp_path):
    points, cells = quadrado_dois_triangulos()
    point_data = {
        "vx": np.array([0.0, 1.0, 1.0, 0.0]),
        "vy": np.array([0.0, 0.0, 0.5, 0.5]),
        "p": np.array([1.0, 0.5, 0.0, 0.5]),
    }
    vtk_path = tmp_path / "solucao -1.vtk"
    write_vtk(str(vtk_path), points, cells, point_data)

    caminhos = plot_ultima_solucao(str(vtk_path), str(tmp_path / "imagens"))

    assert len(caminhos) == 4
    assert all(os.path.exists(c) for c in caminhos)


def test_plot_campo_comparacao_gera_png_nao_vazio(tmp_path):
    triang = Triangulation([0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0], np.array([[0, 1, 2], [0, 2, 3]]))
    path = tmp_path / "comparacao.png"

    plot_campo_comparacao(triang, np.array([0.0, 1.0, 2.0, 1.0]), triang, np.array([0.0, 2.0, 4.0, 2.0]),
                           "Teste", "A", "B", str(path))

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_comparacao_solucao_gera_um_png_por_variavel_mais_a_magnitude(tmp_path):
    points, cells = quadrado_dois_triangulos()
    point_data_a = {
        "vx": np.array([0.0, 1.0, 1.0, 0.0]),
        "vy": np.array([0.0, 0.0, 0.5, 0.5]),
        "p": np.array([1.0, 0.5, 0.0, 0.5]),
    }
    point_data_b = {
        "vx": np.array([0.0, 2.0, 2.0, 0.0]),
        "vy": np.array([0.0, 0.0, 1.0, 1.0]),
        "p": np.array([2.0, 1.0, 0.0, 1.0]),
    }

    caminhos = plot_comparacao_solucao(points, cells, point_data_a, points, cells, point_data_b, "mini", "tri6", str(tmp_path))

    nomes = {os.path.basename(c) for c in caminhos}
    assert nomes == {"vx_comparacao.png", "vy_comparacao.png", "p_comparacao.png", "velocidade_magnitude_comparacao.png"}
    for caminho in caminhos:
        assert os.path.exists(caminho)
        assert os.path.getsize(caminho) > 0


def test_plot_comparacao_ultimas_solucoes_le_dois_vtk(tmp_path):
    points, cells = quadrado_dois_triangulos()
    point_data_a = {"vx": np.array([0.0, 1.0, 1.0, 0.0]), "vy": np.array([0.0, 0.0, 0.5, 0.5]), "p": np.array([1.0, 0.5, 0.0, 0.5])}
    point_data_b = {"vx": np.array([0.0, 2.0, 2.0, 0.0]), "vy": np.array([0.0, 0.0, 1.0, 1.0]), "p": np.array([2.0, 1.0, 0.0, 1.0])}

    vtk_a = tmp_path / "a" / "solucao -1.vtk"
    vtk_b = tmp_path / "b" / "solucao -1.vtk"
    write_vtk(str(vtk_a), points, cells, point_data_a)
    write_vtk(str(vtk_b), points, cells, point_data_b)

    caminhos = plot_comparacao_ultimas_solucoes(str(vtk_a), str(vtk_b), "mini", "tri6", str(tmp_path / "imagens"))

    assert len(caminhos) == 4
    assert all(os.path.exists(c) for c in caminhos)


def test_ultimo_vtk_acha_maior_numero_de_iteracao(tmp_path):
    for n in [1, 2, 10, 9]:
        (tmp_path / f"solucao -{n}.vtk").write_text("")
    (tmp_path / "CondicaoDeContorno.vtk").write_text("")  # nao deve ser confundido com solucao

    assert ultimo_vtk(str(tmp_path)) == str(tmp_path / "solucao -10.vtk")


def test_ultimo_vtk_sem_arquivos_leva_erro_claro(tmp_path):
    try:
        ultimo_vtk(str(tmp_path))
        assert False, "deveria ter levantado FileNotFoundError"
    except FileNotFoundError:
        pass
