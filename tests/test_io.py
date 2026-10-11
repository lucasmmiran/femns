import meshio
import numpy as np
import pytest

from femns.io import (
    blocos_do_tipo,
    caminho_frame,
    conectividade,
    listar_frames,
    numero_frame,
    ultimo_frame,
)


def test_caminho_frame_sem_espaco_antes_do_hifen(tmp_path):
    assert caminho_frame(str(tmp_path), 7) == str(tmp_path / "solucao-7.vtk")


def test_numero_frame_aceita_nome_novo_e_antigo():
    assert numero_frame("solucao-12.vtk") == 12
    assert numero_frame("/algum/dir/solucao -12.vtk") == 12  # nome da v1.0
    assert numero_frame("CondicaoDeContorno.vtk") is None
    assert numero_frame("solucao-12.vtk.bak") is None


def test_listar_frames_ordena_por_numero_e_mistura_formatos(tmp_path):
    for n in (1, 10):
        (tmp_path / f"solucao-{n}.vtk").write_text("")
    for n in (2, 9):
        (tmp_path / f"solucao -{n}.vtk").write_text("")  # run antigo
    (tmp_path / "CondicaoDeContorno.vtk").write_text("")

    frames = listar_frames(str(tmp_path))

    assert list(frames) == [1, 2, 9, 10]
    assert frames[2].endswith("solucao -2.vtk")


def test_listar_frames_mesmo_numero_nos_dois_formatos_vale_o_novo(tmp_path):
    (tmp_path / "solucao -5.vtk").write_text("")
    (tmp_path / "solucao-5.vtk").write_text("")
    assert listar_frames(str(tmp_path))[5].endswith("solucao-5.vtk")


def test_listar_frames_diretorio_inexistente(tmp_path):
    assert listar_frames(str(tmp_path / "nao_existe")) == {}


def test_ultimo_frame(tmp_path):
    for n in (1, 2, 10, 9):
        (tmp_path / f"solucao-{n}.vtk").write_text("")
    assert ultimo_frame(str(tmp_path)) == str(tmp_path / "solucao-10.vtk")
    with pytest.raises(FileNotFoundError):
        ultimo_frame(str(tmp_path / "vazio"))


def test_conectividade_concatena_todos_os_blocos_do_tipo():
    """Malha com mais de uma superficie fisica (ex.: valvula com tubo/arco/corda)
    tem varios blocos `triangle` -- todos entram, na ordem em que aparecem."""
    cells = [
        meshio.CellBlock("line", np.array([[0, 1]])),
        meshio.CellBlock("triangle", np.array([[0, 1, 2]])),
        meshio.CellBlock("vertex", np.array([[3]])),
        meshio.CellBlock("triangle", np.array([[1, 2, 3], [2, 3, 4]])),
    ]
    assert blocos_do_tipo(cells, "triangle") == [1, 3]
    assert conectividade(cells, "triangle").tolist() == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    assert conectividade(cells, "vertex").tolist() == [[3]]


def test_conectividade_aceita_tuplas_e_tipo_ausente():
    cells = [("triangle", np.array([[0, 1, 2]]))]  # formato de write_vtk
    assert conectividade(cells, "triangle").tolist() == [[0, 1, 2]]
    assert conectividade(cells, "line", obrigatorio=False).shape == (0, 2)
    with pytest.raises(ValueError):
        conectividade(cells, "line")
