import numpy as np
import pytest

from femns.io import write_vtk
from femns.webui import results


def quadrado_dois_triangulos():
    """Mesmo quadrado unitario de tests/test_plotting.py, em formato de ponto/celula VTK."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    cells = [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))]
    return points, cells


def escreve_frames(run_dir, numeros):
    points, cells = quadrado_dois_triangulos()
    for n in numeros:
        point_data = {
            "vx": np.array([0.0, 1.0, 1.0, 0.0]) * n,
            "vy": np.array([0.0, 0.0, 0.5, 0.5]),
            "p": np.array([1.0, 0.5, 0.0, 0.5]),
        }
        write_vtk(str(run_dir / f"solucao -{n}.vtk"), points, cells, point_data)


def test_scan_runs_acha_so_diretorios_com_frames(tmp_path):
    run_dir = tmp_path / "poiseuille"
    run_dir.mkdir()
    escreve_frames(run_dir, [1, 2, 3])
    (tmp_path / "sem_frames").mkdir()
    (tmp_path / "sem_frames" / "CondicaoDeContorno.vtk").write_text("x")

    runs = results.scan_runs(str(tmp_path))

    assert len(runs) == 1
    info = next(iter(runs.values()))
    assert info["dir"] == str(run_dir)
    assert info["label"] == "poiseuille"
    assert info["nframes"] == 3


def test_scan_runs_diretorio_inexistente_retorna_vazio(tmp_path):
    assert results.scan_runs(str(tmp_path / "nao_existe")) == {}


def test_scan_runs_ids_sao_estaveis_para_o_mesmo_path(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    escreve_frames(run_dir, [1])

    runs1 = results.scan_runs(str(tmp_path))
    runs2 = results.scan_runs(str(tmp_path))

    assert set(runs1.keys()) == set(runs2.keys())


def test_read_meta(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    escreve_frames(run_dir, [5, 10, 15])

    meta = results.read_meta(str(run_dir))

    assert meta["nframes"] == 3
    assert meta["first_frame"] == 5
    assert meta["last_frame"] == 15
    assert meta["frames"] == [5, 10, 15]
    assert meta["fields"] == ["vx", "vy", "p"]
    assert meta["npoints"] == 4
    assert meta["bbox"] == [0.0, 0.0, 1.0, 1.0]


def test_read_frame_retorna_pontos_triangulos_e_campos(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    escreve_frames(run_dir, [1, 2])

    frame = results.read_frame(str(run_dir), 2)

    assert frame["n"] == 2
    assert len(frame["points"]) == 4
    assert len(frame["triangles"]) == 2
    assert frame["vx"] == [0.0, 2.0, 2.0, 0.0]  # escrito como base * n, n=2


def test_read_frame_numero_inexistente_leva_filenotfounderror(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    escreve_frames(run_dir, [1])

    with pytest.raises(FileNotFoundError):
        results.read_frame(str(run_dir), 99)
