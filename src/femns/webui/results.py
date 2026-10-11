"""Descoberta de runs em `solucoes/` e leitura de frames `.vtk` para JSON (para o visualizador web).

Isolado de `server.py` para poder testar sem subir um servidor HTTP: dado um
diretorio raiz, `scan_runs` acha os runs e `read_frame`/`read_meta` leem os
`.vtk` escritos por `femns.io.write_vtk` (mesmo formato usado por
`femns.plotting`).
"""

import hashlib
import os

import meshio

from femns.io import conectividade, listar_frames, numero_frame

CAMPOS = ("vx", "vy", "p")


def _run_id(path: str) -> str:
    """Id opaco e estavel para um diretorio de run -- evita expor/aceitar paths do cliente."""
    return hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:16]


def scan_runs(root: str) -> dict[str, dict]:
    """Varre `root` recursivamente por diretorios com frames `solucao-N.vtk` (ver `femns.io.listar_frames`).

    Retorna {id: {"dir": path, "label": nome relativo, "nframes": int}}.
    """
    runs = {}
    if not os.path.isdir(root):
        return runs
    for dirpath, _dirnames, filenames in os.walk(root):
        frames = [f for f in filenames if numero_frame(f) is not None]
        if not frames:
            continue
        rid = _run_id(dirpath)
        runs[rid] = {
            "dir": dirpath,
            "label": os.path.relpath(dirpath, root).replace(os.sep, "/"),
            "nframes": len(frames),
        }
    return runs


def _frame_path(run_dir: str, n: int) -> str:
    frames = listar_frames(run_dir)
    if n not in frames:
        raise FileNotFoundError(f"frame {n} nao encontrado em {run_dir!r} (disponiveis: {sorted(frames)[:5]}...)")
    return frames[n]


def frame_numbers(run_dir: str) -> list[int]:
    return list(listar_frames(run_dir))


def read_meta(run_dir: str) -> dict:
    """Metadados de um run: numero de frames, campos disponiveis e bbox (do primeiro frame)."""
    numeros = frame_numbers(run_dir)
    if not numeros:
        raise FileNotFoundError(f"nenhum frame em {run_dir!r}")
    malha = meshio.read(_frame_path(run_dir, numeros[0]))
    campos = [c for c in CAMPOS if c in malha.point_data]
    x, y = malha.points[:, 0], malha.points[:, 1]
    return {
        "nframes": len(numeros),
        "first_frame": numeros[0],
        "last_frame": numeros[-1],
        # lista dos numeros de frame REAIS -- com `vtk_interval > 1` eles nao
        # sao contiguos (ex.: 20, 40, ..., 1000), entao o cliente nao pode
        # reconstruir o intervalo por conta propria.
        "frames": numeros,
        "fields": campos,
        "bbox": [float(x.min()), float(y.min()), float(x.max()), float(y.max())],
        "npoints": int(malha.points.shape[0]),
    }


def read_frame(run_dir: str, n: int) -> dict:
    """Le o frame `n` e retorna um dict pronto pra serializar em JSON.

    `points` como lista de [x, y], `triangles` como lista de [i, j, k],
    campos escalares (`vx`, `vy`, `p`, quando presentes) como listas de float.
    """
    malha = meshio.read(_frame_path(run_dir, n))
    triangulos = conectividade(malha.cells, "triangle")
    pontos = malha.points[:, :2]

    out = {
        "n": n,
        "points": pontos.tolist(),
        "triangles": triangulos.tolist(),
    }
    for campo in CAMPOS:
        if campo in malha.point_data:
            out[campo] = malha.point_data[campo].tolist()
    return out
