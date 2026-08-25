"""Descoberta de runs em `solucoes/` e leitura de frames `.vtk` para JSON (para o visualizador web).

Isolado de `server.py` para poder testar sem subir um servidor HTTP: dado um
diretorio raiz, `scan_runs` acha os runs e `read_frame`/`read_meta` leem os
`.vtk` escritos por `femns.io.write_vtk` (mesmo formato usado por
`femns.plotting`).
"""

import glob
import hashlib
import os
import re

import meshio
import numpy as np

FRAME_RE = re.compile(r"solucao -(\d+)\.vtk$")

CAMPOS = ("vx", "vy", "p")


def _run_id(path: str) -> str:
    """Id opaco e estavel para um diretorio de run -- evita expor/aceitar paths do cliente."""
    return hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:16]


def scan_runs(root: str) -> dict[str, dict]:
    """Varre `root` recursivamente por diretorios com `solucao -N.vtk` (ver `femns.plotting.ultimo_vtk`).

    Retorna {id: {"dir": path, "label": nome relativo, "nframes": int}}.
    """
    runs = {}
    if not os.path.isdir(root):
        return runs
    for dirpath, _dirnames, filenames in os.walk(root):
        frames = [f for f in filenames if FRAME_RE.search(f)]
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
    candidatos = glob.glob(os.path.join(run_dir, "solucao -*.vtk"))
    por_numero = {int(FRAME_RE.search(c).group(1)): c for c in candidatos}
    if n not in por_numero:
        raise FileNotFoundError(f"frame {n} nao encontrado em {run_dir!r} (disponiveis: {sorted(por_numero)[:5]}...)")
    return por_numero[n]


def frame_numbers(run_dir: str) -> list[int]:
    candidatos = glob.glob(os.path.join(run_dir, "solucao -*.vtk"))
    return sorted(int(FRAME_RE.search(c).group(1)) for c in candidatos)


def _triangle_cells(cells) -> np.ndarray:
    for bloco in cells:
        tipo, dados = (bloco.type, bloco.data) if hasattr(bloco, "type") else bloco
        if tipo == "triangle":
            return np.asarray(dados)
    raise ValueError("nenhum bloco 'triangle' encontrado no .vtk")


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
    triangulos = _triangle_cells(malha.cells)
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
