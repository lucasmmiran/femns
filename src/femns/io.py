"""Arquivos da simulacao: escrita do VTK, nome dos frames e blocos de celulas do meshio.

E' o unico lugar que sabe o nome dos arquivos de frame (`solucao-N.vtk`) e
como achar os blocos de um tipo (`triangle`, `line`, `vertex`) numa lista de
celulas do meshio -- o solver, a geracao de imagens, os scripts e a GUI
chamam estas funcoes em vez de repetir glob/regex/laco.

Sem torch: importado tambem pelo servidor da GUI.
"""

import os
import re

import meshio
import numpy as np

# Nome gravado a partir da v2.0. A v1.0 gravava `solucao -N.vtk` (com espaco
# antes do hifen); a leitura aceita os dois, para nao perder resultados antigos.
_FRAME = "solucao-{n}.vtk"
_FRAME_RE = re.compile(r"^solucao ?-(\d+)\.vtk$")


def write_vtk(path: str, points: np.ndarray, cells, point_data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meshio.write_points_cells(path, points, cells, point_data=point_data)


# -- frames -----------------------------------------------------------------

def caminho_frame(output_dir: str, n: int) -> str:
    """Caminho do frame da iteracao `n` (`<output_dir>/solucao-N.vtk`)."""
    return os.path.join(output_dir, _FRAME.format(n=n))


def numero_frame(nome: str):
    """Numero da iteracao de um nome de arquivo de frame, ou None se nao for um frame."""
    m = _FRAME_RE.match(os.path.basename(nome))
    return int(m.group(1)) if m else None


def listar_frames(output_dir: str) -> dict[int, str]:
    """`{N: caminho}` de todos os frames em `output_dir`, em ordem crescente de N.

    Diretorio inexistente da `{}`. Se um mesmo N existir nos dois formatos
    (antigo e novo), vale o novo.
    """
    if not os.path.isdir(output_dir):
        return {}
    frames = {}
    for nome in sorted(os.listdir(output_dir)):  # "solucao -" < "solucao-": o novo sobrescreve
        n = numero_frame(nome)
        if n is not None:
            frames[n] = os.path.join(output_dir, nome)
    return dict(sorted(frames.items()))


def ultimo_frame(output_dir: str) -> str:
    """O frame de maior N em `output_dir` (ultimo passo de tempo escrito)."""
    frames = listar_frames(output_dir)
    if not frames:
        raise FileNotFoundError(f"nenhum 'solucao-*.vtk' encontrado em {output_dir!r} -- rode a simulacao primeiro")
    return frames[max(frames)]


# -- blocos de celulas do meshio --------------------------------------------

def _tipo_e_dados(bloco):
    """Aceita `meshio.CellBlock` ou a tupla `(tipo, dados)` (formato de `write_vtk`)."""
    return (bloco.type, bloco.data) if hasattr(bloco, "type") else bloco


def blocos_do_tipo(cells, tipo: str) -> list[int]:
    """Indices, na lista `cells`, dos blocos de um `tipo` (ex. 'triangle').

    Malhas do Gmsh podem ter blocos em qualquer ordem/quantidade -- ex.
    `lid.msh` tem um bloco `vertex` (no de referencia de pressao) antes do
    `line`, e uma malha com varias superficies fisicas tem varios blocos
    `triangle`. Por isso nunca se usa indice fixo (`cells[0]`).
    """
    return [i for i, bloco in enumerate(cells) if _tipo_e_dados(bloco)[0] == tipo]


def conectividade(cells, tipo: str, obrigatorio: bool = True) -> np.ndarray:
    """Conectividade de **todos** os blocos de um `tipo`, concatenados na ordem de `cells`.

    Com `obrigatorio=False`, a falta do tipo devolve um array vazio
    `(0, nos_por_celula)` em vez de erro (ex.: malha sem bloco `vertex`).
    """
    blocos = [np.asarray(_tipo_e_dados(cells[i])[1]) for i in blocos_do_tipo(cells, tipo)]
    if blocos:
        return np.concatenate(blocos, axis=0)
    if obrigatorio:
        raise ValueError(f"nenhum bloco {tipo!r} encontrado nas celulas")
    nos_por_celula = {"vertex": 1, "line": 2, "triangle": 3}[tipo]
    return np.empty((0, nos_por_celula), dtype=int)
