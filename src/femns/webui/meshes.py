"""Introspeccao leve de malhas `.msh` para o formulario de nova simulacao (sem depender de torch)."""

import glob
import os

import meshio
import numpy as np

from femns.io import blocos_do_tipo, conectividade
from femns.mesh import DIMENSAO_DO_BLOCO, nome_por_tag


def list_meshes(root: str) -> list[dict]:
    """Lista `*.msh` em `root` com o nome de cada contorno (physical group) definido na malha.

    So le geometria/physical groups via `meshio` (nao monta elemento MINI/Tri6
    nem conectividade -- ver `femns.mesh.read_mesh` pra isso), entao e barato
    o bastante pra rodar a cada carregamento da tela de nova simulacao.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(root, "*.msh"))):
        raw = meshio.read(path)
        boundary_names = [nome for nome in raw.field_data if _tem_celulas(raw, nome)]
        triangulos = conectividade(raw.cells, "triangle")
        out.append({
            "name": os.path.basename(path),
            "path": path,
            "npoints": int(raw.points.shape[0]),
            # elementos e arestas: o front calcula os nos de cada elemento
            # (MINI = vertices + elementos; Tri6 = vertices + arestas)
            "nelements": int(triangulos.shape[0]),
            "nedges": contar_arestas(triangulos),
            "boundary_names": boundary_names,
        })
    return out


def contar_arestas(triangulos: np.ndarray) -> int:
    """Arestas distintas de uma triangulacao (cada aresta interna e' compartilhada por 2 triangulos)."""
    if len(triangulos) == 0:
        return 0
    arestas = np.concatenate([triangulos[:, [0, 1]], triangulos[:, [1, 2]], triangulos[:, [2, 0]]])
    return int(np.unique(np.sort(arestas, axis=1), axis=0).shape[0])


def _tem_celulas(raw: meshio.Mesh, nome_grupo: str) -> bool:
    """True se algum bloco de celula (line/vertex) do .msh referencia esse physical group.

    `field_data` do gmsh às vezes inclui o nome do dominio 2D (o proprio
    'triangle') junto dos nomes de contorno -- filtramos pra so sobrar o que
    de fato pode ser usado em `boundary.conditions` no config.
    """
    tag, dim = raw.field_data[nome_grupo]
    for i, bloco in enumerate(raw.cells):
        if bloco.type not in ("line", "vertex") or DIMENSAO_DO_BLOCO[bloco.type] != dim:
            continue
        if (raw.cell_data["gmsh:physical"][i] == tag).any():
            return True
    return False


def mesh_geometry(path: str) -> dict:
    """Geometria de uma malha pra pre-visualizacao na tela de nova simulacao (so vertices, sem elemento MINI/Tri6).

    Retorna `points` (lista de [x, y]), `triangles` (lista de [i, j, k]) e
    `boundaries` (nome do contorno -> lista de segmentos [i, j], arestas do
    bloco `line` daquele physical group) -- o frontend desenha o wireframe
    da malha com os contornos coloridos por cima, pra correlacionar
    visualmente com o editor de `boundary.conditions` logo abaixo.
    """
    raw = meshio.read(path)
    por_tag = nome_por_tag(raw)

    triangles = conectividade(raw.cells, "triangle").tolist()

    boundaries: dict[str, list] = {}
    for i in blocos_do_tipo(raw.cells, "line"):
        tags = raw.cell_data["gmsh:physical"][i]
        for segmento, tag in zip(raw.cells[i].data.tolist(), tags):
            nome = por_tag[(1, int(tag))]
            boundaries.setdefault(nome, []).append(segmento)

    return {
        "points": raw.points[:, :2].tolist(),
        "triangles": triangles,
        "boundaries": boundaries,
    }
