"""Introspeccao leve de malhas `.msh` para o formulario de nova simulacao (sem depender de torch)."""

import glob
import os

import meshio


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
        out.append({
            "name": os.path.basename(path),
            "path": path,
            "npoints": int(raw.points.shape[0]),
            "boundary_names": boundary_names,
        })
    return out


def _tem_celulas(raw: meshio.Mesh, nome_grupo: str) -> bool:
    """True se algum bloco de celula (line/vertex) do .msh referencia esse physical group.

    `field_data` do gmsh às vezes inclui o nome do dominio 2D (o proprio
    'triangle') junto dos nomes de contorno -- filtramos pra so sobrar o que
    de fato pode ser usado em `boundary.conditions` no config.
    """
    tag = raw.field_data[nome_grupo][0]
    for i, bloco in enumerate(raw.cells):
        if bloco.type not in ("line", "vertex"):
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
    boundNames = list(raw.field_data.keys())

    triangles = []
    for bloco in raw.cells:
        if bloco.type == "triangle":
            triangles.extend(bloco.data.tolist())

    boundaries: dict[str, list] = {}
    for i, bloco in enumerate(raw.cells):
        if bloco.type != "line":
            continue
        tags = raw.cell_data["gmsh:physical"][i] - 1
        for segmento, tag in zip(bloco.data.tolist(), tags):
            nome = boundNames[tag]
            boundaries.setdefault(nome, []).append(segmento)

    return {
        "points": raw.points[:, :2].tolist(),
        "triangles": triangles,
        "boundaries": boundaries,
    }
