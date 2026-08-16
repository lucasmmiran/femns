"""Geracao de imagens (PNG) da distribuicao espacial das variaveis de uma solucao."""

import glob
import os
import re

import matplotlib

matplotlib.use("Agg")  # sem display -- roda em servidor/CI sem X11

import meshio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation

# (chave em point_data, titulo) -- todas plotadas com o mesmo colormap/numero
# de cores (ver N_CORES/CMAP_PADRAO abaixo).
CAMPOS = [
    ("vx", "Velocidade vx"),
    ("vy", "Velocidade vy"),
    ("p", "Pressao"),
]

CMAP_PADRAO = "jet"
N_CORES = 26


def plot_campo(triang: Triangulation, valores: np.ndarray, titulo: str, path: str,
                cmap: str = CMAP_PADRAO, n_cores: int = N_CORES, mostrar_malha: bool = True):
    """Plota um campo escalar sobre a malha triangular (`tricontourf`) e salva em `path` (PNG).

    `n_cores` discretiza a faixa de valores em `n_cores` bandas de cor (em
    vez do gradiente continuo padrao do matplotlib). `mostrar_malha`
    sobrepoe os nos (marcadores) e arestas dos elementos (linhas finas).
    """
    diretorio = os.path.dirname(path)
    if diretorio:
        os.makedirs(diretorio, exist_ok=True)

    vmin, vmax = float(np.min(valores)), float(np.max(valores))
    levels = np.linspace(vmin, vmax, n_cores + 1) if vmax > vmin else n_cores

    fig, ax = plt.subplots(figsize=(8, 6))
    contorno = ax.tricontourf(triang, valores, levels=levels, cmap=cmap)
    if mostrar_malha:
        ax.triplot(triang, color="black", linewidth=0.2, alpha=0.4)
        ax.plot(triang.x, triang.y, "o", color="black", markersize=1.5, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(titulo)
    fig.colorbar(contorno, ax=ax, label=titulo)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _bloco_triangulos(cells):
    """Acha o bloco `triangle` em `cells` (lista de blocos meshio ou de tuplas (tipo, dados))."""
    for bloco in cells:
        tipo, dados = (bloco.type, bloco.data) if hasattr(bloco, "type") else bloco
        if tipo == "triangle":
            return dados
    raise ValueError("nenhum bloco 'triangle' encontrado em `cells`")


def plot_solucao(points: np.ndarray, cells, point_data: dict, output_dir: str) -> list[str]:
    """Plota vx, vy, p e a magnitude da velocidade sobre a malha, um PNG por variavel.

    `points`, `cells`, `point_data` no mesmo formato de `io.write_vtk` (`cells`
    pode incluir blocos de linha/vertice de contorno alem do `triangle` --
    so o `triangle` e usado aqui). Campos ausentes em `point_data` sao
    pulados (nao quebra se so um subconjunto das variaveis foi salvo).

    Retorna a lista de caminhos dos PNGs gerados.
    """
    triangulos = _bloco_triangulos(cells)
    triang = Triangulation(points[:, 0], points[:, 1], triangulos)

    caminhos = []
    for chave, titulo in CAMPOS:
        if chave not in point_data:
            continue
        path = os.path.join(output_dir, f"{chave}.png")
        plot_campo(triang, point_data[chave], titulo, path)
        caminhos.append(path)

    if "vx" in point_data and "vy" in point_data:
        vel = np.sqrt(point_data["vx"] ** 2 + point_data["vy"] ** 2)
        path = os.path.join(output_dir, "velocidade_magnitude.png")
        plot_campo(triang, vel, "Magnitude da velocidade |v|", path)
        caminhos.append(path)

    return caminhos


def plot_ultima_solucao(vtk_path: str, output_dir: str) -> list[str]:
    """Le um `.vtk` de solucao (ver `io.write_vtk`) e plota suas variaveis (ver `plot_solucao`)."""
    malha = meshio.read(vtk_path)
    return plot_solucao(malha.points, malha.cells, malha.point_data, output_dir)


def plot_campo_comparacao(triang_a: Triangulation, valores_a: np.ndarray, triang_b: Triangulation, valores_b: np.ndarray,
                           titulo: str, label_a: str, label_b: str, path: str,
                           cmap: str = CMAP_PADRAO, n_cores: int = N_CORES, mostrar_malha: bool = True):
    """Plota o mesmo campo escalar de duas solucoes lado a lado, na mesma escala de cor.

    As duas malhas precisam ter a mesma faixa de coordenadas pra fazer
    sentido comparar visualmente (ex.: mesmo `.msh` de origem, elementos
    de velocidade diferentes -- o `.vtk` de saida de `run_simulation.py`
    sempre usa so os vertices originais, ver `io.write_vtk`).
    """
    diretorio = os.path.dirname(path)
    if diretorio:
        os.makedirs(diretorio, exist_ok=True)

    vmin = float(min(np.min(valores_a), np.min(valores_b)))
    vmax = float(max(np.max(valores_a), np.max(valores_b)))
    levels = np.linspace(vmin, vmax, n_cores + 1) if vmax > vmin else n_cores

    fig, eixos = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    contorno = None
    for ax, triang, valores, label in zip(eixos, (triang_a, triang_b), (valores_a, valores_b), (label_a, label_b)):
        contorno = ax.tricontourf(triang, valores, levels=levels, cmap=cmap)
        if mostrar_malha:
            ax.triplot(triang, color="black", linewidth=0.2, alpha=0.4)
            ax.plot(triang.x, triang.y, "o", color="black", markersize=1.5, alpha=0.4)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_title(label)
    eixos[0].set_ylabel("y")
    fig.suptitle(titulo)
    fig.colorbar(contorno, ax=list(eixos), label=titulo, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparacao_solucao(points_a: np.ndarray, cells_a, point_data_a: dict,
                             points_b: np.ndarray, cells_b, point_data_b: dict,
                             label_a: str, label_b: str, output_dir: str) -> list[str]:
    """Plota vx, vy, p e a magnitude da velocidade de duas solucoes lado a lado, mesma escala de cor por variavel.

    Pensado pra comparar duas simulacoes da mesma malha com elementos de
    velocidade diferentes (ex. MINI vs Tri6, ver `femns.assembly`) -- o
    `.vtk` de saida sempre usa so os vertices originais, independente do
    elemento usado internamente, entao as duas malhas batem.

    Retorna a lista de caminhos dos PNGs gerados.
    """
    triang_a = Triangulation(points_a[:, 0], points_a[:, 1], _bloco_triangulos(cells_a))
    triang_b = Triangulation(points_b[:, 0], points_b[:, 1], _bloco_triangulos(cells_b))

    caminhos = []
    for chave, titulo in CAMPOS:
        if chave not in point_data_a or chave not in point_data_b:
            continue
        path = os.path.join(output_dir, f"{chave}_comparacao.png")
        plot_campo_comparacao(triang_a, point_data_a[chave], triang_b, point_data_b[chave], titulo, label_a, label_b, path)
        caminhos.append(path)

    if all(k in point_data_a for k in ("vx", "vy")) and all(k in point_data_b for k in ("vx", "vy")):
        vel_a = np.sqrt(point_data_a["vx"] ** 2 + point_data_a["vy"] ** 2)
        vel_b = np.sqrt(point_data_b["vx"] ** 2 + point_data_b["vy"] ** 2)
        path = os.path.join(output_dir, "velocidade_magnitude_comparacao.png")
        plot_campo_comparacao(triang_a, vel_a, triang_b, vel_b, "Magnitude da velocidade |v|", label_a, label_b, path)
        caminhos.append(path)

    return caminhos


def plot_comparacao_ultimas_solucoes(vtk_path_a: str, vtk_path_b: str, label_a: str, label_b: str, output_dir: str) -> list[str]:
    """Le dois `.vtk` de solucao e plota suas variaveis lado a lado (ver `plot_comparacao_solucao`)."""
    malha_a = meshio.read(vtk_path_a)
    malha_b = meshio.read(vtk_path_b)
    return plot_comparacao_solucao(malha_a.points, malha_a.cells, malha_a.point_data,
                                    malha_b.points, malha_b.cells, malha_b.point_data,
                                    label_a, label_b, output_dir)


def ultimo_vtk(output_dir: str) -> str:
    """Acha, em `output_dir`, o `solucao -N.vtk` com o maior N (ultimo passo de tempo escrito)."""
    candidatos = glob.glob(os.path.join(output_dir, "solucao -*.vtk"))
    if not candidatos:
        raise FileNotFoundError(f"nenhum 'solucao -*.vtk' encontrado em {output_dir!r} -- rode a simulacao primeiro")

    def numero_iter(caminho):
        return int(re.search(r"solucao -(\d+)\.vtk$", caminho).group(1))

    return max(candidatos, key=numero_iter)
