"""Geracao de imagens (PNG) da distribuicao espacial das variaveis de uma solucao."""

import os

import matplotlib

matplotlib.use("Agg")  # sem display -- roda em servidor/CI sem X11

import meshio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation

from femns.io import conectividade

# (chave em point_data, titulo) -- todas plotadas com o mesmo colormap/numero
# de cores (ver N_CORES/CMAP_PADRAO abaixo).
CAMPOS = [
    ("vx", "Velocidade vx"),
    ("vy", "Velocidade vy"),
    ("p", "Pressao"),
]

CMAP_PADRAO = "jet"
N_CORES = 26


def plot_paineis(triangs: list, valores: list, rotulos: list, titulo: str, path: str,
                 cmap: str = CMAP_PADRAO, n_cores: int = N_CORES, mostrar_malha: bool = True):
    """Plota o mesmo campo escalar de N solucoes em N paineis lado a lado e salva em `path` (PNG).

    A escala de cor e' **comum** aos paineis (`vmin`/`vmax` sobre todos os
    campos) -- senao a comparacao visual engana. Com N=1 sai a imagem simples
    de um campo (titulo no proprio eixo); com N>1, `rotulos` vira o titulo de
    cada painel e `titulo` o titulo geral. `n_cores` discretiza a faixa em
    `n_cores` bandas; `mostrar_malha` sobrepoe nos e arestas dos elementos.

    Os paineis precisam ter a mesma faixa de coordenadas para a comparacao
    fazer sentido (ex.: mesmo `.msh` de origem, elementos diferentes -- o
    `.vtk` de saida sempre usa so os vertices originais).
    """
    diretorio = os.path.dirname(path)
    if diretorio:
        os.makedirs(diretorio, exist_ok=True)

    vmin = float(min(np.min(v) for v in valores))
    vmax = float(max(np.max(v) for v in valores))
    levels = np.linspace(vmin, vmax, n_cores + 1) if vmax > vmin else n_cores

    n = len(triangs)
    fig, eixos = plt.subplots(1, n, figsize=(8, 6) if n == 1 else (7 * n, 6), sharex=True, sharey=True, squeeze=False)
    eixos = list(eixos[0])
    contorno = None
    for ax, triang, v, rotulo in zip(eixos, triangs, valores, rotulos):
        contorno = ax.tricontourf(triang, v, levels=levels, cmap=cmap)
        if mostrar_malha:
            ax.triplot(triang, color="black", linewidth=0.2, alpha=0.4)
            ax.plot(triang.x, triang.y, "o", color="black", markersize=1.5, alpha=0.4)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_title(titulo if n == 1 else rotulo)
    eixos[0].set_ylabel("y")

    if n == 1:
        fig.colorbar(contorno, ax=eixos[0], label=titulo)
        fig.tight_layout()
    else:
        fig.suptitle(titulo)
        fig.colorbar(contorno, ax=eixos, label=titulo, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ler_solucao(vtk_path: str):
    """`(points, cells, point_data)` de um `.vtk` de solucao (ver `io.write_vtk`)."""
    malha = meshio.read(vtk_path)
    return malha.points, malha.cells, malha.point_data


def plot_solucoes(solucoes: list, rotulos: list, output_dir: str) -> list[str]:
    """Plota vx, vy, p e |v| de N solucoes, um PNG por variavel (N paineis em cada).

    `solucoes` e' uma lista de `(points, cells, point_data)` no formato de
    `io.write_vtk` (`cells` pode trazer blocos de linha/vertice de contorno;
    so os triangulos sao usados). Com uma solucao os arquivos sao `vx.png`,
    ...; com mais de uma, `vx_comparacao.png`, ... Variavel ausente em alguma
    solucao e' pulada. Retorna os caminhos dos PNGs gerados.
    """
    triangs = [Triangulation(pts[:, 0], pts[:, 1], conectividade(cells, "triangle")) for pts, cells, _ in solucoes]
    dados = [pd for _, _, pd in solucoes]
    sufixo = "" if len(solucoes) == 1 else "_comparacao"

    campos = [(chave, titulo, [pd[chave] for pd in dados]) for chave, titulo in CAMPOS if all(chave in pd for pd in dados)]
    if all("vx" in pd and "vy" in pd for pd in dados):
        campos.append(("velocidade_magnitude", "Magnitude da velocidade |v|",
                       [np.sqrt(pd["vx"] ** 2 + pd["vy"] ** 2) for pd in dados]))

    caminhos = []
    for chave, titulo, valores in campos:
        path = os.path.join(output_dir, f"{chave}{sufixo}.png")
        plot_paineis(triangs, valores, rotulos, titulo, path)
        caminhos.append(path)
    return caminhos
