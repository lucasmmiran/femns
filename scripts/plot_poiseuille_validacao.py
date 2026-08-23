#!/usr/bin/env python3
"""CLI: valida uma simulacao de Poiseuille contra a solucao analitica exata.

Gera dois graficos, numa secao transversal do canal:

1. `perfil_vx.png`  -- vx(y) numerico vs analitico
2. `erro_relativo.png` -- erro relativo de vx ao longo de y

A solucao exata do escoamento de Poiseuille desenvolvido entre placas e

    vx(y) = 4 * vmax * (y - y0) * (y1 - y) / (y1 - y0)^2,    vy = 0

com `vmax` o pico no centro do canal. Ela so vale como referencia ponto a
ponto se o **perfil ja for imposto na entrada** (`inlet: {vx: {perfil:
parabolico, ...}}`) -- com entrada uniforme existe uma regiao de
desenvolvimento perto de x=0 e a comparacao so faz sentido bem a jusante.
O script avisa se o config nao usa perfil na entrada.

O erro relativo e' normalizado por `vmax` (nao pelo valor local): perto das
paredes o valor exato tende a zero e um erro relativo local explodiria por
divisao por ~0, escondendo o comportamento no resto do canal.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import meshio
import numpy as np
import yaml
from matplotlib import pyplot as plt

from femns.plotting import ultimo_vtk


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Config yaml da simulacao a validar")
    parser.add_argument("--config-b", default=None,
                         help="Config de uma segunda simulacao, plotada junto (ex.: malha estatica vs ALE)")
    parser.add_argument("--label", default=None, help="Rotulo da simulacao principal")
    parser.add_argument("--label-b", default=None, help="Rotulo da segunda simulacao")
    parser.add_argument("--x-secao", type=float, default=None,
                         help="x da secao transversal analisada (padrao: meio do canal)")
    parser.add_argument("--tol", type=float, default=None,
                         help="Semi-largura da faixa em x que seleciona os nos da secao "
                              "(padrao: 1%% do comprimento do canal)")
    parser.add_argument("--output-dir", default=None, help="Onde salvar os PNGs (padrao: <output_dir>/validacao)")
    return parser.parse_args()


def vmax_do_config(cfg: dict) -> float:
    """Le `vmax` do perfil de entrada; avisa se a entrada nao for um perfil.

    Sem perfil na entrada a solucao analitica nao vale ponto a ponto no
    canal inteiro (ha regiao de desenvolvimento), entao o grafico de erro
    mediria sobretudo isso -- melhor dizer do que deixar interpretar
    errado. Nesse caso usa vmax = 1.5 * (velocidade uniforme), que e' o
    perfil desenvolvido de mesma vazao.
    """
    entrada = cfg["boundary"]["conditions"]["inlet"]["vx"]
    if isinstance(entrada, dict):
        return float(entrada["vmax"])

    print(f"AVISO: a entrada deste config e' uniforme (vx = {entrada}), nao um perfil. "
          f"Existe regiao de desenvolvimento, entao o erro medido inclui esse efeito -- "
          f"usando vmax = {1.5 * float(entrada)} (perfil de mesma vazao) como referencia.")
    return 1.5 * float(entrada)


def secao_transversal(vtk_path: str, x_secao: float = None, tol: float = None):
    """Extrai (y, vx) dos nos numa faixa estreita de x, ordenados por y.

    Seleciona por faixa (e nao por igualdade) porque a malha nao e'
    estruturada e, com ALE, os nos ainda saem da posicao de repouso -- nao
    existe uma coluna exata de nos num x fixo.
    """
    malha = meshio.read(vtk_path)
    x, y = malha.points[:, 0], malha.points[:, 1]
    vx = np.asarray(malha.point_data["vx"]).ravel()

    if x_secao is None:
        x_secao = 0.5 * (x.min() + x.max())
    if tol is None:
        tol = 0.01 * (x.max() - x.min())

    sel = np.abs(x - x_secao) <= tol
    if sel.sum() < 3:
        raise ValueError(
            f"so {sel.sum()} nos na secao x={x_secao} +- {tol} -- aumente --tol")

    ordem = np.argsort(y[sel])
    return y[sel][ordem], vx[sel][ordem], x_secao, tol


def analitico(y: np.ndarray, vmax: float, y0: float, y1: float) -> np.ndarray:
    return 4.0 * vmax * (y - y0) * (y1 - y) / (y1 - y0) ** 2


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    vmax = vmax_do_config(cfg)

    casos = [(args.label or "numerico", ultimo_vtk(cfg["output_dir"]))]
    if args.config_b:
        with open(args.config_b) as f:
            cfg_b = yaml.safe_load(f)
        casos.append((args.label_b or "numerico (B)", ultimo_vtk(cfg_b["output_dir"])))

    output_dir = args.output_dir or os.path.join(cfg["output_dir"], "validacao")
    os.makedirs(output_dir, exist_ok=True)

    fig_perfil, ax_perfil = plt.subplots(figsize=(7, 6))
    fig_erro, ax_erro = plt.subplots(figsize=(7, 6))

    y_ref = None
    for i, (label, vtk_path) in enumerate(casos):
        y, vx, x_secao, tol = secao_transversal(vtk_path, args.x_secao, args.tol)
        # As paredes do canal saem dos proprios nos da secao (mesma convencao
        # de boundary.perfil_parabolico), nao de uma altura fixa no codigo.
        y0, y1 = y.min(), y.max()
        exato = analitico(y, vmax, y0, y1)
        erro_rel = (vx - exato) / vmax

        if y_ref is None:
            y_ref = np.linspace(y0, y1, 400)
            ax_perfil.plot(analitico(y_ref, vmax, y0, y1), y_ref, "k-", lw=2, label="analitico  6y(1-y)")
            print(f"Secao: x = {x_secao:.4f} +- {tol:.4f}   |   {len(y)} nos   |   vmax = {vmax}")

        cor = f"C{i}"
        ax_perfil.plot(vx, y, "o", color=cor, ms=4, alpha=0.75, label=label)
        ax_erro.plot(erro_rel * 100.0, y, "o-", color=cor, ms=3, lw=1, alpha=0.8, label=label)

        print(f"  {label:28s} erro rel. max = {np.abs(erro_rel).max() * 100:7.3f}%   "
              f"L2 = {np.linalg.norm(erro_rel) / np.sqrt(len(erro_rel)) * 100:7.3f}%   "
              f"vx_pico = {vx.max():.5f} (exato {vmax})")

    ax_perfil.set_xlabel("vx")
    ax_perfil.set_ylabel("y")
    ax_perfil.set_title(f"Perfil de velocidade em x = {x_secao:.3f}")
    ax_perfil.grid(alpha=0.3)
    ax_perfil.legend()
    fig_perfil.tight_layout()
    caminho_perfil = os.path.join(output_dir, "perfil_vx.png")
    fig_perfil.savefig(caminho_perfil, dpi=150)

    ax_erro.axvline(0.0, color="black", lw=1)
    ax_erro.set_xlabel("erro relativo de vx  [% de vmax]")
    ax_erro.set_ylabel("y")
    ax_erro.set_title(f"Erro relativo vs solucao analitica (x = {x_secao:.3f})")
    ax_erro.grid(alpha=0.3)
    ax_erro.legend()
    fig_erro.tight_layout()
    caminho_erro = os.path.join(output_dir, "erro_relativo.png")
    fig_erro.savefig(caminho_erro, dpi=150)

    plt.close(fig_perfil)
    plt.close(fig_erro)

    print("\nImagens geradas:")
    for c in (caminho_perfil, caminho_erro):
        print(f"  {c}")


if __name__ == "__main__":
    main()
