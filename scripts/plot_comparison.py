#!/usr/bin/env python3
"""CLI: gera imagens lado a lado comparando a ultima solucao de duas simulacoes.

Pensado pra comparar duas simulacoes da mesma malha (ex.: `simulation.element:
mini` vs `tri6`, ou com e sem `mesh_motion`) --
mesma escala de cor por variavel nos dois lados, pra comparacao visual direta.
"""

import argparse

import yaml

from femns.io import ultimo_frame
from femns.plotting import ler_solucao, plot_solucoes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-a", required=True, help="Config yaml da simulacao A")
    parser.add_argument("--config-b", required=True, help="Config yaml da simulacao B")
    parser.add_argument("--label-a", default=None, help="Rotulo da simulacao A (padrao: simulation.element ou 'A')")
    parser.add_argument("--label-b", default=None, help="Rotulo da simulacao B (padrao: simulation.element ou 'B')")
    parser.add_argument("--output-dir", default="solucoes/comparacao", help="Onde salvar as imagens")
    return parser.parse_args()


def _rotulo_padrao(cfg: dict, default: str) -> str:
    return cfg["simulation"].get("element", default)


def main():
    args = parse_args()

    with open(args.config_a) as f:
        cfg_a = yaml.safe_load(f)
    with open(args.config_b) as f:
        cfg_b = yaml.safe_load(f)

    vtk_a = ultimo_frame(cfg_a["output_dir"])
    vtk_b = ultimo_frame(cfg_b["output_dir"])
    label_a = args.label_a or _rotulo_padrao(cfg_a, "A")
    label_b = args.label_b or _rotulo_padrao(cfg_b, "B")

    caminhos = plot_solucoes([ler_solucao(vtk_a), ler_solucao(vtk_b)], [label_a, label_b], args.output_dir)

    print(f"Solucao A ({label_a}): {vtk_a}")
    print(f"Solucao B ({label_b}): {vtk_b}")
    print("Imagens geradas:")
    for caminho in caminhos:
        print(f"  {caminho}")


if __name__ == "__main__":
    main()
