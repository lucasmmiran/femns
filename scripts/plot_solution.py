#!/usr/bin/env python3
"""CLI: gera imagens (PNG) da distribuicao das variaveis na ultima solucao de uma simulacao."""

import argparse
import os

import yaml

from femns.io import ultimo_frame
from femns.plotting import ler_solucao, plot_solucoes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/poiseuille.yaml", help="Caminho do arquivo de config yaml")
    parser.add_argument("--vtk", default=None, help="Caminho de um .vtk especifico (ignora --config/output_dir)")
    parser.add_argument("--output-dir", default=None,
                         help="Onde salvar as imagens (padrao: <output_dir da simulacao>/imagens)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.vtk:
        vtk_path = args.vtk
        output_dir = args.output_dir or os.path.join(os.path.dirname(vtk_path) or ".", "imagens")
    else:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        vtk_path = ultimo_frame(cfg["output_dir"])
        output_dir = args.output_dir or os.path.join(cfg["output_dir"], "imagens")

    caminhos = plot_solucoes([ler_solucao(vtk_path)], [None], output_dir)

    print(f"Ultima solucao: {vtk_path}")
    print("Imagens geradas:")
    for caminho in caminhos:
        print(f"  {caminho}")


if __name__ == "__main__":
    main()
