#!/usr/bin/env python3
"""CLI: compara numericamente os campos da ultima solucao de duas simulacoes.

Complementa `scripts/plot_comparison.py` (que compara visualmente): aqui a
saida e' quantitativa -- por campo (`vx`, `vy`, `p`), a faixa de valores de
cada lado, a diferenca maxima em modulo, a diferenca relativa em norma L2 e
a correlacao. Pensado pra responder "as duas variantes convergem para a
mesma solucao?" quando so a imagem nao decide (ex.: `sl_boundary:
dirichlet` vs `intercept`, `element: mini` vs `tri6`).

Le o `.vtk` de maior numero de iteracao de cada `output_dir`, entao as duas
simulacoes precisam ter rodado o mesmo numero de passos na mesma malha pra
comparacao fazer sentido (isso e' checado).
"""

import argparse

import meshio
import numpy as np
import yaml

from femns.plotting import ultimo_vtk

CAMPOS = ("vx", "vy", "p")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-a", required=True, help="Config yaml da simulacao A")
    parser.add_argument("--config-b", required=True, help="Config yaml da simulacao B")
    parser.add_argument("--label-a", default="A", help="Rotulo da simulacao A")
    parser.add_argument("--label-b", default="B", help="Rotulo da simulacao B")
    return parser.parse_args()


def comparar(a: np.ndarray, b: np.ndarray) -> dict:
    """Metricas de discrepancia entre dois campos nodais do mesmo tamanho.

    `l2_rel` e normalizada pela norma de `a`, entao e' adimensional e
    comparavel entre campos de escalas diferentes (velocidade ~1,
    pressao ~100 no degrau). `corr` e o coeficiente de Pearson; fica
    `nan` se um dos lados for constante (ex.: campo identicamente zero),
    caso em que nao ha correlacao definida -- e' informacao, nao erro.
    """
    dif = a - b
    norma_a = np.linalg.norm(a)
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan

    return {
        "max_abs": np.abs(dif).max(),
        "l2_rel": np.linalg.norm(dif) / norma_a if norma_a > 0 else np.nan,
        "corr": corr,
        "min_a": a.min(), "max_a": a.max(),
        "min_b": b.min(), "max_b": b.max(),
    }


def main():
    args = parse_args()

    with open(args.config_a) as f:
        cfg_a = yaml.safe_load(f)
    with open(args.config_b) as f:
        cfg_b = yaml.safe_load(f)

    vtk_a, vtk_b = ultimo_vtk(cfg_a["output_dir"]), ultimo_vtk(cfg_b["output_dir"])
    malha_a, malha_b = meshio.read(vtk_a), meshio.read(vtk_b)

    if malha_a.points.shape != malha_b.points.shape:
        raise ValueError(
            f"malhas de tamanhos diferentes ({malha_a.points.shape} vs {malha_b.points.shape}) -- "
            "a comparacao no a no exige a mesma malha nos dois lados")

    print(f"A ({args.label_a}): {vtk_a}")
    print(f"B ({args.label_b}): {vtk_b}")
    print(f"Nos: {malha_a.points.shape[0]}\n")

    cabecalho = f"{'campo':<6} {'faixa A':>22} {'faixa B':>22} {'max|A-B|':>11} {'L2 rel':>10} {'corr':>8}"
    print(cabecalho)
    print("-" * len(cabecalho))

    for campo in CAMPOS:
        if campo not in malha_a.point_data or campo not in malha_b.point_data:
            print(f"{campo:<6} (ausente em um dos .vtk -- pulado)")
            continue

        m = comparar(np.asarray(malha_a.point_data[campo]).ravel(),
                      np.asarray(malha_b.point_data[campo]).ravel())

        faixa_a = f"[{m['min_a']:.4g}, {m['max_a']:.4g}]"
        faixa_b = f"[{m['min_b']:.4g}, {m['max_b']:.4g}]"
        print(f"{campo:<6} {faixa_a:>22} {faixa_b:>22} "
              f"{m['max_abs']:>11.4e} {m['l2_rel']:>10.4e} {m['corr']:>8.5f}")


if __name__ == "__main__":
    main()
