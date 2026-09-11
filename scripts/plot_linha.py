#!/usr/bin/env python3
"""CLI: amostra um campo ao longo de uma linha reta sobre a solucao final e
grava um `.json` no formato que a GUI (`plots/*.json`) carrega -- pra comparar
com dados de referencia na aba Resultados.

Replica a `sampleLine`/`interpolateFieldAt` do frontend: localiza o triangulo
que contem cada ponto de amostragem (busca linear) e interpola o valor nodal
por coordenadas baricentricas (mesma logica do sombreamento suave do
visualizador). Pontos fora da malha entram no `.json` como `null`.

Exemplo::

    python scripts/plot_linha.py --config configs/gui/....yaml \\
        --campo vx --x 7 --y0 -0.5 --y1 0.5 --nome degrau-x7 --out plots/degrau-x7.json
"""

import argparse
import datetime
import json

import meshio
import numpy as np
import yaml

from femns.plotting import ultimo_vtk


def _triangulos(malha) -> np.ndarray:
    for cb in malha.cells:
        if cb.type == "triangle":
            return cb.data
    raise ValueError("malha sem celulas 'triangle'")


def interpolar_no_ponto(x, y, pts, tris, valores):
    """Valor nodal interpolado em (x, y) por coordenadas baricentricas; None se
    o ponto cai fora da malha. Mesmo criterio do frontend (tolerancia -1e-9)."""
    px, py = pts[:, 0], pts[:, 1]
    for a, b, c in tris:
        xa, ya = px[a], py[a]
        xb, yb = px[b], py[b]
        xc, yc = px[c], py[c]
        if x < min(xa, xb, xc) or x > max(xa, xb, xc):
            continue
        if y < min(ya, yb, yc) or y > max(ya, yb, yc):
            continue
        denom = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
        if denom == 0.0:
            continue
        wa = ((yb - yc) * (x - xc) + (xc - xb) * (y - yc)) / denom
        wb = ((yc - ya) * (x - xc) + (xa - xc) * (y - yc)) / denom
        wc = 1.0 - wa - wb
        if wa < -1e-9 or wb < -1e-9 or wc < -1e-9:
            continue
        return float(wa * valores[a] + wb * valores[b] + wc * valores[c])
    return None


def amostrar_linha(pts, tris, valores, x1, y1, x2, y2, n):
    comp = float(np.hypot(x2 - x1, y2 - y1))
    amostras = []
    for i in range(n):
        t = 0.0 if n == 1 else i / (n - 1)
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        amostras.append({"dist": t * comp, "x": x, "y": y,
                         "value": interpolar_no_ponto(x, y, pts, tris, valores)})
    return amostras, comp


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Config yaml da simulacao (usa output_dir)")
    p.add_argument("--vtk", default=None, help="VTK especifico (default: ultimo passo de output_dir)")
    p.add_argument("--campo", default="vx", choices=["vx", "vy", "p", "mag"])
    p.add_argument("--x", type=float, help="x fixo (linha vertical)")
    p.add_argument("--y", type=float, help="y fixo (linha horizontal)")
    p.add_argument("--y0", type=float, help="y inicial da linha vertical")
    p.add_argument("--y1", type=float, help="y final da linha vertical")
    p.add_argument("--x0", type=float, help="x inicial da linha horizontal")
    p.add_argument("--x1", type=float, help="x final da linha horizontal")
    p.add_argument("--n", type=int, default=200, help="pontos de amostragem (default 200, = GUI)")
    p.add_argument("--nome", required=True, help="nome da plotagem (campo 'name' do json)")
    p.add_argument("--rotulo", default=None, help="rotulo (campo 'run_label')")
    p.add_argument("--cor", default="#4f8cff")
    p.add_argument("--out", required=True, help="caminho do .json de saida")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    vtk = args.vtk or ultimo_vtk(cfg["output_dir"])
    malha = meshio.read(vtk)
    pts = malha.points[:, :2]
    tris = _triangulos(malha)

    if args.campo == "mag":
        valores = np.hypot(malha.point_data["vx"], malha.point_data["vy"])
    else:
        valores = malha.point_data[args.campo]

    if args.x is not None:
        x1, y1, x2, y2 = args.x, args.y0, args.x, args.y1
    elif args.y is not None:
        x1, y1, x2, y2 = args.x0, args.y, args.x1, args.y
    else:
        raise SystemExit("informe --x (linha vertical) ou --y (linha horizontal)")

    amostras, comp = amostrar_linha(pts, tris, valores, x1, y1, x2, y2, args.n)
    n_validos = sum(1 for a in amostras if a["value"] is not None)

    frame = int(vtk.rsplit("-", 1)[-1].split(".")[0])
    doc = {
        "name": args.nome,
        "campo": "|v|" if args.campo == "mag" else args.campo,
        "color": args.cor,
        "dash": "solid",
        "marker": "none",
        "marker_count": 10,
        "line": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "length": comp,
        "samples": amostras,
        "run_label": args.rotulo or args.nome,
        "frame": frame,
        "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "fonte": (f"Amostrado de {vtk} (passo {frame}) por scripts/plot_linha.py -- "
                  f"interpolacao baricentrica nodal, {n_validos}/{args.n} pontos dentro da malha."),
    }
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"{args.out}  ({n_validos}/{args.n} pontos na malha, passo {frame})")


if __name__ == "__main__":
    main()
