#!/usr/bin/env python3
"""CLI: benchmark so de geometria do movimento de malha da valvula (plano 12/10, §7.1).

Para cada configuracao do movimento (`femns.movimento_malha`), fecha a valvula
sem resolver o escoamento e registra, por angulo: qualidade minima e media,
elementos invertidos, area total, gradacao perto da esfera (`h/h0`, 1 = a da
malha inicial) e o tempo por passo de malha.

Duas rodadas por configuracao:

- **lei**: `theta0 -> theta_fim` com a lei temporal da simulacao
  (`valvula_rotacao.theta_em`) em `--passos` passos -- o mesmo `dTheta` por
  passo que a simulacao vai ver;
- **alcance**: `theta0 -> --theta-max` em passos fixos de `--passo-alcance`
  graus, ate inverter ou chegar ao fim.

Exemplo::

    python scripts/benchmark_movimento_malha.py --malha meshes/valvula_10g.msh --L 2.0
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from timeit import default_timer as timer

import numpy as np

import femns.movimento_malha as mm
import femns.valvula_rotacao as vr
from femns.mesh import read_mesh

CONFIGS = {
    "E0 extensao harmonica (controle)": {"extensao": {"a3": 1.0, "chi": 2.0}},
    "P1 centroide uniforme": {"suavizacao": {"pesos": "uniforme", "a2": 0.5, "varreduras": 3}},
    "P2 valor medio": {"suavizacao": {"pesos": "valor_medio", "a2": 0.5, "varreduras": 3}},
    "P3 radial (r, 1.3r) + valor medio": {"rotacao": {"tipo": "radial", "rho1": 1.0, "rho2": 1.3},
                                          "suavizacao": {"pesos": "valor_medio", "a2": 0.5, "varreduras": 3}},
    "P3b radial (0.5r, r) + valor medio": {"rotacao": {"tipo": "radial", "rho1": 0.5, "rho2": 1.0},
                                           "suavizacao": {"pesos": "valor_medio", "a2": 0.5, "varreduras": 3}},
    "P4 P3b + arcos uniformes": {"rotacao": {"tipo": "radial", "rho1": 0.5, "rho2": 1.0},
                                 "suavizacao": {"pesos": "valor_medio", "a2": 0.5, "varreduras": 3,
                                                "a2_arco": 0.5, "pesos_arco": "uniforme"}},
    "P5 P3b + extensao": {"rotacao": {"tipo": "radial", "rho1": 0.5, "rho2": 1.0},
                          "suavizacao": {"pesos": "valor_medio", "a2": 0.5, "varreduras": 3},
                          "extensao": {"a3": 1.0, "chi": 2.0}},
    "P5b extensao + valor medio": {"suavizacao": {"pesos": "valor_medio", "a2": 0.5, "varreduras": 3},
                                   "extensao": {"a3": 1.0, "chi": 2.0}},
    "P6 IDW + valor medio": {"rotacao": {"tipo": "idw", "p": 3},
                             "suavizacao": {"pesos": "valor_medio", "a2": 0.5, "varreduras": 3}},
    "P7 ODT": {"suavizacao": {"pesos": "odt", "a2": 0.5, "varreduras": 3}},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--malha", default="meshes/valvula_10g.msh")
    p.add_argument("--H", type=float, default=0.5)
    p.add_argument("--L", type=float, default=2.0)
    p.add_argument("--theta0", type=float, default=10.0)
    p.add_argument("--theta-fim", type=float, default=20.0)
    p.add_argument("--passos", type=int, default=8000, help="passos da rodada 'lei' (os da simulacao)")
    p.add_argument("--theta-max", type=float, default=50.0)
    p.add_argument("--passo-alcance", type=float, default=0.01, help="graus por passo na rodada 'alcance'")
    p.add_argument("--configs", nargs="*", default=None, help="prefixos dos nomes (padrao: todas)")
    p.add_argument("--saida", default="solucoes/benchmark_movimento_malha")
    p.add_argument("--processos", type=int, default=4)
    return p.parse_args()


def rodar(args, nome, bloco):
    m = read_mesh(args.malha)
    th0 = np.radians(args.theta0)
    c = vr.preparar(m.IEN, m.X, m.Y, m.IENbound, m.IENboundElem, H=args.H, L=args.L, theta0=th0)
    raio = 1.5 * c.geo.r
    res = {"nome": nome, "bloco": bloco}

    # rodada 'lei'
    mv = mm.MovimentoValvula(c, m.IEN, m.X, m.Y, mm.opcoes_movimento(bloco))
    marcos = {15.0: None, 20.0: None}
    t0, inv = timer(), None
    for k in range(1, args.passos + 1):
        th = vr.theta_em(k / args.passos, th0, np.radians(args.theta_fim), 1.0)
        info = mv.passo(th)
        if info["invertidos"] and inv is None:
            inv = info["theta_deg"]
            break
        for alvo in marcos:
            if marcos[alvo] is None and info["theta_deg"] >= alvo - 1e-9:
                marcos[alvo] = dict(info, gradacao=mv.gradacao(raio))
    res["lei"] = dict(ms_por_passo=1e3 * (timer() - t0) / k, inverte_em=inv,
                      marcos={str(a): v for a, v in marcos.items()})

    # rodada 'alcance'
    mv = mm.MovimentoValvula(c, m.IEN, m.X, m.Y, mm.opcoes_movimento(bloco))
    limiares = {0.3: None, 0.2: None, 0.1: None}
    hist, inv = [], None
    for th in np.arange(args.theta0 + args.passo_alcance, args.theta_max + 1e-9, args.passo_alcance):
        info = mv.passo(np.radians(th))
        if abs(th - round(th)) < 1e-6:
            hist.append((round(th), info["q_min"], mv.gradacao(raio)))
        for q in limiares:
            if limiares[q] is None and info["q_min"] < q:
                limiares[q] = info["theta_deg"]
        if info["invertidos"]:
            inv = info["theta_deg"]
            break
    res["alcance"] = dict(qmin_abaixo_de={str(q): v for q, v in limiares.items()}, inverte_em=inv, historico=hist)
    return res


def main():
    args = parse_args()
    os.makedirs(args.saida, exist_ok=True)
    sel = {k: v for k, v in CONFIGS.items() if not args.configs or any(k.startswith(p) for p in args.configs)}
    with ProcessPoolExecutor(args.processos) as ex:
        resultados = list(ex.map(rodar, [args] * len(sel), sel.keys(), sel.values()))
    with open(os.path.join(args.saida, "resultados.json"), "w") as f:
        json.dump(resultados, f, indent=1)

    def f(v, fmt="{:.3f}"):
        return "—" if v is None else fmt.format(v)

    linhas = [f"# Movimento de malha, so geometria ({os.path.basename(args.malha)})", "",
              f"Rodada **lei**: {args.theta0}° → {args.theta_fim}° com a lei temporal da simulação em "
              f"{args.passos} passos. Rodada **alcance**: {args.theta0}° → {args.theta_max}° em passos de "
              f"{args.passo_alcance}°. Gradação = média de h/h0 em |p| < 1,5 r (1 = a da malha inicial).", "",
              "| configuração | q_min 15° | q_min 20° | q_méd 20° | gradação 20° | inverte (lei) | ms/passo "
              "| q_min<0,3 em | q_min<0,1 em | inverte (alcance) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in resultados:
        lei, alc = r["lei"], r["alcance"]
        m15, m20 = lei["marcos"]["15.0"] or {}, lei["marcos"]["20.0"] or {}
        linhas.append("| " + " | ".join([
            r["nome"], f(m15.get("q_min")), f(m20.get("q_min")), f(m20.get("q_med")), f(m20.get("gradacao")),
            f(lei["inverte_em"], "{:.2f}°"), f(lei["ms_por_passo"], "{:.1f}"),
            f(alc["qmin_abaixo_de"]["0.3"], "{:.2f}°"), f(alc["qmin_abaixo_de"]["0.1"], "{:.2f}°"),
            f(alc["inverte_em"], "{:.2f}°")]) + " |")
    texto = "\n".join(linhas) + "\n"
    with open(os.path.join(args.saida, "relatorio.md"), "w") as fh:
        fh.write(texto)
    print(texto)


if __name__ == "__main__":
    sys.exit(main())
