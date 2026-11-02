#!/usr/bin/env python3
"""CLI: teste de regressao do solver no Poiseuille com malha movel (`configs/poiseuille_test.yaml`).

Para mudancas que nao dao resultado bit a bit igual (passos de Chebyshev,
criterio de parada, outro pre-condicionador...): em vez de exigir igualdade,
mede quanto a solucao se afastou da referencia gravada e se o erro contra a
solucao exata piorou.

Tres subcomandos:

    rodar      roda a simulacao numa pasta propria e extrai o resultado
               (`resultado.npz` com os frames, `resultado.json` com o resumo)
    referencia copia o resultado de um run para `benchmarks/poiseuille_test/`
    comparar   compara um run contra a referencia gravada

Exemplo (validar outros passos de Chebyshev):

    python scripts/teste_poiseuille.py rodar --saida solucoes/pt_cheb3 \\
        --set simulation.solver.chebyshev.velocidade=3
    python scripts/teste_poiseuille.py comparar --saida solucoes/pt_cheb3

Solucao exata (perfil imposto na entrada, canal de altura 1, vazao 1):
`vx = 6 y (1 - y)`, `vy = 0`, `dp/dx = -12 / Re`. O canal e' longo (7) e a
saida e' livre, entao a queda de pressao e' medida pela inclinacao do ajuste
linear de `p(x)` na metade central do canal, longe da entrada e da saida.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

import meshio
import numpy as np
import yaml

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(RAIZ, "configs", "poiseuille_test.yaml")
REFERENCIA = os.path.join(RAIZ, "benchmarks", "poiseuille_test")
ARQUIVOS_DO_RUN = ("resultado.npz", "resultado.json", "historico.csv", "config.yaml")

# Criterios do `comparar` (diferenca relativa a escala do campo na referencia).
# O MINRES para em 1e-8 de residuo por passo; o que sobra entre dois solvers
# corretos e' esse erro acumulado ao longo dos passos, bem abaixo de 1e-5.
LIMITE_CAMPO = 1e-5
# o erro contra a solucao exata pode piorar no maximo isto (relativo)
LIMITE_PIORA_EXATA = 0.01


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("rodar", help="roda a simulacao e extrai o resultado")
    p.add_argument("--saida", required=True, help="pasta do run (criada; nao pode existir com resultado)")
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--iteracoes", type=int, default=None, help="sobrescreve simulation.iterations (teste rapido)")
    p.add_argument("--set", action="append", default=[], metavar="CHAVE=VALOR",
                   help="sobrescreve uma chave do config (ex.: simulation.solver.chebyshev.velocidade=3)")

    p = sub.add_parser("referencia", help="grava o resultado de um run como referencia")
    p.add_argument("--de", required=True, help="pasta de um run feito com `rodar`")
    p.add_argument("--destino", default=REFERENCIA)

    p = sub.add_parser("comparar", help="compara um run contra a referencia")
    p.add_argument("--saida", required=True, help="pasta de um run feito com `rodar`")
    p.add_argument("--referencia", default=REFERENCIA)
    return parser.parse_args()


# -- rodar ------------------------------------------------------------------

def _definir(cfg: dict, chave: str, valor: str):
    """`cfg[a][b][c] = valor` para `chave = "a.b.c"`; o valor e' lido como yaml (numero, bool...)."""
    *caminho, ultima = chave.split(".")
    no = cfg
    for parte in caminho:
        no = no.setdefault(parte, {})
    no[ultima] = yaml.safe_load(valor)


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True).stdout.strip()


def solucao_exata(Y: np.ndarray) -> np.ndarray:
    return 6.0 * Y * (1.0 - Y)


def erros_exatos(X, Y, vx, vy, p, reynolds: float) -> dict:
    """Erro do frame contra a solucao exata do Poiseuille (escala: vmax = 1,5)."""
    ex = vx - solucao_exata(Y)
    centro = (X > 1.75) & (X < 5.25)  # metade central do canal
    inclinacao = np.polyfit(X[centro], p[centro], 1)[0]
    exata = -12.0 / reynolds
    return {
        "vx_max": float(np.abs(ex).max() / 1.5),
        "vx_rms": float(np.sqrt(np.mean(ex ** 2)) / 1.5),
        "vy_max": float(np.abs(vy).max() / 1.5),
        "dpdx": float(inclinacao),
        "dpdx_exato": exata,
        "dpdx_erro_rel": float(abs(inclinacao - exata) / abs(exata)),
    }


def extrair(saida: str, cfg: dict) -> dict:
    """Le os frames gravados e o historico; grava `resultado.npz` e devolve o resumo."""
    from femns.io import listar_frames

    frames = listar_frames(saida)
    if not frames:
        raise RuntimeError(f"nenhum frame em {saida!r} -- a simulacao falhou? ver {saida}/log.txt")
    re_ = float(cfg["simulation"]["reynolds"])
    arrays, por_frame = {}, {}
    for n in sorted(frames):
        m = meshio.read(frames[n])
        X, Y = m.points[:, 0], m.points[:, 1]
        vx, vy, p = (np.asarray(m.point_data[k]) for k in ("vx", "vy", "p"))
        for nome, valor in zip(("X", "Y", "vx", "vy", "p"), (X, Y, vx, vy, p)):
            arrays[f"{nome}_{n}"] = valor
        por_frame[str(n)] = erros_exatos(X, Y, vx, vy, p, re_)
    np.savez_compressed(os.path.join(saida, "resultado.npz"), frames=np.array(sorted(frames)), **arrays)

    h = np.genfromtxt(os.path.join(saida, "historico.csv"), delimiter=",", names=True)
    return {
        "frames": sorted(frames),
        "erro_exato": por_frame,
        "iters_media": float(h["iters"].mean()),
        "iters_max": int(h["iters"].max()),
        "iters_total": int(h["iters"].sum()),
        "residual_max": float(h["residual"].max()),
        "passos_nao_convergidos": int((h["residual"] > cfg["simulation"].get("solver", {}).get("tol", 1e-8)).sum()),
        "tempo_passo_media_s": float(h["tempo_passo_s"].mean()),
        "tempo_passo_mediana_s": float(np.median(h["tempo_passo_s"])),
        "tempo_passos_total_s": float(h["tempo_passo_s"].sum()),
    }


def rodar(args):
    saida = os.path.abspath(args.saida)
    if os.path.exists(os.path.join(saida, "resultado.json")):
        sys.exit(f"{saida} ja tem um resultado; use outra pasta")
    os.makedirs(saida, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.iteracoes is not None:
        cfg["simulation"]["iterations"] = args.iteracoes
    for item in args.set:
        chave, _, valor = item.partition("=")
        _definir(cfg, chave, valor)
    cfg["output_dir"] = saida  # `mesh` relativa continua valendo: run_simulation roda com cwd = RAIZ
    cfg["historico_csv"] = os.path.join(saida, "historico.csv")
    cfg.pop("benchmark_xlsx", None)
    caminho_cfg = os.path.join(saida, "config.yaml")
    with open(caminho_cfg, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    inicio = datetime.now()
    with open(os.path.join(saida, "log.txt"), "w") as log:
        r = subprocess.run([sys.executable, os.path.join(RAIZ, "scripts", "run_simulation.py"), "--config", caminho_cfg],
                           cwd=RAIZ, stdout=log, stderr=subprocess.STDOUT)
    duracao = (datetime.now() - inicio).total_seconds()
    if r.returncode != 0:
        sys.exit(f"run_simulation.py saiu com codigo {r.returncode}; ver {saida}/log.txt")

    resumo = {
        "data": inicio.isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "arvore_suja": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "sobrescritas": args.set,
        "iteracoes": cfg["simulation"]["iterations"],
        "tempo_total_s": round(duracao, 2),
        **extrair(saida, cfg),
    }
    with open(os.path.join(saida, "resultado.json"), "w") as f:
        json.dump(resumo, f, indent=2)
    _imprimir_run(resumo)


def _imprimir_run(r: dict):
    final = r["erro_exato"][str(r["frames"][-1])]
    print(f"\n{r['iteracoes']} passos em {r['tempo_total_s']:.1f} s "
          f"(passo: media {r['tempo_passo_media_s'] * 1e3:.1f} ms, mediana {r['tempo_passo_mediana_s'] * 1e3:.1f} ms)")
    print(f"MINRES: {r['iters_media']:.2f} iteracoes/passo (max {r['iters_max']}), "
          f"residuo max {r['residual_max']:.1e}, {r['passos_nao_convergidos']} passos nao convergidos")
    print(f"erro exato no passo {r['frames'][-1]}: vx max {final['vx_max']:.3e}, rms {final['vx_rms']:.3e}, "
          f"vy max {final['vy_max']:.3e}, dp/dx {final['dpdx']:.5f} (exato {final['dpdx_exato']:.5f}, "
          f"erro {final['dpdx_erro_rel']:.2%})")


# -- referencia ---------------------------------------------------------------

def gravar_referencia(args):
    os.makedirs(args.destino, exist_ok=True)
    for nome in ARQUIVOS_DO_RUN:
        shutil.copy2(os.path.join(args.de, nome), os.path.join(args.destino, nome))
    print(f"referencia gravada em {args.destino}")


# -- comparar -------------------------------------------------------------------

def comparar(args):
    def carregar(pasta):
        with open(os.path.join(pasta, "resultado.json")) as f:
            return json.load(f), np.load(os.path.join(pasta, "resultado.npz"))

    ref, campos_ref = carregar(args.referencia)
    run, campos_run = carregar(args.saida)
    if ref["frames"] != run["frames"]:
        sys.exit(f"frames diferentes: referencia {ref['frames']}, run {run['frames']} -- mesmo numero de passos?")

    print(f"referencia: {ref['commit'][:7]} ({ref['branch']}, {ref['data']})")
    print(f"run:        {run['commit'][:7]} ({run['branch']}, {run['data']})"
          + (f", sobrescritas {run['sobrescritas']}" if run["sobrescritas"] else ""))
    print("\ndiferenca relativa ao run de referencia (max|run - ref| / max|ref|)")
    print(f"{'passo':>6} {'vx':>10} {'vy':>10} {'p':>10}")
    pior = 0.0
    for n in ref["frames"]:
        for eixo in ("X", "Y"):
            if not np.array_equal(campos_ref[f"{eixo}_{n}"], campos_run[f"{eixo}_{n}"]):
                sys.exit(f"a malha do passo {n} difere da referencia -- o movimento da malha mudou?")
        # vy e' ~0 na solucao exata: normaliza pela escala da velocidade (max |vx|)
        escala = {"vx": np.abs(campos_ref[f"vx_{n}"]).max(), "vy": np.abs(campos_ref[f"vx_{n}"]).max(),
                  "p": np.abs(campos_ref[f"p_{n}"]).max()}
        difs = {k: np.abs(campos_run[f"{k}_{n}"] - campos_ref[f"{k}_{n}"]).max() / escala[k] for k in escala}
        pior = max(pior, *difs.values())
        print(f"{n:6d} {difs['vx']:10.2e} {difs['vy']:10.2e} {difs['p']:10.2e}")

    print("\nerro contra a solucao exata (ultimo passo)")
    final = str(ref["frames"][-1])
    print(f"{'':16}{'referencia':>12}{'run':>12}")
    piora = 0.0
    for k in ("vx_max", "vx_rms", "vy_max", "dpdx_erro_rel"):
        a, b = ref["erro_exato"][final][k], run["erro_exato"][final][k]
        piora = max(piora, (b - a) / a if a > 0 else 0.0)
        print(f"{k:16}{a:12.3e}{b:12.3e}")

    print("\ndesempenho")
    print(f"{'':26}{'referencia':>12}{'run':>12}{'variacao':>10}")
    for k, fmt in (("iters_media", "{:12.2f}"), ("iters_max", "{:12d}"), ("passos_nao_convergidos", "{:12d}"),
                   ("tempo_passo_media_s", "{:12.4f}"), ("tempo_passo_mediana_s", "{:12.4f}"),
                   ("tempo_total_s", "{:12.1f}")):
        a, b = ref[k], run[k]
        var = f"{(b - a) / a:+10.1%}" if a else f"{'':>10}"
        print(f"{k:26}{fmt.format(a)}{fmt.format(b)}{var}")

    ok_campo = pior <= LIMITE_CAMPO
    ok_exata = piora <= LIMITE_PIORA_EXATA
    ok_conv = run["passos_nao_convergidos"] <= ref["passos_nao_convergidos"]
    print(f"\n[{'ok' if ok_campo else 'FALHOU'}] maior diferenca de campo {pior:.2e} (limite {LIMITE_CAMPO:.0e})")
    print(f"[{'ok' if ok_exata else 'FALHOU'}] erro contra a exata piorou no maximo {piora:+.2%} "
          f"(limite {LIMITE_PIORA_EXATA:+.0%})")
    print(f"[{'ok' if ok_conv else 'FALHOU'}] passos nao convergidos: {run['passos_nao_convergidos']} "
          f"(referencia {ref['passos_nao_convergidos']})")
    sys.exit(0 if ok_campo and ok_exata and ok_conv else 1)


def main():
    args = parse_args()
    {"rodar": rodar, "referencia": gravar_referencia, "comparar": comparar}[args.comando](args)


if __name__ == "__main__":
    main()
