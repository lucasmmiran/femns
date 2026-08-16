#!/usr/bin/env python3
"""Benchmark do custo por passo dos dois tratamentos de "fora do dominio" do semi-Lagrangeano.

Mede, na malha real, o custo de um passo de `femns.semi_lagrangian.calculo_sl`
com `fora_dominio="dirichlet"` (historico do projeto) e `"intercept"`
(interceptacao geometrica da aresta de saida, como o codigo de referencia do
professor), e compara com `femns.semi_lagrangian_tri.SemiLagrangianMini` --
que ja implementa a mesma interceptacao, mas em torch (roda na GPU, sem
round-trip por passo).

Tambem mede o piso de custo de paralelizar por processos (o que o codigo de
referencia faz em `getDepartElem`, via `multiprocessing.Pool`): so o
ida-e-volta de dados entre processos, sem nenhuma conta. Com `--iter-ms`
(tempo por iteracao da simulacao completa) fecha o argumento de Amdahl: o
que decide nao e' o piso de transporte, e sim o quanto o passo
semi-Lagrangeano pesa na iteracao inteira -- o resto e' BiCGSTAB na GPU.

So mede desempenho -- a concordancia numerica entre as duas fronteiras e
comparada em `scripts/compare_fields.py` sobre a simulacao completa.
"""

import argparse
from multiprocessing import get_context
from timeit import default_timer as timer

import numpy as np
import torch
import yaml

from benchmark_semi_lagrangian import cronometrar, preparar_campo_real
from femns.semi_lagrangian import calculo_sl
from femns.semi_lagrangian_tri import SemiLagrangianMini


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/degrau.yaml", help="Config yaml (usa so mesh/dt/Re/contorno)")
    parser.add_argument("--repeats", type=int, default=50, help="Repeticoes cronometradas de cada variante")
    parser.add_argument("--processes", type=int, default=4, help="Processos no teste de paralelizacao")
    parser.add_argument("--iter-ms", type=float, default=None,
                         help="Tempo medio por iteracao da simulacao completa, em ms "
                              "(coluna tempo_medio_por_iter_s do benchmark_xlsx). Se dado, "
                              "reporta o peso do passo semi-Lagrangeano na iteracao inteira.")
    return parser.parse_args()


def _eco(chunk):
    """Worker do piso de IPC: devolve o que recebeu, sem conta nenhuma.

    Isola o custo de transporte (pickle na ida + pickle na volta) de um
    volume de dados equivalente ao de um passo semi-Lagrangeano.
    """
    return chunk


def piso_multiprocessing(n_total: int, processes: int, repeats: int) -> float:
    """Custo medio de um round-trip `Pool.map` com o volume de dados de um passo.

    Cada chunk carrega as coordenadas do pe da caracteristica (2 arrays) e
    volta com a velocidade interpolada (2 arrays) -- 4 arrays float64 de
    `n_total` elementos no total, divididos entre os processos. O Pool e'
    criado uma vez e reaproveitado, entao isto mede so o `map` (o fork
    inicial ficaria por conta de quem chamasse uma vez por simulacao).
    """
    chunks = [np.empty((4, len(c)), dtype=np.float64)
              for c in np.array_split(np.arange(n_total), processes)]

    with get_context("fork").Pool(processes) as pool:
        pool.map(_eco, chunks)  # aquece (fork dos workers fora da medicao)
        inicio = timer()
        for _ in range(repeats):
            pool.map(_eco, chunks)
        return (timer() - inicio) / repeats


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    campo = preparar_campo_real(cfg, device)
    npoints, ne, dt = campo["npoints"], campo["ne"], campo["dt"]
    n_total = npoints + ne

    vx_np, vy_np = campo["vx"].cpu().numpy(), campo["vy"].cpu().numpy()
    X_np, Y_np = campo["X"], campo["Y"]

    def rodar(fora_dominio):
        return lambda: calculo_sl(
            X_np, Y_np, campo["IEN"], campo["EToE"], campo["node_to_elem"],
            campo["ccName"], campo["conditions"], vx_np, vy_np, dt, npoints, ne,
            fora_dominio=fora_dominio)

    node_to_elem_t = torch.from_numpy(campo["node_to_elem"]).to(device)
    sl_torch = SemiLagrangianMini(
        campo["IEN_t"], campo["EToE"], campo["X_t"], campo["Y_t"], node_to_elem_t, npoints, ne)

    print(f"Malha: {npoints} pontos, {ne} elementos, {n_total} nos (com centroide)")
    print(f"Cronometrando {args.repeats} repeticoes de cada...\n")

    variantes = {
        "dirichlet": rodar("dirichlet"),
        "intercept": rodar("intercept"),
        "torch": lambda: sl_torch.compute(campo["vx"], campo["vy"], dt),
    }

    # Aquecimento obrigatorio antes de cronometrar: a primeira chamada da via
    # torch/cuda paga compilacao de kernel + warm-up do alocador (~2.5 s), o
    # que com poucas repeticoes vira ~90 ms/passo de artefato e faz a via
    # torch parecer ~20x mais lenta do que e' de fato.
    for func in variantes.values():
        func()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t_dir = cronometrar(variantes["dirichlet"], args.repeats, device)
    t_int = cronometrar(variantes["intercept"], args.repeats, device)
    t_torch = cronometrar(variantes["torch"], args.repeats, device)
    t_ipc = piso_multiprocessing(n_total, args.processes, args.repeats)

    print("Custo por passo semi-Lagrangeano:")
    print(f"  numpy vetorizado, fronteira 'dirichlet'  : {t_dir * 1e3:8.3f} ms")
    print(f"  numpy vetorizado, fronteira 'intercept'  : {t_int * 1e3:8.3f} ms  "
          f"({(t_int / t_dir - 1) * 100:+.1f}% vs dirichlet)")
    print(f"  torch/{device}, fronteira 'intercept'{'':<7}: {t_torch * 1e3:8.3f} ms  "
          f"(SemiLagrangianMini, mesma interceptacao)")
    print()
    print(f"Piso de paralelizar por processos ({args.processes} procs):")
    print(f"  so o transporte de dados (sem conta)     : {t_ipc * 1e3:8.3f} ms "
          f"({t_ipc / t_int * 100:.0f}% do passo vetorizado)")
    print(f"  melhor caso teorico (conta / {args.processes} + transporte) : "
          f"{(t_int / args.processes + t_ipc) * 1e3:8.3f} ms\n")

    # Amdahl: o passo semi-Lagrangeano e' so uma fatia da iteracao (o resto e'
    # BiCGSTAB na GPU), entao o teto de ganho da iteracao inteira e' essa fatia.
    if args.iter_ms:
        frac = t_int / (args.iter_ms * 1e-3)
        ganho_max = frac * 100
        print(f"Peso na iteracao inteira ({args.iter_ms:.0f} ms/iter medidos em run_simulation.py):")
        print(f"  passo semi-Lagrangeano                   : {frac * 100:.1f}% do tempo de iteracao")
        print(f"  teto de ganho zerando esse passo         : {ganho_max:.1f}% do tempo total\n")
        print("Conclusao: nao vale paralelizar por processos. Nao e' que o transporte domine -- e'")
        print("que o passo inteiro ja e' uma fatia pequena da iteracao (o resto e' BiCGSTAB na GPU),")
        print(f"entao mesmo um passo instantaneo economizaria no maximo {ganho_max:.1f}% do tempo total.")
        print("O multiprocessing existe na referencia do professor porque la o laco e' Python no a")
        print("no; vetorizado, esse custo ja saiu do caminho critico.")


if __name__ == "__main__":
    main()
