#!/usr/bin/env python3
"""CLI: sobe a interface web opcional do femns (nova simulacao + visualizacao de resultados).

Modulo a parte do pipeline de simulacao -- `run_simulation.py` nao depende
disso, e nao e preciso ter a GUI rodando para simular pela linha de comando.
"""

import argparse

from femns.webui.server import run


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765, help="Porta do servidor local (padrao: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Nao abrir o navegador automaticamente")
    return parser.parse_args()


def main():
    args = parse_args()
    run(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
