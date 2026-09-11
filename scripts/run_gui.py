#!/usr/bin/env python3
"""CLI: sobe a interface web opcional do femns (nova simulacao + visualizacao de resultados).

Modulo a parte do pipeline de simulacao -- `run_simulation.py` nao depende
disso, e nao e preciso ter a GUI rodando para simular pela linha de comando.
"""

import argparse
import sys

from femns.webui.server import run


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765, help="Porta do servidor (padrao: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Nao abrir o navegador automaticamente")
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Endereco de bind (padrao: 127.0.0.1, so aceita conexao local). "
             "Use 0.0.0.0 para aceitar conexoes da rede -- exige a variavel FEMNS_GUI_TOKEN "
             "no ambiente (ou --allow-no-auth). Ver README, secao 'Interface web', e docs/planos/plano_remoto.md")
    parser.add_argument(
        "--allow-no-auth", action="store_true",
        help="Permite bind nao-local sem token (INSEGURO -- so para rede isolada/confiavel)")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        run(port=args.port, open_browser=not args.no_browser, host=args.host, allow_no_auth=args.allow_no_auth)
    except RuntimeError as e:  # bind inseguro (ver server.check_bind) -- mensagem ja e' explicativa
        print(f"erro: {e}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
