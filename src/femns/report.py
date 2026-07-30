"""Registro de metricas de simulacao numa planilha .xlsx acumulada (1 linha por run).

Pensado pra comparar simulacoes entre si (ex.: `explicit` vs `semi_lagrangian`,
diferentes `dt`/malha/Re) numa tabela unica, nao pra analise por-iteracao
dentro de um unico run -- ver `scripts/run_simulation.py` sobre quais
metricas sao agregadas (media/max) antes de virar uma linha aqui.
"""

from pathlib import Path

from openpyxl import Workbook, load_workbook

COLUNAS = [
    "timestamp", "mesh", "advection", "dt", "reynolds", "iterations",
    "npoints", "ne", "device",
    "tempo_total_s", "tempo_assembly_s", "tempo_medio_por_iter_s",
    "bicg_iters_media", "bicg_iters_max",
    "bicg_residual_media", "bicg_residual_max",
    "passos_nao_convergidos",
    "vel_l2_media", "vel_l2_max", "vel_linf_max",
    "divergencia_l2_media", "divergencia_l2_max",
    "pressao_min", "pressao_max", "pressao_media",
]


def salvar_resumo(path: str, dados: dict):
    """Acrescenta uma linha de resumo em `path` (.xlsx), criando arquivo/cabecalho se preciso.

    `dados` mapeia (um subconjunto de) `COLUNAS` para valores dessa
    simulacao; chaves de `COLUNAS` ausentes em `dados` viram celula vazia
    (nao quebra se uma metrica nao foi calculada) e chaves extras em
    `dados` sao ignoradas (nao aparecem na planilha).
    """
    caminho = Path(path)

    if caminho.exists():
        wb = load_workbook(caminho)
        ws = wb.active
    else:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        wb = Workbook()
        ws = wb.active
        ws.title = "Resumo"
        ws.append(COLUNAS)

    ws.append([dados.get(coluna, "") for coluna in COLUNAS])
    wb.save(caminho)
