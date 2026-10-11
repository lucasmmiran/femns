"""Validacao do config de simulacao -- unico lugar com as regras.

Sem dependencia de torch/numpy -- e' importado tanto pelo CLI
(`scripts/run_simulation.py`) quanto pela GUI (`femns.webui.jobs`, que valida
antes de abrir o subprocesso, para devolver o erro na hora no formulario, e
evita carregar o solver no processo do servidor). As mensagens de erro saem
iguais no terminal e na GUI.
"""

from dataclasses import dataclass

DEFAULT_BICGSTAB_TOL = 1e-8
DEFAULT_BICGSTAB_MAX_ITER = 2000
DEFAULT_VTK_INTERVAL = 10

# Nomes aceitos em `simulation.element`. `femns.elemento` (que importa torch)
# tem um teste garantindo que tem exatamente esses elementos.
ELEMENTOS = ("mini", "tri6")


@dataclass(frozen=True)
class ParametrosSimulacao:
    """Bloco `simulation` ja validado, com os defaults aplicados."""

    dt: float
    iterations: int
    reynolds: float
    element: str
    vtk_interval: int
    tol: float
    max_iter: int
    beta_por_passo: bool


def _numero(valor) -> bool:
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)


def _inteiro(valor) -> bool:
    return isinstance(valor, int) and not isinstance(valor, bool)


def validar_config(cfg: dict) -> ParametrosSimulacao:
    """Valida o config inteiro (o que nao depende de arquivo) e devolve o bloco `simulation`.

    Confere: `mesh` presente; `simulation.dt`/`reynolds` numeros positivos,
    `iterations`/`vtk_interval` inteiros positivos, `element` conhecido, o
    bloco `simulation.solver` (`solver_options`); chaves removidas na v2.0
    (`advection` diferente de `semi_lagrangian`, `sl_boundary` diferente de
    `intercept`, `moving_point` ligado -- com valor da v2.0 ou desligadas,
    vindas de config antigo, sao aceitas e ignoradas); `boundary.priority` e
    `boundary.conditions` presentes.

    Se a malha existe e e' segura (caminho dentro de `meshes/`) e' checado
    por quem chama -- o CLI le o arquivo, a GUI confere o caminho.

    Levanta ValueError com a mensagem do primeiro problema.
    """
    mesh = cfg.get("mesh")
    if not mesh or not isinstance(mesh, str):
        raise ValueError("config.mesh e obrigatorio")

    sim = cfg.get("simulation")
    if not isinstance(sim, dict):
        raise ValueError("config.simulation e obrigatorio (dt, iterations, reynolds, ...)")

    dt, iterations, reynolds = sim.get("dt"), sim.get("iterations"), sim.get("reynolds")
    if not _numero(dt) or dt <= 0:
        raise ValueError(f"simulation.dt precisa ser um numero positivo (recebido: {dt!r})")
    if not _inteiro(iterations) or iterations <= 0:
        raise ValueError(f"simulation.iterations precisa ser um inteiro positivo (recebido: {iterations!r})")
    if not _numero(reynolds) or reynolds <= 0:
        raise ValueError(f"simulation.reynolds precisa ser um numero positivo (recebido: {reynolds!r})")

    element = sim.get("element", "mini")
    if element not in ELEMENTOS:
        raise ValueError(f"simulation.element desconhecido: {element!r} (use {' ou '.join(map(repr, ELEMENTOS))})")

    vtk_interval = sim.get("vtk_interval", DEFAULT_VTK_INTERVAL)
    if not _inteiro(vtk_interval) or vtk_interval <= 0:
        raise ValueError(
            f"simulation.vtk_interval precisa ser um inteiro positivo (recebido: {vtk_interval!r}) -- "
            f"grava um .vtk a cada N iteracoes; a ultima iteracao e' sempre gravada")

    if sim.get("advection", "semi_lagrangian") != "semi_lagrangian":
        raise ValueError(
            f"simulation.advection={sim.get('advection')!r} foi removido na v2.0: a adveccao e' sempre "
            "semi-Lagrangeana. Apague a chave do config; para a adveccao Euleriana "
            "('eulerian'/'explicit'), use a tag v1.0.")
    if sim.get("sl_boundary", "intercept") != "intercept":
        raise ValueError(
            f"simulation.sl_boundary={sim.get('sl_boundary')!r} foi removido na v2.0: o pe da "
            "caracteristica fora do dominio e' sempre tratado por interceptacao da aresta de saida "
            "(o antigo 'intercept'). Apague a chave do config; para o modo 'dirichlet', use a tag v1.0.")
    if (cfg.get("moving_point") or {}).get("enabled", False):
        raise ValueError(
            "moving_point foi removido na v2.0 (movia um no so). Apague o bloco do config; "
            "para malha movel use mesh_motion, ou a tag v1.0 para o experimento antigo.")

    tol, max_iter, beta_por_passo = solver_options(sim)

    boundary = cfg.get("boundary") or {}
    if not boundary.get("priority") or not boundary.get("conditions"):
        raise ValueError("boundary.priority e boundary.conditions sao obrigatorios")

    return ParametrosSimulacao(dt=float(dt), iterations=iterations, reynolds=float(reynolds), element=element,
                               vtk_interval=vtk_interval, tol=tol, max_iter=max_iter,
                               beta_por_passo=beta_por_passo)


def solver_options(sim_cfg: dict) -> tuple[float, int, bool]:
    """`(tol, max_iter, beta_por_passo)` do bloco opcional `simulation.solver`.

        simulation:
          solver:
            tol: 1.0e-8            # residuo relativo alvo (padrao 1e-8)
            max_iter: 2000         # teto de iteracoes por passo de tempo (padrao 2000)
            beta_por_passo: false  # recalcular a equilibracao de pressao a cada
                                   #   passo de malha movel (padrao false)

    `tol`/`max_iter` reproduzem, nos defaults, o comportamento anterior
    (valores fixos em `femns.solver.time_step`). Uma malha maior ou um caso
    mais rigido pode precisar de outro `max_iter`; afrouxar `tol` troca
    precisao por velocidade (ver `docs/bicgstab_solver.tex`).

    `beta_por_passo` so tem efeito com `mesh_motion`: quando a malha deforma
    muito (ex.: fechamento de valvula), `beta = ||A_vv|| / ||(Gx;Gy)||`
    afinado na geometria inicial vira uma equilibracao ruim. Ligado,
    `pressure_scale_factor` e' recalculado a cada passo e o componente de
    pressao do warm-start e' reescalado. Default `false` = escala fixa,
    identica ao comportamento anterior.

    Levanta ValueError (bloco nao e' mapa, chave desconhecida, valor nao
    positivo / de tipo errado) -- o chamador decide se vira erro de CLI ou
    `ConfigError` da GUI.
    """
    cfg = sim_cfg.get("solver") or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"simulation.solver precisa ser um mapa (recebido: {cfg!r})")

    extras = set(cfg) - {"tol", "max_iter", "beta_por_passo"}
    if extras:
        raise ValueError(
            f"simulation.solver: chave(s) desconhecida(s) {sorted(extras)} "
            f"(use 'tol', 'max_iter' e/ou 'beta_por_passo')")

    tol = cfg.get("tol", DEFAULT_BICGSTAB_TOL)
    max_iter = cfg.get("max_iter", DEFAULT_BICGSTAB_MAX_ITER)
    beta_por_passo = cfg.get("beta_por_passo", False)
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or tol <= 0:
        raise ValueError(f"simulation.solver.tol precisa ser um numero positivo (recebido: {tol!r})")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError(f"simulation.solver.max_iter precisa ser um inteiro positivo (recebido: {max_iter!r})")
    if not isinstance(beta_por_passo, bool):
        raise ValueError(f"simulation.solver.beta_por_passo precisa ser true/false (recebido: {beta_por_passo!r})")
    return float(tol), int(max_iter), beta_por_passo
