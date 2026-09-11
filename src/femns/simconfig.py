"""Leitura e validacao de blocos opcionais do config de simulacao.

Sem dependencia de torch/numpy -- e' importado tanto pelo CLI
(`scripts/run_simulation.py`) quanto pela GUI (`femns.webui.jobs`, que roda
a simulacao em subprocesso e evita carregar o solver no processo do
servidor).
"""

DEFAULT_BICGSTAB_TOL = 1e-8
DEFAULT_BICGSTAB_MAX_ITER = 2000


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
