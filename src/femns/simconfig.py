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

# Solver linear (`simulation.solver`). `femns.precondicionadores` (que importa
# torch) tem um teste garantindo que tem exatamente esses pre-condicionadores.
METODOS = ("bicgstab", "minres")
PRECONDICIONADORES = ("nenhum", "jacobi", "bloco_triangular", "bloco_diagonal")
# o MINRES exige pre-condicionador simetrico e definido positivo
PRECONDICIONADORES_MINRES = ("nenhum", "jacobi", "bloco_diagonal")
SUBSOLVERS = ("chebyshev", "fsai", "lu")
# Padrao (plano 12/10, §9.6-9.7): MINRES + bloco-diagonal (M1) com FSAI, so GPU,
# o unico que convergiu em todos os passos da valvula com erro de pressao ~1e-8.
DEFAULT_METODO = "minres"
DEFAULT_PRECONDICIONADOR = "bloco_diagonal"
# (com `metodo: bicgstab` e sem precondicionador explicito, o padrao vira bloco_triangular)
DEFAULT_SUBSOLVER = "fsai"
# Passos de Chebyshev por sub-solver. chebyshev 40/5/80: com menos passos
# (5/5/20) o B2 quebra no Poiseuille (F e Lp mal aproximados); fsai 5/5/10:
# os do benchmark da valvula. O `lu` nao usa (so o Mp, com `massa`).
DEFAULT_CHEBYSHEV = {"velocidade": 40, "massa": 5, "laplaciano": 80}
DEFAULT_CHEBYSHEV_POR_SUBSOLVER = {
    "chebyshev": DEFAULT_CHEBYSHEV,
    "fsai": {"velocidade": 5, "massa": 5, "laplaciano": 10},
    "lu": DEFAULT_CHEBYSHEV,
}

# Nomes aceitos em `simulation.element`. `femns.elemento` (que importa torch)
# tem um teste garantindo que tem exatamente esses elementos.
ELEMENTOS = ("mini", "tri6")


@dataclass(frozen=True)
class OpcoesSolver:
    """Bloco `simulation.solver` ja validado, com os defaults aplicados."""

    tol: float = DEFAULT_BICGSTAB_TOL
    max_iter: int = DEFAULT_BICGSTAB_MAX_ITER
    beta_por_passo: bool = False
    metodo: str = DEFAULT_METODO
    precondicionador: str = DEFAULT_PRECONDICIONADOR
    chebyshev: tuple = tuple(DEFAULT_CHEBYSHEV_POR_SUBSOLVER[DEFAULT_SUBSOLVER].values())  # (velocidade, massa, laplaciano)
    subsolver: str = DEFAULT_SUBSOLVER
    verificar_direto: int = 0


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
    solver: OpcoesSolver


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

    opcoes = solver_options(sim)

    boundary = cfg.get("boundary") or {}
    if not boundary.get("priority") or not boundary.get("conditions"):
        raise ValueError("boundary.priority e boundary.conditions sao obrigatorios")

    return ParametrosSimulacao(dt=float(dt), iterations=iterations, reynolds=float(reynolds), element=element,
                               vtk_interval=vtk_interval, tol=opcoes.tol, max_iter=opcoes.max_iter,
                               beta_por_passo=opcoes.beta_por_passo, solver=opcoes)


def solver_options(sim_cfg: dict) -> OpcoesSolver:
    """Opcoes do solver linear, do bloco opcional `simulation.solver`.

        simulation:
          solver:
            tol: 1.0e-8            # residuo relativo alvo (padrao 1e-8)
            max_iter: 2000         # teto de iteracoes por passo de tempo (padrao 2000)
            beta_por_passo: false  # recalcular a equilibracao de pressao a cada
                                   #   passo de malha movel (padrao false)
            metodo: minres         # minres (padrao) ou bicgstab
            precondicionador: bloco_diagonal  # bloco_diagonal (padrao), nenhum,
                                       #   jacobi ou bloco_triangular (so bicgstab)
            subsolver: fsai        # blocos dos pre-condicionadores de bloco: fsai
                                   #   (FSAI, GPU, padrao), chebyshev (Jacobi, GPU)
                                   #   ou lu (LU exata, CPU)
            chebyshev:             # passos de Chebyshev dos blocos aproximados
              velocidade: 5        #   (so pre-condicionadores de bloco; padrao
              massa: 5             #   5/5/10 com fsai, 40/5/80 com chebyshev)
              laplaciano: 10
            verificar_direto: 0    # a cada N passos, compara com a solucao
                                   #   direta (0 = nunca; caro, so validacao)

    `tol`/`max_iter` reproduzem, nos defaults, o comportamento anterior; o
    metodo padrao passou a ser MINRES + bloco-diagonal com FSAI (M1, plano
    12/10) -- `metodo: bicgstab, precondicionador: nenhum` reproduz o antigo.
    `beta_por_passo` so tem efeito com `mesh_motion`: ligado,
    `pressure_scale_factor` e' recalculado a cada passo e o componente de
    pressao do warm-start e' reescalado. Os pre-condicionadores estao em
    `femns.precondicionadores` (plano 12/10); o MINRES exige um simetrico e
    definido positivo (`nenhum`, `jacobi` ou `bloco_diagonal`).

    Levanta ValueError (bloco nao e' mapa, chave desconhecida, valor
    invalido) -- o chamador decide se vira erro de CLI ou `ConfigError` da GUI.
    """
    cfg = sim_cfg.get("solver") or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"simulation.solver precisa ser um mapa (recebido: {cfg!r})")

    chaves = {"tol", "max_iter", "beta_por_passo", "metodo", "precondicionador", "chebyshev", "subsolver",
              "verificar_direto"}
    extras = set(cfg) - chaves
    if extras:
        raise ValueError(
            f"simulation.solver: chave(s) desconhecida(s) {sorted(extras)} (use {', '.join(sorted(chaves))})")

    tol = cfg.get("tol", DEFAULT_BICGSTAB_TOL)
    max_iter = cfg.get("max_iter", DEFAULT_BICGSTAB_MAX_ITER)
    beta_por_passo = cfg.get("beta_por_passo", False)
    if not _numero(tol) or tol <= 0:
        raise ValueError(f"simulation.solver.tol precisa ser um numero positivo (recebido: {tol!r})")
    if not _inteiro(max_iter) or max_iter <= 0:
        raise ValueError(f"simulation.solver.max_iter precisa ser um inteiro positivo (recebido: {max_iter!r})")
    if not isinstance(beta_por_passo, bool):
        raise ValueError(f"simulation.solver.beta_por_passo precisa ser true/false (recebido: {beta_por_passo!r})")

    metodo = cfg.get("metodo", DEFAULT_METODO)
    if metodo not in METODOS:
        raise ValueError(f"simulation.solver.metodo desconhecido: {metodo!r} (use {' ou '.join(METODOS)})")
    # sem precondicionador explicito, o de bloco que combina com o metodo (B2 no BiCGSTAB)
    precondicionador = cfg.get("precondicionador",
                               DEFAULT_PRECONDICIONADOR if metodo == "minres" else "bloco_triangular")
    if precondicionador not in PRECONDICIONADORES:
        raise ValueError(f"simulation.solver.precondicionador desconhecido: {precondicionador!r} "
                         f"(use {', '.join(PRECONDICIONADORES)})")
    if metodo == "minres" and precondicionador not in PRECONDICIONADORES_MINRES:
        raise ValueError(f"simulation.solver: o MINRES exige pre-condicionador simetrico e definido positivo "
                         f"({', '.join(PRECONDICIONADORES_MINRES)}); recebido {precondicionador!r}")

    subsolver = cfg.get("subsolver", DEFAULT_SUBSOLVER)
    if subsolver not in SUBSOLVERS:
        raise ValueError(f"simulation.solver.subsolver desconhecido: {subsolver!r} (use {' ou '.join(SUBSOLVERS)})")

    cheb = cfg.get("chebyshev") or {}
    if not isinstance(cheb, dict) or set(cheb) - set(DEFAULT_CHEBYSHEV):
        raise ValueError(f"simulation.solver.chebyshev precisa ser um mapa com {', '.join(DEFAULT_CHEBYSHEV)} "
                         f"(recebido: {cheb!r})")
    passos = {k: cheb.get(k, v) for k, v in DEFAULT_CHEBYSHEV_POR_SUBSOLVER[subsolver].items()}
    for k, v in passos.items():
        if not _inteiro(v) or v <= 0:
            raise ValueError(f"simulation.solver.chebyshev.{k} precisa ser um inteiro positivo (recebido: {v!r})")

    verificar_direto = cfg.get("verificar_direto", 0)
    if not _inteiro(verificar_direto) or verificar_direto < 0:
        raise ValueError(f"simulation.solver.verificar_direto precisa ser um inteiro >= 0 (recebido: {verificar_direto!r})")

    return OpcoesSolver(tol=float(tol), max_iter=int(max_iter), beta_por_passo=beta_por_passo,
                        metodo=metodo, precondicionador=precondicionador,
                        chebyshev=tuple(passos.values()), subsolver=subsolver, verificar_direto=verificar_direto)
