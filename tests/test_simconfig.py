import pytest

from femns.simconfig import DEFAULT_BICGSTAB_MAX_ITER, DEFAULT_BICGSTAB_TOL, solver_options


def test_solver_options_default_sem_bloco():
    o = solver_options({"dt": 0.01})
    assert (o.tol, o.max_iter, o.beta_por_passo) == (DEFAULT_BICGSTAB_TOL, DEFAULT_BICGSTAB_MAX_ITER, False)
    assert (o.metodo, o.precondicionador, o.subsolver, o.verificar_direto) == ("minres", "bloco_diagonal", "fsai", 0)
    assert o.chebyshev == (5, 5, 10)


def test_solver_options_bloco_vazio_cai_no_default():
    o = solver_options({"solver": {}})
    assert (o.tol, o.max_iter, o.beta_por_passo) == (DEFAULT_BICGSTAB_TOL, DEFAULT_BICGSTAB_MAX_ITER, False)


def test_solver_options_le_valores():
    o = solver_options({"solver": {"tol": 1e-6, "max_iter": 500}})
    assert (o.tol, o.max_iter) == (1e-6, 500)


def test_solver_options_um_valor_so_mantem_o_outro_no_default():
    o = solver_options({"solver": {"max_iter": 5000}})
    assert (o.tol, o.max_iter) == (DEFAULT_BICGSTAB_TOL, 5000)
    o = solver_options({"solver": {"tol": 1e-10}})
    assert (o.tol, o.max_iter) == (1e-10, DEFAULT_BICGSTAB_MAX_ITER)


def test_solver_options_beta_por_passo():
    assert solver_options({"solver": {"beta_por_passo": True}}).beta_por_passo is True
    assert solver_options({"solver": {}}).beta_por_passo is False


def test_solver_options_rejeita_beta_por_passo_nao_bool():
    import pytest as _pytest
    with _pytest.raises(ValueError, match="beta_por_passo"):
        solver_options({"solver": {"beta_por_passo": "sim"}})


def test_solver_options_tol_inteiro_vira_float():
    tol = solver_options({"solver": {"tol": 1}}).tol
    assert isinstance(tol, float) and tol == 1.0


@pytest.mark.parametrize("valor", [0, -1, -1e-8, "1e-8", True, None])
def test_solver_options_rejeita_tol_invalido(valor):
    with pytest.raises(ValueError, match="tol"):
        solver_options({"solver": {"tol": valor}})


@pytest.mark.parametrize("valor", [0, -1, 1.5, "2000", True, None])
def test_solver_options_rejeita_max_iter_invalido(valor):
    with pytest.raises(ValueError, match="max_iter"):
        solver_options({"solver": {"max_iter": valor}})


def test_solver_options_rejeita_chave_desconhecida():
    with pytest.raises(ValueError, match="desconhecida"):
        solver_options({"solver": {"tol": 1e-8, "precond": "jacobi"}})


def test_solver_options_rejeita_bloco_nao_mapa():
    with pytest.raises(ValueError, match="mapa"):
        solver_options({"solver": 1e-8})


def config_minimo(**sim):
    simulation = {"dt": 0.01, "iterations": 10, "reynolds": 1.0}
    simulation.update(sim)
    return {
        "mesh": "meshes/poiseuille.msh",
        "simulation": simulation,
        "boundary": {"priority": ["inlet"], "conditions": {"inlet": {"vx": 1.0}}},
    }


def test_validar_config_aplica_defaults():
    from femns.simconfig import DEFAULT_VTK_INTERVAL, validar_config

    p = validar_config(config_minimo())
    assert (p.dt, p.iterations, p.reynolds) == (0.01, 10, 1.0)
    assert p.element == "mini" and p.vtk_interval == DEFAULT_VTK_INTERVAL
    assert (p.tol, p.max_iter, p.beta_por_passo) == (DEFAULT_BICGSTAB_TOL, DEFAULT_BICGSTAB_MAX_ITER, False)


@pytest.mark.parametrize("sim, trecho", [
    ({"dt": 0}, "dt"), ({"dt": True}, "dt"), ({"iterations": 1.5}, "iterations"),
    ({"iterations": True}, "iterations"), ({"reynolds": -1}, "reynolds"),
    ({"element": "q4"}, "element"), ({"vtk_interval": 0}, "vtk_interval"),
    ({"advection": "eulerian"}, "v1.0"), ({"sl_boundary": "dirichlet"}, "v1.0"),
    ({"solver": {"tol": -1}}, "tol"),
])
def test_validar_config_rejeita(sim, trecho):
    from femns.simconfig import validar_config

    with pytest.raises(ValueError, match=trecho):
        validar_config(config_minimo(**sim))


def test_validar_config_aceita_chaves_legadas_com_valor_da_v2():
    from femns.simconfig import validar_config

    cfg = config_minimo(advection="semi_lagrangian", sl_boundary="intercept")
    cfg["moving_point"] = {"enabled": False}
    validar_config(cfg)  # nao deve levantar

    cfg["moving_point"]["enabled"] = True
    with pytest.raises(ValueError, match="moving_point"):
        validar_config(cfg)


def test_validar_config_exige_mesh_e_boundary():
    from femns.simconfig import validar_config

    sem_mesh = config_minimo()
    del sem_mesh["mesh"]
    with pytest.raises(ValueError, match="mesh"):
        validar_config(sem_mesh)

    sem_boundary = config_minimo()
    sem_boundary["boundary"] = {}
    with pytest.raises(ValueError, match="boundary"):
        validar_config(sem_boundary)


def test_elementos_do_config_batem_com_as_classes():
    """`simconfig` nao pode importar `femns.elemento` (torch); a lista de nomes
    fica duplicada de proposito e este teste impede que as duas se separem."""
    from femns.elemento import _POR_NOME
    from femns.simconfig import ELEMENTOS

    assert set(ELEMENTOS) == set(_POR_NOME)


def test_solver_options_metodo_e_precondicionador():
    o = solver_options({"solver": {"metodo": "minres", "precondicionador": "bloco_diagonal",
                                   "chebyshev": {"laplaciano": 30}, "verificar_direto": 10}})
    assert (o.metodo, o.precondicionador, o.verificar_direto) == ("minres", "bloco_diagonal", 10)
    assert o.chebyshev == (5, 5, 30)  # passos padrao do fsai


def test_solver_options_padroes_por_metodo_e_subsolver():
    o = solver_options({"solver": {"metodo": "bicgstab"}})
    assert (o.precondicionador, o.subsolver) == ("bloco_triangular", "fsai")
    o = solver_options({"solver": {"subsolver": "chebyshev"}})
    assert o.chebyshev == (40, 5, 80)
    o = solver_options({"solver": {"metodo": "bicgstab", "precondicionador": "nenhum"}})  # o padrao antigo
    assert (o.metodo, o.precondicionador) == ("bicgstab", "nenhum")


@pytest.mark.parametrize("solver, trecho", [
    ({"metodo": "gmres"}, "metodo"),
    ({"precondicionador": "ilu"}, "precondicionador"),
    ({"metodo": "minres", "precondicionador": "bloco_triangular"}, "simetrico"),
    ({"chebyshev": {"velocidade": 0}}, "velocidade"),
    ({"chebyshev": {"outro": 3}}, "chebyshev"),
    ({"verificar_direto": -1}, "verificar_direto"),
])
def test_solver_options_rejeita_opcoes_do_solver_linear(solver, trecho):
    with pytest.raises(ValueError, match=trecho):
        solver_options({"solver": solver})


def test_precondicionadores_do_config_batem_com_o_modulo():
    from femns.precondicionadores import NOMES
    from femns.simconfig import PRECONDICIONADORES

    assert set(NOMES) == set(PRECONDICIONADORES)


def test_subsolvers_do_config_batem_com_o_modulo():
    from femns.precondicionadores import SUBSOLVERS as do_modulo
    from femns.simconfig import SUBSOLVERS

    assert set(do_modulo) == set(SUBSOLVERS)

