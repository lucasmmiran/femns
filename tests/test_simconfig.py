import pytest

from femns.simconfig import DEFAULT_BICGSTAB_MAX_ITER, DEFAULT_BICGSTAB_TOL, solver_options


def test_solver_options_default_sem_bloco():
    assert solver_options({"dt": 0.01}) == (DEFAULT_BICGSTAB_TOL, DEFAULT_BICGSTAB_MAX_ITER, False)


def test_solver_options_bloco_vazio_cai_no_default():
    assert solver_options({"solver": {}}) == (DEFAULT_BICGSTAB_TOL, DEFAULT_BICGSTAB_MAX_ITER, False)


def test_solver_options_le_valores():
    assert solver_options({"solver": {"tol": 1e-6, "max_iter": 500}}) == (1e-6, 500, False)


def test_solver_options_um_valor_so_mantem_o_outro_no_default():
    assert solver_options({"solver": {"max_iter": 5000}}) == (DEFAULT_BICGSTAB_TOL, 5000, False)
    assert solver_options({"solver": {"tol": 1e-10}}) == (1e-10, DEFAULT_BICGSTAB_MAX_ITER, False)


def test_solver_options_beta_por_passo():
    assert solver_options({"solver": {"beta_por_passo": True}})[2] is True
    assert solver_options({"solver": {}})[2] is False


def test_solver_options_rejeita_beta_por_passo_nao_bool():
    import pytest as _pytest
    with _pytest.raises(ValueError, match="beta_por_passo"):
        solver_options({"solver": {"beta_por_passo": "sim"}})


def test_solver_options_tol_inteiro_vira_float():
    tol, _, _ = solver_options({"solver": {"tol": 1}})
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
