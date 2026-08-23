import pytest

from femns.webui.jobs import ConfigError, slugify, validate_config


@pytest.fixture
def meshes_root(tmp_path):
    root = tmp_path / "meshes"
    root.mkdir()
    (root / "poiseuille.msh").write_text("fake")
    return str(root)


def config_valido(mesh="poiseuille.msh"):
    return {
        "mesh": mesh,
        "simulation": {"dt": 0.01, "iterations": 10, "reynolds": 1.0},
        "boundary": {
            "priority": ["inlet"],
            "conditions": {"inlet": {"vx": 1.0, "vy": 0.0}},
        },
    }


def test_slugify_troca_caracteres_especiais():
    assert slugify("Degrau Teste #1!") == "degrau-teste-1"


def test_slugify_string_vazia_vira_sim():
    assert slugify("   ") == "sim"


def test_validate_config_aceita_config_valido(meshes_root):
    validate_config(config_valido(), meshes_root)  # nao deve levantar


@pytest.mark.parametrize("chave", ["mesh"])
def test_validate_config_rejeita_mesh_ausente(meshes_root, chave):
    cfg = config_valido()
    del cfg[chave]
    with pytest.raises(ConfigError):
        validate_config(cfg, meshes_root)


def test_validate_config_rejeita_malha_desconhecida(meshes_root):
    with pytest.raises(ConfigError):
        validate_config(config_valido(mesh="nao_existe.msh"), meshes_root)


def test_validate_config_rejeita_path_traversal_na_malha(meshes_root):
    with pytest.raises(ConfigError):
        validate_config(config_valido(mesh="../../etc/passwd"), meshes_root)


@pytest.mark.parametrize("campo,valor", [("dt", 0), ("dt", -1), ("iterations", 0), ("iterations", 1.5), ("reynolds", -1)])
def test_validate_config_rejeita_parametro_numerico_invalido(meshes_root, campo, valor):
    cfg = config_valido()
    cfg["simulation"][campo] = valor
    with pytest.raises(ConfigError):
        validate_config(cfg, meshes_root)


@pytest.mark.parametrize("campo,valor", [("advection", "bogus"), ("element", "bogus"), ("sl_boundary", "bogus")])
def test_validate_config_rejeita_enum_invalido(meshes_root, campo, valor):
    cfg = config_valido()
    cfg["simulation"][campo] = valor
    with pytest.raises(ConfigError):
        validate_config(cfg, meshes_root)


def test_validate_config_rejeita_boundary_ausente(meshes_root):
    cfg = config_valido()
    cfg["boundary"] = {}
    with pytest.raises(ConfigError):
        validate_config(cfg, meshes_root)
