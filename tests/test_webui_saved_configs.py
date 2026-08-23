import pytest

from femns.webui.jobs import ConfigError
from femns.webui.saved_configs import delete_saved_config, list_saved_configs, save_config


def config_exemplo():
    return {
        "mesh": "poiseuille.msh",
        "simulation": {"dt": 0.01, "iterations": 10, "reynolds": 1.0},
        "boundary": {"priority": ["inlet"], "conditions": {"inlet": {"vx": 1.0}}},
    }


def test_list_saved_configs_diretorio_inexistente_retorna_vazio(tmp_path):
    assert list_saved_configs(str(tmp_path / "nao_existe")) == []


def test_save_e_list_roundtrip(tmp_path):
    slug = save_config(str(tmp_path), "Minha Config", config_exemplo())

    saved = list_saved_configs(str(tmp_path))

    assert len(saved) == 1
    assert saved[0]["slug"] == slug
    assert saved[0]["name"] == "Minha Config"
    assert saved[0]["content"] == config_exemplo()


def test_save_config_nome_vazio_levanta_configerror(tmp_path):
    with pytest.raises(ConfigError):
        save_config(str(tmp_path), "   ", config_exemplo())


def test_save_config_mesmo_nome_sobrescreve(tmp_path):
    slug1 = save_config(str(tmp_path), "Config A", config_exemplo())
    cfg2 = config_exemplo()
    cfg2["simulation"]["dt"] = 0.05
    slug2 = save_config(str(tmp_path), "Config A", cfg2)

    saved = list_saved_configs(str(tmp_path))

    assert slug1 == slug2
    assert len(saved) == 1
    assert saved[0]["content"]["simulation"]["dt"] == 0.05


def test_save_config_nomes_diferentes_geram_slugs_diferentes(tmp_path):
    save_config(str(tmp_path), "Config A", config_exemplo())
    save_config(str(tmp_path), "Config B", config_exemplo())

    saved = list_saved_configs(str(tmp_path))

    assert {s["name"] for s in saved} == {"Config A", "Config B"}
    assert len({s["slug"] for s in saved}) == 2


def test_delete_saved_config_remove_arquivo(tmp_path):
    slug = save_config(str(tmp_path), "Config A", config_exemplo())

    assert delete_saved_config(str(tmp_path), slug) is True
    assert list_saved_configs(str(tmp_path)) == []


def test_delete_saved_config_inexistente_retorna_false(tmp_path):
    assert delete_saved_config(str(tmp_path), "nao-existe") is False
