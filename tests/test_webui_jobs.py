import pytest

from femns.webui.jobs import ConfigError, expected_frame_count, slugify, validate_config


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


@pytest.mark.parametrize("valor", [0, -1, 1.5, "10", True])
def test_validate_config_rejeita_vtk_interval_invalido(meshes_root, valor):
    cfg = config_valido()
    cfg["simulation"]["vtk_interval"] = valor
    with pytest.raises(ConfigError):
        validate_config(cfg, meshes_root)


def test_validate_config_aceita_sem_vtk_interval(meshes_root):
    cfg = config_valido()
    assert "vtk_interval" not in cfg["simulation"]
    validate_config(cfg, meshes_root)  # opcional -- cai no default 10


@pytest.mark.parametrize("iterations,intervalo,esperado", [
    (1000, 10, 100),   # multiplo exato: 10, 20, ..., 1000
    (95, 10, 10),       # 10..90 (9) + a ultima (95) = 10
    (7, 10, 1),         # so a ultima
    (200, 1, 200),      # todo passo
    (1, 5, 1),
])
def test_expected_frame_count(iterations, intervalo, esperado):
    assert expected_frame_count(iterations, intervalo) == esperado


@pytest.mark.parametrize("campo,valor", [("advection", "bogus"), ("advection", "eulerian"),  # eulerian: removido na v2.0
                                         ("element", "bogus"), ("sl_boundary", "bogus"),
                                         ("sl_boundary", "dirichlet")])  # dirichlet: removido na v2.0
def test_validate_config_rejeita_enum_invalido(meshes_root, campo, valor):
    cfg = config_valido()
    cfg["simulation"][campo] = valor
    with pytest.raises(ConfigError):
        validate_config(cfg, meshes_root)


def test_validate_config_aceita_bloco_solver(meshes_root):
    cfg = config_valido()
    cfg["simulation"]["solver"] = {"tol": 1e-6, "max_iter": 500}
    validate_config(cfg, meshes_root)  # nao deve levantar


@pytest.mark.parametrize("solver", [
    {"tol": 0},
    {"tol": "1e-8"},
    {"max_iter": -1},
    {"max_iter": 1.5},
    {"precond": "jacobi"},
])
def test_validate_config_rejeita_bloco_solver_invalido(meshes_root, solver):
    cfg = config_valido()
    cfg["simulation"]["solver"] = solver
    with pytest.raises(ConfigError):
        validate_config(cfg, meshes_root)


def test_validate_config_rejeita_boundary_ausente(meshes_root):
    cfg = config_valido()
    cfg["boundary"] = {}
    with pytest.raises(ConfigError):
        validate_config(cfg, meshes_root)


def test_validate_config_rejeita_moving_point(meshes_root):
    cfg = config_valido()
    cfg["moving_point"] = {"enabled": True, "node_index": 0}
    with pytest.raises(ConfigError, match="moving_point"):
        validate_config(cfg, meshes_root)
    cfg["moving_point"]["enabled"] = False  # bloco desligado de config antigo: aceito (ignorado)
    validate_config(cfg, meshes_root)



# -- progresso pelo log (tqdm) e historico de sessoes anteriores -------------

from femns.webui.jobs import JobManager, progresso_do_log  # noqa: E402


def test_progresso_do_log_usa_a_ultima_barra_do_tqdm():
    texto = (" 10%|#  | 100/1000 [00:20<03:00,  5.00it/s]\r"
             " 45%|#### | 450/1000 [1:01:23<01:40,  5.40it/s, bicg_iters=12, residual=9.8e-09]\r")
    p = progresso_do_log(texto)
    assert p == {"iteration": 450, "elapsed_s": 3683.0, "eta_s": 100.0, "finished": False}


def test_progresso_do_log_sem_estimativa_e_sem_barra():
    assert progresso_do_log("  0%|          | 0/1000 [00:00<?, ?it/s]")["eta_s"] is None
    p = progresso_do_log("Leitura de malha completa:\n")
    assert p["iteration"] is None and p["elapsed_s"] is None and not p["finished"]


def test_progresso_do_log_tempo_final_marca_concluida():
    texto = "100%|##| 8000/8000 [3:33:27<00:00,  1.60s/it]\nTempo final: 12810.57 segundos\n"
    p = progresso_do_log(texto)
    assert p["finished"] and p["iteration"] == 8000 and p["elapsed_s"] == 12810.57 and p["eta_s"] == 0.0


def _job_antigo(repo, name, cfg, log, meta=None):
    (repo / "configs" / "gui").mkdir(parents=True, exist_ok=True)
    (repo / "solucoes" / "gui" / "_logs").mkdir(parents=True, exist_ok=True)
    import yaml
    (repo / "configs" / "gui" / f"{name}.yaml").write_text(yaml.safe_dump(cfg))
    if log is not None:
        (repo / "solucoes" / "gui" / "_logs" / f"{name}.log").write_text(log)
    if meta is not None:
        import json
        (repo / "solucoes" / "gui" / "_logs" / f"{name}.json").write_text(json.dumps(meta))


def test_jobmanager_reconstroi_jobs_de_sessao_anterior(tmp_path):
    import os
    import time
    cfg = config_valido()
    _job_antigo(tmp_path, "20260901-194818-degrau-re800-042c02", cfg,
                "100%|##| 10/10 [00:05<00:00]\nTempo final: 5.1 segundos\n",
                meta={"label": "degrau-re800_20260901_1948", "started_at": 123.0})
    _job_antigo(tmp_path, "20260902-101500-lid-aaaaaa", cfg, " 30%|# | 3/10 [00:03<00:07]\r")
    log_velho = tmp_path / "solucoes" / "gui" / "_logs" / "20260902-101500-lid-aaaaaa.log"
    os.utime(log_velho, (time.time() - 3600, time.time() - 3600))  # sem atividade: morreu com o servidor
    (tmp_path / "configs" / "gui" / "quebrado.yaml").write_text("simulation: [")  # yaml invalido: pulado

    jobs = {j.label: j.status() for j in JobManager(str(tmp_path)).list()}

    assert set(jobs) == {"degrau-re800_20260901_1948", "lid"}  # sem meta: rotulo sai do nome
    concluido, interrompido = jobs["degrau-re800_20260901_1948"], jobs["lid"]
    assert concluido["status"] == "finished" and concluido["started_at"] == 123.0
    assert concluido["elapsed_s"] == 5.1 and concluido["iteration"] == 10
    assert concluido["config"]["simulation"]["dt"] == 0.01
    assert interrompido["status"] == "failed" and not interrompido["cancellable"]
    assert interrompido["iteration"] == 3 and interrompido["eta_s"] is None


def test_jobmanager_log_recente_sem_processo_conta_como_rodando(tmp_path):
    _job_antigo(tmp_path, "20260902-101500-lid-bbbbbb", config_valido(), " 30%|# | 3/10 [00:03<00:07]\r")
    (job,) = JobManager(str(tmp_path)).list()
    st = job.status()
    assert st["status"] == "running" and st["eta_s"] == 7.0
    assert not st["cancellable"]  # processo de outra sessao do servidor: nao da pra cancelar daqui
    assert not JobManager(str(tmp_path)).cancel(job.id)
