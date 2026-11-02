"""Teste de fumaca ponta a ponta do laco de `scripts/run_simulation.py` (A1).

Chama `run_simulation.main()` de verdade, em CPU, por 3 passos, e compara
normas dos campos gravados com uma referencia em
`tests/referencias/laco_ponta_a_ponta.json`. Os dois casos cobrem os dois
caminhos de malha movel do laco:

- `poiseuille`: MINI + `mesh_motion` (oscilacao sintetica, remontagem a cada passo);
- `valvula`: Tri6 + `geometria_movel` (rotacao, Dirichlet movel na parede,
  extensao harmonica do interior, `beta_por_passo`).

E' a rede de seguranca para refatorar `main()` (A2/A3): uma mudanca so de
estrutura tem que passar sem mexer na referencia. Se a mudanca alterar a
numerica de proposito, regrave com

    FEMNS_REGRAVAR_REFERENCIA=1 pytest tests/test_laco_ponta_a_ponta.py

e diga no commit por que a referencia mudou.
"""

import importlib.util
import json
import os

import numpy as np
import pytest
import torch
import yaml

from femns.io import listar_frames
from femns.mesh import read_mesh

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCIA = os.path.join(RAIZ, "tests", "referencias", "laco_ponta_a_ponta.json")
REGRAVAR = os.environ.get("FEMNS_REGRAVAR_REFERENCIA") == "1"
PASSOS = 3

# Tolerancia relativa a escala do campo no frame. O MINRES para em tol=1e-8,
# entao duas execucoes que diferem so no arredondamento (outra BLAS, outro
# numero de threads) ficam bem abaixo disto; um erro de indice ou de sinal fica
# muito acima.
RTOL = 1e-6

CASOS = {
    "poiseuille": ("configs/poiseuille_test.yaml", {}),
    # a esfera gira 10 -> 20 graus em 0,03 s (no config sao 8 s): em 3 passos ela
    # anda ~0,24 grau, o bastante para a parede, a extensao e a remontagem pesarem
    "valvula": ("configs/valvula_10g.yaml", {"geometria_movel.corpos.esfera.duracao": 0.03}),
}


def _run_simulation():
    spec = importlib.util.spec_from_file_location("run_simulation", os.path.join(RAIZ, "scripts", "run_simulation.py"))
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _definir(cfg, chave, valor):
    *caminho, ultima = chave.split(".")
    for parte in caminho:
        cfg = cfg[parte]
    cfg[ultima] = valor


def _metricas(output_dir, malha):
    """Normas por frame: rms, max|.| e media (com sinal) de vx, vy, p, e o deslocamento rms da malha."""
    import meshio

    pontos0 = malha.raw.points[:, :2]
    metricas = {}
    for n, caminho in listar_frames(output_dir).items():
        frame = meshio.read(caminho)
        m = {}
        for campo in ("vx", "vy", "p"):
            v = np.asarray(frame.point_data[campo], dtype=np.float64)
            m[f"{campo}_rms"] = float(np.sqrt(np.mean(v**2)))
            m[f"{campo}_max"] = float(np.abs(v).max())
            m[f"{campo}_media"] = float(v.mean())  # com sinal: pega campo de sinal trocado
        m["desloc_rms"] = float(np.sqrt(np.mean(np.sum((frame.points[:, :2] - pontos0) ** 2, axis=1))))
        metricas[str(n)] = m
    return metricas


def _rodar(nome, tmp_path, monkeypatch):
    config, sobrescritas = CASOS[nome]
    with open(os.path.join(RAIZ, config)) as f:
        cfg = yaml.safe_load(f)
    cfg["simulation"]["iterations"] = PASSOS
    cfg["simulation"]["vtk_interval"] = 1
    cfg["output_dir"] = str(tmp_path / "saida")
    cfg.pop("benchmark_xlsx", None)
    cfg.pop("historico_csv", None)
    for chave, valor in sobrescritas.items():
        _definir(cfg, chave, valor)
    caminho_cfg = tmp_path / "config.yaml"
    caminho_cfg.write_text(yaml.safe_dump(cfg))

    modulo = _run_simulation()
    monkeypatch.chdir(RAIZ)  # a malha do config e' relativa a raiz
    monkeypatch.setattr("sys.argv", ["run_simulation.py", "--config", str(caminho_cfg)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # CPU, igual a CI
    modulo.main()
    return _metricas(cfg["output_dir"], read_mesh(os.path.join(RAIZ, cfg["mesh"])))


@pytest.mark.parametrize("nome", list(CASOS))
def test_laco_bate_com_a_referencia(nome, tmp_path, monkeypatch):
    obtido = _rodar(nome, tmp_path, monkeypatch)
    assert list(obtido) == [str(n) for n in range(1, PASSOS + 1)]

    referencias = {}
    if os.path.exists(REFERENCIA):
        with open(REFERENCIA) as f:
            referencias = json.load(f)
    if REGRAVAR:
        referencias[nome] = obtido
        os.makedirs(os.path.dirname(REFERENCIA), exist_ok=True)
        with open(REFERENCIA, "w") as f:
            json.dump(referencias, f, indent=2, sort_keys=True)
            f.write("\n")
        pytest.skip(f"referencia de {nome!r} regravada em {REFERENCIA}")

    assert nome in referencias, f"sem referencia para {nome!r}: rode com FEMNS_REGRAVAR_REFERENCIA=1"
    ref = referencias[nome]
    assert list(obtido) == list(ref)
    for passo, m_ref in ref.items():
        m = obtido[passo]
        assert set(m) == set(m_ref)
        escala_v = max(m_ref["vx_max"], m_ref["vy_max"])
        for chave, valor_ref in m_ref.items():
            escala = {"v": escala_v, "p": m_ref["p_max"], "d": m_ref["desloc_rms"]}[chave[0]]
            assert abs(m[chave] - valor_ref) <= RTOL * escala, (
                f"{nome}, passo {passo}, {chave}: {m[chave]!r} != referencia {valor_ref!r} "
                f"(diferenca {abs(m[chave] - valor_ref):.3e}, limite {RTOL * escala:.3e})")
