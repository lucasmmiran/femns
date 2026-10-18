"""Lancamento e acompanhamento de simulacoes disparadas pela GUI.

Cada job roda `scripts/run_simulation.py` como subprocesso (mesmo caminho do
CLI, sem duplicar a logica de simulacao aqui) contra um config `.yaml` gerado
a partir do JSON recebido do formulario web. Progresso e medido contando
frames `solucao-N.vtk` escritos em `output_dir` -- robusto e independente de
conseguir (ou nao) parsear a barra de progresso do `tqdm` do subprocesso.
"""

import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid

import yaml

from femns.io import listar_frames
from femns.simconfig import DEFAULT_VTK_INTERVAL, validar_config

from .results import _run_id as results_run_id


class ConfigError(ValueError):
    """Config invalido recebido do formulario web (nao deve nem tentar rodar)."""


def slugify(texto: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", texto.strip()).strip("-").lower()
    return slug or "sim"


def expected_frame_count(iterations: int, vtk_interval: int) -> int:
    """Quantos frames `solucao-N.vtk` a simulacao vai escrever.

    `run_simulation.py` grava um a cada `vtk_interval` iteracoes e a ultima
    sempre -- entao o ultimo "balde" parcial conta, dai o `ceil`. Usado como
    denominador da barra de progresso (contar iteracoes daria uma barra que
    trava em ~1/vtk_interval).
    """
    return math.ceil(iterations / vtk_interval)


def validate_config(cfg: dict, meshes_root: str) -> None:
    """Valida antes de gravar o yaml e disparar o subprocesso.

    As regras do config sao as de `femns.simconfig.validar_config` (as mesmas
    que o `run_simulation.py` aplica); aqui so se acrescenta o que e' da GUI:
    a malha tem que ser um arquivo existente **dentro** de `meshes/` (o nome
    vem do cliente -- nao aceitar caminho para fora).
    """
    try:
        validar_config(cfg)
    except ValueError as e:
        raise ConfigError(str(e)) from e

    mesh_abs = os.path.abspath(os.path.join(meshes_root, os.path.basename(cfg["mesh"])))
    if not mesh_abs.startswith(os.path.abspath(meshes_root) + os.sep) or not os.path.isfile(mesh_abs):
        raise ConfigError(f"malha desconhecida: {cfg['mesh']!r}")


# Linha de progresso do tqdm de `run_simulation.py`, ex.:
#   " 45%|####5     | 450/1000 [01:23<01:40,  5.40it/s, bicg_iters=12, residual=9.8e-09]"
# grupos: iteracao, total, decorrido, restante ("?" no primeiro quadro).
_TQDM = re.compile(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+)")
_TEMPO_FINAL = re.compile(r"Tempo final: ([\d.]+) segundos")
# nome dos arquivos de um job: "<aaaammdd-hhmmss>-<slug>-<hex6>"
_NOME_JOB = re.compile(r"^(\d{8}-\d{6})-(.*)-[0-9a-f]{6}$")
# log de job sem processo (servidor reiniciado) escrito ha menos que isso conta
# como simulacao ainda rodando (o subprocesso tem sessao propria e sobrevive
# ao servidor -- ver `start_new_session` em `JobManager.launch`)
_LOG_VIVO_S = 120
# bytes lidos do fim do log a cada consulta de status: cobre as 40 linhas do
# `log_tail` e a ultima barra do tqdm sem reler um log de horas inteiro
_CAUDA_BYTES = 32 * 1024


def _segundos(hms: str) -> float | None:
    """'1:56:23' / '01:23' -> segundos; None para o '?' que o tqdm usa antes de estimar."""
    if "?" in hms:
        return None
    total = 0
    for parte in hms.split(":"):
        total = 60 * total + int(parte)
    return float(total)


def progresso_do_log(texto: str) -> dict:
    """Iteracao atual, tempo decorrido e restante a partir do texto do log.

    Usa a ultima barra do tqdm; `Tempo final` (impresso por
    `run_simulation.py` ao terminar) tem precedencia no decorrido. Campos
    ausentes ficam None -- ex. antes do primeiro passo de tempo.
    """
    barras = _TQDM.findall(texto)
    out = {"iteration": None, "elapsed_s": None, "eta_s": None, "finished": False}
    if barras:
        it, _total, decorrido, restante = barras[-1]
        out["iteration"] = int(it)
        out["elapsed_s"] = _segundos(decorrido)
        out["eta_s"] = _segundos(restante)
    final = _TEMPO_FINAL.findall(texto)
    if final:
        out["elapsed_s"] = float(final[-1])
        out["eta_s"] = 0.0
        out["finished"] = True
    return out


def _ler_cauda(path: str, nbytes: int = _CAUDA_BYTES) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - nbytes))
            return f.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


class Job:
    """Uma simulacao da GUI: lancada nesta sessao do servidor (`process`) ou
    reconstruida dos arquivos de uma sessao anterior (`process=None`)."""

    def __init__(self, job_id: str, label: str, config_path: str, output_dir: str, log_path: str,
                 iterations: int, vtk_interval: int, process: subprocess.Popen | None,
                 config: dict | None = None, started_at: float | None = None):
        self.id = job_id
        self.label = label
        self.config_path = config_path
        self.output_dir = output_dir
        self.log_path = log_path
        self.iterations = iterations
        self.vtk_interval = vtk_interval
        self.expected_frames = expected_frame_count(iterations, vtk_interval)
        self.process = process
        self.config = config or {}
        self.started_at = time.time() if started_at is None else started_at
        self._status_final = None  # cache: um job terminado nao muda mais

    def frames_done(self) -> int:
        return len(listar_frames(self.output_dir))

    def _estado(self, progresso: dict) -> tuple[str, int | None]:
        if self.process is not None:
            returncode = self.process.poll()
            if returncode is None:
                return "running", None
            return ("finished" if returncode == 0 else "failed"), returncode
        # sem processo: deduz do log
        if progresso["finished"]:
            return "finished", None
        try:
            idade = time.time() - os.path.getmtime(self.log_path)
        except OSError:
            idade = float("inf")
        return ("running" if idade < _LOG_VIVO_S else "failed"), None

    def status(self) -> dict:
        if self._status_final is not None:
            return self._status_final

        cauda = _ler_cauda(self.log_path)
        progresso = progresso_do_log(cauda)
        estado, returncode = self._estado(progresso)
        elapsed = progresso["elapsed_s"]
        if elapsed is None and estado == "running" and self.process is not None:
            elapsed = time.time() - self.started_at

        st = {
            "id": self.id,
            "label": self.label,
            "status": estado,
            "returncode": returncode,
            # cancelar so funciona para processo lancado por esta sessao do servidor
            "cancellable": estado == "running" and self.process is not None,
            "iterations": self.iterations,
            "iteration": progresso["iteration"],
            "elapsed_s": elapsed,
            "eta_s": progresso["eta_s"] if estado == "running" else None,
            "vtk_interval": self.vtk_interval,
            "expected_frames": self.expected_frames,
            "frames_done": self.frames_done(),
            "output_dir": self.output_dir,
            "results_id": results_run_id(self.output_dir),
            "config_path": self.config_path,
            "config": self.config,
            "started_at": self.started_at,
            "log_tail": self._log_tail(cauda),
        }
        if estado != "running":
            self._status_final = st
        return st

    def _log_tail(self, texto: str | None = None, n_linhas: int = 40) -> str:
        if texto is None:
            texto = _ler_cauda(self.log_path)
        # tqdm escreve progresso com \r (sem \n) -- so a ultima "linha" (por \r
        # ou \n) de cada bloco importa, o resto e escreve-reescreve da barra.
        blocos = re.split(r"[\r\n]+", texto)
        blocos = [b for b in blocos if b.strip()]
        return "\n".join(blocos[-n_linhas:])


class JobManager:
    def __init__(self, repo_root: str):
        self.repo_root = repo_root
        self.configs_dir = os.path.join(repo_root, "configs", "gui")
        self.output_root = os.path.join(repo_root, "solucoes", "gui")
        self.logs_dir = os.path.join(repo_root, "solucoes", "gui", "_logs")
        os.makedirs(self.configs_dir, exist_ok=True)
        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._carregar_historico()

    def _meta_path(self, name: str) -> str:
        return os.path.join(self.logs_dir, f"{name}.json")

    def _carregar_historico(self) -> None:
        """Reconstroi os jobs de sessoes anteriores do servidor a partir dos arquivos.

        Cada job deixa `configs/gui/<nome>.yaml`, `solucoes/gui/<nome>/` e
        `solucoes/gui/_logs/<nome>.log` (+ `<nome>.json` com o rotulo, desde a
        GUI de 18/10). O estado sai do log (ver `Job._estado`). Config que nao
        abre e' pulado -- o historico nunca impede o servidor de subir.
        """
        for config_path in sorted(glob.glob(os.path.join(self.configs_dir, "*.yaml"))):
            name = os.path.splitext(os.path.basename(config_path))[0]
            try:
                with open(config_path) as f:
                    cfg = yaml.safe_load(f) or {}
                sim = cfg.get("simulation") or {}
                iterations = int(sim["iterations"])
                vtk_interval = int(sim.get("vtk_interval", DEFAULT_VTK_INTERVAL))
            except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError):
                continue

            meta = {}
            try:
                with open(self._meta_path(name)) as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
            m = _NOME_JOB.match(name)
            label = meta.get("label") or (m.group(2) if m else name)
            started_at = meta.get("started_at")
            if started_at is None:
                try:
                    started_at = time.mktime(time.strptime(m.group(1), "%Y%m%d-%H%M%S")) if m else None
                except ValueError:
                    started_at = None
            if started_at is None:
                started_at = os.path.getmtime(config_path)

            job_id = hashlib.sha1(name.encode()).hexdigest()[:12]
            self._jobs[job_id] = Job(
                job_id, label, config_path, os.path.join(self.output_root, name),
                os.path.join(self.logs_dir, f"{name}.log"), iterations, vtk_interval, None,
                config=cfg, started_at=started_at,
            )

    def launch(self, cfg: dict, label: str | None = None) -> Job:
        meshes_root = os.path.join(self.repo_root, "meshes")
        validate_config(cfg, meshes_root)

        slug = slugify(label or os.path.splitext(cfg["mesh"])[0])
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}-{slug}-{uuid.uuid4().hex[:6]}"

        cfg = dict(cfg)
        cfg["mesh"] = os.path.join("meshes", os.path.basename(cfg["mesh"]))
        output_dir = os.path.join(self.output_root, name)
        cfg["output_dir"] = os.path.relpath(output_dir, self.repo_root)
        cfg.pop("benchmark_xlsx", None)  # runs da GUI nao entram na planilha de benchmark do CLI

        config_path = os.path.join(self.configs_dir, f"{name}.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(self.logs_dir, f"{name}.log")
        logfile = open(log_path, "w")  # noqa: SIM115 - fica aberto para o subprocesso, fechado no cancel/gc do Popen
        process = subprocess.Popen(
            # `-u`: sem isso o stdout do subprocesso vai em blocos de 8 KB e se
            # perde inteiro quando ele morre de forma dura (sinal, falha de
            # GPU) -- o log fica em 0 byte justo no caso em que seria util.
            [sys.executable, "-u", "scripts/run_simulation.py", "--config", config_path],
            cwd=self.repo_root,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            # Sessao propria: sem isso o subprocesso fica no mesmo process
            # group do servidor, e um Ctrl+C no terminal do `femns-gui` (ou
            # fechar o terminal, via SIGHUP) mata junto toda simulacao em
            # andamento, sem deixar rastro no log.
            start_new_session=True,
        )

        job = Job(
            uuid.uuid4().hex[:12], label or slug, config_path, output_dir, log_path,
            cfg["simulation"]["iterations"], cfg["simulation"].get("vtk_interval", DEFAULT_VTK_INTERVAL), process,
            config=cfg,
        )
        with open(self._meta_path(name), "w") as f:
            json.dump({"label": job.label, "started_at": job.started_at}, f)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.process is None or job.process.poll() is not None:
            return False
        job.process.terminate()
        try:
            job.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            job.process.kill()
        return True
