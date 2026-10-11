"""Lancamento e acompanhamento de simulacoes disparadas pela GUI.

Cada job roda `scripts/run_simulation.py` como subprocesso (mesmo caminho do
CLI, sem duplicar a logica de simulacao aqui) contra um config `.yaml` gerado
a partir do JSON recebido do formulario web. Progresso e medido contando
frames `solucao-N.vtk` escritos em `output_dir` -- robusto e independente de
conseguir (ou nao) parsear a barra de progresso do `tqdm` do subprocesso.
"""

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


class Job:
    def __init__(self, job_id: str, label: str, config_path: str, output_dir: str, log_path: str,
                 iterations: int, vtk_interval: int, process: subprocess.Popen):
        self.id = job_id
        self.label = label
        self.config_path = config_path
        self.output_dir = output_dir
        self.log_path = log_path
        self.iterations = iterations
        self.vtk_interval = vtk_interval
        self.expected_frames = expected_frame_count(iterations, vtk_interval)
        self.process = process
        self.started_at = time.time()

    def frames_done(self) -> int:
        return len(listar_frames(self.output_dir))

    def status(self) -> dict:
        returncode = self.process.poll()
        if returncode is None:
            estado = "running"
        elif returncode == 0:
            estado = "finished"
        else:
            estado = "failed"

        return {
            "id": self.id,
            "label": self.label,
            "status": estado,
            "returncode": returncode,
            "iterations": self.iterations,
            "vtk_interval": self.vtk_interval,
            "expected_frames": self.expected_frames,
            "frames_done": self.frames_done(),
            "output_dir": self.output_dir,
            "config_path": self.config_path,
            "started_at": self.started_at,
            "log_tail": self._log_tail(),
        }

    def _log_tail(self, n_linhas: int = 40) -> str:
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                linhas = f.readlines()
        except FileNotFoundError:
            return ""
        # tqdm escreve progresso com \r (sem \n) -- so a ultima "linha" (por \r
        # ou \n) de cada bloco importa, o resto e escreve-reescreve da barra.
        texto = "".join(linhas)
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
        )
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
        if job is None or job.process.poll() is not None:
            return False
        job.process.terminate()
        try:
            job.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            job.process.kill()
        return True
