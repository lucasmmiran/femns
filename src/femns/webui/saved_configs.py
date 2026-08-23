"""Configuracoes salvas pelo usuario na tela de nova simulacao (botao "Salvar configuracao").

Distinto de `jobs.py`: um config salvo aqui nunca dispara subprocesso nem
grava em `output_dir` -- e so um preset nomeado, editavel, que a tela
recarrega pra preencher o formulario de novo (ver `POST/GET/DELETE
/api/saved-configs` em `server.py`). Tambem distinto dos `configs/*.yaml`
"template" do projeto (`server.py:api_configs`): aqueles sao presets
curados e versionados; os daqui sao locais do usuario (`configs/gui_saved/`
e git-ignorado, mesmo tratamento de `configs/gui/`).
"""

import glob
import os

import yaml

from .jobs import ConfigError, slugify


def save_config(root: str, name: str, config: dict) -> str:
    """Grava `config` em `root/<slug(name)>.yaml`; um nome ja usado sobrescreve (save-as)."""
    if not name or not name.strip():
        raise ConfigError("nome obrigatorio para salvar a configuracao")
    os.makedirs(root, exist_ok=True)
    slug = slugify(name)
    path = os.path.join(root, f"{slug}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump({"name": name, "config": config}, f, allow_unicode=True, sort_keys=False)
    return slug


def list_saved_configs(root: str) -> list[dict]:
    if not os.path.isdir(root):
        return []
    out = []
    for path in sorted(glob.glob(os.path.join(root, "*.yaml"))):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        slug = os.path.splitext(os.path.basename(path))[0]
        out.append({"slug": slug, "name": data.get("name", slug), "content": data.get("config", {})})
    return out


def delete_saved_config(root: str, slug: str) -> bool:
    """`slug` ja vem restrito a `[a-z0-9_-]+` pelo padrao da rota em `server.py` -- sem risco de path traversal."""
    path = os.path.join(root, f"{slug}.yaml")
    if not os.path.isfile(path):
        return False
    os.remove(path)
    return True
