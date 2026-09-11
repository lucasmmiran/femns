"""Servidor web da GUI opcional do femns -- so biblioteca padrao (sem flask/fastapi).

Serve o frontend estatico (`static/`) e uma API JSON pequena que: lista
malhas/configs disponiveis, lanca simulacoes como subprocesso (`jobs.py`) e
serve os `.vtk` de saida como JSON pro visualizador (`results.py`). Nao e
importado por nenhum modulo do solver -- `scripts/run_simulation.py` continua
funcionando sem isso instalado/rodando (ver README, secao "Interface web").
"""

import hmac
import json
import mimetypes
import os
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import yaml

from . import results, saved_configs
from .jobs import ConfigError, JobManager
from .meshes import list_meshes, mesh_geometry

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
MESHES_DIR = os.path.join(REPO_ROOT, "meshes")
CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")
SOLUCOES_DIR = os.path.join(REPO_ROOT, "solucoes")
SAVED_CONFIGS_DIR = os.path.join(REPO_ROOT, "configs", "gui_saved")

_ROUTES = []  # (method, re.Pattern, handler_name)

# Enderecos que so aceitam conexao da propria maquina -- bind em qualquer
# um deles dispensa autenticacao (ver `check_bind`).
LOCAL_HOSTS = {"", "127.0.0.1", "localhost", "::1"}


def is_local_host(host: str) -> bool:
    return host in LOCAL_HOSTS


def check_token(expected: str | None, auth_header: str | None, query: dict) -> bool:
    """True se a requisicao pode acessar `/api/*`.

    `expected` vazio/`None` => servidor sem autenticacao, tudo liberado.
    Caso contrario exige `Authorization: Bearer <token>` **ou** `?token=<token>`
    igual a `expected` (comparacao de tempo constante).
    """
    if not expected:
        return True
    supplied = ""
    if auth_header and auth_header.startswith("Bearer "):
        supplied = auth_header[len("Bearer "):].strip()
    elif query.get("token"):
        supplied = query["token"][0]
    return bool(supplied) and hmac.compare_digest(supplied, expected)


def check_bind(host: str, token: str | None, allow_no_auth: bool) -> str | None:
    """Valida a combinacao host/token antes de subir o servidor.

    - bind local (127.0.0.1/localhost): nada a fazer, retorna `None`.
    - bind exposto **com** token: retorna um aviso (str) lembrando de HTTPS.
    - bind exposto **sem** token, com `allow_no_auth`: retorna um aviso forte.
    - bind exposto **sem** token, sem `allow_no_auth`: levanta `RuntimeError`
      com as opcoes (tunel SSH, definir `FEMNS_GUI_TOKEN`, ou `--allow-no-auth`).

    Ver `docs/planos/plano_remoto.md` para o passo a passo de acesso remoto.
    """
    if is_local_host(host):
        return None
    if token:
        return (f"femns webui vai aceitar conexoes da rede (bind em {host}) -- protegido por FEMNS_GUI_TOKEN. "
                f"Use HTTPS (proxy reverso) se o trafego passar por rede nao confiavel.")
    if allow_no_auth:
        return (f"ATENCAO: femns webui exposto em {host} SEM AUTENTICACAO (--allow-no-auth). "
                f"Qualquer um que alcance esta porta pode lancar/cancelar simulacao e ler resultados.")
    raise RuntimeError(
        f"bind em {host!r} expoe a GUI na rede e nao ha autenticacao configurada. Opcoes:\n"
        f"  - manter o bind local (127.0.0.1) e usar um tunel SSH  -- ver docs/planos/plano_remoto.md\n"
        f"  - definir um token:  FEMNS_GUI_TOKEN=<segredo> ./femns-gui --host {host}\n"
        f"  - por sua conta e risco (rede isolada/confiavel):  --allow-no-auth")


def route(method: str, pattern: str):
    compiled = re.compile(pattern)

    def registra(func):
        _ROUTES.append((method, compiled, func.__name__))
        return func

    return registra


class Handler(BaseHTTPRequestHandler):
    job_manager: JobManager = None  # setado por `make_server`
    auth_token: str | None = None  # setado por `make_server` a partir de FEMNS_GUI_TOKEN
    _results_index: dict = {}  # id -> dir, preenchido por /api/results
    _results_lock = threading.Lock()

    def log_message(self, fmt, *args):
        pass  # silencioso -- o log do subprocesso da simulacao ja vai pro arquivo em jobs.py

    # -- despacho -----------------------------------------------------

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def _dispatch(self, method):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # So as rotas /api/* sao protegidas: o shell estatico (HTML/CSS/JS)
        # precisa carregar sem token para conseguir mandar o token depois.
        if path.startswith("/api/") and not check_token(
            self.auth_token, self.headers.get("Authorization"), query
        ):
            self._json(401, {"error": "token ausente ou invalido (o servidor exige FEMNS_GUI_TOKEN)"})
            return

        for m, pattern, handler_name in _ROUTES:
            if m != method:
                continue
            match = pattern.fullmatch(path)
            if match:
                try:
                    getattr(self, handler_name)(match, query)
                except ConfigError as e:
                    self._json(400, {"error": str(e)})
                except json.JSONDecodeError as e:
                    self._json(400, {"error": f"JSON invalido no corpo da requisicao: {e}"})
                except FileNotFoundError as e:
                    self._json(404, {"error": str(e)})
                except Exception as e:  # noqa: BLE001 - qualquer outra falha vira 500 em vez de derrubar a thread
                    self._json(500, {"error": f"{type(e).__name__}: {e}"})
                return
        if method == "GET":
            self._serve_static(path)
        else:
            self._json(404, {"error": "not found"})

    # -- helpers --------------------------------------------------------

    def _json(self, code: int, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _serve_static(self, path: str):
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        full = os.path.abspath(os.path.join(STATIC_DIR, rel))
        if not full.startswith(os.path.abspath(STATIC_DIR) + os.sep) and full != os.path.abspath(STATIC_DIR):
            self._json(403, {"error": "forbidden"})
            return
        if not os.path.isfile(full):
            self._json(404, {"error": "not found"})
            return
        ctype, _ = mimetypes.guess_type(full)
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _run_dir(self, run_id: str) -> str:
        with self._results_lock:
            entry = self._results_index.get(run_id)
        if entry is None:
            raise FileNotFoundError(f"run desconhecido: {run_id!r} (recarregue a lista de resultados)")
        return entry

    # -- rotas: malhas/configs -------------------------------------------

    @route("GET", r"/api/meshes")
    def api_meshes(self, match, query):
        self._json(200, list_meshes(MESHES_DIR))

    @route("GET", r"/api/meshes/(?P<name>[^/]+)/geometry")
    def api_mesh_geometry(self, match, query):
        name = match.group("name")
        mesh_abs = os.path.abspath(os.path.join(MESHES_DIR, os.path.basename(name)))
        if not mesh_abs.startswith(os.path.abspath(MESHES_DIR) + os.sep) or not os.path.isfile(mesh_abs):
            raise FileNotFoundError(f"malha desconhecida: {name!r}")
        self._json(200, mesh_geometry(mesh_abs))

    @route("GET", r"/api/configs")
    def api_configs(self, match, query):
        out = []
        for nome in sorted(os.listdir(CONFIGS_DIR)):
            if not nome.endswith(".yaml") or nome == "gui":
                continue
            caminho = os.path.join(CONFIGS_DIR, nome)
            if not os.path.isfile(caminho):
                continue
            with open(caminho) as f:
                conteudo = yaml.safe_load(f)
            out.append({"name": nome, "content": conteudo})
        self._json(200, out)

    # -- rotas: configs salvas pelo usuario (ver saved_configs.py) --------

    @route("GET", r"/api/saved-configs")
    def api_saved_configs_list(self, match, query):
        self._json(200, saved_configs.list_saved_configs(SAVED_CONFIGS_DIR))

    @route("POST", r"/api/saved-configs")
    def api_saved_configs_save(self, match, query):
        body = self._read_json_body()
        slug = saved_configs.save_config(SAVED_CONFIGS_DIR, body.get("name", ""), body.get("config", {}))
        self._json(200, {"slug": slug, "name": body.get("name", "")})

    @route("DELETE", r"/api/saved-configs/(?P<slug>[a-z0-9_-]+)")
    def api_saved_configs_delete(self, match, query):
        ok = saved_configs.delete_saved_config(SAVED_CONFIGS_DIR, match.group("slug"))
        self._json(200, {"deleted": ok})

    # -- rotas: simulacoes -------------------------------------------------

    @route("POST", r"/api/simulations")
    def api_simulations_launch(self, match, query):
        body = self._read_json_body()
        job = self.job_manager.launch(body.get("config", {}), label=body.get("label"))
        self._json(200, job.status())

    @route("GET", r"/api/simulations")
    def api_simulations_list(self, match, query):
        jobs = sorted(self.job_manager.list(), key=lambda j: j.started_at, reverse=True)
        self._json(200, [j.status() for j in jobs])

    @route("GET", r"/api/simulations/(?P<job_id>[a-f0-9]+)")
    def api_simulations_get(self, match, query):
        job = self.job_manager.get(match.group("job_id"))
        if job is None:
            self._json(404, {"error": "job desconhecido"})
            return
        self._json(200, job.status())

    @route("POST", r"/api/simulations/(?P<job_id>[a-f0-9]+)/cancel")
    def api_simulations_cancel(self, match, query):
        ok = self.job_manager.cancel(match.group("job_id"))
        self._json(200, {"cancelled": ok})

    # -- rotas: resultados -------------------------------------------------

    @route("GET", r"/api/results")
    def api_results_list(self, match, query):
        runs = results.scan_runs(SOLUCOES_DIR)
        with self._results_lock:
            self._results_index.update({rid: info["dir"] for rid, info in runs.items()})
        out = [{"id": rid, **{k: v for k, v in info.items() if k != "dir"}} for rid, info in runs.items()]
        out.sort(key=lambda r: r["label"])
        self._json(200, out)

    @route("GET", r"/api/results/(?P<run_id>[a-f0-9]+)/meta")
    def api_results_meta(self, match, query):
        self._json(200, results.read_meta(self._run_dir(match.group("run_id"))))

    @route("GET", r"/api/results/(?P<run_id>[a-f0-9]+)/frame/(?P<n>\d+)")
    def api_results_frame(self, match, query):
        self._json(200, results.read_frame(self._run_dir(match.group("run_id")), int(match.group("n"))))


def make_server(port: int = 8765, host: str = "127.0.0.1", allow_no_auth: bool = False) -> ThreadingHTTPServer:
    token = os.environ.get("FEMNS_GUI_TOKEN") or None
    aviso = check_bind(host, token, allow_no_auth)  # levanta RuntimeError se o bind for inseguro
    if aviso:
        print(aviso, file=sys.stderr)
    Handler.job_manager = JobManager(REPO_ROOT)
    Handler.auth_token = token
    return ThreadingHTTPServer((host, port), Handler)


def run(port: int = 8765, open_browser: bool = True, host: str = "127.0.0.1", allow_no_auth: bool = False):
    httpd = make_server(port, host, allow_no_auth)
    display_host = "127.0.0.1" if is_local_host(host) else host
    url = f"http://{display_host}:{port}/"
    token = Handler.auth_token
    url_abrir = f"{url}?token={token}" if token else url
    print(f"femns webui em {url}"
          + (" (token exigido -- use a URL com ?token=... que o navegador abriu)" if token else "")
          + " (Ctrl+C para parar)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url_abrir)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
