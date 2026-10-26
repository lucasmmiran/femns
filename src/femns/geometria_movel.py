"""Geometria movel: quais contornos se movem, como, e com que velocidade o fluido os ve.

Bloco `geometria_movel` do config. Dois conceitos:

**Corpos** (`corpos:`) -- a lei de movimento, compartilhada por todos os
contornos presos ao mesmo corpo. O tipo do movimento e' um *atributo* do
corpo, por composicao (`Rotacao`, `Translacao`), e nao uma subclasse de
contorno: um contorno "rigido de rotacao" e' um `Rigido` preso a um corpo
`Rotacao`. Assim a corda e a superficie da esfera da valvula giram com o
**mesmo** angulo, sem repetir a lei temporal.

- `Rotacao`: gira em torno de `centro`; o estado e' um angulo (graus no config).
- `Translacao`: anda ao longo de `direcao`; o estado e' um deslocamento.

Os dois seguem a rampa em cosseno `inicio -> fim` em `duracao` s apos
`atraso` s (`valvula_rotacao.theta_em`). `malha` e' o estado em que a malha
foi gerada (padrao: `inicio`).

**Contornos** (`contornos:`, por grupo fisico) -- *como* os nos do contorno
andam. Aqui a distincao e' por subclasse, porque o comportamento muda:

- `Fixo` (padrao de todo grupo nao citado): parado.
- `Rigido`: anda junto com o corpo, a partir da posicao da malha.
- `Deslizante`: se estica entre uma ponta parada e uma ponta presa ao corpo;
  cada no mantem a fracao `eta` que ocupava. A trajetoria vem do tipo do
  corpo:
  - corpo `Rotacao` -> **arco** do circulo de raio `raio` em torno do
    centro. Nos do grupo *sobre* o circulo deslizam; *dentro* dele andam
    rigidos com o corpo (ex.: filete da quina da esfera); *fora* ficam
    parados (ex.: filete da carcaca). Cada arco e' um caminho de nos sobre o
    circulo com exatamente uma ponta ligada a parte rigida do corpo.
  - corpo `Translacao` -> **reta** na direcao do movimento. Sendo `s` a
    projecao na direcao, os nos com o menor `s` do grupo ficam parados
    (base), os com o maior `s` andam rigidos (ponta) e os do meio deslizam
    com `eta = (s - s_base)/(s_ponta - s_base)`.

Cada contorno movel diz tambem a velocidade que o **fluido** ve nele
(`velocidade_fluido`), independente de como os nos da malha andam:

- `malha`: velocidade dos nos, `(x^{n+1} - x^n)/dt` (padrao do `Rigido`);
- `corpo`: velocidade do ponto material do corpo -- `omega x (p - c)` ou
  `v*direcao` -- nos pontos da superficie do corpo, zero nos nos parados
  fora dela (padrao do `Deslizante`: o no desliza a `eta` vezes isso);
- `condicao`: mantem o valor de `boundary.conditions` (ex.: parede parada
  sobre a qual os nos deslizam).

So vertices; os nos extras do elemento seguem por `mesh.atualiza_nos_extras`.
numpy puro, sem torch (o servidor da GUI valida o bloco com `ler_config`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from femns.valvula_rotacao import theta_em

TOL_GEOMETRIA = 1e-6
VELOCIDADES = ("malha", "corpo", "condicao")


# =============================================================================
# 1. Corpos (lei de movimento)
# =============================================================================
@dataclass
class Corpo:
    """Estado `valor(t)` em rampa de cosseno; `malha` e' o estado da malha gerada."""

    nome: str
    inicio: float
    fim: float
    duracao: float
    atraso: float = 0.0
    malha: float | None = None

    def __post_init__(self):
        if self.malha is None:
            self.malha = self.inicio

    def valor(self, t: float) -> float:
        return theta_em(t, self.inicio, self.fim, self.duracao, self.atraso)


@dataclass
class Rotacao(Corpo):
    """Rotacao em torno de `centro`; `inicio`, `fim`, `malha` em radianos."""

    centro: tuple = (0.0, 0.0)
    tipo = "rotacao"

    def mover(self, X0, Y0, valor: float):
        """Posicao de pontos materiais, a partir da posicao na malha."""
        cx, cy = self.centro
        c, s = np.cos(valor - self.malha), np.sin(valor - self.malha)
        dx, dy = X0 - cx, Y0 - cy
        return cx + c * dx - s * dy, cy + s * dx + c * dy

    def velocidade(self, X, Y, taxa: float):
        """`omega x (p - c)`, com `omega = taxa`."""
        return -taxa * (np.asarray(Y) - self.centro[1]), taxa * (np.asarray(X) - self.centro[0])

    def desloc_max(self, X0, Y0, idx, dvalor: float) -> float:
        return float(np.hypot(X0[idx] - self.centro[0], Y0[idx] - self.centro[1]).max()) * abs(dvalor)


@dataclass
class Translacao(Corpo):
    """Translacao ao longo de `direcao` (normalizada); o estado e' o deslocamento."""

    direcao: tuple = (0.0, 1.0)
    tipo = "translacao"

    def __post_init__(self):
        super().__post_init__()
        d = np.asarray(self.direcao, float)
        self.direcao = tuple(d / np.hypot(*d))

    def mover(self, X0, Y0, valor: float):
        dx, dy = self.direcao
        return X0 + (valor - self.malha) * dx, Y0 + (valor - self.malha) * dy

    def velocidade(self, X, Y, taxa: float):
        n = np.shape(X)
        return np.full(n, taxa * self.direcao[0]), np.full(n, taxa * self.direcao[1])

    def desloc_max(self, X0, Y0, idx, dvalor: float) -> float:
        return abs(dvalor)


# =============================================================================
# 2. Contornos (como os nos andam)
# =============================================================================
@dataclass
class Contorno:
    grupo: str


@dataclass
class Fixo(Contorno):
    pass


@dataclass
class Movel(Contorno):
    corpo: str = ""
    velocidade_fluido: str = "malha"


@dataclass
class Rigido(Movel):
    velocidade_fluido: str = "malha"


@dataclass
class Deslizante(Movel):
    velocidade_fluido: str = "corpo"
    raio: float | None = None   # so para corpo Rotacao; None = inferido da malha


# =============================================================================
# 3. Leitura do config
# =============================================================================
@dataclass
class ConfigGeometria:
    corpos: dict             # nome -> Corpo
    contornos: dict          # grupo -> Contorno (so os moveis; o resto e' Fixo)
    interior: dict | None    # bloco para movimento_malha.opcoes_movimento


def _num(v, nome):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"geometria_movel: {nome} precisa ser um numero (recebido {v!r})")
    return float(v)


def _par(v, nome):
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise ValueError(f"geometria_movel: {nome} precisa ser [x, y] (recebido {v!r})")
    return (_num(v[0], nome), _num(v[1], nome))


def ler_config(cfg: dict, duracao_padrao: float | None = None) -> ConfigGeometria:
    """Valida o bloco `geometria_movel` e devolve corpos e contornos.

    Sem acesso a malha: quem chama confere se os grupos existem. Angulos em
    graus no config, radianos nos objetos.
    """
    if not isinstance(cfg, dict):
        raise ValueError("geometria_movel precisa ser um mapa")
    extras = set(cfg) - {"enabled", "corpos", "contornos", "interior"}
    if extras:
        raise ValueError(f"geometria_movel: chave(s) desconhecida(s) {sorted(extras)} "
                         f"(use corpos, contornos, interior)")
    corpos_cfg = cfg.get("corpos") or {}
    if not isinstance(corpos_cfg, dict) or not corpos_cfg:
        raise ValueError("geometria_movel.corpos: declare pelo menos um corpo (nome: {tipo: rotacao|translacao, ...})")

    corpos = {}
    for nome, c in corpos_cfg.items():
        if not isinstance(c, dict):
            raise ValueError(f"geometria_movel.corpos.{nome} precisa ser um mapa")
        tipo = c.get("tipo")
        comuns = {"tipo", "inicio", "fim", "duracao", "atraso", "malha"}
        if tipo == "rotacao":
            permitidas = comuns | {"centro"}
        elif tipo == "translacao":
            permitidas = comuns | {"direcao"}
        else:
            raise ValueError(f"geometria_movel.corpos.{nome}.tipo: use rotacao ou translacao (recebido {tipo!r})")
        if set(c) - permitidas:
            raise ValueError(f"geometria_movel.corpos.{nome}: chave(s) desconhecida(s) {sorted(set(c) - permitidas)}")
        for k in ("inicio", "fim"):
            if k not in c:
                raise ValueError(f"geometria_movel.corpos.{nome}.{k} e' obrigatorio")
        duracao = c.get("duracao", duracao_padrao)
        if duracao is None or _num(duracao, f"corpos.{nome}.duracao") <= 0:
            raise ValueError(f"geometria_movel.corpos.{nome}.duracao precisa ser > 0")
        args = dict(nome=nome, inicio=_num(c["inicio"], f"corpos.{nome}.inicio"),
                    fim=_num(c["fim"], f"corpos.{nome}.fim"), duracao=float(duracao),
                    atraso=_num(c.get("atraso", 0.0), f"corpos.{nome}.atraso"),
                    malha=None if c.get("malha") is None else _num(c["malha"], f"corpos.{nome}.malha"))
        if tipo == "rotacao":
            for k in ("inicio", "fim", "malha"):
                if args[k] is not None:
                    args[k] = float(np.radians(args[k]))
            corpos[nome] = Rotacao(**args, centro=_par(c.get("centro", [0.0, 0.0]), f"corpos.{nome}.centro"))
        else:
            d = _par(c.get("direcao", [0.0, 1.0]), f"corpos.{nome}.direcao")
            if np.hypot(*d) == 0:
                raise ValueError(f"geometria_movel.corpos.{nome}.direcao nula")
            corpos[nome] = Translacao(**args, direcao=d)

    contornos_cfg = cfg.get("contornos") or {}
    if not isinstance(contornos_cfg, dict):
        raise ValueError("geometria_movel.contornos precisa ser um mapa grupo: {tipo: ...}")
    contornos = {}
    for grupo, c in contornos_cfg.items():
        if c == "fixo" or (isinstance(c, dict) and c.get("tipo") == "fixo"):
            continue
        if not isinstance(c, dict):
            raise ValueError(f"geometria_movel.contornos.{grupo}: use fixo ou {{tipo: rigido|deslizante, corpo: ...}}")
        tipo = c.get("tipo")
        permitidas = {"tipo", "corpo", "velocidade_fluido"} | ({"raio"} if tipo == "deslizante" else set())
        if tipo not in ("rigido", "deslizante"):
            raise ValueError(f"geometria_movel.contornos.{grupo}.tipo: use fixo, rigido ou deslizante (recebido {tipo!r})")
        if set(c) - permitidas:
            raise ValueError(f"geometria_movel.contornos.{grupo}: chave(s) desconhecida(s) {sorted(set(c) - permitidas)}")
        corpo = c.get("corpo")
        if corpo not in corpos:
            raise ValueError(f"geometria_movel.contornos.{grupo}.corpo: {corpo!r} nao esta em corpos "
                             f"({', '.join(corpos)})")
        classe = Rigido if tipo == "rigido" else Deslizante
        vel = c.get("velocidade_fluido", classe.velocidade_fluido)
        if vel not in VELOCIDADES:
            raise ValueError(f"geometria_movel.contornos.{grupo}.velocidade_fluido: use {', '.join(VELOCIDADES)}")
        kw = dict(grupo=grupo, corpo=corpo, velocidade_fluido=vel)
        if tipo == "deslizante" and c.get("raio") is not None:
            if not isinstance(corpos[corpo], Rotacao):
                raise ValueError(f"geometria_movel.contornos.{grupo}.raio so vale para corpo de rotacao")
            kw["raio"] = _num(c["raio"], f"contornos.{grupo}.raio")
        contornos[grupo] = classe(**kw)
    if not contornos:
        raise ValueError("geometria_movel.contornos: nenhum contorno movel (rigido ou deslizante)")
    for nome in corpos:
        if not any(c.corpo == nome for c in contornos.values()):
            raise ValueError(f"geometria_movel: o corpo {nome!r} nao tem nenhum contorno preso a ele")
    return ConfigGeometria(corpos=corpos, contornos=contornos, interior=cfg.get("interior"))


# =============================================================================
# 4. Classificacao dos vertices (com a malha)
# =============================================================================
@dataclass
class Arco:
    """Nos sobre um arco, da ponta parada `nos[0]` a ponta movel `nos[-1]`."""

    corpo: str
    grupo: str
    nos: np.ndarray
    phi0: np.ndarray
    raio: float
    lam: np.ndarray = field(init=False)

    def __post_init__(self):
        d_esq = self.phi0[1:-1] - self.phi0[:-2]
        d_dir = self.phi0[2:] - self.phi0[1:-1]
        self.lam = d_dir / (d_esq + d_dir)


@dataclass
class Reta:
    """Nos deslizantes de um grupo em translacao: `s = s_base + eta*(s_ponta - s_base)`."""

    corpo: str
    grupo: str
    nos: np.ndarray
    eta: np.ndarray
    s0: np.ndarray        # projecao de cada no na malha
    s_base: float
    s_ponta0: float


def _nos_dos_grupos(IENbound, nomes, grupos) -> np.ndarray:
    sel = np.isin(np.asarray(nomes), list(grupos))
    return np.unique(np.asarray(IENbound)[sel][:, :2])


def _vizinhos_contorno(IENbound, n: int) -> list:
    viz = [[] for _ in range(n)]
    for a, b in np.asarray(IENbound)[:, :2]:
        viz[a].append(b)
        viz[b].append(a)
    return viz


def _caminhos(nos: np.ndarray, viz: list) -> list:
    """Componentes conexas (caminhos) do subgrafo de contorno induzido por `nos`, em ordem."""
    em = set(int(i) for i in nos)
    grau = {i: sum(j in em for j in viz[i]) for i in em}
    vistos, caminhos = set(), []
    for ponta in sorted(i for i in em if grau[i] < 2):
        if ponta in vistos:
            continue
        caminho, anterior = [ponta], None
        while True:
            prox = [j for j in viz[caminho[-1]] if j in em and j != anterior]
            if not prox:
                break
            anterior = caminho[-1]
            caminho.append(prox[0])
        vistos.update(caminho)
        caminhos.append(np.array(caminho, dtype=int))
    if vistos != em:
        raise ValueError("nos de arco formando um ciclo fechado no contorno")
    return caminhos


def _raio_inferido(X, Y, nos, centro) -> float:
    """Raio em que mais nos do grupo caem (os do arco, contra os poucos de filetes)."""
    rho = np.hypot(X[nos] - centro[0], Y[nos] - centro[1])
    valores, contagem = np.unique(np.round(rho / TOL_GEOMETRIA).astype(np.int64), return_counts=True)
    return float(np.median(rho[np.round(rho / TOL_GEOMETRIA).astype(np.int64) == valores[contagem.argmax()]]))


@dataclass
class GeometriaMovel:
    """Classificacao dos vertices de contorno e a cinematica de cada classe."""

    cfg: ConfigGeometria
    X0: np.ndarray
    Y0: np.ndarray
    contorno: np.ndarray              # mascara: todo vertice de contorno
    rigido: dict                      # corpo -> vertices rigidos
    arcos: list
    retas: list
    superficie: dict                  # corpo -> mascara dos pontos da superficie do corpo

    @property
    def corpos(self) -> dict:
        return self.cfg.corpos

    @property
    def idx_arco(self) -> np.ndarray:
        return np.concatenate([a.nos[1:-1] for a in self.arcos]) if self.arcos else np.empty(0, int)

    @property
    def idx_reta(self) -> np.ndarray:
        return np.concatenate([r.nos for r in self.retas]) if self.retas else np.empty(0, int)

    @property
    def idx_rigido(self) -> np.ndarray:
        partes = list(self.rigido.values())
        return np.concatenate(partes) if partes else np.empty(0, int)

    @property
    def idx_movel(self) -> np.ndarray:
        return np.concatenate([self.idx_rigido, self.idx_arco, self.idx_reta])

    def estado_malha(self) -> dict:
        return {n: c.malha for n, c in self.corpos.items()}

    # --- cinematica do contorno -------------------------------------------
    def contorno_em(self, X, Y, atual: dict, novo: dict):
        """Posicoes do contorno no estado `novo`, partindo da geometria atual `(X, Y)`.

        Rigidos: exato a partir da malha. Arcos: cada no mantem a fracao
        angular atual entre as pontas (preditor). Retas: fracao da malha.
        """
        Xc, Yc = np.array(X, float), np.array(Y, float)
        for nome, idx in self.rigido.items():
            Xc[idx], Yc[idx] = self.corpos[nome].mover(self.X0[idx], self.Y0[idx], novo[nome])
        for a in self.arcos:
            corpo = self.corpos[a.corpo]
            cx, cy = corpo.centro
            r = a.raio
            phi = _phi_atual(a, Xc, Yc, corpo.centro)
            phiA = a.phi0[0]
            phiB = a.phi0[-1] + (atual[a.corpo] - corpo.malha)
            phiB_novo = a.phi0[-1] + (novo[a.corpo] - corpo.malha)
            eta = (phi[1:-1] - phiA) / (phiB - phiA)
            phi_novo = phiA + eta * (phiB_novo - phiA)
            Xc[a.nos[1:-1]] = cx + r * np.cos(phi_novo)
            Yc[a.nos[1:-1]] = cy + r * np.sin(phi_novo)
        for rt in self.retas:
            corpo = self.corpos[rt.corpo]
            s_ponta = rt.s_ponta0 + (novo[rt.corpo] - corpo.malha)
            ds = rt.s_base + rt.eta * (s_ponta - rt.s_base) - rt.s0
            Xc[rt.nos] = self.X0[rt.nos] + ds * corpo.direcao[0]
            Yc[rt.nos] = self.Y0[rt.nos] + ds * corpo.direcao[1]
        return Xc, Yc

    def suavizar_arcos(self, X, Y, a2: float, varreduras: int, pesos: str = "referencia") -> None:
        """Corretor 1D dos arcos (no lugar), Jacobi em coordenada angular (ver movimento_malha)."""
        if a2 <= 0 or varreduras <= 0:
            return
        for a in self.arcos:
            cx, cy = self.corpos[a.corpo].centro
            r = a.raio
            lam = a.lam if pesos == "referencia" else np.full(len(a.nos) - 2, 0.5)
            phi = _phi_atual(a, X, Y, (cx, cy))
            for _ in range(varreduras):
                alvo = lam * phi[:-2] + (1 - lam) * phi[2:]
                phi[1:-1] += a2 * (alvo - phi[1:-1])
            X[a.nos[1:-1]] = cx + r * np.cos(phi[1:-1])
            Y[a.nos[1:-1]] = cy + r * np.sin(phi[1:-1])

    def desloc_max(self, atual: dict, novo: dict) -> float:
        """Maior deslocamento de um ponto de contorno entre os dois estados."""
        d = 0.0
        for nome, corpo in self.corpos.items():
            idx = np.concatenate([self.rigido.get(nome, np.empty(0, int))]
                                 + [a.nos for a in self.arcos if a.corpo == nome]
                                 + [r.nos for r in self.retas if r.corpo == nome])
            if len(idx):
                d = max(d, corpo.desloc_max(self.X0, self.Y0, idx, novo[nome] - atual[nome]))
        return d

    # --- velocidade do fluido ----------------------------------------------
    def velocidade_corpo(self, nome: str, X, Y, taxa: float):
        """Velocidade do ponto material do corpo nos vertices da superficie dele; zero no resto."""
        m = self.superficie[nome]
        ux, uy = self.corpos[nome].velocidade(X, Y, taxa)
        return np.where(m, ux, 0.0), np.where(m, uy, 0.0)


def _phi_atual(a: Arco, X, Y, centro) -> np.ndarray:
    ang = np.arctan2(Y[a.nos] - centro[1], X[a.nos] - centro[0])
    return a.phi0 + np.angle(np.exp(1j * (ang - a.phi0)))  # perto da referencia, sem salto de 2*pi


def preparar(cfg: ConfigGeometria, X, Y, IENbound, nomes) -> GeometriaMovel:
    """Classifica os vertices de contorno da malha segundo `cfg`.

    `X, Y` sao so os vertices, na geometria em que a malha foi gerada (estado
    `malha` de cada corpo).
    """
    X = np.asarray(X, float).copy()
    Y = np.asarray(Y, float).copy()
    n = len(X)
    IENbound = np.asarray(IENbound)[:, :2]
    nomes = list(nomes)
    existentes = set(nomes)
    faltando = set(cfg.contornos) - existentes
    if faltando:
        raise ValueError(f"geometria_movel.contornos: grupo(s) {sorted(faltando)} nao existem na malha "
                         f"(grupos: {sorted(existentes)})")

    contorno = np.zeros(n, bool)
    contorno[np.unique(IENbound)] = True
    viz = _vizinhos_contorno(IENbound, n)
    rigido_mask = {nome: np.zeros(n, bool) for nome in cfg.corpos}
    superficie = {nome: np.zeros(n, bool) for nome in cfg.corpos}
    arcos, retas = [], []

    # rigidos (tem precedencia sobre deslizantes nos vertices compartilhados)
    em_rigido = np.zeros(n, bool)
    for c in cfg.contornos.values():
        if isinstance(c, Rigido):
            nos = _nos_dos_grupos(IENbound, nomes, [c.grupo])
            rigido_mask[c.corpo][nos] = True
            em_rigido[nos] = True

    # deslizantes de rotacao: por (corpo, raio), como os arcos da valvula
    rot = [c for c in cfg.contornos.values() if isinstance(c, Deslizante) and isinstance(cfg.corpos[c.corpo], Rotacao)]
    conjuntos = {}
    for c in rot:
        corpo = cfg.corpos[c.corpo]
        nos = _nos_dos_grupos(IENbound, nomes, [c.grupo])
        nos = nos[~em_rigido[nos]]
        r = c.raio if c.raio is not None else _raio_inferido(X, Y, nos, corpo.centro)
        chave = next((k for k in conjuntos if k[0] == c.corpo and abs(k[1] - r) <= TOL_GEOMETRIA), (c.corpo, r))
        conjuntos.setdefault(chave, []).append((c.grupo, nos))
    for (nome, r), partes in conjuntos.items():
        corpo = cfg.corpos[nome]
        cx, cy = corpo.centro
        nos = np.unique(np.concatenate([p[1] for p in partes]))
        grupo_do_no = {}
        for g, ns in partes:
            for i in ns:
                grupo_do_no.setdefault(int(i), g)
        rho = np.hypot(X[nos] - cx, Y[nos] - cy)
        no_circulo = nos[np.abs(rho - r) <= TOL_GEOMETRIA]
        dentro = nos[rho < r - TOL_GEOMETRIA]
        if len(no_circulo) == 0:
            raise ValueError(f"geometria_movel: nenhum no de {[p[0] for p in partes]} sobre o circulo de raio "
                             f"{r:.6f} em torno de {corpo.centro}; confira o centro do corpo {nome!r} e o raio")
        rigido_mask[nome][dentro] = True
        rig = rigido_mask[nome]
        for cam in _caminhos(no_circulo, viz):
            anc = []
            for ponta, interno in ((cam[0], cam[1] if len(cam) > 1 else None),
                                   (cam[-1], cam[-2] if len(cam) > 1 else None)):
                fora = [j for j in viz[ponta] if j != interno and j not in set(cam.tolist())]
                if len(fora) != 1:
                    raise ValueError(f"arco com ponta {ponta} sem vizinho unico fora do arco")
                anc.append(fora[0])
            lado_rigido = [bool(rig[a]) for a in anc]
            if lado_rigido.count(True) != 1:
                raise ValueError(f"geometria_movel: arco de {grupo_do_no.get(int(cam[0]))} precisa de exatamente "
                                 f"uma ponta ligada a parte rigida do corpo {nome!r}; achei {lado_rigido.count(True)}")
            if lado_rigido[0]:
                cam, anc = cam[::-1], anc[::-1]
            # ponta movel: a propria ancora, se for rigida sobre o circulo
            # (quina viva); senao o ultimo no do caminho e' ponto material do corpo
            if em_rigido[anc[1]]:
                cam = np.append(cam, anc[1])
            else:
                rig[cam[-1]] = True
            ang = np.arctan2(Y[cam] - cy, X[cam] - cx)
            phi = ang[0] + np.unwrap(ang - ang[0])
            if not (np.all(np.diff(phi) > 0) or np.all(np.diff(phi) < 0)):
                raise ValueError(f"arco de {grupo_do_no.get(int(cam[1]))}: angulos nao monotonos")
            if abs(phi[-1] - phi[0]) > np.pi:
                raise ValueError("arco com mais de 180 graus: classificacao errada")
            if len(cam) >= 3:
                arcos.append(Arco(corpo=nome, grupo=grupo_do_no.get(int(cam[1]), "?"), nos=cam, phi0=phi, raio=r))
                superficie[nome][cam] = True

    # deslizantes de translacao: base parada, ponta rigida, meio deslizando
    for c in cfg.contornos.values():
        if not (isinstance(c, Deslizante) and isinstance(cfg.corpos[c.corpo], Translacao)):
            continue
        corpo = cfg.corpos[c.corpo]
        nos = _nos_dos_grupos(IENbound, nomes, [c.grupo])
        superficie[c.corpo][nos] = True
        nos = nos[~em_rigido[nos]]
        s = X[nos] * corpo.direcao[0] + Y[nos] * corpo.direcao[1]
        s_base, s_ponta = s.min(), s.max()
        if s_ponta - s_base <= TOL_GEOMETRIA:
            raise ValueError(f"geometria_movel: o contorno {c.grupo!r} nao tem extensao na direcao do corpo {c.corpo!r}")
        ponta = np.abs(s - s_ponta) <= TOL_GEOMETRIA
        base = np.abs(s - s_base) <= TOL_GEOMETRIA
        rigido_mask[c.corpo][nos[ponta]] = True
        meio = ~ponta & ~base
        if meio.any():
            retas.append(Reta(corpo=c.corpo, grupo=c.grupo, nos=nos[meio],
                              eta=(s[meio] - s_base) / (s_ponta - s_base), s0=s[meio],
                              s_base=float(s_base), s_ponta0=float(s_ponta)))

    rigido = {nome: np.flatnonzero(m) for nome, m in rigido_mask.items()}
    for nome in rigido:
        superficie[nome] |= rigido_mask[nome]
    return GeometriaMovel(cfg=cfg, X0=X, Y0=Y, contorno=contorno, rigido=rigido, arcos=arcos, retas=retas,
                          superficie=superficie)


# =============================================================================
# 5. Dirichlet do fluido nas paredes moveis
# =============================================================================
class ParedesMoveis:
    """Velocidade que o fluido ve em cada contorno movel (`velocidade_fluido`).

    `ccName` e' o nome de contorno de cada no (extras inclusive), ja resolvido
    por `boundary.priority`: so os nos cujo nome e' o grupo recebem o valor.
    `estender(ux, uy)` leva valores de vertice aos nos extras pela mesma regra
    com que eles se movem (`mesh.atualiza_nos_extras`).
    """

    def __init__(self, geo: GeometriaMovel, ccName, nnodes: int, estender):
        self.g = geo
        self.nnodes = nnodes
        self.estender = estender
        self.grupos = []   # (contorno, nos do grupo)
        for c in geo.cfg.contornos.values():
            if c.velocidade_fluido == "condicao":
                continue
            nos = np.array([i for i in range(nnodes) if ccName[i] == c.grupo], dtype=int)
            self.grupos.append((c, nos))
        self.idx = np.concatenate([n for _, n in self.grupos]) if self.grupos else np.empty(0, int)

    def velocidades(self, X_ant, Y_ant, X, Y, estado_ant: dict, estado: dict, dt: float):
        """`(ux, uy)` em `self.idx`, entre as geometrias/estados anterior e atual (so vertices)."""
        npts = len(self.g.X0)
        wx, wy = np.zeros(self.nnodes), np.zeros(self.nnodes)
        wx[:npts], wy[:npts] = (np.asarray(X) - X_ant) / dt, (np.asarray(Y) - Y_ant) / dt
        self.estender(wx, wy)
        por_corpo = {}
        partes_x, partes_y = [], []
        for c, nos in self.grupos:
            if c.velocidade_fluido == "malha":
                partes_x.append(wx[nos])
                partes_y.append(wy[nos])
                continue
            if c.corpo not in por_corpo:
                ux, uy = np.zeros(self.nnodes), np.zeros(self.nnodes)
                ux[:npts], uy[:npts] = self.g.velocidade_corpo(
                    c.corpo, X, Y, (estado[c.corpo] - estado_ant[c.corpo]) / dt)
                self.estender(ux, uy)
                por_corpo[c.corpo] = (ux, uy)
            ux, uy = por_corpo[c.corpo]
            partes_x.append(ux[nos])
            partes_y.append(uy[nos])
        if not partes_x:
            return np.empty(0), np.empty(0)
        return np.concatenate(partes_x), np.concatenate(partes_y)


# =============================================================================
# 6. Configs antigos -> geometria_movel
# =============================================================================
def de_valvula_rotacao(vr: dict) -> dict:
    """Bloco `valvula_rotacao` (valvula esfera) no formato `geometria_movel`."""
    H, L = float(vr.get("H", 0.5)), float(vr.get("L", 2.0))
    fechamento = 2 * np.degrees(np.arcsin(H / np.hypot(H, L / 2)))
    if max(vr.get("theta_inicio", vr["theta0"]), vr["theta_fim"]) >= fechamento:
        raise ValueError(f"valvula_rotacao: theta ate {max(vr.get('theta_inicio', vr['theta0']), vr['theta_fim'])} "
                         f"graus, mas a valvula fecha em 2*asen(H/r) = {fechamento:.2f} graus.")
    vel_malha = list(vr.get("velocidade_malha", []))
    vel_esfera = list(vr.get("velocidade_esfera", []))

    def vel(g):
        return "malha" if g in vel_malha else "corpo" if g in vel_esfera else "condicao"

    contornos = {g: {"tipo": "rigido", "corpo": "esfera", "velocidade_fluido": vel(g)}
                 for g in vr.get("grupos_corda", ["valve_up", "valve_down"])}
    for g in vr.get("grupos_arco", ["outter_wall", "inner_wall"]):
        contornos[g] = {"tipo": "deslizante", "corpo": "esfera", "velocidade_fluido": vel(g),
                        "raio": float(np.hypot(H, L / 2))}
    esfera = {"tipo": "rotacao", "centro": [0.0, 0.0], "malha": vr["theta0"],
              "inicio": vr.get("theta_inicio", vr["theta0"]), "fim": vr["theta_fim"],
              "atraso": vr.get("atraso", 0.0)}
    if "duracao" in vr:
        esfera["duracao"] = vr["duracao"]
    out = {"enabled": True, "corpos": {"esfera": esfera}, "contornos": contornos}
    if vr.get("movimento") is not None:
        out["interior"] = vr["movimento"]
    return out


def de_dente_crescendo(dc: dict) -> dict:
    """Bloco `dente_crescendo` (pistao saindo do piso) no formato `geometria_movel`."""
    if "altura_inicio" not in dc:
        raise ValueError("dente_crescendo: informe altura_inicio (a altura do dente na malha)")
    pistao = {"tipo": "translacao", "direcao": [0.0, 1.0], "malha": dc["altura_inicio"],
              "inicio": dc["altura_inicio"], "fim": dc["altura_fim"], "atraso": dc.get("atraso", 0.0)}
    if "duracao" in dc:
        pistao["duracao"] = dc["duracao"]
    return {"enabled": True, "corpos": {"pistao": pistao},
            "contornos": {dc.get("grupo", "dente"): {"tipo": "deslizante", "corpo": "pistao"}},
            "interior": {"extensao": {"a3": 1.0, "chi": float(dc.get("chi", 2.0))}}}


def bloco_do_config(cfg: dict) -> dict | None:
    """O bloco `geometria_movel` ligado do config, convertendo `valvula_rotacao`/`dente_crescendo`."""
    blocos = [(k, cfg.get(k)) for k in ("geometria_movel", "valvula_rotacao", "dente_crescendo")
              if (cfg.get(k) or {}).get("enabled", False)]
    if len(blocos) > 1:
        raise ValueError(f"{', '.join(k for k, _ in blocos)}: ligue um so bloco de geometria movel")
    if not blocos:
        return None
    k, b = blocos[0]
    return b if k == "geometria_movel" else de_valvula_rotacao(b) if k == "valvula_rotacao" else de_dente_crescendo(b)
