"""Definicao e aplicacao de condicoes de contorno a partir da malha e do config."""

import numpy as np
import torch


def assign_boundary_names(IENbound: np.ndarray, IENboundElem: list, npoints: int, priority: list[str],
                           IENpoint: np.ndarray = None, IENpointElem: list = None, nnodes: int = None):
    """Atribui a cada no de contorno o nome do contorno ao qual ele pertence.

    `priority` lista os nomes de contorno da prioridade mais baixa para a
    mais alta: quando um no pertence a mais de um contorno (ex.: quinas da
    malha), prevalece o ultimo nome da lista que o contem.

    `IENpoint`/`IENpointElem` (opcionais, de `mesh.read_mesh`) cobrem
    contornos definidos por um unico no (ex.: ponto de referencia de
    pressao numa cavidade tampada, sem saida fisica de fluido) -- tratados
    dentro da mesma ordem de prioridade que as arestas.

    `IENbound` pode ter qualquer numero de colunas por segmento (2 para
    elementos so com vertices, ex. MINI; 3 com no de aresta no meio, ex.
    Tri6 -- ver `mesh.estende_IENbound_tri6`). `nnodes` (opcional, default
    `npoints`) e o total de nos de velocidade -- precisa ser maior que
    `npoints` quando o contorno inclui nos que nao sao vertice (arestas
    do Tri6).
    """
    nnodes = nnodes if nnodes is not None else npoints
    ccName = [None for _ in range(nnodes)]
    IENpoint = IENpoint if IENpoint is not None else np.empty(0, dtype=int)
    IENpointElem = IENpointElem if IENpointElem is not None else []

    for nome in priority:
        for segmento in IENbound[np.array(IENboundElem) == nome]:
            for a in segmento:
                ccName[a] = nome
        for a in IENpoint[np.array(IENpointElem) == nome]:
            ccName[a] = nome

    return ccName


def perfil_parabolico(coord: np.ndarray, vmax: float,
                       s0: float = None, s1: float = None) -> np.ndarray:
    """Perfil parabolico sobre um contorno: zero em `s0`/`s1`, `vmax` no meio.

        v(s) = 4 * vmax * (s - s0) * (s1 - s) / (s1 - s0)^2

    `coord` sao as coordenadas dos nos que recebem o valor, ao longo do
    eixo em que o contorno se estende. `s0`/`s1` sao as **pontas do
    contorno** (default: min/max de `coord`); vem da malha e nao de uma
    altura fixa no codigo, entao o mesmo config vale para qualquer canal.
    Passe-as explicitamente quando `coord` nao cobrir o contorno inteiro
    -- ver `valores_da_condicao`, onde as quinas ficam de fora por
    prioridade de contorno.

    E' o perfil desenvolvido de Poiseuille: com `vmax = 1.5` a velocidade
    media na secao e' 1, mesma vazao da entrada uniforme `vx: 1.0`, o que
    permite comparar as duas condicoes sem mudar a vazao.
    """
    s0 = coord.min() if s0 is None else s0
    s1 = coord.max() if s1 is None else s1
    if s1 == s0:
        raise ValueError("perfil parabolico num contorno degenerado (todos os nos na mesma posicao)")
    return 4.0 * vmax * (coord - s0) * (s1 - coord) / (s1 - s0) ** 2


_FUNCOES_PERMITIDAS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt,
    "abs": np.abs, "sign": np.sign, "floor": np.floor, "ceil": np.ceil,
    "minimum": np.minimum, "maximum": np.maximum, "clip": np.clip,
    "pi": np.pi, "e": np.e,
}


def funcao_de_y(expressao: str, y: np.ndarray) -> np.ndarray:
    """Avalia uma expressao Python de `y` (ex.: "4*y*(1-y)") nas coordenadas dadas.

    `y` e' a coordenada **absoluta** do no na malha (nao normalizada pelo
    contorno) -- a expressao precisa ser escrita pensando no y real do
    canal, do mesmo jeito que aparece no preview da malha.

    E' um `eval` restrito (builtins bloqueados, so os nomes de
    `_FUNCOES_PERMITIDAS` mais `y` no escopo), nao um parser matematico
    de verdade nem uma fronteira de seguranca -- a expressao so e'
    digitada pelo proprio usuario local, nunca por terceiro. A restricao
    existe so pra transformar erro de digitacao em `ValueError` claro em
    vez de deixar passar um nome indefinido.
    """
    ambiente = dict(_FUNCOES_PERMITIDAS)
    ambiente["y"] = y
    try:
        resultado = eval(expressao, {"__builtins__": {}}, ambiente)  # noqa: S307
    except Exception as exc:
        raise ValueError(f"expressao de contorno invalida {expressao!r}: {exc}") from exc
    return np.broadcast_to(np.asarray(resultado, dtype=float), y.shape).copy()


def valores_da_condicao(valor, idx: np.ndarray, X: np.ndarray, Y: np.ndarray,
                         idx_span: np.ndarray = None):
    """Resolve o valor de uma condicao de contorno nos nos `idx`.

    Aceita um numero (condicao uniforme, o caso historico) ou um dict
    descrevendo a condicao de outra forma:
      - `{"perfil": "parabolico", "vmax": v}` -- perfil parabolico
        normalizado pela extensao geometrica do contorno (ver abaixo).
      - `{"funcao": "4*y*(1-y)"}` -- expressao Python arbitraria do `y`
        **absoluto** de cada no (ver `funcao_de_y`); ao contrario do
        perfil parabolico, nao ha normalizacao pela extensao do contorno
        -- a expressao precisa valer para o y real da malha.

    `idx_span` (default: `idx`) sao os nos que definem a **extensao**
    do contorno, e nao precisam ser os mesmos que recebem o valor. A
    distincao importa pro perfil parabolico: `idx` ja passou pela
    resolucao de prioridade (`assign_boundary_names`), entao as quinas do
    contorno normalmente pertencem a parede e **saem** de `idx`. Calcular
    a extensao so com `idx` encolhe o perfil de `h` em cada ponta -- a
    parabola zera dentro do canal em vez de na parede, e a vazao sai
    baixa por ~2h/H (medido: 6,4% de erro numa malha com h~0.03, com a
    vazao ainda conservada de secao a secao, o que faz o erro parecer
    discretizacao). Passe aqui os nos *geometricos* do contorno. `funcao`
    nao usa `idx_span` -- cada no recebe o valor no seu proprio y, sem
    depender da extensao do contorno.

    O eixo do perfil parabolico e' detectado pela geometria do proprio
    contorno: usa a coordenada em que os nos daquele contorno mais se
    espalham (y para uma entrada vertical, x para uma horizontal). Evita
    ter que declarar o eixo no config e errar silenciosamente.
    """
    if not isinstance(valor, dict):
        return float(valor)

    if X is None or Y is None:
        raise ValueError(
            "condicao de contorno com perfil/funcao exige as coordenadas da malha -- "
            "passe X e Y para build_boundary_conditions")

    if "funcao" in valor:
        return funcao_de_y(valor["funcao"], Y[idx])

    nome = valor.get("perfil")
    if nome != "parabolico":
        raise ValueError(
            f"condicao de contorno desconhecida: {valor!r} -- use um numero, "
            "{'perfil': 'parabolico', 'vmax': ...} ou {'funcao': '4*y*(1-y)'}")

    idx_span = idx if idx_span is None else idx_span
    xs, ys = X[idx_span], Y[idx_span]
    usa_y = (ys.max() - ys.min()) >= (xs.max() - xs.min())

    coord = Y[idx] if usa_y else X[idx]
    s0, s1 = (ys.min(), ys.max()) if usa_y else (xs.min(), xs.max())

    return perfil_parabolico(coord, float(valor["vmax"]), s0=s0, s1=s1)


def build_boundary_conditions(IENbound: np.ndarray, ccName: list, conditions: dict, npoints: int, device,
                               IENpoint: np.ndarray = None, nnodes: int = None,
                               X: np.ndarray = None, Y: np.ndarray = None,
                               IENboundElem: list = None):
    """Monta os vetores de condicao de contorno (valores e indices) a partir do config.

    `conditions` mapeia nome do contorno -> {"vx": v, "vy": v, "p": v}; um
    componente so e restringido (Dirichlet) nos contornos que o listam.
    Cada valor pode ser um numero (condicao uniforme) ou um dict de perfil
    (ex. `{"perfil": "parabolico", "vmax": 1.5}`, ver
    `valores_da_condicao`) -- perfis exigem `X`, `Y` (coordenadas dos nos).
    `IENpoint` (opcional) acrescenta nos marcados por contorno de ponto
    unico (ver `assign_boundary_names`) ao conjunto de nos considerado.
    `nnodes` (opcional, default `npoints`) e o total de nos de velocidade
    (ver `assign_boundary_names`) -- `vx_cc`/`vy_cc` sao alocados nesse
    tamanho, `p_cc` sempre em `npoints` (pressao so tem grau de liberdade
    nos vertices).

    Retorno: vx_cc, vy_cc, p_cc (tensores com o valor da condicao em cada no)
    e vx_cc_pts, vy_cc_pts, p_cc_pts (indices dos nos restringidos por componente).
    """
    nnodes = nnodes if nnodes is not None else npoints
    IENpoint = IENpoint if IENpoint is not None else np.empty(0, dtype=int)
    cc = np.unique(np.concatenate([IENbound.reshape(IENbound.size), IENpoint]))
    ccName_arr = np.array(ccName, dtype=object)

    vx_cc = torch.zeros(nnodes, dtype=torch.float64, device=device)
    vy_cc = torch.zeros(nnodes, dtype=torch.float64, device=device)
    p_cc = torch.zeros(npoints, dtype=torch.float64, device=device)

    vx_idx, vy_idx, p_idx = [], [], []

    elem_arr = np.array(IENboundElem) if IENboundElem is not None else None

    for nome, valores in conditions.items():
        idx = cc[ccName_arr[cc] == nome]
        if idx.size == 0:
            continue

        # Extensao geometrica do contorno (todos os nos dos segmentos com esse
        # nome, quinas incluidas), que pode ser maior que `idx` -- ver
        # `valores_da_condicao` sobre por que a diferenca importa nos perfis.
        span = np.unique(IENbound[elem_arr == nome]) if elem_arr is not None else idx

        if "vx" in valores:
            vx_cc[idx] = torch.as_tensor(
                valores_da_condicao(valores["vx"], idx, X, Y, span), dtype=vx_cc.dtype, device=device)
            vx_idx.append(idx)
        if "vy" in valores:
            vy_cc[idx] = torch.as_tensor(
                valores_da_condicao(valores["vy"], idx, X, Y, span), dtype=vy_cc.dtype, device=device)
            vy_idx.append(idx)
        if "p" in valores:
            # pressao so tem grau de liberdade nos vertices (P1): descarta
            # nos de aresta do Tri6 que porventura estejam em `idx`.
            idx_p = idx[idx < npoints]
            p_cc[idx_p] = torch.as_tensor(
                valores_da_condicao(valores["p"], idx_p, X, Y, span), dtype=p_cc.dtype, device=device)
            p_idx.append(idx_p)

    vx_cc_pts = np.concatenate(vx_idx) if vx_idx else np.array([], dtype=int)
    vy_cc_pts = np.concatenate(vy_idx) if vy_idx else np.array([], dtype=int)
    p_cc_pts = np.concatenate(p_idx) if p_idx else np.array([], dtype=int)

    vx_cc_pts = torch.from_numpy(vx_cc_pts).to(device)
    vy_cc_pts = torch.from_numpy(vy_cc_pts).to(device)
    p_cc_pts = torch.from_numpy(p_cc_pts).to(device)

    return vx_cc, vy_cc, p_cc, vx_cc_pts, vy_cc_pts, p_cc_pts


def apply_boundary_conditions(A: torch.Tensor, vx_cc_pts, vy_cc_pts, p_cc_pts, npoints: int, ne: int):
    """Zera as linhas de A referentes as condicoes de contorno e coloca 1 na diagonal.

    `npoints + ne` deve ser o tamanho do bloco de velocidade (linhas/colunas
    de vx e de vy em `A`, ver `solver.build_system_matrix`). Para o MINI
    isso e `npoints + numero_de_elementos` (1 bolha por elemento); para o
    Tri6, `ne` deve ser `nnodes - npoints` (nos de aresta, nao ha relacao
    direta com o numero de elementos).
    """
    rows_vx = vx_cc_pts
    rows_vy = vy_cc_pts + npoints + ne
    rows_p = p_cc_pts + 2 * (npoints + ne)
    rows_bc = torch.cat([rows_vx, rows_vy, rows_p])

    A = A.coalesce()
    mask = ~torch.isin(A.indices()[0], rows_bc)
    new_indices = A.indices()[:, mask]
    new_values = A.values()[mask]

    diag_indices = torch.stack([rows_bc, rows_bc])
    diag_values = torch.ones_like(rows_bc, dtype=A.dtype, device=A.device)

    final_indices = torch.cat([new_indices, diag_indices], dim=1)
    final_values = torch.cat([new_values, diag_values])

    return torch.sparse_coo_tensor(final_indices, final_values, size=A.shape, device=A.device, dtype=A.dtype)
