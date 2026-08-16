"""Deslocamento senoidal da malha (experimento de malha movel/ALE).

Dois experimentos, independentes entre si:

- `mover_ponto`: desloca **um** no em orbita circular, reinterpolando a
  velocidade dele a partir dos vizinhos (IDW). E' o experimento antigo,
  ligado por `moving_point` no config.
- `preparar_oscilacao`/`aplicar_oscilacao`: faz **todos os nos interiores**
  oscilarem, cada um ao longo de uma direcao propria e com fase propria.
  Ligado por `mesh_motion` no config.
"""

from dataclasses import dataclass

import numpy as np
import torch

from femns.mesh import atualiza_nos_extras


def nova_velocidade(vx: torch.Tensor, vy: torch.Tensor, dist: torch.Tensor, pts_viz):
    """Interpola a velocidade de um no deslocado a partir de seus vizinhos (IDW)."""
    device = dist.device
    beta = 2

    vx_viz = vx[pts_viz].to(device)
    vy_viz = vy[pts_viz].to(device)

    pesos = 1.0 / dist**beta
    pesos_norm = pesos / pesos.sum()

    return torch.dot(pesos_norm, vx_viz), torch.dot(pesos_norm, vy_viz)


def mover_ponto(ponto: int, X, Y, vx: torch.Tensor, vy: torch.Tensor, x0: float, y0: float,
                NToN, t: float, amplitude_factor: float = 0.1, omega: float = np.pi * 200):
    """Move um no da malha em uma trajetoria senoidal: pos_{m+1} = pos_0 + A*[cos, sin](omega*t).

    A amplitude A e proporcional a distancia ao vizinho mais proximo, para
    evitar que o no cruze elementos vizinhos.
    """
    device = X.device
    pts_viz = NToN[ponto]

    coo_ponto = torch.tensor([[x0, y0]]).to(device)
    coo_viz = torch.stack([X[pts_viz], Y[pts_viz]], dim=1).to(device)
    dist = torch.norm(coo_viz - coo_ponto, dim=1).to(device)

    amplitude = amplitude_factor * min(dist)

    X[ponto] = x0 + amplitude * np.cos(omega * t)
    Y[ponto] = y0 + amplitude * np.sin(omega * t)

    vx[ponto], vy[ponto] = nova_velocidade(vx, vy, dist, pts_viz)

    return X, Y, vx, vy


@dataclass
class Oscilacao:
    """Parametros pre-computados da oscilacao (constantes ao longo da simulacao).

    `nos` sao os indices dos nos que oscilam; `x0`/`y0` as posicoes de
    repouso desses nos; `amplitude` o deslocamento maximo de cada um;
    `dir_x`/`dir_y` o versor da direcao de oscilacao; `fase` a defasagem.
    """

    nos: np.ndarray
    x0: np.ndarray
    y0: np.ndarray
    amplitude: np.ndarray
    dir_x: np.ndarray
    dir_y: np.ndarray
    fase: np.ndarray
    omega: float


def distancia_min_vizinhos(X: np.ndarray, Y: np.ndarray, NToN, nos: np.ndarray) -> np.ndarray:
    """Distancia de cada no em `nos` ao seu vizinho mais proximo (`h` local).

    Usa a vizinhanca nó-a-nó (`mesh.montar_NToN`), a mesma nocao de
    "vizinho" de `mover_ponto`. Fica um laco Python sobre os nos porque
    `NToN` tem numero variavel de vizinhos por no (array de objetos), mas
    roda uma unica vez por malha, no precomputo -- nao por passo.
    """
    h = np.empty(len(nos))
    for i, n in enumerate(nos):
        viz = NToN[n]
        h[i] = np.min(np.hypot(X[viz] - X[n], Y[viz] - Y[n]))
    return h


def preparar_oscilacao(X: np.ndarray, Y: np.ndarray, NToN, nos: np.ndarray,
                        fator: float = 0.3, omega: float = np.pi * 200,
                        seed: int = 0) -> Oscilacao:
    """Pre-computa a oscilacao de `nos`: amplitude local, direcao e fase.

    A amplitude de cada no e `fator * h_local`, com `h_local` a distancia
    ao vizinho mais proximo (`distancia_min_vizinhos`) -- e' local, e nao
    um `h` global, para que a amplitude acompanhe o refino: numa regiao
    refinada os nos andam menos, em vez de a malha grossa ditar um
    deslocamento minusculo em todo lugar. Com `fator < 0.5` o no nunca
    alcanca o vizinho mais proximo, entao a malha nao inverte elemento
    por causa do deslocamento de um no isolado (nao e' garantia formal de
    nao-inversao: vizinhos se movem ao mesmo tempo, em direcoes
    independentes -- ver `aplicar_oscilacao`).

    Direcao e fase saem de um gerador com `seed` fixa, entao a mesma
    malha e a mesma semente dao sempre a mesma oscilacao (a simulacao
    continua reproduzivel).
    """
    rng = np.random.default_rng(seed)
    nos = np.asarray(nos, dtype=int)

    theta = rng.uniform(0.0, 2.0 * np.pi, size=len(nos))

    return Oscilacao(
        nos=nos,
        x0=X[nos].copy(),
        y0=Y[nos].copy(),
        amplitude=fator * distancia_min_vizinhos(X, Y, NToN, nos),
        dir_x=np.cos(theta),
        dir_y=np.sin(theta),
        fase=rng.uniform(0.0, 2.0 * np.pi, size=len(nos)),
        omega=omega,
    )


def aplicar_oscilacao(osc: Oscilacao, X: torch.Tensor, Y: torch.Tensor, t: float):
    """Escreve em `X`, `Y` as posicoes dos nos oscilantes no instante `t`.

    Cada no oscila **ao longo de uma reta**, na sua propria direcao:

        pos(t) = pos_0 + A * sin(omega*t + fase) * [dir_x, dir_y]

    (diferente de `mover_ponto`, que percorre uma *orbita circular* de raio
    `A` -- ali a posicao nunca volta a `pos_0`.) Como a fase e' propria de
    cada no, os vizinhos nao se movem juntos: a malha "treme" em vez de
    transladar em bloco, que e' o caso mais severo para a busca por
    elemento da adveccao semi-Lagrangeana.

    Em `t=0` os nos ja saem deslocados de `A*sin(fase)` -- a oscilacao e'
    centrada em `pos_0`, nao ancorada nela.

    Modifica `X`, `Y` no lugar (sao os tensores da simulacao) e os devolve.
    Os nos fora de `osc.nos` (contorno, e os nos extras do elemento) ficam
    intactos -- ver `mesh.atualiza_nos_extras` para reposicionar os nos de
    centroide/aresta depois desta chamada.
    """
    d = osc.amplitude * np.sin(osc.omega * t + osc.fase)

    nos = torch.as_tensor(osc.nos, dtype=torch.long, device=X.device)
    X[nos] = torch.as_tensor(osc.x0 + d * osc.dir_x, dtype=X.dtype, device=X.device)
    Y[nos] = torch.as_tensor(osc.y0 + d * osc.dir_y, dtype=Y.dtype, device=Y.device)

    return X, Y


def passos_por_periodo(omega: float, dt: float) -> float:
    """Quantos passos de tempo cabem num periodo da oscilacao: `2*pi / (omega*dt)`.

    Serve para detectar amostragem degenerada. O caso patologico e' inteiro:
    com `omega*dt = 2*pi` (exatamente 1 passo por periodo) a malha e'
    amostrada sempre na MESMA fase e fica **parada** numa configuracao
    deslocada -- a simulacao roda sem erro nenhum e parece uma malha movel,
    mas nada se move entre passos. `omega = pi*200` (o default herdado de
    `moving_point`) com `dt = 0.01` cai exatamente nesse caso.
    """
    return 2.0 * np.pi / (omega * dt)


def velocidade_malha(osc: Oscilacao, IEN, t: float, nnodes: int,
                      device=None, dtype=torch.float64):
    """Velocidade da malha `w` em cada no, no instante `t` (formulacao ALE).

    Derivada analitica de `aplicar_oscilacao` (nao diferenca finita entre
    passos, que herdaria o erro de amostragem):

        w(t) = A * omega * cos(omega*t + fase) * [dir_x, dir_y]

    Nos de contorno ficam com `w = 0` (nao se movem). Os nos extras do
    elemento (centroide/aresta) sao preenchidos por `atualiza_nos_extras`
    aplicado ao **campo de velocidade**, e nao as coordenadas: a posicao do
    centroide e' uma combinacao linear das posicoes dos vertices, entao a
    velocidade dele e' a mesma combinacao linear das velocidades -- a media
    que aquela funcao ja faz.

    Retorna `(wx, wy)`, tamanho `nnodes`.
    """
    wx = torch.zeros(nnodes, dtype=dtype, device=device)
    wy = torch.zeros(nnodes, dtype=dtype, device=device)

    d = osc.amplitude * osc.omega * np.cos(osc.omega * t + osc.fase)

    nos = torch.as_tensor(osc.nos, dtype=torch.long, device=device)
    wx[nos] = torch.as_tensor(d * osc.dir_x, dtype=dtype, device=device)
    wy[nos] = torch.as_tensor(d * osc.dir_y, dtype=dtype, device=device)

    return atualiza_nos_extras(IEN, wx, wy)
