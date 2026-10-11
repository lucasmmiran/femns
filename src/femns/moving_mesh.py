"""Deslocamento senoidal da malha (experimento de malha movel/ALE).

`preparar_oscilacao`/`aplicar_oscilacao` fazem **todos os nos interiores**
oscilarem, cada um ao longo de uma direcao propria e com fase propria.
Ligado por `mesh_motion` no config. (A v1.0 tinha tambem `moving_point`, que
movia um no so, e a velocidade da malha para a correcao ALE da adveccao
Euleriana -- os dois sairam na v2.0.)
"""

from dataclasses import dataclass

import numpy as np
import torch


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

    Usa a vizinhanca nó-a-nó (`mesh.montar_NToN`). Fica um laco Python sobre os nos porque
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

    Como a fase e' propria de
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
    mas nada se move entre passos. `omega = pi*200` (o default de
    `mesh_motion`) com `dt = 0.01` cai exatamente nesse caso.
    """
    return 2.0 * np.pi / (omega * dt)
