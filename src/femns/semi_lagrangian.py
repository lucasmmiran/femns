"""Backtrace, localizacao de ponto e interpolacao para adveccao semi-Lagrangeana.

Logica pura de malha + interpolacao (numpy, sem torch) -- ver
docs/semi_lagrangian_strategy.pdf para a derivacao e as decisoes de
projeto por tras de cada funcao aqui.
"""

import numpy as np


def barycentric(xd: float, yd: float, X: np.ndarray, Y: np.ndarray, IEN: np.ndarray, elem: int):
    """Coordenadas baricentricas de (xd, yd) em relacao aos 3 vertices do `elem`.

    Retorna (li, lj, lk), razoes de area com sinal -- todas >= 0 (dentro
    de uma tolerancia numerica) se e somente se o ponto esta dentro do
    triangulo. li + lj + lk == 1 sempre.
    """
    i, j, k = IEN[elem, 0], IEN[elem, 1], IEN[elem, 2]
    xi, yi = X[i], Y[i]
    xj, yj = X[j], Y[j]
    xk, yk = X[k], Y[k]

    def cross(xa, ya, xb, yb, xc, yc):
        return (xb - xa) * (yc - ya) - (xc - xa) * (yb - ya)

    denom = cross(xi, yi, xj, yj, xk, yk)
    li = cross(xd, yd, xj, yj, xk, yk) / denom
    lj = cross(xi, yi, xd, yd, xk, yk) / denom
    lk = 1.0 - li - lj

    return li, lj, lk


def locate_point(xd: float, yd: float, elem_start: int, X: np.ndarray, Y: np.ndarray,
                  IEN: np.ndarray, EToE: np.ndarray, tol: float = 1e-9, max_iter: int = 50):
    """Localiza o elemento que contem (xd, yd), partindo de `elem_start`.

    Busca por caminhada: anda pelos vizinhos de `EToE` na direcao da
    coordenada baricentrica mais negativa (vertice local `m` mais
    negativo -> face `(m+1)%3`, convencao de `mesh.montar_EToE`) ate
    localizar o ponto ou sair do dominio.

    Retorna (elem, (li, lj, lk)) se localizado, ou (None, None) se o
    ponto estiver fora do dominio (face de contorno, EToE == -1) ou o
    limite de iteracoes for atingido -- os dois casos sao tratados da
    mesma forma por quem chama (fallback fora do dominio).
    """
    elem = elem_start
    for _ in range(max_iter):
        lambdas = barycentric(xd, yd, X, Y, IEN, elem)
        if min(lambdas) >= -tol:
            return elem, lambdas

        m = int(np.argmin(lambdas))
        face = (m + 1) % 3
        proximo = EToE[elem, face]
        if proximo == -1:
            return None, None
        elem = proximo

    return None, None


def interpolate_mini(li: float, lj: float, lk: float, vi: float, vj: float, vk: float, vb: float):
    """Avalia um campo do espaco MINI (P1 + bolha) num ponto de coordenadas baricentricas dadas.

    `vi, vj, vk` sao os valores nodais nos 3 vertices do elemento e `vb`
    o valor no no de centroide (coeficiente modal da bolha, nao uma
    velocidade fisica no centroide -- ver docs/semi_lagrangian_strategy.pdf,
    Passo 3). Os lambdas sao recortados para [0,1] e renormalizados antes
    de usar como peso, para absorver pequenos negativos residuais da
    tolerancia numerica de `locate_point`.
    """
    li, lj, lk = (max(0.0, min(1.0, x)) for x in (li, lj, lk))
    soma = li + lj + lk
    li, lj, lk = li / soma, lj / soma, lk / soma

    return li * vi + lj * vj + lk * vk + 27.0 * li * lj * lk * vb


def backtrace_euler(x: np.ndarray, y: np.ndarray, vx_node: np.ndarray, vy_node: np.ndarray, dt: float):
    """Pe da caracteristica por Euler explicito: x_d = x - dt*v(x).

    Vetorizado em numpy (sem busca, so aritmetica por no) -- `x, y,
    vx_node, vy_node` tem o mesmo shape, um valor por no.
    """
    return x - dt * vx_node, y - dt * vy_node


def backtrace_and_interpolate(X: np.ndarray, Y: np.ndarray, IEN: np.ndarray, EToE: np.ndarray,
                               node_to_elem: np.ndarray, ccName: list, conditions: dict,
                               vx: np.ndarray, vy: np.ndarray, dt: float, npoints: int, ne: int,
                               tol: float = 1e-9, max_iter: int = 50):
    """Avanca a adveccao por um passo semi-Lagrangeano: backtrace + localizacao + interpolacao.

    `X, Y, IEN` sao os arrays completos (pos-`mesh.elem_mini`, com o no
    de centroide/bolha na 4a coluna de `IEN`); `EToE` vem de
    `mesh.montar_EToE` sobre a malha original (so vertices); `vx, vy`
    sao os campos do passo atual (tamanho `npoints + ne`), em numpy.

    Para cada no, calcula o pe da caracteristica (Euler explicito) e
    localiza o elemento que o contem partindo de um "chute" (
    `node_to_elem[n]` para vertices, ou o proprio elemento `n - npoints`
    para nos de centroide -- ver `mesh.montar_node_to_elem`). Se o pe da
    caracteristica sair do dominio, usa o valor de Dirichlet do
    contorno (`ccName`/`conditions`, mesma estrutura de
    `boundary.build_boundary_conditions`) quando definido para aquele
    componente; senao mantem o valor atual do proprio no (nos de
    centroide nunca tem nome de contorno, entao sempre caem nesse
    ultimo caso quando saem do dominio).

    Retorna (vx_star, vy_star), tamanho `npoints + ne`.
    """
    n_total = npoints + ne
    vx_star = np.empty(n_total)
    vy_star = np.empty(n_total)

    for n in range(n_total):
        elem_start = node_to_elem[n] if n < npoints else n - npoints
        xd, yd = backtrace_euler(X[n], Y[n], vx[n], vy[n], dt)
        elem, lambdas = locate_point(xd, yd, elem_start, X, Y, IEN, EToE, tol, max_iter)

        if elem is not None:
            i, j, k, b = IEN[elem]
            li, lj, lk = lambdas
            vx_star[n] = interpolate_mini(li, lj, lk, vx[i], vx[j], vx[k], vx[b])
            vy_star[n] = interpolate_mini(li, lj, lk, vy[i], vy[j], vy[k], vy[b])
        else:
            nome = ccName[n] if n < npoints else None
            cond = conditions.get(nome, {}) if nome is not None else {}
            vx_star[n] = cond.get("vx", vx[n])
            vy_star[n] = cond.get("vy", vy[n])

    return vx_star, vy_star
