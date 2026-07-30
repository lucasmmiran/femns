"""Backtrace, localizacao de ponto e interpolacao para adveccao semi-Lagrangeana.

Logica pura de malha + interpolacao (numpy, sem torch) -- ver
docs/semi_lagrangian_strategy.pdf para a derivacao e as decisoes de
projeto por tras de cada funcao aqui. A busca por elemento
(`localizar_pontos_partida`) e vetorizada entre nos (mascara booleana, sem
laco Python por no) -- so o numero de saltos da caminhada em si e
sequencial, por natureza do algoritmo.
"""

import numpy as np


def backtrace(x: np.ndarray, y: np.ndarray, vx_node: np.ndarray, vy_node: np.ndarray, dt: float):
    """Pe da caracteristica por Euler explicito: x_d = x - dt*v(x).

    Vetorizado em numpy (sem busca, so aritmetica por no) -- `x, y,
    vx_node, vy_node` tem o mesmo shape, um valor por no.
    """
    return x - dt * vx_node, y - dt * vy_node


def baricentro(xd, yd, X: np.ndarray, Y: np.ndarray, IEN: np.ndarray, elem):
    """Coordenadas baricentricas de (xd, yd) em relacao aos 3 vertices do `elem`.

    Retorna li, lj, lk se o ponto esta dentro do triangulo. li + lj + lk = 1 sempre.

    `xd, yd, elem` podem ser escalares ou arrays numpy de mesma dimensão (shape).
    """
    
    i, j, k = IEN[elem, 0], IEN[elem, 1], IEN[elem, 2]
    xi, yi = X[i], Y[i]
    xj, yj = X[j], Y[j]
    xk, yk = X[k], Y[k]

    def bar_area(xa, ya, xb, yb, xc, yc):
        return (1.0/2.0)*((xb*yc+xa*yb+ya*xc)-(xa*yc+yb*xc+ya*xb))

    A = bar_area(xi, yi, xj, yj, xk, yk)
    li = bar_area(xd, yd, xj, yj, xk, yk) / A
    lj = bar_area(xi, yi, xd, yd, xk, yk) / A
    lk = 1.0 - li - lj

    return li, lj, lk


def localizar_pontos_partida(xd: np.ndarray, yd: np.ndarray, elem_start: np.ndarray, X: np.ndarray,
                         Y: np.ndarray, IEN: np.ndarray, EToE: np.ndarray, tol: float = 1e-9,
                         max_iter: int = 100):
    """Localiza, para cada ponto, o elemento que o contem -- versao em lote.

    Mesmo algoritmo de busca por caminhada de `locate_point` (anda
    pelos vizinhos de `EToE` na direcao da coordenada baricentrica mais
    negativa -- vertice local `m` mais negativo -> face `(m+1)%3`,
    convencao de `mesh.montar_EToE`), mas processando todos os pontos
    em paralelo *entre si*: cada iteracao do laco externo da "um passo
    de caminhada" simultaneo em todos os pontos ainda nao resolvidos
    (localizado ou fora do dominio), via mascara booleana -- em vez de
    um laco Python por ponto. O numero de iteracoes do laco externo e'
    o maior numero de saltos entre todos os pontos ainda ativos, nao
    `len(xd)` vezes algo.

    `xd, yd, elem_start` tem o mesmo shape `(n,)`. Retorna `(elem,
    lambdas)`: `elem` e' `(n,)` com `-1` nas posicoes nao localizadas
    (fora do dominio ou que nao convergiram em `max_iter`); `lambdas`
    e' `(n,3)`, com lixo nas linhas nao localizadas (quem usa o
    retorno deve filtrar por `elem != -1` antes).
    """
    n = len(xd)
    elem = np.asarray(elem_start, dtype=int).copy() # Vetor com os elementos atuais de cada ponto
    done = np.zeros(n, dtype=bool) # Inicializa o vetor de pontos com seu ponto de partida já encontrados
    lambdas_out = np.zeros((n, 3)) # Inicializa o vetor de lambdas dos elementos finais

    for _ in range(max_iter):
        ativos = np.where(~done)[0] # Identifica quais pontos de partida ainda estão sendo buscados 
        if ativos.size == 0: # Interrompe antes de max_iter se todos os pontos tiverem sido encontrados
            break

        lambdas = np.stack(baricentro(xd[ativos], yd[ativos], X, Y, IEN, elem[ativos]), axis=1) # Calcula os lambdas de todos os pontos que estão sendo buscados
        dentro = lambdas.min(axis=1) >= -tol # Para os lambdas calculados, verifica quais indicam que o ponto está dentro do elemento

        idx_dentro = ativos[dentro] # Recebe o index da mascara "dentro", indicando quais elementos de partida foram encontrados
        lambdas_out[idx_dentro] = lambdas[dentro] # Marca os lambdas finais dos elementos de partida encontrados
        done[idx_dentro] = True # Inclui nos finalizados

        idx_fora = ativos[~dentro] # Pega o index de quem não encontrou um elemento
        m = lambdas[~dentro].argmin(axis=1) # Identifica qual o index do menor lambda de quem ficou fora
        face = (m + 1) % 3 # Indica qual a face oposta ao vértice com menor lambda de quem ficou fora
        proximo = EToE[elem[idx_fora], face] # Indica qual o próximo elemento a ser analisado para buscar o ponto de partida

        saiu_dominio = idx_fora[proximo == -1] # Indica o index dos pontos em que a busca caiu fora do domínio 
        done[saiu_dominio] = True # Se caiu fora do domínio, ele para de buscar
        elem[saiu_dominio] = -1 # Indica que o elemento de partida está fora do domínio 

        tem_vizinho = proximo != -1
        andando = idx_fora[tem_vizinho] # Atualiza o vetor de index dos pontos que vão continuar a busca no próximo passo
        elem[andando] = proximo[tem_vizinho] # Atualiza o vetor de elemento com os próximos elementos a serem analisados

    elem[~done] = -1  # nao convergiu em max_iter: mesmo tratamento de "fora do dominio"

    return elem, lambdas_out


def locate_point(xd: float, yd: float, elem_start: int, X: np.ndarray, Y: np.ndarray,
                  IEN: np.ndarray, EToE: np.ndarray, tol: float = 1e-9, max_iter: int = 50):
    """Localiza o elemento que contem (xd, yd), partindo de `elem_start`.

    Wrapper escalar sobre `localizar_pontos_partida` (arrays de tamanho 1)
    -- mantido para uso pontual/testes; a caminhada em si vive so' em
    `localizar_pontos_partida`.

    Retorna (elem, (li, lj, lk)) se localizado, ou (None, None) se o
    ponto estiver fora do dominio (face de contorno, EToE == -1) ou o
    limite de iteracoes for atingido -- os dois casos sao tratados da
    mesma forma por quem chama (fallback fora do dominio).
    """
    elem, lambdas = localizar_pontos_partida(
        np.array([xd]), np.array([yd]), np.array([elem_start]), X, Y, IEN, EToE, tol, max_iter)

    if elem[0] == -1:
        return None, None
    return int(elem[0]), tuple(lambdas[0])


def interpolar_mini(li: float, lj: float, lk: float, vi: float, vj: float, vk: float, vb: float):
    """Avalia um campo do espaco MINI (base nodal) num ponto de coordenadas baricentricas dadas.

    `vi, vj, vk` sao os valores nodais nos 3 vertices do elemento e `vb`
    o valor no no de centroide -- os 4 DOFs sao nodais (propriedade de
    Lagrange), `vb` E a velocidade fisica no centroide, igual a
    qualquer outro no. As funcoes de forma correspondentes sao
    `Ni = li - 9*li*lj*lk` (e ciclicamente `Nj, Nk`) e `Nb = 27*li*lj*lk`
    -- verificado batendo a matriz de rigidez resultante, entrada a
    entrada, contra `tests/test_assembly.py::test_assemble_mini_triangulo_referencia`
    (a base "P1 + bolha crua", sem a correcao `9*li*lj*lk`, NAO bate
    com essa matriz). Os lambdas sao recortados para [0,1] e
    renormalizados antes de usar como peso, para absorver pequenos
    negativos residuais da tolerancia numerica de `locate_point`.
    Funciona tanto com escalares quanto com arrays numpy (`np.clip`
    vetoriza, ao contrario dos `min`/`max` builtins do Python).
    """
    li, lj, lk = np.clip(li, 0.0, 1.0), np.clip(lj, 0.0, 1.0), np.clip(lk, 0.0, 1.0)
    soma = li + lj + lk
    li, lj, lk = li / soma, lj / soma, lk / soma
    corr = 9.0 * li * lj * lk
    Ni, Nj, Nk, Nb = (li - corr), (lj - corr), (lk - corr), (27.0 * li * lj * lk)

    return Ni * vi + Nj * vj + Nk * vk + Nb * vb


def _fallback_dirichlet(ccName: list, conditions: dict, npoints: int, n_total: int):
    """Monta, uma vez por chamada, os arrays de valor/disponibilidade de Dirichlet por no.

    So os `npoints` nos de vertice tem nome de contorno (`ccName`); nos
    de centroide ficam com `has_*_bc=False` sempre. Usado como fallback
    vetorizado em `calculo_sl` para nos cujo pe da
    caracteristica saiu do dominio.
    """
    vx_bc = np.zeros(n_total)
    vy_bc = np.zeros(n_total)
    has_vx_bc = np.zeros(n_total, dtype=bool)
    has_vy_bc = np.zeros(n_total, dtype=bool)

    for n in range(npoints):
        nome = ccName[n]
        if nome is None:
            continue
        valores = conditions.get(nome, {})
        if "vx" in valores:
            vx_bc[n] = valores["vx"]
            has_vx_bc[n] = True
        if "vy" in valores:
            vy_bc[n] = valores["vy"]
            has_vy_bc[n] = True

    return vx_bc, has_vx_bc, vy_bc, has_vy_bc


def calculo_sl(X: np.ndarray, Y: np.ndarray, IEN: np.ndarray, EToE: np.ndarray,
                               node_to_elem: np.ndarray, ccName: list, conditions: dict,
                               vx: np.ndarray, vy: np.ndarray, dt: float, npoints: int, ne: int,
                               tol: float = 1e-9, max_iter: int = 50):
    """Avanca a adveccao por um passo semi-Lagrangeano: backtrace + localizacao + interpolacao.

    `X, Y, IEN` sao os arrays completos (pos-`mesh.elem_mini`, com o no
    de centroide/bolha na 4a coluna de `IEN`); `EToE` vem de
    `mesh.montar_EToE` sobre a malha original (so vertices); `vx, vy`
    sao os campos do passo atual (tamanho `npoints + ne`), em numpy.

    Vetorizado: calcula o pe da caracteristica de todos os nos de uma
    vez (Euler explicito, usando `vx/vy` diretamente -- os DOFs sao
    todos nodais, inclusive o de centroide, ver `interpolar_mini`),
    localiza todos em lote (`localizar_pontos_partida`, chute inicial
    `node_to_elem[n]` para vertices ou `n - npoints` para nos de
    centroide -- ver `mesh.montar_node_to_elem`) e interpola. Nos cujo
    pe da caracteristica saiu do dominio usam o valor de Dirichlet do
    contorno (`ccName`/`conditions`, mesma estrutura de
    `boundary.build_boundary_conditions`) quando definido para aquele
    componente; senao mantem o valor atual do proprio no (nos de
    centroide nunca tem nome de contorno, entao sempre caem nesse
    ultimo caso quando saem do dominio).

    Retorna (vx_star, vy_star), tamanho `npoints + ne`.
    """
    n_total = npoints + ne
    elem_start = np.concatenate([node_to_elem, np.arange(ne)])

    xd, yd = backtrace(X, Y, vx, vy, dt)
    elem, lambdas = localizar_pontos_partida(xd, yd, elem_start, X, Y, IEN, EToE, tol, max_iter)

    vx_star, vy_star = np.copy(vx), np.copy(vy)  # default: mantem o proprio valor

    localizados = np.where(elem != -1)[0]
    i, j, k, b = IEN[elem[localizados]].T
    li, lj, lk = lambdas[localizados].T
    vx_star[localizados] = interpolar_mini(li, lj, lk, vx[i], vx[j], vx[k], vx[b])
    vy_star[localizados] = interpolar_mini(li, lj, lk, vy[i], vy[j], vy[k], vy[b])

    exited = elem == -1
    vx_bc, has_vx_bc, vy_bc, has_vy_bc = _fallback_dirichlet(ccName, conditions, npoints, n_total)
    usa_bc_x = exited & has_vx_bc
    usa_bc_y = exited & has_vy_bc
    vx_star[usa_bc_x] = vx_bc[usa_bc_x]
    vy_star[usa_bc_y] = vy_bc[usa_bc_y]

    return vx_star, vy_star