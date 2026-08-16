"""Backtrace, localizacao de ponto e interpolacao para adveccao semi-Lagrangeana.

Logica pura de malha + interpolacao (numpy, sem torch) -- ver
docs/semi_lagrangian_strategy.pdf para a derivacao e as decisoes de
projeto por tras de cada funcao aqui. A busca por elemento
(`localizar_pontos_partida`) e vetorizada entre nos (mascara booleana, sem
laco Python por no) -- so o numero de saltos da caminhada em si e
sequencial, por natureza do algoritmo.

Ha dois tratamentos para o pe da caracteristica que cai fora da malha,
selecionaveis em `calculo_sl(fora_dominio=...)` e por
`simulation.sl_boundary` no config: `"dirichlet"` (historico deste
projeto) e `"intercept"` (interceptacao geometrica da aresta de saida,
como o codigo de referencia do professor em
`referencia/clSemiLagrangian.py`) -- ver `calculo_sl`.
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
    lambdas, saida)`: `elem` e' `(n,)` com `-1` nas posicoes nao localizadas
    (fora do dominio ou que nao convergiram em `max_iter`); `lambdas`
    e' `(n,3)`, com lixo nas linhas nao localizadas (quem usa o
    retorno deve filtrar por `elem != -1` antes); `saida` e' `(n,2)`
    com `[elemento, face]` de contorno por onde a caminhada saiu do
    dominio, e `-1` nas duas colunas para quem foi localizado ou nao
    convergiu em `max_iter` (nesse caso nao ha face de saida). A face e'
    a convencao de `mesh.montar_EToE`: face `f` liga os vertices locais
    `f` e `(f+1)%3`. E' o que o tratamento de contorno por interceptacao
    geometrica precisa para saber em qual aresta interpolar (ver
    `interceptar_contorno`).
    """
    n = len(xd)
    elem = np.asarray(elem_start, dtype=int).copy() # Vetor com os elementos atuais de cada ponto
    done = np.zeros(n, dtype=bool) # Inicializa o vetor de pontos com seu ponto de partida já encontrados
    lambdas_out = np.zeros((n, 3)) # Inicializa o vetor de lambdas dos elementos finais
    saida = np.full((n, 2), -1, dtype=int) # [elemento, face] de contorno por onde cada ponto saiu

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

        sem_vizinho = proximo == -1
        saiu_dominio = idx_fora[sem_vizinho] # Indica o index dos pontos em que a busca caiu fora do domínio
        saida[saiu_dominio, 0] = elem[saiu_dominio] # Guarda o elemento de contorno antes de marcar -1 abaixo
        saida[saiu_dominio, 1] = face[sem_vizinho] # Guarda a face de contorno atravessada
        done[saiu_dominio] = True # Se caiu fora do domínio, ele para de buscar
        elem[saiu_dominio] = -1 # Indica que o elemento de partida está fora do domínio

        tem_vizinho = ~sem_vizinho
        andando = idx_fora[tem_vizinho] # Atualiza o vetor de index dos pontos que vão continuar a busca no próximo passo
        elem[andando] = proximo[tem_vizinho] # Atualiza o vetor de elemento com os próximos elementos a serem analisados

    elem[~done] = -1  # nao convergiu em max_iter: mesmo tratamento de "fora do dominio"

    return elem, lambdas_out, saida


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
    elem, lambdas, _ = localizar_pontos_partida(
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


def interceptar_contorno(xn, yn, xd, yd, xb1, yb1, xb2, yb2, eps: float = 1e-14):
    """Onde o segmento no->pe da caracteristica cruza a aresta de contorno (b1,b2).

    Resolve, para cada ponto, o cruzamento entre o segmento R1->R2 (R1 =
    posicao do proprio no, R2 = pe da caracteristica, fora do dominio) e a
    reta da aresta de contorno B1->B2, escrevendo o ponto de cruzamento
    como `w1*B1 + w2*B2` (w1 + w2 = 1). Sistema 2x2 por ponto, resolvido
    por Cramer -- versao vetorizada de `computeIntercept` do codigo de
    referencia do professor (`referencia/clSemiLagrangian.py`):

        w1*(B1-B2) + t*(R1-R2) = R1 - B2

    Todos os argumentos sao arrays de mesmo shape `(n,)`. Retorna
    `(w1, w2, ok)`: os dois pesos ao longo da aresta e uma mascara de
    quais pontos tiveram cruzamento bem definido. `ok` e' False quando o
    determinante e' ~0, isto e' quando a caracteristica e' paralela a
    aresta de saida -- geometricamente degenerado, sem ponto de
    cruzamento util; quem chama trata esses a parte (cai no fallback de
    Dirichlet). Os pesos sao recortados para [0,1] para que a
    interpolacao resultante seja sempre uma combinacao convexa dos dois
    nos da aresta: sem isso, um cruzamento calculado fora do segmento
    (possivel em quinas, quando a face escolhida pela caminhada nao e'
    exatamente a face fisica de saida) viraria extrapolacao.
    """
    a1, b1, c1 = xb1 - xb2, xn - xd, xn - xb2
    a2, b2, c2 = yb1 - yb2, yn - yd, yn - yb2

    det = a1 * b2 - a2 * b1
    ok = np.abs(det) >= eps

    w1 = np.clip((c1 * b2 - c2 * b1) / np.where(ok, det, 1.0), 0.0, 1.0)

    return w1, 1.0 - w1, ok


def _fallback_interceptacao(X: np.ndarray, Y: np.ndarray, IEN: np.ndarray, xd: np.ndarray,
                             yd: np.ndarray, saida: np.ndarray, alvos: np.ndarray,
                             vx: np.ndarray, vy: np.ndarray, eps: float = 1e-14,
                             X_no: np.ndarray = None, Y_no: np.ndarray = None):
    """Velocidade no ponto onde a caracteristica cruzou o contorno, por interpolacao na aresta.

    Para cada no marcado em `alvos` (booleano, os que sairam do dominio e
    tem face de saida registrada em `saida`), acha o cruzamento com a
    aresta de contorno (`interceptar_contorno`) e avalia o campo ali.

    A interpolacao na aresta e' linear entre os dois vertices, sem termo
    de bolha -- e' exato, nao uma aproximacao: sobre qualquer aresta do
    triangulo uma das coordenadas baricentricas e' zero, entao a
    correcao de bolha `9*li*lj*lk` das funcoes de forma do MINI se anula
    e `Ni = li` (ver `interpolar_mini`). O campo MINI restrito a uma
    aresta E' P1. Por isso o `Tri4.jumpToElem` da referencia tambem
    devolve so os dois nos da aresta nesse caso.

    `X, Y` sao a geometria em que o campo vive (a aresta de contorno);
    `X_no, Y_no` as posicoes de chegada dos nos (o ponto R1 do segmento).
    Sao iguais em malha parada; com malha movel a aresta e' a da malha do
    passo anterior e o no ja esta na posicao nova (ver `calculo_sl`).

    Retorna `(indices, vx_int, vy_int)`: os indices de no efetivamente
    tratados (subconjunto de `alvos`, tirando os degenerados) e os
    valores interpolados para cada um.
    """
    X_no = X if X_no is None else X_no
    Y_no = Y if Y_no is None else Y_no

    idx = np.where(alvos)[0]
    e, f = saida[idx, 0], saida[idx, 1]

    ib1 = IEN[e, f]
    ib2 = IEN[e, (f + 1) % 3]

    w1, w2, ok = interceptar_contorno(
        X_no[idx], Y_no[idx], xd[idx], yd[idx], X[ib1], Y[ib1], X[ib2], Y[ib2], eps)

    ib1, ib2 = ib1[ok], ib2[ok]
    w1, w2 = w1[ok], w2[ok]

    return idx[ok], w1 * vx[ib1] + w2 * vx[ib2], w1 * vy[ib1] + w2 * vy[ib2]


def calculo_sl(X: np.ndarray, Y: np.ndarray, IEN: np.ndarray, EToE: np.ndarray,
                               node_to_elem: np.ndarray, ccName: list, conditions: dict,
                               vx: np.ndarray, vy: np.ndarray, dt: float, npoints: int, ne: int,
                               tol: float = 1e-9, max_iter: int = 50,
                               fora_dominio: str = "intercept",
                               X_campo: np.ndarray = None, Y_campo: np.ndarray = None):
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
    centroide -- ver `mesh.montar_node_to_elem`) e interpola.

    `fora_dominio` escolhe o tratamento dos nos cujo pe da caracteristica
    caiu fora da malha (os dois ficam lado a lado para comparacao, mesmo
    padrao de `advection`/`element`):

    - `"dirichlet"` (comportamento historico do projeto, mantido para
      reproduzir runs antigos): usa o valor de Dirichlet do contorno
      *do proprio no de chegada*
      (`ccName`/`conditions`, mesma estrutura de
      `boundary.build_boundary_conditions`) quando definido para aquele
      componente; senao mantem o valor atual do proprio no. Nos de
      centroide nunca tem nome de contorno, entao sempre caem nesse
      ultimo caso.
    - `"intercept"` (padrao): tratamento do codigo de referencia do
      professor -- calcula onde a caracteristica de fato cruzou o contorno e
      interpola o campo *atual* nos dois nos daquela aresta (ver
      `_fallback_interceptacao`). Nao depende de o no de chegada ter
      Dirichlet, entao trata corretamente saida por contorno que nao
      prescreve velocidade (ex.: `outlet`, que so define `p`), onde o
      modo `"dirichlet"` congela o no. So cai no fallback de Dirichlet
      no residuo geometricamente degenerado (caracteristica paralela a
      aresta) e nos que nao convergiram em `max_iter` (sem face de
      saida registrada).

    **Malha movel (ALE)**: `X, Y` sao as posicoes de *chegada* (a malha
    deste passo) e `X_campo, Y_campo` a geometria em que `vx, vy` de
    fato vivem (a malha do passo *anterior*, onde aquele campo foi
    resolvido). Se `X_campo/Y_campo` forem omitidos valem `X, Y` -- o
    caso de malha parada, em que as duas coincidem.

    A separacao importa porque o pe da caracteristica e' um ponto do
    espaco fisico: o backtrace parte da posicao nova do no (por onde o
    fluido chega), mas quem e' interpolado la e' o campo do passo
    anterior, discretizado na malha *velha*. Localizar elemento e avaliar
    funcao de forma na malha nova seria avaliar o campo numa geometria que
    nao e' a dele -- com os nos deslocados de ~0.3*h, um erro de ordem do
    proprio elemento. Note que o backtrace usa `vx[i]` (valor nodal do
    passo anterior) como velocidade do no `i`, e nao o campo reinterpolado
    na posicao nova: e' um erro O(h) no *pe*, da mesma ordem do Euler
    explicito do proprio backtrace, e evita uma interpolacao extra por
    passo.

    Retorna (vx_star, vy_star), tamanho `npoints + ne`.
    """
    if fora_dominio not in ("dirichlet", "intercept"):
        raise ValueError(f"fora_dominio desconhecido: {fora_dominio!r} (use 'dirichlet' ou 'intercept')")

    X_campo = X if X_campo is None else X_campo
    Y_campo = Y if Y_campo is None else Y_campo

    n_total = npoints + ne
    elem_start = np.concatenate([node_to_elem, np.arange(ne)])

    xd, yd = backtrace(X, Y, vx, vy, dt)
    elem, lambdas, saida = localizar_pontos_partida(
        xd, yd, elem_start, X_campo, Y_campo, IEN, EToE, tol, max_iter)

    vx_star, vy_star = np.copy(vx), np.copy(vy)  # default: mantem o proprio valor

    localizados = np.where(elem != -1)[0]
    i, j, k, b = IEN[elem[localizados]].T
    li, lj, lk = lambdas[localizados].T
    vx_star[localizados] = interpolar_mini(li, lj, lk, vx[i], vx[j], vx[k], vx[b])
    vy_star[localizados] = interpolar_mini(li, lj, lk, vy[i], vy[j], vy[k], vy[b])

    exited = elem == -1
    restantes = exited  # quem ainda precisa do fallback de Dirichlet

    if fora_dominio == "intercept":
        idx, vx_int, vy_int = _fallback_interceptacao(
            X_campo, Y_campo, IEN, xd, yd, saida, exited & (saida[:, 0] != -1), vx, vy,
            X_no=X, Y_no=Y)
        vx_star[idx], vy_star[idx] = vx_int, vy_int
        restantes = restantes.copy()
        restantes[idx] = False

    vx_bc, has_vx_bc, vy_bc, has_vy_bc = _fallback_dirichlet(ccName, conditions, npoints, n_total)
    usa_bc_x = restantes & has_vx_bc
    usa_bc_y = restantes & has_vy_bc
    vx_star[usa_bc_x] = vx_bc[usa_bc_x]
    vy_star[usa_bc_y] = vy_bc[usa_bc_y]

    return vx_star, vy_star