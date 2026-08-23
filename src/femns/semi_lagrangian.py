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

from femns.boundary import valores_da_condicao


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


def interpolar_tri6(li: float, lj: float, lk: float,
                     vi: float, vj: float, vk: float,
                     vij: float, vjk: float, vki: float):
    """Avalia um campo do espaco Tri6 (base nodal P2) num ponto de coordenadas baricentricas dadas.

    `vi, vj, vk` sao os valores nodais nos 3 vertices; `vij, vjk, vki`
    os valores nos 3 nos de aresta (convencao de `mesh.elem_tri6`: `vij`
    e' o no da aresta (i,j) -- a face local `f` do elemento tem seu no
    de aresta na coluna `3+f` de `IEN`, na mesma ordem `arestas_locais =
    [(0,1),(1,2),(2,0)]`). Os 6 DOFs sao nodais (mesma propriedade de
    Lagrange do MINI, ver `interpolar_mini`) -- as funcoes de forma
    correspondentes sao as do triangulo P2 padrao (Zienkiewicz vol. 1,
    cap. 8, mesma base de `assembly.assemble_tri6`):
    `Ni = li*(2*li - 1)` (e ciclicamente `Nj, Nk`) para os vertices e
    `Nij = 4*li*lj` (e ciclicamente `Njk, Nki`) para os nos de aresta.

    Igual ao MINI, os lambdas sao recortados para [0,1] e renormalizados
    antes de usar como peso (ver `interpolar_mini` sobre o motivo e a
    ressalva: essa politica de clamp foi herdada por analogia, nao
    re-derivada para a base quadratica -- ver `docs/lambda.md`).
    """
    li, lj, lk = np.clip(li, 0.0, 1.0), np.clip(lj, 0.0, 1.0), np.clip(lk, 0.0, 1.0)
    soma = li + lj + lk
    li, lj, lk = li / soma, lj / soma, lk / soma

    Ni, Nj, Nk = li * (2.0 * li - 1.0), lj * (2.0 * lj - 1.0), lk * (2.0 * lk - 1.0)
    Nij, Njk, Nki = 4.0 * li * lj, 4.0 * lj * lk, 4.0 * lk * li

    return Ni * vi + Nj * vj + Nk * vk + Nij * vij + Njk * vjk + Nki * vki


def _fallback_dirichlet(ccName: list, conditions: dict, npoints: int, n_total: int,
                         X: np.ndarray = None, Y: np.ndarray = None):
    """Monta, uma vez por chamada, os arrays de valor/disponibilidade de Dirichlet por no.

    So os `npoints` primeiros nos (vertices) sao consultados aqui --
    `nomes = ccName[:npoints]` abaixo -- entao todo no extra (indice >=
    `npoints`) sai desta funcao com `has_*_bc=False`, incondicionalmente.
    Para o elemento MINI isso e' exatamente correto: o no de
    centroide/bolha e' sempre interior ao elemento, nunca tem nome de
    contorno de verdade. Para o elemento Tri6 isso e' uma limitacao real,
    nao so' uma correcao de docstring: os nos de aresta que caem sobre o
    contorno **tem** nome (`ccName` estendido por
    `boundary.assign_boundary_names` ate `nnodes`, via
    `mesh.estende_IENbound_tri6`), mas esta funcao os ignora do mesmo
    jeito. Na pratica isso nao morde porque `run_simulation.py` sempre
    passa `bc_dirichlet` ja resolvido por
    `boundary.build_boundary_conditions` (que enxerga todos os `nnodes`
    nos) em vez de deixar `calculo_sl` cair neste fallback -- ver
    `calculo_sl`. Este fallback so e' de fato exercitado, hoje, por
    chamadas diretas/testes sem `bc_dirichlet`, e la' so' para contorno
    uniforme (ver aviso sobre perfil abaixo). Generalizar para nos de
    aresta nomeados fica para quando/se este caminho precisar valer para
    Tri6 tambem.

    Usado como fallback vetorizado em `calculo_sl` para nos cujo pe da
    caracteristica saiu do dominio.

    Agrupa por nome de contorno em vez de percorrer no a no: alem de ser
    vetorizado, e' o que permite condicao com **perfil**
    (`boundary.valores_da_condicao`), que precisa ver todos os nos daquele
    contorno de uma vez para saber onde ficam as pontas. `X, Y` so sao
    necessarios se algum contorno usar perfil.
    """
    vx_bc = np.zeros(n_total)
    vy_bc = np.zeros(n_total)
    has_vx_bc = np.zeros(n_total, dtype=bool)
    has_vy_bc = np.zeros(n_total, dtype=bool)

    nomes = np.array(ccName[:npoints], dtype=object)

    for nome, valores in conditions.items():
        idx = np.where(nomes == nome)[0]
        if idx.size == 0:
            continue
        if "vx" in valores:
            vx_bc[idx] = valores_da_condicao(valores["vx"], idx, X, Y)
            has_vx_bc[idx] = True
        if "vy" in valores:
            vy_bc[idx] = valores_da_condicao(valores["vy"], idx, X, Y)
            has_vy_bc[idx] = True

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
                             X_no: np.ndarray = None, Y_no: np.ndarray = None,
                             elemento: str = "mini"):
    """Velocidade no ponto onde a caracteristica cruzou o contorno, por interpolacao na aresta.

    Para cada no marcado em `alvos` (booleano, os que sairam do dominio e
    tem face de saida registrada em `saida`), acha o cruzamento com a
    aresta de contorno (`interceptar_contorno`) e avalia o campo ali.

    **MINI** (`elemento="mini"`): a interpolacao na aresta e' linear
    entre os dois vertices, sem termo de bolha -- e' exato, nao uma
    aproximacao: sobre qualquer aresta do triangulo uma das coordenadas
    baricentricas e' zero, entao a correcao de bolha `9*li*lj*lk` das
    funcoes de forma do MINI se anula e `Ni = li` (ver
    `interpolar_mini`). O campo MINI restrito a uma aresta E' P1. Por
    isso o `Tri4.jumpToElem` da referencia tambem devolve so os dois nos
    da aresta nesse caso.

    **Tri6** (`elemento="tri6"`): pelo mesmo argumento (uma coordenada
    baricentrica -- a do vertice oposto -- e' zero sobre a aresta), so
    3 das 6 funcoes de forma do Tri6 sao nao-nulas ali: as dos 2
    vertices da aresta e a do no de aresta correspondente; as outras 3
    (vertice oposto e os 2 nos das outras arestas) se anulam porque tem
    a coordenada do vertice oposto como fator (ver `interpolar_tri6`).
    Com `s` a posicao relativa do cruzamento ao longo da aresta (`s=0`
    no primeiro vertice, `s=1` no segundo -- `s = w2` abaixo), essas 3
    funcoes de forma restritas a aresta sao os 3 polinomios de Lagrange
    quadraticos em `s`: `(1-s)*(1-2*s)`, `s*(2*s-1)`, `4*s*(1-s)`. E'
    exato para o campo Tri6 pelo mesmo motivo do MINI, so que com base
    quadratica em vez de linear -- por isso precisa do no de aresta
    (`IEN[e, 3+f]`, mesma convencao de `mesh.elem_tri6`), nao so os 2
    vertices.

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

    if elemento == "tri6":
        ibm = IEN[e, 3 + f][ok]
        s = w2  # w1*B1 + w2*B2 == (1-s)*B1 + s*B2
        peso1, peso2, pesom = (1.0 - s) * (1.0 - 2.0 * s), s * (2.0 * s - 1.0), 4.0 * s * (1.0 - s)
        vx_int = peso1 * vx[ib1] + peso2 * vx[ib2] + pesom * vx[ibm]
        vy_int = peso1 * vy[ib1] + peso2 * vy[ib2] + pesom * vy[ibm]
    else:
        vx_int = w1 * vx[ib1] + w2 * vx[ib2]
        vy_int = w1 * vy[ib1] + w2 * vy[ib2]

    return idx[ok], vx_int, vy_int


def calculo_sl(X: np.ndarray, Y: np.ndarray, IEN: np.ndarray, EToE: np.ndarray,
                               node_to_elem: np.ndarray, ccName: list, conditions: dict,
                               vx: np.ndarray, vy: np.ndarray, dt: float, npoints: int, ne: int,
                               tol: float = 1e-9, max_iter: int = 50,
                               fora_dominio: str = "intercept",
                               elemento: str = "mini",
                               elem_start_extra: np.ndarray = None,
                               X_campo: np.ndarray = None, Y_campo: np.ndarray = None,
                               bc_dirichlet: tuple = None):
    """Avanca a adveccao por um passo semi-Lagrangeano: backtrace + localizacao + interpolacao.

    `X, Y, IEN` sao os arrays completos (pos-`mesh.elem_mini` ou
    `mesh.elem_tri6`, conforme `elemento`, com os nos extras nas colunas
    depois da 3a de `IEN`); `EToE` vem de `mesh.montar_EToE` sobre a
    malha original (so vertices, 3 colunas -- e' a mesma `EToE` para
    qualquer `elemento`, a caminhada so anda por vertice); `vx, vy` sao
    os campos do passo atual (tamanho `npoints` + numero de nos extras),
    em numpy.

    `elemento` escolhe a base de interpolacao usada nos nos localizados
    dentro da malha:

    - `"mini"` (padrao): 4 DOFs por elemento (3 vertices + 1
      centroide/bolha, na 4a coluna de `IEN`), interpolados por
      `interpolar_mini`.
    - `"tri6"`: 6 DOFs por elemento (3 vertices + 3 nos de aresta, nas
      colunas 3:6 de `IEN`, convencao de `mesh.elem_tri6`), interpolados
      por `interpolar_tri6`.

    Vetorizado: calcula o pe da caracteristica de todos os nos de uma
    vez (Euler explicito, usando `vx/vy` diretamente -- os DOFs sao
    todos nodais, inclusive os extras, ver `interpolar_mini`/
    `interpolar_tri6`), localiza todos em lote
    (`localizar_pontos_partida`, chute inicial `node_to_elem[n]` para
    vertices ou `elem_start_extra[n - npoints]` para nos extras -- ver
    `mesh.montar_node_to_elem`/`mesh.montar_edge_to_elem`) e interpola.

    `elem_start_extra` e' a semente de caminhada para os nos extras
    (tamanho = numero de nos extras, isto e' `len(vx) - npoints`). Se
    omitido, so e' valido para `elemento="mini"`: usa `np.arange(ne)`
    (o no extra `npoints + e` E' o centroide do elemento `e`, unico caso
    em que essa correspondencia direta existe -- ver a nota sobre `ne`
    abaixo). Para `elemento="tri6"` e' obrigatorio, porque o no de
    aresta nao tem uma unica "semente natural": ele e' compartilhado por
    ate 2 elementos, nenhum privilegiado (ver `mesh.montar_edge_to_elem`,
    que constroi essa semente varrendo `IEN[:, 3:6]`).

    Nota sobre `ne`: e' o numero de elementos da malha (`mesh.ne`), nao
    o numero de nos extras -- os dois coincidem so' no MINI (um
    centroide por elemento). No Tri6 nao coincidem (nos de aresta sao
    deduplicados entre vizinhos), por isso `ne` so' e' usado aqui para
    construir o `elem_start_extra` default do MINI; o numero de nos
    extras de fato usado (`n_total - npoints`) vem de `len(vx)`, nunca
    de `ne` diretamente.

    `fora_dominio` escolhe o tratamento dos nos cujo pe da caracteristica
    caiu fora da malha (os dois ficam lado a lado para comparacao, mesmo
    padrao de `advection`/`element`):

    - `"dirichlet"` (comportamento historico do projeto, mantido para
      reproduzir runs antigos): usa o valor de Dirichlet do contorno
      *do proprio no de chegada*
      (`ccName`/`conditions`, mesma estrutura de
      `boundary.build_boundary_conditions`) quando definido para aquele
      componente; senao mantem o valor atual do proprio no. Nos
      extras: no MINI (centroide) nunca tem nome de contorno, entao
      sempre caem no fallback do proprio no; no Tri6 o no de aresta
      *pode* ter nome (se cair no contorno), mas so' quando
      `bc_dirichlet` e' passado pronto -- o fallback interno
      `_fallback_dirichlet` nao ve nos extras nomeados de jeito nenhum,
      ver seu docstring.
    - `"intercept"` (padrao): tratamento do codigo de referencia do
      professor -- calcula onde a caracteristica de fato cruzou o contorno e
      interpola o campo *atual* nos nos daquela aresta (ver
      `_fallback_interceptacao`: 2 nos de vertice no MINI, +1 no de
      aresta no Tri6). Nao depende de o no de chegada ter Dirichlet,
      entao trata corretamente saida por contorno que nao prescreve
      velocidade (ex.: `outlet`, que so define `p`), onde o modo
      `"dirichlet"` congela o no. So cai no fallback de Dirichlet no
      residuo geometricamente degenerado (caracteristica paralela a
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

    `bc_dirichlet`, se dado, e' a tupla `(vx_bc, has_vx_bc, vy_bc,
    has_vy_bc)` ja resolvida por no -- e' o que `run_simulation.py`
    passa, reaproveitando o que `boundary.build_boundary_conditions` ja
    calculou. Sem ela, `ccName`/`conditions` sao reinterpretados aqui por
    `_fallback_dirichlet`, que **nao enxerga a extensao geometrica dos
    contornos** e por isso encolhe condicoes com perfil (ver
    `boundary.valores_da_condicao`); esse caminho so serve para contorno
    uniforme, onde os dois coincidem exatamente.

    Retorna (vx_star, vy_star), mesmo tamanho de `vx`/`vy`.
    """
    if fora_dominio not in ("dirichlet", "intercept"):
        raise ValueError(f"fora_dominio desconhecido: {fora_dominio!r} (use 'dirichlet' ou 'intercept')")
    if elemento not in ("mini", "tri6"):
        raise ValueError(f"elemento desconhecido: {elemento!r} (use 'mini' ou 'tri6')")

    if elem_start_extra is None:
        if elemento == "tri6":
            raise ValueError(
                "elem_start_extra e' obrigatorio para elemento='tri6' -- o no de aresta nao tem "
                "uma semente de elemento generica como o centroide do MINI (`np.arange(ne)` so' "
                "vale quando o no extra `npoints + e` E' o centroide do elemento `e`). "
                "Ver mesh.montar_edge_to_elem."
            )
        elem_start_extra = np.arange(ne)

    X_campo = X if X_campo is None else X_campo
    Y_campo = Y if Y_campo is None else Y_campo

    n_total = len(vx)
    elem_start = np.concatenate([node_to_elem, elem_start_extra])

    xd, yd = backtrace(X, Y, vx, vy, dt)
    elem, lambdas, saida = localizar_pontos_partida(
        xd, yd, elem_start, X_campo, Y_campo, IEN, EToE, tol, max_iter)

    vx_star, vy_star = np.copy(vx), np.copy(vy)  # default: mantem o proprio valor

    localizados = np.where(elem != -1)[0]
    li, lj, lk = lambdas[localizados].T
    if elemento == "tri6":
        i, j, k, eij, ejk, eki = IEN[elem[localizados]].T
        vx_star[localizados] = interpolar_tri6(li, lj, lk, vx[i], vx[j], vx[k], vx[eij], vx[ejk], vx[eki])
        vy_star[localizados] = interpolar_tri6(li, lj, lk, vy[i], vy[j], vy[k], vy[eij], vy[ejk], vy[eki])
    else:
        i, j, k, b = IEN[elem[localizados]].T
        vx_star[localizados] = interpolar_mini(li, lj, lk, vx[i], vx[j], vx[k], vx[b])
        vy_star[localizados] = interpolar_mini(li, lj, lk, vy[i], vy[j], vy[k], vy[b])

    exited = elem == -1
    restantes = exited  # quem ainda precisa do fallback de Dirichlet

    if fora_dominio == "intercept":
        idx, vx_int, vy_int = _fallback_interceptacao(
            X_campo, Y_campo, IEN, xd, yd, saida, exited & (saida[:, 0] != -1), vx, vy,
            X_no=X, Y_no=Y, elemento=elemento)
        vx_star[idx], vy_star[idx] = vx_int, vy_int
        restantes = restantes.copy()
        restantes[idx] = False

    if bc_dirichlet is not None:
        vx_bc, has_vx_bc, vy_bc, has_vy_bc = bc_dirichlet
    else:
        vx_bc, has_vx_bc, vy_bc, has_vy_bc = _fallback_dirichlet(
            ccName, conditions, npoints, n_total, X_campo, Y_campo)
    usa_bc_x = restantes & has_vx_bc
    usa_bc_y = restantes & has_vy_bc
    vx_star[usa_bc_x] = vx_bc[usa_bc_x]
    vy_star[usa_bc_y] = vy_bc[usa_bc_y]

    return vx_star, vy_star