"""Backtrace, localizacao de ponto e interpolacao para adveccao semi-Lagrangeana.

Logica pura de malha + interpolacao (numpy, sem torch) -- ver
docs/00.archives/semi_lagrangian_strategy.pdf para a derivacao e as decisoes de
projeto por tras de cada funcao aqui. A busca por elemento
(`Localizador.localizar`) e vetorizada entre nos (mascara booleana, sem
laco Python por no) -- so o numero de saltos da caminhada em si e
sequencial, por natureza do algoritmo.

O pe da caracteristica que cai fora da malha e' tratado por interceptacao
geometrica da aresta de saida, como o codigo de referencia do professor em
`referencia/clSemiLagrangian.py` -- ver `calculo_sl`. (A v1.0 tinha tambem
um modo `"dirichlet"`, removido na v2.0.)
"""

import numpy as np

from femns.elemento import MINI
from femns.mesh import montar_EToE, montar_no_para_elem


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


class Localizador:
    """Busca por elemento (caminhada pela vizinhanca) sobre a topologia de uma malha.

    Guarda so o que **nao muda** quando a malha se move: a conectividade
    `IEN`, os vizinhos por face (`EToE`) e a semente de cada no (o elemento
    de partida da caminhada). E' criado uma vez por malha; a cada passo --
    a malha tendo se movido ou nao -- chama-se `localizar` com a geometria
    em que a busca deve acontecer.

    A geometria (`X, Y`) entra **em cada chamada**, e nao fica guardada no
    objeto, de proposito: com malha movel (ALE) a busca tem que acontecer na
    malha em que o campo vive (a do passo anterior), enquanto o backtrace
    parte da malha nova -- ver `calculo_sl`. Guardar coordenadas no objeto e
    atualiza-las "quando a malha mover" abriria espaco para buscar na malha
    errada sem erro nenhum (medido em 2026-08-16: ate 15% da escala de
    velocidade).

    `IEN` pode ser a estendida (MINI ou Tri6): a caminhada so usa as 3
    primeiras colunas, e a semente cobre todos os nos (vertices e extras).
    """

    def __init__(self, IEN: np.ndarray, n_nos: int = None, tol: float = 1e-9, max_iter: int = 50):
        self.IEN = np.asarray(IEN)
        self.EToE = montar_EToE(self.IEN)
        n_nos = int(self.IEN.max()) + 1 if n_nos is None else n_nos
        self.semente = montar_no_para_elem(self.IEN, n_nos)
        self.tol = tol
        self.max_iter = max_iter

    def localizar(self, xd: np.ndarray, yd: np.ndarray, X: np.ndarray, Y: np.ndarray,
                  inicio: np.ndarray = None):
        """Localiza, para cada ponto `(xd, yd)`, o elemento que o contem na geometria `(X, Y)`.

        `inicio` e' o elemento de partida de cada ponto; omitido, usa
        `self.semente` -- o caso do semi-Lagrangeano, em que o ponto `i` e'
        o pe da caracteristica do no `i` (`len(xd)` = numero de nos).

        Caminhada: anda pelos vizinhos de `EToE` na direcao da coordenada
        baricentrica mais negativa (vertice local `m` -> face `(m+1)%3`,
        convencao de `mesh.montar_EToE`), todos os pontos em paralelo via
        mascara booleana -- o laco externo roda o maior numero de saltos
        entre os pontos ainda ativos, nao um laco por ponto.

        Retorna `(elem, lambdas, saida)`: `elem` `(n,)` com `-1` nos nao
        localizados (fora do dominio ou sem convergir em `max_iter`);
        `lambdas` `(n,3)`, com lixo nas linhas nao localizadas (filtre por
        `elem != -1`); `saida` `(n,2)` com `[elemento, face]` de contorno por
        onde a caminhada saiu do dominio, e `-1` para quem foi localizado ou
        nao convergiu (sem face de saida) -- e' o que a interceptacao de
        contorno usa (ver `interceptar_contorno`).
        """
        if inicio is None:
            inicio = self.semente
        n = len(xd)
        elem = np.asarray(inicio, dtype=int).copy() # Vetor com os elementos atuais de cada ponto
        done = np.zeros(n, dtype=bool) # Inicializa o vetor de pontos com seu ponto de partida já encontrados
        lambdas_out = np.zeros((n, 3)) # Inicializa o vetor de lambdas dos elementos finais
        saida = np.full((n, 2), -1, dtype=int) # [elemento, face] de contorno por onde cada ponto saiu

        for _ in range(self.max_iter):
            ativos = np.where(~done)[0] # Identifica quais pontos de partida ainda estão sendo buscados 
            if ativos.size == 0: # Interrompe antes de max_iter se todos os pontos tiverem sido encontrados
                break

            lambdas = np.stack(baricentro(xd[ativos], yd[ativos], X, Y, self.IEN, elem[ativos]), axis=1) # Calcula os lambdas de todos os pontos que estão sendo buscados
            dentro = lambdas.min(axis=1) >= -self.tol # Para os lambdas calculados, verifica quais indicam que o ponto está dentro do elemento

            idx_dentro = ativos[dentro] # Recebe o index da mascara "dentro", indicando quais elementos de partida foram encontrados
            lambdas_out[idx_dentro] = lambdas[dentro] # Marca os lambdas finais dos elementos de partida encontrados
            done[idx_dentro] = True # Inclui nos finalizados

            idx_fora = ativos[~dentro] # Pega o index de quem não encontrou um elemento
            m = lambdas[~dentro].argmin(axis=1) # Identifica qual o index do menor lambda de quem ficou fora
            face = (m + 1) % 3 # Indica qual a face oposta ao vértice com menor lambda de quem ficou fora
            proximo = self.EToE[elem[idx_fora], face] # Indica qual o próximo elemento a ser analisado para buscar o ponto de partida

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

    def localizar_ponto(self, x: float, y: float, X: np.ndarray, Y: np.ndarray, inicio: int = 0):
        """Um ponto so: `(elem, (li, lj, lk))`, ou `(None, None)` se estiver
        fora do dominio ou a caminhada nao convergir."""
        elem, lambdas, _ = self.localizar(np.array([x]), np.array([y]), X, Y, np.array([inicio]))
        if elem[0] == -1:
            return None, None
        return int(elem[0]), tuple(lambdas[0])


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
                             elemento=MINI):
    """Velocidade no ponto onde a caracteristica cruzou o contorno, por interpolacao na aresta.

    Para cada no marcado em `alvos` (booleano, os que sairam do dominio e
    tem face de saida registrada em `saida`), acha o cruzamento com a
    aresta de contorno (`interceptar_contorno`) e avalia o campo ali.

    O valor na aresta vem de `elemento.valores_na_aresta` (`femns.elemento`):

    **MINI**: a interpolacao na aresta e' linear
    entre os dois vertices, sem termo de bolha -- e' exato, nao uma
    aproximacao: sobre qualquer aresta do triangulo uma das coordenadas
    baricentricas e' zero, entao a correcao de bolha `9*li*lj*lk` das
    funcoes de forma do MINI se anula e `Ni = li` (ver
    `elemento.Mini.funcoes_forma`). O campo MINI restrito a uma aresta E' P1. Por
    isso o `Tri4.jumpToElem` da referencia tambem devolve so os dois nos
    da aresta nesse caso.

    **Tri6**: pelo mesmo argumento (uma coordenada
    baricentrica -- a do vertice oposto -- e' zero sobre a aresta), so
    3 das 6 funcoes de forma do Tri6 sao nao-nulas ali: as dos 2
    vertices da aresta e a do no de aresta correspondente; as outras 3
    (vertice oposto e os 2 nos das outras arestas) se anulam porque tem
    a coordenada do vertice oposto como fator (ver `elemento.Tri6.funcoes_forma`).
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

    e, f = e[ok], f[ok]
    ib1, ib2 = ib1[ok], ib2[ok]
    w1, w2 = w1[ok], w2[ok]

    vx_int = elemento.valores_na_aresta(IEN, e, f, ib1, ib2, w1, w2, vx)
    vy_int = elemento.valores_na_aresta(IEN, e, f, ib1, ib2, w1, w2, vy)

    return idx[ok], vx_int, vy_int


def calculo_sl(X: np.ndarray, Y: np.ndarray, localizador: Localizador,
                               vx: np.ndarray, vy: np.ndarray, dt: float,
                               bc_dirichlet: tuple,
                               elemento=MINI,
                               X_campo: np.ndarray = None, Y_campo: np.ndarray = None):
    """Avanca a adveccao por um passo semi-Lagrangeano: backtrace + localizacao + interpolacao.

    `X, Y` sao as coordenadas de todos os nos (pos-`mesh.elem_mini` ou
    `mesh.elem_tri6`, conforme o elemento); `localizador` e' o
    `Localizador` da malha, criado uma vez com a `IEN` estendida (ele
    carrega a `IEN`, os vizinhos e a semente da caminhada, e define `tol`
    e `max_iter` da busca); `vx, vy` sao os campos do passo atual (um
    valor por no, vertices e extras), em numpy.

    `elemento` e' o tipo de elemento (`femns.elemento.MINI`, padrao, ou
    `TRI6`). Ele decide a interpolacao nos nos localizados
    (`elemento.interpolar`: funcoes de forma do MINI ou do Tri6) e na
    aresta de saida (`elemento.valores_na_aresta`).

    Vetorizado: calcula o pe da caracteristica de todos os nos de uma
    vez (Euler explicito, usando `vx/vy` diretamente -- os DOFs sao
    todos nodais, inclusive os extras), localiza todos em lote
    (`localizador.localizar`) e interpola.

    Nos cujo pe da caracteristica caiu fora da malha: tratamento do
    codigo de referencia do professor -- calcula onde a caracteristica de
    fato cruzou o contorno e interpola o campo *atual* nos nos daquela
    aresta (ver `_fallback_interceptacao`: 2 nos de vertice no MINI, +1 no
    de aresta no Tri6). Nao depende de o no de chegada ter Dirichlet,
    entao trata corretamente saida por contorno que nao prescreve
    velocidade (ex.: `outlet`, que so define `p`). Sobra um residuo sem
    cruzamento util -- caracteristica paralela a aresta (geometricamente
    degenerado) ou caminhada que nao convergiu em `max_iter` (sem face de
    saida registrada): esses nos usam o valor de Dirichlet de
    `bc_dirichlet` quando o componente e' prescrito, senao mantem o
    proprio valor.

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

    `bc_dirichlet` e' a tupla `(vx_bc, has_vx_bc, vy_bc, has_vy_bc)`, um
    valor e uma mascara por no (tamanho `len(vx)`), ja resolvida por
    `boundary.build_boundary_conditions` -- e' o que `run_simulation.py`
    passa. Obrigatoria: reinterpretar o config aqui (o que a v1.0 fazia
    sem ela) nao enxerga a extensao geometrica dos contornos e encolhia
    condicoes com perfil.

    Retorna (vx_star, vy_star), mesmo tamanho de `vx`/`vy`.
    """
    IEN = localizador.IEN
    X_campo = X if X_campo is None else X_campo
    Y_campo = Y if Y_campo is None else Y_campo

    xd, yd = backtrace(X, Y, vx, vy, dt)
    elem, lambdas, saida = localizador.localizar(xd, yd, X_campo, Y_campo)

    vx_star, vy_star = np.copy(vx), np.copy(vy)  # default: mantem o proprio valor

    localizados = np.where(elem != -1)[0]
    li, lj, lk = lambdas[localizados].T
    nos = IEN[elem[localizados]]
    vx_star[localizados] = elemento.interpolar(li, lj, lk, nos, vx)
    vy_star[localizados] = elemento.interpolar(li, lj, lk, nos, vy)

    exited = elem == -1
    idx, vx_int, vy_int = _fallback_interceptacao(
        X_campo, Y_campo, IEN, xd, yd, saida, exited & (saida[:, 0] != -1), vx, vy,
        X_no=X, Y_no=Y, elemento=elemento)
    vx_star[idx], vy_star[idx] = vx_int, vy_int

    # residuo sem cruzamento util (degenerado ou sem convergencia): Dirichlet
    restantes = exited.copy()
    restantes[idx] = False
    vx_bc, has_vx_bc, vy_bc, has_vy_bc = bc_dirichlet
    usa_bc_x = restantes & has_vx_bc
    usa_bc_y = restantes & has_vy_bc
    vx_star[usa_bc_x] = vx_bc[usa_bc_x]
    vy_star[usa_bc_y] = vy_bc[usa_bc_y]

    return vx_star, vy_star