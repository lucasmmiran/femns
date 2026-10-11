import numpy as np

from femns.elemento import MINI, TRI6
from femns.mesh import elem_mini, elem_tri6, montar_EToE
from femns.semi_lagrangian import (
    Localizador,
    backtrace,
    baricentro,
    calculo_sl,
    interceptar_contorno,
)
from geometrias import quadrado_dois_triangulos


def interpolar_mini(li, lj, lk, vi, vj, vk, vb):
    """Atalho escalar: `MINI.interpolar` num ponto de um elemento com valores nodais dados."""
    v = np.array([vi, vj, vk, vb], dtype=float)
    return MINI.interpolar(np.array([li]), np.array([lj]), np.array([lk]), np.array([[0, 1, 2, 3]]), v)[0]


def interpolar_tri6(li, lj, lk, vi, vj, vk, vij, vjk, vki):
    """Atalho escalar: `TRI6.interpolar` num ponto de um elemento com valores nodais dados."""
    v = np.array([vi, vj, vk, vij, vjk, vki], dtype=float)
    return TRI6.interpolar(np.array([li]), np.array([lj]), np.array([lk]), np.array([[0, 1, 2, 3, 4, 5]]), v)[0]


def sem_dirichlet(n):
    """`bc_dirichlet` de `calculo_sl` sem nenhum componente prescrito (n nos)."""
    return np.zeros(n), np.zeros(n, dtype=bool), np.zeros(n), np.zeros(n, dtype=bool)


def test_baricentro_nos_vertices():
    X, Y, IEN = quadrado_dois_triangulos()

    li, lj, lk = baricentro(X[0], Y[0], X, Y, IEN, elem=0)
    assert np.allclose([li, lj, lk], [1.0, 0.0, 0.0])

    li, lj, lk = baricentro(X[1], Y[1], X, Y, IEN, elem=0)
    assert np.allclose([li, lj, lk], [0.0, 1.0, 0.0])

    li, lj, lk = baricentro(X[2], Y[2], X, Y, IEN, elem=0)
    assert np.allclose([li, lj, lk], [0.0, 0.0, 1.0])


def test_baricentro_centroide():
    X, Y, IEN = quadrado_dois_triangulos()
    v0, v1, v2 = IEN[0]
    xc = (X[v0] + X[v1] + X[v2]) / 3.0
    yc = (Y[v0] + Y[v1] + Y[v2]) / 3.0

    li, lj, lk = baricentro(xc, yc, X, Y, IEN, elem=0)
    assert np.allclose([li, lj, lk], [1 / 3, 1 / 3, 1 / 3])


def test_baricentro_ponto_fora():
    X, Y, IEN = quadrado_dois_triangulos()
    # (0,1) e o no 3, do outro lado da diagonal do elemento 0 (0,1,2)
    li, lj, lk = baricentro(0.0, 1.0, X, Y, IEN, elem=0)
    assert min(li, lj, lk) < 0.0
    assert np.isclose(li + lj + lk, 1.0)


def test_localizar_ponto_mesmo_elemento():
    X, Y, IEN = quadrado_dois_triangulos()

    elem, lambdas = Localizador(IEN).localizar_ponto(0.5, 0.2, X, Y, inicio=0)

    assert elem == 0
    assert min(lambdas) >= 0.0


def test_localizar_ponto_um_salto():
    X, Y, IEN = quadrado_dois_triangulos()

    # (0.2, 0.5) esta no elemento 1 (0,2,3), mas a busca comeca no elemento 0
    elem, lambdas = Localizador(IEN).localizar_ponto(0.2, 0.5, X, Y, inicio=0)

    assert elem == 1
    assert min(lambdas) >= 0.0
    assert np.allclose(lambdas, [0.5, 0.2, 0.3])


def test_localizar_ponto_sai_do_dominio():
    X, Y, IEN = quadrado_dois_triangulos()

    elem, lambdas = Localizador(IEN).localizar_ponto(-0.5, -0.5, X, Y, inicio=0)

    assert elem is None
    assert lambdas is None


def test_localizador_criado_uma_vez_serve_para_a_malha_movida():
    """A topologia nao muda quando a malha se move: o mesmo `Localizador`
    localiza na geometria que for passada em cada chamada. Movendo o no 2 de
    (1,1) para (0.5,1), a diagonal passa a ser y = 2x e o ponto (0.4, 0.6)
    muda do elemento 1 para o elemento 0."""
    X, Y, IEN = quadrado_dois_triangulos()
    loc = Localizador(IEN)

    assert loc.localizar_ponto(0.4, 0.6, X, Y)[0] == 1

    X_movido = X.copy()
    X_movido[2] = 0.5
    assert loc.localizar_ponto(0.4, 0.6, X_movido, Y)[0] == 0
    assert loc.localizar_ponto(0.4, 0.6, X, Y)[0] == 1  # nada ficou guardado no objeto


def test_baricentro_com_arrays():
    # A mesma funcao, sem nenhuma mudanca, deve aceitar arrays no lugar
    # de escalares (usado em lote por Localizador.localizar).
    X, Y, IEN = quadrado_dois_triangulos()

    xd = np.array([X[0], X[1], X[2]])
    yd = np.array([Y[0], Y[1], Y[2]])
    elem = np.array([0, 0, 0])

    li, lj, lk = baricentro(xd, yd, X, Y, IEN, elem)

    assert np.allclose(li, [1.0, 0.0, 0.0])
    assert np.allclose(lj, [0.0, 1.0, 0.0])
    assert np.allclose(lk, [0.0, 0.0, 1.0])


def test_localizar_saltos_diferentes_na_mesma_chamada():
    """Regressao da vetorizacao: 3 pontos que precisam de 0 saltos, 1
    salto e saida de dominio, todos na MESMA chamada, comecando do
    mesmo elemento -- valida que a mascara nao mistura resultados
    entre linhas (risco real de bug ao vetorizar por mascara).
    Os valores esperados sao os mesmos ja validados individualmente em
    test_localizar_ponto_mesmo_elemento/um_salto/sai_do_dominio.
    """
    X, Y, IEN = quadrado_dois_triangulos()

    xd = np.array([0.5, 0.2, -0.5])
    yd = np.array([0.2, 0.5, -0.5])
    elem_start = np.array([0, 0, 0])

    elem, lambdas, saida = Localizador(IEN).localizar(xd, yd, X, Y, inicio=elem_start)

    assert elem[0] == 0  # 0 saltos
    assert min(lambdas[0]) >= 0.0

    assert elem[1] == 1  # 1 salto
    assert np.allclose(lambdas[1], [0.5, 0.2, 0.3])

    assert elem[2] == -1  # saiu do dominio

    # Quem foi localizado nao tem face de saida; quem saiu tem a face registrada
    assert (saida[0] == -1).all() and (saida[1] == -1).all()
    assert saida[2, 0] != -1 and saida[2, 1] != -1


def test_localizar_registra_aresta_de_saida():
    """A face de saida guardada tem que ser a aresta de contorno que a
    caminhada de fato tentou atravessar -- e' ela que o tratamento por
    interceptacao usa para interpolar. Aqui o pe cai reto abaixo da aresta
    de baixo do elemento 0 (nos 0-1), entao a saida tem que ser
    [elem=0, face=0] (convencao de montar_EToE: face f liga os vertices
    locais f e (f+1)%3).
    """
    X, Y, IEN = quadrado_dois_triangulos()
    EToE = montar_EToE(IEN)

    elem, _, saida = Localizador(IEN).localizar(np.array([0.5]), np.array([-0.5]), X, Y, inicio=np.array([0]))

    assert elem[0] == -1
    assert saida[0, 0] == 0  # saiu pelo elemento 0
    assert saida[0, 1] == 0  # pela face 0 = vertices locais 0 e 1 = nos 0 e 1 (aresta de baixo)
    assert EToE[saida[0, 0], saida[0, 1]] == -1  # e essa face e' mesmo contorno


def test_interceptar_contorno_valores_a_mao():
    # Caracteristica vertical descendo de (0.25, 0.5) ate (0.25, -0.5),
    # cruzando a aresta (0,0)-(1,0) em (0.25, 0). Como o cruzamento e'
    # escrito como w1*B1 + w2*B2 = w1*(0,0) + w2*(1,0) = (w2, 0), tem que
    # sair w2 = 0.25 e w1 = 0.75.
    w1, w2, ok = interceptar_contorno(
        xn=np.array([0.25]), yn=np.array([0.5]),
        xd=np.array([0.25]), yd=np.array([-0.5]),
        xb1=np.array([0.0]), yb1=np.array([0.0]),
        xb2=np.array([1.0]), yb2=np.array([0.0]),
    )

    assert ok[0]
    assert np.isclose(w1[0], 0.75)
    assert np.isclose(w2[0], 0.25)
    assert np.isclose(w1[0] + w2[0], 1.0)


def test_interceptar_contorno_paralelo_e_degenerado():
    # Caracteristica horizontal contra uma aresta horizontal: nao ha
    # cruzamento bem definido (det == 0) -- tem que sair ok=False, para
    # quem chama mandar esses nos para o fallback de Dirichlet.
    _, _, ok = interceptar_contorno(
        xn=np.array([0.5]), yn=np.array([0.5]),
        xd=np.array([1.5]), yd=np.array([0.5]),
        xb1=np.array([0.0]), yb1=np.array([0.0]),
        xb2=np.array([1.0]), yb2=np.array([0.0]),
    )

    assert not ok[0]


def test_interpolar_mini_centroide():
    # Base nodal: no centroide (li=lj=lk=1/3), Ni=Nj=Nk=0 e Nb=1 -> v_star = vb, diretamente.
    v_star = interpolar_mini(1 / 3, 1 / 3, 1 / 3, vi=2.0, vj=4.0, vk=9.0, vb=10.0)
    assert np.isclose(v_star, 10.0)


def test_interpolar_mini_valores_arbitrarios():
    # Conferido a mao com a base nodal (Ni=li-9*li*lj*lk, Nb=27*li*lj*lk):
    # corr = 9*0.5*0.3*0.2 = 0.27
    # (0.5-0.27)*1 + (0.3-0.27)*2 + (0.2-0.27)*3 + 27*0.5*0.3*0.2*10 = 8.18
    v_star = interpolar_mini(0.5, 0.3, 0.2, vi=1.0, vj=2.0, vk=3.0, vb=10.0)
    assert np.isclose(v_star, 8.18)


def test_interpolar_mini_clampa_lambda_negativo():
    # li levemente negativo (tolerancia da busca por elemento) e recortado pra 0
    # antes de renormalizar -- nao deve gerar peso negativo no resultado.
    v_star = interpolar_mini(-1e-10, 0.5, 0.5, vi=100.0, vj=1.0, vk=1.0, vb=0.0)
    assert np.isclose(v_star, 1.0)


def test_interpolar_tri6_reproduz_quadratico_arbitrario():
    """A base P2 do Tri6 tem que reproduzir EXATAMENTE (precisao de
    maquina) um polinomio quadratico arbitrario -- e' a mesma logica da
    licao de 2026-07-12 (CLAUDE.local.md): um teste com valores
    escolhidos a mao pode "passar" com uma base errada se coincidir por
    acidente (foi o caso do bug da bolha do MINI, com bolha=0). Um
    polinomio com coeficientes aleatorios e triangulo nao-equilatero nao
    deixa esse espaco de coincidencia -- uma base P2 errada (ex.: usar
    so os 3 vertices, sem os nos de aresta) nao passaria aqui.
    """
    rng = np.random.default_rng(42)
    X0 = np.array([0.0, 1.0, 0.3])  # triangulo arbitrario, nao alinhado com os eixos
    Y0 = np.array([0.0, 0.2, 1.1])

    a, b, c, d, e, f = rng.uniform(-3.0, 3.0, size=6)

    def poly(x, y):
        return a + b * x + c * y + d * x**2 + e * x * y + f * y**2

    vi, vj, vk = poly(X0[0], Y0[0]), poly(X0[1], Y0[1]), poly(X0[2], Y0[2])
    xij, yij = (X0[0] + X0[1]) / 2, (Y0[0] + Y0[1]) / 2
    xjk, yjk = (X0[1] + X0[2]) / 2, (Y0[1] + Y0[2]) / 2
    xki, yki = (X0[2] + X0[0]) / 2, (Y0[2] + Y0[0]) / 2
    vij, vjk, vki = poly(xij, yij), poly(xjk, yjk), poly(xki, yki)

    li, lj, lk = rng.uniform(0.05, 0.9, size=3)
    soma = li + lj + lk
    li, lj, lk = li / soma, lj / soma, lk / soma
    x_teste = li * X0[0] + lj * X0[1] + lk * X0[2]
    y_teste = li * Y0[0] + lj * Y0[1] + lk * Y0[2]

    obtido = interpolar_tri6(li, lj, lk, vi, vj, vk, vij, vjk, vki)
    assert np.isclose(obtido, poly(x_teste, y_teste), atol=1e-12)


def test_interpolar_tri6_vertices_e_nos_de_aresta():
    # No vertice i (li=1): so Ni e' nao-nulo (Ni=1*(2*1-1)=1).
    assert np.isclose(interpolar_tri6(1.0, 0.0, 0.0, vi=5.0, vj=1.0, vk=1.0, vij=1.0, vjk=1.0, vki=1.0), 5.0)
    # No no de aresta (i,j) (li=lj=0.5, lk=0): Nij=4*0.5*0.5=1, resto 0.
    assert np.isclose(interpolar_tri6(0.5, 0.5, 0.0, vi=1.0, vj=1.0, vk=1.0, vij=7.0, vjk=1.0, vki=1.0), 7.0)


def test_backtrace_euler_translacao():
    X, Y, IEN = quadrado_dois_triangulos()
    vx_node = np.full_like(X, 2.0)
    vy_node = np.full_like(Y, -1.0)
    dt = 0.1

    xd, yd = backtrace(X, Y, vx_node, vy_node, dt)

    assert np.allclose(xd, X - 0.2)
    assert np.allclose(yd, Y + 0.1)


def test_calculo_sl_campo_constante():
    """Campo MINI que representa a constante (c, d): TODOS os DOFs (vertices
    e centroide/bolha) valem (c, d) -- a base e nodal (ver interpolar_mini),
    entao um campo constante de verdade tem o mesmo valor em todo DOF,
    igual a qualquer elemento P1 comum. A interpolacao deve recuperar
    exatamente (c,d) em todo no: nos que caem dentro da malha (no 0 e
    centroides) e tambem nos que saem do dominio (1, 2, 3), cujo valor vem
    da interceptacao da aresta de saida -- interpolar constante numa aresta
    tambem da a constante.
    """
    X0, Y0, IEN0 = quadrado_dois_triangulos()

    IEN, X, Y = elem_mini(IEN0, X0, Y0)
    loc = Localizador(IEN, len(X))

    c, d, dt = -0.05, -0.02, 1.0

    vx = np.full(6, c)
    vy = np.full(6, d)

    vx_star, vy_star = calculo_sl(
        X, Y, loc, vx, vy, dt, sem_dirichlet(6))

    assert np.allclose(vx_star, c) and np.allclose(vy_star, d)


def test_calculo_sl_residuo_sem_cruzamento_usa_dirichlet():
    """Nos que a caminhada nao resolve (aqui forcado com `max_iter=0`: nenhum
    passo de caminhada, ninguem localizado e ninguem com face de saida)
    nao tem cruzamento de contorno para interpolar. Cobre os 2 caminhos
    desse residuo: componente com Dirichlet prescrito em `bc_dirichlet`
    -> usa o valor prescrito; sem prescricao -> mantem o proprio valor.
    """
    X0, Y0, IEN0 = quadrado_dois_triangulos()

    IEN, X, Y = elem_mini(IEN0, X0, Y0)

    vx = np.arange(6, dtype=float)
    vy = -np.arange(6, dtype=float)

    vx_bc, has_vx, vy_bc, has_vy = sem_dirichlet(6)
    vx_bc[1], has_vx[1] = 99.0, True    # no 1: so vx prescrito
    vy_bc[2], has_vy[2] = -77.0, True   # no 2: so vy prescrito

    vx_star, vy_star = calculo_sl(
        X, Y, Localizador(IEN, len(X), max_iter=0), vx, vy, 0.1,
        (vx_bc, has_vx, vy_bc, has_vy))

    assert vx_star[1] == 99.0 and vy_star[1] == vy[1]
    assert vx_star[2] == vx[2] and vy_star[2] == -77.0
    # demais nos: nada prescrito -> proprio valor
    for n in (0, 3, 4, 5):
        assert vx_star[n] == vx[n] and vy_star[n] == vy[n]


def test_calculo_sl_centroide_e_nodal():
    """Regressao: o DOF de centroide/bolha e nodal (propriedade de
    Lagrange) -- a velocidade fisica no centroide E vx[centroide]
    diretamente, nao `(vi+vj+vk)/3 + vb` (formula errada descartada
    depois de bater a matriz de rigidez de `elemento.Mini`, entrada a
    entrada, contra a base nodal `Ni = li - 9*li*lj*lk`; ver
    `interpolar_mini`). Com dt desprezivel, v_star no no de centroide
    deve ficar bem perto do proprio vx[n] (nao da media dos vertices
    mais vx[n]).
    """
    X0, Y0, IEN0 = quadrado_dois_triangulos()

    IEN, X, Y = elem_mini(IEN0, X0, Y0)
    loc = Localizador(IEN, len(X))

    vx = np.array([1.0, 2.0, 3.0, 4.0, 5.0, -1.0])
    vy = np.zeros(6)
    dt = 1e-6  # deslocamento desprezivel: v_star deve ficar bem perto da velocidade fisica atual

    vx_star, _ = calculo_sl(
        X, Y, loc, vx, vy, dt, sem_dirichlet(6))

    # No de centroide do elemento 0 e' o no 4 (vb=5.0) -- deve recuperar isso diretamente
    assert np.isclose(vx_star[4], 5.0, atol=1e-4)


def test_calculo_sl_intercept_interpola_na_aresta_de_saida():
    """Valor da interceptacao calculado a mao, num no sem Dirichlet.

    O no 4 e' o centroide do elemento 0, em (2/3, 1/3). Com vx=0, vy=1 e
    dt=0.5 o pe da caracteristica cai em (2/3, -1/6) -- fora do dominio,
    reto abaixo da aresta de baixo (nos 0-1). O cruzamento e' em (2/3, 0),
    isto e' 1/3 do no 0 e 2/3 do no 1, entao o resultado tem que ser
    1/3*vx[0] + 2/3*vx[1] = 1/3*10 + 2/3*4 = 6.0 -- e nao o proprio valor
    do no (0.0), que e' o que o antigo modo `dirichlet` (v1.0) dava aqui,
    "congelando" a adveccao num no sem valor prescrito.
    """
    X0, Y0, IEN0 = quadrado_dois_triangulos()

    IEN, X, Y = elem_mini(IEN0, X0, Y0)
    loc = Localizador(IEN, len(X))

    vx = np.array([10.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    vy = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    dt = 0.5
    vx_int, _ = calculo_sl(X, Y, loc, vx, vy, dt,
                            sem_dirichlet(6))

    assert np.isclose(vx_int[4], 6.0)


def test_calculo_sl_intercept_reproduz_campo_linear_na_saida():
    """A interpolacao na aresta de saida e' exata para campo linear -- nao e'
    aproximacao. Sobre uma aresta uma coordenada baricentrica e' zero, a
    correcao de bolha `9*li*lj*lk` do MINI se anula e a base vira P1 puro
    (ver `interpolar_mini`), entao interpolar so nos dois vertices da
    aresta reproduz qualquer campo linear exatamente no ponto de
    cruzamento.

    Aqui vx = X em todo DOF (campo linear). O no 4 (centroide, em
    (2/3,1/3)) tem entao vx=2/3 e vy=1, e com dt=1/2 o pe cai em
    (1/3,-1/6) -- fora do dominio. A reta (2/3,1/3)->(1/3,-1/6) cruza
    a aresta de baixo (y=0) em x = 4/9, logo vx_star[4] tem que dar
    exatamente 4/9 (o valor do campo linear no ponto de cruzamento).
    """
    X0, Y0, IEN0 = quadrado_dois_triangulos()
    npoints, ne = 4, 2

    IEN, X, Y = elem_mini(IEN0, X0, Y0)
    loc = Localizador(IEN, len(X))

    vx = X.copy()  # campo linear vx = x, em todo DOF (base nodal)
    vy = np.zeros(npoints + ne)
    vy[4] = 1.0  # so o no 4 e' empurrado para fora, pela aresta de baixo

    vx_star, _ = calculo_sl(X, Y, loc, vx, vy, dt=0.5,
                             bc_dirichlet=sem_dirichlet(npoints + ne))

    assert np.isclose(vx_star[4], 4.0 / 9.0)


def _malha_tri6_quadrado():
    """Quadrado unitario dividido em 2 triangulos, ja estendido pro Tri6
    (nos de aresta deduplicados). Retorna tambem o `Localizador` que
    `calculo_sl` precisa e uma funcao `no(a, b)` que da o no de aresta entre os
    vertices `a` e `b`."""
    X0, Y0, IEN0 = quadrado_dois_triangulos()
    npoints = 4
    IEN, X, Y, arestas = elem_tri6(IEN0, X0, Y0)
    loc = Localizador(IEN, len(X))

    def no(a, b):
        return npoints + int(arestas.indice(a, b))

    return X, Y, IEN, loc, no


def test_calculo_sl_tri6_reproduz_campo_quadratico_interior():
    """Igual em espirito a `test_interpolar_tri6_reproduz_quadratico_arbitrario`,
    mas passando pelo pipeline inteiro de `calculo_sl` (backtrace +
    localizacao + dispatch pra `interpolar_tri6`) -- cobre o
    desempacotamento `i, j, k, eij, ejk, eki = IEN[elem].T` especifico
    do Tri6 dentro de `calculo_sl`, que o teste unitario de
    `interpolar_tri6` sozinho nao exercita. O MINI nao reproduziria um
    campo quadratico geral (so e' exato para linear); aqui, com o campo
    definido por um polinomio quadratico em todo DOF nodal, o Tri6 tem
    que recuperar o valor exato do polinomio no pe da caracteristica
    (que fica dentro do proprio elemento, dt pequeno).
    """
    X, Y, IEN, loc, no_aresta = _malha_tri6_quadrado()

    rng = np.random.default_rng(7)
    a1, b1, c1, d1, e1, f1 = rng.uniform(-1.0, 1.0, size=6)
    a2, b2, c2, d2, e2, f2 = rng.uniform(-1.0, 1.0, size=6)

    def poly1(x, y):
        return a1 + b1 * x + c1 * y + d1 * x**2 + e1 * x * y + f1 * y**2

    def poly2(x, y):
        return a2 + b2 * x + c2 * y + d2 * x**2 + e2 * x * y + f2 * y**2

    vx, vy = poly1(X, Y), poly2(X, Y)

    no = no_aresta(0, 2)  # no da diagonal -- interior ao quadrado, nao no contorno externo
    dt = 0.05
    xd, yd = X[no] - dt * vx[no], Y[no] - dt * vy[no]

    vx_star, vy_star = calculo_sl(
        X, Y, loc, vx, vy, dt, sem_dirichlet(len(vx)),
        elemento=TRI6)

    assert np.isclose(vx_star[no], poly1(xd, yd))
    assert np.isclose(vy_star[no], poly2(xd, yd))


def test_calculo_sl_tri6_intercept_reproduz_campo_quadratico_na_saida():
    """Analogo Tri6 de `test_calculo_sl_intercept_reproduz_campo_linear_na_saida`,
    mas discriminante: usa um campo genuinamente quadratico ao longo da
    aresta de saida (nao um que colapsa em linear), pra provar que a
    restricao quadratica de `_fallback_interceptacao` (decisao A,
    docs/semi_lagrangian_tri6.md) e' de fato usada -- se o codigo
    caisse de volta pra interpolacao linear so nos 2 vertices (o
    tratamento do MINI), o valor bateria com uma reta entre `vx[0]` e
    `vx[1]`, nao com o polinomio real no ponto de cruzamento.

    Fundo do quadrado (y=0, nos 0, 1 e o no de aresta (0,1)) recebe
    `h(x) = 2 - 3x + 5x^2`. O no da diagonal (0,2), em (0.5, 0.5), e'
    empurrado com `vx=0.3, vy=1.0` e `dt=0.8`: backtrace cai em
    (0.26, -0.3), fora do dominio, cruzando a aresta de baixo (reta
    vertical partindo de x=0.5 na direcao (0.3,1.0)) em x=0.35 -- logo
    o valor esperado e' `h(0.35)`.
    """
    X, Y, IEN, loc, no_aresta = _malha_tri6_quadrado()

    def h(x):
        return 2.0 - 3.0 * x + 5.0 * x**2

    no4 = no_aresta(0, 1)
    no6 = no_aresta(0, 2)

    vx = np.zeros(9)
    vy = np.zeros(9)
    vx[0], vx[1], vx[no4] = h(0.0), h(1.0), h(0.5)
    vx[no6], vy[no6] = 0.3, 1.0
    dt = 0.8

    vx_star, _ = calculo_sl(
        X, Y, loc, vx, vy, dt, sem_dirichlet(9),
        elemento=TRI6)

    assert np.isclose(vx_star[no6], h(0.35))
