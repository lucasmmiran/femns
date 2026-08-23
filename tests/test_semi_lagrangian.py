import numpy as np

from femns.mesh import elem_mini, elem_tri6, montar_edge_to_elem, montar_EToE, montar_node_to_elem
from femns.semi_lagrangian import (
    backtrace,
    baricentro,
    calculo_sl,
    interceptar_contorno,
    interpolar_mini,
    interpolar_tri6,
    localizar_pontos_partida,
    locate_point,
)


def quadrado_dois_triangulos():
    """Quadrado unitario dividido em 2 triangulos: 0=(0,0) 1=(1,0) 2=(1,1) 3=(0,1)."""
    X = np.array([0.0, 1.0, 1.0, 0.0])
    Y = np.array([0.0, 0.0, 1.0, 1.0])
    IEN = np.array([[0, 1, 2], [0, 2, 3]])
    return X, Y, IEN


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


def test_locate_point_mesmo_elemento():
    X, Y, IEN = quadrado_dois_triangulos()
    EToE, _ = montar_EToE(IEN)

    elem, lambdas = locate_point(0.5, 0.2, elem_start=0, X=X, Y=Y, IEN=IEN, EToE=EToE)

    assert elem == 0
    assert min(lambdas) >= 0.0


def test_locate_point_um_salto():
    X, Y, IEN = quadrado_dois_triangulos()
    EToE, _ = montar_EToE(IEN)

    # (0.2, 0.5) esta no elemento 1 (0,2,3), mas a busca comeca no elemento 0
    elem, lambdas = locate_point(0.2, 0.5, elem_start=0, X=X, Y=Y, IEN=IEN, EToE=EToE)

    assert elem == 1
    assert min(lambdas) >= 0.0
    assert np.allclose(lambdas, [0.5, 0.2, 0.3])


def test_locate_point_sai_do_dominio():
    X, Y, IEN = quadrado_dois_triangulos()
    EToE, _ = montar_EToE(IEN)

    elem, lambdas = locate_point(-0.5, -0.5, elem_start=0, X=X, Y=Y, IEN=IEN, EToE=EToE)

    assert elem is None
    assert lambdas is None


def test_baricentro_com_arrays():
    # A mesma funcao, sem nenhuma mudanca, deve aceitar arrays no lugar
    # de escalares (usado em lote por localizar_pontos_partida).
    X, Y, IEN = quadrado_dois_triangulos()

    xd = np.array([X[0], X[1], X[2]])
    yd = np.array([Y[0], Y[1], Y[2]])
    elem = np.array([0, 0, 0])

    li, lj, lk = baricentro(xd, yd, X, Y, IEN, elem)

    assert np.allclose(li, [1.0, 0.0, 0.0])
    assert np.allclose(lj, [0.0, 1.0, 0.0])
    assert np.allclose(lk, [0.0, 0.0, 1.0])


def test_localizar_pontos_partida_saltos_diferentes_na_mesma_chamada():
    """Regressao da vetorizacao: 3 pontos que precisam de 0 saltos, 1
    salto e saida de dominio, todos na MESMA chamada, comecando do
    mesmo elemento -- valida que a mascara nao mistura resultados
    entre linhas (risco real de bug ao vetorizar por mascara).
    Os valores esperados sao os mesmos ja validados individualmente em
    test_locate_point_mesmo_elemento/um_salto/sai_do_dominio.
    """
    X, Y, IEN = quadrado_dois_triangulos()
    EToE, _ = montar_EToE(IEN)

    xd = np.array([0.5, 0.2, -0.5])
    yd = np.array([0.2, 0.5, -0.5])
    elem_start = np.array([0, 0, 0])

    elem, lambdas, saida = localizar_pontos_partida(xd, yd, elem_start, X, Y, IEN, EToE)

    assert elem[0] == 0  # 0 saltos
    assert min(lambdas[0]) >= 0.0

    assert elem[1] == 1  # 1 salto
    assert np.allclose(lambdas[1], [0.5, 0.2, 0.3])

    assert elem[2] == -1  # saiu do dominio

    # Quem foi localizado nao tem face de saida; quem saiu tem a face registrada
    assert (saida[0] == -1).all() and (saida[1] == -1).all()
    assert saida[2, 0] != -1 and saida[2, 1] != -1


def test_localizar_pontos_partida_registra_aresta_de_saida():
    """A face de saida guardada tem que ser a aresta de contorno que a
    caminhada de fato tentou atravessar -- e' ela que o tratamento por
    interceptacao usa para interpolar. Aqui o pe cai reto abaixo da aresta
    de baixo do elemento 0 (nos 0-1), entao a saida tem que ser
    [elem=0, face=0] (convencao de montar_EToE: face f liga os vertices
    locais f e (f+1)%3).
    """
    X, Y, IEN = quadrado_dois_triangulos()
    EToE, _ = montar_EToE(IEN)

    elem, _, saida = localizar_pontos_partida(
        np.array([0.5]), np.array([-0.5]), np.array([0]), X, Y, IEN, EToE)

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
    # li levemente negativo (tolerancia do locate_point) e recortado pra 0
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
    igual a qualquer elemento P1 comum. Com isso, a interpolacao deve
    recuperar exatamente (c,d) em qualquer ponto interno, nao importa
    onde o pe da caracteristica caia.

    Fixa `fora_dominio="dirichlet"` de proposito: este teste existe para
    cobrir os 3 caminhos *desse* fallback, entao nao deve seguir o padrao
    do modulo (hoje `intercept`) -- se seguisse, deixaria de testar o que
    se propoe no dia em que o padrao mudar (foi o que aconteceu).
    """
    X0, Y0, IEN0 = quadrado_dois_triangulos()
    npoints, ne = 4, 2

    EToE, _ = montar_EToE(IEN0)
    node_to_elem = montar_node_to_elem(IEN0, npoints)
    IEN, X, Y = elem_mini(IEN0, X0, Y0)

    c, d, dt = -0.05, -0.02, 1.0

    vx = np.array([c, c, c, c, c, c])
    vy = np.array([d, d, d, d, d, d])

    # Nomes de contorno arbitrarios so pra exercitar os 3 caminhos do
    # fallback: valor de Dirichlet explicito, contorno sem esse
    # componente definido, e contorno sem nenhuma condicao no dict.
    ccName = ["origem", "direita", "topo", "esquerda"]
    conditions = {
        "direita": {"vx": 99.0, "vy": 99.0},
        "topo": {"p": 0.0},
    }

    vx_star, vy_star = calculo_sl(
        X, Y, IEN, EToE, node_to_elem, ccName, conditions, vx, vy, dt, npoints, ne,
        fora_dominio="dirichlet")

    # No 0: pe da caracteristica cai dentro do elemento 0 -> interpolado, recupera (c,d)
    assert np.isclose(vx_star[0], c) and np.isclose(vy_star[0], d)
    # No 1: sai do dominio por um contorno com Dirichlet explicito -> usa esse valor
    assert np.isclose(vx_star[1], 99.0) and np.isclose(vy_star[1], 99.0)
    # Nos 2 e 3: saem do dominio por contornos sem vx/vy definido -> fallback pro proprio no
    assert np.isclose(vx_star[2], c) and np.isclose(vy_star[2], d)
    assert np.isclose(vx_star[3], c) and np.isclose(vy_star[3], d)
    # Nos de centroide (4, 5): campo constante em todo DOF -> interpolar_mini
    # recupera (c,d) exatamente em qualquer ponto do elemento (propriedade de
    # reproducao de constante da base nodal), nao importa onde o pe da
    # caracteristica caia dentro do proprio elemento.
    assert np.isclose(vx_star[4], c) and np.isclose(vy_star[4], d)
    assert np.isclose(vx_star[5], c) and np.isclose(vy_star[5], d)


def test_calculo_sl_centroide_e_nodal():
    """Regressao: o DOF de centroide/bolha e nodal (propriedade de
    Lagrange) -- a velocidade fisica no centroide E vx[centroide]
    diretamente, nao `(vi+vj+vk)/3 + vb` (formula errada descartada
    depois de bater a matriz de rigidez de assembly.py, entrada a
    entrada, contra a base nodal `Ni = li - 9*li*lj*lk`; ver
    `interpolar_mini`). Com dt desprezivel, v_star no no de centroide
    deve ficar bem perto do proprio vx[n] (nao da media dos vertices
    mais vx[n]).
    """
    X0, Y0, IEN0 = quadrado_dois_triangulos()
    npoints, ne = 4, 2

    EToE, _ = montar_EToE(IEN0)
    node_to_elem = montar_node_to_elem(IEN0, npoints)
    IEN, X, Y = elem_mini(IEN0, X0, Y0)

    vx = np.array([1.0, 2.0, 3.0, 4.0, 5.0, -1.0])
    vy = np.zeros(6)
    dt = 1e-6  # deslocamento desprezivel: v_star deve ficar bem perto da velocidade fisica atual

    vx_star, _ = calculo_sl(
        X, Y, IEN, EToE, node_to_elem, ["a", "b", "c", "d"], {}, vx, vy, dt, npoints, ne)

    # No de centroide do elemento 0 e' o no 4 (vb=5.0) -- deve recuperar isso diretamente
    assert np.isclose(vx_star[4], 5.0, atol=1e-4)


def test_calculo_sl_intercept_interpola_na_aresta_de_saida():
    """Discrimina os dois tratamentos de "fora do dominio" num no onde eles
    dao respostas diferentes, com o valor esperado calculado a mao.

    O no 4 e' o centroide do elemento 0, em (2/3, 1/3). Com vx=0, vy=1 e
    dt=0.5 o pe da caracteristica cai em (2/3, -1/6) -- fora do dominio,
    reto abaixo da aresta de baixo (nos 0-1). O cruzamento e' em (2/3, 0),
    isto e' 1/3 do no 0 e 2/3 do no 1, entao o modo `intercept` tem que
    dar 1/3*vx[0] + 2/3*vx[1] = 1/3*10 + 2/3*4 = 6.0.

    Sem nenhuma condicao de contorno no dict, o modo `dirichlet` nao tem
    valor prescrito para cair e mantem o valor do proprio no (0.0) --
    "congela" a adveccao nesse no, que e' exatamente a diferenca de
    comportamento entre os dois.
    """
    X0, Y0, IEN0 = quadrado_dois_triangulos()
    npoints, ne = 4, 2

    EToE, _ = montar_EToE(IEN0)
    node_to_elem = montar_node_to_elem(IEN0, npoints)
    IEN, X, Y = elem_mini(IEN0, X0, Y0)

    vx = np.array([10.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    vy = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    dt = 0.5
    ccName = [None, None, None, None]

    vx_int, _ = calculo_sl(X, Y, IEN, EToE, node_to_elem, ccName, {}, vx, vy, dt,
                            npoints, ne, fora_dominio="intercept")
    vx_dir, _ = calculo_sl(X, Y, IEN, EToE, node_to_elem, ccName, {}, vx, vy, dt,
                            npoints, ne, fora_dominio="dirichlet")

    assert np.isclose(vx_int[4], 6.0)
    assert np.isclose(vx_dir[4], 0.0)


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

    EToE, _ = montar_EToE(IEN0)
    node_to_elem = montar_node_to_elem(IEN0, npoints)
    IEN, X, Y = elem_mini(IEN0, X0, Y0)

    vx = X.copy()  # campo linear vx = x, em todo DOF (base nodal)
    vy = np.zeros(npoints + ne)
    vy[4] = 1.0  # so o no 4 e' empurrado para fora, pela aresta de baixo

    vx_star, _ = calculo_sl(X, Y, IEN, EToE, node_to_elem, [None] * npoints, {},
                             vx, vy, dt=0.5, npoints=npoints, ne=ne, fora_dominio="intercept")

    assert np.isclose(vx_star[4], 4.0 / 9.0)


def _malha_tri6_quadrado():
    """Quadrado unitario dividido em 2 triangulos, ja estendido pro Tri6
    (nos de aresta deduplicados). Retorna tambem os elementos que
    `calculo_sl` precisa para o elemento='tri6' (EToE, node_to_elem,
    elem_start_extra) e o dict `edge_para_no` para os testes acharem os
    nos de aresta que precisam pelo nome (vertices que a aresta liga)."""
    X0, Y0, IEN0 = quadrado_dois_triangulos()
    npoints = 4
    IEN, X, Y, edge_para_no = elem_tri6(IEN0, X0, Y0)
    EToE, _ = montar_EToE(IEN0)
    node_to_elem = montar_node_to_elem(IEN0, npoints)
    elem_start_extra = montar_edge_to_elem(IEN, npoints, len(edge_para_no))
    return X, Y, IEN, EToE, node_to_elem, elem_start_extra, edge_para_no, npoints


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
    X, Y, IEN, EToE, node_to_elem, elem_start_extra, edge_para_no, npoints = _malha_tri6_quadrado()

    rng = np.random.default_rng(7)
    a1, b1, c1, d1, e1, f1 = rng.uniform(-1.0, 1.0, size=6)
    a2, b2, c2, d2, e2, f2 = rng.uniform(-1.0, 1.0, size=6)

    def poly1(x, y):
        return a1 + b1 * x + c1 * y + d1 * x**2 + e1 * x * y + f1 * y**2

    def poly2(x, y):
        return a2 + b2 * x + c2 * y + d2 * x**2 + e2 * x * y + f2 * y**2

    vx, vy = poly1(X, Y), poly2(X, Y)

    no = edge_para_no[(0, 2)]  # no da diagonal -- interior ao quadrado, nao no contorno externo
    dt = 0.05
    xd, yd = X[no] - dt * vx[no], Y[no] - dt * vy[no]

    vx_star, vy_star = calculo_sl(
        X, Y, IEN, EToE, node_to_elem, [None] * npoints, {}, vx, vy, dt, npoints, ne=2,
        elemento="tri6", elem_start_extra=elem_start_extra)

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
    X, Y, IEN, EToE, node_to_elem, elem_start_extra, edge_para_no, npoints = _malha_tri6_quadrado()

    def h(x):
        return 2.0 - 3.0 * x + 5.0 * x**2

    no4 = edge_para_no[(0, 1)]
    no6 = edge_para_no[(0, 2)]

    vx = np.zeros(9)
    vy = np.zeros(9)
    vx[0], vx[1], vx[no4] = h(0.0), h(1.0), h(0.5)
    vx[no6], vy[no6] = 0.3, 1.0
    dt = 0.8

    vx_star, _ = calculo_sl(
        X, Y, IEN, EToE, node_to_elem, [None] * npoints, {}, vx, vy, dt, npoints, ne=2,
        elemento="tri6", elem_start_extra=elem_start_extra, fora_dominio="intercept")

    assert np.isclose(vx_star[no6], h(0.35))

    # modo dirichlet, sem nenhuma condicao no dict: sem valor prescrito
    # pra cair, mantem o proprio no (mesma diferenca de comportamento
    # que o teste MINI equivalente demonstra).
    vx_dir, _ = calculo_sl(
        X, Y, IEN, EToE, node_to_elem, [None] * npoints, {}, vx, vy, dt, npoints, ne=2,
        elemento="tri6", elem_start_extra=elem_start_extra, fora_dominio="dirichlet")

    assert np.isclose(vx_dir[no6], 0.3)


def test_calculo_sl_tri6_requer_elem_start_extra():
    """O no de aresta do Tri6 nao tem uma semente generica por indice
    como o centroide do MINI (`np.arange(ne)`) -- omitir
    `elem_start_extra` para `elemento='tri6'` tem que falhar cedo e com
    mensagem clara, nao silenciosamente usar uma semente errada."""
    X, Y, IEN, EToE, node_to_elem, _, _, npoints = _malha_tri6_quadrado()

    try:
        calculo_sl(X, Y, IEN, EToE, node_to_elem, [None] * npoints, {},
                   np.zeros(9), np.zeros(9), 0.1, npoints, ne=2, elemento="tri6")
    except ValueError as e:
        assert "elem_start_extra" in str(e)
    else:
        raise AssertionError("deveria ter levantado ValueError")


def test_calculo_sl_fora_dominio_invalido():
    X0, Y0, IEN0 = quadrado_dois_triangulos()
    EToE, _ = montar_EToE(IEN0)
    node_to_elem = montar_node_to_elem(IEN0, 4)
    IEN, X, Y = elem_mini(IEN0, X0, Y0)

    try:
        calculo_sl(X, Y, IEN, EToE, node_to_elem, [None] * 4, {},
                    np.zeros(6), np.zeros(6), 0.1, 4, 2, fora_dominio="qualquer")
    except ValueError as e:
        assert "fora_dominio" in str(e)
    else:
        raise AssertionError("deveria ter levantado ValueError")
