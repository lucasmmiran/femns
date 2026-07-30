import numpy as np

from femns.mesh import elem_mini, montar_EToE, montar_node_to_elem
from femns.semi_lagrangian import (
    backtrace,
    baricentro,
    calculo_sl,
    interpolar_mini,
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

    elem, lambdas = localizar_pontos_partida(xd, yd, elem_start, X, Y, IEN, EToE)

    assert elem[0] == 0  # 0 saltos
    assert min(lambdas[0]) >= 0.0

    assert elem[1] == 1  # 1 salto
    assert np.allclose(lambdas[1], [0.5, 0.2, 0.3])

    assert elem[2] == -1  # saiu do dominio


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
        X, Y, IEN, EToE, node_to_elem, ccName, conditions, vx, vy, dt, npoints, ne)

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
