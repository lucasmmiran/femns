import numpy as np

from femns.mesh import elem_mini, montar_EToE, montar_node_to_elem
from femns.semi_lagrangian import backtrace_and_interpolate, backtrace_euler, barycentric, interpolate_mini, locate_point


def quadrado_dois_triangulos():
    """Quadrado unitario dividido em 2 triangulos: 0=(0,0) 1=(1,0) 2=(1,1) 3=(0,1)."""
    X = np.array([0.0, 1.0, 1.0, 0.0])
    Y = np.array([0.0, 0.0, 1.0, 1.0])
    IEN = np.array([[0, 1, 2], [0, 2, 3]])
    return X, Y, IEN


def test_barycentric_nos_vertices():
    X, Y, IEN = quadrado_dois_triangulos()

    li, lj, lk = barycentric(X[0], Y[0], X, Y, IEN, elem=0)
    assert np.allclose([li, lj, lk], [1.0, 0.0, 0.0])

    li, lj, lk = barycentric(X[1], Y[1], X, Y, IEN, elem=0)
    assert np.allclose([li, lj, lk], [0.0, 1.0, 0.0])

    li, lj, lk = barycentric(X[2], Y[2], X, Y, IEN, elem=0)
    assert np.allclose([li, lj, lk], [0.0, 0.0, 1.0])


def test_barycentric_centroide():
    X, Y, IEN = quadrado_dois_triangulos()
    v0, v1, v2 = IEN[0]
    xc = (X[v0] + X[v1] + X[v2]) / 3.0
    yc = (Y[v0] + Y[v1] + Y[v2]) / 3.0

    li, lj, lk = barycentric(xc, yc, X, Y, IEN, elem=0)
    assert np.allclose([li, lj, lk], [1 / 3, 1 / 3, 1 / 3])


def test_barycentric_ponto_fora():
    X, Y, IEN = quadrado_dois_triangulos()
    # (0,1) e o no 3, do outro lado da diagonal do elemento 0 (0,1,2)
    li, lj, lk = barycentric(0.0, 1.0, X, Y, IEN, elem=0)
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


def test_interpolate_mini_centroide():
    # No centroide (li=lj=lk=1/3), a bolha vale 1: v_star = media(vi,vj,vk) + vb
    v_star = interpolate_mini(1 / 3, 1 / 3, 1 / 3, vi=2.0, vj=4.0, vk=9.0, vb=10.0)
    assert np.isclose(v_star, (2.0 + 4.0 + 9.0) / 3.0 + 10.0)


def test_interpolate_mini_valores_arbitrarios():
    # Conferido a mao: 0.5*1 + 0.3*2 + 0.2*3 + 27*0.5*0.3*0.2*10 = 9.8
    v_star = interpolate_mini(0.5, 0.3, 0.2, vi=1.0, vj=2.0, vk=3.0, vb=10.0)
    assert np.isclose(v_star, 9.8)


def test_interpolate_mini_clampa_lambda_negativo():
    # li levemente negativo (tolerancia do locate_point) e recortado pra 0
    # antes de renormalizar -- nao deve gerar peso negativo no resultado.
    v_star = interpolate_mini(-1e-10, 0.5, 0.5, vi=100.0, vj=1.0, vk=1.0, vb=0.0)
    assert np.isclose(v_star, 1.0)


def test_backtrace_euler_translacao():
    X, Y, IEN = quadrado_dois_triangulos()
    vx_node = np.full_like(X, 2.0)
    vy_node = np.full_like(Y, -1.0)
    dt = 0.1

    xd, yd = backtrace_euler(X, Y, vx_node, vy_node, dt)

    assert np.allclose(xd, X - 0.2)
    assert np.allclose(yd, Y + 0.1)


def test_backtrace_and_interpolate_campo_constante():
    """Campo MINI que representa a constante (c, d): vertices = (c,d), bolha = 0.

    A bolha precisa ser 0 (nao (c,d)) para representar uma constante de
    verdade no espaco MINI -- ver Passo 3 do docs/semi_lagrangian_strategy.pdf.
    Com isso, a interpolacao deve recuperar exatamente (c,d) em qualquer
    ponto interno, nao importa onde o pe da caracteristica caia.
    """
    X0, Y0, IEN0 = quadrado_dois_triangulos()
    npoints, ne = 4, 2

    EToE, _ = montar_EToE(IEN0)
    node_to_elem = montar_node_to_elem(IEN0, npoints)
    IEN, X, Y = elem_mini(IEN0, X0, Y0)

    c, d, dt = -0.05, -0.02, 1.0

    vx = np.array([c, c, c, c, 0.0, 0.0])
    vy = np.array([d, d, d, d, 0.0, 0.0])

    # Nomes de contorno arbitrarios so pra exercitar os 3 caminhos do
    # fallback: valor de Dirichlet explicito, contorno sem esse
    # componente definido, e contorno sem nenhuma condicao no dict.
    ccName = ["origem", "direita", "topo", "esquerda"]
    conditions = {
        "direita": {"vx": 99.0, "vy": 99.0},
        "topo": {"p": 0.0},
    }

    vx_star, vy_star = backtrace_and_interpolate(
        X, Y, IEN, EToE, node_to_elem, ccName, conditions, vx, vy, dt, npoints, ne)

    # No 0: pe da caracteristica cai dentro do elemento 0 -> interpolado, recupera (c,d)
    assert np.isclose(vx_star[0], c) and np.isclose(vy_star[0], d)
    # No 1: sai do dominio por um contorno com Dirichlet explicito -> usa esse valor
    assert np.isclose(vx_star[1], 99.0) and np.isclose(vy_star[1], 99.0)
    # Nos 2 e 3: saem do dominio por contornos sem vx/vy definido -> fallback pro proprio no
    assert np.isclose(vx_star[2], c) and np.isclose(vy_star[2], d)
    assert np.isclose(vx_star[3], c) and np.isclose(vy_star[3], d)
    # Nos de centroide (4, 5): bolha=0 nao se move (backtrace usa a propria velocidade
    # da bolha), fica no proprio elemento -> interpolado, recupera (c,d)
    assert np.isclose(vx_star[4], c) and np.isclose(vy_star[4], d)
    assert np.isclose(vx_star[5], c) and np.isclose(vy_star[5], d)
