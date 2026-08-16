import numpy as np
import torch

from femns.mesh import atualiza_nos_extras, elem_mini, elem_tri6, montar_NToN
from femns.moving_mesh import (
    aplicar_oscilacao,
    distancia_min_vizinhos,
    passos_por_periodo,
    preparar_oscilacao,
    velocidade_malha,
)


def malha_5_nos():
    """Quadrado unitario com um no no centro: 4 vertices de contorno + no 4 interior.

    0=(0,0) 1=(1,0) 2=(1,1) 3=(0,1) 4=(0.5,0.5), 4 triangulos em volta do centro.
    O no 4 e' o unico interior -- e o unico que deve oscilar.
    """
    X = np.array([0.0, 1.0, 1.0, 0.0, 0.5])
    Y = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
    IEN = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    return X, Y, IEN


def test_distancia_min_vizinhos():
    X, Y, IEN = malha_5_nos()
    NToN = montar_NToN(IEN, len(X))

    # O no 4 (centro) esta a sqrt(0.5)/... de cada canto: dist = sqrt(0.5^2+0.5^2)
    h = distancia_min_vizinhos(X, Y, NToN, np.array([4]))
    assert np.isclose(h[0], np.hypot(0.5, 0.5))

    # O no 0 tem vizinhos 1 (dist 1), 3 (dist 1) e 4 (dist sqrt(0.5)) -> o mais proximo e o 4
    h0 = distancia_min_vizinhos(X, Y, NToN, np.array([0]))
    assert np.isclose(h0[0], np.hypot(0.5, 0.5))


def test_oscilacao_respeita_amplitude_e_nao_move_contorno():
    """O deslocamento de cada no oscilante nunca passa de `fator * h local`, e
    nenhum no fora da lista (contorno) se move -- em nenhum instante.
    """
    X0, Y0, IEN = malha_5_nos()
    NToN = montar_NToN(IEN, len(X0))
    nos = np.array([4])  # so o interior
    fator = 0.3

    osc = preparar_oscilacao(X0, Y0, NToN, nos, fator=fator, omega=2.0, seed=42)
    h = distancia_min_vizinhos(X0, Y0, NToN, nos)

    X = torch.from_numpy(X0.copy())
    Y = torch.from_numpy(Y0.copy())

    for t in np.linspace(0.0, 10.0, 200):
        X, Y = aplicar_oscilacao(osc, X, Y, t)
        desloc = np.hypot(X[4].item() - X0[4], Y[4].item() - Y0[4])
        assert desloc <= fator * h[0] + 1e-12
        # nos de contorno intactos
        assert np.allclose(X.numpy()[:4], X0[:4])
        assert np.allclose(Y.numpy()[:4], Y0[:4])


def test_oscilacao_atinge_a_amplitude_maxima():
    """Regressao: a oscilacao tem que de fato *usar* a amplitude, nao ficar
    perto de zero. Ao longo de um periodo inteiro o deslocamento maximo
    deve encostar em `fator * h` (o seno passa por +-1).
    """
    X0, Y0, IEN = malha_5_nos()
    NToN = montar_NToN(IEN, len(X0))
    nos = np.array([4])
    fator, omega = 0.3, 2.0

    osc = preparar_oscilacao(X0, Y0, NToN, nos, fator=fator, omega=omega, seed=1)
    h = distancia_min_vizinhos(X0, Y0, NToN, nos)[0]

    X = torch.from_numpy(X0.copy())
    Y = torch.from_numpy(Y0.copy())

    maximo = 0.0
    for t in np.linspace(0.0, 2 * np.pi / omega, 500):  # um periodo
        X, Y = aplicar_oscilacao(osc, X, Y, t)
        maximo = max(maximo, np.hypot(X[4].item() - X0[4], Y[4].item() - Y0[4]))

    assert np.isclose(maximo, fator * h, rtol=1e-3)


def test_oscilacao_reprodutivel_pela_semente():
    X0, Y0, IEN = malha_5_nos()
    NToN = montar_NToN(IEN, len(X0))
    nos = np.array([4])

    a = preparar_oscilacao(X0, Y0, NToN, nos, seed=7)
    b = preparar_oscilacao(X0, Y0, NToN, nos, seed=7)
    c = preparar_oscilacao(X0, Y0, NToN, nos, seed=8)

    assert np.allclose(a.fase, b.fase) and np.allclose(a.dir_x, b.dir_x)
    assert not np.allclose(a.fase, c.fase)


def test_direcao_e_versor():
    """dir_x/dir_y tem que ter norma 1, senao a amplitude efetiva nao seria
    `fator * h` (o deslocamento e' amplitude * seno * versor).
    """
    X0, Y0, IEN = malha_5_nos()
    NToN = montar_NToN(IEN, len(X0))

    osc = preparar_oscilacao(X0, Y0, NToN, np.array([4]), seed=3)

    assert np.allclose(np.hypot(osc.dir_x, osc.dir_y), 1.0)


def test_atualiza_nos_extras_mini_recoloca_centroide():
    """Depois de mover um vertice, o no de centroide do MINI tem que voltar a
    ser a media dos 3 vertices -- se nao, ele sai de dentro do triangulo e
    deixa de ser o DOF que `assembly.assemble_mini` assume.
    """
    X0, Y0, IEN0 = malha_5_nos()
    IEN, X, Y = elem_mini(IEN0, X0, Y0)

    X[4] += 0.1  # move o no interior (vertice)
    Y[4] -= 0.05
    X, Y = atualiza_nos_extras(IEN, X, Y)

    for e in range(IEN.shape[0]):
        v1, v2, v3, c = IEN[e]
        assert np.isclose(X[c], (X[v1] + X[v2] + X[v3]) / 3.0)
        assert np.isclose(Y[c], (Y[v1] + Y[v2] + Y[v3]) / 3.0)


def test_atualiza_nos_extras_tri6_recoloca_pontos_medios():
    X0, Y0, IEN0 = malha_5_nos()
    IEN, X, Y, _ = elem_tri6(IEN0, X0, Y0)

    X[4] += 0.1
    Y[4] -= 0.05
    X, Y = atualiza_nos_extras(IEN, X, Y)

    for e in range(IEN.shape[0]):
        v1, v2, v3, a12, a23, a31 = IEN[e]
        assert np.isclose(X[a12], (X[v1] + X[v2]) / 2.0)
        assert np.isclose(X[a23], (X[v2] + X[v3]) / 2.0)
        assert np.isclose(X[a31], (X[v3] + X[v1]) / 2.0)
        assert np.isclose(Y[a12], (Y[v1] + Y[v2]) / 2.0)


def test_velocidade_malha_bate_com_diferenca_finita():
    """`velocidade_malha` e' a derivada analitica de `aplicar_oscilacao`;
    tem que bater com a derivada numerica das posicoes.
    """
    X0, Y0, IEN0 = malha_5_nos()
    NToN = montar_NToN(IEN0, len(X0))
    IEN, Xm, Ym = elem_mini(IEN0, X0, Y0)
    nnodes = len(Xm)

    osc = preparar_oscilacao(Xm, Ym, NToN, np.array([4]), fator=0.3, omega=3.0, seed=5)

    t, h = 0.37, 1e-7
    Xa, Ya = aplicar_oscilacao(osc, torch.from_numpy(Xm.copy()), torch.from_numpy(Ym.copy()), t - h)
    Xa, Ya = atualiza_nos_extras(IEN, Xa, Ya)
    Xb, Yb = aplicar_oscilacao(osc, torch.from_numpy(Xm.copy()), torch.from_numpy(Ym.copy()), t + h)
    Xb, Yb = atualiza_nos_extras(IEN, Xb, Yb)

    wx, wy = velocidade_malha(osc, IEN, t, nnodes)

    assert np.allclose(wx.numpy(), (Xb - Xa).numpy() / (2 * h), atol=1e-6)
    assert np.allclose(wy.numpy(), (Yb - Ya).numpy() / (2 * h), atol=1e-6)


def test_velocidade_malha_zera_no_contorno_e_propaga_ao_centroide():
    """Nos de contorno nao se movem (w=0); o centroide do MINI herda a media
    das velocidades dos 3 vertices (mesma combinacao linear que define a
    posicao dele).
    """
    X0, Y0, IEN0 = malha_5_nos()
    NToN = montar_NToN(IEN0, len(X0))
    IEN, Xm, Ym = elem_mini(IEN0, X0, Y0)

    osc = preparar_oscilacao(Xm, Ym, NToN, np.array([4]), omega=3.0, seed=2)
    wx, wy = velocidade_malha(osc, IEN, t=0.11, nnodes=len(Xm))

    assert np.allclose(wx.numpy()[:4], 0.0) and np.allclose(wy.numpy()[:4], 0.0)
    assert not np.isclose(wx[4].item(), 0.0) or not np.isclose(wy[4].item(), 0.0)

    for e in range(IEN.shape[0]):
        v1, v2, v3, c = IEN[e]
        assert np.isclose(wx[c].item(), (wx[v1] + wx[v2] + wx[v3]).item() / 3.0)
        assert np.isclose(wy[c].item(), (wy[v1] + wy[v2] + wy[v3]).item() / 3.0)


def test_passos_por_periodo_detecta_o_caso_congelado():
    """Regressao do achado: omega = pi*200 com dt = 0.01 da exatamente 1 passo
    por periodo -- a malha e' amostrada sempre na mesma fase e fica parada.
    """
    assert np.isclose(passos_por_periodo(np.pi * 200, 0.01), 1.0)

    # amostrada sempre na mesma fase -> deslocamento identico em todo passo
    X0, Y0, IEN0 = malha_5_nos()
    NToN = montar_NToN(IEN0, len(X0))
    osc = preparar_oscilacao(X0, Y0, NToN, np.array([4]), omega=np.pi * 200, seed=0)

    X = torch.from_numpy(X0.copy())
    Y = torch.from_numpy(Y0.copy())
    posicoes = []
    for n in range(5):
        X, Y = aplicar_oscilacao(osc, X, Y, n * 0.01)
        posicoes.append((X[4].item(), Y[4].item()))

    assert all(np.allclose(p, posicoes[0]) for p in posicoes)

    # com omega 20x menor a malha de fato anda entre passos
    osc2 = preparar_oscilacao(X0, Y0, NToN, np.array([4]), omega=np.pi * 10, seed=0)
    assert np.isclose(passos_por_periodo(np.pi * 10, 0.01), 20.0)
    X, Y = aplicar_oscilacao(osc2, X, Y, 0.0)
    p0 = (X[4].item(), Y[4].item())
    X, Y = aplicar_oscilacao(osc2, X, Y, 0.01)
    assert not np.allclose((X[4].item(), Y[4].item()), p0)


def test_atualiza_nos_extras_rejeita_ien_desconhecida():
    X0, Y0, IEN0 = malha_5_nos()
    try:
        atualiza_nos_extras(IEN0, X0, Y0)  # 3 colunas: so vertices, sem no extra
    except ValueError as e:
        assert "3 colunas" in str(e)
    else:
        raise AssertionError("deveria ter levantado ValueError")
