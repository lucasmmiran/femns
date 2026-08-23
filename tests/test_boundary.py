import numpy as np
import torch

from femns.boundary import assign_boundary_names, build_boundary_conditions, perfil_parabolico


def quadrado_com_contornos():
    """Quadrado com 4 nos, uma aresta de contorno por lado."""
    IENbound = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    IENboundElem = ["bottom", "outlet", "top", "inlet"]
    return IENbound, IENboundElem


def test_assign_boundary_names_prioridade():
    IENbound, IENboundElem = quadrado_com_contornos()
    # 'bottom' tem a prioridade mais alta (ultimo da lista) e vence nas quinas
    priority = ["outlet", "inlet", "top", "bottom"]

    ccName = assign_boundary_names(IENbound, IENboundElem, npoints=4, priority=priority)

    assert ccName == ["bottom", "bottom", "top", "top"]


def test_build_boundary_conditions():
    IENbound, IENboundElem = quadrado_com_contornos()
    priority = ["outlet", "inlet", "top", "bottom"]
    ccName = assign_boundary_names(IENbound, IENboundElem, npoints=4, priority=priority)

    conditions = {
        "bottom": {"vx": 0.0, "vy": 0.0},
        "top": {"vx": 1.0, "vy": 0.0},
    }

    device = torch.device("cpu")
    vx_cc, vy_cc, p_cc, vx_cc_pts, vy_cc_pts, p_cc_pts = build_boundary_conditions(
        IENbound, ccName, conditions, npoints=4, device=device)

    assert sorted(vx_cc_pts.tolist()) == [0, 1, 2, 3]
    assert sorted(vy_cc_pts.tolist()) == [0, 1, 2, 3]
    assert p_cc_pts.numel() == 0

    assert vx_cc[0] == 0.0 and vx_cc[1] == 0.0
    assert vx_cc[2] == 1.0 and vx_cc[3] == 1.0
    assert torch.all(vy_cc == 0.0)


def test_assign_boundary_names_com_contorno_de_ponto_unico():
    """Cavidade tampada (lid.msh): 'outlet' e um unico no de referencia de pressao, nao uma aresta."""
    IENbound, IENboundElem = quadrado_com_contornos()
    IENboundElem = ["bottom", "wall", "top", "wall"]  # sem 'outlet' em nenhuma aresta
    priority = ["wall", "top", "bottom", "outlet"]
    IENpoint = np.array([2])
    IENpointElem = ["outlet"]

    ccName = assign_boundary_names(IENbound, IENboundElem, npoints=4, priority=priority,
                                    IENpoint=IENpoint, IENpointElem=IENpointElem)

    assert ccName == ["bottom", "bottom", "outlet", "top"]


def test_assign_e_build_boundary_conditions_com_no_de_aresta_tri6():
    """IENbound de 3 colunas (v1, no_aresta, v2), como o Tri6 produz via
    `mesh.estende_IENbound_tri6` -- a condicao de contorno de velocidade
    deve valer tambem sobre o no de aresta, nao so sobre os vertices."""
    IENbound = np.array([[0, 4, 1], [1, 5, 2], [2, 6, 3], [3, 7, 0]])
    IENboundElem = ["bottom", "outlet", "top", "inlet"]
    priority = ["outlet", "inlet", "top", "bottom"]
    nnodes = 8  # 4 vertices + 4 nos de aresta

    ccName = assign_boundary_names(IENbound, IENboundElem, npoints=4, priority=priority, nnodes=nnodes)

    assert len(ccName) == nnodes
    assert ccName[4] == "bottom"  # no de aresta entre 0 e 1
    assert ccName[6] == "top"     # no de aresta entre 2 e 3

    conditions = {
        "bottom": {"vx": 0.0, "vy": 0.0},
        "top": {"vx": 1.0, "vy": 0.0},
    }
    device = torch.device("cpu")
    vx_cc, vy_cc, p_cc, vx_cc_pts, vy_cc_pts, p_cc_pts = build_boundary_conditions(
        IENbound, ccName, conditions, npoints=4, device=device, nnodes=nnodes)

    assert vx_cc.numel() == nnodes
    assert vy_cc.numel() == nnodes
    assert p_cc.numel() == 4  # pressao so nos vertices, independente de nnodes

    assert sorted(vx_cc_pts.tolist()) == [0, 1, 2, 3, 4, 6]
    assert vx_cc[4] == 0.0  # no de aresta do bottom
    assert vx_cc[6] == 1.0  # no de aresta do top


def test_build_boundary_conditions_ignora_no_de_aresta_para_pressao():
    """Pressao e P1 (so vertice) -- um contorno de pressao que tambem cobre
    nos de aresta (Tri6) nao pode tentar indexar p_cc (tamanho npoints) fora
    dos limites."""
    IENbound = np.array([[0, 4, 1], [1, 5, 2], [2, 6, 3], [3, 7, 0]])
    IENboundElem = ["outlet", "outlet", "outlet", "outlet"]
    priority = ["outlet"]
    nnodes = 8

    ccName = assign_boundary_names(IENbound, IENboundElem, npoints=4, priority=priority, nnodes=nnodes)

    conditions = {"outlet": {"p": 0.0}}
    device = torch.device("cpu")
    _, _, p_cc, _, _, p_cc_pts = build_boundary_conditions(
        IENbound, ccName, conditions, npoints=4, device=device, nnodes=nnodes)

    assert p_cc.numel() == 4
    assert sorted(p_cc_pts.tolist()) == [0, 1, 2, 3]  # nunca inclui os nos de aresta (4..7)


def test_build_boundary_conditions_com_contorno_de_ponto_unico():
    IENbound, _ = quadrado_com_contornos()
    IENboundElem = ["bottom", "wall", "top", "wall"]
    priority = ["wall", "top", "bottom", "outlet"]
    IENpoint = np.array([2])
    IENpointElem = ["outlet"]

    ccName = assign_boundary_names(IENbound, IENboundElem, npoints=4, priority=priority,
                                    IENpoint=IENpoint, IENpointElem=IENpointElem)

    conditions = {"outlet": {"p": 0.0}}
    device = torch.device("cpu")
    vx_cc, vy_cc, p_cc, vx_cc_pts, vy_cc_pts, p_cc_pts = build_boundary_conditions(
        IENbound, ccName, conditions, npoints=4, device=device, IENpoint=IENpoint)

    assert p_cc_pts.tolist() == [2]
    assert p_cc[2] == 0.0


def test_perfil_parabolico_zera_nas_pontas_e_pico_no_meio():
    coord = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    v = perfil_parabolico(coord, vmax=1.5)

    assert np.isclose(v[0], 0.0) and np.isclose(v[-1], 0.0)
    assert np.isclose(v[2], 1.5)  # pico no centro
    # Poiseuille classico: 6y(1-y) para H=1, vmax=1.5
    assert np.allclose(v, 6.0 * coord * (1.0 - coord))


def test_perfil_parabolico_independe_da_altura_do_canal():
    """s0/s1 saem dos proprios nos, entao o mesmo config vale pra qualquer
    altura de canal -- o pico continua vmax e as pontas continuam zero.
    """
    coord = np.linspace(-2.0, 3.0, 11)  # canal de altura 5, deslocado
    v = perfil_parabolico(coord, vmax=2.0)

    assert np.isclose(v[0], 0.0) and np.isclose(v[-1], 0.0)
    assert np.isclose(v.max(), 2.0)


def test_perfil_parabolico_tem_media_dois_tercos_do_pico():
    """A media de uma parabola e' 2/3 do pico -- e' por isso que vmax=1.5
    reproduz a vazao da entrada uniforme vx=1.0.
    """
    coord = np.linspace(0.0, 1.0, 2001)
    v = perfil_parabolico(coord, vmax=1.5)

    assert np.isclose(np.trapezoid(v, coord), 1.0, rtol=1e-4)


def test_build_boundary_conditions_com_perfil_parabolico():
    """Entrada vertical: o perfil deve ser detectado ao longo de y e escrito
    nos nos do contorno, com os nos de parede zerados nas pontas.
    """
    # canal [0,1]x[0,1]: contorno esquerdo (inlet) com 3 nos em y=0, 0.5, 1
    X = np.array([0.0, 0.0, 0.0])
    Y = np.array([0.0, 0.5, 1.0])
    IENbound = np.array([[0, 1], [1, 2]])
    ccName = ["inlet", "inlet", "inlet"]
    conditions = {"inlet": {"vx": {"perfil": "parabolico", "vmax": 1.5}, "vy": 0.0}}

    vx_cc, vy_cc, _, vx_pts, _, _ = build_boundary_conditions(
        IENbound, ccName, conditions, npoints=3, device=torch.device("cpu"), X=X, Y=Y)

    assert np.isclose(vx_cc[0].item(), 0.0)
    assert np.isclose(vx_cc[1].item(), 1.5)
    assert np.isclose(vx_cc[2].item(), 0.0)
    assert set(vx_pts.tolist()) == {0, 1, 2}
    assert torch.allclose(vy_cc, torch.zeros(3, dtype=torch.float64))


def test_build_boundary_conditions_perfil_sem_coordenadas_falha():
    IENbound = np.array([[0, 1]])
    conditions = {"inlet": {"vx": {"perfil": "parabolico", "vmax": 1.0}}}
    try:
        build_boundary_conditions(IENbound, ["inlet", "inlet"], conditions,
                                   npoints=2, device=torch.device("cpu"))
    except ValueError as e:
        assert "coordenadas" in str(e)
    else:
        raise AssertionError("deveria ter levantado ValueError")


def test_perfil_usa_extensao_geometrica_nao_os_nos_restantes():
    """Regressao: as quinas do contorno de entrada pertencem a parede (por
    prioridade), entao saem do conjunto de nos que recebem o perfil. Se a
    extensao da parabola for calculada so com os nos restantes, ela zera
    DENTRO do canal em vez de na parede e a vazao sai baixa -- erro de
    ~2h/H (6,4% medido numa malha real), com a vazao ainda conservada de
    secao a secao, o que faz parecer erro de discretizacao.

    Canal [0,1] com 5 nos: as quinas (y=0 e y=1) viram parede; o perfil e'
    aplicado so em y=0.25, 0.5, 0.75, mas tem que continuar valendo
    6y(1-y) -- isto e', ancorado em y=0 e y=1.
    """
    X = np.zeros(5)
    Y = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    IENbound = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    IENboundElem = ["inlet"] * 4
    # prioridade resolvida: quinas sao parede, miolo e' inlet
    ccName = ["parede", "inlet", "inlet", "inlet", "parede"]
    conditions = {
        "inlet": {"vx": {"perfil": "parabolico", "vmax": 1.5}},
        "parede": {"vx": 0.0},
    }

    vx_cc, *_ = build_boundary_conditions(
        IENbound, ccName, conditions, npoints=5, device=torch.device("cpu"),
        X=X, Y=Y, IENboundElem=IENboundElem)

    esperado = 6.0 * Y * (1.0 - Y)  # ancorado nas paredes reais
    assert np.allclose(vx_cc.numpy(), esperado), f"{vx_cc.numpy()} != {esperado}"
    assert np.isclose(vx_cc[2].item(), 1.5)  # pico no centro do canal


def test_perfil_sem_IENboundElem_usa_os_proprios_nos():
    """Sem a informacao geometrica a extensao cai nos nos que recebem o valor
    -- comportamento documentado (e o que torna o parametro necessario).
    """
    X = np.zeros(5)
    Y = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    IENbound = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    ccName = ["parede", "inlet", "inlet", "inlet", "parede"]
    conditions = {"inlet": {"vx": {"perfil": "parabolico", "vmax": 1.5}}}

    vx_cc, *_ = build_boundary_conditions(
        IENbound, ccName, conditions, npoints=5, device=torch.device("cpu"), X=X, Y=Y)

    # parabola ancorada em y=0.25 e y=0.75: zera nos extremos do miolo
    assert np.isclose(vx_cc[1].item(), 0.0)
    assert np.isclose(vx_cc[3].item(), 0.0)
    assert np.isclose(vx_cc[2].item(), 1.5)
