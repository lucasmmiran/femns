import numpy as np
import torch

from femns.boundary import assign_boundary_names, build_boundary_conditions


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
