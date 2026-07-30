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
