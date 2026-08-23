#!/usr/bin/env python3
"""Benchmark: adveccao semi-Lagrangeana ponto a ponto (numpy) vs matriz esparsa (torch).

Compara `femns.semi_lagrangian.calculo_sl` (metodo atual, busca+interpolacao
vetorizada em numpy, fallback de Dirichlet no contorno) com
`femns.semi_lagrangian_tri.SemiLagrangianMini` (metodo novo, CLAUDE_sl.md:
matriz de interpolacao esparsa Pi em torch, interceptacao de aresta no
contorno). Roda na malha real do Poiseuille, num campo de velocidade nao
trivial (1 passo real de `solver.time_step`, nao velocidade nula/uniforme --
mesma estrategia ja usada para validar a vetorizacao de
`locate_points_batch`).

So mede desempenho e concordancia numerica -- nao escreve .vtk, nao
integra com `run_simulation.py`.
"""

import argparse
from timeit import default_timer as timer

import numpy as np
import torch
import yaml

from femns.assembly import assemble_mini
from femns.boundary import apply_boundary_conditions, assign_boundary_names, build_boundary_conditions
from femns.mesh import elem_mini, montar_EToE, montar_node_to_elem, read_mesh
from femns.semi_lagrangian import calculo_sl, localizar_pontos_partida
from femns.semi_lagrangian_tri import SemiLagrangianMini
from femns.solver import build_system_matrix, pressure_scale_factor, time_step


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/poiseuille.yaml", help="Config yaml (usa so mesh/dt/Re/contorno)")
    parser.add_argument("--repeats", type=int, default=50, help="Repeticoes cronometradas de cada metodo")
    return parser.parse_args()


def preparar_campo_real(cfg: dict, device: torch.device):
    """Monta a malha/sistema e roda 1 passo eulerian real -- devolve um campo vx,vy nao trivial."""
    dt = cfg["simulation"]["dt"]
    Re = cfg["simulation"]["reynolds"]

    mesh = read_mesh(cfg["mesh"])
    npoints, ne = mesh.npoints, mesh.ne

    ccName = assign_boundary_names(mesh.IENbound, mesh.IENboundElem, npoints, cfg["boundary"]["priority"])
    vx_cc, vy_cc, p_cc, vx_cc_pts, vy_cc_pts, p_cc_pts = build_boundary_conditions(
        mesh.IENbound, ccName, cfg["boundary"]["conditions"], npoints, device)

    EToE, _ = montar_EToE(mesh.IEN)
    node_to_elem = montar_node_to_elem(mesh.IEN, npoints)

    IEN, X, Y = elem_mini(mesh.IEN, mesh.X, mesh.Y)
    IEN_t = torch.from_numpy(IEN).to(device)
    X_t, Y_t = torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)

    K, M, Gx, Gy, Gvx, Gvy = assemble_mini(X_t, Y_t, IEN_t, ne, npoints)
    beta = pressure_scale_factor(K, M, Gx, Gy, dt, Re)
    A = build_system_matrix(dt, Re, K, M, Gx, Gy, beta)
    A = apply_boundary_conditions(A, vx_cc_pts, vy_cc_pts, p_cc_pts, npoints, ne).to_sparse_csr()
    Gvx, Gvy, M = Gvx.to_sparse_csr(), Gvy.to_sparse_csr(), M.to_sparse_csr()

    vx = torch.zeros(npoints + ne, dtype=torch.float64, device=device)
    vy = torch.zeros(npoints + ne, dtype=torch.float64, device=device)
    p = torch.zeros(npoints, dtype=torch.float64, device=device)
    vx[vx_cc_pts] = vx_cc[vx_cc_pts]
    vy[vy_cc_pts] = vy_cc[vy_cc_pts]
    p[p_cc_pts] = p_cc[p_cc_pts]
    x0 = torch.cat([vx, vy, p / beta])

    # 1 passo eulerian real (advection default) -- campo pos-1-passo, nao trivial
    vx, vy, p, x0, info = time_step(
        A, M, Gvx, Gvy, vx, vy, dt, vx_cc, vy_cc, p_cc, vx_cc_pts, vy_cc_pts, p_cc_pts, npoints, ne, beta=beta, x0=x0)
    print(f"Campo de teste: 1 passo eulerian real (bicg iters={info['iters']}, residuo={info['residual']:.1e})")

    return dict(
        IEN=IEN, IEN_t=IEN_t, X=X, Y=Y, X_t=X_t, Y_t=Y_t, EToE=EToE, node_to_elem=node_to_elem,
        ccName=ccName, conditions=cfg["boundary"]["conditions"], vx=vx, vy=vy, dt=dt, npoints=npoints, ne=ne, device=device,
    )


def cronometrar(func, repeats: int, device: torch.device):
    inicio = timer()
    for _ in range(repeats):
        func()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (timer() - inicio) / repeats


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    campo = preparar_campo_real(cfg, device)
    npoints, ne, dt = campo["npoints"], campo["ne"], campo["dt"]

    # --- metodo atual (numpy, ponto a ponto) ---
    vx_np, vy_np = campo["vx"].cpu().numpy(), campo["vy"].cpu().numpy()
    X_np, Y_np = campo["X"], campo["Y"]

    def rodar_atual():
        return calculo_sl(
            X_np, Y_np, campo["IEN"], campo["EToE"], campo["node_to_elem"],
            campo["ccName"], campo["conditions"], vx_np, vy_np, dt, npoints, ne)

    def rodar_atual_com_conversao():
        """Inclui o round-trip torch<->numpy que run_simulation.py faz a cada passo -- custo real de producao."""
        vx_np_ = campo["vx"].cpu().numpy()
        vy_np_ = campo["vy"].cpu().numpy()
        vx_star_np, vy_star_np = calculo_sl(
            X_np, Y_np, campo["IEN"], campo["EToE"], campo["node_to_elem"],
            campo["ccName"], campo["conditions"], vx_np_, vy_np_, dt, npoints, ne)
        return torch.from_numpy(vx_star_np).to(device), torch.from_numpy(vy_star_np).to(device)

    # --- metodo novo (torch, matriz esparsa) ---
    node_to_elem_t = torch.from_numpy(campo["node_to_elem"]).to(device)
    sl = SemiLagrangianMini(campo["IEN_t"], campo["EToE"], campo["X_t"], campo["Y_t"], node_to_elem_t, npoints, ne)
    vx_t, vy_t = campo["vx"], campo["vy"]

    def rodar_novo():
        return sl.compute(vx_t, vy_t, dt)

    # aplicar Pi ja montada (custo amortizado por campo extra, sem rebuscar elemento)
    # -- sl.conv ja sai em CSR de compute(), sem conversao extra aqui.
    sl.compute(vx_t, vy_t, dt)
    conv_pronta = sl.conv

    def aplicar_apenas():
        torch.mm(conv_pronta, vx_t.unsqueeze(1)).squeeze(1)
        torch.mm(conv_pronta, vy_t.unsqueeze(1)).squeeze(1)

    print(f"Malha: {npoints} pontos, {ne} elementos, {npoints + ne} nos (com centroide)\n")
    print(f"Cronometrando {args.repeats} repeticoes de cada...\n")

    t_atual = cronometrar(rodar_atual, args.repeats, device)
    t_atual_conv = cronometrar(rodar_atual_com_conversao, args.repeats, device)
    t_novo = cronometrar(rodar_novo, args.repeats, device)
    t_aplicar = cronometrar(aplicar_apenas, args.repeats, device)

    print("Desempenho (tempo medio por passo):")
    print(f"  atual   (numpy, so calculo)          : {t_atual * 1e3:8.3f} ms")
    print(f"  atual   (numpy, + round-trip torch)  : {t_atual_conv * 1e3:8.3f} ms  <- custo real em run_simulation.py")
    print(f"  novo    (torch, monta Pi + aplica)   : {t_novo * 1e3:8.3f} ms")
    print(f"  novo    (torch, so aplica Pi pronta) : {t_aplicar * 1e3:8.3f} ms  <- custo por campo extra amortizado")
    print(f"  speedup novo/atual (custo real)      : {t_atual_conv / t_novo:8.2f}x\n")

    # --- concordancia numerica (so nos interiores nos dois metodos) ---
    vx_star_atual, vy_star_atual = calculo_sl(
        X_np, Y_np, campo["IEN"], campo["EToE"], campo["node_to_elem"],
        campo["ccName"], campo["conditions"], vx_np, vy_np, dt, npoints, ne)
    vx_star_novo, vy_star_novo = rodar_novo()

    elem_atual, _, _ = localizar_pontos_partida(
        X_np - dt * vx_np, Y_np - dt * vy_np,
        np.concatenate([campo["node_to_elem"], np.arange(ne)]),
        X_np, Y_np, campo["IEN"], campo["EToE"])
    interior_atual = elem_atual != -1

    # sl.conv ja sai em CSR -- nnz por linha direto de crow_indices, sem to_dense()
    crow = sl.conv.crow_indices()
    nnz_por_linha = crow[1:] - crow[:-1]
    interior_novo = (nnz_por_linha == sl.NEN).cpu().numpy()

    interior_ambos = interior_atual & interior_novo
    n_interior = interior_ambos.sum()

    vx_star_novo_np = vx_star_novo.cpu().numpy()
    vy_star_novo_np = vy_star_novo.cpu().numpy()

    dif_vx = np.abs(vx_star_atual[interior_ambos] - vx_star_novo_np[interior_ambos])
    dif_vy = np.abs(vy_star_atual[interior_ambos] - vy_star_novo_np[interior_ambos])

    print("Concordancia numerica:")
    print(f"  nos interiores em ambos os metodos   : {n_interior} / {npoints + ne}")
    print(f"  max|dif| vx (so interior)             : {dif_vx.max():.3e}")
    print(f"  max|dif| vy (so interior)             : {dif_vy.max():.3e}")
    print(f"  nos que saem do dominio (atual/novo)  : {(~interior_atual).sum()} / {(~interior_novo).sum()}")
    print("  (divergencia nos nos de contorno e esperada -- estrategias de contorno diferentes por desenho:")
    print("   fallback de Dirichlet no metodo atual vs interceptacao de aresta no metodo novo)")


if __name__ == "__main__":
    main()
