#!/usr/bin/env python3
"""CLI: roda uma simulacao de Navier-Stokes (elemento MINI, Euler implicito) a partir de um config yaml."""

import argparse
import os
from timeit import default_timer as timer

import numpy as np
import torch
import yaml
from tqdm import tqdm

from femns.assembly import assemble_mini
from femns.boundary import assign_boundary_names, build_boundary_conditions, apply_boundary_conditions
from femns.mesh import elem_mini, montar_NToN, read_mesh
from femns.moving_mesh import mover_ponto
from femns.solver import build_system_matrix, time_step
from femns.io import write_vtk


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/poiseuille.yaml", help="Caminho do arquivo de config yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dt = cfg["simulation"]["dt"]
    Iter = cfg["simulation"]["iterations"]
    Re = cfg["simulation"]["reynolds"]
    output_dir = cfg["output_dir"]

    start = timer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    print(f'Device utilizado: {device}')
    print('\n--------------------------------------------\n')
    print(f'Iniciando a simulacao com o dispositivo: {device}')
    print(f'Numero de iteracoes: {Iter}')
    print(f'Passo de tempo: {dt}')
    print(f'Numero de Reynolds: {Re}')
    print('\n--------------------------------------------')

    mesh = read_mesh(cfg["mesh"])
    npoints, ne = mesh.npoints, mesh.ne

    print('Leitura de malha completa:')
    print(f'Numero de elementos: {ne}')
    print(f'Numero de pontos: {npoints}')
    print(f'Numero de nos da simulacao: {npoints + ne}')
    print('\n--------------------------------------------\n')

    # Condicoes de contorno (nomes por no, depois valores/indices)
    ccName = assign_boundary_names(mesh.IENbound, mesh.IENboundElem, npoints, cfg["boundary"]["priority"])
    vx_cc, vy_cc, p_cc, vx_cc_pts, vy_cc_pts, p_cc_pts = build_boundary_conditions(
        mesh.IENbound, ccName, cfg["boundary"]["conditions"], npoints, device)

    # Conectividade de vizinhanca (antes de acrescentar os centroides do elemento MINI)
    NToN = montar_NToN(mesh.IEN, npoints)

    IEN, X, Y = elem_mini(mesh.IEN, mesh.X, mesh.Y)
    IEN, X, Y = torch.from_numpy(IEN).to(device), torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)

    print(f"Tempo ate o assembly: {round(timer() - start, 2)}")

    K, M, Gx, Gy, Gvx, Gvy = assemble_mini(X, Y, IEN, ne, npoints)
    A = build_system_matrix(dt, Re, K, M, Gx, Gy)

    print(f"Tempo depois do assembly: {round(timer() - start, 2)}")

    A = apply_boundary_conditions(A, vx_cc_pts, vy_cc_pts, p_cc_pts, npoints, ne)
    A = A.to_sparse_csr()
    Gvx = Gvx.to_sparse_csr()
    Gvy = Gvy.to_sparse_csr()
    M = M.to_sparse_csr()

    updated_points = mesh.raw.points.copy()

    write_vtk(
        os.path.join(output_dir, "CondicaoDeContorno.vtk"),
        mesh.raw.points,
        mesh.raw.cells,
        point_data={
            "vx_cc": vx_cc.cpu().numpy(),
            "vy_cc": vy_cc.cpu().numpy(),
            "p_cc": p_cc.cpu().numpy(),
        },
    )

    vx = torch.zeros(npoints + ne, dtype=torch.float64, device=device)
    vy = torch.zeros(npoints + ne, dtype=torch.float64, device=device)
    p = torch.zeros(npoints, dtype=torch.float64, device=device)

    vx[vx_cc_pts] = vx_cc[vx_cc_pts]
    vy[vy_cc_pts] = vy_cc[vy_cc_pts]
    p[p_cc_pts] = p_cc[p_cc_pts]

    print(f"Tempo ate inicio das iteracoes: {round(timer() - start, 2)}")

    mp_cfg = cfg.get("moving_point", {})
    ponto = mp_cfg.get("node_index")
    mp_enabled = mp_cfg.get("enabled", False) and ponto is not None
    amplitude_factor = mp_cfg.get("amplitude_factor", 0.1)
    omega = mp_cfg.get("omega", np.pi * 200)

    if mp_enabled:
        x0, y0 = X[ponto].item(), Y[ponto].item()
        X, Y, vx, vy = mover_ponto(ponto, X, Y, vx, vy, x0, y0, NToN, 0, amplitude_factor, omega)

    for n in tqdm(range(Iter)):
        vx, vy, p = time_step(
            A, M, Gvx, Gvy, vx, vy, dt,
            vx_cc, vy_cc, p_cc, vx_cc_pts, vy_cc_pts, p_cc_pts,
            npoints, ne,
        )

        vx_sol = vx[:npoints]
        vy_sol = vy[:npoints]

        updated_points[:, 0] = X[:npoints].cpu().numpy()
        updated_points[:, 1] = Y[:npoints].cpu().numpy()

        write_vtk(
            os.path.join(output_dir, f"solucao -{n + 1}.vtk"),
            updated_points,
            mesh.raw.cells,
            point_data={
                "vx": vx_sol.cpu().numpy(),
                "vy": vy_sol.cpu().numpy(),
                "p": p.cpu().numpy(),
            },
        )

        if mp_enabled:
            X, Y, vx, vy = mover_ponto(ponto, X, Y, vx, vy, x0, y0, NToN, n * dt, amplitude_factor, omega)
            tqdm.write(f"Posicao do ponto {ponto}: X={X[ponto]:.6f}, Y={Y[ponto]:.6f}")

    print(f"Tempo final: {round(timer() - start, 2)} segundos")


if __name__ == "__main__":
    main()
