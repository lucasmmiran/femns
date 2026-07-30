#!/usr/bin/env python3
"""CLI: roda uma simulacao de Navier-Stokes (elemento MINI, Euler implicito) a partir de um config yaml."""

import argparse
import os
from datetime import datetime
from timeit import default_timer as timer

import numpy as np
import torch
import yaml
from tqdm import tqdm

from femns.assembly import assemble_mini
from femns.boundary import apply_boundary_conditions, assign_boundary_names, build_boundary_conditions
from femns.io import write_vtk
from femns.mesh import elem_mini, montar_EToE, montar_node_to_elem, montar_NToN, read_mesh
from femns.moving_mesh import mover_ponto
from femns.report import salvar_resumo
from femns.semi_lagrangian import calculo_sl
from femns.solver import build_system_matrix, pressure_scale_factor, time_step


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
    advection = cfg["simulation"].get("advection", "explicit")
    output_dir = cfg["output_dir"]

    start = timer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print('\n')
    print('Iniciando simulação de dinâmica dos fluidos')
    print('\n--------------------------------------------\n')
    print(f'Iniciando a simulacao com o dispositivo: {device}')
    print(f'Numero de iteracoes: {Iter}')
    print(f'Passo de tempo: {dt}')
    print(f'Numero de Reynolds: {Re}')
    print(f'Método escolhido: {advection}')
    print('\n--------------------------------------------')

    mesh = read_mesh(cfg["mesh"]) # Criado um objeto Mesh para a malha utilizada
    npoints, ne = mesh.npoints, mesh.ne

    print('Leitura de malha completa:')
    print(f'Numero de elementos: {ne}')
    print(f'Numero de pontos: {npoints}')
    print(f'Numero de nos da simulacao: {npoints + ne}')
    print('\n--------------------------------------------\n')

    # Condicoes de contorno (nomes por no, depois valores/indices)
    ccName = assign_boundary_names(mesh.IENbound, mesh.IENboundElem, npoints, cfg["boundary"]["priority"],
                                    mesh.IENpoint, mesh.IENpointElem) # Roda a função que nomeia os pontos dos contornos com a priorização
    vx_cc, vy_cc, p_cc, vx_cc_pts, vy_cc_pts, p_cc_pts = build_boundary_conditions(
        mesh.IENbound, ccName, cfg["boundary"]["conditions"], npoints, device, mesh.IENpoint)

    # Conectividade de vizinhanca (antes de acrescentar os centroides do elemento MINI)
    NToN = montar_NToN(mesh.IEN, npoints)

    if advection == "semi_lagrangian":
        EToE, _ = montar_EToE(mesh.IEN)
        node_to_elem = montar_node_to_elem(mesh.IEN, npoints)

    IEN, X, Y = elem_mini(mesh.IEN, mesh.X, mesh.Y)
    IEN, X, Y = torch.from_numpy(IEN).to(device), torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)

    print(f"Tempo ate o assembly: {round(timer() - start, 2)}")

    K, M, Gx, Gy, Gvx, Gvy = assemble_mini(X, Y, IEN, ne, npoints)
    beta = pressure_scale_factor(K, M, Gx, Gy, dt, Re)
    print(f"Fator de escala da pressao (beta): {beta:.4f}")
    A = build_system_matrix(dt, Re, K, M, Gx, Gy, beta)

    print(f"Tempo depois do assembly: {round(timer() - start, 2)}")
    t_assembly = timer() - start

    A = apply_boundary_conditions(A, vx_cc_pts, vy_cc_pts, p_cc_pts, npoints, ne)
    A = A.to_sparse_csr()
    Gvx = Gvx.to_sparse_csr()
    Gvy = Gvy.to_sparse_csr()
    M = M.to_sparse_csr()

    # Registro opcional de metricas (femns.report) pra comparar simulacoes
    # entre si -- so ativa se o config tiver benchmark_xlsx, pra nao mudar
    # nada no caminho padrao. Dx,Dy = -Gx^T,-Gy^T (sem beta -- e a restricao
    # fisica de incompressibilidade; beta e so escala numerica do solver,
    # ver solver.build_system_matrix) usados so pra medir divergencia por
    # passo, nunca entram no sistema linear resolvido.
    benchmark_xlsx = cfg.get("benchmark_xlsx")
    if benchmark_xlsx:
        Dx = (-torch.transpose(Gx, 0, 1)).coalesce().to_sparse_csr()
        Dy = (-torch.transpose(Gy, 0, 1)).coalesce().to_sparse_csr()
        bicg_iters_hist, bicg_residual_hist = [], []
        vel_l2_hist, vel_linf_hist, div_l2_hist = [], [], []

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

    x0 = torch.cat([vx, vy, p / beta])  # chute inicial do BiCGSTAB (p_tilde = p/beta)

    print(f"Tempo ate inicio das iteracoes: {round(timer() - start, 2)}")

    mp_cfg = cfg.get("moving_point", {})
    ponto = mp_cfg.get("node_index")
    mp_enabled = mp_cfg.get("enabled", False) and ponto is not None
    amplitude_factor = mp_cfg.get("amplitude_factor", 0.1)
    omega = mp_cfg.get("omega", np.pi * 200)

    if mp_enabled:
        ponto_x0, ponto_y0 = X[ponto].item(), Y[ponto].item()
        X, Y, vx, vy = mover_ponto(ponto, X, Y, vx, vy, ponto_x0, ponto_y0, NToN, 0, amplitude_factor, omega)

    pbar = tqdm(range(Iter))
    for n in pbar:
        if advection == "semi_lagrangian":
            vx_star_np, vy_star_np = calculo_sl(
                X.cpu().numpy(), Y.cpu().numpy(), IEN.cpu().numpy(),
                EToE, node_to_elem, ccName, cfg["boundary"]["conditions"],
                vx.cpu().numpy(), vy.cpu().numpy(), dt, npoints, ne,
            )
            vx_star = torch.from_numpy(vx_star_np).to(device)
            vy_star = torch.from_numpy(vy_star_np).to(device)
        else:
            vx_star = vy_star = None

        vx, vy, p, x0, info = time_step(
            A, M, Gvx, Gvy, vx, vy, dt,
            vx_cc, vy_cc, p_cc, vx_cc_pts, vy_cc_pts, p_cc_pts,
            npoints, ne, beta=beta, x0=x0, vx_star=vx_star, vy_star=vy_star,
        )
        pbar.set_postfix(bicg_iters=info["iters"], residual=f'{info["residual"]:.1e}')

        vx_sol = vx[:npoints]
        vy_sol = vy[:npoints]

        if benchmark_xlsx:
            bicg_iters_hist.append(info["iters"])
            bicg_residual_hist.append(info["residual"])
            vel_mag = torch.sqrt(vx_sol**2 + vy_sol**2)
            vel_l2_hist.append(torch.sqrt(torch.mean(vel_mag**2)).item())
            vel_linf_hist.append(torch.max(vel_mag).item())
            div = torch.mm(Dx, vx.unsqueeze(1)).squeeze(1) + torch.mm(Dy, vy.unsqueeze(1)).squeeze(1)
            div_l2_hist.append(torch.sqrt(torch.mean(div**2)).item())

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
            X, Y, vx, vy = mover_ponto(ponto, X, Y, vx, vy, ponto_x0, ponto_y0, NToN, n * dt, amplitude_factor, omega)
            tqdm.write(f"Posicao do ponto {ponto}: X={X[ponto]:.6f}, Y={Y[ponto]:.6f}")

    print(f"Tempo final: {round(timer() - start, 2)} segundos")

    if benchmark_xlsx:
        tempo_total = timer() - start
        dados = dict(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            mesh=cfg["mesh"], advection=advection, dt=dt, reynolds=Re, iterations=Iter,
            npoints=npoints, ne=ne, device=str(device),
            tempo_total_s=round(tempo_total, 3),
            tempo_assembly_s=round(t_assembly, 3),
            tempo_medio_por_iter_s=round((tempo_total - t_assembly) / Iter, 5),
            bicg_iters_media=round(sum(bicg_iters_hist) / len(bicg_iters_hist), 2),
            bicg_iters_max=max(bicg_iters_hist),
            bicg_residual_media=sum(bicg_residual_hist) / len(bicg_residual_hist),
            bicg_residual_max=max(bicg_residual_hist),
            passos_nao_convergidos=sum(1 for r in bicg_residual_hist if r > 1e-8),  # mesmo tol padrao de solver.time_step
            vel_l2_media=sum(vel_l2_hist) / len(vel_l2_hist),
            vel_l2_max=max(vel_l2_hist),
            vel_linf_max=max(vel_linf_hist),
            divergencia_l2_media=sum(div_l2_hist) / len(div_l2_hist),
            divergencia_l2_max=max(div_l2_hist),
            pressao_min=p.min().item(),
            pressao_max=p.max().item(),
            pressao_media=p.mean().item(),
        )
        salvar_resumo(benchmark_xlsx, dados)
        print(f"Resumo de benchmark salvo em {benchmark_xlsx}")


if __name__ == "__main__":
    main()
