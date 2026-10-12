#!/usr/bin/env python3
"""CLI: roda uma simulacao de Navier-Stokes (elemento MINI ou Tri6, Euler implicito, adveccao semi-Lagrangeana) a partir de um config yaml."""

import argparse
import os
from datetime import datetime
from timeit import default_timer as timer

import numpy as np
import torch
import yaml
from tqdm import tqdm

from femns.boundary import assign_boundary_names, build_boundary_conditions
from femns.elemento import elemento_por_nome
from femns.io import caminho_frame, write_vtk
from femns.mesh import (
    area_com_sinal,
    atualiza_nos_extras,
    montar_NToN,
    read_mesh,
)
from femns.moving_mesh import aplicar_oscilacao, passos_por_periodo, preparar_oscilacao
from femns.precondicionadores import DE_BLOCO, PassosChebyshev
from femns.precondicionadores import montar as montar_precondicionador
from femns.reassembly import Remontador
from femns.report import salvar_resumo
from femns.semi_lagrangian import Localizador, calculo_sl
from femns.simconfig import validar_config
from femns.solver import time_step


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/poiseuille.yaml", help="Caminho do arquivo de config yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    params = validar_config(cfg)  # todas as regras do config (femns.simconfig, as mesmas da GUI)
    dt, Iter, Re = params.dt, params.iterations, params.reynolds
    element, vtk_interval = params.element, params.vtk_interval
    bicgstab_tol, bicgstab_max_iter, beta_por_passo = params.tol, params.max_iter, params.beta_por_passo
    opcoes = params.solver
    passos_cheb = PassosChebyshev(*opcoes.chebyshev)
    elemento = elemento_por_nome(element)  # a unica escolha do elemento: o objeto vai adiante
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
    print(f'Elemento: {element}')
    if vtk_interval == 1:
        print('Gravacao de .vtk: a cada iteracao')
    else:
        print(f'Gravacao de .vtk: a cada {vtk_interval} iteracoes (+ a ultima)')
    print(f'Solver linear: {opcoes.metodo}, pre-condicionador {opcoes.precondicionador} '
          f'(sub-solver {opcoes.subsolver}), '
          f'tol={bicgstab_tol:g}, max_iter={bicgstab_max_iter}')
    if beta_por_passo:
        print('Equilibracao de pressao: recalculada a cada passo (beta_por_passo)')
    print('\n--------------------------------------------')

    mesh = read_mesh(cfg["mesh"]) # Criado um objeto Mesh para a malha utilizada
    npoints, ne = mesh.npoints, mesh.ne

    # Augmenta a malha de vertices (mesh.IEN) com os nos extras do elemento
    # de velocidade escolhido, e estende IENbound de acordo (so o Tri6 tem
    # no extra em contorno -- a bolha do MINI e sempre interna ao elemento).
    IEN_np, X_np, Y_np, IENbound_bc = elemento.adiciona_nos(mesh.IEN, mesh.X, mesh.Y, mesh.IENbound)
    nnodes = X_np.shape[0]
    n_extra = nnodes - npoints  # equivalente a `ne` para boundary/solver (ver docstrings)

    print('Leitura de malha completa:')
    print(f'Numero de elementos: {ne}')
    print(f'Numero de pontos: {npoints}')
    print(f'Numero de nos da simulacao: {nnodes}')
    print('\n--------------------------------------------\n')

    # Condicoes de contorno (nomes por no, depois valores/indices)
    ccName = assign_boundary_names(IENbound_bc, mesh.IENboundElem, npoints, cfg["boundary"]["priority"],
                                    mesh.IENpoint, mesh.IENpointElem, nnodes=nnodes) # Roda a função que nomeia os pontos dos contornos com a priorização
    vx_cc, vy_cc, p_cc, vx_cc_pts, vy_cc_pts, p_cc_pts = build_boundary_conditions(
        IENbound_bc, ccName, cfg["boundary"]["conditions"], npoints, device, mesh.IENpoint, nnodes=nnodes,
        X=X_np, Y=Y_np, IENboundElem=mesh.IENboundElem)

    # Dirichlet por no, ja resolvido (perfis inclusive) -- o semi-Lagrangeano
    # usa no residuo de nos sem cruzamento de contorno util (ver calculo_sl).
    has_vx_bc = np.zeros(nnodes, dtype=bool)
    has_vy_bc = np.zeros(nnodes, dtype=bool)
    has_vx_bc[vx_cc_pts.cpu().numpy()] = True
    has_vy_bc[vy_cc_pts.cpu().numpy()] = True
    bc_dirichlet = (vx_cc.cpu().numpy(), has_vx_bc, vy_cc.cpu().numpy(), has_vy_bc)

    # Busca por elemento do semi-Lagrangeano: so a topologia (vizinhos e semente
    # de cada no), que nao muda com a malha movel -- criada uma vez.
    localizador = Localizador(IEN_np, nnodes)

    IEN, X, Y = torch.from_numpy(IEN_np).to(device), torch.from_numpy(X_np).to(device), torch.from_numpy(Y_np).to(device)

    print(f"Tempo ate o assembly: {round(timer() - start, 2)}")

    # Montagem do sistema (assembly + matriz de bloco + condicoes de contorno).
    # `Remontador` e' o unico caminho: monta o sistema inicial, conferindo
    # contra a montagem de referencia (build_system_matrix +
    # apply_boundary_conditions), e com mesh_motion remonta a cada passo so
    # recomputando valores no padrao de esparsidade fixo.
    # Saidas livres (contornos sem velocidade prescrita): Dirichlet so no Lp
    # do pre-condicionador de bloco (ver Remontador), para ele nao ser singular
    # quando nao ha pressao prescrita.
    sem_velocidade = [nome for nome in set(mesh.IENboundElem)
                      if not {"vx", "vy"} & set(cfg["boundary"]["conditions"].get(nome, {}))]
    nos_saida_livre = np.unique(np.concatenate(
        [mesh.IENbound[np.array(mesh.IENboundElem) == nome].ravel() for nome in sem_velocidade]
        or [np.empty(0, dtype=int)]))
    nos_saida_livre = nos_saida_livre[~(has_vx_bc[nos_saida_livre] | has_vy_bc[nos_saida_livre])]

    # As condicoes de Dirichlet saem de A de forma simetrica (linha e coluna;
    # a parcela delas vai para o lado direito, sistema.rhs_cc).
    remontador = Remontador(
        dt, Re, X, Y, IEN, elemento, npoints, n_extra,
        vx_cc_pts, vy_cc_pts, p_cc_pts, valores_cc=(vx_cc, vy_cc, p_cc),
        precisa_blocos=opcoes.precondicionador in DE_BLOCO, nos_saida_livre=nos_saida_livre)
    sistema = remontador.sistema_inicial
    beta = sistema.beta
    precond = montar_precondicionador(opcoes.precondicionador, sistema, dt, Re, passos_cheb, opcoes.subsolver)
    print(f"Fator de escala da pressao (beta): {beta:.4f}")
    print(f"Montagem: erro de verificacao {remontador.erro_verificacao:.1e} (vs montagem de referencia)")

    print(f"Tempo depois do assembly: {round(timer() - start, 2)}")
    t_assembly = timer() - start

    mm_cfg = cfg.get("mesh_motion", {})
    mm_enabled = mm_cfg.get("enabled", False)

    # Registro opcional de metricas (femns.report) pra comparar simulacoes
    # entre si -- so ativa se o config tiver benchmark_xlsx, pra nao mudar
    # nada no caminho padrao. A divergencia por passo vem de
    # `remontador.divergencia` (D = -G^T, sem beta).
    benchmark_xlsx = cfg.get("benchmark_xlsx")
    if benchmark_xlsx:
        bicg_iters_hist, bicg_residual_hist = [], []
        tempo_passo_hist, erro_direto_v_hist, erro_direto_p_hist = [], [], []
        vel_l2_hist, vel_linf_hist, div_l2_hist = [], [], []

    updated_points = mesh.raw.points.copy()

    write_vtk(
        os.path.join(output_dir, "CondicaoDeContorno.vtk"),
        mesh.raw.points,
        mesh.raw.cells,
        point_data={
            "vx_cc": vx_cc[:npoints].cpu().numpy(),
            "vy_cc": vy_cc[:npoints].cpu().numpy(),
            "p_cc": p_cc.cpu().numpy(),
        },
    )

    vx = torch.zeros(nnodes, dtype=torch.float64, device=device)
    vy = torch.zeros(nnodes, dtype=torch.float64, device=device)
    p = torch.zeros(npoints, dtype=torch.float64, device=device)

    vx[vx_cc_pts] = vx_cc[vx_cc_pts]
    vy[vy_cc_pts] = vy_cc[vy_cc_pts]
    p[p_cc_pts] = p_cc[p_cc_pts]

    x0 = torch.cat([vx, vy, p / beta])  # chute inicial do BiCGSTAB (p_tilde = p/beta)

    print(f"Tempo ate inicio das iteracoes: {round(timer() - start, 2)}")

    # Oscilacao de todos os nos interiores (experimento de malha movel/ALE).
    # So nos de VERTICE que nao estao em nenhum contorno (`ccName[n] is None`):
    # os nos extras do elemento sao definidos pelos vertices e sao
    # reposicionados por `atualiza_nos_extras` depois de cada deslocamento,
    # nao oscilam por conta propria.  (`mm_cfg`/`mm_enabled` ja lidos acima.)
    oscilacao = None
    if mm_enabled:
        mm_omega = mm_cfg.get("omega", np.pi * 200)
        NToN = montar_NToN(mesh.IEN, npoints)  # vizinhanca: amplitude = fator x h local
        nos_interiores = np.array([n for n in range(npoints) if ccName[n] is None], dtype=int)
        oscilacao = preparar_oscilacao(
            X_np, Y_np, NToN, nos_interiores,
            fator=mm_cfg.get("amplitude_factor", 0.3),
            omega=mm_omega,
            seed=mm_cfg.get("seed", 0),
        )

        # Amostragem da oscilacao: com poucos passos por periodo a malha e'
        # sempre vista quase na mesma fase. No caso inteiro (omega*dt = 2*pi)
        # ela fica literalmente PARADA numa configuracao deslocada, e a
        # simulacao roda sem erro nenhum parecendo uma malha movel.
        n_por_periodo = passos_por_periodo(mm_omega, dt)
        if n_por_periodo < 4:
            raise ValueError(
                f"mesh_motion: {n_por_periodo:.2f} passos por periodo da oscilacao "
                f"(omega={mm_omega}, dt={dt}) -- abaixo de Nyquist, a malha nem chega a oscilar "
                f"(em exatamente 1.0 ela fica parada numa posicao deslocada). "
                f"Reduza mesh_motion.omega para <= {2 * np.pi / (10 * dt):.4f} "
                f"(10 passos por periodo) ou reduza dt."
            )
        if n_por_periodo < 10:
            print(f"AVISO: so {n_por_periodo:.1f} passos por periodo da oscilacao -- "
                  f"a malha movel esta mal resolvida no tempo.")

        vel_malha_max = (oscilacao.amplitude * oscilacao.omega).max()
        print(f"Malha movel: {len(nos_interiores)} nos interiores de {npoints} oscilando "
              f"(amplitude = {mm_cfg.get('amplitude_factor', 0.3)} x h local, "
              f"max = {oscilacao.amplitude.max():.3e})")
        print(f"  {n_por_periodo:.1f} passos por periodo | velocidade maxima da malha = {vel_malha_max:.3f}")

    # Geometria em que o campo (vx, vy) atual vive. Com malha movel ela fica
    # uma "foto" atras de X, Y: o campo foi resolvido na malha do passo
    # anterior, e e' nela que a interpolacao semi-Lagrangeana tem que
    # avaliar (ver semi_lagrangian.calculo_sl).
    X_campo, Y_campo = X.clone(), Y.clone()

    sincronizar = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    pbar = tqdm(range(Iter))
    for n in pbar:
        t_passo = timer()
        if mm_enabled:
            # Move a malha para o instante deste passo e remonta o sistema na
            # geometria nova -- sem isso as matrizes ficariam as da malha
            # original, inconsistentes com as coordenadas que a adveccao
            # semi-Lagrangeana e o VTK usam.
            X, Y = aplicar_oscilacao(oscilacao, X, Y, n * dt)
            X, Y = atualiza_nos_extras(IEN, X, Y)

            # Um elemento invertido nao levanta erro na montagem (que toma
            # abs da area) -- entraria no sistema com Gx/Gy de sinal trocado,
            # calado. Melhor parar aqui do que produzir uma solucao errada.
            area_min = area_com_sinal(IEN, X, Y).min().item()
            if area_min <= 0.0:
                raise RuntimeError(
                    f"malha movel inverteu elemento na iteracao {n} (area minima {area_min:.3e}). "
                    f"Reduza mesh_motion.amplitude_factor (atual: {mm_cfg.get('amplitude_factor', 0.3)})."
                )

            beta_anterior = beta
            sistema = remontador.remontar(X, Y, beta=(None if beta_por_passo else beta))
            beta = sistema.beta
            precond = montar_precondicionador(opcoes.precondicionador, sistema, dt, Re, passos_cheb, opcoes.subsolver)
            if beta_por_passo and beta != beta_anterior:
                # x0 carrega p_tilde = p / beta_anterior; o novo sistema resolve
                # p / beta, entao reescala so o componente de pressao do warm-start.
                x0[2 * nnodes:] *= beta_anterior / beta

        # Malha movel (ALE): a correcao e' geometrica -- o campo do passo
        # anterior e' interpolado na malha em que ele vive (X_campo/Y_campo).
        vx_star_np, vy_star_np = calculo_sl(
            X.cpu().numpy(), Y.cpu().numpy(), localizador,
            vx.cpu().numpy(), vy.cpu().numpy(), dt, bc_dirichlet,
            elemento=elemento,
            X_campo=X_campo.cpu().numpy(), Y_campo=Y_campo.cpu().numpy(),
        )
        vx_star = torch.from_numpy(vx_star_np).to(device)
        vy_star = torch.from_numpy(vy_star_np).to(device)

        verificar = opcoes.verificar_direto > 0 and (n + 1) % opcoes.verificar_direto == 0
        vx, vy, p, x0, info = time_step(
            sistema, vx_star, vy_star, dt, npoints, n_extra, x0=x0,
            metodo=opcoes.metodo, precond=precond, tol=bicgstab_tol, max_iter=bicgstab_max_iter,
            verificar_direto=verificar,
        )
        sincronizar()
        tempo_passo = timer() - t_passo - info.get("tempo_direto", 0.0)  # sem a checagem de validacao
        pbar.set_postfix(iters=info["iters"], residual=f'{info["residual_verdadeiro"]:.1e}')

        # O campo recem-resolvido vive na geometria deste passo: ela vira a
        # referencia de interpolacao do proximo (ver X_campo acima).
        if mm_enabled:
            X_campo, Y_campo = X.clone(), Y.clone()

        vx_sol = vx[:npoints]
        vy_sol = vy[:npoints]

        if benchmark_xlsx:
            bicg_iters_hist.append(info["iters"])
            bicg_residual_hist.append(info["residual_verdadeiro"])
            tempo_passo_hist.append(tempo_passo)
            if verificar:
                erro_direto_v_hist.append(info["erro_direto_v"])
                erro_direto_p_hist.append(info["erro_direto_p"])
            vel_mag = torch.sqrt(vx_sol**2 + vy_sol**2)
            vel_l2_hist.append(torch.sqrt(torch.mean(vel_mag**2)).item())
            vel_linf_hist.append(torch.max(vel_mag).item())
            div = remontador.divergencia(vx, vy)
            div_l2_hist.append(torch.sqrt(torch.mean(div**2)).item())

        # Grava a solucao a cada `vtk_interval` iteracoes; a ultima e' sempre
        # gravada para que o estado final nao dependa de `Iter` ser multiplo
        # do intervalo (e para `femns.io.ultimo_frame` / o visualizador
        # web continuarem achando o campo final).
        if (n + 1) % vtk_interval == 0 or (n + 1) == Iter:
            updated_points[:, 0] = X[:npoints].cpu().numpy()
            updated_points[:, 1] = Y[:npoints].cpu().numpy()

            write_vtk(
                caminho_frame(output_dir, n + 1),
                updated_points,
                mesh.raw.cells,
                point_data={
                    "vx": vx_sol.cpu().numpy(),
                    "vy": vy_sol.cpu().numpy(),
                    "p": p.cpu().numpy(),
                },
            )

    print(f"Tempo final: {round(timer() - start, 2)} segundos")

    if benchmark_xlsx:
        tempo_total = timer() - start
        dados = dict(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            mesh=cfg["mesh"], advection="semi_lagrangian",
            element=element, dt=dt, reynolds=Re, iterations=Iter,
            npoints=npoints, ne=ne, device=str(device),
            tempo_total_s=round(tempo_total, 3),
            tempo_assembly_s=round(t_assembly, 3),
            tempo_medio_por_iter_s=round((tempo_total - t_assembly) / Iter, 5),
            bicg_iters_media=round(sum(bicg_iters_hist) / len(bicg_iters_hist), 2),
            bicg_iters_max=max(bicg_iters_hist),
            bicg_iters_min=min(bicg_iters_hist),
            metodo=opcoes.metodo, precondicionador=opcoes.precondicionador, subsolver=opcoes.subsolver,
            tempo_passo_media_s=round(sum(tempo_passo_hist) / len(tempo_passo_hist), 5),
            tempo_passo_max_s=round(max(tempo_passo_hist), 5),
            tempo_passo_min_s=round(min(tempo_passo_hist), 5),
            erro_direto_v_max=max(erro_direto_v_hist) if erro_direto_v_hist else "",
            erro_direto_p_max=max(erro_direto_p_hist) if erro_direto_p_hist else "",
            bicg_residual_media=sum(bicg_residual_hist) / len(bicg_residual_hist),
            bicg_residual_max=max(bicg_residual_hist),
            passos_nao_convergidos=sum(1 for r in bicg_residual_hist if r > bicgstab_tol),  # tol efetivo (simulation.solver.tol)
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
