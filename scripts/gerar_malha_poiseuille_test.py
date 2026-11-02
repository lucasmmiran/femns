#!/usr/bin/env python3
"""CLI: gera `meshes/Poiseuille_test.msh`, a malha do teste de regressao do Poiseuille.

Canal [0, 7] x [0, 1], triangulos estruturados (`nx` x `ny` quadrados, cada um
partido em 2 pela diagonal, alternada em xadrez para a malha nao ter direcao
preferencial e ser simetrica em torno de y = 0,5). Contornos (physical
groups): `inlet` (x = 0), `outlet` (x = 7), `down` (y = 0), `upper` (y = 1).

Gerada sem gmsh (nao e' dependencia do projeto): o arquivo sai no formato
MSH 2.2 ascii, o mesmo das outras malhas de `meshes/`, e e' deterministico --
rodar de novo com os mesmos argumentos reproduz o arquivo byte a byte.

Uso: python scripts/gerar_malha_poiseuille_test.py [--nx 140 --ny 20]
"""

import argparse

COMPRIMENTO, ALTURA = 7.0, 1.0
# tag de cada physical group (dimensao 1: arestas; 2: superficie)
GRUPOS = {"inlet": 1, "down": 2, "outlet": 3, "upper": 4}
TAG_SUPERFICIE = 5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=140, help="divisoes em x (padrao 140: h = 0,05)")
    parser.add_argument("--ny", type=int, default=20, help="divisoes em y (padrao 20: h = 0,05)")
    parser.add_argument("--saida", default="meshes/Poiseuille_test.msh")
    return parser.parse_args()


def gerar(nx: int, ny: int):
    """`(pontos, triangulos, arestas)`: arestas como `(grupo, a, b)`; indices a partir de 1."""
    def no(i, j):  # no (i, j) da grade, i em x e j em y
        return j * (nx + 1) + i + 1

    pontos = [(COMPRIMENTO * i / nx, ALTURA * j / ny) for j in range(ny + 1) for i in range(nx + 1)]

    triangulos = []  # todos em sentido anti-horario
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = no(i, j), no(i + 1, j), no(i + 1, j + 1), no(i, j + 1)
            if (i + j) % 2 == 0:  # diagonal a-c
                triangulos += [(a, b, c), (a, c, d)]
            else:                 # diagonal b-d
                triangulos += [(a, b, d), (b, c, d)]

    arestas = [("down", no(i, 0), no(i + 1, 0)) for i in range(nx)]
    arestas += [("outlet", no(nx, j), no(nx, j + 1)) for j in range(ny)]
    arestas += [("upper", no(i + 1, ny), no(i, ny)) for i in reversed(range(nx))]
    arestas += [("inlet", no(0, j + 1), no(0, j)) for j in reversed(range(ny))]
    return pontos, triangulos, arestas


def escrever_msh(caminho: str, pontos, triangulos, arestas):
    linhas = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(GRUPOS) + 1)]
    linhas += [f'1 {tag} "{nome}"' for nome, tag in GRUPOS.items()]
    linhas += [f'2 {TAG_SUPERFICIE} "surface"', "$EndPhysicalNames", "$Nodes", str(len(pontos))]
    linhas += [f"{k} {x!r} {y!r} 0" for k, (x, y) in enumerate(pontos, start=1)]
    linhas += ["$EndNodes", "$Elements", str(len(arestas) + len(triangulos))]
    k = 0
    for nome, a, b in arestas:  # tipo 1 = linha de 2 nos; tags: physical, entidade geometrica
        k += 1
        linhas.append(f"{k} 1 2 {GRUPOS[nome]} {GRUPOS[nome]} {a} {b}")
    for a, b, c in triangulos:  # tipo 2 = triangulo de 3 nos
        k += 1
        linhas.append(f"{k} 2 2 {TAG_SUPERFICIE} 1 {a} {b} {c}")
    linhas.append("$EndElements")
    with open(caminho, "w") as f:
        f.write("\n".join(linhas) + "\n")


def main():
    args = parse_args()
    pontos, triangulos, arestas = gerar(args.nx, args.ny)
    escrever_msh(args.saida, pontos, triangulos, arestas)
    print(f"{args.saida}: {len(pontos)} nos, {len(triangulos)} triangulos, {len(arestas)} arestas de contorno")


if __name__ == "__main__":
    main()
