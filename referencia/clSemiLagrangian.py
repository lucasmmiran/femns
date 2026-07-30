#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Gustavo R. Anjos
# Email: gustavo.rabello@coppe.ufrj.br
# Date: 2025-06-23
# File: clSemiLagrangian.py

"""
Description: Semi-Lagrangian Interpolation for Finite Element Methods

This module implements a flexible, extensible framework for semi-Lagrangian
interpolation and particle tracking in the context of finite element methods
(FEM) for advection-dominated problems. The core concept is to backtrack nodes
or integration points along velocity fields to determine departure points, then
compute interpolation weights using appropriate shape functions for various
finite element types: triangles, quadrilaterals, tetrahedrons.

The design is object-oriented, featuring abstract base classes for 2D and 3D
elements (`SemiLagrangian2D`, `SemiLagrangian3D`) and concrete implementations
for specific element types (e.g., tri3, tri4 -mini-, tri6, tri7, Quad4, Quad5
-mini-, Quad8, Quad9). Common procedures—such as element search
(`testElement`), non-linear coordinate mapping, and blending on boundaries—are
encapsulated and reused across the hierarchy.

The module supports parallel processing via Python's multiprocessing tools,
allowing efficient construction of large sparse interpolation matrices. The
design allows easy extension to new element types or higher polynomial orders,
making this code suitable for research and production environments involving
unstructured meshes and high-order FEM for 2D and 3D advection-dominated flows.

Key features:
 - General base classes for both 2D and 3D semi-Lagrangian FEM interpolation
 - Support for linear, quadratic, and bubble-enriched elements (triangles,
   quads)
 - Efficient element search and local coordinate mapping
 - Parallel assembly of sparse interpolation matrices
 - Easily extensible to new element types or spatial dimensions
"""
from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import coo_matrix
from multiprocessing import get_context, cpu_count
from functools import partial
from typing import Tuple, List

# ------------------------------------------------------------------------------
# Implementation of 2D Finite Elements (Semi-Lagrangian Methods)
#
# This codebase establishes a structured approach to implementing
# semi-Lagrangian interpolation and tracking for various 2D finite elements,
# including triangles and quadrilaterals with different interpolation orders
# (e.g., linear, quadratic, bubble-enriched). The base class,
# `SemiLagrangian2D`, defines a standard interface and essential methods (such
# as `compute`, `getDepartElem`, `jumpToElem`, and `testElement`) that are
# shared among all 2D elements. Specialized classes inherit from this base and
# implement the appropriate shape functions and element-specific logic. The
# modular structure enables seamless extension to higher-order elements or
# alternative formulations while ensuring code reuse and clarity.
# ------------------------------------------------------------------------------
def depart_chunk2D(chunk, dt, jumpToElem, X, Y, velU, velV,
                 neighborElem, Xmin, Xmax, Lx, hasPBC):
    rows, cols, vals = [], [], []
    for i in chunk:
        mele = neighborElem[i][0]
        xp = X[i] - velU[i] * dt
        yp = Y[i] - velV[i] * dt

        if hasPBC:
            if xp < Xmin:
                xp += Lx
            elif xp > Xmax:
                xp -= Lx

        phi, target_cols = jumpToElem(mele, i, xp, yp)
        rows.extend([i] * len(phi))
        cols.extend(target_cols)
        vals.extend(phi)

    return np.array(rows), np.array(cols), np.array(vals)

class SemiLagrangian2D(ABC):
    def __init__(self, IEN, X, Y,
                 neighborElem, oface,
                 velU, velV, IDleft=None):
        self.X = X
        self.Y = Y
        self.IEN = IEN
        self.neighborElem = neighborElem
        self.oface = oface
        self.velU = velU
        self.velV = velV
        self.IDleft = IDleft

        self.numNodes = len(X)
        self.numElems = len(IEN)
        self.conv = None  # Montada em compute

    def compute(self, dt, processes=None):
        """Calcula a matriz de interpolação semi-Lagrangiana"""
        self.getDepartElem(dt, processes)

    def getDepartElem(self, dt, processes=None):
        """Paraleliza a busca dos pontos de partida para todos os nós"""
        if processes is None:
            processes = max(1, cpu_count() - 1)

        Xmin, Xmax = min(self.X), max(self.X)
        Lx = Xmax - Xmin
        hasPBC = (self.IDleft is not None and len(self.IDleft) > 0)

        block = max(10_000, self.numNodes // (4 * processes))
        chunks = [range(i, min(i + block, self.numNodes))
                  for i in range(0, self.numNodes, block)]

        args_common = dict(
            dt=dt,
            jumpToElem=self.jumpToElem,
            X=self.X,
            Y=self.Y,
            velU=self.velU,
            velV=self.velV,
            neighborElem=self.neighborElem,
            Xmin=Xmin,
            Xmax=Xmax,
            Lx=Lx,
            hasPBC=hasPBC
        )

        with get_context("fork").Pool(processes) as pool:
            func = partial(depart_chunk2D, **args_common)
            results = pool.map(func, chunks)

        rows, cols, vals = map(np.concatenate, zip(*results))
        self.conv = coo_matrix((vals, (rows, cols)),
                               shape=(self.numNodes, self.numNodes)).tocsr()

    def setBC(self, idv, vbc, vd):
        """Imposição de condições de contorno Dirichlet"""
        for i in idv:
            vd[i] = vbc[i]
        return vd

    @abstractmethod
    def jumpToElem(self, destElem, iiVert,
                   R2X, R2Y) -> Tuple[List[float],List[int]]:
        """
        Busca recursiva/interpolação do elemento de partida.
        Deve retornar (phi, target_cols)
        """
        pass

    @abstractmethod
    def testElement(self, mele, xp, yp):
        """
        Testa se o ponto (xp, yp) está dentro do elemento 'mele'
        """
        pass

    def computeIntercept(self, i, R2X, R2Y, ib1, ib2):
        R1X, R1Y = self.X[i], self.Y[i]
        B1X, B2X = self.X[ib1], self.X[ib2]
        B1Y, B2Y = self.Y[ib1], self.Y[ib2]
        a1, b1, c1 = B1X-B2X, R1X-R2X, R1X-B2X
        a2, b2, c2 = B1Y-B2Y, R1Y-R2Y, R1Y-B2Y
        det = (a1*b2) - (a2*b1)
        detx = (c1*b2) - (c2*b1)
        x1 = detx/det
        return x1, 1.0-x1

class Triangle(SemiLagrangian2D):
    def __init__(self, IEN, X, Y,
                 neighborElem, oface,
                 velU, velV, IDleft=None):
        super().__init__(IEN, X, Y,
                         neighborElem, oface,
                         velU, velV, IDleft)
        self.numVerts = np.max(IEN[:, :3]) + 1

    def testElement(self, mele, xp, yp):
        """
        Testa se (xp, yp) está no triângulo 'mele' e calcula l1, l2, l3.
        Retorna True se está dentro (ou suficientemente próximo).
        """
        v1, v2, v3 = self.IEN[mele][0:3]
        EPSlocal = 1e-04

        A = (1.0/2.0) * ( self.X[v2]*self.Y[v3] +
                          self.X[v1]*self.Y[v2] +
                          self.Y[v1]*self.X[v3] -
                          self.X[v1]*self.Y[v3] -
                          self.Y[v2]*self.X[v3] -
                          self.Y[v1]*self.X[v2] )

        A1 = (1.0/2.0) * ( self.X[v2]*self.Y[v3] +
                           xp*self.Y[v2] +
                           yp*self.X[v3] -
                           xp*self.Y[v3] -
                           self.Y[v2]*self.X[v3] -
                           yp*self.X[v2] )

        A2 = (1.0/2.0) * ( xp*self.Y[v3] +
                           self.X[v1]*yp +
                           self.Y[v1]*self.X[v3] -
                           self.X[v1]*self.Y[v3] -
                           yp*self.X[v3] -
                           self.Y[v1]*xp )

        self.l1 = A1/A
        self.l2 = A2/A
        self.l3 = 1.0 - self.l2 - self.l1

        return ( (self.l1>=0.0-EPSlocal) and (self.l1<=1.0+EPSlocal) and
                 (self.l2>=0.0-EPSlocal) and (self.l2<=1.0+EPSlocal) and
                 (self.l3>=0.0-EPSlocal) and (self.l3<=1.0+EPSlocal) )

class Tri3(Triangle):
    def __init__(self, IEN, X, Y,
                 neighborElem, oface,
                 velU, velV, IDleft=None):
        super().__init__(IEN, X, Y,
                         neighborElem, oface,
                         velU, velV, IDleft)
    def jumpToElem(self, destElem, iiVert, R2X, R2Y):
        v1, v2, v3 = self.IEN[destElem][0:3]
        if self.testElement(destElem, R2X, R2Y):
            phi = [self.l1, self.l2, self.l3]
            return phi, [v1, v2, v3]
        else:
            if (self.l1 <= self.l2) and (self.l1 <= self.l3):
                vjump = 0
                ib1 = v2
                ib2 = v3
            elif (self.l2 <= self.l1) and (self.l2 <= self.l3):
                vjump = 1
                ib1 = v1
                ib2 = v3
            else:
                vjump = 2
                ib1 = v1
                ib2 = v2

            if self.oface[destElem, vjump] != -1:
                return self.jumpToElem(self.oface[destElem, vjump], iiVert, R2X, R2Y)
            else:
                Bl1, Bl2 = self.computeIntercept(iiVert, R2X, R2Y, ib1, ib2)
                return [Bl1, Bl2], [ib1, ib2]

class Tri4(Triangle):
    def __init__(self, IEN, X, Y,
                 neighborElem, oface,
                 velU, velV, IDleft=None):
        super().__init__(IEN, X, Y,
                         neighborElem, oface,
                         velU, velV, IDleft)
    def jumpToElem(self, destElem, iiVert, R2X, R2Y):
        v1, v2, v3, v4 = self.IEN[destElem][0:4]
        if self.testElement(destElem, R2X, R2Y):
            # funcoes de forma MINI
            N1 = self.l1-9.0*self.l1*self.l2*self.l3
            N2 = self.l2-9.0*self.l1*self.l2*self.l3
            N3 = self.l3-9.0*self.l1*self.l2*self.l3
            N4 = 27*self.l1*self.l2*self.l3
            phi = [N1,N2,N3,N4]
            return phi, [v1, v2, v3, v4]
        else:
            if (self.l1 <= self.l2) and (self.l1 <= self.l3):
                vjump = 0
                ib1 = v2
                ib2 = v3
            elif (self.l2 <= self.l1) and (self.l2 <= self.l3):
                vjump = 1
                ib1 = v1
                ib2 = v3
            else:
                vjump = 2
                ib1 = v1
                ib2 = v2

            if self.oface[destElem, vjump] != -1:
                return self.jumpToElem(self.oface[destElem, vjump], iiVert, R2X, R2Y)
            else:
                Bl1, Bl2 = self.computeIntercept(iiVert, R2X, R2Y, ib1, ib2)
                return [Bl1, Bl2], [ib1, ib2]

class Tri6(Triangle):
    def __init__(self, IEN, X, Y,
                 neighborElem, oface,
                 velU, velV, IDleft=None):
        super().__init__(IEN, X, Y,
                         neighborElem, oface,
                         velU, velV, IDleft)
    def jumpToElem(self, destElem, iiVert, R2X, R2Y):
        v1, v2, v3, v4, v5, v6 = self.IEN[destElem][0:6]  # v4, v5, v6 são os nós de aresta (meio)

        if self.testElement(destElem, R2X, R2Y):
            # Funções de forma P2 (quadráticas) para triângulo
            l1, l2, l3 = self.l1, self.l2, self.l3
            N1 = l1 * (2 * l1 - 1)
            N2 = l2 * (2 * l2 - 1)
            N3 = l3 * (2 * l3 - 1)
            N4 = 4 * l1 * l2
            N5 = 4 * l2 * l3
            N6 = 4 * l1 * l3
            phi = [N1, N2, N3, N4, N5, N6]
            return phi, [v1, v2, v3, v4, v5, v6]
        else:
            # A lógica de busca recursiva segue igual à Linear/Mini
            if (self.l1 <= self.l2) and (self.l1 <= self.l3):
                vjump = 0
                ib1 = v2
                ib2 = v3
                ib3 = v5  # meio da aresta entre v2-v3
            elif (self.l2 <= self.l1) and (self.l2 <= self.l3):
                vjump = 1
                ib1 = v3
                ib2 = v1
                ib3 = v6  # meio da aresta entre v3-v1
            else:
                vjump = 2
                ib1 = v1
                ib2 = v2
                ib3 = v4  # meio da aresta entre v1-v2

            if self.oface[destElem, vjump] != -1:
                return self.jumpToElem(self.oface[destElem, vjump], iiVert, R2X, R2Y)
            else:
                Bl1, Bl2 = self.computeIntercept(iiVert, R2X, R2Y, ib1, ib2)
                # Recalcula funções de forma da aresta (P2 de linha) — local
                BN1 = Bl1 * (2 * Bl1 - 1)
                BN2 = Bl2 * (2 * Bl2 - 1)
                BN3 = 4 * Bl1 * Bl2
                return [BN1, BN2, BN3], [ib1, ib2, ib3]

class Tri7(Triangle):
    def __init__(self, IEN, X, Y,
                 neighborElem, oface,
                 velU, velV, IDleft=None):
        super().__init__(IEN, X, Y,
                         neighborElem, oface,
                         velU, velV, IDleft)
    def jumpToElem(self, destElem, iiVert, R2X, R2Y):
        v1, v2, v3, v4, v5, v6, v7 = self.IEN[destElem][0:7]
        if self.testElement(destElem, R2X, R2Y):
            l1, l2, l3 = self.l1, self.l2, self.l3
            # Funções de forma Tri7 (P2+bubble)
            N1 = l1 * (2.0 * l1 - 1.0) + 3.0 * l1 * l2 * l3
            N2 = l2 * (2.0 * l2 - 1.0) + 3.0 * l1 * l2 * l3
            N3 = l3 * (2.0 * l3 - 1.0) + 3.0 * l1 * l2 * l3
            N4 = 4.0 * l1 * l2 - 12.0 * l1 * l2 * l3
            N5 = 4.0 * l2 * l3 - 12.0 * l1 * l2 * l3
            N6 = 4.0 * l1 * l3 - 12.0 * l1 * l2 * l3
            N7 = 27.0 * l1 * l2 * l3
            phi = [N1, N2, N3, N4, N5, N6, N7]
            return phi, [v1, v2, v3, v4, v5, v6, v7]
        else:
            # A lógica de backtracking segue o padrão
            if (self.l1 <= self.l2) and (self.l1 <= self.l3):
                vjump = 0
                ib1 = v2
                ib2 = v3
                ib3 = v5  # meio da aresta v2-v3
            elif (self.l2 <= self.l1) and (self.l2 <= self.l3):
                vjump = 1
                ib1 = v3
                ib2 = v1
                ib3 = v6  # meio da aresta v3-v1
            else:
                vjump = 2
                ib1 = v1
                ib2 = v2
                ib3 = v4  # meio da aresta v1-v2

            if self.oface[destElem, vjump] != -1:
                return self.jumpToElem(self.oface[destElem, vjump], iiVert, R2X, R2Y)
            else:
                Bl1, Bl2 = self.computeIntercept(iiVert, R2X, R2Y, ib1, ib2)
                # Funções de forma quadráticas de linha para a aresta
                BN1 = Bl1 * (2.0 * Bl1 - 1.0)
                BN2 = Bl2 * (2.0 * Bl2 - 1.0)
                BN3 = 4.0 * Bl1 * Bl2
                return [BN1, BN2, BN3], [ib1, ib2, ib3]

class Tri10(Triangle):
    def __init__(self, IEN, X, Y,
                 neighborElem, oface,
                 velU, velV, IDleft=None):
        super().__init__(IEN, X, Y,
                         neighborElem, oface,
                         velU, velV, IDleft)
    def jumpToElem(self, destElem, iiVert, R2X, R2Y):
        v = self.IEN[destElem][0:10]  # v1...v10

        if self.testElement(destElem, R2X, R2Y):
            l1, l2, l3 = self.l1, self.l2, self.l3

            # Funções de forma P3 para triângulo (ver Hughes, Zienkiewicz, etc)
            N1  = 0.5 * (3*l1 - 1)*(3*l1 - 2)*l1
            N2  = 0.5 * (3*l2 - 1)*(3*l2 - 2)*l2
            N3  = 0.5 * (3*l3 - 1)*(3*l3 - 2)*l3
            N4  = 4.5 * l1 * l2 * (3*l1 - 1)
            N5  = 4.5 * l1 * l2 * (3*l2 - 1)
            N6  = 4.5 * l2 * l3 * (3*l2 - 1)
            N7  = 4.5 * l2 * l3 * (3*l3 - 1)
            N8  = 4.5 * l1 * l3 * (3*l3 - 1)
            N9  = 4.5 * l1 * l3 * (3*l1 - 1)
            N10 = 27.0 * l1 * l2 * l3
            phi = [N1, N2, N3, N4, N5, N6, N7, N8, N9, N10]
            return phi, v
        else:
            # Backtracking na aresta: devolve interpolação cúbica de linha (4 nós por aresta)
            if (self.l1 <= self.l2) and (self.l1 <= self.l3):
                vjump = 0
                ib1, ib2, ib3, ib4 = v[1], v[2], v[5], v[6]   # v2, v3, meio1, meio2
            elif (self.l2 <= self.l1) and (self.l2 <= self.l3):
                vjump = 1
                ib1, ib2, ib3, ib4 = v[2], v[0], v[7], v[8]   # v3, v1, meio1, meio2
            else:
                vjump = 2
                ib1, ib2, ib3, ib4 = v[0], v[1], v[3], v[4]   # v1, v2, meio1, meio2

            if self.oface[destElem, vjump] != -1:
                return self.jumpToElem(self.oface[destElem, vjump], iiVert, R2X, R2Y)
            else:
                Bl1, Bl2 = self.computeIntercept(iiVert, R2X, R2Y, ib1, ib2)
                # Funções de forma cúbicas de linha para a aresta (4 nós)
                BN1 = 0.5 * (1 - Bl1) * (2 - 3*Bl1) * (1 - 3*Bl1)
                BN2 = 0.5 * Bl1 * (2 - 3*Bl1) * (1 - 3*Bl1)
                BN3 = (9.0 / 2.0) * Bl1 * (1 - Bl1) * (2 - 3*Bl1)
                BN4 = (9.0 / 2.0) * Bl1 * (1 - Bl1) * (3*Bl1 - 1)
                return [BN1, BN2, BN3, BN4], [ib1, ib2, ib3, ib4]


class Quadrilateral(SemiLagrangian2D):
    def __init__(self, IEN, X, Y,
                 neighborElem, oface,
                 velU, velV, IDleft=None):
        super().__init__(IEN, X, Y,
                         neighborElem, oface,
                         velU, velV, IDleft)
        self.numVerts = np.max(IEN[:, :4]) + 1

        # Armazenamento auxiliar distances d1,d2,d3 e d4
        self.d1 = 0
        self.d2 = 0
        self.d3 = 0
        self.d4 = 0

    def compute(self, dt, processes=None):
        self.getDepartElem(dt, processes)


    def setBC(self, idv, vbc, vd):
        for i in idv:
            vd[i] = vbc[i]
        return vd

    # solve non-linear sys eqs:
    #   a0+a1*xi+a2*eta+a3*xi*eta = xp
    #   b0+b1*xi+b2*eta+b3*xi*eta = yp
    # obs.: testes indicaram que 5 iteracoes sao suficientes para convergencia
    def newtonBilinear(self,_xi,_eta,_alpha,_beta,xp,yp):
     # jacobian's matrix coeficients
     a = (_alpha[1]+_alpha[3]*_eta)
     b = (_alpha[2]+_alpha[3]*_xi)
     c = (_beta[1]+_beta[3]*_eta)
     d = (_beta[2]+_beta[3]*_xi)

     # jacobian
     detJ = a*d - b*c

     xi  = _xi  - \
           (d/detJ) * \
           (_alpha[0]+_alpha[1]*_xi+_alpha[2]*_eta+_alpha[3]*_xi*_eta-xp) - \
           (-b/detJ) * \
           (_beta[0]+_beta[1]*_xi+_beta[2]*_eta+_beta[3]*_xi*_eta-yp)

     eta = _eta - \
           (-c/detJ) * \
           (_alpha[0]+_alpha[1]*_xi+_alpha[2]*_eta+_alpha[3]*_xi*_eta-xp) - \
           (a/detJ) * \
           (_beta[0]+_beta[1]*_xi+_beta[2]*_eta+_beta[3]*_xi*_eta-yp)

     return (xi,eta)

    @abstractmethod
    def jumpToElem(self, destElem, iiVert,
                   R2X, R2Y) -> Tuple[List[float],List[int]]:
        """Sobrescreva: retorna (phi, v)"""
        pass

    def testElement(self, mele, xp, yp):
        v1, v2, v3, v4 = self.IEN[mele][:4]
        EPSlocal = 1e-04
        # This method computes the distance of departure point xp,yp to the
        # straight line defined by the edges v1-v2, v2-v3, v3-v4 and v4-v1.
        #
        #       [vec] x P[v1]p  (area of the quadrilateral formed by vec and P)
        #   d = --------------
        #            [vec]
        #
        #          v4             v3
        #           o ---------- o
        #          /            /            o
        #         /            /              P (xp,yp)
        #        /            /
        #       o ---------- o
        #     v1             v2
        #
        # If d1,d2,d3 and d4 >= 0.0, thus P is inside quad v1-v2-v3-v4
        # If some d is negative (d(v2-v3) is negative), thus v2-v3 is used to
        # jump to the opposite element
        # Checagem via áreas (distância do ponto para cada lado)
        # v1->v2
        x1vec = self.X[v2]-self.X[v1]
        y1vec = self.Y[v2]-self.Y[v1]
        l1vec = np.sqrt(x1vec**2 + y1vec**2)
        # vector p1: v1-->p
        px1vec = xp-self.X[v1]
        py1vec = yp-self.Y[v1]

        # v2->v3
        x2vec = self.X[v3]-self.X[v2]
        y2vec = self.Y[v3]-self.Y[v2]
        l2vec = np.sqrt(x2vec**2 + y2vec**2)
        # vector p2: v2-->p
        px2vec = xp-self.X[v2]
        py2vec = yp-self.Y[v2]

        # v3->v4
        x3vec = self.X[v4]-self.X[v3]
        y3vec = self.Y[v4]-self.Y[v3]
        l3vec = np.sqrt(x3vec**2 + y3vec**2)
        # vector p3: v3-->p
        px3vec = xp-self.X[v3]
        py3vec = yp-self.Y[v3]

        # v4->v1
        x4vec = self.X[v1]-self.X[v4]
        y4vec = self.Y[v1]-self.Y[v4]
        l4vec = np.sqrt(x4vec**2 + y4vec**2)
        # vector p4: v4-->p
        px4vec = xp-self.X[v4]
        py4vec = yp-self.Y[v4]

        # area paralelogram
        area1 = x1vec*py1vec - y1vec*px1vec
        area2 = x2vec*py2vec - y2vec*px2vec
        area3 = x3vec*py3vec - y3vec*px3vec
        area4 = x4vec*py4vec - y4vec*px4vec

        # distances
        self.d1 = area1/l1vec
        self.d2 = area2/l2vec
        self.d3 = area3/l3vec
        self.d4 = area4/l4vec

        return (self.d1>=0.0-EPSlocal and
                self.d2>=0.0-EPSlocal and
                self.d3>=0.0-EPSlocal and
                self.d4>=0.0-EPSlocal)


class Quad4(Quadrilateral):
    def __init__(self, IEN, X, Y,
                 neighborElem, oface,
                 velU, velV, IDleft=None):
        super().__init__(IEN, X, Y,
                         neighborElem, oface,
                         velU, velV, IDleft)

    def jumpToElem(self, destElem, iiVert, R2X, R2Y):
        v1, v2, v3, v4 = self.IEN[destElem][0:4]

        # Se o ponto está dentro do elemento
        #
        #    3         2
        #     o ----- o
        #     |       |    xi,eta = [-1,+1]
        #     |       |
        #     o ----- o
        #    0         1
        #
        # computing phi1...phi9, for arbitrary quadrilateral

        # Se o ponto está dentro do elemento
        if self.testElement(destElem, R2X, R2Y):
            Xelem = [self.X[v1], self.X[v2], self.X[v3], self.X[v4]]
            Yelem = [self.Y[v1], self.Y[v2], self.Y[v3], self.Y[v4]]

            Ainv = np.array([[ 0.25,  0.25,  0.25,  0.25],
                             [-0.25,  0.25,  0.25, -0.25],
                             [-0.25, -0.25,  0.25,  0.25],
                             [ 0.25, -0.25,  0.25, -0.25]])

            a = Ainv @ Xelem
            b = Ainv @ Yelem

            xi, eta = self.newtonBilinear(0.0, 0.0, a, b, R2X, R2Y)
            for _ in range(10):
                xiOld, etaOld = xi, eta
                xi, eta = self.newtonBilinear(xi, eta, a, b, R2X, R2Y)
                if abs(xi - xiOld) < 1E-12 and abs(eta - etaOld) < 1E-12:
                    break

            xi = max(min(xi, 1.0), -1.0)
            eta = max(min(eta, 1.0), -1.0)

            phi1 = (1.0-xi)*(1.0-eta)/4.0
            phi2 = (1.0+xi)*(1.0-eta)/4.0
            phi3 = (1.0+xi)*(1.0+eta)/4.0
            phi4 = (1.0-xi)*(1.0+eta)/4.0
            phi = [phi1, phi2, phi3, phi4]
            return phi, [v1,v2,v3,v4]

        # Se ponto fora do elemento, tenta vizinho oposto
        else:
          # check lower distance
          distances = [self.d1, self.d2, self.d3, self.d4]
          indexMin = distances.index(min(distances))

          if indexMin == 0:
              ib1 = v1
              ib2 = v2
              vjump = self.oface[destElem,0]
          elif indexMin == 1:
              ib1 = v2
              ib2 = v3
              vjump = self.oface[destElem,1]
          elif indexMin == 2:
              ib1 = v3
              ib2 = v4
              vjump = self.oface[destElem,2]
          else: # indexMin == 3:
              ib1 = v4
              ib2 = v1
              vjump = self.oface[destElem,3]

          if vjump != -1:
            return self.jumpToElem(vjump,iiVert, R2X, R2Y)
          else:
            Bl1, Bl2 = self.computeIntercept(iiVert, R2X, R2Y, ib1, ib2)
            return [Bl1, Bl2], [ib1, ib2]

class Quad5(Quadrilateral):
    def __init__(self, IEN, X, Y,
                 neighborElem, oface,
                 velU, velV, IDleft=None):
        super().__init__(IEN, X, Y,
                         neighborElem, oface,
                         velU, velV, IDleft)

    def jumpToElem(self, destElem, iiVert, R2X, R2Y):
        v1, v2, v3, v4, v5 = self.IEN[destElem][0:5]

        # Se o ponto está dentro do elemento
        #
        #    3         2
        #     o ----- o
        #     |       |    xi,eta = [-1,+1]
        #     |       |
        #     o ----- o
        #    0         1
        #
        # computing phi1...phi9, for arbitrary quadrilateral

        # Se o ponto está dentro do elemento
        if self.testElement(destElem, R2X, R2Y):
            Xelem = [self.X[v1], self.X[v2], self.X[v3], self.X[v4]]
            Yelem = [self.Y[v1], self.Y[v2], self.Y[v3], self.Y[v4]]

            Ainv = np.array([[ 0.25,  0.25,  0.25,  0.25],
                             [-0.25,  0.25,  0.25, -0.25],
                             [-0.25, -0.25,  0.25,  0.25],
                             [ 0.25, -0.25,  0.25, -0.25]])

            a = Ainv @ Xelem
            b = Ainv @ Yelem

            xi, eta = self.newtonBilinear(0.0, 0.0, a, b, R2X, R2Y)
            for _ in range(10):
                xiOld, etaOld = xi, eta
                xi, eta = self.newtonBilinear(xi, eta, a, b, R2X, R2Y)
                if abs(xi - xiOld) < 1E-12 and abs(eta - etaOld) < 1E-12:
                    break

            xi = max(min(xi, 1.0), -1.0)
            eta = max(min(eta, 1.0), -1.0)

            phi1 = (1.0/4.0)*(1.0-xi)*(1.0-eta)-\
                   (1.0/4.0)*(1.0-xi*xi)*(1.0-eta*eta)
            phi2 = (1.0/4.0)*(1.0+xi)*(1.0-eta)-\
                   (1.0/4.0)*(1.0-xi*xi)*(1.0-eta*eta)
            phi3 = (1.0/4.0)*(1.0+xi)*(1.0+eta)-\
                   (1.0/4.0)*(1.0-xi*xi)*(1.0-eta*eta)
            phi4 = (1.0/4.0)*(1.0-xi)*(1.0+eta)-\
                   (1.0/4.0)*(1.0-xi*xi)*(1.0-eta*eta)
            phi5 = (1.0-eta*eta)*(1.0-xi*xi)
            phi = [phi1, phi2, phi3, phi4, phi5]
            return phi, [v1,v2,v3,v4,v5]

        # Se ponto fora do elemento, tenta vizinho oposto
        else:
          # check lower distance
          distances = [self.d1, self.d2, self.d3, self.d4]
          indexMin = distances.index(min(distances))

          if indexMin == 0:
              ib1 = v1
              ib2 = v2
              vjump = self.oface[destElem,0]
          elif indexMin == 1:
              ib1 = v2
              ib2 = v3
              vjump = self.oface[destElem,1]
          elif indexMin == 2:
              ib1 = v3
              ib2 = v4
              vjump = self.oface[destElem,2]
          else: # indexMin == 3:
              ib1 = v4
              ib2 = v1
              vjump = self.oface[destElem,3]

          if vjump != -1:
            return self.jumpToElem(vjump,iiVert, R2X, R2Y)
          else:
            Bl1, Bl2 = self.computeIntercept(iiVert, R2X, R2Y, ib1, ib2)
            return [Bl1, Bl2], [ib1, ib2]

class Quad8(Quadrilateral):
    def __init__(self, IEN, X, Y,
                 neighborElem, oface,
                 velU, velV, IDleft=None):
        super().__init__(IEN, X, Y,
                         neighborElem, oface,
                         velU, velV, IDleft)

    def jumpToElem(self, destElem, iiVert, R2X, R2Y):
        v1, v2, v3, v4, v5, v6, v7, v8 = self.IEN[destElem][0:8]

        # Se o ponto está dentro do elemento
        #
        #    3         2
        #     o ----- o
        #     |       |    xi,eta = [-1,+1]
        #     |       |
        #     o ----- o
        #    0         1
        #
        # computing phi1...phi9, for arbitrary quadrilateral

        # Se o ponto está dentro do elemento
        if self.testElement(destElem, R2X, R2Y):
            Xelem = [self.X[v1], self.X[v2], self.X[v3], self.X[v4]]
            Yelem = [self.Y[v1], self.Y[v2], self.Y[v3], self.Y[v4]]

            Ainv = np.array([[ 0.25,  0.25,  0.25,  0.25],
                             [-0.25,  0.25,  0.25, -0.25],
                             [-0.25, -0.25,  0.25,  0.25],
                             [ 0.25, -0.25,  0.25, -0.25]])

            a = Ainv @ Xelem
            b = Ainv @ Yelem

            xi, eta = self.newtonBilinear(0.0, 0.0, a, b, R2X, R2Y)
            for _ in range(10):
                xiOld, etaOld = xi, eta
                xi, eta = self.newtonBilinear(xi, eta, a, b, R2X, R2Y)
                if abs(xi - xiOld) < 1E-12 and abs(eta - etaOld) < 1E-12:
                    break

            xi = max(min(xi, 1.0), -1.0)
            eta = max(min(eta, 1.0), -1.0)

            phi1 = -(1.0/4.0)*(1.0-xi)*(1.0-eta)*(1.0+xi+eta)
            phi2 = -(1.0/4.0)*(1.0+xi)*(1.0-eta)*(1.0-xi+eta)
            phi3 = -(1.0/4.0)*(1.0+xi)*(1.0+eta)*(1.0-xi-eta)
            phi4 = -(1.0/4.0)*(1.0-xi)*(1.0+eta)*(1.0+xi-eta)
            phi5 =  (1.0/2.0)*(1.0-(xi*xi))*(1.0-eta)
            phi6 =  (1.0/2.0)*(1.0+xi)*(1.0-(eta*eta))
            phi7 =  (1.0/2.0)*(1.0-(xi*xi))*(1.0+eta)
            phi8 =  (1.0/2.0)*(1.0-xi)*(1.0-(eta*eta))
            phi = [phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8]
            return phi, [v1,v2,v3,v4,v5,v6,v7,v8]

        # Se ponto fora do elemento, tenta vizinho oposto
        else:
          # check lower distance
          distances = [self.d1, self.d2, self.d3, self.d4]
          indexMin = distances.index(min(distances))

          if indexMin == 0:
              ib1 = v1
              ib2 = v2
              ib3 = v5  # nó do meio da aresta v1-v2
              vjump = self.oface[destElem, 0]
          elif indexMin == 1:
              ib1 = v2
              ib2 = v3
              ib3 = v6  # nó do meio da aresta v2-v3
              vjump = self.oface[destElem, 1]
          elif indexMin == 2:
              ib1 = v3
              ib2 = v4
              ib3 = v7  # nó do meio da aresta v3-v4
              vjump = self.oface[destElem, 2]
          else: # indexMin == 3:
              ib1 = v4
              ib2 = v1
              ib3 = v8  # nó do meio da aresta v4-v1
              vjump = self.oface[destElem, 3]

          if vjump != -1:
            return self.jumpToElem(vjump,iiVert, R2X, R2Y)
          else:
            Bl1, Bl2 = self.computeIntercept(iiVert, R2X, R2Y, ib1, ib2)
            return [Bl1, Bl2], [ib1, ib2]


class Quad9(Quadrilateral):
    def __init__(self, IEN, X, Y,
                 neighborElem, oface,
                 velU, velV, IDleft=None):
        super().__init__(IEN, X, Y,
                         neighborElem, oface,
                         velU, velV, IDleft)

    def jumpToElem(self, destElem, iiVert, R2X, R2Y):
        v1, v2, v3, v4, v5, v6, v7, v8, v9 = self.IEN[destElem][0:9]

        # Se o ponto está dentro do elemento
        #
        #    3         2
        #     o ----- o
        #     |       |    xi,eta = [-1,+1]
        #     |       |
        #     o ----- o
        #    0         1
        #
        # computing phi1...phi9, for arbitrary quadrilateral

        # Se o ponto está dentro do elemento
        if self.testElement(destElem, R2X, R2Y):
            Xelem = [self.X[v1], self.X[v2], self.X[v3], self.X[v4]]
            Yelem = [self.Y[v1], self.Y[v2], self.Y[v3], self.Y[v4]]

            Ainv = np.array([[ 0.25,  0.25,  0.25,  0.25],
                             [-0.25,  0.25,  0.25, -0.25],
                             [-0.25, -0.25,  0.25,  0.25],
                             [ 0.25, -0.25,  0.25, -0.25]])

            a = Ainv @ Xelem
            b = Ainv @ Yelem

            xi, eta = self.newtonBilinear(0.0, 0.0, a, b, R2X, R2Y)
            for _ in range(10):
                xiOld, etaOld = xi, eta
                xi, eta = self.newtonBilinear(xi, eta, a, b, R2X, R2Y)
                if abs(xi - xiOld) < 1E-12 and abs(eta - etaOld) < 1E-12:
                    break

            xi = max(min(xi, 1.0), -1.0)
            eta = max(min(eta, 1.0), -1.0)

            # shape functions in xi,eta coordinates for _mele:
            phi1 = (1.0/4.0)*(xi*xi-xi)*(eta*eta-eta)
            phi2 = (1.0/4.0)*(xi*xi+xi)*(eta*eta-eta)
            phi3 = (1.0/4.0)*(xi*xi+xi)*(eta*eta+eta)
            phi4 = (1.0/4.0)*(xi*xi-xi)*(eta*eta+eta)
            phi5 = (1.0/2.0)*(eta*eta-eta)*(1.0-xi*xi)
            phi6 = (1.0/2.0)*(xi*xi+xi)*(1.0-eta*eta)
            phi7 = (1.0/2.0)*(eta*eta+eta)*(1.0-xi*xi)
            phi8 = (1.0/2.0)*(xi*xi-xi)*(1.0-eta*eta)
            phi9 = (1.0-(xi*xi))*(1.0-(eta*eta))
            phi = [phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9]
            return phi, [v1,v2,v3,v4,v5,v6,v7,v8,v9]

        # Se ponto fora do elemento, tenta vizinho oposto
        else:
          distances = [self.d1, self.d2, self.d3, self.d4]
          indexMin = distances.index(min(distances))

          if indexMin == 0:
              ib1 = v1
              ib2 = v2
              ib3 = v5  # nó do meio da aresta v1-v2
              vjump = self.oface[destElem, 0]
          elif indexMin == 1:
              ib1 = v2
              ib2 = v3
              ib3 = v6  # nó do meio da aresta v2-v3
              vjump = self.oface[destElem, 1]
          elif indexMin == 2:
              ib1 = v3
              ib2 = v4
              ib3 = v7  # nó do meio da aresta v3-v4
              vjump = self.oface[destElem, 2]
          else: # indexMin == 3:
              ib1 = v4
              ib2 = v1
              ib3 = v8  # nó do meio da aresta v4-v1
              vjump = self.oface[destElem, 3]

          if vjump != -1:
            return self.jumpToElem(vjump, iiVert, R2X, R2Y)
          # Caso de fronteira: calcula Blending
          else:
            Bl1, Bl2 = self.computeIntercept(iiVert, R2X, R2Y, ib1, ib2)
            return [Bl1, Bl2], [ib1, ib2]

class Quad12(Quadrilateral):
    def __init__(self, IEN, X, Y,
                 neighborElem, oface,
                 velU, velV, IDleft=None):
        super().__init__(IEN, X, Y,
                         neighborElem, oface,
                         velU, velV, IDleft)

    def jumpToElem(self, destElem, iiVert, R2X, R2Y):
        v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12 = self.IEN[destElem][0:12]

        # Se o ponto está dentro do elemento
        #
        #    3         2
        #     o ----- o
        #     |       |    xi,eta = [-1,+1]
        #     |       |
        #     o ----- o
        #    0         1
        #
        # computing phi1...phi9, for arbitrary quadrilateral

        # Se o ponto está dentro do elemento
        if self.testElement(destElem, R2X, R2Y):
            Xelem = [self.X[v1], self.X[v2], self.X[v3], self.X[v4]]
            Yelem = [self.Y[v1], self.Y[v2], self.Y[v3], self.Y[v4]]

            Ainv = np.array([[ 0.25,  0.25,  0.25,  0.25],
                             [-0.25,  0.25,  0.25, -0.25],
                             [-0.25, -0.25,  0.25,  0.25],
                             [ 0.25, -0.25,  0.25, -0.25]])

            a = Ainv @ Xelem
            b = Ainv @ Yelem

            xi, eta = self.newtonBilinear(0.0, 0.0, a, b, R2X, R2Y)
            for _ in range(10):
                xiOld, etaOld = xi, eta
                xi, eta = self.newtonBilinear(xi, eta, a, b, R2X, R2Y)
                if abs(xi - xiOld) < 1E-12 and abs(eta - etaOld) < 1E-12:
                    break

            xi = max(min(xi, 1.0), -1.0)
            eta = max(min(eta, 1.0), -1.0)

            # shape functions in xi,eta coordinates for _mele:
            phi1  = (1.0/32.0)*(1.0-xi)*(1.0-eta)*(9.0*(xi*xi+eta*eta)-10.0)
            phi2  = (1.0/32.0)*(1.0+xi)*(1.0-eta)*(9.0*(xi*xi+eta*eta)-10.0)
            phi3  = (1.0/32.0)*(1.0+xi)*(1.0+eta)*(9.0*(xi*xi+eta*eta)-10.0)
            phi4  = (1.0/32.0)*(1.0-xi)*(1.0+eta)*(9.0*(xi*xi+eta*eta)-10.0)
            # bottom edge
            phi5  = (9.0/32.0)*(1.0-3.0*xi)*(1.0-eta)*(1.0-xi*xi) # xi=-1/3
            phi6  = (9.0/32.0)*(1.0+3.0*xi)*(1.0-eta)*(1.0-xi*xi) # xi=+1/3
            # right edge
            phi7  = (9.0/32.0)*(1.0+xi)*(1.0-3.0*eta)*(1.0-eta*eta)  # eta=-1/3
            phi8  = (9.0/32.0)*(1.0+xi)*(1.0+3.0*eta)*(1.0-eta*eta)  # eta=+1/3
            # top edge
            phi9  = (9.0/32.0)*(1.0+3.0*xi)*(1.0+eta)*(1.0-xi*xi) # xi=+1/3
            phi10 = (9.0/32.0)*(1.0-3.0*xi)*(1.0+eta)*(1.0-xi*xi) # xi=-1/3
            # left edge
            phi11 = (9.0/32.0)*(1.0-xi)*(1.0+3.0*eta)*(1.0-eta*eta) # eta=+1/3
            phi12 = (9.0/32.0)*(1.0-xi)*(1.0-3.0*eta)*(1.0-eta*eta) # eta=-1/3
            phi = [phi1,phi2,phi3,phi4,phi5,phi6,
                   phi7,phi8,phi9,phi10,phi11,phi12]
            return phi, [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12]

        # Se ponto fora do elemento, tenta vizinho oposto
        else:
          distances = [self.d1, self.d2, self.d3, self.d4]
          indexMin = distances.index(min(distances))

          if indexMin == 0:
              ib1 = v1
              ib2 = v2
              ib3 = v5  # 1 nó da aresta v1-v2
              ib4 = v6  # 2 nó da aresta v1-v2
              vjump = self.oface[destElem, 0]
          elif indexMin == 1:
              ib1 = v2
              ib2 = v3
              ib3 = v7  # 1 nó da aresta v2-v3
              ib4 = v8  # 2 nó da aresta v2-v3
              vjump = self.oface[destElem, 1]
          elif indexMin == 2:
              ib1 = v3
              ib2 = v4
              ib3 = v9  # 1 nó da aresta v3-v4
              ib4 = v10 # 2 nó da aresta v3-v4
              vjump = self.oface[destElem, 2]
          else: # indexMin == 3:
              ib1 = v4
              ib2 = v1
              ib3 = v11 # 1 nó da aresta v4-v1
              ib4 = v12 # 2 nó da aresta v4-v1
              vjump = self.oface[destElem, 3]

          if vjump != -1:
            return self.jumpToElem(vjump, iiVert, R2X, R2Y)
          # Caso de fronteira: calcula Blending
          else:
            Bl1, Bl2 = self.computeIntercept(iiVert, R2X, R2Y, ib1, ib2)
            return [Bl1, Bl2], [ib1, ib2]

class Quad16(Quadrilateral):
    def __init__(self, IEN, X, Y,
                 neighborElem, oface,
                 velU, velV, IDleft=None):
        super().__init__(IEN, X, Y,
                         neighborElem, oface,
                         velU, velV, IDleft)

    def jumpToElem(self, destElem, iiVert, R2X, R2Y):
        v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16 = self.IEN[destElem][0:16]

        # Se o ponto está dentro do elemento
        #
        #    3         2
        #     o ----- o
        #     |       |    xi,eta = [-1,+1]
        #     |       |
        #     o ----- o
        #    0         1
        #
        # computing phi1...phi9, for arbitrary quadrilateral

        # Se o ponto está dentro do elemento
        if self.testElement(destElem, R2X, R2Y):
            Xelem = [self.X[v1], self.X[v2], self.X[v3], self.X[v4]]
            Yelem = [self.Y[v1], self.Y[v2], self.Y[v3], self.Y[v4]]

            Ainv = np.array([[ 0.25,  0.25,  0.25,  0.25],
                             [-0.25,  0.25,  0.25, -0.25],
                             [-0.25, -0.25,  0.25,  0.25],
                             [ 0.25, -0.25,  0.25, -0.25]])

            a = Ainv @ Xelem
            b = Ainv @ Yelem

            xi, eta = self.newtonBilinear(0.0, 0.0, a, b, R2X, R2Y)
            for _ in range(10):
                xiOld, etaOld = xi, eta
                xi, eta = self.newtonBilinear(xi, eta, a, b, R2X, R2Y)
                if abs(xi - xiOld) < 1E-12 and abs(eta - etaOld) < 1E-12:
                    break

            xi = max(min(xi, 1.0), -1.0)
            eta = max(min(eta, 1.0), -1.0)

            # shape functions in xi,eta coordinates for _mele:
            # Cantos
            phi1  = ((-9*xi**3 + 9*xi**2 + xi - 1)/16) * \
                    ((-9*eta**3 + 9*eta**2 + eta - 1)/16)
            phi2  = ((xi + 1)*(3*xi - 1)*(3*xi + 1)/16) * \
                    ((-9*eta**3 + 9*eta**2 + eta - 1)/16)
            phi3  = ((xi + 1)*(3*xi - 1)*(3*xi + 1)/16) * \
                    ((eta + 1)*(3*eta - 1)*(3*eta + 1)/16)
            phi4  = ((-9*xi**3 + 9*xi**2 + xi - 1)/16) * \
                    ((eta + 1)*(3*eta - 1)*(3*eta + 1)/16)
            # Bordas
            phi5  = (9*(xi - 1)*(xi + 1)*(3*xi - 1)/16) * \
                    ((-9*eta**3 + 9*eta**2 + eta - 1)/16)
            phi6  = (-9*(xi - 1)*(xi + 1)*(3*xi + 1)/16) * \
                    ((-9*eta**3 + 9*eta**2 + eta - 1)/16)
            phi7  = ((xi + 1)*(3*xi - 1)*(3*xi + 1)/16) * \
                    (9*(eta - 1)*(eta + 1)*(3*eta - 1)/16)
            phi8  = ((xi + 1)*(3*xi - 1)*(3*xi + 1)/16) * \
                    (-9*(eta - 1)*(eta + 1)*(3*eta + 1)/16)
            phi9  = (-9*(xi - 1)*(xi + 1)*(3*xi + 1)/16) * \
                    ((eta + 1)*(3*eta - 1)*(3*eta + 1)/16)
            phi10 = (9*(xi - 1)*(xi + 1)*(3*xi - 1)/16) * \
                    ((eta + 1)*(3*eta - 1)*(3*eta + 1)/16)
            phi11 = ((-9*xi**3 + 9*xi**2 + xi - 1)/16) * \
                    (-9*(eta - 1)*(eta + 1)*(3*eta + 1)/16)
            phi12 = ((-9*xi**3 + 9*xi**2 + xi - 1)/16) * \
                    (9*(eta - 1)*(eta + 1)*(3*eta - 1)/16)
            # Internos
            phi13 = (9*(xi - 1)*(xi + 1)*(3*xi - 1)/16) * \
                    (9*(eta - 1)*(eta + 1)*(3*eta - 1)/16)
            phi14 = (-9*(xi - 1)*(xi + 1)*(3*xi + 1)/16) * \
                    (9*(eta - 1)*(eta + 1)*(3*eta - 1)/16)
            phi15 = (-9*(xi - 1)*(xi + 1)*(3*xi + 1)/16) * \
                    (-9*(eta - 1)*(eta + 1)*(3*eta + 1)/16)
            phi16 = (9*(xi - 1)*(xi + 1)*(3*xi - 1)/16) * \
                    (-9*(eta - 1)*(eta + 1)*(3*eta + 1)/16)
            phi = [phi1,phi2,phi3,phi4,phi5,phi6,
                   phi7,phi8,phi9,phi10,phi11,phi12,
                   phi13,phi14,phi15,phi16]
            return phi, [v1,v2,v3,v4,v5,v6,v7,v8,v9,
                         v10,v11,v12,v13,v14,v15,v16]

        # Se ponto fora do elemento, tenta vizinho oposto
        else:
          distances = [self.d1, self.d2, self.d3, self.d4]
          indexMin = distances.index(min(distances))

          if indexMin == 0:
              ib1 = v1
              ib2 = v2
              ib3 = v5  # 1 nó da aresta v1-v2
              ib4 = v6  # 2 nó da aresta v1-v2
              vjump = self.oface[destElem, 0]
          elif indexMin == 1:
              ib1 = v2
              ib2 = v3
              ib3 = v7  # 1 nó da aresta v2-v3
              ib4 = v8  # 2 nó da aresta v2-v3
              vjump = self.oface[destElem, 1]
          elif indexMin == 2:
              ib1 = v3
              ib2 = v4
              ib3 = v9  # 1 nó da aresta v3-v4
              ib4 = v10 # 2 nó da aresta v3-v4
              vjump = self.oface[destElem, 2]
          else: # indexMin == 3:
              ib1 = v4
              ib2 = v1
              ib3 = v11 # 1 nó da aresta v4-v1
              ib4 = v12 # 2 nó da aresta v4-v1
              vjump = self.oface[destElem, 3]

          if vjump != -1:
            return self.jumpToElem(vjump, iiVert, R2X, R2Y)
          # Caso de fronteira: calcula Blending
          else:
            Bl1, Bl2 = self.computeIntercept(iiVert, R2X, R2Y, ib1, ib2)
            return [Bl1, Bl2], [ib1, ib2]

# ------------------------------------------------------------------------------
# Extension to 3D Finite Elements
#
# The general structure adopted for 2D elements provides a natural pathway for
# extending the framework to 3D finite elements, such as tetrahedra, hexahedra,
# and prismatic elements. In a 3D context, the shape functions, local
# coordinate transformations, and element search routines are adapted to three
# spatial dimensions, but the inheritance-based design pattern remains the
# same. The implementation of a `SemiLagrangian3D` base class and its derived
# element types will follow similar principles, ensuring consistent interfaces
# and facilitating code maintenance for semi-Lagrangian tracking in 3D
# simulations.
# ------------------------------------------------------------------------------
def depart_chunk3D(chunk, dt, jumpToElem, X, Y, Z,
                   velU, velV, velW,
                   neighborElem, Xmin, Xmax, Lx, hasPBC):
    rows, cols, vals = [], [], []
    for i in chunk:
        mele = neighborElem[i][0]
        xp = X[i] - velU[i] * dt
        yp = Y[i] - velV[i] * dt
        zp = Y[i] - velW[i] * dt

        if hasPBC:
            if xp < Xmin:
                xp += Lx
            elif xp > Xmax:
                xp -= Lx

        phi, target_cols = jumpToElem(mele, i, xp, yp)
        rows.extend([i] * len(phi))
        cols.extend(target_cols)
        vals.extend(phi)

    return np.array(rows), np.array(cols), np.array(vals)

class SemiLagrangian3D(ABC):
    def __init__(self, IEN, X, Y, Z,
                 neighborElem, oface,
                 velU, velV, velW, IDleft=None):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.IEN = IEN
        self.neighborElem = neighborElem
        self.oface = oface
        self.velU = velU
        self.velV = velV
        self.velW = velW
        self.IDleft = IDleft

        self.numNodes = len(X)
        self.numElems = len(IEN)
        self.conv = None  # Montada em compute

    def compute(self, dt, processes=None):
        """Calcula a matriz de interpolação semi-Lagrangiana"""
        self.getDepartElem(dt, processes)

    def getDepartElem(self, dt, processes=None):
        """Paraleliza a busca dos pontos de partida para todos os nós"""
        if processes is None:
            processes = max(1, cpu_count() - 1)

        Xmin, Xmax = min(self.X), max(self.X)
        Lx = Xmax - Xmin
        hasPBC = (self.IDleft is not None and len(self.IDleft) > 0)

        block = max(10_000, self.numNodes // (4 * processes))
        chunks = [range(i, min(i + block, self.numNodes))
                  for i in range(0, self.numNodes, block)]

        args_common = dict(
            dt=dt,
            jumpToElem=self.jumpToElem,
            X=self.X,
            Y=self.Y,
            Z=self.Z,
            velU=self.velU,
            velV=self.velV,
            velW=self.velW,
            neighborElem=self.neighborElem,
            Xmin=Xmin,
            Xmax=Xmax,
            Lx=Lx,
            hasPBC=hasPBC
        )

        with get_context("fork").Pool(processes) as pool:
            func = partial(depart_chunk3D, **args_common)
            results = pool.map(func, chunks)

        rows, cols, vals = map(np.concatenate, zip(*results))
        self.conv = coo_matrix((vals, (rows, cols)),
                               shape=(self.numNodes, self.numNodes)).tocsr()

    def setBC(self, idv, vbc, vd):
        """Imposição de condições de contorno Dirichlet"""
        for i in idv:
            vd[i] = vbc[i]
        return vd

    @abstractmethod
    def jumpToElem(self, destElem, iiVert,
                   R2X, R2Y, R2Z) -> Tuple[List[float],List[int]]:
        """
        Busca recursiva/interpolação do elemento de partida.
        Deve retornar (phi, target_cols)
        """
        pass

    @abstractmethod
    def testElement(self, mele, xp, yp, zp):
        """
        Testa se o ponto (xp, yp) está dentro do elemento 'mele'
        """
        pass

    def computeIntercept(self, i, R2X, R2Y, R2Z, ib1, ib2, ib3):
        R1X = self.X[i]
        R1Y = self.Y[i]
        R1Z = self.Z[i]

        B1X = self.X[ib1]
        B2X = self.X[ib2]
        B3X = self.X[ib3]
        B1Y = self.Y[ib1]
        B2Y = self.Y[ib2]
        B3Y = self.Y[ib3]
        B1Z = self.Z[ib1]
        B2Z = self.Z[ib2]
        B3Z = self.Z[ib3]

        a1 = B1X-B3X; b1 = B2X-B3X; c1 = R1X-R2X; d1 = R1X-B3X;
        a2 = B1Y-B3Y; b2 = B2Y-B3Y; c2 = R1Y-R2Y; d2 = R1Y-B3Y;
        a3 = B1Z-B3Z; b3 = B2Z-B3Z; c3 = R1Z-R2Z; d3 = R1Z-B3Z;

        det  = (a1*b2*c3)+(a3*b1*c2)+(a2*b3*c1)-(a3*b2*c1)-(a1*b3*c2)-(a2*b1*c3)
        detx = (d1*b2*c3)+(d3*b1*c2)+(d2*b3*c1)-(d3*b2*c1)-(d1*b3*c2)-(d2*b1*c3)
        dety = (a1*d2*c3)+(a3*d1*c2)+(a2*d3*c1)-(a3*d2*c1)-(a1*d3*c2)-(a2*d1*c3)

        x1 = detx/det
        x2 = dety/det

        Bl1 = x1
        Bl2 = x2
        Bl3 = 1.0-Bl1-Bl2

        # return C++ equivalent Bl1, Bl2 and Bl3 (see femSIM3d)
        return Bl1,Bl2,Bl3

class Tetrahedron(SemiLagrangian3D):
    def __init__(self, IEN, X, Y, Z,
                 neighborElem, oface,
                 velU, velV, velW, IDleft=None):
        super().__init__(IEN, X, Y, Z,
                         neighborElem, oface,
                         velU, velV, velW, IDleft)
        self.numVerts = np.max(IEN[:, :4]) + 1

    def testElement(self,mele,xp,yp,zp):
        """
        Testa se (xp, yp) está no triângulo 'mele' e calcula l1, l2, l3.
        Retorna True se está dentro (ou suficientemente próximo).
        """
        v1, v2, v3, v4 = self.IEN[mele][0:4]
        EPSlocal = 1e-04

        V = (1.0/6.0) * (+1*( (self.X[v2]*self.Y[v3]*self.Z[v4])
                             +(self.Y[v2]*self.Z[v3]*self.X[v4])
                             +(self.Z[v2]*self.X[v3]*self.Y[v4])
                             -(self.Y[v2]*self.X[v3]*self.Z[v4])
                             -(self.X[v2]*self.Z[v3]*self.Y[v4])
                             -(self.Z[v2]*self.Y[v3]*self.X[v4]) )
               -self.X[v1]*( +self.Y[v3]*self.Z[v4]
                              +self.Y[v2]*self.Z[v3]
                              +self.Z[v2]*self.Y[v4]
                              -self.Y[v2]*self.Z[v4]
                              -self.Z[v3]*self.Y[v4]
                              -self.Z[v2]*self.Y[v3] )
               +self.Y[v1]*( +self.X[v3]*self.Z[v4]
                              +self.X[v2]*self.Z[v3]
                              +self.Z[v2]*self.X[v4]
                              -self.X[v2]*self.Z[v4]
                              -self.Z[v3]*self.X[v4]
                              -self.Z[v2]*self.X[v3] )
               -self.Z[v1]*( +self.X[v3]*self.Y[v4]
                              +self.X[v2]*self.Y[v3]
                              +self.Y[v2]*self.X[v4]
                              -self.X[v2]*self.Y[v4]
                              -self.Y[v3]*self.X[v4]
                              -self.Y[v2]*self.X[v3] ) )

        V1 = (1.0/6.0) * (+1*( (self.X[v2]*self.Y[v3]*self.Z[v4])
                              +(self.Y[v2]*self.Z[v3]*self.X[v4])
                              +(self.Z[v2]*self.X[v3]*self.Y[v4])
                              -(self.Y[v2]*self.X[v3]*self.Z[v4])
                              -(self.X[v2]*self.Z[v3]*self.Y[v4])
                              -(self.Z[v2]*self.Y[v3]*self.X[v4]) )
                        -xp*( +self.Y[v3]*self.Z[v4]
                               +self.Y[v2]*self.Z[v3]
                               +self.Z[v2]*self.Y[v4]
                               -self.Y[v2]*self.Z[v4]
                               -self.Z[v3]*self.Y[v4]
                               -self.Z[v2]*self.Y[v3] )
                        +yp*( +self.X[v3]*self.Z[v4]
                               +self.X[v2]*self.Z[v3]
                               +self.Z[v2]*self.X[v4]
                               -self.X[v2]*self.Z[v4]
                               -self.Z[v3]*self.X[v4]
                               -self.Z[v2]*self.X[v3] )
                        -zp*( +self.X[v3]*self.Y[v4]
                               +self.X[v2]*self.Y[v3]
                               +self.Y[v2]*self.X[v4]
                               -self.X[v2]*self.Y[v4]
                               -self.Y[v3]*self.X[v4]
                               -self.Y[v2]*self.X[v3] ) )

        V2 = (1.0/6.0) * (+1*( (xp*self.Y[v3]*self.Z[v4])
                              +(yp*self.Z[v3]*self.X[v4])
                              +(zp*self.X[v3]*self.Y[v4])
                              -(yp*self.X[v3]*self.Z[v4])
                              -(xp*self.Z[v3]*self.Y[v4])
                              -(zp*self.Y[v3]*self.X[v4]) )
                   -self.X[v1]*( +self.Y[v3]*self.Z[v4]
                               +yp*self.Z[v3]
                               +zp*self.Y[v4]
                               -yp*self.Z[v4]
                                  -self.Z[v3]*self.Y[v4]
                               -zp*self.Y[v3] )
                   +self.Y[v1]*( +self.X[v3]*self.Z[v4]
                               +xp*self.Z[v3]
                               +zp*self.X[v4]
                               -xp*self.Z[v4]
                                  -self.Z[v3]*self.X[v4]
                               -zp*self.X[v3] )
                   -self.Z[v1]*( +self.X[v3]*self.Y[v4]
                               +xp*self.Y[v3]
                               +yp*self.X[v4]
                               -xp*self.Y[v4]
                               -self.Y[v3]*self.X[v4]
                               -yp*self.X[v3] ) )

        V3 = (1.0/6.0) * (+1*( (self.X[v2]*yp*self.Z[v4])
                              +(self.Y[v2]*zp*self.X[v4])
                              +(self.Z[v2]*xp*self.Y[v4])
                              -(self.Y[v2]*xp*self.Z[v4])
                              -(self.X[v2]*zp*self.Y[v4])
                              -(self.Z[v2]*yp*self.X[v4]) )
             -self.X[v1]*( +yp*self.Z[v4]
                               +self.Y[v2]*zp
                               +self.Z[v2]*self.Y[v4]
                               -self.Y[v2]*self.Z[v4]
                               -zp*self.Y[v4]
                               -self.Z[v2]*yp )
             +self.Y[v1]*( +xp*self.Z[v4]
                               +self.X[v2]*zp
                               +self.Z[v2]*self.X[v4]
                               -self.X[v2]*self.Z[v4]
                               -zp*self.X[v4]
                               -self.Z[v2]*xp )
             -self.Z[v1]*( +xp*self.Y[v4]
                               +self.X[v2]*yp
                               +self.Y[v2]*self.X[v4]
                               -self.X[v2]*self.Y[v4]
                               -yp*self.X[v4]
                               -self.Y[v2]*xp ) );

        self.l1 = V1/V
        self.l2 = V2/V
        self.l3 = V3/V
        self.l4 = 1.0 - self.l3 - self.l2 - self.l1

        return ( (self.l1>=0.0-EPSlocal) and (self.l1<=1.0+EPSlocal) and \
                 (self.l2>=0.0-EPSlocal) and (self.l2<=1.0+EPSlocal) and \
                 (self.l3>=0.0-EPSlocal) and (self.l3<=1.0+EPSlocal) and \
                 (self.l4>=0.0-EPSlocal) and (self.l4<=1.0+EPSlocal) )

class tet3(Tetrahedron):
    def __init__(self, IEN, X, Y, Z,
                 neighborElem, oface,
                 velU, velV, velW, IDleft=None):
        super().__init__(IEN, X, Y, Z,
                         neighborElem, oface,
                         velU, velV, velW, IDleft)


    # R2X = xp, R2Y = yp, R2Z = zp (departure point)
    def jumpToElem(self,destElem,iiVert,R2X,R2Y,R2Z):
       v1 = self.IEN[destElem][0]
       v2 = self.IEN[destElem][1]
       v3 = self.IEN[destElem][2]
       v4 = self.IEN[destElem][3]

       if self.testElement(destElem,R2X,R2Y,R2Z):
        phi = [self.l1, self.l2, self.l3, self.l4]
        return phi, [v1, v2, v3, v4]
       else:
        if (self.l1<=self.l2) and (self.l1<=self.l3) and (self.l1<=self.l4):
         vjump=0
         ib1=v2
         ib2=v3
         ib3=v4
        if (self.l2<=self.l1) and (self.l2<=self.l3) and (self.l2<=self.l4):
         vjump=1
         ib1=v3
         ib2=v1
         ib3=v4
        if (self.l3<=self.l1) and (self.l3<=self.l2) and (self.l3<=self.l4):
         vjump=2
         ib1=v1
         ib2=v2
         ib3=v4
        if (self.l4<=self.l1) and (self.l4<=self.l2) and (self.l4<=self.l3):
         vjump=3
         ib1=v2
         ib2=v1
         ib3=v3

        if self.oface[destElem,vjump] != -1:
         return self.jumpToElem(self.oface[destElem,vjump],iiVert,R2X,R2Y,R2Z)
        else:
         Bl1,Bl2,Bl3 = self.computeIntercept(iiVert,R2X,R2Y,R2Z,ib1,ib2,ib3)
         return [Bl1, Bl2, Bl3], [ib1, ib2, ib3]

class tet10(Tetrahedron):
    def __init__(self, IEN, X, Y, Z,
                 neighborElem, oface,
                 velU, velV, velW, IDleft=None):
        super().__init__(IEN, X, Y, Z,
                         neighborElem, oface,
                         velU, velV, velW, IDleft)

    # R2X = xp, R2Y = yp, R2Z = zp (departure point)
    def jumpToElem(self,destElem,iiVert,R2X,R2Y,R2Z):
     v1,v2,v3,v4,v5,v6,v7,v8,v9,v10  = self.IEN[destElem]

     if self.testElement(destElem,R2X,R2Y,R2Z):
      N1  = (2*self.l1-1.0)*self.l1
      N2  = (2*self.l2-1.0)*self.l2
      N3  = (2*self.l3-1.0)*self.l3
      N4  = (2*self.l4-1.0)*self.l4
      N5  = 4*self.l1*self.l2
      N6  = 4*self.l2*self.l3
      N7  = 4*self.l1*self.l3
      N8  = 4*self.l1*self.l4
      N9  = 4*self.l2*self.l4
      N10 = 4*self.l3*self.l4
      phi = [N1, N2, N3, N4, N5, N6, N7, N8, N9, N10]
      return phi, [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]
     else:
      if (self.l1<=self.l2) and (self.l1<=self.l3) and (self.l1<=self.l4):
       vjump=0
       ib1=v2
       ib2=v3
       ib3=v4
      if (self.l2<=self.l1) and (self.l2<=self.l3) and (self.l2<=self.l4):
       vjump=1
       ib1=v3
       ib2=v1
       ib3=v4
      if (self.l3<=self.l1) and (self.l3<=self.l2) and (self.l3<=self.l4):
       vjump=2
       ib1=v1
       ib2=v2
       ib3=v4
      if (self.l4<=self.l1) and (self.l4<=self.l2) and (self.l4<=self.l3):
       vjump=3
       ib1=v2
       ib2=v1
       ib3=v3

      if self.oface[destElem,vjump] != -1:
       return self.jumpToElem(self.oface[destElem,vjump],iiVert,R2X,R2Y,R2Z)
      else:
       Bl1,Bl2,Bl3 = self.computeIntercept(iiVert,R2X,R2Y,R2Z,ib1,ib2,ib3)
       return [Bl1, Bl2, Bl3], [ib1, ib2, ib3]
