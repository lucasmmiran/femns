"""Geometrias pequenas compartilhadas entre arquivos de teste.

Importadas como `from geometrias import ...` (o pytest poe `tests/` no
`sys.path`). So entra aqui o que mais de um arquivo usa -- geometria que
um unico teste precisa fica no proprio arquivo.
"""

import numpy as np
import torch


def quadrado_dois_triangulos():
    """Quadrado unitario dividido em 2 triangulos: 0=(0,0) 1=(1,0) 2=(1,1) 3=(0,1).

    So vertices (numpy), orientacao anti-horaria. Os dois elementos
    compartilham a diagonal (0,2).
    """
    X = np.array([0.0, 1.0, 1.0, 0.0])
    Y = np.array([0.0, 0.0, 1.0, 1.0])
    IEN = np.array([[0, 1, 2], [0, 2, 3]])
    return X, Y, IEN


def triangulo_mini_referencia():
    """Triangulo retangulo de referencia (0,0),(1,0),(0,1) + centroide, elemento MINI (torch).

    No 3 e' o centroide (1/3, 1/3). `ne=1`, `npoints=3`.
    """
    X = torch.tensor([0.0, 1.0, 0.0, 1.0 / 3.0], dtype=torch.float64)
    Y = torch.tensor([0.0, 0.0, 1.0, 1.0 / 3.0], dtype=torch.float64)
    IEN = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    return X, Y, IEN


def triangulo_tri6_referencia():
    """Triangulo retangulo de referencia (0,0),(1,0),(0,1) + pontos medios, elemento Tri6 (torch).

    Nos 3, 4, 5 sao os pontos medios das arestas (0,1), (1,2), (2,0) --
    convencao de `mesh.elem_tri6`. `npoints=3`, `nnodes=6`.
    """
    X = torch.tensor([0.0, 1.0, 0.0, 0.5, 0.5, 0.0], dtype=torch.float64)
    Y = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.5, 0.5], dtype=torch.float64)
    IEN = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    return X, Y, IEN
