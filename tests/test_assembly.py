import torch

from femns.assembly import assemble_mini, assemble_tri6


def test_assemble_mini_triangulo_referencia():
    """Triangulo retangulo (0,0),(1,0),(0,1), area=0.5, checado contra valores analiticos."""
    X = torch.tensor([0.0, 1.0, 0.0, 1.0 / 3.0], dtype=torch.float64)
    Y = torch.tensor([0.0, 0.0, 1.0, 1.0 / 3.0], dtype=torch.float64)
    IEN = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    K, M, Gx, Gy, Gvx, Gvy = assemble_mini(X, Y, IEN, ne=1, npoints=3)

    K_dense = K.to_dense()
    M_dense = M.to_dense()

    # Bloco P1 padrao (sem a correcao da bolha) para este triangulo e conhecido:
    # [[1,-0.5,-0.5],[-0.5,0.5,0],[-0.5,0,0.5]]. A correcao da bolha soma
    # 0.9 (= 9/10*(zx+zy), com zx=zy=0.5) a cada entrada do bloco de vertices.
    K_esperado = torch.tensor([
        [1.9, 0.4, 0.4, -2.7],
        [0.4, 1.4, 0.9, -2.7],
        [0.4, 0.9, 1.4, -2.7],
        [-2.7, -2.7, -2.7, 8.1],
    ], dtype=torch.float64)

    assert torch.allclose(K_dense, K_esperado, atol=1e-10)

    # M = (area/840) * [[83,13,13,45]x3, [45,45,45,243]], area=0.5
    M_esperado = (0.5 / 840.0) * torch.tensor([
        [83, 13, 13, 45],
        [13, 83, 13, 45],
        [13, 13, 83, 45],
        [45, 45, 45, 243],
    ], dtype=torch.float64)

    assert torch.allclose(M_dense, M_esperado, atol=1e-10)

    # Matrizes de rigidez/massa devem ser simetricas
    assert torch.allclose(K_dense, K_dense.T, atol=1e-10)
    assert torch.allclose(M_dense, M_dense.T, atol=1e-10)

    assert Gx.shape == (4, 3)
    assert Gy.shape == (4, 3)
    # valores de Gvx/Gvy: ver test_assemble_mini_gvx_gvy_valores


def test_assemble_mini_gvx_gvy_valores():
    """Gvx/Gvy do MINI conferidos entrada a entrada (nao so shape).

    Sao as matrizes do termo convectivo, montadas e usadas em TODA iteracao
    do caminho eulerian (`solver.time_step`, `vg = vx_diag@Gvx + vy_diag@Gvy`)
    -- um erro de sinal ou de termo aqui nao quebra nenhum outro teste.

    Convencao (confirmada simbolicamente com sympy, base NODAL do MINI --
    `Ni = li - 9*li*lj*lk`, `Nb = 27*li*lj*lk`, ver CLAUDE.local.md 2026-07-12):
        Gvx[i][j] = INT( Ni * dNj/dx )   (derivada na funcao de forma)
    Triangulo de referencia (0,0),(1,0),(0,1), area 1/2.
    """
    X = torch.tensor([0.0, 1.0, 0.0, 1.0 / 3.0], dtype=torch.float64)
    Y = torch.tensor([0.0, 0.0, 1.0, 1.0 / 3.0], dtype=torch.float64)
    IEN = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    _, _, _, _, Gvx, Gvy = assemble_mini(X, Y, IEN, ne=1, npoints=3)
    Gvx, Gvy = Gvx.to_dense(), Gvy.to_dense()

    Gvx_esperado = (1.0 / 120.0) * torch.tensor([
        [-20,   2,  -9,  27],
        [ -2,  20,   9, -27],
        [-11,  11,   0,   0],
        [-27,  27,   0,   0],
    ], dtype=torch.float64)
    Gvy_esperado = (1.0 / 120.0) * torch.tensor([
        [-20,  -9,   2,  27],
        [-11,   0,  11,   0],
        [ -2,   9,  20, -27],
        [-27,   0,  27,   0],
    ], dtype=torch.float64)
    assert torch.allclose(Gvx, Gvx_esperado, atol=1e-12)
    assert torch.allclose(Gvy, Gvy_esperado, atol=1e-12)

    # -- invariantes estruturais (valem em qualquer malha, nao so nesta) --
    z4 = torch.zeros(4, dtype=torch.float64)
    # particao da unidade: campo de velocidade uniforme => conveccao nula.
    assert torch.allclose(Gvx.sum(dim=1), z4, atol=1e-12)
    assert torch.allclose(Gvy.sum(dim=1), z4, atol=1e-12)
    # Gvx @ x_nodal = INT(Ni): fixa a ESCALA e o SINAL global (soma de linha
    # nula sozinha nao pega Gvx -> -Gvx). Vertices 11/120, bolha 27/120.
    int_Ni = torch.tensor([11, 11, 11, 27], dtype=torch.float64) / 120.0
    assert torch.allclose(Gvx @ X, int_Ni, atol=1e-12)
    assert torch.allclose(Gvy @ Y, int_Ni, atol=1e-12)
    # Gvx so envolve as derivadas em x (bi); Gvx @ y_nodal tem que zerar.
    assert torch.allclose(Gvx @ Y, z4, atol=1e-12)
    assert torch.allclose(Gvy @ X, z4, atol=1e-12)


def test_assemble_tri6_triangulo_referencia():
    """Triangulo retangulo (0,0),(1,0),(0,1), area=0.5, nos de aresta nos pontos
    medios (v4=aresta(v1,v2), v5=aresta(v2,v3), v6=aresta(v3,v1)). Valores
    esperados obtidos por integracao simbolica exata (sympy) das funcoes de
    forma P2 (velocidade) e P1 (pressao) e conferidos contra a saida da funcao.
    """
    X = torch.tensor([0.0, 1.0, 0.0, 0.5, 0.5, 0.0], dtype=torch.float64)
    Y = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.5, 0.5], dtype=torch.float64)
    IEN = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)

    K, M, Gx, Gy, Gvx, Gvy = assemble_tri6(X, Y, IEN, npoints=3, nnodes=6)

    K_esperado = (1.0 / 6.0) * torch.tensor([
        [6, 1, 1, -4, 0, -4],
        [1, 3, 0, -4, 0, 0],
        [1, 0, 3, 0, 0, -4],
        [-4, -4, 0, 16, -8, 0],
        [0, 0, 0, -8, 16, -8],
        [-4, 0, -4, 0, -8, 16],
    ], dtype=torch.float64)
    assert torch.allclose(K.to_dense(), K_esperado, atol=1e-10)

    M_esperado = (0.5 / 180.0) * torch.tensor([
        [6, -1, -1, 0, -4, 0],
        [-1, 6, -1, 0, 0, -4],
        [-1, -1, 6, -4, 0, 0],
        [0, 0, -4, 32, 16, 16],
        [-4, 0, 0, 16, 32, 16],
        [0, -4, 0, 16, 16, 32],
    ], dtype=torch.float64)
    assert torch.allclose(M.to_dense(), M_esperado, atol=1e-10)

    # Rigidez/massa devem ser simetricas; K deve zerar campo de velocidade constante.
    assert torch.allclose(K.to_dense(), K.to_dense().T, atol=1e-10)
    assert torch.allclose(M.to_dense(), M.to_dense().T, atol=1e-10)
    assert torch.allclose(K.to_dense().sum(dim=1), torch.zeros(6, dtype=torch.float64), atol=1e-10)

    # Gx/Gy: gradiente P2 COMPLETO, INT(phi_j dN_i/dx) -- `gxele`/`gyele` da
    # referencia, nao a variante "slip" (ver docstring de assemble_tri6).
    Gx_esperado = (1.0 / 6.0) * torch.tensor([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
        [1, -1, 0],
        [1, 1, 2],
        [-1, -1, -2],
    ], dtype=torch.float64)
    assert torch.allclose(Gx.to_dense(), Gx_esperado, atol=1e-10)

    Gy_esperado = (1.0 / 6.0) * torch.tensor([
        [-1, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
        [-1, -2, -1],
        [1, 2, 1],
        [1, 0, -1],
    ], dtype=torch.float64)
    assert torch.allclose(Gy.to_dense(), Gy_esperado, atol=1e-10)

    # Uma pressao constante nao pode gerar forca liquida: soma de cada coluna = 0.
    assert torch.allclose(Gx.to_dense().sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(Gy.to_dense().sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-10)
    # valores de Gvx/Gvy: ver test_assemble_tri6_gvx_gvy_valores


def test_assemble_tri6_gvx_gvy_valores():
    """Gvx/Gvy do Tri6 conferidos entrada a entrada (nao so shape).

    Mesma motivacao de `test_assemble_mini_gvx_gvy_valores` (termo convectivo
    usado em toda iteracao eulerian) e mesma forma fraca -- `Gvx[i][j] =
    INT(Ni dNj/dx)` com base P2 (`Ni = li(2li-1)`, `N_ij = 4 li lj`),
    conferida simbolicamente e contra `gvxele` de `referencia/ref tri6/
    assembly-tri6.py`. Triangulo de referencia com nos de aresta nos pontos
    medios (v4=aresta(v1,v2), v5=aresta(v2,v3), v6=aresta(v3,v1)).
    """
    X = torch.tensor([0.0, 1.0, 0.0, 0.5, 0.5, 0.0], dtype=torch.float64)
    Y = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.5, 0.5], dtype=torch.float64)
    IEN = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)

    _, _, _, _, Gvx, Gvy = assemble_tri6(X, Y, IEN, npoints=3, nnodes=6)
    Gvx, Gvy = Gvx.to_dense(), Gvy.to_dense()

    Gvx_esperado = (1.0 / 30.0) * torch.tensor([
        [-2, -1,  0,  3, -1,  1],
        [ 1,  2,  0, -3, -1,  1],
        [ 1, -1,  0,  0,  2, -2],
        [-3,  3,  0,  0,  4, -4],
        [ 1,  3,  0, -4,  8, -8],
        [-3, -1,  0,  4,  8, -8],
    ], dtype=torch.float64)
    Gvy_esperado = (1.0 / 30.0) * torch.tensor([
        [-2,  0, -1,  1, -1,  3],
        [ 1,  0, -1, -2,  2,  0],
        [ 1,  0,  2,  1, -1, -3],
        [-3,  0, -1, -8,  8,  4],
        [ 1,  0,  3, -8,  8, -4],
        [-3,  0,  3, -4,  4,  0],
    ], dtype=torch.float64)
    assert torch.allclose(Gvx, Gvx_esperado, atol=1e-12)
    assert torch.allclose(Gvy, Gvy_esperado, atol=1e-12)

    # -- invariantes estruturais --
    z6 = torch.zeros(6, dtype=torch.float64)
    assert torch.allclose(Gvx.sum(dim=1), z6, atol=1e-12)   # particao da unidade
    assert torch.allclose(Gvy.sum(dim=1), z6, atol=1e-12)
    # INT(Ni): 0 nos vertices P2, 1/6 nos nos de aresta -- fixa escala e sinal.
    int_Ni = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float64) / 6.0
    assert torch.allclose(Gvx @ X, int_Ni, atol=1e-12)
    assert torch.allclose(Gvy @ Y, int_Ni, atol=1e-12)
    assert torch.allclose(Gvx @ Y, z6, atol=1e-12)
    assert torch.allclose(Gvy @ X, z6, atol=1e-12)


def _scatter_add(local: torch.Tensor, linhas: list[int], colunas: list[int], shape: tuple[int, int]) -> torch.Tensor:
    """Soma uma matriz local `len(linhas) x len(colunas)` numa global `shape`,
    no laco explicito -- a referencia "lenta" contra a qual conferir a
    montagem vetorizada (`repeat_interleave`/`repeat` + acumulacao COO).
    """
    G = torch.zeros(shape, dtype=torch.float64)
    for a, ga in enumerate(linhas):
        for b, gb in enumerate(colunas):
            G[ga, gb] += local[a, b]
    return G


def test_assemble_mini_dois_elementos_acumula_no_no_compartilhado():
    """2 triangulos partilhando a aresta (1)-(2): a montagem vetorizada tem
    que acumular no vertice compartilhado exatamente o que o laco elemento a
    elemento acumula. Cobre o que os testes de 1 triangulo nao alcancam --
    os indices COO (`repeat_interleave`/`repeat`) e o scatter-add.
    """
    # verts 0=(0,0) 1=(1,0) 2=(0,1) 3=(1,1); centroides: no 4 (elem 0), no 5 (elem 1)
    X = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0], dtype=torch.float64)
    Y = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0 / 3.0, 2.0 / 3.0], dtype=torch.float64)
    IEN = torch.tensor([[0, 1, 2, 4], [1, 3, 2, 5]], dtype=torch.long)
    npoints, ne = 4, 2

    matrizes = assemble_mini(X, Y, IEN, ne=ne, npoints=npoints)
    nomes = ["K", "M", "Gx", "Gy", "Gvx", "Gvy"]
    n = npoints + ne

    for nome, glob in zip(nomes, matrizes):
        largura = npoints if nome in ("Gx", "Gy") else n
        ref = torch.zeros((n, largura), dtype=torch.float64)
        for e in range(ne):
            i, j, k, _ = IEN[e].tolist()
            vel = [i, j, k, npoints + e]
            xl = X[[i, j, k, npoints + e]]
            yl = Y[[i, j, k, npoints + e]]
            locais = dict(zip(nomes, (m.to_dense() for m in assemble_mini(
                xl, yl, torch.tensor([[0, 1, 2, 3]]), ne=1, npoints=3))))
            colunas = vel[:3] if nome in ("Gx", "Gy") else vel
            ref += _scatter_add(locais[nome], vel, colunas, (n, largura))
        assert torch.allclose(glob.to_dense(), ref, atol=1e-12), f"{nome} diverge do laco elemento a elemento"

    # o vertice 1 e o vertice 2 pertencem aos 2 elementos: a diagonal de K
    # ali tem que ser a soma das duas contribuicoes locais, > que a de um
    # vertice de canto (0 ou 3), que so aparece num elemento.
    Kd = matrizes[0].to_dense()
    assert Kd[1, 1] > Kd[0, 0]
    assert Kd[2, 2] > Kd[3, 3]


def test_assemble_tri6_dois_elementos_acumula_no_no_compartilhado():
    """Mesma ideia para o Tri6: 2 triangulos partilham a aresta (1)-(2),
    logo partilham os vertices 1 e 2 E o no de aresta do meio (no 5). No Tri6
    os nos de aresta sao deduplicados entre vizinhos (`mesh.elem_tri6`), entao
    esse no recebe contribuicao dos dois elementos -- caso que o MINI (nos
    extras por elemento) nao exercita.
    """
    # verts 0..3; nos de aresta: 4=(0,1) 5=(1,2) 6=(2,0) 7=(1,3) 8=(3,2)
    X = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.0, 1.0, 0.5], dtype=torch.float64)
    Y = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.5, 1.0], dtype=torch.float64)
    IEN = torch.tensor([[0, 1, 2, 4, 5, 6], [1, 3, 2, 7, 8, 5]], dtype=torch.long)
    npoints, nnodes = 4, 9

    matrizes = assemble_tri6(X, Y, IEN, npoints=npoints, nnodes=nnodes)
    nomes = ["K", "M", "Gx", "Gy", "Gvx", "Gvy"]

    for nome, glob in zip(nomes, matrizes):
        largura = npoints if nome in ("Gx", "Gy") else nnodes
        ref = torch.zeros((nnodes, largura), dtype=torch.float64)
        for e in range(2):
            vel = IEN[e].tolist()
            press = IEN[e][:3].tolist()
            locais = dict(zip(nomes, (m.to_dense() for m in assemble_tri6(
                X[IEN[e]], Y[IEN[e]], torch.tensor([[0, 1, 2, 3, 4, 5]]), npoints=3, nnodes=6))))
            colunas = press if nome in ("Gx", "Gy") else vel
            ref += _scatter_add(locais[nome], vel, colunas, (nnodes, largura))
        assert torch.allclose(glob.to_dense(), ref, atol=1e-12), f"{nome} diverge do laco elemento a elemento"

    # no de aresta 5 (meio da aresta compartilhada) aparece nos 2 elementos;
    # nos 4/6/7/8 so num -- a diagonal de K no 5 tem que ser a maior.
    Kd = matrizes[0].to_dense()
    diag_arestas = {i: Kd[i, i].item() for i in (4, 5, 6, 7, 8)}
    assert diag_arestas[5] == max(diag_arestas.values())
