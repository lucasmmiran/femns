import numpy as np
import torch

from femns.mesh import elem_mini, montar_EToE, montar_node_to_elem
from femns.semi_lagrangian_tri import SemiLagrangianMini


def quadrado_dois_triangulos():
    """Quadrado unitario dividido em 2 triangulos: 0=(0,0) 1=(1,0) 2=(1,1) 3=(0,1)."""
    X = np.array([0.0, 1.0, 1.0, 0.0])
    Y = np.array([0.0, 0.0, 1.0, 1.0])
    IEN = np.array([[0, 1, 2], [0, 2, 3]])
    return X, Y, IEN


def malha_grade(n: int = 4):
    """Grade n x n de quadrados unitarios (cada um em 2 triangulos, CCW).

    Da nos genuinamente interiores (que nao tocam o contorno) -- necessario
    para o teste de reproducao polinomial: nos de contorno saem do dominio
    sob qualquer velocidade uniforme nao nula, e nao servem pra validar a
    interpolacao interior.
    """
    xs, ys = np.meshgrid(np.arange(n + 1, dtype=float), np.arange(n + 1, dtype=float), indexing="ij")
    X = xs.ravel()
    Y = ys.ravel()

    def idx(i, j):
        return i * (n + 1) + j

    elems = []
    for i in range(n):
        for j in range(n):
            v00, v10, v11, v01 = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            elems.append([v00, v10, v11])
            elems.append([v00, v11, v01])

    IEN = np.array(elems)
    return X, Y, IEN


def montar_sl(X: np.ndarray, Y: np.ndarray, IEN: np.ndarray, npoints: int):
    EToE, _ = montar_EToE(IEN)
    node_to_elem = montar_node_to_elem(IEN, npoints)
    IEN_full, X_full, Y_full = elem_mini(IEN, X, Y)
    ne = IEN.shape[0]

    X_t = torch.from_numpy(X_full)
    Y_t = torch.from_numpy(Y_full)

    sl = SemiLagrangianMini(IEN_full, EToE, X_t, Y_t, node_to_elem, npoints, ne)
    return sl, X_full, Y_full, ne


def nnz_por_linha(conv: torch.Tensor) -> torch.Tensor:
    return (conv.to_dense() != 0).sum(dim=1)


def test_reproducao_polinomial_exata_campo_afim():
    """Protocolo de validacao do CLAUDE_sl.md: campo afim deve ser reproduzido a ~1e-15."""
    X, Y, IEN = malha_grade(4)
    npoints = len(X)
    sl, X_full, Y_full, ne = montar_sl(X, Y, IEN, npoints)
    n_total = npoints + ne

    dt = 0.01
    vx_val, vy_val = 0.3, -0.2
    vx = torch.full((n_total,), vx_val, dtype=torch.float64)
    vy = torch.full((n_total,), vy_val, dtype=torch.float64)
    sl.compute(vx, vy, dt)  # so pra montar sl.conv com velocidade uniforme

    a, b, c = 1.7, 2.3, -0.9
    phi = torch.from_numpy(a + b * X_full + c * Y_full)
    phi_d = torch.mm(sl.conv, phi.unsqueeze(1)).squeeze(1)

    xd = X_full - dt * vx_val
    yd = Y_full - dt * vy_val
    phi_analitico = torch.from_numpy(a + b * xd + c * yd)

    dentro = nnz_por_linha(sl.conv) == sl.NEN
    assert dentro.sum() > 0  # a grade precisa realmente ter nos interiores no teste

    assert torch.allclose(phi_d[dentro], phi_analitico[dentro], atol=1e-12)


def test_reproducao_polinomial_falha_para_campo_quadratico():
    """Teste negativo obrigatorio: sem ele, o positivo acima nao prova nada."""
    X, Y, IEN = malha_grade(4)
    npoints = len(X)
    sl, X_full, Y_full, ne = montar_sl(X, Y, IEN, npoints)
    n_total = npoints + ne

    dt = 0.01
    vx_val, vy_val = 0.3, -0.2
    vx = torch.full((n_total,), vx_val, dtype=torch.float64)
    vy = torch.full((n_total,), vy_val, dtype=torch.float64)
    sl.compute(vx, vy, dt)

    phi = torch.from_numpy(X_full**2)
    phi_d = torch.mm(sl.conv, phi.unsqueeze(1)).squeeze(1)

    xd = X_full - dt * vx_val
    phi_analitico = torch.from_numpy(xd**2)

    dentro = nnz_por_linha(sl.conv) == sl.NEN
    erro = (phi_d[dentro] - phi_analitico[dentro]).abs().max().item()

    assert erro > 1e-4  # ordem de 1e-3 esperada -- deve, sim, falhar


def test_particao_da_unidade():
    X, Y, IEN = malha_grade(3)
    npoints = len(X)
    sl, X_full, Y_full, ne = montar_sl(X, Y, IEN, npoints)
    n_total = npoints + ne

    vx = torch.full((n_total,), 0.4, dtype=torch.float64)
    vy = torch.full((n_total,), 0.1, dtype=torch.float64)
    sl.compute(vx, vy, dt=0.05)

    ones = torch.ones(n_total, dtype=torch.float64)
    soma = torch.mm(sl.conv, ones.unsqueeze(1)).squeeze(1)

    assert torch.allclose(soma, ones, atol=1e-12)


def test_contagem_status_velocidade_nula():
    """Com dt=0, o pe da caracteristica e o proprio no -- todo mundo fica 'dentro'."""
    X, Y, IEN = quadrado_dois_triangulos()
    npoints = len(X)
    sl, _, _, ne = montar_sl(X, Y, IEN, npoints)
    n_total = npoints + ne

    vx = torch.zeros(n_total, dtype=torch.float64)
    vy = torch.zeros(n_total, dtype=torch.float64)
    sl.compute(vx, vy, dt=0.0)

    assert sl.status == {"dentro": n_total, "fronteira": 0, "nao_convergido": 0}


def test_linha_de_fronteira_no_que_sai_do_dominio():
    """No cujo backtrace sai do dominio tem exatamente 2 entradas na linha, somando 1.

    Usa a grade (nao o quadrado de 2 triangulos) e velocidade diagonal de
    proposito: um no de vertice de contorno com deslocamento alinhado aos
    eixos numa grade regular tende a cair exatamente sobre outro no da
    malha (peso 0/1, caso degenerado do caso degenerado -- nao exercita o
    blend fracionario que este teste quer validar). Um no interior com
    deslocamento diagonal garante uma interceptacao generica.
    """
    X, Y, IEN = malha_grade(4)
    npoints = len(X)
    sl, X_full, _, ne = montar_sl(X, Y, IEN, npoints)
    n_total = npoints + ne

    no = 1 * 5 + 2  # (i=1, j=2): interior, uma linha dentro do contorno esquerdo (i=0)
    assert X_full[no] == 1.0

    vx = torch.zeros(n_total, dtype=torch.float64)
    vy = torch.zeros(n_total, dtype=torch.float64)
    vx[no], vy[no] = 5.0, 1.3  # forte e diagonal o bastante pra sair por x=0 fora de um vertice
    sl.compute(vx, vy, dt=1.0)

    linha = sl.conv.to_dense()[no]
    entradas = linha[linha != 0]

    assert entradas.numel() == 2
    assert torch.allclose(entradas.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-12)
    assert torch.all(entradas >= -1e-12) and torch.all(entradas <= 1.0 + 1e-12)
    assert sl.status["fronteira"] >= 1


def test_contagem_status_soma_bate_com_total_de_nos():
    """Numa malha maior com deslocamento nao trivial, dentro+fronteira+nao_convergido = total de nos."""
    X, Y, IEN = malha_grade(4)
    npoints = len(X)
    sl, _, _, ne = montar_sl(X, Y, IEN, npoints)
    n_total = npoints + ne

    vx = torch.full((n_total,), 0.3, dtype=torch.float64)
    vy = torch.full((n_total,), -0.2, dtype=torch.float64)
    sl.compute(vx, vy, dt=0.01)

    soma_status = sum(sl.status.values())
    assert soma_status == n_total
    assert sl.status["dentro"] > 0
    assert sl.status["fronteira"] > 0
