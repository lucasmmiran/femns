"""Montagem vetorizada das matrizes de elemento finito para o elemento MINI (P1+bolha/P1)."""

import torch


def assemble_mini(X: torch.Tensor, Y: torch.Tensor, IEN: torch.Tensor, ne: int, npoints: int):
    """Monta as matrizes esparsas COO do elemento MINI.

    Parametros:
    - X, Y: coordenadas dos nos (inclui centroides do elemento MINI).
    - IEN: conectividade dos elementos (4 nos: 3 vertices + centroide).
    - ne: numero de elementos.
    - npoints: numero de nos "reais" (sem contar os centroides).

    Retorno:
    - K, M, Gx, Gy, Gvx, Gvy: matrizes esparsas COO no mesmo device de X/Y.
    """
    device = X.device

    vi = IEN[:, 0].to(device)
    vj = IEN[:, 1].to(device)
    vk = IEN[:, 2].to(device)

    xi, yi = X[vi], Y[vi]
    xj, yj = X[vj], Y[vj]
    xz, yz = X[vk], Y[vk]

    area = 0.5 * torch.abs(
        xi * (yj - yz) +
        xj * (yz - yi) +
        xz * (yi - yj)
    )

    bi = Y[vj] - Y[vk]
    bj = Y[vk] - Y[vi]
    bk = Y[vi] - Y[vj]

    ci = X[vk] - X[vj]
    cj = X[vi] - X[vk]
    ck = X[vj] - X[vi]

    zx = (1.0 / (4.0 * area)) * (bj**2 + bj * bk + bk**2)
    zy = (1.0 / (4.0 * area)) * (cj**2 + cj * ck + ck**2)

    kx_elem_values = torch.stack([
        (1.0 / (4.0 * area)) * bi * bi + (9 / 10) * zx, (1.0 / (4.0 * area)) * bi * bj + (9 / 10) * zx, (1.0 / (4.0 * area)) * bi * bk + (9 / 10) * zx, (-27 / 10) * zx,
        (1.0 / (4.0 * area)) * bj * bi + (9 / 10) * zx, (1.0 / (4.0 * area)) * bj * bj + (9 / 10) * zx, (1.0 / (4.0 * area)) * bj * bk + (9 / 10) * zx, (-27 / 10) * zx,
        (1.0 / (4.0 * area)) * bk * bi + (9 / 10) * zx, (1.0 / (4.0 * area)) * bk * bj + (9 / 10) * zx, (1.0 / (4.0 * area)) * bk * bk + (9 / 10) * zx, (-27 / 10) * zx,
        (-27 / 10) * zx, (-27 / 10) * zx, (-27 / 10) * zx, (81 / 10) * zx], dim=1)

    ky_elem_values = torch.stack([
        (1.0 / (4.0 * area)) * ci * ci + (9 / 10) * zy, (1.0 / (4.0 * area)) * ci * cj + (9 / 10) * zy, (1.0 / (4.0 * area)) * ci * ck + (9 / 10) * zy, (-27 / 10) * zy,
        (1.0 / (4.0 * area)) * cj * ci + (9 / 10) * zy, (1.0 / (4.0 * area)) * cj * cj + (9 / 10) * zy, (1.0 / (4.0 * area)) * cj * ck + (9 / 10) * zy, (-27 / 10) * zy,
        (1.0 / (4.0 * area)) * ck * ci + (9 / 10) * zy, (1.0 / (4.0 * area)) * ck * cj + (9 / 10) * zy, (1.0 / (4.0 * area)) * ck * ck + (9 / 10) * zy, (-27 / 10) * zy,
        (-27 / 10) * zy, (-27 / 10) * zy, (-27 / 10) * zy, (81 / 10) * zy], dim=1)

    gx_elem_values = torch.stack([
        (1.0 / 6.0) * ((9 / 20) * bi + bi), (1.0 / 6.0) * ((9 / 20) * bj + bi), (1.0 / 6.0) * ((9 / 20) * bk + bi),
        (1.0 / 6.0) * ((9 / 20) * bi + bj), (1.0 / 6.0) * ((9 / 20) * bj + bj), (1.0 / 6.0) * ((9 / 20) * bk + bj),
        (1.0 / 6.0) * ((9 / 20) * bi + bk), (1.0 / 6.0) * ((9 / 20) * bj + bk), (1.0 / 6.0) * ((9 / 20) * bk + bk),
        -(9 / 40) * bi, -(9 / 40) * bj, -(9 / 40) * bk], dim=1)

    gy_elem_values = torch.stack([
        (1.0 / 6.0) * ((9 / 20) * ci + ci), (1.0 / 6.0) * ((9 / 20) * cj + ci), (1.0 / 6.0) * ((9 / 20) * ck + ci),
        (1.0 / 6.0) * ((9 / 20) * ci + cj), (1.0 / 6.0) * ((9 / 20) * cj + cj), (1.0 / 6.0) * ((9 / 20) * ck + cj),
        (1.0 / 6.0) * ((9 / 20) * ci + ck), (1.0 / 6.0) * ((9 / 20) * cj + ck), (1.0 / 6.0) * ((9 / 20) * ck + ck),
        -(9 / 40) * ci, -(9 / 40) * cj, -(9 / 40) * ck], dim=1)

    gvx_elem_values = torch.stack([
        (1.0 / 6.0) * ((11 / 20) * bi + (9 / 20) * bi), (1.0 / 6.0) * ((11 / 20) * bj + (9 / 20) * bi), (1.0 / 6.0) * ((11 / 20) * bk + (9 / 20) * bi), -(9 / 40) * bi,
        (1.0 / 6.0) * ((11 / 20) * bi + (9 / 20) * bj), (1.0 / 6.0) * ((11 / 20) * bj + (9 / 20) * bj), (1.0 / 6.0) * ((11 / 20) * bk + (9 / 20) * bj), -(9 / 40) * bj,
        (1.0 / 6.0) * ((11 / 20) * bi + (9 / 20) * bk), (1.0 / 6.0) * ((11 / 20) * bj + (9 / 20) * bk), (1.0 / 6.0) * ((11 / 20) * bk + (9 / 20) * bk), -(9 / 40) * bk,
        (9 / 40) * bi, (9 / 40) * bj, (9 / 40) * bk, torch.zeros_like(bi)], dim=1)

    gvy_elem_values = torch.stack([
        (1.0 / 6.0) * ((11 / 20) * ci + (9 / 20) * ci), (1.0 / 6.0) * ((11 / 20) * cj + (9 / 20) * ci), (1.0 / 6.0) * ((11 / 20) * ck + (9 / 20) * ci), -(9 / 40) * ci,
        (1.0 / 6.0) * ((11 / 20) * ci + (9 / 20) * cj), (1.0 / 6.0) * ((11 / 20) * cj + (9 / 20) * cj), (1.0 / 6.0) * ((11 / 20) * ck + (9 / 20) * cj), -(9 / 40) * cj,
        (1.0 / 6.0) * ((11 / 20) * ci + (9 / 20) * ck), (1.0 / 6.0) * ((11 / 20) * cj + (9 / 20) * ck), (1.0 / 6.0) * ((11 / 20) * ck + (9 / 20) * ck), -(9 / 40) * ck,
        (9 / 40) * ci, (9 / 40) * cj, (9 / 40) * ck, torch.zeros_like(ci)], dim=1)

    m_elem_values = (area / 840.0).unsqueeze(1) * torch.tensor(
        [83, 13, 13, 45,
         13, 83, 13, 45,
         13, 13, 83, 45,
         45, 45, 45, 243],
        dtype=area.dtype, device=area.device).unsqueeze(0)

    K_values = (kx_elem_values + ky_elem_values).flatten()

    # 4x4 (16 entradas por elemento)
    row_idx_4 = IEN[:, :4].repeat_interleave(4, dim=1)
    col_idx_4 = IEN[:, :4].repeat(1, 4)
    indices_4x4 = torch.stack([row_idx_4, col_idx_4], dim=0).reshape(2, -1)

    # 4x3 (12 entradas por elemento)
    row_idx_3 = IEN[:, :4].repeat_interleave(3, dim=1)
    col_idx_3 = IEN[:, :3].repeat(1, 4)
    indices_4x3 = torch.stack([row_idx_3, col_idx_3], dim=0).reshape(2, -1)

    K = torch.sparse_coo_tensor(indices_4x4, K_values, size=(npoints + ne, npoints + ne), dtype=torch.float64).to(device)
    M = torch.sparse_coo_tensor(indices_4x4, m_elem_values.flatten(), size=(npoints + ne, npoints + ne), dtype=torch.float64).to(device)
    Gx = torch.sparse_coo_tensor(indices_4x3, gx_elem_values.flatten(), size=(npoints + ne, npoints), dtype=torch.float64).to(device)
    Gy = torch.sparse_coo_tensor(indices_4x3, gy_elem_values.flatten(), size=(npoints + ne, npoints), dtype=torch.float64).to(device)
    Gvx = torch.sparse_coo_tensor(indices_4x4, gvx_elem_values.flatten(), size=(npoints + ne, npoints + ne), dtype=torch.float64).to(device)
    Gvy = torch.sparse_coo_tensor(indices_4x4, gvy_elem_values.flatten(), size=(npoints + ne, npoints + ne), dtype=torch.float64).to(device)

    return K, M, Gx, Gy, Gvx, Gvy

def assemble_tri6(X: torch.Tensor, Y: torch.Tensor, IEN: torch.Tensor, ne: int, npoints: int):
    """Monta as matrizes esparsas COO do elemento Tri6.

    Parametros:
    - X, Y: coordenadas dos nos (inclui centroides do elemento MINI).
    - IEN: conectividade dos elementos (6 nos: 3 vertices + 3 faces).
    - ne: numero de elementos.
    - npoints: numero de nos "reais" (sem contar os centroides).

    Retorno:
    - K, M, Gx, Gy, Gvx, Gvy: matrizes esparsas COO no mesmo device de X/Y.
    """
    device = X.device

    vi = IEN[:, 0].to(device)
    vj = IEN[:, 1].to(device)
    vk = IEN[:, 2].to(device)

    xi, yi = X[vi], Y[vi]
    xj, yj = X[vj], Y[vj]
    xz, yz = X[vk], Y[vk]

    area = 0.5 * torch.abs(
        xi * (yj - yz) +
        xj * (yz - yi) +
        xz * (yi - yj)
    )

    bi = Y[vj] - Y[vk]
    bj = Y[vk] - Y[vi]
    bk = Y[vi] - Y[vj]

    ci = X[vk] - X[vj]
    cj = X[vi] - X[vk]
    ck = X[vj] - X[vi]

    m_elem_values = (area / 180.0).unsqueeze(1)*torch.tensor([ 6.0,-1.0,-1.0, 0.0,-4.0, 0.0,
                                                              -1.0, 6.0,-1.0, 0.0, 0.0,-4.0,
                                                              -1.0,-1.0, 6.0,-4.0, 0.0, 0.0,
                                                               0.0, 0.0,-4.0,32.0,16.0,16.0,
                                                              -4.0, 0.0, 0.0,16.0,32.0,16.0,
                                                               0.0,-4.0, 0.0,16.0,16.0,32.0],
                                                            dtype=area.dtype, device=area.device).unsqueeze(0)

    m_area = (1.0 / (12.0 * area))

    kx_elem_values = torch.stack([
        m_area * 3.0 * bi * bi, m_area * -bi * bj, m_area * -bi * bk, m_area * 4 * bi * bj, torch.zeros_like(bi), m_area * 4 * bi * bk,
        m_area * - bi * bj, m_area * 3 * bj * bj, m_area * - bj * bk, m_area * 4 * bi * bj, m_area * 4 * bj * bk, torch.zeros_like(bi),
        m_area * - bi * bk, m_area * - bj * bk, m_area * 3 * bk * bk, torch.zeros_like(bi), m_area * 4 * bj * bk, m_area * 4 * bi * bk,
        m_area * 4 * bi * bj, m_area * 4 * bi * bj, torch.zeros_like(bi), m_area * 8 * (bi * bi + bi * bj + bj *bj), 
                                                         m_area * 4 * (bj * bj + bj * bk + bi * bj + 2 * bi * bk),
                                                         m_area * 4 * (bi * bi + bi * bk + bi * bj + 2 * bj * bk),
        torch.zeros_like(bi), m_area * 4 * bj * bk, m_area * 4 * bj * bk, m_area * 4 * (bi * bj + 2 * bi * bk + bj * bj + bj * bk),
                                                         m_area * 8 * (bj * bj + bj * bk + bk * bk),
                                                         m_area * 4 * (2 * bi * bj + bi * bk + bj * bk + bk * bk),
        m_area * 4 * bi * bk, torch.zeros_like(bi), m_area * 4 * bi * bk, m_area * 4 * (bi * bi + bi * bj + bi * bk + 2 * bj * bk),
                                                         m_area * 4 * (2 * bi * bj + bi * bk + bj * bk + bk * bk),
                                                         m_area * 8 * (bi * bi + bi * bk + bk * bk)], dim=1)

    ky_elem_values = torch.stack([
        m_area * 3 * ci * ci, m_area * -ci * cj, m_area * -ci * ck, m_area * 4 * ci * cj, torch.zeros_like(bi), m_area * 4 * ci * ck,
        m_area * -ci * cj, m_area * 3 * cj * cj, m_area * -cj * ck, m_area * 4 * ci * cj, m_area * 4 * cj * ck, torch.zeros_like(bi),
        m_area * -ci * ck, m_area * -cj * ck, m_area * 3 * ck * ck, torch.zeros_like(bi), m_area * 4 * cj * ck, m_area * 4 * ci * ck,
        m_area * 4 * ci * cj, m_area * 4 * ci * cj, torch.zeros_like(bi), m_area * 8 * (ci * ci + ci * cj + cj *cj),
                                                         m_area * 4 * (cj * cj + cj * ck + ci * cj + 2 * ci * ck),
                                                         m_area * 4 * (ci * ci + ci * ck + ci * cj + 2 * cj * ck),
        torch.zeros_like(bi), m_area * 4 * cj * ck, m_area * 4 * cj * ck, m_area * 4 * (ci * cj + 2 * ci * ck + cj * cj + cj * ck),
                                                         m_area * 8 * (cj * cj + cj * ck + ck * ck),
                                                         m_area * 4 * (2 * ci * cj + ci * ck + cj * ck + ck * ck),
        m_area * 4 * ci * ck, torch.zeros_like(bi), m_area * 4 * ci * ck, m_area * 4 * (ci * ci + ci * cj + ci * ck + 2 * cj * ck),
                                                         m_area * 4 * (2 * ci * cj + ci * ck + cj * ck + ck * ck),
                                                         m_area * 8 * (ci * ci + ci * ck + ck * ck)], dim=1)

    gx_elem_values = torch.stack([
        (1.0 / 6.0) * bi         ,          torch.zeros_like(bi), torch.zeros_like(bi),
        torch.zeros_like(bi)     ,              (1.0 / 6.0) * bj, torch.zeros_like(bi),
        torch.zeros_like(bi)     ,          torch.zeros_like(bi), (1.0 / 6.0) * bk,
        (1.0 / 6.0) * (bi + 2 * bj), (1.0 / 6.0) * (2 * bi + bj), (1.0 / 6.0) *bi + bj,
        (1.0 / 6.0) * (bj + bk)    , (1.0 / 6.0) * (bj + 2 * bk), (1.0 / 6.0) * 2*bj + bk,
        (1.0 / 6.0) * (bi + 2 * bk), (1.0 / 6.0) * (bi + bk)    , (1.0 / 6.0) * 2*bi + bk], dim=1)
    

    gy_elem_values = torch.stack([
        (1.0 / 6.0) * ci         ,          torch.zeros_like(ci), torch.zeros_like(ci),
        torch.zeros_like(ci)     ,              (1.0 / 6.0) * cj, torch.zeros_like(ci),
        torch.zeros_like(ci)     ,          torch.zeros_like(ci), (1.0 / 6.0) * ck,
        (1.0 / 6.0) * (ci + 2 * cj), (1.0 / 6.0) * (2 * ci + cj), (1.0 / 6.0) *ci + cj,
        (1.0 / 6.0) * (cj + ck)    , (1.0 / 6.0) * (cj + 2 * ck), (1.0 / 6.0) * 2*cj + ck,
        (1.0 / 6.0) * (ci + 2 * ck), (1.0 / 6.0) * (ci + ck)    , (1.0 / 6.0) * 2*ci + ck], dim=1)
    
    gvx_elem_values = torch.stack([
        (1.0/30.0) * 2 * bi, (1.0/30.0) * -bj, (1.0/30.0) * -bk, (1.0/30.0) * (-bi + 2 * bj), (1.0/30.0) * -(bj + bk), (1.0/30.0) * (-bi + 2 * bk),
        (1.0/30.0) * -bi, (1.0/30.0) * 2 * bj, (1.0/30.0) * -bk, (1.0/30.0) * (2 * bi - bj), (1.0/30.0) * (-bj + 2 * bk), (1.0/30.0) * -(bi + bk),
        (1.0/30.0) * -bi, (1.0/30.0) * -bj, (1.0/30.0) * 2 * bk, (1.0/30.0) * -(bi + bj), (1.0/30.0) * (2 * bj-bk), (1.0/30.0) * (2 * bi - bk),
        (1.0/30.0) * 3 * bi, (1.0/30.0) * 3 * bj, (1.0/30.0) * -bk, (1.0/30.0) * 8 * (bi + bj), (1.0/30.0) * 4 * (bj + 2 * bk), (1.0/30.0) * 4 * (bi + 2 * bk),
        (1.0/30.0) * -bi, (1.0/30.0) * 3 * bj, (1.0/30.0) * 3 * bk, (1.0/30.0) * 4 * (2 * bi + bj), (1.0/30.0) * 8 * (bj + bk), (1.0/30.0) * 4 * (2 * bi + bk),
        (1.0/30.0) * 3 * bi, (1.0/30.0) * -bj, (1.0/30.0) * 3 * bk, (1.0/30.0) * 4 * (bi + 2 * bj), (1.0/30.0) * 4 * (2 * bj + bk), (1.0/30.0) * 8 * (bi + bk),
        ], dim=1)

    gvy_elem_values = torch.stack([
        (1.0/30.0) * 2 * ci, (1.0/30.0) * -cj, (1.0/30.0) * -ck, (1.0/30.0) * (-ci + 2 * cj), (1.0/30.0) * -(cj+ck), (1.0/30.0) * (-ci + 2 * ck),
        (1.0/30.0) * -ci, (1.0/30.0) * 2 * cj, (1.0/30.0) * -ck, (1.0/30.0) * (2 * ci - cj), (1.0/30.0) * (-cj + 2 * ck), (1.0/30.0) * -(ci + ck),
        (1.0/30.0) * -ci, (1.0/30.0) * -cj, (1.0/30.0) * 2 * ck, (1.0/30.0) * -(ci + cj), (1.0/30.0) * (2 * cj - ck), (1.0/30.0) * (2 * ci - ck),
        (1.0/30.0) * 3 * ci, (1.0/30.0) * 3 * cj, (1.0/30.0) * -ck, (1.0/30.0) * 8 * (ci + cj), (1.0/30.0) * 4 * (cj + 2 * ck), (1.0/30.0) * 4 * (ci + 2 * ck),
        (1.0/30.0) * -ci, (1.0/30.0) * 3 * cj, (1.0/30.0) * 3 * ck, (1.0/30.0) * 4 * (2 * ci + cj), (1.0/30.0) * 8 * (cj + ck), (1.0/30.0) * 4 * (2 * ci + ck),
        (1.0/30.0) * 3 * ci, (1.0/30.0) * -cj, (1.0/30.0) * 3 * ck, (1.0/30.0) * 4 * (ci + 2 * cj), (1.0/30.0) * 4 * (2 * cj + ck), (1.0/30.0) * 8 * (ci + ck)
        ], dim=1)

    K_values = (kx_elem_values + ky_elem_values).flatten()

    # 6x6 (36 entradas por elemento)
    row_idx_4 = IEN[:, :4].repeat_interleave(4, dim=1)
    col_idx_4 = IEN[:, :4].repeat(1, 4)
    indices_4x4 = torch.stack([row_idx_4, col_idx_4], dim=0).reshape(2, -1)

    # 4x3 (12 entradas por elemento)
    row_idx_3 = IEN[:, :4].repeat_interleave(3, dim=1)
    col_idx_3 = IEN[:, :3].repeat(1, 4)
    indices_4x3 = torch.stack([row_idx_3, col_idx_3], dim=0).reshape(2, -1)

    K = torch.sparse_coo_tensor(indices_4x4, K_values, size=(npoints + ne, npoints + ne), dtype=torch.float64).to(device)
    M = torch.sparse_coo_tensor(indices_4x4, m_elem_values.flatten(), size=(npoints + ne, npoints + ne), dtype=torch.float64).to(device)
    Gx = torch.sparse_coo_tensor(indices_4x3, gx_elem_values.flatten(), size=(npoints + ne, npoints), dtype=torch.float64).to(device)
    Gy = torch.sparse_coo_tensor(indices_4x3, gy_elem_values.flatten(), size=(npoints + ne, npoints), dtype=torch.float64).to(device)
    Gvx = torch.sparse_coo_tensor(indices_4x4, gvx_elem_values.flatten(), size=(npoints + ne, npoints + ne), dtype=torch.float64).to(device)
    Gvy = torch.sparse_coo_tensor(indices_4x4, gvy_elem_values.flatten(), size=(npoints + ne, npoints + ne), dtype=torch.float64).to(device)
    Dx 
    Dy

    return K, M, Gx, Gy, Gvx, Gvy, Dx, Dy

