import numpy as np

from femns.webui.meshes import contar_arestas, list_meshes, mesh_geometry


def test_list_meshes_le_malhas_reais_do_projeto():
    """Malhas reais em `meshes/` (mesmas de configs/*.yaml) -- ver tests/test_boundary.py
    para os nomes de contorno esperados de cada uma.
    """
    malhas = {m["name"]: m for m in list_meshes("meshes")}

    assert set(malhas) == {"degrau.msh", "lid.msh", "poiseuille.msh"}
    assert set(malhas["poiseuille.msh"]["boundary_names"]) == {"inlet", "outlet", "top", "bottom"}
    assert set(malhas["degrau.msh"]["boundary_names"]) == {"inlet", "outlet", "top", "bottom"}
    assert set(malhas["lid.msh"]["boundary_names"]) == {"left", "right", "top", "bottom", "outlet"}
    assert all(m["npoints"] > 0 for m in malhas.values())


def test_list_meshes_diretorio_vazio_retorna_lista_vazia(tmp_path):
    assert list_meshes(str(tmp_path)) == []


def test_mesh_geometry_malha_real():
    geo = mesh_geometry("meshes/poiseuille.msh")

    assert len(geo["points"]) > 0
    assert all(len(p) == 2 for p in geo["points"])
    assert len(geo["triangles"]) > 0
    assert all(len(t) == 3 for t in geo["triangles"])
    assert set(geo["boundaries"]) == {"inlet", "outlet", "top", "bottom"}
    # cada segmento de contorno referencia indices validos de `points`
    npoints = len(geo["points"])
    for segmentos in geo["boundaries"].values():
        assert len(segmentos) > 0
        for i, j in segmentos:
            assert 0 <= i < npoints and 0 <= j < npoints


def test_contar_arestas_dois_triangulos_com_aresta_compartilhada():
    assert contar_arestas(np.array([[0, 1, 2], [1, 3, 2]])) == 5
    assert contar_arestas(np.zeros((0, 3), dtype=int)) == 0


def test_list_meshes_conta_elementos_e_arestas():
    """Euler numa triangulacao simplesmente conexa: V - A + E = 1."""
    malha = next(m for m in list_meshes("meshes") if m["name"] == "poiseuille.msh")
    assert malha["nelements"] > 0
    assert malha["npoints"] - malha["nedges"] + malha["nelements"] == 1
