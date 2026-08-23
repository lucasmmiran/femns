from femns.webui.meshes import list_meshes, mesh_geometry


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
