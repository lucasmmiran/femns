from openpyxl import load_workbook

from femns.report import COLUNAS, salvar_resumo


def test_salvar_resumo_cria_arquivo_com_cabecalho(tmp_path):
    path = tmp_path / "benchmarks.xlsx"
    salvar_resumo(path, {"mesh": "meshes/poiseuille.msh", "advection": "explicit", "dt": 0.001})

    wb = load_workbook(path)
    ws = wb.active
    linhas = list(ws.iter_rows(values_only=True))

    assert linhas[0] == tuple(COLUNAS)
    assert len(linhas) == 2

    linha = dict(zip(COLUNAS, linhas[1]))
    assert linha["mesh"] == "meshes/poiseuille.msh"
    assert linha["advection"] == "explicit"
    assert linha["dt"] == 0.001


def test_salvar_resumo_chave_ausente_vira_celula_vazia(tmp_path):
    path = tmp_path / "benchmarks.xlsx"
    salvar_resumo(path, {"mesh": "meshes/poiseuille.msh"})

    wb = load_workbook(path)
    ws = wb.active
    linha = dict(zip(COLUNAS, next(ws.iter_rows(min_row=2, values_only=True))))

    # openpyxl le uma celula de string vazia de volta como None
    assert linha["advection"] is None


def test_salvar_resumo_chave_extra_e_ignorada(tmp_path):
    path = tmp_path / "benchmarks.xlsx"
    salvar_resumo(path, {"mesh": "meshes/poiseuille.msh", "chave_que_nao_existe": 42})

    wb = load_workbook(path)
    ws = wb.active
    linha = dict(zip(COLUNAS, next(ws.iter_rows(min_row=2, values_only=True))))

    assert "chave_que_nao_existe" not in linha
    assert linha["mesh"] == "meshes/poiseuille.msh"


def test_salvar_resumo_acumula_uma_linha_por_chamada(tmp_path):
    path = tmp_path / "benchmarks.xlsx"
    salvar_resumo(path, {"mesh": "a", "advection": "explicit"})
    salvar_resumo(path, {"mesh": "a", "advection": "semi_lagrangian"})
    salvar_resumo(path, {"mesh": "b", "advection": "explicit"})

    wb = load_workbook(path)
    ws = wb.active
    linhas = list(ws.iter_rows(min_row=2, values_only=True))

    assert len(linhas) == 3
    advections = [dict(zip(COLUNAS, linha))["advection"] for linha in linhas]
    assert advections == ["explicit", "semi_lagrangian", "explicit"]


def test_salvar_resumo_cria_diretorio_pai_se_nao_existir(tmp_path):
    path = tmp_path / "subdir" / "benchmarks.xlsx"
    salvar_resumo(path, {"mesh": "a"})

    assert path.exists()
