import pytest

from femns.webui.server import check_bind, check_token, is_local_host


@pytest.mark.parametrize("host,esperado", [
    ("127.0.0.1", True), ("localhost", True), ("::1", True), ("", True),
    ("0.0.0.0", False), ("192.168.0.10", False), ("meu-servidor", False),
])
def test_is_local_host(host, esperado):
    assert is_local_host(host) is esperado


def test_check_token_sem_token_no_servidor_libera_tudo():
    assert check_token(None, None, {}) is True
    assert check_token("", "Bearer qualquer", {}) is True


def test_check_token_bearer_correto():
    assert check_token("segredo", "Bearer segredo", {}) is True


def test_check_token_bearer_errado():
    assert check_token("segredo", "Bearer outro", {}) is False


def test_check_token_via_query_string():
    assert check_token("segredo", None, {"token": ["segredo"]}) is True
    assert check_token("segredo", None, {"token": ["errado"]}) is False


def test_check_token_ausente_quando_exigido():
    assert check_token("segredo", None, {}) is False
    assert check_token("segredo", "", {}) is False
    assert check_token("segredo", "Basic Zm9v", {}) is False


def test_check_bind_local_nao_avisa_nem_levanta():
    assert check_bind("127.0.0.1", None, allow_no_auth=False) is None
    assert check_bind("localhost", "tok", allow_no_auth=False) is None


def test_check_bind_exposto_com_token_avisa_sobre_https():
    aviso = check_bind("0.0.0.0", "tok", allow_no_auth=False)
    assert aviso and "HTTPS" in aviso


def test_check_bind_exposto_sem_token_com_allow_avisa_forte():
    aviso = check_bind("0.0.0.0", None, allow_no_auth=True)
    assert aviso and "SEM AUTENTICACAO" in aviso


def test_check_bind_exposto_sem_token_sem_allow_levanta():
    with pytest.raises(RuntimeError, match="autenticacao"):
        check_bind("0.0.0.0", None, allow_no_auth=False)
