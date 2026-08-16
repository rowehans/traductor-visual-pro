"""Regresiones de alcance del rate limiting."""

from flask import Flask

from ratelimit import _rate_limit_key


def test_red_local_no_se_exonera_para_red_privada():
    app = Flask(__name__)
    with app.test_request_context(
        "/api/process-page",
        environ_base={"REMOTE_ADDR": "192.168.1.25"},
    ):
        assert _rate_limit_key() == "192.168.1.25"


def test_loopback_real_sigue_exento():
    app = Flask(__name__)
    with app.test_request_context(
        "/api/translate",
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
    ):
        assert _rate_limit_key() is None
