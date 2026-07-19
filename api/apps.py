import logging

from django.apps import AppConfig

log = logging.getLogger(__name__)


class ApiConfig(AppConfig):
    name = "api"

    def ready(self):
        from LSTM.predict import _get_session
        try:
            _get_session()
        except Exception:
            log.exception("Model failed to preload at startup")
