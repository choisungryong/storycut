from typing import Dict, Any, List

from utils.logger import get_logger

_logger = get_logger("error_manager")


class ErrorManager:
    """
    Centralized error logging wrapper using Python logging module.
    """

    @classmethod
    def log_error(
        cls,
        service: str,
        error_message: str,
        details: Any = None,
        severity: str = "error"
    ):
        level_map = {
            "warning": _logger.warning,
            "error": _logger.error,
            "critical": _logger.critical,
        }
        log_fn = level_map.get(severity, _logger.error)
        msg = f"[{service}] {error_message}"
        if details:
            msg += f" | {details}"
        log_fn(msg)

    @classmethod
    def get_recent_errors(cls, limit: int = 20) -> List[Dict]:
        return []

    @classmethod
    def clear_logs(cls):
        pass
