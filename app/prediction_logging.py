"""JSON logging for the prediction API.

One JSON line is printed to stdout for each successful prediction. The line
holds only the timestamp, the event name and the predicted energy. Customer
details and the request body are never logged.

Usage in app/main.py, after the model has returned a value:

    from app.prediction_logging import log_prediction
    log_prediction(prediction)
"""
import json
import logging
import sys
from datetime import datetime, timezone

LOGGER_NAME = "mp019.prediction"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc).isoformat(),
            "event": record.getMessage(),
            "predicted_energy_kwh": getattr(record, "predicted_energy_kwh", None),
        })


class _StdoutHandler(logging.StreamHandler):
    """Always writes to the current sys.stdout (also works under pytest)."""

    def __init__(self):
        logging.Handler.__init__(self)

    @property
    def stream(self):
        return sys.stdout


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:  # add the handler only once
        handler = _StdoutHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # no duplicate lines from the root logger
    return logger


def log_prediction(predicted_energy_kwh: float) -> None:
    """Log one successful prediction. Call it once per /predict response."""
    get_logger().info(
        "prediction",
        extra={"predicted_energy_kwh": round(float(predicted_energy_kwh), 4)},
    )
