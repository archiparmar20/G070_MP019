import json

from app.prediction_logging import log_prediction


def test_prints_one_json_line(capsys):
    log_prediction(42.12)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["event"] == "prediction"
    assert data["predicted_energy_kwh"] == 42.12
    assert "timestamp" in data


def test_only_safe_fields_are_logged(capsys):
    log_prediction(10.0)
    data = json.loads(capsys.readouterr().out.strip())
    assert set(data) == {"timestamp", "event", "predicted_energy_kwh"}


def test_two_predictions_two_lines(capsys):
    log_prediction(1.0)
    log_prediction(2.0)
    assert len(capsys.readouterr().out.strip().splitlines()) == 2
