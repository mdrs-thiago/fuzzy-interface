"""API tests. Run with: pytest test_api.py"""
import base64

import pytest
from fastapi.testclient import TestClient

from api import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def build_tipping(client):
    assert client.post("/variable", json={"name": "Service", "min_val": 0, "max_val": 10, "role": "input"}).status_code == 200
    assert client.post("/variable", json={"name": "Tip", "min_val": 0, "max_val": 30, "role": "output"}).status_code == 200
    assert client.post("/variable/Service/term", json={"name": "Poor", "type": "tri", "params": [0, 0, 5]}).status_code == 200
    assert client.post("/variable/Service/term", json={"name": "Good", "type": "tri", "params": [0, 5, 10]}).status_code == 200
    assert client.post("/variable/Tip/term", json={"name": "Low", "type": "tri", "params": [0, 0, 15]}).status_code == 200
    assert client.post("/variable/Tip/term", json={"name": "High", "type": "tri", "params": [15, 30, 30]}).status_code == 200
    assert client.post("/rule", json={"antecedents": {"Service": "Poor"}, "consequent": {"Tip": "Low"}}).status_code == 200
    assert client.post("/rule", json={"antecedents": {"Service": "Good"}, "consequent": {"Tip": "High"}}).status_code == 200


def test_ui_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/app.js").status_code == 200
    assert client.get("/app.css").status_code == 200


def test_build_simulate_and_delete(client):
    build_tipping(client)
    assert client.post("/build").status_code == 200
    data = client.post("/simulate", json={"inputs": {"Service": 2.5}}).json()
    assert data["output"] is not None and len(data["rules_triggered"]) > 0
    assert client.delete("/rule/0").status_code == 200
    assert client.delete("/variable/Service/term/Good").status_code == 200
    assert client.get("/config").json()["rules"] == []
    assert client.delete("/variable/Tip").status_code == 200
    assert client.delete("/variable/Tip").status_code == 404


def test_sessions_are_isolated():
    with TestClient(app) as a, TestClient(app) as b:
        build_tipping(a)
        assert b.get("/config").json() == {"variables": {}, "rules": []}
        assert "Service" in a.get("/config").json()["variables"]


def test_simulate_without_explicit_build_and_after_preset(client):
    assert client.post("/preset/tank").status_code == 200
    assert client.post("/simulate", json={"inputs": {"level": 0.3, "rate": 0}}).status_code == 200
    assert client.post("/preset/crane").status_code == 200
    r = client.post("/simulate", json={"inputs": {"PosErr": 1, "Vel": 0, "Angle": 0, "W": 0}})
    assert r.status_code == 200


def test_edits_take_effect_without_rebuild(client):
    client.post("/preset/tank")
    before = client.post("/simulate", json={"inputs": {"level": 0.05, "rate": 0}}).json()["output"]
    client.post("/variable/valve/term", json={"name": "no_change", "type": "tri", "params": [0.5, 0.6, 0.7]})
    after = client.post("/simulate", json={"inputs": {"level": 0.05, "rate": 0}}).json()["output"]
    assert after > before + 0.3


def test_shoulder_terms_are_one_at_range_edges(client):
    client.post("/preset/tank")
    low = client.post("/simulate", json={"inputs": {"level": -1, "rate": 0}}).json()
    assert low["input_memberships"]["level"]["low"] == pytest.approx(1.0)
    assert low["output"] > 0.5  # open_fast
    high = client.post("/simulate", json={"inputs": {"level": 1, "rate": 0.1}}).json()
    assert high["input_memberships"]["level"]["high"] == pytest.approx(1.0)
    assert high["input_memberships"]["rate"]["positive"] == pytest.approx(1.0)


def test_no_rule_fired_returns_null_output(client):
    build_tipping(client)
    client.delete("/rule/0")
    data = client.post("/simulate", json={"inputs": {"Service": 0}}).json()
    assert data["output"] is None and data["no_rule_fired"] is True


def test_dc_motor_exercise_gives_clear_error(client):
    assert client.post("/preset/dc_motor").status_code == 200
    r = client.post("/simulate", json={"inputs": {"error": 10, "error_dot": 0}})
    assert r.status_code == 400 and "fuzzy sets" in r.json()["detail"]
    assert client.post("/build").status_code == 400


def test_wang_mendel_result_imports(client):
    csv = "x,y\n" + "\n".join(f"{i},{i * 2}" for i in range(20))
    wm = client.post("/wang-mendel", json={"csv_data": base64.b64encode(csv.encode()).decode(), "target_col": "y", "n_terms": 3})
    assert wm.status_code == 200
    assert client.post("/import", json=wm.json()).status_code == 200
    cfg = client.get("/config").json()
    assert len(cfg["variables"]["x"]["terms"]) == 3 and cfg["variables"]["y"]["role"] == "output"
    assert len(cfg["rules"]) > 0
    assert client.post("/simulate", json={"inputs": {"x": 5}}).json()["output"] is not None


def test_wang_mendel_rejects_bad_input(client):
    def run(csv, target="y", n=3):
        return client.post("/wang-mendel", json={"csv_data": base64.b64encode(csv.encode()).decode(), "target_col": target, "n_terms": n})
    assert run("x,y\na,1\n").status_code == 400
    assert run("x,y\n1,2\n", target="z").status_code == 400
    assert run("x,y\n1,2\n", n=1).status_code == 422
    assert client.post("/wang-mendel", json={"csv_data": "not base64!", "target_col": "y"}).status_code == 400


def test_export_import_roundtrip_all_mf_types(client):
    client.post("/variable", json={"name": "a", "min_val": 0, "max_val": 10})
    terms = {"t": ("tri", [0, 5, 10]), "p": ("trap", [0, 1, 2, 3]), "g": ("gauss", [5, 1]), "b": ("bell", [2, 2, 5]), "s": ("sig", [1, 5])}
    for name, (kind, params) in terms.items():
        assert client.post("/variable/a/term", json={"name": name, "type": kind, "params": params}).status_code == 200
    exported = client.get("/export").json()
    client.post("/reset")
    assert client.post("/import", json=exported).status_code == 200
    restored = client.get("/config").json()["variables"]["a"]["terms"]
    assert {k: v["params"] for k, v in restored.items()} == {k: pytest.approx(p) for k, (_, p) in terms.items()}


def test_failed_import_keeps_current_system(client):
    build_tipping(client)
    r = client.post("/import", json={"variables": {"x": {"role": "input"}}})
    assert r.status_code == 400
    assert "Service" in client.get("/config").json()["variables"]


@pytest.mark.parametrize("payload", [
    {"name": "t", "type": "tri", "params": [1, 2]},
    {"name": "t", "type": "tri", "params": [3, 2, 1]},
    {"name": "t", "type": "gauss", "params": [1, 0]},
    {"name": "t", "type": "nope", "params": [1]},
])
def test_bad_terms_are_400(client, payload):
    client.post("/variable", json={"name": "a", "min_val": 0, "max_val": 10})
    assert client.post("/variable/a/term", json=payload).status_code == 400


def test_validation_errors(client):
    assert client.post("/variable", json={"name": "z", "min_val": 10, "max_val": 0}).status_code == 400
    assert client.post("/variable", json={"name": "a/b", "min_val": 0, "max_val": 1}).status_code == 422
    assert client.post("/rule", json={"antecedents": {}, "consequent": {}}).status_code == 400
    client.post("/preset/tank")
    r = client.post("/simulate", json={"inputs": {"level": 0.1}})
    assert r.status_code == 400 and "rate" in r.json()["detail"]
    client.post("/variable", json={"name": "out2", "min_val": 0, "max_val": 1, "role": "output"})
    assert "Only one output" in client.post("/build").json()["detail"]


def test_names_with_spaces(client):
    assert client.post("/variable", json={"name": "my var", "min_val": 0, "max_val": 1}).status_code == 200
    assert client.post("/variable/my%20var/term", json={"name": "very low", "type": "tri", "params": [0, 0, 1]}).status_code == 200
    assert client.get("/plot/my%20var").json()["terms"]["very low"][0] == pytest.approx(1.0)
    assert client.delete("/variable/my%20var/term/very%20low").status_code == 200
