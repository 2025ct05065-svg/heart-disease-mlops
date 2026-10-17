from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_prediction_endpoint():
    patient = {
        "age": 55,
        "trestbps": 130,
        "chol": 250,
        "thalach": 150,
        "oldpeak": 1.0,
        "sex": 1,
        "cp": 2,
        "fbs": 0,
        "restecg": 1,
        "exang": 0,
        "slope": 2,
        "ca": 0,
        "thal": 3
    }

    response = client.post("/predict", json=patient)

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert 0 <= result["probability"] <= 1
    assert "label" in result
    assert "note" in result

def test_prediction_rejects_missing_fields():
    response = client.post("/predict", json={"age": 55})
    assert response.status_code == 422
