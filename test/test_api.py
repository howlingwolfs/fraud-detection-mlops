import pytest
from fastapi.testclient import TestClient
from src.api import app

SAMPLE_PAYLOAD = {
    "Time": 0.0,
    "V1": -1.3598071336738,
    "V2": -0.0727811733098,
    "V3": 2.5363467379698,
    "V4": 1.3781552242744,
    "V5": -0.3383207699425,
    "V6": 0.4623880443422,
    "V7": 0.2395985540608,
    "V8": 0.098697901261,
    "V9": 0.3637869696116,
    "V10": 0.0907941719789,
    "V11": -0.5515995332608,
    "V12": -0.6178008557616,
    "V13": -0.9913898472354,
    "V14": -0.3111693536999,
    "V15": 1.4681769720943,
    "V16": -0.4704005252594,
    "V17": 0.2079712419292,
    "V18": 0.025790580198,
    "V19": 0.4039929602557,
    "V20": 0.2514120982397,
    "V21": -0.0183067779441,
    "V22": 0.2778375755589,
    "V23": -0.110474010131,
    "V24": 0.066928074905,
    "V25": 0.1285393582735,
    "V26": -0.1891148438888,
    "V27": 0.1335583767403,
    "V28": -0.0265233482418,
    "Amount": 149.62
}

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

def test_predict():
    with TestClient(app) as client:
        response = client.post("/predict", json=SAMPLE_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data