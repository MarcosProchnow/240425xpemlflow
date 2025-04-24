import pytest
from fastapi.testclient import TestClient
from app import app  # Importa a instância FastAPI do arquivo principal

# Cria um cliente de teste para simular requisições à API
client = TestClient(app)

# Teste para verificar se a API está funcionando (endpoint /health)
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# Teste para verificar a previsão da API (endpoint /predict/)
@pytest.mark.parametrize("input_data, expected_species", [
    (
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        },
        "setosa"
    ),
    (
        {
            "sepal_length": 6.0,
            "sepal_width": 2.2,
            "petal_length": 4.0,
            "petal_width": 1.0
        },
        "versicolor"
    ),
    (
        {
            "sepal_length": 6.3,
            "sepal_width": 3.3,
            "petal_length": 6.0,
            "petal_width": 2.5
        },
        "virginica"
    )
])
def test_predict(input_data, expected_species):
    response = client.post("/predict/", json=input_data)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_species" in data
    assert data["predicted_species"] == expected_species
