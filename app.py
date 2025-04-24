# Importações necessárias
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from datetime import datetime

# Inicialização do FastAPI
app = FastAPI()

# Carregar o dataset Iris
iris = sns.load_dataset('iris')

# Preparação dos dados
X = iris.drop(columns=['species'])  # Features
y = iris['species']  # Rótulos

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar o MLflow
# mlflow.set_tracking_uri("http://127.0.0.1:5000")  # URI do servidor MLflow (opcional)
mlflow.set_experiment("Iris_Classification_Experiment")

# Treinar o modelo Random Forest com MLflow
def train_model():
    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Definir hiperparâmetros
        n_estimators = 100
        max_depth = 5
        random_state = 42

        # Log dos hiperparâmetros no MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        # Treinar o modelo
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        # Fazer previsões no conjunto de teste
        y_pred = model.predict(X_test)

        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)

        # Log das métricas no MLflow
        mlflow.log_metric("accuracy", accuracy)

        # Salvar o modelo no MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")

        return model
    
# Treinar o modelo inicialmente
model = train_model()

# Definir o esquema de entrada para a API
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Endpoint para previsão
@app.post("/predict/")
async def predict(input_data: IrisInput):
    try:
        # Converter os dados de entrada em um array NumPy
        data = np.array([[input_data.sepal_length, input_data.sepal_width,
                          input_data.petal_length, input_data.petal_width]])
        
        # Fazer a previsão usando o modelo treinado
        prediction = model.predict(data)
        
        # Retornar a previsão como resposta
        return {"predicted_species": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint de verificação de saúde
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Executar o servidor localmente (para testes)
if __name__ == "__main__":
    import uvicorn
    #uvicorn.run(app, host="127.0.0.1", port=8000)
    port = int(os.environ.get("PORT", 8000))  # Obtém a porta do ambiente (Render define automaticamente)
    uvicorn.run(app, host="0.0.0.0", port=port)