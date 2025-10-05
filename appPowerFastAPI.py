from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import powerClient
from neuralNetwork.weather_predictions import predict_by_date

app = FastAPI()

# ------------------ Esquema del request ------------------
class ClimaRequest(BaseModel):
    latitude: float = Field(..., description="Latitude", ge=-90, le=90)
    longitude: float = Field(..., description="Longitude", ge=-180, le=180)
    day: int = Field(..., description="Day", ge=1, le=31)
    month: int = Field(..., description="Month", ge=1, le=12)
    year: int = Field(default=2025, description="Year for prediction")


# ------------------ Endpoint principal ------------------
@app.post("/clima")
def get_clima(request: ClimaRequest):
    try:
        # 1️⃣ Obtener datos históricos
        start = datetime(1981, 9, 6)
        end = datetime(2024, 10, 7)
        raw_data = powerClient.fetch_power_daily(request.latitude, request.longitude, start, end)

        # 2️⃣ Crear CSV filtrando por mes y día
        formatted_data = powerClient.convert_to_csv(raw_data, "datos.csv", request.month, request.day)

        # 3️⃣ Entrenar y obtener mejor epoch
        resultado = predict_by_date(request.year, request.month, request.day)

        # 4️⃣ Retornar resultado
        return {
            "mensaje": f"Predicción generada con el mejor modelo (loss={resultado['mejor_loss']:.6f})",
            "prediccion": resultado["prediccion"]
        }

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento: {str(e)}")
