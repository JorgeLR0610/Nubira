from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import powerClient
from neuralNetwork.weather_predictions import predict_by_date
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 🚀 Permitir peticiones desde cualquier origen (solo para desarrollo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o ["http://localhost:3000"] si quieres ser más estricto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Esquema del request ------------------
class ClimaRequest(BaseModel):
    latitude: float = Field(..., description="Latitude", ge=-90, le=90)
    longitude: float = Field(..., description="Longitude", ge=-180, le=180)
    day: int = Field(..., description="Day", ge=1, le=31)
    month: int = Field(..., description="Month", ge=1, le=12)
    year: int = Field(default=2025, description="Year for prediction")

res_enviar = []

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
        res = resultado["prediccion"]
        res_enviar = res

        # 4️⃣ Retornar resultado
        return {
            "temp_max": res[0],
            "temp_min": res[1],
            "precipitacion": res[2],
            "vel_viento": res[3]
        }

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento: {str(e)}")
