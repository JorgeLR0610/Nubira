from functools import lru_cache

from fastapi import FastAPI, HTTPException
import torch
from pydantic import BaseModel, Field
import powerClient
from datetime import datetime
import pandas as pd
import numpy as np
from neuralNetwork.weather_predictions import predict_by_date

app=FastAPI()
    
class ClimaRequest(BaseModel):
    latitude: float = Field(..., description="latitude", ge=-90, le=90) #ge = greater or equal, le = lesser or equal
    longitude: float = Field(..., description="longitude", ge=-180, le=180)
    day: int = Field(..., description="day", ge=1, le=31)
    month: int = Field(..., description="month", ge=1, le=12)
    year: int = Field(..., description="year")


@app.post('/clima')
def getClima(request: ClimaRequest):   
    try:
        start = datetime(1981, 9, 6)
        end = datetime(2024, 10, 7)
        
        raw_data = powerClient.fetch_power_daily(request.latitude, request.longitude, start, end)
        
        formated_data = powerClient.convert_to_csv(raw_data, 'datos.csv', request.month, request.day)

         # 3️⃣ Entrenar modelo y predecir (todo ocurre dentro de la función)
        prediccion = predict_by_date(request.year, request.month, request.day)

        # 4️⃣ Retornar resultado
        return {
            "mensaje": "Predicción generada correctamente (modelo entrenado en esta ejecución).",
            "prediccion": prediccion
        }
        
    except Exception as e:
        print('Error', e)
        raise HTTPException(status_code=500, detail= 'Error en el procesamiento de datos')
    