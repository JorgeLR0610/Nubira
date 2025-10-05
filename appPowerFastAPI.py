from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import powerClient
from datetime import datetime
import pandas as pd

app=FastAPI()
    
class ClimaRequest(BaseModel):
    latitude: float = Field(..., description="latitude", ge=-90, le=90) #ge = greater or equal, le = lesser or equal
    longitude: float = Field(..., description="longitude", ge=-180, le=180)
    day: int = Field(..., description="day", ge=1, le=31)
    month: int = Field(..., description="month", ge=1, le=12)

@app.post('/clima')
def getClima(request: ClimaRequest):   
    try:
        start = datetime(2010, 9, 6)
        end = datetime(2010, 10, 7)
        
        raw_data = powerClient.fetch_power_daily(request.latitude, request.longitude, start, end)
        formated_data = powerClient.convert_to_list_of_dicts(raw_data)
           
                     
        return #Devolver datos
        
    except Exception as e:
        print('Error', e)
        raise HTTPException(status_code=500, detail= 'Error en el procesamiento de datos')
    