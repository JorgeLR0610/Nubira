import requests
from datetime import datetime
import pandas as pd
#Eliminar probabilidad de precipitación
def fetch_power_daily(lat, lon, start, end, parameters=['T2M_MAX','T2M_MIN','PRECTOTCORR', 'PRECSNO', 'WS2M_MAX']):
    # Construye URL base
    base = "https://power.larc.nasa.gov/api/temporal/daily/point"
    # parámetros
    params = {
        "latitude": lat,
        "longitude": lon,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "parameters": ",".join(parameters),
        "community": "AG",
        "format": "JSON"
    }
    resp = requests.get(base, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data

def convert_to_list_of_dicts(api_response):
    # Extrae los parámetros
    parameters_data = api_response['properties']['parameter']
    
    # Obtiene todas las fechas (usando el primer parámetro como referencia)
    first_param = list(parameters_data.keys())[0]
    dates = list(parameters_data[first_param].keys())
    
    # Crea la lista de diccionarios
    result = []
    for date in dates:
        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:8])
        
        daily_dict = {
            'year': year,
            'month': month,
            'day': day
        }
        
        for param_name, param_values in parameters_data.items():
            daily_dict[param_name] = param_values.get(date)
        
        result.append(daily_dict)
        
    df = pd.DataFrame(result)
    
    return df
    