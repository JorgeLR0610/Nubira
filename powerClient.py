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



def convert_to_csv(api_response, filename='datos.csv', filter_month=None, filter_day=None):
    parameters_data = api_response['properties']['parameter']
    first_param = list(parameters_data.keys())[0]
    dates = list(parameters_data[first_param].keys())
    
    result = []
    for date in dates:
        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:8])
        
        # Aplicar filtro si se especificó
        if filter_month is not None and month != filter_month:
            continue
        if filter_day is not None and day != filter_day:
            continue
        
        daily_dict = {
            'year': year,
            'month': month,
            'day': day
        }
        
        for param_name, param_values in parameters_data.items():
            daily_dict[param_name] = param_values.get(date)
        
        result.append(daily_dict)
    
    df = pd.DataFrame(result)
    df.to_csv(filename, index=False)
    return df


