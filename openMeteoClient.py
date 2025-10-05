import requests
from datetime import datetime

def fetch_openmeteo_daily(lat, lon, start_date, end_date, variables = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'wind_speed_10m_max'], timezone = 'UTC'):
    
    base = 'https://archive-api.open-meteo.com/v1/archive'
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date.strftime("%Y-%m-%d"),
        'end_date': end_date.strftime("%Y-%m-%d"),
        'daily': ','.join(variables),
        'timezone': timezone
    }
    
    resp = requests.get(base, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data

def json_to_registries(data): #Este es el método que devuelve la respuesta de la api en forma de lista de diccionarios jeje
    registros = []
    daily = data.get("daily", {})
    times = daily.get("time", [])
    tmax = daily.get("temperature_2m_max", [])
    tmin = daily.get("temperature_2m_min", [])
    precip = daily.get("precipitation_sum", [])

    for i, fecha in enumerate(times):
        year, month, day = fecha.split("-") #Para separar la fecha en año, mes y día
        registro = {
            "año": int(year),
            "mes": int(month),
            "día": int(day),
            "temperature_2m_max": tmax[i] if i < len(tmax) else None,
            "temperature_2m_min": tmin[i] if i < len(tmin) else None,
            "precipitation_sum": precip[i] if i < len(precip) else None
        }
        registros.append(registro)
        
        print(registro)
    return registros
