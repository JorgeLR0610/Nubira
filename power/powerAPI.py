import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from shapely.geometry import Point, shape
import json
import time
from tqdm import tqdm
from io import StringIO
import logging
from datetime import datetime

# ---------------- Configuración ----------------
output_file = "weatherData_2000_2025.csv"
variables = [
    'T2M_MAX','T2M_MIN','PRECTOTCORR',
    'IMERG_PRECLIQUID_PROB','PRECSNO','WS2M_MAX'
]

# Rango de años 2000-2025
start_date = "20000101"
end_date = "20251231"

# Paso de 5x5 grados
lat_step = 5.0
lon_step = 5.0

max_workers = 5  # 5 hilos paralelos
pause_between_requests = 1.0  # segundos de descanso entre requests

failed_points = []
successful_requests = 0
total_requests = 0

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('power_api_log_2000_2025.txt'),
        logging.StreamHandler()
    ]
)

# ---------------- Cargar GeoJSON público ----------------
print("🌍 Cargando datos geográficos para identificar zonas terrestres...")
geojson_url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
resp = requests.get(geojson_url)
countries = json.loads(resp.text)['features']
print(f"✅ Cargados {len(countries)} países/regiones")

def is_land(lat, lon):
    """Verifica si las coordenadas están en tierra"""
    point = Point(lon, lat)
    for country in countries:
        if shape(country['geometry']).contains(point):
            return True
    return False

# ---------------- Función de descarga ----------------
def fetch_power_data(lat, lon):
    global successful_requests, total_requests
    
    if not is_land(lat, lon):
        logging.info(f"Punto ({lat:.1f}, {lon:.1f}) - SALTADO: No es zona terrestre")
        return None  # Saltar puntos sobre agua
    
    total_requests += 1
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": ",".join(variables),
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "CSV"
    }
    
    try:
        # Mostrar inicio de petición
        start_msg = f"🚀 Punto ({lat:.1f}, {lon:.1f}) - INICIANDO petición para {start_date}-{end_date}"
        logging.info(start_msg)
        print(f"   {start_msg}")
        
        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()

        text = response.text
        if "YYYYMMDD" not in text:
            error_msg = f"❌ Punto ({lat:.1f}, {lon:.1f}) - ERROR: Respuesta no contiene datos válidos"
            logging.error(error_msg)
            print(f"   {error_msg}")
            failed_points.append((lat, lon, "No data in response"))
            return None

        df = pd.read_csv(StringIO(text), skiprows=11, on_bad_lines='skip')
        df['lat'] = lat
        df['lon'] = lon

        # Agregar transformaciones trigonométricas
        df['lat_sin'] = np.sin(np.radians(lat))
        df['lat_cos'] = np.cos(np.radians(lat))
        df['lon_sin'] = np.sin(np.radians(lon))
        df['lon_cos'] = np.cos(np.radians(lon))

        successful_requests += 1
        success_msg = f"✅ Punto ({lat:.1f}, {lon:.1f}) - ÉXITO: {len(df)} registros obtenidos"
        logging.info(success_msg)
        print(f"   {success_msg}")
        
        time.sleep(pause_between_requests)
        return df
        
    except requests.exceptions.Timeout as e:
        error_msg = f"⏰ Punto ({lat:.1f}, {lon:.1f}) - ERROR TIMEOUT: {str(e)}"
        logging.error(error_msg)
        print(f"   {error_msg}")
        failed_points.append((lat, lon, f"Timeout: {str(e)}"))
        return None
        
    except requests.exceptions.RequestException as e:
        error_msg = f"🌐 Punto ({lat:.1f}, {lon:.1f}) - ERROR REQUEST: {str(e)}"
        logging.error(error_msg)
        print(f"   {error_msg}")
        failed_points.append((lat, lon, f"Request error: {str(e)}"))
        return None
        
    except Exception as e:
        error_msg = f"💥 Punto ({lat:.1f}, {lon:.1f}) - ERROR GENERAL: {str(e)}"
        logging.error(error_msg)
        print(f"   {error_msg}")
        failed_points.append((lat, lon, f"General error: {str(e)}"))
        return None

# ---------------- Generar lista de puntos terrestres ----------------
print("🗺️  Generando puntos de cuadrícula terrestres...")
latitudes = np.arange(-90 + lat_step, 90, lat_step)
longitudes = np.arange(-180, 180 + lon_step, lon_step)
points = [(lat, lon) for lat in latitudes for lon in longitudes if is_land(lat, lon)]

print(f"📊 Configuración:")
print(f"   • Rango de años: {start_date} - {end_date}")
print(f"   • Resolución: {lat_step}° x {lon_step}°")
print(f"   • Puntos terrestres a procesar: {len(points)}")
print(f"   • Workers paralelos: {max_workers}")
print(f"   • Pausa entre requests: {pause_between_requests}s")
print(f"   • Variables: {', '.join(variables)}")

logging.info(f"Iniciando descarga de datos meteorológicos 2000-2025")
logging.info(f"Total de puntos terrestres a procesar: {len(points)}")
logging.info(f"Resolución de cuadrícula: {lat_step}° x {lon_step}°")
logging.info(f"Workers paralelos: {max_workers}")

# ---------------- Descarga paralela con barra de progreso ----------------
print(f"\n🚀 Iniciando descarga con {max_workers} hilos paralelos...")
all_data = []
progress_bar = tqdm(total=len(points), desc="Descargando datos", ncols=100)

start_time = time.time()

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_point = {executor.submit(fetch_power_data, lat, lon): (lat, lon) for lat, lon in points}
    for future in as_completed(future_to_point):
        lat, lon = future_to_point[future]
        df = future.result()
        if df is not None:
            all_data.append(df)
        # Actualizar barra de progreso
        progress_bar.update(1)
        
        # Mostrar estadísticas cada 10 requests
        if (successful_requests + len(failed_points)) % 10 == 0:
            progress_bar.set_postfix({
                'Exitosos': successful_requests,
                'Fallidos': len(failed_points),
                'Tasa éxito': f"{(successful_requests/(successful_requests + len(failed_points))*100):.1f}%"
            })

progress_bar.close()

end_time = time.time()
execution_time = end_time - start_time

# ---------------- Guardar CSV global ----------------
print(f"\n💾 Guardando datos en {output_file}...")
if all_data:
    global_df = pd.concat(all_data, ignore_index=True)
    global_df.to_csv(output_file, index=False)
    logging.info(f"Archivo CSV guardado exitosamente: {output_file}")
    logging.info(f"Total de registros guardados: {len(global_df)}")
    print(f"✅ Descarga completa. Archivo guardado en {output_file}")
    print(f"📊 Total de registros: {len(global_df)}")
    print(f"📅 Rango de fechas: {global_df['YYYYMMDD'].min()} - {global_df['YYYYMMDD'].max()}")
else:
    logging.warning("No se obtuvieron datos para guardar")
    print("\n❌ No se obtuvieron datos para guardar")

# ---------------- Guardar puntos fallidos ----------------
if failed_points:
    with open("failed_points_2000_2025.txt", "w") as f:
        f.write("lat,lon,error\n")
        for point_data in failed_points:
            if len(point_data) == 3:
                lat, lon, error = point_data
                f.write(f"{lat},{lon},{error}\n")
            else:
                lat, lon = point_data
                f.write(f"{lat},{lon},Unknown error\n")
    logging.info(f"Puntos fallidos guardados en failed_points_2000_2025.txt: {len(failed_points)}")
    print(f"⚠️  Puntos fallidos guardados en failed_points_2000_2025.txt: {len(failed_points)}")

# ---------------- Estadísticas finales ----------------
print(f"\n📈 ESTADÍSTICAS FINALES:")
print(f"⏱️  Tiempo de ejecución: {execution_time:.2f} segundos ({execution_time/60:.1f} minutos)")
print(f"✅ Peticiones exitosas: {successful_requests}")
print(f"❌ Peticiones fallidas: {len(failed_points)}")
print(f"📊 Total de peticiones: {total_requests}")
if total_requests > 0:
    success_rate = (successful_requests / total_requests) * 100
    print(f"🎯 Tasa de éxito: {success_rate:.2f}%")

logging.info("=== ESTADÍSTICAS FINALES ===")
logging.info(f"Tiempo total de ejecución: {execution_time:.2f} segundos")
logging.info(f"Peticiones exitosas: {successful_requests}")
logging.info(f"Peticiones fallidas: {len(failed_points)}")
logging.info(f"Total de peticiones: {total_requests}")
if total_requests > 0:
    success_rate = (successful_requests / total_requests) * 100
    logging.info(f"Tasa de éxito: {success_rate:.2f}%")

print(f"\n🎉 ¡Proceso completado!")
