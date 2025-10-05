// app/paginaRes/page.js
"use client"
import { useState, useEffect } from 'react';

const EnhancedWeatherPage = () => {
  const [weatherType, setWeatherType] = useState('normal');
  const [temperature, setTemperature] = useState(22); // Variable temporal en °C
  const [precipitation, setPrecipitation] = useState(0); // Nueva variable para precipitación
  const [windSpeed, setWindSpeed] = useState(5); // Nueva variable para velocidad del viento

  // Función para determinar el tipo de clima basado en múltiples factores
  const getWeatherTypeFromConditions = (temp, precip, wind) => {
    // Condiciones para tormenta eléctrica
    if (precip > 70 && wind > 30 && temp > 15) return 'thunderstorm';
    
    // Condiciones para granizo
    if (precip > 60 && temp <= 5) return 'hail';
    
    // Condiciones para nevado
    if (precip > 50 && temp <= 0) return 'snowy';
    
    // Condiciones originales
    if (temp >= 30) return 'sunny';
    if (temp >= 20) return 'normal';
    if (temp >= 10) return 'cloudy';
    return 'rainy';
  };

  // Simular datos del backend - expandido para incluir más variables
  const simulateBackendData = () => {
    // Temperaturas aleatorias para simular diferentes condiciones
    const temps = [-5, -2, 0, 2, 5, 8, 12, 15, 18, 22, 25, 28, 32, 35];
    const randomTemp = temps[Math.floor(Math.random() * temps.length)];
    
    // Precipitación aleatoria (0-100%)
    const randomPrecip = Math.floor(Math.random() * 101);
    
    // Velocidad del viento aleatoria (0-50 km/h)
    const randomWind = Math.floor(Math.random() * 51);
    
    return { temp: randomTemp, precip: randomPrecip, wind: randomWind };
  };

  // Efecto para simular la obtención de datos del backend
  useEffect(() => {
    // Simular llamada al backend cada 5 segundos (solo para demo)
    const interval = setInterval(() => {
      const newData = simulateBackendData();
      setTemperature(newData.temp);
      setPrecipitation(newData.precip);
      setWindSpeed(newData.wind);
      setWeatherType(getWeatherTypeFromConditions(newData.temp, newData.precip, newData.wind));
    }, 5000);

    // Datos iniciales
    const initialData = simulateBackendData();
    setTemperature(initialData.temp);
    setPrecipitation(initialData.precip);
    setWindSpeed(initialData.wind);
    setWeatherType(getWeatherTypeFromConditions(initialData.temp, initialData.precip, initialData.wind));

    return () => clearInterval(interval);
  }, []);

  const weatherConfig = {
    normal: {
      background: 'bg-gradient-to-br from-blue-400 via-blue-500 to-blue-600',
      icon: '⛅',
      name: 'Normal Day',
      description: 'Clear sky with some clouds',
      tempRange: '20-29°C',
      textColor: 'text-white'
    },
    sunny: {
      background: 'bg-gradient-to-br from-yellow-400 via-orange-400 to-red-500',
      icon: '☀️',
      name: 'Hot Day',
      description: 'Intense sunlight, and warm weather',
      tempRange: '30°C+',
      textColor: 'text-white'
    },
    cloudy: {
      background: 'bg-gradient-to-br from-gray-400 via-gray-500 to-gray-600',
      icon: '☁️',
      name: 'Cloudy Day',
      description: 'Cloud covered sky',
      tempRange: '10-19°C',
      textColor: 'text-white'
    },
    rainy: {
      background: 'bg-gradient-to-br from-blue-600 via-blue-700 to-gray-800',
      icon: '🌧️',
      name: 'Rainy Day',
      description: 'Persistent precipitation throughout the day',
      tempRange: '0-9°C',
      textColor: 'text-white'
    },
    snowy: {
      background: 'bg-gradient-to-br from-blue-100 via-blue-200 to-white',
      icon: '❄️',
      name: 'Snowy Day',
      description: 'Snow falling and cold temperatures',
      tempRange: 'Below 0°C',
      textColor: 'text-gray-800'
    },
    hail: {
      background: 'bg-gradient-to-br from-gray-300 via-gray-400 to-gray-600',
      icon: '🌨️',
      name: 'Hail Storm',
      description: 'Falling ice pellets, be careful!',
      tempRange: '0-5°C',
      textColor: 'text-gray-800'
    },
    thunderstorm: {
      background: 'bg-gradient-to-br from-purple-800 via-gray-900 to-black',
      icon: '⛈️',
      name: 'Thunderstorm',
      description: 'Heavy rain with lightning and thunder',
      tempRange: '15°C+',
      textColor: 'text-white'
    }
  };

  const currentWeather = weatherConfig[weatherType];

  return (
    <div className={`min-h-screen transition-all duration-1000 ${currentWeather.background}`}>
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold text-white text-center mb-2">
          Enhanced Weather App
        </h1>
        
        {/* Tarjeta de información del clima - Ahora con fondo blanco parcialmente transparente */}
        <div className="bg-white/30 backdrop-blur-lg rounded-2xl p-8 max-w-md mx-auto border border-white/40 shadow-xl">
          <div className="text-center">
            <div className="text-6xl mb-4">{currentWeather.icon}</div>
            <h2 className={`text-3xl font-bold mb-2 ${currentWeather.textColor}`}>
              {currentWeather.name}
            </h2>
            
            {/* Display de temperatura */}
            <div className={`text-5xl font-bold my-4 ${currentWeather.textColor}`}>
              {temperature}°C
            </div>
            
            <p className={`opacity-90 mb-2 ${currentWeather.textColor}`}>
              {currentWeather.description}
            </p>
            <p className={`opacity-80 text-sm mb-4 ${currentWeather.textColor}`}>
              Range: {currentWeather.tempRange}
            </p>
            
            {/* Información adicional de condiciones */}
            <div className="mt-6 grid grid-cols-2 gap-4">
              <div className="bg-black/20 rounded-lg p-3 text-white backdrop-blur-sm">
                <div className="opacity-80 text-sm">Precipitation</div>
                <div className="text-xl font-bold">{precipitation}%</div>
              </div>
              <div className="bg-black/20 rounded-lg p-3 text-white backdrop-blur-sm">
                <div className="opacity-80 text-sm">Wind Speed</div>
                <div className="text-xl font-bold">{windSpeed} km/h</div>
              </div>
            </div>

            {/* Indicadores de condiciones especiales */}
            {(weatherType === 'thunderstorm' || weatherType === 'hail') && (
              <div className="mt-4 p-3 bg-red-500/70 rounded-lg backdrop-blur-sm">
                <p className="text-white font-bold text-sm">
                  ⚠️ {weatherType === 'thunderstorm' 
                    ? 'Lightning danger - Seek shelter' 
                    : 'Hail warning - Protect yourself'}
                </p>
              </div>
            )}

            {weatherType === 'snowy' && (
              <div className="mt-4 p-3 bg-blue-500/70 rounded-lg backdrop-blur-sm">
                <p className="text-white font-bold text-sm">
                  ❄️ Cold weather alert - Dress warmly
                </p>
              </div>
            )}
            
          </div>
        </div>

        {/* Panel informativo adicional */}
        <div className="mt-8 max-w-md mx-auto">
          <div className="bg-black/40 rounded-xl p-6 backdrop-blur-md border border-white/20">
            <h3 className="text-white text-xl font-bold mb-4">Weather Conditions Guide</h3>
            <div className="space-y-2 text-white text-sm">
              <p>❄️ <strong>Snowy:</strong> Temperature ≤ 0°C + Precipitation</p>
              <p>🌨️ <strong>Hail:</strong> Temperature ≤ 5°C + Heavy Precipitation</p>
              <p>⛈️ <strong>Thunderstorm:</strong> High Precipitation + Strong Wind</p>
              <p className="text-xs opacity-70 mt-4">Updates every 5 seconds</p>
            </div>
          </div>
        </div>
        
      </div>
    </div>
  );
};

export default EnhancedWeatherPage;