// app/paginaRes/page.js
"use client"
import { useState, useEffect } from 'react';

const EnhancedWeatherPage = () => {
  const [weatherType, setWeatherType] = useState('normal');
  const [temperature, setTemperature] = useState(22); // Variable temporal en °C

  // Función para determinar el tipo de clima basado en temperatura
  const getWeatherTypeFromTemp = (temp) => {
    if (temp >= 30) return 'sunny';
    if (temp >= 20) return 'normal';
    if (temp >= 10) return 'cloudy';
    return 'rainy';
  };

  // Simular datos del backend - esto se reemplazará luego con una API real
  const simulateBackendData = () => {
    // Temperaturas aleatorias para simular diferentes condiciones
    const temps = [15, 18, 22, 25, 28, 32, 8, 12];
    const randomTemp = temps[Math.floor(Math.random() * temps.length)];
    return randomTemp;
  };

  // Efecto para simular la obtención de datos del backend
  useEffect(() => {
    // Simular llamada al backend cada 5 segundos (solo para demo)
    const interval = setInterval(() => {
      const newTemp = simulateBackendData();
      setTemperature(newTemp);
      setWeatherType(getWeatherTypeFromTemp(newTemp));
    }, 5000);

    // Datos iniciales
    const initialTemp = simulateBackendData();
    setTemperature(initialTemp);
    setWeatherType(getWeatherTypeFromTemp(initialTemp));

    return () => clearInterval(interval);
  }, []);

  const weatherConfig = {
    normal: {
      background: 'bg-gradient-to-br from-blue-400 via-blue-500 to-blue-600',
      icon: '⛅',
      name: 'Normal Day',
      description: 'Clear sky with some clouds',
      tempRange: '20-29°C'
    },
    sunny: {
      background: 'bg-gradient-to-br from-yellow-400 via-orange-400 to-red-500',
      icon: '☀️',
      name: 'Hot Day',
      description: 'Intense sunligh, and warm weather',
      tempRange: '30°C+'
    },
    cloudy: {
      background: 'bg-gradient-to-br from-gray-400 via-gray-500 to-gray-600',
      icon: '☁️',
      name: 'Cloudy Day',
      description: 'Cloud covered sky',
      tempRange: '10-19°C'
    },
    rainy: {
      background: 'bg-gradient-to-br from-blue-600 via-blue-700 to-gray-800',
      icon: '🌧️',
      name: 'Rainy Day',
      description: 'persistent precipitation trought the day',
      tempRange: '0-9°C'
    }
  };

  const currentWeather = weatherConfig[weatherType];

  return (
    <div className={`min-h-screen transition-all duration-1000 ${currentWeather.background}`}>
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold text-white text-center mb-2">
          Weather App
        </h1>
        <p className="text-black text-center opacity-80 mb-8">
          El fondo cambia automáticamente según la temperatura
        </p>

        {/* Tarjeta de información del clima */}
        <div className="bg-white bg-opacity-20 backdrop-blur-lg rounded-2xl p-8 max-w-md mx-auto border border-white border-opacity-30">
          <div className="text-center">
            <div className="text-6xl mb-4">{currentWeather.icon}</div>
            <h2 className="text-3xl font-bold text-black mb-2">
              {currentWeather.name}
            </h2>
            
            {/* Display de temperatura */}
            <div className="text-5xl font-bold text-black my-4">
              {temperature}°C
            </div>
            
            <p className="text-black opacity-90 mb-2">
              {currentWeather.description}
            </p>
            <p className="text-black opacity-80 text-sm mb-4">
              Rango: {currentWeather.tempRange}
            </p>
            
            <div className="text-black opacity-70 text-xs">
              ⚡ Los datos se actualizan automáticamente (simulando backend)
            </div>
          </div>
        </div>

        {/* Información de debug (puedes quitarlo luego) */}
        <div className="mt-8 text-center text-black opacity-60 text-sm">
          <p>Variable temporal: {temperature}°C → Clima: {weatherType}</p>
          <p>Próxima actualización automática en 5 segundos</p>
        </div>
      </div>
    </div>
  );
};

export default EnhancedWeatherPage;