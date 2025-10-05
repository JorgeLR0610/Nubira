// app/paginaRes/page.js
"use client"
import { useState, useEffect } from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

// Registrar componentes de Chart.js para gráfica de barras
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const EnhancedWeatherPage = () => {
  const [weatherType, setWeatherType] = useState('normal');
  const [temperature, setTemperature] = useState(22);
  const [precipitation, setPrecipitation] = useState(0);
  const [windSpeed, setWindSpeed] = useState(5);
  const [weatherHistory, setWeatherHistory] = useState([]);
  const [sampleCount, setSampleCount] = useState(0);

  // Función para determinar el tipo de clima
  const getWeatherTypeFromConditions = (temp, precip, wind) => {
    if (precip > 70 && wind > 30 && temp > 15) return 'thunderstorm';
    if (precip > 60 && temp <= 5) return 'hail';
    if (precip > 50 && temp <= 0) return 'snowy';
    if (temp >= 30) return 'sunny';
    if (temp >= 20) return 'normal';
    if (temp >= 10) return 'cloudy';
    return 'rainy';
  };

  // Simular datos del backend
  const simulateBackendData = () => {
    const temps = [-5, -2, 0, 2, 5, 8, 12, 15, 18, 22, 25, 28, 32, 35];
    const randomTemp = temps[Math.floor(Math.random() * temps.length)];
    const randomPrecip = Math.floor(Math.random() * 101);
    const randomWind = Math.floor(Math.random() * 51);
    
    return { 
      temp: randomTemp, 
      precip: randomPrecip, 
      wind: randomWind,
      timestamp: new Date().toLocaleTimeString(),
      id: Date.now() + Math.random() // ID único para cada muestra
    };
  };

  // Configuración de la gráfica de barras
  const getChartData = () => {
    const maxSamples = 5;
    const currentSamples = weatherHistory.slice(-maxSamples);
    
    const labels = currentSamples.map((data, index) => {
      const globalIndex = weatherHistory.length - maxSamples + index + 1;
      return `M${globalIndex}`;
    });
    
    return {
      labels,
      datasets: [
        {
          label: 'Temperatura (°C)',
          data: currentSamples.map(data => data.temp),
          backgroundColor: 'rgba(255, 99, 132, 0.8)',
          borderColor: 'rgb(255, 99, 132)',
          borderWidth: 1,
          yAxisID: 'y',
        },
        {
          label: 'Precipitación (%)',
          data: currentSamples.map(data => data.precip),
          backgroundColor: 'rgba(54, 162, 235, 0.8)',
          borderColor: 'rgb(54, 162, 235)',
          borderWidth: 1,
          yAxisID: 'y1',
        },
        {
          label: 'Viento (km/h)',
          data: currentSamples.map(data => data.wind),
          backgroundColor: 'rgba(75, 192, 192, 0.8)',
          borderColor: 'rgb(75, 192, 192)',
          borderWidth: 1,
          yAxisID: 'y2',
        }
      ]
    };
  };

  const chartOptions = {
    responsive: true,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        ticks: {
          color: 'white'
        }
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        ticks: {
          color: 'white'
        },
        title: {
          display: true,
          text: 'Temperatura (°C)',
          color: 'white'
        }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        grid: {
          drawOnChartArea: false,
        },
        ticks: {
          color: 'white'
        },
        title: {
          display: true,
          text: 'Precipitación (%)',
          color: 'white'
        },
        max: 100
      },
      y2: {
        type: 'linear',
        display: false,
      },
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: 'white',
          font: {
            size: 12
          }
        }
      },
      title: {
        display: true,
        text: `Carrusel de Muestras - Mostrando últimas 5 de ${sampleCount} totales`,
        color: 'white',
        font: {
          size: 16,
          weight: 'bold'
        }
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'white',
        bodyColor: 'white',
        borderColor: 'rgba(255, 255, 255, 0.2)',
        borderWidth: 1
      }
    },
    animation: {
      duration: 800,
      easing: 'easeOutQuart'
    }
  };

  useEffect(() => {
    const updateWeatherData = () => {
      const newData = simulateBackendData();
      setTemperature(newData.temp);
      setPrecipitation(newData.precip);
      setWindSpeed(newData.wind);
      setWeatherType(getWeatherTypeFromConditions(newData.temp, newData.precip, newData.wind));
      
      // Incrementar el contador de muestras
      setSampleCount(prev => prev + 1);
      
      // Agregar nueva muestra al historial
      setWeatherHistory(prev => {
        return [...prev, newData];
      });
    };

    // Datos iniciales
    updateWeatherData();

    // Actualizar cada 5 segundos
    const interval = setInterval(updateWeatherData, 5000);

    return () => clearInterval(interval);
  }, []);

  // Obtener las últimas 5 muestras para mostrar
  const currentSamples = weatherHistory.slice(-5);

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
        
        {/* Contenedor principal con información del clima y gráfica */}
        <div className="flex flex-col lg:flex-row gap-8 max-w-6xl mx-auto">
          
          {/* Tarjeta de información del clima */}
          <div className="flex-1">
            <div className="bg-white/30 backdrop-blur-lg rounded-2xl p-8 border border-white/40 shadow-xl">
              <div className="text-center">
                <div className="text-6xl mb-4">{currentWeather.icon}</div>
                <h2 className={`text-3xl font-bold mb-2 ${currentWeather.textColor}`}>
                  {currentWeather.name}
                </h2>
                
                <div className={`text-5xl font-bold my-4 ${currentWeather.textColor}`}>
                  {temperature}°C
                </div>
                
                <p className={`opacity-90 mb-2 ${currentWeather.textColor}`}>
                  {currentWeather.description}
                </p>
                <p className={`opacity-80 text-sm mb-4 ${currentWeather.textColor}`}>
                  Range: {currentWeather.tempRange}
                </p>
                
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
          </div>

          {/* Gráfica de Barras - Carrusel */}
          <div className="flex-1">
            <div className="bg-black/40 backdrop-blur-lg rounded-2xl p-6 border border-white/20 shadow-xl">
              {weatherHistory.length > 0 ? (
                <div>
                  <Bar data={getChartData()} options={chartOptions} />
                  
                  {/* Información del carrusel */}
                  <div className="mt-6 grid grid-cols-3 gap-4">
                    <div className="bg-red-500/20 rounded-lg p-3 text-center">
                      <div className="text-white text-sm">Temp. Actual</div>
                      <div className="text-white font-bold text-lg">
                        {temperature}°C
                      </div>
                    </div>
                    <div className="bg-blue-500/20 rounded-lg p-3 text-center">
                      <div className="text-white text-sm">Precip. Actual</div>
                      <div className="text-white font-bold text-lg">
                        {precipitation}%
                      </div>
                    </div>
                    <div className="bg-green-500/20 rounded-lg p-3 text-center">
                      <div className="text-white text-sm">Viento Actual</div>
                      <div className="text-white font-bold text-lg">
                        {windSpeed} km/h
                      </div>
                    </div>
                  </div>

                  {/* Información del carrusel */}
                  <div className="mt-4 text-center">
                    <div className="text-white text-sm">
                      {`Mostrando muestras ${Math.max(1, sampleCount - 4)} a ${sampleCount} de ${sampleCount} totales`}
                    </div>
                    <div className="text-white text-xs opacity-70 mt-1">
                      {sampleCount > 5 
                        ? 'Carrusel activo - Las muestras más antiguas se desplazan fuera de la vista'
                        : 'Completando primeras 5 muestras...'
                      }
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center h-64">
                  <p className="text-white text-lg">Cargando datos del clima...</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnhancedWeatherPage;