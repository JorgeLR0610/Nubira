// src/app/page.js
'use client';

import { useState } from 'react';
import ProfessionalMapModal from '../components/ProfessionalMapModal';

export default function Home() {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedLocation, setSelectedLocation] = useState(null);

  const handleOpenModal = () => {
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
  };

  const handleLocationSelect = (location) => {
    console.log('📍 Ubicación seleccionada:', location);
    setSelectedLocation(location);
    setShowButton(false); 
  };

  const [showButton, setShowButton] = useState(true);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-950 relative overflow-hidden">
      {/* Fondo animado de planetas y estrellas */}
      <div className="absolute inset-0 pointer-events-none z-0">
        {/* Planetas */}
        <div className="planet planet-1"></div>
        <div className="planet planet-2"></div>
        <div className="planet planet-3"></div>
        <div className="planet planet-4"></div>
        {/* Estrellas */}
        <div className="star star-1"></div>
        <div className="star star-2"></div>
        <div className="star star-3"></div>
        <div className="star star-4"></div>
        <div className="star star-5"></div>
        <div className="star star-6"></div>
        <div className="star star-7"></div>
        <div className="star star-8"></div>
        <div className="star star-9"></div>
        <div className="star star-10"></div>
        <div className="star star-11"></div>
        <div className="star star-12"></div>
        <div className="star star-13"></div>
        <div className="star star-14"></div>
        <div className="star star-15"></div>
        <div className="star star-16"></div>
      </div>

      <div className="text-center z-10 relative">
        {showButton && (
          <button 
            onClick={handleOpenModal}
            className="px-8 py-4 bg-blue-600 text-white rounded-lg text-xl font-semibold hover:bg-blue-700 transition transform duration-300 hover:scale-105 shadow-lg mb-6 focus:outline-none focus:ring-2 focus:ring-blue-400"
          >
            🗺️ Abrir Mapa Profesional
          </button>
        )}
        
        
        {selectedLocation && (
          <div className="bg-white p-6 rounded-lg shadow-lg mt-6 max-w-md mx-auto border-2 border-green-200">
            <h3 className="font-semibold text-green-600 text-lg mb-3 flex items-center justify-center gap-2">
              <span>✅</span>
              Ubicación Confirmada
            </h3>
            <div className="text-left space-y-3">
              <div>
                <span className="font-medium text-gray-700">Coordenadas:</span>
                <div className="bg-gray-100 p-2 rounded mt-1 font-mono text-black">
                  {selectedLocation.lat.toFixed(6)}, {selectedLocation.lng.toFixed(6)}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-700">Dirección:</span>
                <div className="bg-gray-100 p-2 rounded mt-1 text-sm text-gray-600 max-h-20 overflow-y-auto">
                  {selectedLocation.address}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-700">Fecha del Pronóstico:</span>
                <div className="bg-gray-100 p-2 rounded mt-1 text-sm text-gray-600">
                  {selectedLocation.date
                    ? new Date(selectedLocation.date).toLocaleDateString()
                    : 'No seleccionada'}
                </div>
              </div>
            </div>
            
            <div className="flex gap-4">
              <button 
                onClick={handleOpenModal}
                className="mt-4 px-4 py-2 bg-blue-500 text-white rounded transition transform duration-300 hover:scale-105 hover:bg-blue-600  text-sm "
              >
                🗺️ Cambiar Ubicación
              </button>
              <a
                href={`/pronostico`}
                className="mt-4 inline-block px-4 py-2 bg-purple-600 text-white rounded transition transform duration-300 hover:scale-105 hover:bg-purple-700 text-sm"
              >
                🔮 Pronóstico del Clima
              </a>
            </div>
          </div>
        )}

        {!selectedLocation && (
          <div className="bg-white p-6 rounded-lg shadow-lg mt-6 max-w-md mx-auto border-2 border-blue-200">
            <div className="text-center text-gray-600">
              <div className="text-4xl mb-2">🌎</div>
              <p className="font-medium">No hay ubicación seleccionada</p>
              <p className="text-sm mt-1">Haz click en el botón de arriba para seleccionar una ubicación en el mapa profesional</p>
            </div>
          </div>
        )}
      </div>

      {/* Modal Profesional de Mapa */}
      <ProfessionalMapModal 
        isOpen={isModalOpen}
        onClose={handleCloseModal}
        onLocationSelect={handleLocationSelect}
      />

      <style jsx>{`
        .planet {
          position: absolute;
          border-radius: 50%;
          opacity: 0.9;
          animation-timing-function: linear;
        }
        .planet-1 {
          width: 50px;
          height: 50px;
          background: radial-gradient(circle at 30% 30%, #fbbf24 70%, #f59e42 100%);
          top: 10%;
          left: 5%;
          animation: move1 12s infinite alternate;
        }
        .planet-2 {
          width: 30px;
          height: 30px;
          background: radial-gradient(circle at 60% 60%, #60a5fa 70%, #2563eb 100%);
          top: 70%;
          left: 80%;
          animation: move2 10s infinite alternate;
        }
        .planet-3 {
          width: 40px;
          height: 40px;
          background: radial-gradient(circle at 50% 50%, #a78bfa 70%, #7c3aed 100%);
          top: 40%;
          left: 60%;
          animation: move3 14s infinite alternate;
        }
        .planet-4 {
          width: 20px;
          height: 20px;
          background: radial-gradient(circle at 70% 70%, #34d399 70%, #059669 100%);
          top: 80%;
          left: 20%;
          animation: move4 9s infinite alternate;
        }
        @keyframes move1 {
          0% { transform: translateY(0) translateX(0);}
          100% { transform: translateY(100px) translateX(60vw);}
        }
        @keyframes move2 {
          0% { transform: translateY(0) translateX(0);}
          100% { transform: translateY(-120px) translateX(-70vw);}
        }
        @keyframes move3 {
          0% { transform: translateY(0) translateX(0);}
          100% { transform: translateY(80px) translateX(-40vw);}
        }
        @keyframes move4 {
          0% { transform: translateY(0) translateX(0);}
          100% { transform: translateY(-60px) translateX(50vw);}
        }
        .star {
          position: absolute;
          width: 6px;
          height: 6px;
          background: white;
          border-radius: 50%;
          opacity: 0.8;
          animation: twinkle 3s infinite alternate;
        }

        /* Posiciones diferentes */
        .star-1 { top: 20%; left: 15%; animation-delay: 0s; }
        .star-2 { top: 40%; left: 70%; animation-delay: 0.5s; }
        .star-3 { top: 60%; left: 30%; animation-delay: 1s; }
        .star-4 { top: 80%; left: 50%; animation-delay: 1.5s; }
        .star-5 { top: 10%; left: 80%; animation-delay: 2s; }
        .star-6 { top: 25%; left: 60%; animation-delay: 3s; }
        .star-7 { top: 35%; left: 20%; animation-delay: 1s; }
        .star-8 { top: 55%; left: 85%; animation-delay: 4s; }
        .star-9 { top: 70%; left: 10%; animation-delay: 0.5s; }
        .star-10 { top: 85%; left: 75%; animation-delay: 2.5s; }
        .star-11 { top: 15%; left: 40%; animation-delay: 1.2s; }
        .star-12 { top: 45%; left: 10%; animation-delay: 0.8s; }
        .star-13 { top: 65%; left: 60%; animation-delay: 2.2s; }
        .star-14 { top: 75%; left: 25%; animation-delay: 3.5s; }
        .star-15 { top: 30%; left: 90%; animation-delay: 1.8s; }
        .star-16 { top: 50%; left: 45%; animation-delay: 0.3s; }

        @keyframes twinkle {
          0% { opacity: 0.2; transform: scale(0.8); }
          100% { opacity: 1; transform: scale(1.2); }
        }
      `}</style>


    </div>
  );
}