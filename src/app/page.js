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
    console.log('📍 Location selected:', location);
    setSelectedLocation(location);
    setShowButton(false); 
  };

  const [showButton, setShowButton] = useState(true);

  return (
    <div className="min-h-screen flex items-center justify-center space-bg relative overflow-hidden">
      {/* Fondo animado de planetas y estrellas */}
      <div className="absolute inset-0 pointer-events-none z-0">
        {/* Planetas */}
        {/* PLANETA 1 - Tipo Tierra/Planeta Rocoso */}
        <div
          className="absolute rounded-full w-[50px] h-[50px] top-[10%] left-[5%] animate-move1 overflow-hidden z-5"
          style={{
            background: 'radial-gradient(circle at 30% 30%, #fbbf24 70%, #f59e42 100%)',
          }}
        >
          <div
            className="w-full h-full rounded-full opacity-70"
            style={{
              background: `
                radial-gradient(circle at 20% 30%, rgba(139, 69, 19, 0.3) 0%, transparent 20%),
                radial-gradient(circle at 70% 60%, rgba(105, 105, 105, 0.4) 0%, transparent 25%),
                radial-gradient(circle at 40% 80%, rgba(178, 34, 34, 0.25) 0%, transparent 15%),
                radial-gradient(circle at 80% 20%, rgba(47, 79, 79, 0.35) 0%, transparent 18%),
                radial-gradient(circle at 60% 40%, rgba(160, 82, 45, 0.3) 0%, transparent 22%)
              `,
            }}
          ></div>
        </div>

        {/* PLANETA 2 - Tipo Gaseoso con Bandas */}
        <div
          className="absolute rounded-full w-[30px] h-[30px] top-[70%] left-[80%] animate-move2 overflow-hidden z-5"
          style={{ background: 'radial-gradient(circle at 60% 60%, #60a5fa 70%, #2563eb 100%)' }}
        >
          <div
            className="w-full h-full rounded-full opacity-80"
            style={{
              background: `
                linear-gradient(90deg, 
                  transparent 0%, 
                  rgba(255,255,255,0.1) 10%, 
                  transparent 20%,
                  rgba(37, 99, 235, 0.3) 30%,
                  transparent 40%,
                  rgba(96, 165, 250, 0.4) 50%,
                  transparent 60%,
                  rgba(255,255,255,0.15) 70%,
                  transparent 80%,
                  rgba(37, 99, 235, 0.25) 90%,
                  transparent 100%
                ),
                radial-gradient(circle at 30% 40%, rgba(255,255,255,0.2) 0%, transparent 30%)
              `,
            }}
          ></div>
        </div>

        {/* PLANETA 3 - Con cráteres y montañas */}
        <div
          className="absolute rounded-full w-[40px] h-[40px] top-[40%] left-[60%] animate-move3 overflow-hidden z-5"
          style={{ background: 'radial-gradient(circle at 50% 50%, #a78bfa 70%, #7c3aed 100%)' }}
        >
          <div
            className="w-full h-full rounded-full opacity-60 "
            style={{
              background: `
                radial-gradient(circle at 25% 35%, rgba(0,0,0,0.3) 0%, rgba(0,0,0,0.4) 8%, transparent 12%),
                radial-gradient(circle at 65% 70%, rgba(0,0,0,0.25) 0%, rgba(0,0,0,0.35) 6%, transparent 10%),
                radial-gradient(circle at 45% 20%, rgba(0,0,0,0.2) 0%, rgba(0,0,0,0.3) 5%, transparent 8%),
                radial-gradient(circle at 80% 50%, rgba(255,255,255,0.1) 0%, transparent 7%),
                radial-gradient(circle at 30% 80%, rgba(0,0,0,0.15) 0%, transparent 6%)
              `,
              boxShadow: 'inset 0 0 20px rgba(0,0,0,0.3)'
            }}
          ></div>
        </div>

        {/* PLANETA 4 - Tipo Volcánico */}
        <div
          className="absolute rounded-full w-[20px] h-[20px] top-[80%] left-[20%] animate-move4 overflow-hidden z-5"
          style={{ background: 'radial-gradient(circle at 70% 70%, #34d399 70%, #059669 100%)' }}
        >
          <div
            className="w-full h-full rounded-full opacity-80"
            style={{
              background: `
                radial-gradient(circle at 40% 60%, rgba(220, 38, 38, 0.4) 0%, transparent 25%),
                radial-gradient(circle at 60% 30%, rgba(251, 146, 60, 0.3) 0%, transparent 20%),
                radial-gradient(circle at 20% 80%, rgba(120, 53, 15, 0.35) 0%, transparent 18%),
                radial-gradient(circle at 75% 65%, rgba(154, 52, 18, 0.4) 0%, transparent 15%)
              `,
            }}
          ></div>
        </div>
        {/* Estrellas */}
        <div className="star star-1 z-1"></div>
        <div className="star star-2 z-1"></div>
        <div className="star star-3 z-1"></div>
        <div className="star star-4 z-1"></div>
        <div className="star star-5 z-1"></div>
        <div className="star star-6 z-1"></div>
        <div className="star star-7 z-1"></div>
        <div className="star star-8 z-1"></div>
        <div className="star star-9 z-1"></div>
        <div className="star star-10 z-1"></div>
        <div className="star star-11 z-1"></div>
        <div className="star star-12 z-1"></div>
        <div className="star star-13 z-1"></div>
        <div className="star star-14 z-1"></div>
        <div className="star star-15 z-1"></div>
        <div className="star star-16 z-1"></div>
        <div className="star star-17 z-1"></div>
        <div className="star star-18 z-1"></div>
        <div className="star star-19 z-1"></div>
        <div className="star star-20 z-1"></div>
        <div className="star star-21 z-1"></div>
        <div className="star star-22 z-1"></div>
        <div className="star star-23 z-1"></div>
        <div className="star star-24 z-1"></div>
      </div>

      <div className="text-center z-10 relative">
        {showButton && (
          <button 
            onClick={handleOpenModal}
            className="px-8 py-4 bg-blue-600 text-white rounded-lg text-xl font-semibold hover:bg-blue-700 transition transform duration-300 hover:scale-105 shadow-lg mb-6 focus:outline-none focus:ring-2 focus:ring-blue-400"
          >
            🗺 Open Professional Map
          </button>
        )}
        
        
        {selectedLocation && (
          <div className="bg-white p-6 rounded-lg shadow-lg mt-6 max-w-md mx-auto border-2 border-green-200">
            <h3 className="font-semibold text-green-600 text-lg mb-3 flex items-center justify-center gap-2">
              <span>✅</span>
              Location Confirmed
            </h3>
            <div className="text-left space-y-3">
              <div>
                <span className="font-medium text-gray-700">Coordinates:</span>
                <div className="bg-gray-100 p-2 rounded mt-1 font-mono text-black">
                  {selectedLocation.lat.toFixed(6)}, {selectedLocation.lng.toFixed(6)}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-700">Address:</span>
                <div className="bg-gray-100 p-2 rounded mt-1 text-sm text-gray-600 max-h-20 overflow-y-auto">
                  {selectedLocation.address}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-700">Forecast Date:</span>
                <div className="bg-gray-100 p-2 rounded mt-1 text-sm text-gray-600">
                  {selectedLocation.date
                    ? new Date(selectedLocation.date).toLocaleDateString()
                    : 'Not selected'}
                </div>
              </div>
            </div>
            
            <div className="flex gap-4">
              <button 
                onClick={handleOpenModal}
                className="mt-4 px-4 py-2 bg-blue-500 text-white rounded transition transform duration-300 hover:scale-105 hover:bg-blue-600  text-sm "
              >
                🗺 Change Location
              </button>
              <a
                href="/paginaRes"
                className="mt-4 inline-block px-4 py-2 bg-purple-600 text-white rounded transition transform duration-300 hover:scale-105 hover:bg-purple-700 text-sm"
              >
                🔮 Weather Forecast
              </a>
            </div>
          </div>
        )}

        {!selectedLocation && (
          <div className="bg-white p-6 rounded-lg shadow-lg mt-6 max-w-md mx-auto border-2 border-blue-200">
            <div className="text-center text-gray-600">
              <div className="text-4xl mb-2">🌎</div>
              <p className="font-medium">No location selected</p>
              <p className="text-sm mt-1">Click the button above to select a location on the professional map</p>
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
    </div>
  );
}