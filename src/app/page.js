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
                href="/paginaRes"
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

        <div className="min-h-screen flex items-center justify-center bg-gray-950 relative overflow-hidden">
      {/* Fondo animado de planetas 3D con texturas */}
      <ThreePlanets /> {/* o <Planets3D /> para la versión simple */}
      
      {/* Estrellas (mantener igual) */}
      <div className="absolute inset-0 pointer-events-none z-0">
        <div className="star star-1"></div>
        <div className="star star-2"></div>
        {/* ... resto de estrellas igual */}
      </div>

      {/* El resto de tu código permanece igual */}
      <div className="text-center z-10 relative">
        {/* ... tu contenido existente */}
      </div>
    </div>


    </div>
  );
}