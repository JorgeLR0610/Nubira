// src/app/page.js
'use client';

import { useState, useRef, useEffect } from 'react';
import ProfessionalMapModal from '../components/ProfessionalMapModal';
import * as THREE from 'three';

export default function Home() {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [showButton, setShowButton] = useState(true);
  const mountRef = useRef(null);

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

  // Efecto para los planetas 3D con Three.js
  useEffect(() => {
    if (!mountRef.current) return;

    // Scene
    const scene = new THREE.Scene();
    
    // Camera
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 15;
    
    // Renderer
    const renderer = new THREE.WebGLRenderer({ 
      alpha: true,
      antialias: true 
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0);
    mountRef.current.appendChild(renderer.domElement);

    // Texture loader
    const textureLoader = new THREE.TextureLoader();

    // URLs de texturas de planetas (puedes reemplazarlas con tus propias texturas)
    const textureUrls = [
      'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_atmos_2048.jpg',
      'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/mars_1k_color.jpg',
      'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/jupiter_1k_color.jpg',
      'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/saturn_1k_color.jpg'
    ];

    // Crear planetas con diferentes texturas
    const createPlanet = (radius, textureUrl, position, rotationSpeed) => {
      const geometry = new THREE.SphereGeometry(radius, 32, 32);
      
      // Material con textura
      const material = new THREE.MeshBasicMaterial({ 
        map: textureLoader.load(textureUrl),
        transparent: true,
        opacity: 0.9
      });
      
      const planet = new THREE.Mesh(geometry, material);
      
      planet.position.set(position.x, position.y, position.z);
      scene.add(planet);
      
      return { planet, rotationSpeed, initialPosition: { ...position } };
    };

    // Crear los 4 planetas
    const planets = [
      createPlanet(0.8, textureUrls[0], { x: -6, y: 1, z: -10 }, 0.005),
      createPlanet(0.6, textureUrls[1], { x: 5, y: -2, z: -12 }, 0.008),
      createPlanet(1.2, textureUrls[2], { x: 0, y: 3, z: -15 }, 0.003),
      createPlanet(1.0, textureUrls[3], { x: -4, y: -3, z: -8 }, 0.006)
    ];

    // Variables para animación de movimiento orbital
    let time = 0;

    // Animación
    function animate() {
      requestAnimationFrame(animate);
      
      time += 0.01;
      
      planets.forEach(({ planet, rotationSpeed, initialPosition }, index) => {
        // Rotación del planeta
        planet.rotation.y += rotationSpeed;
        
        // Movimiento orbital
        const orbitSpeed = 0.2 + index * 0.1;
        const orbitRadius = 2 + index * 0.5;
        planet.position.x = initialPosition.x + Math.cos(time * orbitSpeed) * orbitRadius;
        planet.position.y = initialPosition.y + Math.sin(time * orbitSpeed) * orbitRadius * 0.5;
      });
      
      renderer.render(scene, camera);
    }
    
    animate();

    // Handle resize
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-950 relative overflow-hidden">
      {/* Fondo animado de planetas 3D con Three.js */}
      <div ref={mountRef} className="absolute inset-0 pointer-events-none z-0" />
      
      {/* Estrellas de fondo (manteniendo las originales) */}
      <div className="absolute inset-0 pointer-events-none z-0">
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

      <style jsx>{`
        .star {
          position: absolute;
          width: 6px;
          height: 6px;
          background: white;
          border-radius: 50%;
          opacity: 0.8;
          animation: twinkle 3s infinite alternate;
        }

        /* Posiciones diferentes para las estrellas */
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