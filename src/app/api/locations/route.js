// src/app/api/locations/route.js
import { promises as fs } from 'fs';
import path from 'path';

// Ruta al archivo JSON - se creará en la carpeta 'data' en la raíz del proyecto
const locationsFilePath = path.join(process.cwd(), 'data', 'locations.json');

// POST - Guardar nueva ubicación
export async function POST(request) {
  try {
    const locationData = await request.json();
    
    console.log('📨 Receiving location:', locationData);
    
    // Validar datos básicos
    if (typeof locationData.lat !== 'number' || typeof locationData.lng !== 'number') {
      return Response.json({ 
        error: 'Invalid or missing coordinates',
        received: locationData 
      }, { status: 400 });
    }

    // Asegurar que el directorio 'data' existe
    try {
      await fs.mkdir(path.dirname(locationsFilePath), { recursive: true });
      console.log('✅ Data directory created/verified');
    } catch (error) {
      console.error('❌ Error creating directory:', error);
    }

    // Leer archivo existente o crear array vacío
    let locations = [];
    try {
      const fileData = await fs.readFile(locationsFilePath, 'utf8');
      locations = JSON.parse(fileData);
      console.log(`📖 Reading ${locations.length} existing locations`);
    } catch (error) {
      // Archivo no existe, es normal la primera vez
      console.log('📝 Creating new locations file');
      locations = [];
    }

    // Crear nueva ubicación con ID y timestamp
    const newLocation = {
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
      lat: locationData.lat,
      lng: locationData.lng,
      address: locationData.address || 'Address not available',
      type: locationData.type || 'user_selection',
      date: locationData.date || null
    };

    // Agregar al array
    // locations.push(newLocation);
    // console.log('➕ New location added:', newLocation);

    // Guardar en archivo JSON
    await fs.writeFile(locationsFilePath, JSON.stringify(newLocation, null, 2));
    console.log('💾 File saved successfully');

    return Response.json({ 
      success: true, 
      message: 'Location saved successfully',
      location: newLocation,
      //totalLocations: locations.length
    });

  } catch (error) {
    console.error('❌ Error saving location:', error);
    return Response.json({ 
      error: 'Internal server error',
      details: error.message 
    }, { status: 500 });
  }
}

// GET - Obtener todas las ubicaciones
export async function GET() {
  try {
    console.log('📨 Requesting locations list');
    
    // Leer archivo de ubicaciones
    const fileData = await fs.readFile(locationsFilePath, 'utf8');
    const locations = JSON.parse(fileData);
    
    console.log(`📊 Sending ${locations.length} locations`);
    
    return Response.json(locations);
    
  } catch (error) {
    // Si el archivo no existe, devolver array vacío
    console.log('📭 No locations saved yet');
    return Response.json([]);
  }
}
