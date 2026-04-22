import { useRef, useEffect, useState, useMemo } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN;

const Map = () => {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const [lng] = useState(121.5654);
  const [lat] = useState(25.0330);
  const [zoom] = useState(12);
  const [mode, setMode] = useState('TDX Live Speed');
  const [status, setStatus] = useState('Live Feed Active');

  // Persist the random line coordinates so they don't jump around when mode changes
  const baseCoordinates = useMemo(() => {
    const coords = [];
    for (let i = 0; i < 200; i++) {
      const startLng = 121.5654 + (Math.random() - 0.5) * 0.1;
      const startLat = 25.0330 + (Math.random() - 0.5) * 0.1;
      const endLng = startLng + (Math.random() - 0.5) * 0.01;
      const endLat = startLat + (Math.random() - 0.5) * 0.01;
      coords.push([[startLng, startLat], [endLng, endLat]]);
    }
    return coords;
  }, []);

  const getSpeedColor = (speed) => {
    if (speed < 20) return '#ef4444'; // Red
    if (speed < 40) return '#eab308'; // Yellow
    return '#22c55e'; // Green
  };

  // TDX CongestionLevel: 1=順暢, 2=車多, 3=車多擁擠, 4=壅塞, 5=停滯
  const getCongestionColor = (level) => {
    if (level <= 2) return '#22c55e';
    if (level === 3) return '#eab308';
    return '#ef4444';
  };

  const updateMapSource = (features) => {
    if (map.current && map.current.isStyleLoaded()) {
      const source = map.current.getSource('mock-traffic');
      if (source) {
        source.setData({ type: 'FeatureCollection', features });
      }
    }
  };

  const setMapData = (data) => {
    if (map.current && map.current.isStyleLoaded()) {
      const source = map.current.getSource('mock-traffic');
      if (source) source.setData(data);
    }
  };

  useEffect(() => {
    if (map.current) return;

    const mapInstance = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [lng, lat],
      zoom: zoom,
    });

    map.current = mapInstance;

    mapInstance.on('error', (e) => {
      console.error('Mapbox error:', e.error);
    });

    mapInstance.on('load', () => {
      console.log('Map loaded.');

      mapInstance.addSource('mock-traffic', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });

      mapInstance.addLayer({
        id: 'traffic-glow',
        type: 'line',
        source: 'mock-traffic',
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: {
          'line-color': ['get', 'color'],
          'line-width': 8,
          'line-opacity': 0.4,
          'line-blur': 5,
        },
      });

      mapInstance.addLayer({
        id: 'traffic-core',
        type: 'line',
        source: 'mock-traffic',
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: {
          'line-color': ['get', 'color'],
          'line-width': 2,
          'line-opacity': 1.0,
        },
      });

    });

    return () => {
      mapInstance.remove();
      map.current = null;
    };
  }, [lng, lat, zoom, baseCoordinates]);

  // Effect to update map data when mode changes
  useEffect(() => {
    let cancelled = false;

    const run = async () => {
      // Wait until the style is loaded (the map 'load' handler above
      // adds the source/layers); poll briefly if we got here first.
      if (!map.current) return;
      while (!cancelled && !map.current.isStyleLoaded()) {
        await new Promise(r => setTimeout(r, 100));
      }
      if (cancelled) return;

      if (mode === 'TDX Live Speed') {
        setStatus('Loading Live Feed...');
        try {
          const res = await fetch('http://localhost:8000/live-traffic');
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const geojson = await res.json();
          if (cancelled) return;

          geojson.features.forEach(f => {
            const level = f.properties.congestion_level;
            f.properties.color = getCongestionColor(level);
          });
          setMapData(geojson);
          const count = geojson.features.length;
          const stamp = geojson.updated_at
            ? ` · ${new Date(geojson.updated_at).toLocaleTimeString()}`
            : '';
          setStatus(`Live Feed Active (${count} links)${stamp}`);
        } catch (error) {
          console.error('Live traffic fetch failed:', error);
          setStatus('Live Feed Error (Fallback)');
          const features = baseCoordinates.map(coords => {
            const speed = Math.random() * 80;
            return {
              type: 'Feature',
              properties: { speed, color: getSpeedColor(speed) },
              geometry: { type: 'LineString', coordinates: coords }
            };
          });
          updateMapSource(features);
        }
        return;
      }

      if (mode === 'STGCN Prediction') {
        setStatus('Fetching AI Prediction...');
        try {
          const res = await fetch('http://localhost:8000/predict-geojson', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
          });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const geojson = await res.json();
          if (cancelled) return;
          if (!geojson.features || geojson.features.length === 0) {
            throw new Error('No predictions returned');
          }

          geojson.features.forEach(f => {
            const level = f.properties.congestion_level;
            // Backend uses 1=free, 2=moderate, 3=congested
            f.properties.color =
              level === 3 ? '#ef4444' :
                level === 2 ? '#eab308' : '#22c55e';
          });
          setMapData(geojson);
          setStatus(
            `STGCN Prediction Loaded (${geojson.matched}/${geojson.num_nodes} live nodes)`
          );
        } catch (error) {
          console.error("Prediction API failed:", error);
          setStatus('Model API Error (Fallback)');
          updateMapSource(baseCoordinates.map(coords => ({
            type: 'Feature',
            properties: { speed: 10, color: '#ef4444' },
            geometry: { type: 'LineString', coordinates: coords }
          })));
        }
        return;
      } else {
        setStatus('Heatmap Active');
        const features = baseCoordinates.map(coords => {
          const speed = Math.random() * 30; // Mostly congested/moderate
          return {
            type: 'Feature',
            properties: { speed, color: getSpeedColor(speed) },
            geometry: { type: 'LineString', coordinates: coords }
          };
        });
        updateMapSource(features);
      }
    };

    run();
    return () => { cancelled = true; };
  }, [mode, baseCoordinates]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div ref={mapContainer} style={{ position: 'absolute', inset: 0 }} />
      <div className="absolute top-4 left-4 bg-slate-800/80 backdrop-blur-md p-4 rounded-xl border border-slate-700 shadow-2xl z-10 text-slate-200 w-80">
        <h2 className="text-xl font-bold mb-2 text-blue-400">GeoAI Traffic</h2>
        <p className="text-sm text-slate-400 mb-4">Real-time Spatio-Temporal Prediction</p>

        <div className="space-y-4">
          <div className="bg-slate-900/50 p-3 rounded-lg">
            <div className="text-xs text-slate-500 uppercase tracking-widest mb-1">Status</div>
            <div className="flex items-center gap-2">
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
              </span>
              <span className="text-sm font-medium">{status}</span>
            </div>
          </div>

          <div className="bg-slate-900/50 p-3 rounded-lg">
            <div className="text-xs text-slate-500 uppercase tracking-widest mb-1">Mode</div>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              className="w-full bg-slate-800 border border-slate-600 rounded p-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="TDX Live Speed">TDX Live Speed</option>
              <option value="STGCN Prediction">STGCN Prediction</option>
              <option value="Movement Heatmap">Movement Heatmap</option>
            </select>
          </div>

          <div className="bg-slate-900/50 p-3 rounded-lg">
            <div className="text-xs text-slate-500 uppercase tracking-widest mb-2">Traffic Legend</div>
            <div className="space-y-2">
              <div className="flex items-center gap-3">
                <span className="inline-block w-6 h-1.5 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]"></span>
                <span className="text-sm font-medium text-slate-300">Free Flow (順暢)</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="inline-block w-6 h-1.5 rounded-full bg-yellow-500 shadow-[0_0_8px_rgba(234,179,8,0.6)]"></span>
                <span className="text-sm font-medium text-slate-300">Moderate (車多)</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="inline-block w-6 h-1.5 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]"></span>
                <span className="text-sm font-medium text-slate-300">Congested (壅塞)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Map;
