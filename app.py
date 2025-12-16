"""
Sistema de Predicción de Congestión Vehicular - Chillán IA (MEJORADO)
Proyecto de Inteligencia Artificial - Universidad del Bío-Bío
Autores: Diego Loyola, Catalina Toro, Valentina Zúñiga

Versión Flask (sin dependencias problemáticas)
"""

from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json

app = Flask(__name__)

# Variables globales para el modelo
modelo = None
scaler = None
df_global = None
feature_cols = None
metricas = None

# ============================================================================
# GENERACIÓN DE DATASET SINTÉTICO
# ============================================================================

@st.cache_data
def generar_dataset_sintetico():
    """Genera un dataset sintético realista basado en patrones de tráfico de Chillán."""
    np.random.seed(42)
    
    segmentos = [
        {"id": "SEG001", "nombre": "Av. O'Higgins con Av. Collín", "lat": -36.613056, "lon": -72.111597, "long_m": 1200, "tipo": "arterial", "vel_max": 50},
        {"id": "SEG002", "nombre": "Av. O'Higgins con Av. Ecuador", "lat": -36.597912, "lon": -72.105803, "long_m": 1500, "tipo": "arterial", "vel_max": 50},
        {"id": "SEG003", "nombre": "Av. Collín con Av. Argentina", "lat": -36.617130, "lon": -72.097092, "long_m": 2000, "tipo": "arterial", "vel_max": 60},
        {"id": "SEG004", "nombre": "Av. Ecuador", "lat": -36.599991, "lon": -72.099999, "long_m": 1000, "tipo": "colectora", "vel_max": 40},
        {"id": "SEG005", "nombre": "Av. Alonso de Ercilla", "lat": -36.625223, "lon": -72.084603, "long_m": 1800, "tipo": "arterial", "vel_max": 50},
        {"id": "SEG006", "nombre": "Av. O'Higgins Chillán-Chillán Viejo", "lat": -36.620378, "lon": -72.121468, "long_m": 500, "tipo": "arterial", "vel_max": 40},
        {"id": "SEG007", "nombre": "Av. Collín (Clínica)", "lat": -36.614837, "lon": -72.106812, "long_m": 300, "tipo": "arterial", "vel_max": 30},
        {"id": "SEG008", "nombre": "5 de Abril", "lat": -36.608176, "lon": -72.101319, "long_m": 800, "tipo": "colectora", "vel_max": 30},
        {"id": "SEG009", "nombre": "Terminal de Buses María Teresa", "lat": -36.587964, "lon": -72.102649, "long_m": 600, "tipo": "colectora", "vel_max": 30},
        {"id": "SEG010", "nombre": "Av. Libertad Oriente", "lat": -36.608632, "lon": -72.090847, "long_m": 700, "tipo": "colectora", "vel_max": 40},
        {"id": "SEG011", "nombre": "Av. Andrés Bello", "lat": -36.592662, "lon": -72.070978, "long_m": 1200, "tipo": "arterial", "vel_max": 50},
    ]
    
    fechas = pd.date_range(start='2024-10-01', periods=60, freq='D')
    horas = range(24)
    dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    
    datos = []
    
    for fecha in fechas:
        dia_semana = dias_semana[fecha.dayofweek]
        temp_base = 15 + np.random.randn() * 5
        llueve_dia = np.random.random() < 0.15
        
        for hora in horas:
            lluvia_mm = np.random.exponential(3) if llueve_dia and np.random.random() < 0.4 else 0
            temperatura = temp_base + np.sin((hora - 6) * np.pi / 12) * 5 + np.random.randn()
            
            for seg in segmentos:
                vel_base = seg['vel_max'] * 0.85
                
                # Horas punta MÁS severas
                if hora in [8, 9, 18, 19]:
                    factor_hora = 0.35
                elif hora in [7, 10, 17, 20]:
                    factor_hora = 0.55
                elif hora in [12, 13, 14]:
                    factor_hora = 0.70
                elif hora in [0, 1, 2, 3, 4, 5]:
                    factor_hora = 1.0
                else:
                    factor_hora = 0.80
                
                if dia_semana in ['Sábado', 'Domingo']:
                    factor_hora = min(factor_hora * 1.2, 1.0)
                
                factor_lluvia = 1.0 - (lluvia_mm * 0.15) if lluvia_mm > 0 else 1.0
                factor_lluvia = max(factor_lluvia, 0.5)
                
                factor_critico = 1.0
                if seg['id'] in ['SEG006', 'SEG007', 'SEG008']:
                    factor_critico = 0.85
                
                velocidad = vel_base * factor_hora * factor_lluvia * factor_critico
                velocidad += np.random.randn() * 5
                velocidad = max(5, min(velocidad, seg['vel_max']))
                
                reduccion = (seg['vel_max'] - velocidad) / seg['vel_max']
                indice = reduccion * 100
                indice = max(0, min(indice, 100))
                
                if indice < 30:
                    categoria = "Fluido"
                elif indice < 60:
                    categoria = "Moderado"
                else:
                    categoria = "Congestionado"
                
                datos.append({
                    'fecha_hora': f"{fecha.date()} {hora:02d}:00:00",
                    'dia_semana': dia_semana,
                    'hora': hora,
                    'segmento_id': seg['id'],
                    'segmento_nombre': seg['nombre'],
                    'latitud': seg['lat'],
                    'longitud': seg['lon'],
                    'longitud_m': seg['long_m'],
                    'tipo_via': seg['tipo'],
                    'velocidad_maxima_kmh': seg['vel_max'],
                    'temperatura_c': round(temperatura, 1),
                    'lluvia_mm': round(lluvia_mm, 1),
                    'llueve': 1 if lluvia_mm > 0 else 0,
                    'velocidad_promedio_kmh': round(velocidad, 1),
                    'indice_congestion': round(indice, 1),
                    'categoria_flujo': categoria
                })
    
    return pd.DataFrame(datos)


# ============================================================================
# FUNCIONES DE MODELO
# ============================================================================

def cargar_datos():
    """Carga el dataset desde CSV o genera uno sintético."""
    try:
        df = pd.read_csv('dataset_congestion_vehicular_chillan.csv')
        print("✅ Dataset cargado desde archivo CSV")
    except FileNotFoundError:
        df = generar_dataset_sintetico()
        print("ℹ️ Usando dataset sintético generado")
    
    return df


def preprocesar_datos(df):
    """Preprocesa el dataset para entrenamiento."""
    df_proc = df.copy()
    df_proc = pd.get_dummies(df_proc, columns=['dia_semana', 'tipo_via'], drop_first=True)
    
    feature_cols = [col for col in df_proc.columns if col not in [
        'fecha_hora', 'segmento_id', 'segmento_nombre', 'latitud', 'longitud',
        'indice_congestion', 'categoria_flujo'
    ]]
    
    X = df_proc[feature_cols]
    y = df_proc['indice_congestion']
    
    return X, y, feature_cols


def entrenar_modelo(X, y):
    """Entrena el modelo MLPRegressor."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    modelo = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,  # Regularización L2
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False
    )
    
    # Entrenar
    with st.spinner("🧠 Entrenando Red Neuronal MLP..."):
        modelo.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_pred_train = modelo.predict(X_train_scaled)
    y_pred_test = modelo.predict(X_test_scaled)
    
    # Convertir predicciones a categorías
    def indice_a_categoria(indice):
        if indice < 30:
            return "Fluido"
        elif indice < 60:
            return "Moderado"
        else:
            return "Congestionado"
    
    y_pred_cat_train = [indice_a_categoria(i) for i in y_pred_train]
    y_pred_cat_test = [indice_a_categoria(i) for i in y_pred_test]
    
    # Métricas de regresión
    metricas = {
        'train_r2': round(r2_score(y_train, y_pred_train), 4),
        'train_mae': round(mean_absolute_error(y_train, y_pred_train), 2),
        'train_rmse': round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 2),
        'test_r2': round(r2_score(y_test, y_pred_test), 4),
        'test_mae': round(mean_absolute_error(y_test, y_pred_test), 2),
        'test_rmse': round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 2)
    }
    
    return modelo, scaler, metricas


def predecir_congestion(hora, dia_semana, temperatura, llueve, velocidad_promedio):
    """Realiza predicciones para todos los segmentos."""
    global modelo, scaler, df_global, feature_cols
    
    segmentos = df_global[['segmento_id', 'segmento_nombre', 'latitud', 'longitud', 
                           'tipo_via', 'velocidad_maxima_kmh', 'longitud_m']].drop_duplicates()
    
    predicciones = []
    
    for _, seg in segmentos.iterrows():
        entrada = {
            'hora': hora,
            'longitud_m': seg['longitud_m'],
            'velocidad_maxima_kmh': seg['velocidad_maxima_kmh'],
            'temperatura_c': temperatura,
            'lluvia_mm': 2.0 if llueve else 0.0,
            'llueve': 1 if llueve else 0,
            'velocidad_promedio_kmh': velocidad_promedio
        }
        
        for dia in ['Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']:
            entrada[f'dia_semana_{dia}'] = 1 if dia_semana == dia else 0
        
        for tipo in ['colectora', 'local']:
            entrada[f'tipo_via_{tipo}'] = 1 if seg['tipo_via'] == tipo else 0
        
        df_entrada = pd.DataFrame([entrada])
        
        # Asegurar que tenga todas las columnas
        for col in feature_cols:
            if col not in df_entrada.columns:
                df_entrada[col] = 0
        
        df_entrada = df_entrada[feature_cols]
        
        # Predicción estándar
        entrada_scaled = scaler.transform(df_entrada)
        prediccion = modelo.predict(entrada_scaled)[0]
        prediccion = max(0, min(prediccion, 100))
        
        # Intervalos de confianza (opcional, más lento)
        if usar_bootstrap:
            intervalos = predecir_con_bootstrap(modelo, scaler, X_train, y_train, df_entrada, n_bootstrap=5)
            pred_min = max(0, intervalos['percentil_5'][0])
            pred_max = min(100, intervalos['percentil_95'][0])
        else:
            pred_min = max(0, prediccion - 10)
            pred_max = min(100, prediccion + 10)
        
        if prediccion < 30:
            categoria = "Fluido"
            color = "green"
        elif prediccion < 60:
            categoria = "Moderado"
            color = "orange"
        else:
            categoria = "Congestionado"
            color = "red"
        
        predicciones.append({
            'segmento_id': seg['segmento_id'],
            'segmento_nombre': seg['segmento_nombre'],
            'latitud': seg['latitud'],
            'longitud': seg['longitud'],
            'prediccion': round(prediccion, 1),
            'pred_min': round(pred_min, 1),
            'pred_max': round(pred_max, 1),
            'categoria': categoria,
            'color': color,
            'velocidad_estimada': round(velocidad_estimada, 1)
        })
    
    return predicciones


# ============================================================================
# INICIALIZACIÓN
# ============================================================================

print("🔄 Cargando datos y entrenando modelo...")
df_global = cargar_datos()
X, y, feature_cols = preprocesar_datos(df_global)
modelo, scaler, metricas = entrenar_modelo(X, y)
print(f"✅ Modelo entrenado - R²: {metricas['test_r2']}, MAE: {metricas['test_mae']}")


# ============================================================================
# PLANTILLA HTML
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predicción Congestión Vehicular - Chillán</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.1em; opacity: 0.9; }
        
        .container {
            max-width: 1400px;
            margin: 30px auto;
            padding: 0 20px;
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 20px;
        }
        
        .sidebar {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            height: fit-content;
        }
        
        .sidebar h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.4em;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 600;
        }
        
        .form-group input,
        .form-group select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            transition: border 0.3s;
        }
        
        .form-group input:focus,
        .form-group select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .checkbox-group input[type="checkbox"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }
        
        .btn-predict {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .btn-predict:hover {
            transform: translateY(-2px);
        }
        
        .metrics {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
        }
        
        .metric-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding: 8px;
            background: #f9f9f9;
            border-radius: 6px;
        }
        
        .metric-label { color: #666; }
        .metric-value { font-weight: 700; color: #667eea; }
        
        .main-content {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        #map {
            width: 100%;
            height: 600px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .results {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .result-card {
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        .result-card.fluido { border-color: #4caf50; background: #f1f8e9; }
        .result-card.moderado { border-color: #ff9800; background: #fff3e0; }
        .result-card.congestionado { border-color: #f44336; background: #ffebee; }
        
        .result-card h3 { font-size: 0.9em; color: #333; margin-bottom: 8px; }
        .result-card .indice { font-size: 1.5em; font-weight: 700; }
        .result-card .categoria { font-size: 0.9em; opacity: 0.8; }
        
        .loading {
            display: none;
            text-align: center;
            padding: 50px;
            color: #667eea;
            font-size: 1.2em;
        }
        
        @media (max-width: 1024px) {
            .container { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚗 Sistema de Predicción de Congestión Vehicular</h1>
        <p>Chillán - Inteligencia Artificial | Universidad del Bío-Bío</p>
    </div>
    
    <div class="container">
        <div class="sidebar">
            <h2>⚙️ Configuración</h2>
            <form id="predictionForm">
                <div class="form-group">
                    <label>🕐 Hora del día (0-23)</label>
                    <input type="number" id="hora" min="0" max="23" value="8" required>
                </div>
                
                <div class="form-group">
                    <label>📅 Día de la semana</label>
                    <select id="dia_semana">
                        <option value="Lunes">Lunes</option>
                        <option value="Martes">Martes</option>
                        <option value="Miércoles">Miércoles</option>
                        <option value="Jueves">Jueves</option>
                        <option value="Viernes">Viernes</option>
                        <option value="Sábado">Sábado</option>
                        <option value="Domingo">Domingo</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>🌡️ Temperatura (°C)</label>
                    <input type="number" id="temperatura" min="5" max="35" value="18" required>
                </div>
                
                <div class="form-group checkbox-group">
                    <input type="checkbox" id="llueve">
                    <label for="llueve">🌧️ ¿Está lloviendo?</label>
                </div>
                
                <div class="form-group">
                    <label>🚙 Velocidad promedio (km/h)</label>
                    <input type="number" id="velocidad" min="10" max="80" value="40" required>
                </div>
                
                <button type="submit" class="btn-predict">🔮 Realizar Predicción</button>
            </form>
            
            <div class="metrics">
                <h2>📊 Métricas del Modelo</h2>
                <div class="metric-item">
                    <span class="metric-label">R² (Test)</span>
                    <span class="metric-value">{{ metricas.test_r2 }}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">MAE (Test)</span>
                    <span class="metric-value">{{ metricas.test_mae }}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">RMSE (Test)</span>
                    <span class="metric-value">{{ metricas.test_rmse }}</span>
                </div>
            </div>
        </div>
        
        <div class="main-content">
            <h2>🗺️ Mapa de Congestión Predicha</h2>
            <div id="map"></div>
            <div class="loading" id="loading">⏳ Generando predicciones...</div>
            <div class="results" id="results"></div>
        </div>
    </div>
    
    <script>
        let map = L.map('map').setView([-36.606, -72.103], 13);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);
        
        let markers = [];
        
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            
            loading.style.display = 'block';
            results.innerHTML = '';
            markers.forEach(m => map.removeLayer(m));
            markers = [];
            
            const data = {
                hora: parseInt(document.getElementById('hora').value),
                dia_semana: document.getElementById('dia_semana').value,
                temperatura: parseFloat(document.getElementById('temperatura').value),
                llueve: document.getElementById('llueve').checked,
                velocidad_promedio: parseFloat(document.getElementById('velocidad').value)
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const predicciones = await response.json();
                
                predicciones.forEach(pred => {
                    const colorMap = {
                        'green': '#4caf50',
                        'orange': '#ff9800',
                        'red': '#f44336'
                    };
                    
                    const marker = L.circleMarker([pred.latitud, pred.longitud], {
                        radius: 10,
                        fillColor: colorMap[pred.color],
                        color: '#fff',
                        weight: 2,
                        opacity: 1,
                        fillOpacity: 0.8
                    }).addTo(map);
                    
                    marker.bindPopup(`
                        <b>${pred.segmento_nombre}</b><br>
                        Índice: ${pred.prediccion}<br>
                        Estado: ${pred.categoria}
                    `);
                    
                    markers.push(marker);
                    
                    const card = document.createElement('div');
                    card.className = `result-card ${pred.categoria.toLowerCase()}`;
                    card.innerHTML = `
                        <h3>${pred.segmento_nombre}</h3>
                        <div class="indice">${pred.prediccion}</div>
                        <div class="categoria">${pred.categoria}</div>
                    `;
                    results.appendChild(card);
                });
                
            } catch (error) {
                alert('Error al realizar predicción: ' + error);
            } finally {
                loading.style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""


# ============================================================================
# RUTAS FLASK
# ============================================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, metricas=metricas)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    predicciones = predecir_congestion(
        hora=data['hora'],
        dia_semana=data['dia_semana'],
        temperatura=data['temperatura'],
        llueve=data['llueve'],
        velocidad_promedio=data['velocidad_promedio']
    )
    return jsonify(predicciones)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Servidor Flask iniciado")
    print("📍 Abre tu navegador en: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)