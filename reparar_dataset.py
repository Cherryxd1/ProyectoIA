
import pandas as pd
import numpy as np


#  COORDENADAS OFICIALES - PUNTOS  DE CHILLÁN


SEGMENTOS_CHILLAN = [
    {"id": "SEG001", "segmento_nombre": "Av. O'Higgins con Av. Collín", "latitud": -36.613056, "longitud": -72.111597, "longitud_m": 1200, "tipo_via": "arterial", "velocidad_maxima_kmh": 50},
    {"id": "SEG002", "segmento_nombre": "Av. O'Higgins con Av. Ecuador", "latitud": -36.597912, "longitud": -72.105803, "longitud_m": 1500, "tipo_via": "arterial", "velocidad_maxima_kmh": 50},
    {"id": "SEG003", "segmento_nombre": "Av. Collín con Av. Argentina", "latitud": -36.617130, "longitud": -72.097092, "longitud_m": 2000, "tipo_via": "arterial", "velocidad_maxima_kmh": 60},
    {"id": "SEG004", "segmento_nombre": "Av. Ecuador", "latitud": -36.599991, "longitud": -72.099999, "longitud_m": 1000, "tipo_via": "colectora", "velocidad_maxima_kmh": 40},
    {"id": "SEG005", "segmento_nombre": "Av. Alonso de Ercilla", "latitud": -36.625223, "longitud": -72.084603, "longitud_m": 1800, "tipo_via": "arterial", "velocidad_maxima_kmh": 50},
    {"id": "SEG006", "segmento_nombre": "Av. O'Higgins Chillán-Chillán Viejo", "latitud": -36.620378, "longitud": -72.121468, "longitud_m": 500, "tipo_via": "arterial", "velocidad_maxima_kmh": 40},
    {"id": "SEG007", "segmento_nombre": "Av. Collín (Clínica)", "latitud": -36.614837, "longitud": -72.106812, "longitud_m": 300, "tipo_via": "arterial", "velocidad_maxima_kmh": 30},
    {"id": "SEG008", "segmento_nombre": "5 de Abril", "latitud": -36.608176, "longitud": -72.101319, "longitud_m": 800, "tipo_via": "colectora", "velocidad_maxima_kmh": 30},
    {"id": "SEG009", "segmento_nombre": "Terminal de Buses María Teresa", "latitud": -36.587964, "longitud": -72.102649, "longitud_m": 600, "tipo_via": "colectora", "velocidad_maxima_kmh": 30},
    {"id": "SEG010", "segmento_nombre": "Av. Libertad Oriente", "latitud": -36.608632, "longitud": -72.090847, "longitud_m": 700, "tipo_via": "colectora", "velocidad_maxima_kmh": 40},
    {"id": "SEG011", "segmento_nombre": "Av. Andrés Bello", "latitud": -36.592662, "longitud": -72.070978, "longitud_m": 1200, "tipo_via": "arterial", "velocidad_maxima_kmh": 50},
]

def generar_dataset_historico():
    print("Iniciando generación de dataset con coordenadas reales y esquema completo...")
    np.random.seed(42)
    
    # Generar datos para aprox 10.000 registros (aprox 40 días * 24h * 11 segmentos)
    fechas = pd.date_range(start='2024-01-01', end='2024-02-12', freq='h')
    datos = []
    
    total_steps = len(fechas)
    print(f"Generando datos para {total_steps} hitos temporales por {len(SEGMENTOS_CHILLAN)} segmentos...")

    for i, fecha in enumerate(fechas):
        hora = fecha.hour
        dia_semana_num = fecha.dayofweek
        dias_str = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        dia_semana = dias_str[dia_semana_num]
        
        # Factor Clima (Aleatorio, más lluvia en invierno)
        mes = fecha.month
        prob_lluvia = 0.30 if mes in [5, 6, 7] else 0.05
        llueve_bool = np.random.random() < prob_lluvia
        
        # Temperatura y Lluvia mm
        temp_base = 15 + np.random.randn() * 5
        # Temperatura varía con la hora
        temperatura = temp_base + np.sin((hora - 6) * np.pi / 12) * 5 + np.random.randn()
        
        lluvia_mm = np.random.exponential(3) if llueve_bool and np.random.random() < 0.4 else 0.0
        
        for seg in SEGMENTOS_CHILLAN:
            # --- LÓGICA DE TRÁFICO ---
            congestion_base = 10 
            
            # 1. Factor Hora Punta
            es_punta_manana = (7 <= hora <= 9)
            es_punta_tarde = (17 <= hora <= 19)
            es_mediodia = (13 <= hora <= 14)
            es_noche = (23 <= hora) or (0 <= hora <= 5)
            
            if es_punta_manana: congestion_base += 65
            elif es_punta_tarde: congestion_base += 75
            elif es_mediodia: congestion_base += 45
            elif es_noche: congestion_base = 5
            else: congestion_base += 20
                
            # 2. Factor Día
            if dia_semana in ['Sábado', 'Domingo']:
                congestion_base *= 0.5
                if 11 <= hora <= 14: congestion_base += 25  # Fin de semana mediodía
            
            # 3. Factor Clima
            if llueve_bool: congestion_base += 15
                
            # 4. Variación Aleatoria
            variacion = np.random.normal(0, 6)
            indice = congestion_base + variacion
            indice = max(0, min(indice, 100))
            
            # Categoría (Schema: categoria_flujo)
            if indice < 30: cat = "Fluido"
            elif indice < 60: cat = "Moderado"
            else: cat = "Congestionado"
            
            # Velocidad Promedio (Derivada)
            reduccion = indice / 100
            velocidad = seg['velocidad_maxima_kmh'] * (1 - reduccion)
            
            datos.append({
                'fecha_hora': f"{fecha.date()} {hora:02d}:00:00",
                'dia_semana': dia_semana,
                'hora': hora,
                'segmento_id': seg['id'],
                'segmento_nombre': seg['segmento_nombre'],
                'latitud': seg['latitud'],
                'longitud': seg['longitud'],
                'longitud_m': seg['longitud_m'],
                'tipo_via': seg['tipo_via'],
                'velocidad_maxima_kmh': seg['velocidad_maxima_kmh'],
                'temperatura_c': round(temperatura, 1),
                'lluvia_mm': round(lluvia_mm, 1),
                'llueve': 1 if llueve_bool else 0,
                'velocidad_promedio_kmh': round(velocidad, 1),
                'indice_congestion': round(indice, 1),
                'categoria_flujo': cat
            })
            
    df = pd.DataFrame(datos)
    
    # Guardar CSV
    filename = 'dataset_congestion_vehicular_chillan.csv'
    df.to_csv(filename, index=False)
    print(f" Archivo '{filename}' generado exitosamente con {len(df)} registros. Esquema corregido.")

if __name__ == "__main__":
    generar_dataset_historico()
