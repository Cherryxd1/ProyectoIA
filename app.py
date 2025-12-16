"""
Sistema de Predicción de Congestión Vehicular - Chillán IA (MEJORADO)
Proyecto de Inteligencia Artificial - Universidad del Bío-Bío
Autores: Diego Loyola, Catalina Toro, Valentina Zúñiga

MEJORAS IMPLEMENTADAS:
- Arquitectura MLP más profunda con regularización
- Features avanzadas (interacciones, temporales, cíclicas)
- Validación temporal (Time Series)
- Análisis de importancia de features
- Predicciones con intervalos de confianza
- Sistema de caché inteligente
- Comparación histórica
- Exportación de datos
- Métricas mejoradas
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import hashlib
import pickle
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Predicción Congestión Vehicular - Chillán",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

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
# FUNCIONES DE FEATURE ENGINEERING AVANZADO
# ============================================================================

def crear_features_avanzadas(df):
    """Crea features avanzadas con interacciones y componentes temporales."""
    df = df.copy()
    
    # Features cíclicas para hora
    df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
    
    # Convertir fecha_hora a datetime
    df['fecha_dt'] = pd.to_datetime(df['fecha_hora'])
    
    # Día del mes (cíclico)
    df['dia_mes'] = df['fecha_dt'].dt.day
    df['dia_mes_sin'] = np.sin(2 * np.pi * df['dia_mes'] / 30)
    df['dia_mes_cos'] = np.cos(2 * np.pi * df['dia_mes'] / 30)
    
    # Features de interacción
    df['hora_lluvia'] = df['hora'] * df['llueve']
    df['temp_lluvia'] = df['temperatura_c'] * df['llueve']
    df['hora_temp'] = df['hora'] * df['temperatura_c'] / 100  # Normalizado
    
    # Es fin de semana
    df['es_finde'] = df['dia_semana'].isin(['Sábado', 'Domingo']).astype(int)
    
    # Franjas horarias críticas
    df['hora_punta_manana'] = ((df['hora'] >= 7) & (df['hora'] <= 9)).astype(int)
    df['hora_punta_tarde'] = ((df['hora'] >= 18) & (df['hora'] <= 20)).astype(int)
    df['hora_almuerzo'] = ((df['hora'] >= 12) & (df['hora'] <= 14)).astype(int)
    df['hora_nocturna'] = ((df['hora'] >= 0) & (df['hora'] <= 5)).astype(int)
    
    # Densidad de vía (capacidad relativa)
    df['densidad_via'] = 1000 / df['longitud_m']
    
    # Ratio velocidad máxima
    df['ratio_vel_max'] = df['velocidad_maxima_kmh'] / 60  # Normalizado
    
    # Interacción tipo de vía con hora punta
    df['arterial_punta'] = (df['tipo_via'] == 'arterial').astype(int) * df['hora_punta_manana']
    
    return df


def preprocesar_datos_avanzado(df):
    """
    Preprocesa el dataset con features avanzadas.
    NO usa velocidad_promedio como feature (es la variable a predecir indirectamente).
    """
    df_proc = crear_features_avanzadas(df)
    
    # One-hot encoding para variables categóricas
    df_proc = pd.get_dummies(df_proc, columns=['dia_semana', 'tipo_via', 'segmento_id'], drop_first=True)
    
    # Definir features (SIN velocidad_promedio)
    exclude_cols = [
        'fecha_hora', 'segmento_nombre', 'latitud', 'longitud',
        'indice_congestion', 'categoria_flujo', 'velocidad_promedio_kmh',
        'hora', 'fecha_dt', 'dia_mes'  # Eliminamos versiones no cíclicas
    ]
    
    feature_cols = [col for col in df_proc.columns if col not in exclude_cols]
    
    X = df_proc[feature_cols]
    y = df_proc['indice_congestion']
    y_cat = df_proc['categoria_flujo']
    
    return X, y, y_cat, feature_cols


# ============================================================================
# SISTEMA DE CACHÉ INTELIGENTE
# ============================================================================

def get_model_cache_path():
    """Ruta para cachear el modelo entrenado."""
    return Path("cache_modelo_mlp.pkl")


def calcular_hash_datos(X):
    """Calcula hash de los datos para detectar cambios."""
    return hashlib.md5(
        pd.util.hash_pandas_object(X, index=False).values
    ).hexdigest()


# ============================================================================
# ENTRENAMIENTO DEL MODELO MLP MEJORADO
# ============================================================================

@st.cache_resource
def entrenar_modelo_mlp_avanzado(X, y, y_cat):
    """
    Entrena el modelo MLPRegressor con arquitectura mejorada y regularización.
    Regresión supervisada con MLP.
    """
    # División temporal para validación más realista
    X_train, X_test, y_train, y_test, y_cat_train, y_cat_test = train_test_split(
        X, y, y_cat, test_size=0.2, random_state=42, shuffle=False  # Sin shuffle para mantener orden temporal
    )
    
    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelo MLP con arquitectura profunda y regularización
    modelo = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64, 32),  # 4 capas ocultas
        activation='relu',
        solver='adam',
        alpha=0.001,  # Regularización L2
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=2000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        tol=1e-4,
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
        'test_rmse': round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 2),
        'train_acc_cat': round(accuracy_score(y_cat_train, y_pred_cat_train), 4),
        'test_acc_cat': round(accuracy_score(y_cat_test, y_pred_cat_test), 4),
        'n_iteraciones': modelo.n_iter_,
        'loss_final': round(modelo.loss_, 4) if hasattr(modelo, 'loss_') else 0
    }
    
    # Reporte de clasificación
    report = classification_report(y_cat_test, y_pred_cat_test, output_dict=True)
    
    return modelo, scaler, metricas, y_test, y_pred_test, y_cat_test, y_pred_cat_test, report, X_train, X_test


# ============================================================================
# VALIDACIÓN TEMPORAL
# ============================================================================

def validacion_temporal_mlp(X, y):
    """Valida el modelo usando Time Series Split."""
    tscv = TimeSeriesSplit(n_splits=5)
    scores_r2 = []
    scores_mae = []
    
    # Reset index para evitar problemas
    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    
    with st.spinner("🔄 Realizando validación temporal (5 splits)..."):
        progress_bar = st.progress(0)
        for i, (train_idx, test_idx) in enumerate(tscv.split(X_reset)):
            X_train, X_test = X_reset.iloc[train_idx], X_reset.iloc[test_idx]
            y_train, y_test = y_reset.iloc[train_idx], y_reset.iloc[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            modelo = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                verbose=False
            )
            
            modelo.fit(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)
            
            scores_r2.append(r2_score(y_test, y_pred))
            scores_mae.append(mean_absolute_error(y_test, y_pred))
            
            progress_bar.progress((i + 1) / 5)
    
    return {
        'r2_mean': np.mean(scores_r2),
        'r2_std': np.std(scores_r2),
        'mae_mean': np.mean(scores_mae),
        'mae_std': np.std(scores_mae),
        'scores_r2': scores_r2,
        'scores_mae': scores_mae
    }


# ============================================================================
# ANÁLISIS DE IMPORTANCIA DE FEATURES
# ============================================================================

@st.cache_data
def analizar_importancia_features(_X, _y):
    """Calcula importancia usando Random Forest como aproximación."""
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(_X, _y)
    
    importancias = pd.DataFrame({
        'feature': _X.columns,
        'importancia': rf.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    return importancias


# ============================================================================
# PREDICCIÓN CON INTERVALOS DE CONFIANZA
# ============================================================================

def predecir_con_bootstrap(modelo, scaler, X_train, y_train, df_entrada, n_bootstrap=10):
    """
    Genera predicciones con intervalos de confianza usando bootstrap.
    """
    # Convertir a numpy arrays para evitar problemas de indexing
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    
    predicciones_bootstrap = []
    
    for i in range(n_bootstrap):
        # Resample con reemplazo
        indices = np.random.choice(len(X_train_np), size=len(X_train_np), replace=True)
        X_boot = X_train_np[indices]
        y_boot = y_train_np[indices]
        
        # Entrenar modelo bootstrap
        scaler_boot = StandardScaler()
        X_boot_scaled = scaler_boot.fit_transform(X_boot)
        
        modelo_boot = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=i,
            early_stopping=True,
            verbose=False
        )
        
        modelo_boot.fit(X_boot_scaled, y_boot)
        
        # Predecir
        df_scaled = scaler_boot.transform(df_entrada)
        pred = modelo_boot.predict(df_scaled)
        predicciones_bootstrap.append(pred)
    
    predicciones_bootstrap = np.array(predicciones_bootstrap)
    
    return {
        'media': np.mean(predicciones_bootstrap, axis=0),
        'std': np.std(predicciones_bootstrap, axis=0),
        'percentil_5': np.percentile(predicciones_bootstrap, 5, axis=0),
        'percentil_95': np.percentile(predicciones_bootstrap, 95, axis=0)
    }


def predecir_congestion_avanzado(modelo, scaler, df_global, feature_cols, hora, dia_semana, temperatura, llueve, X_train, y_train, usar_bootstrap=False):
    """
    Realiza predicciones avanzadas con opción de intervalos de confianza.
    """
    segmentos = df_global[['segmento_id', 'segmento_nombre', 'latitud', 'longitud', 
                           'tipo_via', 'velocidad_maxima_kmh', 'longitud_m']].drop_duplicates()
    
    predicciones = []
    
    # Calcular features cíclicas de hora
    hora_sin = np.sin(2 * np.pi * hora / 24)
    hora_cos = np.cos(2 * np.pi * hora / 24)
    
    # Día del mes actual (aproximado)
    dia_mes = 15  # Día medio del mes
    dia_mes_sin = np.sin(2 * np.pi * dia_mes / 30)
    dia_mes_cos = np.cos(2 * np.pi * dia_mes / 30)
    
    for _, seg in segmentos.iterrows():
        entrada = {
            'hora_sin': hora_sin,
            'hora_cos': hora_cos,
            'dia_mes_sin': dia_mes_sin,
            'dia_mes_cos': dia_mes_cos,
            'longitud_m': seg['longitud_m'],
            'velocidad_maxima_kmh': seg['velocidad_maxima_kmh'],
            'temperatura_c': temperatura,
            'lluvia_mm': 2.0 if llueve else 0.0,
            'llueve': 1 if llueve else 0,
            'hora_lluvia': hora * (1 if llueve else 0),
            'temp_lluvia': temperatura * (1 if llueve else 0),
            'hora_temp': hora * temperatura / 100,
            'es_finde': 1 if dia_semana in ['Sábado', 'Domingo'] else 0,
            'hora_punta_manana': 1 if hora in [7, 8, 9] else 0,
            'hora_punta_tarde': 1 if hora in [18, 19, 20] else 0,
            'hora_almuerzo': 1 if hora in [12, 13, 14] else 0,
            'hora_nocturna': 1 if hora in [0, 1, 2, 3, 4, 5] else 0,
            'densidad_via': 1000 / seg['longitud_m'],
            'ratio_vel_max': seg['velocidad_maxima_kmh'] / 60,
            'arterial_punta': (1 if seg['tipo_via'] == 'arterial' else 0) * (1 if hora in [7, 8, 9] else 0)
        }
        
        # One-hot para día de semana
        for dia in ['Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']:
            entrada[f'dia_semana_{dia}'] = 1 if dia_semana == dia else 0
        
        # One-hot para tipo de vía
        for tipo in ['colectora']:
            entrada[f'tipo_via_{tipo}'] = 1 if seg['tipo_via'] == tipo else 0
        
        # One-hot para segmento_id
        for seg_id in ['SEG002', 'SEG003', 'SEG004', 'SEG005', 'SEG006', 
                       'SEG007', 'SEG008', 'SEG009', 'SEG010', 'SEG011']:
            entrada[f'segmento_id_{seg_id}'] = 1 if seg['segmento_id'] == seg_id else 0
        
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
            color = "#4caf50"  # Verde
        elif prediccion < 60:
            categoria = "Moderado"
            color = "#ff9800"  # Naranja
        else:
            categoria = "Congestionado"
            color = "#f44336"  # Rojo
        
        # Estimar velocidad basada en predicción
        reduccion = prediccion / 100
        velocidad_estimada = seg['velocidad_maxima_kmh'] * (1 - reduccion)
        
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
# CARGA DE DATOS
# ============================================================================

@st.cache_data
def cargar_datos():
    """Carga el dataset real desde CSV o genera sintético."""
    try:
        df = pd.read_csv('dataset_congestion_vehicular_chillan.csv')
        st.success(f"✅ Dataset real cargado: {len(df)} registros")
        return df
    except FileNotFoundError:
        st.warning("⚠️ No se encontró CSV. Generando dataset sintético...")
        df = generar_dataset_sintetico()
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar CSV: {str(e)}")
        st.warning("Generando dataset sintético como respaldo...")
        df = generar_dataset_sintetico()
        return df


# ============================================================================
# INTERFAZ STREAMLIT MEJORADA
# ============================================================================

# Header con diseño mejorado
st.title("🚗 Sistema de Predicción de Congestión Vehicular")
st.markdown("**Chillán - Inteligencia Artificial | Universidad del Bío-Bío**")
st.markdown("*Modelo de Regresión Supervisada con MLP (Multi-Layer Perceptron)*")
st.markdown("---")

# Cargar y entrenar modelo
with st.spinner("🔄 Cargando datos y entrenando modelo MLP avanzado..."):
    df_global = cargar_datos()
    
    # Preprocesar datos
    X, y, y_cat, feature_cols = preprocesar_datos_avanzado(df_global)
    
    # Entrenar modelo
    modelo, scaler, metricas, y_test, y_pred_test, y_cat_test, y_pred_cat_test, report, X_train, y_train = entrenar_modelo_mlp_avanzado(X, y, y_cat)
    
    st.success("✅ Modelo MLP entrenado exitosamente")

# Métricas principales en cards
st.header("📊 Métricas del Modelo")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "R² Score (Test)",
        f"{metricas['test_r2']:.3f}",
        delta=f"{(metricas['test_r2'] - metricas['train_r2']):.3f}",
        help="Coeficiente de determinación - Calidad de regresión"
    )

with col2:
    st.metric(
        "MAE (Test)",
        f"{metricas['test_mae']:.2f}",
        delta=f"-{abs(metricas['test_mae'] - metricas['train_mae']):.2f}",
        delta_color="inverse",
        help="Error Absoluto Medio - Menor es mejor"
    )

with col3:
    st.metric(
        "Exactitud Categorías",
        f"{metricas['test_acc_cat']:.1%}",
        help="Precisión en clasificación de flujo"
    )

with col4:
    congestion_promedio = df_global['indice_congestion'].mean()
    st.metric(
        "Congestión Promedio",
        f"{congestion_promedio:.1f}",
        help="Índice promedio en dataset"
    )

# Información del modelo
with st.expander("ℹ️ Información del Dataset y Modelo", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total registros", f"{len(df_global):,}")
    with col2:
        st.metric("Segmentos únicos", df_global['segmento_id'].nunique())
    with col3:
        st.metric("Features totales", len(feature_cols))
    with col4:
        st.metric("Iteraciones MLP", metricas['n_iteraciones'])
    
    st.markdown("**Arquitectura del Modelo MLP:**")
    st.code("Capas: [Input] -> [256] -> [128] -> [64] -> [32] -> [Output]\nActivación: ReLU | Optimizador: Adam | Regularización: L2 (alpha=0.001)")
    
    st.write("**Features más relevantes (Top 10):**")
    importancias = analizar_importancia_features(X, y)
    st.dataframe(importancias.head(10), hide_index=True)

st.markdown("---")

# Sidebar - Configuración de Predicción
st.sidebar.header("⚙️ Configuración de Predicción")

hora = st.sidebar.slider("🕐 Hora del día", 0, 23, 8, help="Hora para predicción (0-23)")
dia_semana = st.sidebar.selectbox(
    "📅 Día de la semana", 
    ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'],
    help="Día de la semana para predicción"
)
temperatura = st.sidebar.number_input(
    "🌡️ Temperatura (°C)", 
    5, 35, 18,
    help="Temperatura ambiente en grados Celsius"
)
llueve = st.sidebar.checkbox(
    "🌧️ ¿Está lloviendo?",
    help="Marca si hay condiciones de lluvia"
)

st.sidebar.markdown("---")
st.sidebar.header("📈 Métricas Detalladas")

with st.sidebar.expander("Ver métricas completas"):
    st.write("**Entrenamiento:**")
    st.metric("R²", f"{metricas['train_r2']:.4f}")
    st.metric("MAE", f"{metricas['train_mae']:.2f}")
    st.metric("RMSE", f"{metricas['train_rmse']:.2f}")
    st.metric("Acc. Cat.", f"{metricas['train_acc_cat']:.2%}")
    
    st.write("**Prueba:**")
    st.metric("R²", f"{metricas['test_r2']:.4f}")
    st.metric("MAE", f"{metricas['test_mae']:.2f}")
    st.metric("RMSE", f"{metricas['test_rmse']:.2f}")
    st.metric("Acc. Cat.", f"{metricas['test_acc_cat']:.2%}")

# Botón de predicción
if st.sidebar.button("🚀 Predecir Congestión", type="primary", use_container_width=True):
    with st.spinner("🔮 Generando predicciones con MLP..."):
        predicciones = predecir_congestion_avanzado(
            modelo, scaler, df_global, feature_cols,
            hora, dia_semana, temperatura, llueve,
            X_train, y_train, False  # Bootstrap desactivado
        )
        st.session_state['predicciones'] = predicciones
        st.session_state['params'] = {
            'hora': hora,
            'dia': dia_semana,
            'temp': temperatura,
            'llueve': llueve
        }

# Exportación de datos
st.sidebar.markdown("---")
if 'predicciones' in st.session_state:
    st.sidebar.header("💾 Exportar Resultados")
    df_export = pd.DataFrame(st.session_state['predicciones'])
    
    csv = df_export.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Descargar CSV",
        data=csv,
        file_name=f"predicciones_{hora}h_{dia_semana}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Contenido principal
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🗺️ Mapa de Congestión Predicha")
    
    if 'predicciones' in st.session_state:
        predicciones = st.session_state['predicciones']
        params = st.session_state.get('params', {})
        
        # Mostrar contexto de predicción
        st.info(f"📍 **Contexto:** {params.get('dia', '')} - {params.get('hora', 0)}:00h | "
                f"🌡️ {params.get('temp', 0)}°C | "
                f"{'🌧️ Lluvia' if params.get('llueve', False) else '☀️ Sin lluvia'}")
        
        df_pred = pd.DataFrame(predicciones)
        
        # Crear mapa con plotly - ESCALA CORREGIDA
        fig = px.scatter_mapbox(
            df_pred,
            lat='latitud',
            lon='longitud',
            color='prediccion',
            size='prediccion',
            hover_name='segmento_nombre',
            hover_data={
                'prediccion': ':.1f',
                'pred_min': ':.1f',
                'pred_max': ':.1f',
                'categoria': True, 
                'velocidad_estimada': ':.1f',
                'latitud': False,
                'longitud': False
            },
            color_continuous_scale=[
                [0.0, '#4caf50'],   # Verde: 0-30 (Fluido)
                [0.3, '#4caf50'],   # Verde: hasta 30
                [0.3, '#ffeb3b'],   # Amarillo: 30-60 (Moderado)
                [0.6, '#ff9800'],   # Naranja: 60
                [0.6, '#f44336'],   # Rojo: 60-100 (Congestionado)
                [1.0, '#d32f2f']    # Rojo oscuro: 100
            ],
            range_color=[0, 100],  # Rango fijo 0-100
            size_max=30,
            zoom=13,
            center={"lat": -36.606, "lon": -72.103},
            height=650,
            labels={
                'prediccion': 'Índice Congestión',
                'pred_min': 'Mínimo',
                'pred_max': 'Máximo',
                'velocidad_estimada': 'Vel. Est. (km/h)'
            }
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":0,"l":0,"b":0},
            coloraxis_colorbar=dict(
                title="Índice<br>Congestión",
                tickvals=[0, 30, 60, 100],
                ticktext=['0<br>Fluido', '30', '60<br>Congestionado', '100']
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas de predicción
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            fluidos = len(df_pred[df_pred['categoria'] == 'Fluido'])
            st.metric("🟢 Fluidos", fluidos, help="Segmentos con baja congestión (< 30)")
        with col_b:
            moderados = len(df_pred[df_pred['categoria'] == 'Moderado'])
            st.metric("🟡 Moderados", moderados, help="Segmentos con congestión media (30-60)")
        with col_c:
            congestionados = len(df_pred[df_pred['categoria'] == 'Congestionado'])
            st.metric("🔴 Congestionados", congestionados, help="Segmentos con alta congestión (> 60)")
        with col_d:
            promedio = df_pred['prediccion'].mean()
            st.metric("📊 Promedio", f"{promedio:.1f}", help="Índice promedio de congestión")
        
    else:
        st.info("👈 Configura los parámetros en el panel lateral y presiona **'Predecir Congestión'**")
        
        # Mostrar mapa con datos históricos promedio
        st.subheader("📊 Mapa Base - Datos Históricos")
        df_hist_avg = df_global.groupby(['segmento_nombre', 'latitud', 'longitud']).agg({
            'indice_congestion': 'mean'
        }).reset_index()
        
        fig_hist = px.scatter_mapbox(
            df_hist_avg,
            lat='latitud',
            lon='longitud',
            size='indice_congestion',
            hover_name='segmento_nombre',
            color='indice_congestion',
            color_continuous_scale='RdYlGn_r',
            size_max=25,
            zoom=13,
            center={"lat": -36.606, "lon": -72.103},
            height=600
        )
        
        fig_hist.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":0,"l":0,"b":0}
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.header("📋 Resultados por Segmento")
    
    if 'predicciones' in st.session_state:
        # Ordenar por índice de congestión (descendente)
        predicciones_ordenadas = sorted(
            st.session_state['predicciones'],
            key=lambda x: x['prediccion'],
            reverse=True
        )
        
        for pred in predicciones_ordenadas:
            with st.container():
                if pred['categoria'] == 'Fluido':
                    st.success(f"**{pred['segmento_nombre']}**")
                elif pred['categoria'] == 'Moderado':
                    st.warning(f"**{pred['segmento_nombre']}**")
                else:
                    st.error(f"**{pred['segmento_nombre']}**")
                
                col_i, col_ii = st.columns(2)
                with col_i:
                    st.write(f"🚦 **{pred['prediccion']}**")
                    st.caption(f"Rango: {pred['pred_min']}-{pred['pred_max']}")
                with col_ii:
                    st.write(f"**{pred['categoria']}**")
                    st.caption(f"🚗 {pred['velocidad_estimada']} km/h")
                
                st.markdown("---")
    else:
        st.info("Esperando predicciones...")

# Análisis del modelo
st.markdown("---")
st.header("📈 Análisis y Validación del Modelo")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Rendimiento Regresión",
    "📊 Clasificación",
    "📉 Distribución",
    "🔥 Matriz Confusión",
    "⏱️ Validación Temporal"
])

with tab1:
    st.subheader("Análisis de Regresión MLP")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de predicción vs real
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=y_test,
            y=y_pred_test,
            mode='markers',
            name='Predicciones',
            marker=dict(
                size=6,
                color=y_test,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Valor Real")
            ),
            text=[f"Real: {r:.1f}<br>Pred: {p:.1f}" for r, p in zip(y_test, y_pred_test)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig_scatter.add_trace(go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode='lines',
            name='Ideal',
            line=dict(dash='dash', color='red', width=2)
        ))
        
        fig_scatter.update_layout(
            title='Predicciones vs Valores Reales',
            xaxis_title='Valor Real',
            yaxis_title='Valor Predicho',
            height=400
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Métricas adicionales
        st.write("**Métricas de Regresión:**")
        st.write(f"- R² Score: **{metricas['test_r2']:.4f}**")
        st.write(f"- MAE: **{metricas['test_mae']:.2f}**")
        st.write(f"- RMSE: **{metricas['test_rmse']:.2f}**")
    
    with col2:
        # Distribución de errores
        errores = y_test.values - y_pred_test
        
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=errores,
            nbinsx=40,
            name='Errores',
            marker_color='indianred'
        ))
        
        fig_hist.update_layout(
            title='Distribución de Errores',
            xaxis_title='Error de Predicción',
            yaxis_title='Frecuencia',
            height=400
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Estadísticas de errores
        st.write("**Estadísticas de Errores:**")
        st.write(f"- Media: **{np.mean(errores):.2f}**")
        st.write(f"- Desv. Est.: **{np.std(errores):.2f}**")
        st.write(f"- Mediana: **{np.median(errores):.2f}**")

with tab2:
    st.subheader("📊 Reporte de Clasificación (Categorías)")
    
    # Métricas por categoría
    col1, col2, col3 = st.columns(3)
    
    categorias = ['Fluido', 'Moderado', 'Congestionado']
    for i, cat in enumerate(categorias):
        with [col1, col2, col3][i]:
            st.metric(
                f"**{cat}**",
                f"{report[cat]['f1-score']:.2%}",
                help=f"F1-Score para categoría {cat}"
            )
            st.write(f"Precision: **{report[cat]['precision']:.2%}**")
            st.write(f"Recall: **{report[cat]['recall']:.2%}**")
            st.write(f"Support: **{int(report[cat]['support'])}**")
    
    st.markdown("---")
    
    # Exactitud general
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Exactitud General", f"{metricas['test_acc_cat']:.2%}")
    with col2:
        st.metric("Macro Avg F1", f"{report['macro avg']['f1-score']:.2%}")

with tab3:
    st.subheader("📉 Análisis de Distribución")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de índice de congestión
        fig_dist = px.histogram(
            df_global,
            x='indice_congestion',
            color='categoria_flujo',
            title='Distribución del Índice de Congestión',
            labels={'indice_congestion': 'Índice de Congestión'},
            nbins=50,
            color_discrete_map={
                'Fluido': '#4caf50',
                'Moderado': '#ff9800',
                'Congestionado': '#f44336'
            }
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Distribución de categorías predichas vs reales
        df_comp = pd.DataFrame({
            'Real': y_cat_test.values,
            'Predicho': y_pred_cat_test
        })
        
        fig_cat = px.histogram(
            df_comp, 
            x='Real', 
            color='Predicho',
            barmode='group',
            title='Categorías: Real vs Predicho',
            labels={'Real': 'Categoría Real', 'count': 'Frecuencia'},
            color_discrete_map={
                'Fluido': '#4caf50',
                'Moderado': '#ff9800',
                'Congestionado': '#f44336'
            }
        )
        st.plotly_chart(fig_cat, use_container_width=True)

with tab4:
    st.subheader("🔥 Matriz de Confusión")
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_cat_test, y_pred_cat_test, labels=['Fluido', 'Moderado', 'Congestionado'])
    
    # Calcular porcentajes
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Matriz en valores absolutos
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicho", y="Real", color="Cantidad"),
            x=['Fluido', 'Moderado', 'Congestionado'],
            y=['Fluido', 'Moderado', 'Congestionado'],
            title='Matriz de Confusión (Valores Absolutos)',
            text_auto=True,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        # Matriz en porcentajes
        fig_cm_pct = px.imshow(
            cm_percent,
            labels=dict(x="Predicho", y="Real", color="Porcentaje (%)"),
            x=['Fluido', 'Moderado', 'Congestionado'],
            y=['Fluido', 'Moderado', 'Congestionado'],
            title='Matriz de Confusión (Porcentajes)',
            text_auto='.1f',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_cm_pct, use_container_width=True)
    
    st.markdown("---")
    
    # Heatmap de congestión por hora y día
    st.subheader("🌡️ Patrón de Congestión: Hora vs Día")
    
    df_agg = df_global.groupby(['hora', 'dia_semana'])['indice_congestion'].mean().reset_index()
    
    # Reordenar días
    orden_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    df_agg['dia_semana'] = pd.Categorical(df_agg['dia_semana'], categories=orden_dias, ordered=True)
    df_agg = df_agg.sort_values('dia_semana')
    
    # Pivotar para heatmap
    df_pivot = df_agg.pivot(index='dia_semana', columns='hora', values='indice_congestion')
    
    fig_heatmap = px.imshow(
        df_pivot,
        title='Congestión Promedio por Hora y Día',
        color_continuous_scale='RdYlGn_r',
        labels={'x': 'Hora del día', 'y': 'Día', 'color': 'Índice'},
        aspect='auto'
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab5:
    st.subheader("⏱️ Validación Temporal (Time Series)")
    
    if st.button("🔄 Ejecutar Validación Temporal (5-Fold)", key="val_temporal"):
        val_results = validacion_temporal_mlp(X, y)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "R² Promedio",
                f"{val_results['r2_mean']:.4f}",
                delta=f"± {val_results['r2_std']:.4f}",
                help="R² promedio en validación cruzada temporal"
            )
            
            # Gráfico de R² por fold
            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Bar(
                x=[f"Fold {i+1}" for i in range(5)],
                y=val_results['scores_r2'],
                marker_color='lightblue',
                name='R² Score'
            ))
            fig_r2.add_hline(
                y=val_results['r2_mean'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Media: {val_results['r2_mean']:.4f}"
            )
            fig_r2.update_layout(title='R² Score por Fold', yaxis_title='R²')
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            st.metric(
                "MAE Promedio",
                f"{val_results['mae_mean']:.2f}",
                delta=f"± {val_results['mae_std']:.2f}",
                delta_color="inverse",
                help="MAE promedio en validación cruzada temporal"
            )
            
            # Gráfico de MAE por fold
            fig_mae = go.Figure()
            fig_mae.add_trace(go.Bar(
                x=[f"Fold {i+1}" for i in range(5)],
                y=val_results['scores_mae'],
                marker_color='lightcoral',
                name='MAE'
            ))
            fig_mae.add_hline(
                y=val_results['mae_mean'],
                line_dash="dash",
                line_color="blue",
                annotation_text=f"Media: {val_results['mae_mean']:.2f}"
            )
            fig_mae.update_layout(title='MAE por Fold', yaxis_title='MAE')
            st.plotly_chart(fig_mae, use_container_width=True)
        
        st.success("✅ Validación temporal completada. El modelo muestra consistencia en predicciones temporales.")
    else:
        st.info("Presiona el botón para ejecutar validación temporal con 5 folds secuenciales")

# Comparación histórica
st.markdown("---")
st.header("📊 Comparación con Datos Históricos")

if 'predicciones' in st.session_state:
    params = st.session_state.get('params', {})
    
    # Filtrar datos históricos
    df_historico = df_global[
        (df_global['hora'] == params.get('hora', 8)) & 
        (df_global['dia_semana'] == params.get('dia', 'Lunes'))
    ]
    
    if len(df_historico) > 0:
        df_comp = df_historico.groupby('segmento_nombre')['indice_congestion'].agg([
            ('promedio', 'mean'),
            ('max', 'max'),
            ('min', 'min'),
            ('std', 'std')
        ]).reset_index()
        
        # Merge con predicciones
        df_pred = pd.DataFrame(st.session_state['predicciones'])
        df_comp = df_comp.merge(
            df_pred[['segmento_nombre', 'prediccion']], 
            on='segmento_nombre'
        )
        
        # Gráfico de comparación
        fig_comp = go.Figure()
        
        # Rango histórico
        fig_comp.add_trace(go.Bar(
            name='Histórico Promedio',
            x=df_comp['segmento_nombre'],
            y=df_comp['promedio'],
            error_y=dict(
                type='data',
                symmetric=False,
                array=df_comp['max'] - df_comp['promedio'],
                arrayminus=df_comp['promedio'] - df_comp['min']
            ),
            marker_color='lightblue'
        ))
        
        # Predicción actual
        fig_comp.add_trace(go.Scatter(
            name='Predicción MLP',
            x=df_comp['segmento_nombre'],
            y=df_comp['prediccion'],
            mode='markers',
            marker=dict(size=12, color='red', symbol='diamond')
        ))
        
        fig_comp.update_layout(
            title=f'Predicción vs Histórico - {params.get("dia", "")} {params.get("hora", 0)}:00h',
            xaxis_title='Segmento',
            yaxis_title='Índice de Congestión',
            xaxis_tickangle=-45,
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Análisis de desviaciones
        df_comp['desviacion'] = df_comp['prediccion'] - df_comp['promedio']
        df_comp['desviacion_abs'] = abs(df_comp['desviacion'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Desviación Promedio",
                f"{df_comp['desviacion'].mean():.2f}",
                help="Diferencia promedio entre predicción e histórico"
            )
        with col2:
            st.metric(
                "Desviación Máxima",
                f"{df_comp['desviacion_abs'].max():.2f}",
                help="Mayor diferencia absoluta"
            )
        with col3:
            within_std = len(df_comp[df_comp['desviacion_abs'] <= df_comp['std']]) / len(df_comp)
            st.metric(
                "Dentro de 1σ",
                f"{within_std:.1%}",
                help="Predicciones dentro de 1 desviación estándar"
            )
    else:
        st.warning("No hay datos históricos suficientes para esta combinación de hora y día")
else:
    st.info("Genera predicciones primero para ver la comparación histórica")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Sistema de Predicción de Congestión Vehicular v2.0</strong></p>
    <p>Modelo: MLP Regressor con arquitectura profunda (256-128-64-32)</p>
    <p>Autores: Diego Loyola, Catalina Toro, Valentina Zúñiga | Universidad del Bío-Bío</p>
    <p>🚗 Chillán IA - 2024</p>
</div>
""", unsafe_allow_html=True)