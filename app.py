"""
Sistema de Predicción de Congestión Vehicular - Chillán IA
Proyecto de Inteligencia Artificial - Universidad del Bío-Bío
Autores: Diego Loyola, Catalina Toro, Valentina Zúñiga


"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Predicción Congestión Vehicular - Chillán",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# GENERACIÓN DE DATASET SINTÉTICO
# ============================================================================

@st.cache_data
def generar_dataset_sintetico():
    """Genera un dataset sintético realista basado en patrones de tráfico de Chillán."""
    np.random.seed(42)
    
    segmentos = [
        {"id": "SEG001", "nombre": "Av. O'Higgins (Ruta 5 - L. Arellano)", "lat": -36.606, "lon": -72.103, "long_m": 1200, "tipo": "arterial", "vel_max": 50},
        {"id": "SEG002", "nombre": "Av. O'Higgins (L. Arellano - P. Piedra)", "lat": -36.606, "lon": -72.095, "long_m": 1500, "tipo": "arterial", "vel_max": 50},
        {"id": "SEG003", "nombre": "Av. Collín (Sector Norte)", "lat": -36.595, "lon": -72.105, "long_m": 2000, "tipo": "arterial", "vel_max": 60},
        {"id": "SEG004", "nombre": "Av. Ecuador (Centro)", "lat": -36.610, "lon": -72.100, "long_m": 1000, "tipo": "colectora", "vel_max": 40},
        {"id": "SEG005", "nombre": "Av. Alonso de Ercilla", "lat": -36.615, "lon": -72.098, "long_m": 1800, "tipo": "arterial", "vel_max": 50},
        {"id": "SEG006", "nombre": "Puente Chillán-Chillán Viejo", "lat": -36.620, "lon": -72.110, "long_m": 500, "tipo": "arterial", "vel_max": 40},
        {"id": "SEG007", "nombre": "Rotonda Collín", "lat": -36.593, "lon": -72.108, "long_m": 300, "tipo": "arterial", "vel_max": 30},
        {"id": "SEG008", "nombre": "Centro Cívico", "lat": -36.606, "lon": -72.104, "long_m": 800, "tipo": "colectora", "vel_max": 30},
        {"id": "SEG009", "nombre": "Terminal de Buses", "lat": -36.608, "lon": -72.106, "long_m": 600, "tipo": "colectora", "vel_max": 30},
        {"id": "SEG010", "nombre": "Hospital Herminda Martín", "lat": -36.614, "lon": -72.107, "long_m": 700, "tipo": "colectora", "vel_max": 40},
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
# FUNCIONES DE FEATURE ENGINEERING MEJORADO
# ============================================================================

def crear_features_ciclicas(df):
    """Convierte hora en features cíclicas (sin y cos)."""
    df = df.copy()
    # Hora cíclica: 0-23 horas
    df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
    return df


def preprocesar_datos_mejorado(df):
    """
    Preprocesa el dataset con mejoras de Feature Engineering:
    1. Eliminamos velocidad_promedio como feature (es resultado de congestión)
    2. Agrega features cíclicas para hora
    3. Incluye segmento_id como contexto
    4. One-hot encoding mejorado
    """
    df_proc = df.copy()
    
    # 1. Crear features cíclicas para hora
    df_proc = crear_features_ciclicas(df_proc)
    
    # 2. One-hot encoding para variables categóricas
    df_proc = pd.get_dummies(df_proc, columns=['dia_semana', 'tipo_via', 'segmento_id'], drop_first=True)
    
    # 3. Definir features (SIN velocidad_promedio)
    exclude_cols = [
        'fecha_hora', 'segmento_nombre', 'latitud', 'longitud',
        'indice_congestion', 'categoria_flujo', 'velocidad_promedio_kmh',
        'hora'  # Eliminamos hora lineal, usamos sin/cos
    ]
    
    feature_cols = [col for col in df_proc.columns if col not in exclude_cols]
    
    X = df_proc[feature_cols]
    y = df_proc['indice_congestion']
    
    # También guardamos categorías para evaluación
    y_cat = df_proc['categoria_flujo']
    
    return X, y, y_cat, feature_cols


# ============================================================================
# FUNCIONES DE MODELO y PREDICCIÓN 
# ============================================================================

@st.cache_data
def cargar_datos():
    """Carga el dataset real desde CSV."""
    try:
        df = pd.read_csv('dataset_congestion_vehicular_chillan.csv')
        st.success(f"Dataset real cargado: {len(df)} registros")
        return df
    except FileNotFoundError:
        st.error("Error: No se encontró el archivo 'dataset_congestion_vehicular_chillan.csv'")
        st.warning("Generando dataset sintético como respaldo...")
        df = generar_dataset_sintetico()
        return df
    except Exception as e:
        st.error(f"Error al cargar CSV: {str(e)}")
        st.warning("Generando dataset sintético como respaldo...")
        df = generar_dataset_sintetico()
        return df


@st.cache_resource
def entrenar_modelo_mejorado(X, y, y_cat):
    """
    Entrena el modelo MLPRegressor con arquitectura mejorada.
    También evalúa precisión de clasificación.
    """
    X_train, X_test, y_train, y_test, y_cat_train, y_cat_test = train_test_split(
        X, y, y_cat, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelo con arquitectura más profunda
    modelo = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),  # 3 capas ocultas
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        verbose=False
    )
    
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
        'test_acc_cat': round(accuracy_score(y_cat_test, y_pred_cat_test), 4)
    }
    
    # Reporte de clasificación
    report = classification_report(y_cat_test, y_pred_cat_test, output_dict=True)
    
    return modelo, scaler, metricas, y_test, y_pred_test, y_cat_test, y_pred_cat_test, report


def predecir_congestion_mejorado(modelo, scaler, df_global, feature_cols, hora, dia_semana, temperatura, llueve):
    """
    Realiza predicciones SIN usar velocidad_promedio como input.
    Predice basándose solo en contexto temporal, climático y del segmento.
    """
    segmentos = df_global[['segmento_id', 'segmento_nombre', 'latitud', 'longitud', 
                           'tipo_via', 'velocidad_maxima_kmh', 'longitud_m']].drop_duplicates()
    
    predicciones = []
    
    # Calcular features cíclicas de hora
    hora_sin = np.sin(2 * np.pi * hora / 24)
    hora_cos = np.cos(2 * np.pi * hora / 24)
    
    for _, seg in segmentos.iterrows():
        entrada = {
            'hora_sin': hora_sin,
            'hora_cos': hora_cos,
            'longitud_m': seg['longitud_m'],
            'velocidad_maxima_kmh': seg['velocidad_maxima_kmh'],
            'temperatura_c': temperatura,
            'lluvia_mm': 2.0 if llueve else 0.0,
            'llueve': 1 if llueve else 0
            # NO incluimos velocidad_promedio
        }
        
        # One-hot para día de semana
        for dia in ['Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']:
            entrada[f'dia_semana_{dia}'] = 1 if dia_semana == dia else 0
        
        # One-hot para tipo de vía
        for tipo in ['colectora', 'local']:
            entrada[f'tipo_via_{tipo}'] = 1 if seg['tipo_via'] == tipo else 0
        
        # One-hot para segmento_id
        for seg_id in ['SEG002', 'SEG003', 'SEG004', 'SEG005', 'SEG006', 
                       'SEG007', 'SEG008', 'SEG009', 'SEG010']:
            entrada[f'segmento_id_{seg_id}'] = 1 if seg['segmento_id'] == seg_id else 0
        
        df_entrada = pd.DataFrame([entrada])
        
        # Asegurar que tenga todas las columnas
        for col in feature_cols:
            if col not in df_entrada.columns:
                df_entrada[col] = 0
        
        df_entrada = df_entrada[feature_cols]
        entrada_scaled = scaler.transform(df_entrada)
        prediccion = modelo.predict(entrada_scaled)[0]
        prediccion = max(0, min(prediccion, 100))
        
        if prediccion < 30:
            categoria = "Fluido"
            color = "#4caf50"
        elif prediccion < 60:
            categoria = "Moderado"
            color = "#ff9800"
        else:
            categoria = "Congestionado"
            color = "#f44336"
        
        # Estimar velocidad basada en predicción
        reduccion = prediccion / 100
        velocidad_estimada = seg['velocidad_maxima_kmh'] * (1 - reduccion)
        
        predicciones.append({
            'segmento_id': seg['segmento_id'],
            'segmento_nombre': seg['segmento_nombre'],
            'latitud': seg['latitud'],
            'longitud': seg['longitud'],
            'prediccion': round(prediccion, 1),
            'categoria': categoria,
            'color': color,
            'velocidad_estimada': round(velocidad_estimada, 1)
        })
    
    return predicciones


# ============================================================================
# INTERFAZ STREAMLIT
# ============================================================================

# Header
st.title("Sistema de Predicción de Congestión Vehicular ")
st.markdown("**Chillán - Inteligencia Artificial | Universidad del Bío-Bío**")
st.markdown("---")

# Cargar y entrenar modelo
with st.spinner("Cargando datos y entrenando modelo MLP mejorado..."):
    df_global = cargar_datos()
    
    # Mostrar información del dataset
    with st.expander("Ver información del dataset cargado", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total registros", f"{len(df_global):,}")
        with col2:
            st.metric("Segmentos únicos", df_global['segmento_id'].nunique())
        with col3:
            st.metric("Días de datos", df_global['fecha_hora'].nunique() // 24)
        
        st.write("**Primeras filas del dataset:**")
        st.dataframe(df_global.head(10))
        
        st.write("**Columnas disponibles:**")
        st.write(list(df_global.columns))
    
    X, y, y_cat, feature_cols = preprocesar_datos_mejorado(df_global)
    modelo, scaler, metricas, y_test, y_pred_test, y_cat_test, y_pred_cat_test, report = entrenar_modelo_mejorado(X, y, y_cat)

# Sidebar - Configuración
st.sidebar.header("Configuración de Predicción")


hora = st.sidebar.slider("🕐 Hora del día", 0, 23, 8)
dia_semana = st.sidebar.selectbox("📅 Día de la semana", 
    ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'])
temperatura = st.sidebar.number_input("🌡️ Temperatura (°C)", 5, 35, 18)
llueve = st.sidebar.checkbox("🌧️ ¿Está lloviendo?")

st.sidebar.markdown("---")
st.sidebar.header("📊 Métricas del Modelo MLP")

col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("R² Train", f"{metricas['train_r2']:.4f}")
    st.metric("MAE Train", f"{metricas['train_mae']:.2f}")
    st.metric("Acc Cat Train", f"{metricas['train_acc_cat']:.2%}")
with col2:
    st.metric("R² Test", f"{metricas['test_r2']:.4f}")
    st.metric("MAE Test", f"{metricas['test_mae']:.2f}")
    st.metric("Acc Cat Test", f"{metricas['test_acc_cat']:.2%}")



# Botón de predicción
if st.sidebar.button("Predecir Congestión Vehicular", type="primary"):
    with st.spinner("Generando predicciones..."):
        predicciones = predecir_congestion_mejorado(
            modelo, scaler, df_global, feature_cols,
            hora, dia_semana, temperatura, llueve
        )
        st.session_state['predicciones'] = predicciones

# Contenido principal
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🗺️ Mapa de Congestión Predicha")
    
    if 'predicciones' in st.session_state:
        predicciones = st.session_state['predicciones']
        df_pred = pd.DataFrame(predicciones)
        
        # Crear mapa con plotly
        fig = px.scatter_mapbox(
            df_pred,
            lat='latitud',
            lon='longitud',
            color='prediccion',
            size='prediccion',
            hover_name='segmento_nombre',
            hover_data={
                'prediccion': ':.1f',              # Con formato
                'categoria': True, 
                'velocidad_estimada': ':.1f',      # Con formato
                'latitud': ':.4f',                 # Mostrar pero con formato
                'longitud': ':.4f'                 # Mostrar pero con formato
            },
            color_continuous_scale=['green', 'yellow', 'red'],
            size_max=25,                           # Puntos más grandes
            zoom=13,                               # Más zoom
            center={"lat": -36.606, "lon": -72.103},  # Centrado en Chillán
            height=600,
            labels={'prediccion': 'Índice de Congestión'}  # Label mejorado
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":0,"l":0,"b":0}
        )
        
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Configura los parámetros en el panel lateral y presiona 'Predecir Congestión'")

with col2:
    st.header("📋 Resultados por Segmento")
    
    if 'predicciones' in st.session_state:
        for pred in st.session_state['predicciones']:
            with st.container():
                if pred['categoria'] == 'Fluido':
                    st.success(f"**{pred['segmento_nombre']}**")
                elif pred['categoria'] == 'Moderado':
                    st.warning(f"**{pred['segmento_nombre']}**")
                else:
                    st.error(f"**{pred['segmento_nombre']}**")
                
                st.write(f"🚦 Índice: **{pred['prediccion']}**")
                st.write(f"📊 Estado: **{pred['categoria']}**")
                st.write(f"🚗 Vel. estimada: **{pred['velocidad_estimada']} km/h**")
                st.markdown("---")

# Análisis del modelo
st.header("📈 Análisis del Modelo")

tab1, tab2, tab3, tab4 = st.tabs(["Rendimiento Regresión", "Clasificación", "Distribución", "Matriz de Confusión"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de predicción vs real
        fig_scatter = px.scatter(
            x=y_test, y=y_pred_test,
            labels={'x': 'Valor Real', 'y': 'Valor Predicho'},
            title='Predicciones vs Valores Reales'
        )
        fig_scatter.add_trace(
            go.Scatter(x=[0, 100], y=[0, 100], mode='lines', name='Ideal', line=dict(dash='dash', color='red'))
        )
        st.plotly_chart(fig_scatter, width='stretch')
    
    with col2:
        # Distribución de errores
        errores = y_test - y_pred_test
        fig_hist = px.histogram(
            x=errores,
            nbins=30,
            labels={'x': 'Error de Predicción'},
            title='Distribución de Errores'
        )
        st.plotly_chart(fig_hist, width='stretch')

with tab2:
    st.subheader("📊 Reporte de Clasificación (Categorías)")
    
    # Mostrar métricas por categoría
    col1, col2, col3 = st.columns(3)
    
    categorias = ['Fluido', 'Moderado', 'Congestionado']
    for i, cat in enumerate(categorias):
        with [col1, col2, col3][i]:
            st.metric(
                f"{cat}",
                f"{report[cat]['f1-score']:.2%}",
                delta=f"Precision: {report[cat]['precision']:.2%}"
            )
            st.caption(f"Recall: {report[cat]['recall']:.2%} | Support: {int(report[cat]['support'])}")
    
    # Exactitud general
    st.metric("Exactitud General (Categorías)", f"{metricas['test_acc_cat']:.2%}")

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de índice de congestión
        fig_dist = px.histogram(
            df_global, x='indice_congestion', color='categoria_flujo',
            title='Distribución del Índice de Congestión',
            labels={'indice_congestion': 'Índice de Congestión'},
            nbins=50
        )
        st.plotly_chart(fig_dist, width='stretch')
    
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
            labels={'Real': 'Categoría Real', 'count': 'Frecuencia'}
        )
        st.plotly_chart(fig_cat, width='stretch')

with tab4:
    # Matriz de confusión
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_cat_test, y_pred_cat_test, labels=['Fluido', 'Moderado', 'Congestionado'])
    
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicho", y="Real", color="Cantidad"),
        x=['Fluido', 'Moderado', 'Congestionado'],
        y=['Fluido', 'Moderado', 'Congestionado'],
        title='Matriz de Confusión',
        text_auto=True,
        color_continuous_scale='Blues'
    )
    
    st.plotly_chart(fig_cm, width='stretch')
    
    # Heatmap de congestión por hora y día
    st.subheader("Patrón de Congestión: Hora vs Día")
    df_agg = df_global.groupby(['hora', 'dia_semana'])['indice_congestion'].mean().reset_index()
    
    # Reordenar días
    orden_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    df_agg['dia_semana'] = pd.Categorical(df_agg['dia_semana'], categories=orden_dias, ordered=True)
    df_agg = df_agg.sort_values('dia_semana')
    
    fig_heatmap = px.density_heatmap(
        df_agg, x='hora', y='dia_semana', z='indice_congestion',
        title='Congestión Promedio por Hora y Día',
        color_continuous_scale='RdYlGn_r',
        labels={'indice_congestion': 'Índice', 'hora': 'Hora del día', 'dia_semana': 'Día'}
    )
    st.plotly_chart(fig_heatmap, width='stretch')

# Comparación de features
st.header("Importancia de Features")

with st.expander("Ver lista de features utilizadas"):
    st.write(f"**Total de features:** {len(feature_cols)}")
    st.write("**Features incluidas:**")
    
    col1, col2 = st.columns(2)
    mid = len(feature_cols) // 2
    
    with col1:
        for feat in feature_cols[:mid]:
            st.text(f"• {feat}")
    
    with col2:
        for feat in feature_cols[mid:]:
            st.text(f"• {feat}")

st.markdown("""

**Autores:** Diego Loyola, Catalina Toro, Valentina Zúñiga | **Universidad del Bío-Bío**
""")