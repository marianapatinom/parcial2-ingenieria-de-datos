import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import graphviz

# Configuración de página
st.set_page_config(page_title="Airline Disruptions Analytics", page_icon="✈️", layout="wide")

# Estilos personalizados premium para Streamlit
st.markdown("""
    <style>
    /* Estilo General y Tipografía */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Fondo General de la App */
    .stApp {
        background-color: #f8fafc;
        background-image: radial-gradient(#e2e8f0 1px, transparent 1px);
        background-size: 20px 20px;
    }

    /* Estilo del Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
        border-right: none;
    }

    /* Tarjetas de Métricas (KPIs) Animadas */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 5px solid #3b82f6;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px -5px rgba(59, 130, 246, 0.15);
        border-left: 5px solid #8b5cf6;
    }

    /* Valores y Etiquetas de KPIs */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        color: #0f172a;
        font-weight: 800;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.05rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Títulos */
    h1 {
        background: -webkit-linear-gradient(45deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    h2, h3 {
        color: #1e293b;
        font-weight: 700;
    }
    
    /* Alertas */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

import os

# Cachear la carga de datos
@st.cache_data
def load_data():
    try:
        # Construir la ruta absoluta correcta independiente de desde dónde se ejecute streamlit
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, "data", "airline_losses.csv")
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        # Fallback de inicialización caso no encuentre la ruta
        df = pd.DataFrame()
    return df

df = load_data()

# Título y Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3163/3163183.png", width=100)
st.sidebar.title("Navegación")
st.sidebar.markdown("---")

menu = st.sidebar.radio("Opciones de Exploración", 
                        ["📊 Dashboard Analítico", "🤖 Análisis Predictivo", "⚙️ Orquestación de Datos"])

st.sidebar.markdown("---")
st.sidebar.info("Desarrollado por Mariana Patiño Múnera.")

if df.empty:
    st.error("Ruta de datos no encontrada. Asegúrate de tener 'data/airline_losses.csv' en la raíz.")
else:
    if menu == "📊 Dashboard Analítico":
        st.title("📊 Dashboard Analítico: Interrupciones en Aerolíneas")
        st.markdown("Visualización integral sobre los principales KPIs e historial de operaciones logísticas.")
        
        # Filtros en columnas
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            region_list = ["All"] + list(df['region'].unique())
            region_filter = st.selectbox("🌍 Filtro por Región:", region_list)
        with col_f2:
            type_list = ["All"] + list(df['airline_type'].unique())
            type_filter = st.selectbox("✈️ Filtro por Tipo de Aerolínea:", type_list)
            
        df_filtered = df.copy()
        if region_filter != "All":
            df_filtered = df_filtered[df_filtered['region'] == region_filter]
        if type_filter != "All":
            df_filtered = df_filtered[df_filtered['airline_type'] == type_filter]
            
        # KPIs Principales
        st.markdown("### 🔑 Indicadores Principales (KPIs)")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Pérdida Total (USD)", f"${df_filtered['estimated_loss_usd'].sum():,.0f}")
        kpi2.metric("Total de Vuelos Cancelados", f"{df_filtered['cancellations_count'].sum():,}")
        kpi3.metric("Total Rerutas Generadas", f"{df_filtered['reroutes_count'].sum():,}")
        kpi4.metric("Desviación Media de Ganancia", f"{df_filtered['revenue_loss_pct'].mean():,.1f}%")
        st.markdown("---")
        
        # Gráficos de Analítica
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig1 = px.bar(df_filtered.sort_values(by='estimated_loss_usd', ascending=False).head(10), 
                          x='airline', y='estimated_loss_usd', 
                          color='airline_type',
                          title="Top 10 Aerolíneas por Pérdida Estimada (USD)",
                          template="plotly_white")
            st.plotly_chart(fig1, use_container_width=True)
            
        with col_c2:
            fig2 = px.pie(df_filtered, names='airline_type', values='cancellations_count',
                          title="Distribución de Cancelaciones por Tipo de Aerolínea",
                          hole=0.4)
            st.plotly_chart(fig2, use_container_width=True)
            
        # Mapa
        st.markdown("### 🗺️ Impacto Operativo Geográfico")
        st.write("Visión global centralizada de los incidentes usando metadatos geográficos (Países).")
        df_country = df_filtered.groupby('country')[['cancellations_count', 'estimated_loss_usd']].sum().reset_index()
        fig_map = px.choropleth(df_country, locations='country', locationmode='country names',
                                color='estimated_loss_usd', title="Mapa Global de Impacto Financiero",
                                hover_name='country', hover_data=['cancellations_count'],
                                color_continuous_scale="Reds")
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Tabla Exploratoria
        with st.expander("Ver tabla de frecuencias de los datos tabulares"):
            st.dataframe(df_filtered.style.background_gradient(subset=["estimated_loss_usd"], cmap="Blues"))

    elif menu == "🤖 Análisis Predictivo":
        st.title("🤖 Análisis Predictivo en Tiempo Real")
        st.markdown("Simulación con Machine Learning para calcular los sobrecostos y la pérdida total en caso de un incidente imprevisto futuro.")
        
        # Preparar modelo sencillo (Random Forest Regressor)
        X = df[['cancellations_count', 'reroutes_count', 'revenue_loss_pct']]
        y = df['estimated_loss_usd']
        
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        
        st.sidebar.markdown("### ⚙️ Parámetros del Evento")
        sim_cancellations = st.sidebar.slider("Cant. Vuelos Cancelados", 0, int(df['cancellations_count'].max() * 1.5), int(df['cancellations_count'].mean()))
