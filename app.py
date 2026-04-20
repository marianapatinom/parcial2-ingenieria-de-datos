import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import graphviz

# Configuración de página
st.set_page_config(page_title="Airline Disruptions Analytics", page_icon="✈️", layout="wide")

# Estilos personalizados intermedios para Streamlit
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1, h2, h3 {color: #1e3a8a;}
    .stMetric {background-color: white; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);}
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
        sim_reroutes = st.sidebar.slider("Cant. Redireccionamientos", 0, int(df['reroutes_count'].max() * 1.5), int(df['reroutes_count'].mean()))
        sim_rev_loss = st.sidebar.number_input("Pérdida de Rev Estimada (%)", min_value=0.0, max_value=100.0, value=float(df['revenue_loss_pct'].mean()))
        
        st.markdown("### 🧠 Simulación de Impacto del Riesgo")
        st.info("Ajuste los parámetros en la barra lateral para observar la inferencia generada por el Random Forest en vivo.")
        
        pred_loss = model.predict(np.array([[sim_cancellations, sim_reroutes, sim_rev_loss]]))[0]
        st.metric(label="Pérdida Financiera Estimada", value=f"${pred_loss:,.2f} USD")
        
    elif menu == "⚙️ Orquestación de Datos":
        st.title("⚙️ Orquestación de Datos (Apache Airflow)")
        st.markdown("Sección interactiva de simulación de flujos de trabajo basados en el ecosistema **Apache Airflow**. Se incluyen conceptos básicos de programación de workflows implementando **DAGs (Directed Acyclic Graphs)**.")
        
        st.markdown("---")
        st.markdown("### 🔹 Ejercicio 1: DAG ETL Pipeline (`covid_etl_pipeline.py`) adaptado a Aerolíneas")
        st.write('''
            **Descripción del Ejercicio:**  
            Este ejercicio define y documenta un flujo de automatización completo de Extracción, Transformación, Verificación de Calidad y Carga (Data Quality ETL). Es uno de los estándares principales enseñados para construir pipelines resilientes que transforman un feed externo en datos listos para el Data Warehouse.
            
            **Secuencia de Tareas:**
            1. **`extract_data`**: Usando sensores y peticiones, descarga los datos crudos diariamente y comunica la ruta temporal usando mecanismos (XComs).
            2. **`transform_data`**: Aplica operaciones de limpieza para sanear valores nulos, estandariza monedas o formatos y procesa las métricas de negocio.
            3. **`quality_check`**: Un punto de validación esencial. Detiene el proceso (falla la tarea) si el archivo procesado está vacío o si superan los umbrales de nulos permitidos.
            4. **`load_data`**: Movimiento final de los datos seguros persistiendo su almacenamiento en BigQuery, Postgres o almacenes de analítica.
        ''')
        
        dag1 = graphviz.Digraph(engine='dot')
        dag1.attr(rankdir='LR')
        dag1.node('A', '🌐 extract_data', shape='box', style='filled', fillcolor='#D4E6F1')
        dag1.node('B', '🔧 transform_data', shape='box', style='filled', fillcolor='#D4E6F1')
        dag1.node('C', '🔍 quality_check', shape='box', style='filled', fillcolor='#F9E79F')
        dag1.node('D', '💾 load_data', shape='box', style='filled', fillcolor='#A9DFBF')
        
        dag1.edges(['AB', 'BC', 'CD'])
        st.graphviz_chart(dag1)
        
        st.markdown("---")
        st.markdown("### 🔹 Ejercicio 2: Monitoreo de Procesos y Control de Flujo (Trigger Rules)")
        st.write('''
            **Descripción del Ejercicio:**  
            En sistemas en producción, no todos los DAGs son estrictamente secuenciales. Este ejercicio básico de programación de workflows demuestra el control avanzado de ejecución mediante las **Trigger Rules**. Sirve como un DAG de monitoreo paralelo para administrar alertas en los sistemas lógicos.
            
            **Conceptos de Reglas Aplicados:**
            - **`all_success` (Ejecución Normal):** La regla estándar de progreso. Las tareas principales deben concluir con éxito para avanzar a la tarea principal de fin de batch.
            - **`one_failed` (Alerta Crítica):** Se define un "nodo listener". Si cualquiera de las tareas de ingesta del workflow llegara a fallar, este nodo se dispara automáticamente para enviar una alerta inmediata a los analistas (vía Slack/Email).
            - **`one_success`:** Variante útil para proceder si al menos un servidor o API espejo logró devolver la respuesta sin esperar al resto.
        ''')
        
        dag2 = graphviz.Digraph()
        dag2.attr(rankdir='TD')
        dag2.node('T1', 'api_ingestion_1', shape='box', style='filled', fillcolor='#D4E6F1')
        dag2.node('T2', 'api_ingestion_2 (Fails)', shape='box', style='filled', fillcolor='#F5B7B1')
        dag2.node('T3', 'api_ingestion_3', shape='box', style='filled', fillcolor='#D4E6F1')
        
        dag2.node('N1', '✅ normal_flow\n(all_success)', shape='ellipse', style='filled', fillcolor='#A9DFBF')
        dag2.node('N2', '🚨 slack_alert\n(one_failed)', shape='octagon', style='filled', fillcolor='#F1948A')
        
        # Conexiones
        for t in ['T1', 'T2', 'T3']:
            dag2.edge(t, 'N1', style='dashed')
            dag2.edge(t, 'N2')
            
        st.graphviz_chart(dag2)
        
        st.success("Ejercicios basados y escalados referenciando los fundamentos presentados en la plataforma: airflowdocker.netlify.app")
