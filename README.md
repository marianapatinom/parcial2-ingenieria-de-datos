# Pipeline de Datos y Despliegue 🛫

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)

**Autora:** Mariana Patiño Múnera

Bienvenido al repositorio oficial del proyecto universitario **"Pipeline de Datos y Despliegue"**. Esta aplicación fue diseñada para el análisis interactivo e inteligencia de incidentes y pérdidas operativas de aerolíneas, cubriendo todo el ciclo de vida del dato desde su procesamiento hasta su despliegue y visualización.

---

## 🛠️ Explicación Técnica del Proceso ETL

El procesamiento de datos fue desarrollado iterativamente para garantizar calidad computacional, reduciendo latencias en cálculos predictivos del Streamlit. A continuación, las bases del flujo ETL (Extract, Transform, Load):

### 1. Ingesta (Extract)
- **Fuentes de datos:** Se extrajeron los datos brutos del dataset "Airline Disruptions and Operational Impact" de Kaggle. La carga hacia un Dataframe en la memoria de Python se completó mediante las APIs de pandas para archivos CSV. Inicialmente proveía información sobre pérdidas económicas diarias (*estimated_daily_loss_usd*), pero la escala financiera requirió estructuración para integrarlo con sistemas de BI.

### 2. Transformación (Transform)
- **Limpieza y Manejo de Nulos:** A través del mapeo con variables lógicas como `isnull().sum()`, detectamos que el volumen requería aplicar en primera instancia `.dropna()` sobre registros incompletos para evitar fallos catastróficos en el Machine Learning Random Forest y subestimaciones visuales de los KPI de pérdida.
- **Normalización de Tipos:** Se hizo *cast* y *type-enforcing* en campos medulares como *cancelled_flights* y *passengers_impacted* garantizando la unicidad al tratarlos como `int` y no cadenas, propiciando su idoneidad para operaciones matemáticas.
- **Creación de Atributos Derivados:** Implementamos lógica vectorizada con numpy (`np.where`) inyectando categorización analítica (e.g., *impact_level* para discernir *High Impact* o *Low Impact* sobre la mediana operativa).

### 3. Carga (Load)
- **Estructura final de los datos:** Finalmente, se consolidó el subset purificado a una ruta local `cleaned_dataset.csv` o de producción. Estos datos son persistidos de manera inmutable tras la transformación y son la capa principal (Staging Area) de consulta a la cual se conecta Streamlit para sus queries rápidas en el backend de Python.

---

## 🚀 Ejecutar la App en Local

Para visualizar este proyecto desplegado desde tu ordenador en un ambiente de desarrollo local, sigue estos pasos:

1. Clona el repositorio a tu máquina:
   ```bash
   git clone https://github.com/marianapatinom/NOMBRE_DEL_REPO.git
   cd NOMBRE_DEL_REPO
   ```

2. Instala las dependencias y bibliotecas Python obligatorias listadas en el entorno:
   ```bash
   pip install -r requirements.txt
   ```

3. Asegúrate de ejecutar el servidor de desarrollo de Streamlit desde la raíz del proyecto para visualizar la topología modular:
   ```bash
   streamlit run app.py
   ```

4. Observa tu entorno interactivo disponible nativamente desde tu navegador web, típicamente en `http://localhost:8501`.
