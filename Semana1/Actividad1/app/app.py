import pandas as pd
import streamlit as st
from modelado_temporal import mostrar_series_tiempo, mostrar_modelado_forecast, bandas_confianza
from vista_general import mostrar_topn_mapa
from clustering_pca import mostrar_clustering_pca
from calidad_datos import mostrar_calidad_datos
from estadistica_avanzada import mostrar_estadistica_avanzada

# Credenciales simples (ejemplo, no usar en producción)
USERS = {
    "pepe123": "123456",
    "admin": "admin"
}

# Si no existe el estado de login, inicializar
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Función para login
def login():
    st.image("LogoEpiWatch.png", width=150)
    st.subheader("EpiWatch: Inicia sesión")
    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")

    if st.button("Ingresar"):
        if username in USERS and USERS[username] == password:
            st.session_state.logged_in = True
            st.success("✅ Sesión iniciada")
            st.rerun()
        else:
            st.error("❌ Usuario o contraseña incorrectos")

# Mostrar login si no está logueado
if not st.session_state.logged_in:
    login()
else:
    # configuración básica
    st.set_page_config(page_title="COVID-19 JHU – Métricas y Análisis",layout="wide")
    colImg, colHeader = st.columns([1, 11])  # adjust proportions
    with colImg:
        st.image("LogoEpiWatch.png", width=80)
    with colHeader:
        st.title("EpiWatch: COVID-19 Dashboard")
    st.caption("Fuente: Johns Hopkins CSSE – Daily Report 2022-04-18")

    # Cargar los datos
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-18-2022.csv"
    @st.cache_data
    def load_data():
        df = pd.read_csv(url)
        df['Last_Update'] = pd.to_datetime(df['Last_Update'])  # Convertir a tipo datetime
        return df

    # Cargar los datos
    df = load_data()
    df['Confirmed'] = df['Confirmed'].fillna(0).astype(int)
    df['Deaths'] = df['Deaths'].fillna(0).astype(int)

    # Sidebar con filtros
    st.sidebar.header("Filtros")

    # Por rango de fechas
    if "Last_Update" in df.columns:
        df["Last_Update"] = pd.to_datetime(df["Last_Update"], errors="coerce")
        min_date = df["Last_Update"].min()
        max_date = df["Last_Update"].max()
        date_range = st.sidebar.date_input("Rango de fechas", [min_date, max_date])
        if len(date_range) == 2:
            df = df[(df["Last_Update"].dt.date >= date_range[0]) &
                    (df["Last_Update"].dt.date <= date_range[1])]

    # Por países
    paises = df["Country_Region"].unique()

    st.sidebar.markdown(
        """
        <style>
        div[data-baseweb="select"] > div {
            max-height: 200px;   /* altura máxima */
            overflow-y: auto;  /* scroll vertical */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Checkbox para seleccionar todos
    select_all = st.sidebar.checkbox("Seleccionar todos los países", value=False)

    if select_all:
        paises_sel = st.sidebar.multiselect("Selecciona países", options=paises, default=paises)
    else:
        paises_sel = st.sidebar.multiselect("Selecciona países", options=paises, default=paises[:5])

    df = df[df["Country_Region"].isin(paises_sel)]

    # Verificar si el DataFrame está vacío después de los filtros
    if df.empty:
        st.error("🚫 No hay datos disponibles para los filtros seleccionados.")
        st.warning("⚠️ Por favor elija al menos 1 país para poder ver los datos.")
        st.stop()  # Detener la ejecución si el DataFrame está vacío

    # Filtro por provincias/estados
    if "Province_State" in df.columns:
        provincias = df["Province_State"].dropna().unique()
        provincias_sel = st.sidebar.multiselect("Selecciona provincias/estados", options=provincias)
        if provincias_sel:
            df = df[df["Province_State"].isin(provincias_sel)]

    # Filtro por umbral de confirmados
    umbral_conf = st.sidebar.slider("Umbral mínimo de confirmados", 0, int(df["Confirmed"].max()), 1000)
    df = df[df["Confirmed"] >= umbral_conf]

    # Filtro por carga de población
    if "Population" in df.columns:
        pop_min, pop_max = st.sidebar.slider("Rango de población", 
                                            int(df["Population"].min()), 
                                            int(df["Population"].max()), 
                                            (int(df["Population"].min()), int(df["Population"].max())))
        df = df[(df["Population"] >= pop_min) & (df["Population"] <= pop_max)]

    #KPIs principales
    # Agrupar por país
    grouped = df.groupby("Country_Region", as_index=False).agg({
        "Confirmed": "sum",
        "Deaths": "sum"
    })

    # Verificar si el DataFrame 'grouped' tiene datos para evitar errores en cálculos
    if grouped.empty:
        st.error("🚫 No hay datos disponibles para los filtros seleccionados.")
        st.stop()  # Detener la ejecución si el DataFrame 'grouped' está vacío

    # Calcular CFR (muertes / confirmados)
    grouped["CFR"] = (grouped["Deaths"] / grouped["Confirmed"]) * 100

    # Calcular un promedio por país:
    incident_rate = df.groupby("Country_Region")["Incident_Rate"].mean().reset_index()
    grouped = grouped.merge(incident_rate, on="Country_Region")

    # Renombrar columnas
    grouped = grouped.rename(columns={
        "Country_Region": "Pais",
        "Confirmed": "Confirmados",
        "Deaths": "Fallecidos",
        "CFR": "CFR (%)",
        "Incident_Rate": "Tasa casos por 100k (Incident_Rate)"
    })

    # Mostrar resultados
    st.subheader("📈 KPIs Principales")
    st.dataframe(grouped)

    # ----------------------------
    # Narrativa automática
    # ----------------------------
    st.subheader("📝 Narrativa de los resultados")

    if len(grouped) < 3:
        st.warning("⚠️ Selecciona al menos **3 países** para generar la narrativa.")
    else:
        # Top 3 países con mayor CFR
        top_cfr = grouped.sort_values(by="CFR (%)", ascending=False).head(3)

        # Top 3 países con mayor Tasa de Incidencia
        top_incidence = grouped.sort_values(by="Tasa casos por 100k (Incident_Rate)", ascending=False).head(3)

        narrativa = f"""
        El análisis de los indicadores epidemiológicos muestra diferencias claras entre países:

        - Los países con **mayor letalidad (CFR)** son:  
        🥇 {top_cfr.iloc[0]['Pais']} ({top_cfr.iloc[0]['CFR (%)']:.2f}%),  
        🥈 {top_cfr.iloc[1]['Pais']} ({top_cfr.iloc[1]['CFR (%)']:.2f}%),  
        🥉 {top_cfr.iloc[2]['Pais']} ({top_cfr.iloc[2]['CFR (%)']:.2f}%).  

        - En cuanto a la **tasa de incidencia por 100k habitantes**, los más afectados son:  
        🥇 {top_incidence.iloc[0]['Pais']} ({top_incidence.iloc[0]['Tasa casos por 100k (Incident_Rate)']:.2f}),  
        🥈 {top_incidence.iloc[1]['Pais']} ({top_incidence.iloc[1]['Tasa casos por 100k (Incident_Rate)']:.2f}),  
        🥉 {top_incidence.iloc[2]['Pais']} ({top_incidence.iloc[2]['Tasa casos por 100k (Incident_Rate)']:.2f}).  

        🔎 Estos resultados permiten contrastar países con **alta propagación pero baja letalidad**, frente a otros con **menor número de casos pero mayor mortalidad relativa**. 
        """
        st.write(narrativa)

    # Definición de las pestañas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📂 Vista General",
        "📈 Estadística Avanzada",
        "📈 Modelado temporal",
        "📊 Clustering y PCA",
        "🔎 Calidad de datos"
    ])

    # ==========================
    # Contenido de las pestañas
    # ==========================

    #Vista general
    with tab1:
        st.header("📂 Vista General")
        mostrar_topn_mapa(df)
    #Estadística
    with tab2:
        st.header("📈 Estadística Avanzada")
        st.write("Aquí se calcularán las métricas clave por país (Confirmados, Fallecidos, CFR, tasas por 100k).")
        mostrar_estadistica_avanzada(df)

    #Modelado temporal
    with tab3:
        st.header("🧪 Modelado temporal")
        # === 3.1 Generación de Series de Tiempo con Suavizado de 7 Días ===
        mostrar_series_tiempo(df)
        mostrar_modelado_forecast(url, df)
        bandas_confianza(df)
    #Clusters
    with tab4:
        st.header("📊 Clustering y PCA")
        st.write("Aquí se construirá el clustering de países con K-means y se mostrarán los grupos.")
        mostrar_clustering_pca(df)

    #Calidad de datos
    with tab5:
        st.header("🔎 Calidad de datos")
        st.write("Aquí se reducirá la dimensionalidad con PCA y se graficarán los componentes principales.")
        mostrar_calidad_datos(df)
