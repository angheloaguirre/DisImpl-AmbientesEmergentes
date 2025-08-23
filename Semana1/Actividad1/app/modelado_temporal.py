import pandas as pd
import streamlit as st
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

# === 3.1 Generación de Series de Tiempo con Suavizado de 7 Días ===
def mostrar_series_tiempo(df):
    # Título y descripción
    st.subheader('🌍 3.1 Generación de Series de Tiempo con Suavizado de 7 Días')
    st.markdown("""
    **Descripción de la actividad:**  
    Este gráfico muestra los **casos confirmados** y **muertes suavizadas** a través de una media móvil de 7 días. El suavizado es importante para eliminar fluctuaciones de datos y observar tendencias más claras.

    Se generan los datos de manera simulada para los días previos a la fecha final y se muestran en un gráfico de líneas para facilitar la visualización de las tendencias.

    """)

    st.caption("⚠️ Solo se dispone de los reportes diarios de: 28/02/2022, 01/03/2022 y 19/04/2022.")

    # Selección de país
    country = st.selectbox("Seleccione un país para ver sus datos suavizados:", df['Country_Region'].unique())

    # Filtrar los datos por el país seleccionado
    df_country = df[df['Country_Region'] == country]

    # Agrupar por fecha y sumar los casos confirmados
    df_country_grouped = df_country.groupby('Last_Update').agg({'Confirmed': 'sum', 'Deaths': 'sum'}).reset_index()

    # Obtener la última fecha (Last_Update) en los datos del país
    last_update = df_country_grouped['Last_Update'].max()
    aux_date = last_update.strftime('%d/%m/%Y')

    colA, colB = st.columns(2)
    with colA:
        st.metric("Confirmados (día base)", df_country_grouped['Confirmed'].iloc[-1])
    with colB:
        st.metric("Muertes (día base)", df_country_grouped['Deaths'].iloc[-1])

    st.write(f"*Fecha base:* **{aux_date}** -- Fuente: reporte diario JHU (1 día).")

    # Generar las 7 fechas: 6 fechas anteriores a la de Last_Update
    date_list = [last_update]  # Agregamos la última fecha (Last_Update)
    for i in range(6):
        date_list.append(last_update - timedelta(days=i+1))  # Restamos un día por cada iteración

    # Convertir las fechas a formato "DD/MM/YYYY"
    date_list = [date.strftime('%d/%m/%Y') for date in reversed(date_list)]  # Reversed para tenerlas en orden ascendente

    # Crear la lista de valores para "Confirmed" en orden ascendente
    confirmed_values = [df_country_grouped['Confirmed'].iloc[-1]]  # Empezamos con el valor más reciente
    for i in range(6):
        # Generar un incremento del 5% a 20% de lo último (en función de los datos reales)
        percent = np.random.uniform(0.05, 0.20)  # Un rango más ajustado
        new_value = max(confirmed_values[-1] * (1 - percent), 0)  # Restamos un porcentaje para los días previos
        confirmed_values.append(new_value)

    # Ordenar los valores de casos confirmados de menor a mayor
    confirmed_values = sorted(confirmed_values)

    # Crear la lista de valores para "Deaths" de manera similar
    deaths_values = [df_country_grouped['Deaths'].iloc[-1]]  # Empezamos con el valor más reciente
    for i in range(6):
        # Generar un incremento del 2% a 10% de lo último (en función de los datos reales)
        percent = np.random.uniform(0.02, 0.10)  # Un rango más ajustado
        new_value = max(deaths_values[-1] * (1 - percent), 0)  # Restamos un porcentaje para los días previos
        deaths_values.append(new_value)

    # Ordenar los valores de muertes de menor a mayor
    deaths_values = sorted(deaths_values)

    # Crear DataFrame con índices que comienzan desde 1
    df_display = pd.DataFrame({
        'Fecha': date_list,
        'Casos Confirmados': np.array(confirmed_values).astype(int),  # Convertir a enteros
        'Muertes': np.array(deaths_values).astype(int)  # Convertir a enteros
    })
    df_display.index += 1  # Cambiar el índice para que empiece desde 1

    # Mostrar la tabla con las fechas y los valores
    st.write("### Tabla de Casos Confirmados y Muertes por Fecha:")
    st.write(df_display)

    # --- Función para formatear el eje Y en millares ---
    def thousands(x, pos):
        return '%1.0fK' % (x * 1e-3)  # Convertir a miles y mostrar 'K'

    # --- construir series con índice de fechas ---
    idx = pd.to_datetime(date_list, dayfirst=True)
    confirmed_sr = pd.Series(confirmed_values, index=idx)
    deaths_sr = pd.Series(deaths_values, index=idx)

    # --- Suavizado 7 días (rolling mean) ---
    confirmed_ma7 = confirmed_sr.rolling(window=7, min_periods=1, center=True).mean()
    deaths_ma7 = deaths_sr.rolling(window=7, min_periods=1, center=True).mean()

    # --- plot lado a lado ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Confirmados
    ax1.plot(idx, confirmed_sr.values, linewidth=1.2, alpha=0.35, label='Serie original')
    ax1.plot(idx, confirmed_ma7.values, linewidth=2.5, label='MA 7 días')
    ax1.set_title(f"Casos Confirmados Suavizados en {country}")
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Número de casos en millares')
    ax1.legend()
    for label in ax1.get_xticklabels(): 
        label.set_rotation(45)

    # Aplicar formateo a los valores del eje Y en 'ax1'
    ax1.yaxis.set_major_formatter(FuncFormatter(thousands))

    # Muertes
    ax2.plot(idx, deaths_sr.values, linewidth=1.2, alpha=0.35, label='Serie original', color='tab:orange')
    ax2.plot(idx, deaths_ma7.values, linewidth=2.5, label='MA 7 días', color='tab:red')
    ax2.set_title(f"Muertes Suavizadas en {country}")
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Número de muertes')
    ax2.legend()
    for label in ax2.get_xticklabels(): 
        label.set_rotation(45)

    # Mostrar los gráficos
    st.pyplot(fig)
