import streamlit as st
import pandas as pd
import numpy as np
from datetime import time

st.title('Semana 1 - Actividad 1')
st.write('Hecho por el grupo 5')

# Leer el dataset
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-18-2022.csv"
df = pd.read_csv(url)

# Convertir la columna "Last Update" a tipo datetime
df['Last Update'] = pd.to_datetime(df['Last Update'])

# Suponiendo que 'Country/Region' y 'Confirmed' son las columnas de interés
df_grouped = df.groupby(['Country/Region', 'Last Update'])[['Confirmed']].sum().reset_index()

# Suavizar los casos confirmados por país con una ventana de 7 días
df_grouped['Confirmed_smoothed'] = df_grouped.groupby('Country/Region')['Confirmed'].rolling(window=7).mean().reset_index(level=0, drop=True)

# Mostrar una serie de tiempo de un país de ejemplo
country_data = df_grouped[df_grouped['Country/Region'] == 'US']  # Cambiar por el país que desees
country_data.set_index('Last Update', inplace=True)

# Visualizar la serie suavizada
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(country_data.index, country_data['Confirmed_smoothed'], label='Casos confirmados (suavizados)')
plt.title('Casos Confirmados Suavizados de COVID-19 por País')
plt.xlabel('Fecha')
plt.ylabel('Casos Confirmados')
plt.legend()
plt.xticks(rotation=45)
plt.show()
