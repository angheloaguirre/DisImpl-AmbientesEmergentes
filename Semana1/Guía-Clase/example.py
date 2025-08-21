import streamlit as st
import pandas as pd
import numpy as np
#from datetime import time, datetime
import datetime
import urllib.request

st.title('Semana 1')
st.write('¡Clase de prueba para aprender **funcionalidades básicas** de Streamlit!')


num = st.slider("Elija un número", 0, 100, step=1)
st.write("El numero ingresado es {}".format(num))

"""
appointment = st.slider(
    "Programe la asesoría:",
    value=(time(11, 30), time(12, 45))
)
st.write("Está agendado para: ", appointment)

start_time = st.slider(
    "Ver casos ocurridos en: ",
    value = datetime(2020, 1, 1, 9, 30),
    format="DD/MM/YY - hh:mm"
)
st.write("Fecha seleccionada: ", start_time)
"""

d = st.date_input(
    "Fecha de cumpleaños",
    datetime.date(2019, 7, 6)
)
st.write("Tu cumpleaños es: ", d)

option = st.selectbox(
    '¿Cómo desaría ser contactado/a?',
    ('Email', 'Teléfono', 'WhatsApp')
)
st.write("Usted seleccióno: ", option)

n = st.slider("Elija un número: ", 5, 100, step=1)
chart_data = pd.DataFrame(np.random.randn(n), columns=['data'])
st.line_chart(chart_data)

df = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon']
)

"""
@st.experimental_set_query_params
def download_data():
    url = 'https://files.minsa.gob.pe/s/eRqxR35ZCxrzNgr/download'
    filename = 'data.csv'
    urllib.request.urlretrieve(url, filename)

download_data()
"""
