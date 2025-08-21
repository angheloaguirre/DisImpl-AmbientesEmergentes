import streamlit as st
import pandas as pd
import numpy as np
from datetime import time

st.title('Semana 1')
st.write('¡Clase de prueba para aprender **funcionalidades básicas** de Streamlit!')


num = st.slider("Elija un número", 0, 100, step=1)
st.write("El numero ingresado es {}".format(num))

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
