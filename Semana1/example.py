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
