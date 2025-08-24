import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mostrar_calidad_datos(df):
    # ==============================
    #Valores nulos
    # ==============================

    st.write("**Valores nulos por columna:**")
    st.write(df.isnull().sum())

    # ==============================
    #Inconsistencias
    # ==============================

    inconsistencias = df[(df["Confirmed"] < 0) | (df["Deaths"] < 0)]
    if inconsistencias.empty:
        st.success("✅ No se detectaron inconsistencias (valores negativos).")
    else:
        st.warning("⚠️ Se encontraron inconsistencias:")
        st.dataframe(inconsistencias)

    # ==============================
    # Gráfico de control
    # ==============================
  
    grafico_control_confirmados(df)
    grafico_control_mortalidad(df)

def grafico_control_confirmados(df):
    # Agrupar por país
    grouped = df.groupby("Country_Region", as_index=False).agg({
        "Confirmed": "sum"
    })

    # Calcular estadísticas
    mean_conf = grouped["Confirmed"].mean()
    std_conf = grouped["Confirmed"].std()
    ucl = mean_conf + 3 * std_conf
    lcl = max(mean_conf - 3 * std_conf, 0)

    # Gráfico de control
    plt.figure(figsize=(14, 6))
    plt.plot(grouped.index, grouped["Confirmed"], marker="o", linestyle="-", label="Casos confirmados")
    plt.axhline(mean_conf, color="green", linestyle="--", label=f"Media ({mean_conf:.0f})")
    plt.axhline(ucl, color="red", linestyle="--", label=f"UCL ({ucl:.0f})")
    plt.axhline(lcl, color="orange", linestyle="--", label=f"LCL ({lcl:.0f})")

    plt.xticks(grouped.index, grouped["Country_Region"], rotation=90)
    plt.ylabel("Casos confirmados")
    plt.title("Gráfico de Control - Casos Confirmados por País")
    plt.legend()

    st.pyplot(plt)


def grafico_control_mortalidad(df):
    # Agrupar por país
    grouped = df.groupby("Country_Region", as_index=False).agg({
        "Confirmed": "sum",
        "Deaths": "sum"
    })
    
    # Calcular CFR (%)
    grouped["CFR"] = (grouped["Deaths"] / grouped["Confirmed"]).replace([float("inf"), None], 0) * 100
    
    # Calcular estadísticas
    mean_cfr = grouped["CFR"].mean()
    std_cfr = grouped["CFR"].std()
    ucl = mean_cfr + 3 * std_cfr
    lcl = max(mean_cfr - 3 * std_cfr, 0)  # No puede ser negativo

    # Gráfico
    plt.figure(figsize=(14, 6))
    plt.plot(grouped.index, grouped["CFR"], marker="o", linestyle="-", label="CFR por país")
    plt.axhline(mean_cfr, color="green", linestyle="--", label=f"Media ({mean_cfr:.2f}%)")
    plt.axhline(ucl, color="red", linestyle="--", label=f"UCL ({ucl:.2f}%)")
    plt.axhline(lcl, color="orange", linestyle="--", label=f"LCL ({lcl:.2f}%)")
    
    plt.xticks(grouped.index, grouped["Country_Region"], rotation=90)
    plt.ylabel("CFR (%)")
    plt.title("Gráfico de Control - Índice de Mortalidad (CFR) por País")
    plt.legend()

    st.pyplot(plt)