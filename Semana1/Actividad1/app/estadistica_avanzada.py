# estadistica_avanzada.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------
# Utilidad base: KPIs mínimos
# ---------------------------
def _kpis_por_pais(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Country_Region", as_index=False).agg({
        "Confirmed": "sum",
        "Deaths": "sum"
    })
    g["CFR"] = np.where(g["Confirmed"] > 0, (g["Deaths"] / g["Confirmed"]) * 100, np.nan)

    if "Incident_Rate" in df.columns:
        ir = df.groupby("Country_Region")["Incident_Rate"].mean().reset_index()
        g = g.merge(ir, on="Country_Region", how="left")
    else:
        g["Incident_Rate"] = np.nan

    g = g.rename(columns={
        "Country_Region": "Pais",
        "Confirmed": "Confirmados",
        "Deaths": "Fallecidos",
        "CFR": "CFR (%)",
        "Incident_Rate": "Tasa casos por 100k (Incident_Rate)"
    })
    return g


def mostrar_estadistica_avanzada(df: pd.DataFrame):
    # Calculamos una tabla base para CFR por país
    grouped = _kpis_por_pais(df)

    # =====================================================
    # 2.2 Generar intervalos de confianza para el CFR
    # =====================================================
    st.subheader("🧪 Intervalos de confianza para el CFR")

    c1, c2 = st.columns(2)
    with c1:
        min_confirm = st.number_input("Mínimo de confirmados por país",
                                      min_value=0, value=100, step=50)
    with c2:
        conf_level = st.slider("Nivel de confianza", 0.80, 0.99, 0.95, 0.01)
    alpha = 1 - conf_level

    mask_valid = grouped["Confirmados"] >= min_confirm
    low_ci = np.full(len(grouped), np.nan)
    up_ci  = np.full(len(grouped), np.nan)

    try:
        from statsmodels.stats.proportion import proportion_confint
        li, ls = proportion_confint(
            grouped.loc[mask_valid, "Fallecidos"].astype(int).values,
            grouped.loc[mask_valid, "Confirmados"].astype(int).values,
            alpha=alpha, method="wilson"
        )
        low_ci[mask_valid.values] = li * 100
        up_ci[mask_valid.values]  = ls * 100
        used_method = "Wilson"
    except Exception:
        from scipy.stats import norm
        z = norm.ppf(1 - alpha/2)
        p = (grouped.loc[mask_valid, "Fallecidos"] / grouped.loc[mask_valid, "Confirmados"]).astype(float)
        n = grouped.loc[mask_valid, "Confirmados"].astype(float)
        se = np.sqrt(np.clip(p*(1-p)/n, 0, None))
        low_ci[mask_valid.values] = (p - z*se) * 100
        up_ci[mask_valid.values]  = (p + z*se) * 100
        used_method = "Aprox. normal"

    ic_df = grouped.copy()
    ic_df["CFR_LI (%)"] = low_ci
    ic_df["CFR_LS (%)"] = up_ci

    st.caption(f"Nota: método usado para IC = {used_method}.")
    st.dataframe(
        ic_df.sort_values("CFR (%)", ascending=False)[
            ["Pais", "Confirmados", "Fallecidos", "CFR (%)",
             "CFR_LI (%)", "CFR_LS (%)", "Tasa casos por 100k (Incident_Rate)"]
        ]
    )
    st.download_button(
        "⬇️ Descargar IC de CFR (CSV)",
        ic_df.to_csv(index=False).encode("utf-8"),
        file_name="intervalos_cfr.csv",
        mime="text/csv"
    )

    # =====================================================
    # 2.3 Test de hipótesis de proporciones (CFR A vs B)
    # =====================================================
    st.subheader("⚖️ Test de hipótesis: comparar CFR entre dos países")

    paises = grouped["Pais"].dropna().sort_values().tolist()
    t1, t2, t3 = st.columns([1, 1, 1.2])
    with t1:
        pais_a = st.selectbox("País A", paises, index=0, key="th_pais_a")
    with t2:
        idx_b = 1 if len(paises) > 1 else 0
        pais_b = st.selectbox("País B", paises, index=idx_b, key="th_pais_b")
    with t3:
        alpha_test = st.slider("α (significancia)", 0.001, 0.20, 0.05, 0.005)

    if pais_a == pais_b:
        st.info("Selecciona dos países distintos.")
        res_df = pd.DataFrame()
    else:
        fila_a = grouped[grouped["Pais"] == pais_a].iloc[0]
        fila_b = grouped[grouped["Pais"] == pais_b].iloc[0]
        x = np.array([int(fila_a["Fallecidos"]), int(fila_b["Fallecidos"])])  # éxitos
        n = np.array([int(fila_a["Confirmados"]), int(fila_b["Confirmados"])])  # ensayos

        if (n <= 0).any():
            st.warning("Alguno de los países tiene 0 confirmados. No se puede realizar el test.")
            res_df = pd.DataFrame()
        else:
            try:
                from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
                stat, pval = proportions_ztest(count=x, nobs=n, alternative="two-sided")
                eff = proportion_effectsize(x[0]/n[0], x[1]/n[1])  # h de Cohen
            except Exception:
                from scipy.stats import norm
                p_pool = x.sum() / n.sum()
                se = np.sqrt(p_pool*(1-p_pool)*(1/n[0] + 1/n[1]))
                stat = np.nan if se == 0 else (x[0]/n[0] - x[1]/n[1]) / se
                pval = np.nan if np.isnan(stat) else 2 * (1 - norm.cdf(abs(stat)))
                eff = np.nan

            cfr_a = (x[0]/n[0])*100
            cfr_b = (x[1]/n[1])*100

            st.markdown(f"""
**Resultados**
- CFR {pais_a}: **{cfr_a:.2f}%**  (Fallecidos: {x[0]} / Confirmados: {n[0]})
- CFR {pais_b}: **{cfr_b:.2f}%**  (Fallecidos: {x[1]} / Confirmados: {n[1]})
- Estadístico z: **{stat:.3f}**
- p-valor: **{pval:.4f}**
- α: **{alpha_test:.3f}**
""")
            if pd.notna(pval) and pval < alpha_test:
                st.success("Conclusión: **Se rechaza H₀**. Hay diferencia significativa en los CFR.")
            elif pd.notna(pval):
                st.info("Conclusión: **No se rechaza H₀**.")
            else:
                st.warning("No fue posible calcular el p-valor.")

            if not np.isnan(eff):
                st.caption(f"Tamaño de efecto (h de Cohen): {eff:.3f} (≈0.2 pequeño, 0.5 mediano, 0.8 grande)")

            res_df = pd.DataFrame({
                "Pais": [pais_a, pais_b],
                "Fallecidos": x,
                "Confirmados": n,
                "CFR (%)": [cfr_a, cfr_b],
                "z": [stat, stat],
                "p_valor": [pval, pval],
                "alpha": [alpha_test, alpha_test]
            })

    if not res_df.empty:
        st.download_button(
            "⬇️ Descargar resultados del test (CSV)",
            res_df.to_csv(index=False).encode("utf-8"),
            file_name="test_hipotesis_cfr.csv",
            mime="text/csv"
        )

    # =====================================================
    # Boxplot de CFR y descargas
    # =====================================================
    st.subheader("📊 Boxplot de CFR (%)")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(grouped["CFR (%)"].dropna(), vert=True, patch_artist=True)
    ax.set_ylabel("CFR (%)")
    ax.set_title("Distribución de CFR (%) por países")
    st.pyplot(fig)

    # Descargas útiles
    st.download_button(
        "⬇️ Descargar tabla base (CFR por país) (CSV)",
        grouped.to_csv(index=False).encode("utf-8"),
        file_name="cfr_por_pais.csv",
        mime="text/csv"
    )
