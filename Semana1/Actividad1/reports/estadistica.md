<h1>Estadísticas de Proyección y Bandas de Confianza</h1>

<p>Este archivo contiene una descripción detallada de los gráficos generados en las funciones de proyección de casos confirmados y muertes, así como las bandas de confianza asociadas. Los gráficos muestran las proyecciones futuras basadas en modelos estadísticos, específicamente el modelo ETS (Exponential Smoothing), junto con sus límites de confianza al 95%.</p>

<h2>1. Proyección a N Días Hacia Adelante</h2>

<h3>Descripción:</h3>
<p>Esta sección muestra las proyecciones de casos confirmados y muertes en función del modelo ETS. Los datos históricos (últimos 60 días) se presentan junto con las proyecciones futuras. Las bandas de confianza al 95% están representadas para ilustrar la incertidumbre del pronóstico.</p>

<h3>Gráficos disponibles:</h3>
<ul>
  <li>Proyección a 7 días</li>
  <li>Proyección a 14 días</li>
  <li>Proyección a X días (ajustable con un slider)</li>
</ul>

<h3>Archivos descargables:</h3>
<ul>
  <li><a href="graficas/series_tiempo_Afghanistan.png">Gráfico de Proyección a 7 días</a></li>
  <li><a href="graficas/proyeccion_Spain_Confirmed_forecast.png">Gráfico de Proyección a 14 días</a></li>
</ul>

<h2>2. Residuos del Modelo ETS</h2>

<h3>Descripción:</h3>
<p>Los residuos del modelo ETS representan las diferencias entre los valores reales y los valores pronosticados. Esta sección muestra los residuos a lo largo del tiempo para evaluar la precisión del modelo y si hay patrones inesperados.</p>

<h3>Gráfico disponible:</h3>
<ul>
  <li>Gráfico de residuos del modelo ETS</li>
</ul>

<h3>Archivos descargables:</h3>
<ul>
  <li><a href="graficas/backtesting.png">Gráfico de Residuos del Modelo ETS</a></li>
</ul>

<h2>3. Estadísticas del Pronóstico</h2>

<h3>Descripción:</h3>
<p>Las estadísticas de las proyecciones incluyen las métricas del error (como el MAE, MAPE) y los valores máximos y mínimos de las bandas de confianza.</p>

<h3>Métricas utilizadas:</h3>
<ul>
  <li>MAE (Error Absoluto Medio)</li>
  <li>MAPE (Error Absoluto Porcentual Medio)</li>
</ul>

<h3>Archivos adicionales:</h3>
<ul>
  <li><a href="graficas/grafico_control_confirmados.png">Gráfico de Control de Confirmados</a></li>
  <li><a href="graficas/grafico_control_mortalidad.png">Gráfico de Control de Mortalidad</a></li>
</ul>
