## Link de la app desplegada en el streamlit
https://energy-prediction-lewagon.streamlit.app/

# Análisis y Predicción de Energías

## Descripción del Proyecto

Esta aplicación web desarrollada con Streamlit proporciona un análisis detallado y predicciones del consumo y generación de energía a nivel mundial, por continente y para países específicos. La herramienta visualiza datos históricos de producción energética, muestra la distribución de diferentes fuentes de energía y utiliza modelos de predicción basados en Prophet para estimar la generación de energía hasta 2035.

## Características Principales

- **Visualización de datos históricos** de generación eléctrica global, por continente y por país
- **Análisis de la distribución de fuentes de energía** (nuclear, solar, eólica, gas, hidráulica, biocombustibles, carbón y petróleo)
- **Modelos predictivos** basados en Prophet para estimar la generación de energía futura
- **Mapas interactivos** que muestran las regiones analizadas
- **Tablas de predicción** con valores estimados para años futuros
- **Gráficos comparativos** de la evolución histórica y proyecciones futuras

## Requisitos

Para ejecutar esta aplicación, necesitas tener instaladas las siguientes librerías de Python:

```
streamlit
pandas
seaborn
matplotlib
prophet
plotly
```

Puedes instalarlas usando pip:

```bash
pip install streamlit pandas seaborn matplotlib prophet plotly
```

## Estructura del Código

El código se organiza en varias funciones principales:

### 1. Carga y Preprocesamiento de Datos

- `load_data()`: Carga los datos desde el archivo CSV y realiza el preprocesamiento inicial
- `data_energia()`: Prepara datos de generación y consumo de electricidad por país/continente
- `data_continente()`: Agrega datos por continente
- `data_prophet()`: Prepara datos específicamente para los modelos de Prophet

### 2. Visualización de Datos

- `plot_energy_distribution()`: Crea gráficos de distribución de energía por tipo
- `mapa_region()`: Genera mapas interactivos para las regiones seleccionadas
- `prophet_energias()`: Genera visualizaciones y predicciones por tipo de energía

### 3. Modelos Predictivos

- `prediction_prophet()`: Ejecuta modelos Prophet con variables exógenas para predecir la generación de energía

### 4. Interfaz de Usuario

- `main()`: Función principal que configura la interfaz de usuario y coordina la visualización de todos los componentes

## Funcionamiento

La aplicación permite al usuario:

1. Seleccionar una región o país específico para analizar
2. Ajustar el año hasta el cual se realizarán las predicciones (2024-2035)
3. Visualizar gráficos de evolución histórica y proyecciones futuras
4. Consultar tablas con valores numéricos de las predicciones
5. Explorar la distribución de diferentes fuentes de energía

## Datos Utilizados

La aplicación utiliza un dataset con información histórica sobre el consumo y la generación de energía a nivel mundial:

- Archivo: "World Energy Consumption.csv"
- Contenido: Datos de consumo y generación de energía por país y por tipo de fuente
- Período: Aproximadamente desde 1985 hasta la actualidad

## Configuración de Modelos

Los modelos Prophet están preconfigurados con hiperparámetros optimizados para cada región:
- `changepoint_prior_scale`: Controla la flexibilidad de la tendencia
- `seasonality_prior_scale`: Ajusta la magnitud de las estacionalidades
- `seasonality_mode`: Define si la estacionalidad es aditiva o multiplicativa
- `train_size`: Porcentaje de datos utilizados para entrenamiento

## Tipos de Energía Analizados

- Nuclear (`nuclear_electricity`)
- Petróleo (`oil_electricity`)
- Solar (`solar_electricity`)
- Eólica (`wind_electricity`)
- Gas (`gas_electricity`)
- Hidroeléctrica (`hydro_electricity`)
- Biocombustibles (`biofuel_electricity`)
- Carbón (`coal_electricity`)

## Regiones Disponibles

La aplicación incluye análisis para:
- Continentes (África, Asia, Europa, Norte América, Oceanía, Sud América)
- Países específicos (Estados Unidos, China, Japón, Rusia, India, entre otros)
- Nivel mundial (agregación de todos los datos)

## Ejecución

Para ejecutar la aplicación, navega hasta el directorio donde se encuentra el script y ejecuta:

```bash
streamlit run nombre_del_script.py
```

## Notas Adicionales

- La aplicación requiere conexión a Internet para cargar mapas y componentes de Plotly
- Se recomienda un navegador moderno para una visualización óptima
- La información se actualiza según los datos disponibles en el archivo CSV
- Los modelos predictivos están optimizados con parámetros específicos para cada región

## Personalización

Si deseas personalizar la aplicación:

1. Puedes modificar los colores y estilos de los gráficos en las funciones de visualización
2. Ajustar los parámetros de los modelos Prophet en el diccionario `PROPHET_PARAMS`
3. Añadir nuevas regiones o tipos de energía modificando los diccionarios correspondientes
