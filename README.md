# Trabajo de Fin de Grado: Clasificación de EEG reales basada en técnicas cuánticas de Reservoir Computing.

Este repositorio contiene el desarrollo de modelos clásicos y cuánticos de **Reservoir Computing (RC)** aplicados a señales sintéticas y reales de EEG. La estructura del proyecto se divide en diferentes etapas de experimentación y complejidad.

## 📁 Estructura del repositorio

- **`senyal_simple/`**  
  Contiene el notebook `RC_senyal_simple.ipynb`, con las primeras versiones del RC clásico y cuántico aplicadas a la señal sintética unidimensional de **dos frecuencias**.

- **`senyal_compleja/`**  
  Contiene `RC_senyal_compleja.ipynb`, donde se extienden los modelos RC a la señal sintética unidimensional de **cuatro frecuencias**.

- **`senyales_simuladas/`**  
  Incluye el notebook `RC_senyales_simuladas.ipynb`, con:
  - RC clásico y cuántico
  - Distintos modelos de clasificación (clasificador simple, pruebas con más modelos y curvas de aprendizaje)
  - El fichero `synthetic_data_auto.npy` contiene las señales sintéticas utilizadas.

- **`senyales_reales/`**  
  Notebook `RC_senyales_reales.ipynb`, con la aplicación de RC clásico y cuántico a señales reales de EEG.  
  ⚠️ **Por motivos de privacidad, las señales EEG no se encuentran subidas**, pero sí se pueden consultar los resultados y modelos de clasificación obtenidos.

- **`YolandaOHBM2024-poster1682_DEFINITIVO.pdf`**  
  Póster utilizado como referencia inicial para el desarrollo del proyecto, que sirvió como punto de partida para estructurar la implementación del RC.
