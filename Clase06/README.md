# ============================================================
# REPASO CLASE 05CLASE DE VISUALIZACIÓN AVANZADA CON SEABORN Y MATPLOTLIB
# ============================================================
# Tema: Herramientas avanzadas de visualización
#
# Objetivo de la clase:
# - Entender qué es Seaborn y por qué usarlo encima de Matplotlib.
# - Diferenciar subplots clásicos de Matplotlib vs. FacetGrid de Seaborn.
# - Practicar con ejemplos reales de gráficos.
#
# ------------------------------------------------------------
# TEORÍA:
# ------------------------------------------------------------
# ¿Qué es Seaborn?
# - Es una librería de Python construida sobre Matplotlib.
# - Se integra directamente con Pandas, lo que facilita graficar DataFrames.
# - Permite hacer gráficos estadísticos de forma rápida, estética y sencilla.
#
# Ventajas principales:
# 1. Sintaxis más simple que Matplotlib.
# 2. Temas visuales predefinidos (mejor estética por defecto).
# 3. Funciona directamente con DataFrames y columnas.
# 4. Tiene funciones para gráficos complejos en pocas líneas.
#
# ------------------------------------------------------------
# Diferencia entre funciones "Axes-level" y "Figure-level":
# - Axes-level (ej: sns.scatterplot, sns.boxplot):
#   -> Se dibujan en un objeto de ejes específico (matplotlib.pyplot.Axes).
#   -> Útiles cuando quiero controlar UN gráfico dentro de subplots.
#
# - Figure-level (ej: sns.relplot, sns.catplot, sns.FacetGrid):
#   -> Controlan toda la figura completa, creando automáticamente subplots.
#   -> Útiles para comparar subgrupos o múltiples distribuciones de datos.
#
# ------------------------------------------------------------
# EJEMPLO 1: SUBPLOTS CLÁSICOS CON MATPLOTLIB + SEABORN
# ------------------------------------------------------------
# Idea: crear una figura con 4 gráficos diferentes para mostrar:
# - Histograma
# - Boxplot
# - Gráfico de dispersión
# - Gráfico de líneas
#
# Objetivo: practicar cómo ubicar diferentes gráficos en una grilla
# (2 filas x 2 columnas) y cómo elegir qué gráfico va en cada lugar.
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Generamos datos de ejemplo usando NumPy
# np.random.seed(42) -> fijamos semilla para reproducibilidad (los datos serán siempre iguales)
np.random.seed(42)
df = pd.DataFrame({
    "variable1": np.random.normal(0, 1, 100),   # distribución normal media=0, sd=1
    "variable2": np.random.exponential(1, 100), # distribución exponencial
    "variable3": np.random.uniform(0, 10, 100), # distribución uniforme
    "variable4": np.random.normal(5, 2, 100),   # normal con media=5, sd=2
    "variable5": range(100),                    # valores consecutivos 0 a 99
    "variable6": np.random.normal(50, 10, 100)  # normal con media=50, sd=10
})

# Creamos la figura con 2 filas y 2 columnas de subplots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Subplot 1: Histograma de variable1
# Histograma = distribución de frecuencias
sns.histplot(data=df, x='variable1', ax=axs[0, 0], color="skyblue")
axs[0, 0].set_title("Histograma de variable1")

# Subplot 2: Boxplot de variable2
# Boxplot = detecta la dispersión de los datos y posibles outliers
sns.boxplot(data=df, x='variable2', ax=axs[0, 1], color="lightgreen")
axs[0, 1].set_title("Boxplot de variable2")

# Subplot 3: Gráfico de dispersión (scatterplot)
# Muestra relación entre dos variables
sns.scatterplot(data=df, x='variable3', y='variable4', ax=axs[1, 0], color="salmon")
axs[1, 0].set_title("Dispersión variable3 vs variable4")

# Subplot 4: Gráfico de líneas
# Útil para series temporales o valores consecutivos
sns.lineplot(data=df, x='variable5', y='variable6', ax=axs[1, 1], color="purple")
axs[1, 1].set_title("Línea variable5 vs variable6")

# Ajustamos para que no se encimen títulos y gráficos
plt.tight_layout()
plt.show()

# ============================================================
# EJEMPLO 2: FACETGRID EN SEABORN
# ============================================================
# Objetivo:
# - Mostrar cómo dividir automáticamente un dataset en múltiples gráficos
#   según categorías.
# - Usaremos el dataset "tips" de Seaborn (propinas en un restaurante).
#
# Casos de uso típicos:
# - Comparar distribuciones entre grupos (ej: hombres vs mujeres).
# - Ver cómo cambia una variable según otra categoría (ej: Lunch vs Dinner).
# ============================================================

# Cargamos dataset "tips"
df_tips = sns.load_dataset("tips")

# Mostramos primeras filas para conocer estructura
print(df_tips.head())

# Creamos un FacetGrid
# col="sex"  -> divide columnas por sexo
# row="time" -> divide filas por tiempo (Lunch/Dinner)
g = sns.FacetGrid(df_tips, col="sex", row="time", margin_titles=True)

# Mapear gráfico: en cada subplot mostrar histograma de "total_bill"
g.map(sns.histplot, "total_bill", bins=15, color="skyblue")

# Agregar leyenda para categorías
g.add_legend()

# Mostrar figura
plt.show()

# ------------------------------------------------------------
# CONCLUSIONES DE LA CLASE:
# - Matplotlib permite un control detallado de gráficos y subplots.
# - Seaborn simplifica la sintaxis y mejora la estética.
# - Subplots clásicos: control manual de cada gráfico.
# - FacetGrid: automatización para comparar categorías.
# ------------------------------------------------------------


#inicio CLase 06
# Tipos de variables en un analisis del tipo EDA
[DIAPOSITIVAS](https://docs.google.com/presentation/d/1bTgblneO_G2WteTeku40AN-3yKmhiVvpcudqcGo0cco/edit?slide=id.p2#slide=id.p2)


#Teoria 6.1

# ============================================================
# 6.1 FUNDAMENTOS DE ESTADÍSTICA
# ============================================================

# 👉 Estadística Descriptiva: Concepto y Relevancia
# - Rama de la estadística que recopila, organiza y resume datos.
# - Su objetivo principal es ofrecer una visión general y comprensible.
# - Permite identificar patrones, tendencias y relaciones.
# - Herramienta clave para la toma de decisiones informadas.
# - Usa medidas de resumen como:
#   -> Media (promedio)
#   -> Mediana (valor central)
#   -> Moda (valor más frecuente)
#   -> Varianza y Desviación Estándar (medidas de dispersión).

# 👉 Importancia práctica:
# Nos ayuda a responder preguntas como:
# - ¿Cuál es el valor típico de los datos?
# - ¿Qué tan dispersos o concentrados están?
# - ¿Existen valores atípicos (outliers)?

# ------------------------------------------------------------
# Introducción al Análisis Exploratorio de Datos (EDA)
# ------------------------------------------------------------
# - Fase inicial del análisis de datos.
# - Objetivo: descubrir patrones, detectar anomalías,
#   probar hipótesis y verificar supuestos.
# - Se realiza combinando:
#   -> Estadística descriptiva
#   -> Visualización de datos
#
# Filosofía del EDA:
# - Acercarse a los datos sin prejuicios.
# - Explorar abiertamente para encontrar características inesperadas.
#
# Beneficios del EDA:
# - Entender la estructura de los datos.
# - Preparar datos para modelos más complejos.
# - Tomar mejores decisiones sobre limpieza y preprocesamiento.

# ------------------------------------------------------------
# RESUMEN PARA LA CLASE:
# ------------------------------------------------------------
# - La estadística descriptiva resume y organiza los datos.
# - El EDA es el paso inicial para explorar datos a fondo.
# - Juntos permiten comprender los datos antes de aplicar técnicas
#   de predicción o machine learning.
#
# EJEMPLOS EN CLASE:
# - Usar Python para calcular media, mediana y moda de un dataset.
# - Hacer histogramas y boxplots para visualizar distribuciones.



### 🧠 1. **Según su naturaleza:**

#### a. **Cuantitativas (Numéricas)**

* Representan cantidades o valores numéricos.
* Se subdividen en:

##### 🔹 *Continuas*

* Pueden tomar **infinitos valores dentro de un rango**.
* Se miden, no se cuentan.
* **Ejemplos**:

  * Altura (1.75 m, 1.76 m, etc.)
  * Peso (65.5 kg, 66.2 kg)
  * Tiempo (2.5 horas)

##### 🔹 *Discretas*

* Solo pueden tomar **valores enteros o finitos**.
* Se cuentan.
* **Ejemplos**:

  * Número de hijos (0, 1, 2…)
  * Número de autos en una familia
  * Cantidad de errores en una prueba

---

#### b. **Cualitativas (Categoricas)**

* Representan **categorías o cualidades**.
* No implican operaciones matemáticas.

##### 🔹 *Nominales*

* No tienen orden lógico.
* **Ejemplos**:

  * Color de ojos (azul, verde, marrón)
  * Género (masculino, femenino, otro)
  * Nacionalidad (argentina, chilena)

##### 🔹 *Ordinales*

* Tienen **orden o jerarquía**, pero no hay distancia precisa entre categorías.
* **Ejemplos**:

  * Nivel educativo (primario, secundario, universitario)
  * Grado de satisfacción (bajo, medio, alto)
  * Rango militar (soldado, cabo, sargento)

---

### 🔁 2. **Según su variabilidad:**

#### 🔹 *Dependientes*

* Son afectadas por otras variables.
* **Ejemplo**: La presión arterial depende del nivel de actividad física.

#### 🔹 *Independientes*

* Se manipulan para ver su efecto.
* **Ejemplo**: Horas de estudio como variable que afecta al rendimiento académico.

---

### 📊 3. **Otras clasificaciones útiles:**

#### 🔹 *Binarias o dicotómicas*

* Solo tienen **dos categorías posibles**.
* **Ejemplo**: Sí / No, Aprobado / Reprobado

#### 🔹 *Polinómicas*

* Tienen más de dos categorías.
* **Ejemplo**: Tipo de sangre (A, B, AB, O)
---
Las **series temporales** son un tipo especial de conjunto de datos en el que **las observaciones están ordenadas en el tiempo**. Se utilizan para analizar y predecir cómo cambian los valores de una variable a lo largo del tiempo.

---


## Consideracion de otras variables sobre Temporalidades

### 🕒 ¿Qué es una serie temporal?

Una **serie temporal** es una **secuencia de datos** registrados a lo largo del tiempo, **en intervalos regulares** (diarios, mensuales, anuales, etc.).

* **Ejemplos**:

  * Temperatura diaria de una ciudad
  * Precio del dólar por mes
  * Ventas de una empresa por trimestre
  * Cantidad de pasajeros por hora en un subte

---

### 🧩 Cualidades distintivas de las series temporales:

1. ### 📐 **Secuencialidad**

   * El orden cronológico **es fundamental**.
   * A diferencia de otros conjuntos de datos, **reordenar los datos rompe la estructura** de la serie.

2. ### 🔁 **Dependencia temporal**

   * Los valores sucesivos están **relacionados entre sí**.
   * El valor en el tiempo `t` suele depender del valor en `t-1`, `t-2`, etc.
   * Esto permite usar modelos que capturan esa dependencia, como **ARIMA, LSTM, Prophet**, etc.

3. ### ⏳ **Estructura del tiempo**

   * El tiempo tiene **características particulares** que afectan los datos:

     * **Estacionalidad**: patrones que se repiten en intervalos regulares (ej. más helado en verano).
     * **Tendencia**: crecimiento o disminución a lo largo del tiempo.
     * **Ciclos**: fluctuaciones más irregulares que pueden durar años (ej. economía).
     * **Eventos especiales**: feriados, años bisiestos, pandemias, etc.
     * **Cambio de estaciones del año**: especialmente relevante en fenómenos naturales o de consumo.

4. ### 📏 **Frecuencia**

   * Puede ser **horaria, diaria, semanal, mensual, trimestral, anual**, etc.
   * La elección de la frecuencia afecta el tipo de análisis (por ejemplo, análisis de estacionalidad mensual vs diaria).

---

### 🎯 Aplicaciones típicas:

* Predicción de ventas o demanda
* Análisis financiero (acciones, divisas)
* Meteorología
* Series biomédicas (ritmo cardíaco, glucosa)
* Sensores IoT (temperatura, vibraciones)

---

### 📘 **Resumen General: Medidas de Resumen y Distribuciones**

Las medidas de resumen se dividen en dos grandes grupos:

1. **Cuantitativas** (valores numéricos):

   * **Media**: Promedio aritmético, sensible a valores extremos.
   * **Mediana**: Valor central, resistente a outliers.
   * **Moda**: Valor más frecuente.
   * **Varianza**: Mide cuán dispersos están los datos respecto a la media.
   * **Desviación estándar**: Dispersión en las mismas unidades que los datos.
   * **Cuartiles y percentiles**: Dividen el conjunto de datos en segmentos para analizar la distribución.

2. **Cualitativas** (valores categóricos):

   * **Conteo de observaciones**: Frecuencia de cada categoría.
   * **Moda**: Categoría más frecuente.

También se analizan **distribuciones**:

* **Distribución uniforme**: Todos los valores tienen igual probabilidad.
* **Distribución normal**: Forma de campana, simétrica, con mayor densidad de datos en torno a la media.

Y se interpretan gráficamente mediante:

* **Histogramas**: Muestran frecuencias por intervalos.
* **Correlación lineal**: Mide fuerza y dirección de la relación entre dos variables (coeficiente de correlación de −1 a 1).

---

### 📊 **Tabla Resumen: Medidas y Distribuciones**

| **Concepto**                | **Tipo de Variable**     | **Descripción**                                                   | **Ventajas**                               | **Limitaciones**                                      |
| --------------------------- | ------------------------ | ----------------------------------------------------------------- | ------------------------------------------ | ----------------------------------------------------- |
| **Media**                   | Cuantitativa             | Promedio de todos los valores                                     | Fácil de calcular e interpretar            | Afectada por valores extremos (outliers)              |
| **Mediana**                 | Cuantitativa             | Valor central de los datos ordenados                              | Robusta frente a outliers                  | No refleja toda la información del conjunto           |
| **Moda**                    | Cualitativa/Cuantitativa | Valor o categoría más frecuente                                   | Intuitiva y útil en cualitativas           | Puede haber más de una o ninguna moda                 |
| **Varianza**                | Cuantitativa             | Promedio de los cuadrados de las desviaciones respecto a la media | Evalúa dispersión con detalle              | Difícil interpretación directa (unidades al cuadrado) |
| **Desviación estándar**     | Cuantitativa             | Raíz cuadrada de la varianza                                      | Más intuitiva que la varianza              | Sensible a valores extremos                           |
| **Cuartiles / Percentiles** | Cuantitativa             | Dividen los datos ordenados en partes iguales                     | Permiten comparar posiciones relativas     | No indican dispersión general completa                |
| **Conteo de observaciones** | Cualitativa              | Número de veces que aparece cada categoría                        | Simple y clara                             | No indica relación entre categorías                   |
| **Distribución uniforme**   | Cualitativa/Cuantitativa | Todos los valores tienen igual probabilidad                       | Ideal para procesos aleatorios controlados | Poco realista en fenómenos naturales                  |
| **Distribución normal**     | Cuantitativa             | Datos concentrados alrededor de la media con forma de campana     | Base de muchos métodos estadísticos        | No todos los fenómenos la siguen                      |
| **Histograma**              | Cualitativa/Cuantitativa | Representa frecuencias por intervalos                             | Visualiza forma de la distribución         | Depende del número de intervalos                      |
| **Correlación lineal**      | Cuantitativa             | Relación entre dos variables (de −1 a 1)                          | Detecta patrones lineales                  | No implica causalidad                                 |

---

### RESPUESTA DE LA FILMINA pag.8 

**índice trimestral de inflación se considera una serie de tiempo**?

### ¿Por qué?

Una **serie de tiempo** es un conjunto de observaciones recogidas **en momentos sucesivos del tiempo**, generalmente a intervalos regulares (diarios, mensuales, trimestrales, etc.). El **índice trimestral de inflación** cumple con estas características:

* **Observaciones ordenadas en el tiempo:** cada dato representa la inflación de un trimestre específico (por ejemplo, Q1 2022, Q2 2022, etc.).
* **Intervalos regulares:** los datos se recopilan **cada tres meses**, lo que constituye un intervalo de tiempo constante.
* **Análisis temporal posible:** se puede analizar su tendencia, estacionalidad, ciclos económicos y variabilidad a lo largo del tiempo.

### En pocas palabras:

✅ **Sí, el índice trimestral de inflación es una serie de tiempo**, y puede ser analizado mediante métodos estadísticos y econométricos propios del análisis de series temporales.
---


