#Recordatorio Clase 06 se hace la primer pre entrega.

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


Parte 6.2

# ============================================================
# 6.2 MEDIDAS DE RESUMEN Y DISTRIBUCIONES
# ============================================================

# 👉 Medidas Cuantitativas
# ------------------------------------------------------------
# MEDIA:
# - Promedio de todos los valores.
# - Fórmula: Media = (Σ xi) / n
# - Representa el valor típico, pero es sensible a outliers.
#
# MEDIANA:
# - Valor central de los datos ordenados.
# - Robusta frente a valores atípicos.
#
# MODA:
# - Valor que más se repite.
# - Puede ser unimodal (1), bimodal (2) o multimodal (+2).
# - Útil especialmente en datos categóricos.
#
# VARIANZA:
# - Mide la dispersión de los datos respecto a la media.
# - Fórmula: Var = Σ(xi - media)^2 / (n-1)
#
# DESVIACIÓN ESTÁNDAR:
# - Raíz cuadrada de la varianza.
# - Se expresa en las mismas unidades que los datos.
# - Indica cuánto se alejan, en promedio, los datos de la media.
#
# CUARTILES Y PERCENTILES:
# - Dividen los datos en partes iguales.
# - Q1: 25% inferior, Q2: mediana, Q3: 75% inferior.
# - Percentiles: dividen en 100 partes.
# - Útiles para ver posiciones relativas (ej: percentil 90).

# 👉 Medidas Cualitativas
# ------------------------------------------------------------
# - Variables categóricas (ej: género, estado civil, color).
# - Se resumen con:
#   -> Conteo de observaciones (frecuencias).
#   -> Moda (categoría más común).
#
# Ejemplo:
#   Género: {Hombres: 60, Mujeres: 40}  -> Conteo
#   Moda = "Hombres"  -> Categoría más frecuente

# 👉 Distribuciones de Variables
# ------------------------------------------------------------
# DISTRIBUCIÓN UNIFORME:
# - Todos los valores tienen la misma probabilidad.
# - Ejemplo: lanzar un dado justo.
# - Gráfico: barras del histograma con alturas similares.
#
# DISTRIBUCIÓN NORMAL (campana de Gauss):
# - Simétrica respecto a la media.
# - La mayoría de datos se concentra en torno a la media.
# - Media = Mediana = Moda.
# - Ejemplo: altura de personas.
#
# DIFERENCIAS:
# - Uniforme = todos los resultados igual de probables.
# - Normal = los valores centrales son más probables.

# 👉 Histogramas
# ------------------------------------------------------------
# - Muestran cómo se distribuyen los datos.
# - Uniforme: barras todas del mismo nivel.
# - Normal: forma de campana (barras altas en el centro).
#
# EJEMPLO VISUAL:
# - Uniforme: █ █ █ █ █
# - Normal:     ▂▄█▄▂

# 👉 Correlación Lineal y Coeficiente de Correlación
# ------------------------------------------------------------
# - Relación entre dos variables (fuerza y dirección).
# - Coeficiente r ∈ [-1, 1]:
#   r ≈ 1  -> correlación positiva fuerte
#   r ≈ -1 -> correlación negativa fuerte
#   r ≈ 0  -> no hay relación lineal
#
# IMPORTANTE: Correlación ≠ causalidad.
# - Que dos variables estén relacionadas no significa
#   que una cause a la otra.
#
# Herramienta clave: diagrama de dispersión (scatter plot).
# - Permite visualizar la relación entre dos variables.
#
# Ejemplo:
# - r = 0.9 -> a mayor estudio, mayor nota (fuerte positiva).
# - r = -0.8 -> a más horas de TV, menor nota (fuerte negativa).
# - r ≈ 0.0 -> variables independientes.

Punto 6.3

# ============================================
# 📌 6.3 Introducción al Preprocesamiento
# ============================================
# El preprocesamiento de datos es una etapa FUNDAMENTAL en cualquier proyecto de Ciencia de Datos.
# Consiste en transformar datos crudos (raw data) en datos limpios y listos para el análisis.
# 
# 🚨 Problema: Los datos reales suelen venir incompletos, ruidosos, inconsistentes.
# ✅ Solución: Aplicar técnicas de preprocesamiento para mejorar su calidad y obtener mejores modelos.
#
# Importancia:
# - Mejora la calidad de los datos.
# - Aumenta el rendimiento de los algoritmos de Machine Learning.
# - Evita errores y conclusiones incorrectas.
#
# Tareas comunes:
# 1. Limpieza de datos
# 2. Integración de datos
# 3. Transformación de datos
# 4. Reducción de datos
# 5. Detección y tratamiento de outliers

# ============================================
# 🔹 EJEMPLOS PRÁCTICOS CON PYTHON
# ============================================

import pandas as pd
import numpy as np

# Creamos un DataFrame de ejemplo
data = {
    "Nombre": ["Ana", "Luis", "Pedro", "María", None],
    "Edad": [23, np.nan, 35, 29, 40],
    "Ciudad": ["Madrid", "Madrid", "Barcelona", "Madrid", "Barcelona"],
    "Salario": [30000, 40000, None, 50000, 1000000]  # <--- hay un valor outlier
}

df = pd.DataFrame(data)

print("📊 Datos originales:")
print(df)

# ============================================
# 1. Limpieza de Datos (Data Cleaning)
# ============================================
# - Manejo de valores faltantes
# - Corrección/eliminación de outliers
# - Reducción del ruido en los datos

# Rellenar valores faltantes en Edad con la media
df["Edad"].fillna(df["Edad"].mean(), inplace=True)

# Rellenar valores faltantes en Salario con la mediana
df["Salario"].fillna(df["Salario"].median(), inplace=True)

# Eliminar filas con Nombre nulo
df.dropna(subset=["Nombre"], inplace=True)

print("\n✅ Datos después de limpieza:")
print(df)

# ============================================
# 2. Integración de Datos (Data Integration)
# ============================================
# - Combinar datos de distintas fuentes.
# (Ejemplo simple con un dataset adicional de bonus)

bonus = pd.DataFrame({
    "Nombre": ["Ana", "Luis", "Pedro", "María"],
    "Bonus": [1000, 1500, 1200, 2000]
})

df = pd.merge(df, bonus, on="Nombre", how="left")

print("\n🔗 Datos después de integración:")
print(df)

# ============================================
# 3. Transformación de Datos (Data Transformation)
# ============================================
# - Normalización / Escalado
# - Codificación de variables categóricas

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Normalizar Salario en rango [0,1]
scaler = MinMaxScaler()
df["Salario_Normalizado"] = scaler.fit_transform(df[["Salario"]])

# Codificar variable categórica Ciudad
encoder = LabelEncoder()
df["Ciudad_Codificada"] = encoder.fit_transform(df["Ciudad"])

print("\n🔄 Datos después de transformación:")
print(df)

# ============================================
# 4. Reducción de Datos (Data Reduction)
# ============================================
# - Selección de características
# - Reducción de dimensionalidad (ej. PCA)
# - Muestreo de datos

# Ejemplo: Nos quedamos solo con las variables más relevantes
df_reducido = df[["Edad", "Salario_Normalizado", "Ciudad_Codificada"]]

print("\n📉 Datos reducidos:")
print(df_reducido)

# ============================================
# 5. Detección y Tratamiento de Outliers
# ============================================
# - Identificar valores atípicos que distorsionan el análisis

Q1 = df["Salario"].quantile(0.25)
Q3 = df["Salario"].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df["Salario"] < (Q1 - 1.5 * IQR)) | (df["Salario"] > (Q3 + 1.5 * IQR))]

print("\n🚨 Outliers detectados:")
print(outliers)

# ============================================
# 📌 Conclusión:
# ============================================
# El preprocesamiento es clave en Data Science porque:
# - Mejora la calidad de los datos.
# - Prepara los datos para modelos de ML.
# - Permite descubrir patrones más claros.
# 
# Sin un buen preprocesamiento, el análisis o modelado puede ser incorrecto.


# ============================================
# 📌 6.4 Detección y Manejo de Outliers
# ============================================
# Los OUTLIERS (valores atípicos) son observaciones que están muy alejadas
# del resto de los datos en un conjunto.
#
# 🚨 Problema: Pueden distorsionar análisis estadísticos y modelos predictivos.
# ✅ Solución: Detectarlos y tratarlos de forma adecuada según el contexto.
#
# --------------------------------------------
# 🔎 Identificación con Diagramas de Caja y Bigotes (Boxplots)
# --------------------------------------------
# - Caja (Box): representa el rango intercuartil (IQR = Q3 - Q1).
# - Bigotes (Whiskers): se extienden desde Q1 y Q3 hasta 1.5 * IQR.
# - Outliers: cualquier punto fuera de los bigotes.
#
# --------------------------------------------
# 🎯 Opciones de Tratamiento de Outliers
# --------------------------------------------
# 1. Eliminación → si son errores de medición o ingreso de datos.
# 2. Transformación → suavizar su efecto (ej. logaritmo).
# 3. Imputación → reemplazarlos con valores como cuartiles o medianas.
# 4. Análisis de sensibilidad → comparar resultados con y sin outliers.
#
# --------------------------------------------
# 📌 Conclusión:
# Detectar y manejar outliers es crucial para obtener modelos más precisos.
# La decisión de cómo tratarlos depende del contexto y de su origen.

# ============================================
# 🔹 EJEMPLO PRÁCTICO CON PYTHON
# ============================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Generamos un dataset con outliers
np.random.seed(42)
data = np.random.normal(50, 10, 100)  # distribución normal
data = np.append(data, [150, 200, -50])  # agregamos valores extremos (outliers)

df = pd.DataFrame({"Valor": data})

print("📊 Datos originales (con outliers):")
print(df.describe())

# ============================================
# 1. Identificación de Outliers con Boxplot
# ============================================
plt.figure(figsize=(6,4))
sns.boxplot(x=df["Valor"])
plt.title("Diagrama de Caja y Bigotes - Outliers visibles")
plt.show()

# ============================================
# 2. Detección numérica de Outliers con IQR
# ============================================
Q1 = df["Valor"].quantile(0.25)
Q3 = df["Valor"].quantile(0.75)
IQR = Q3 - Q1

# Definimos los límites
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers = df[(df["Valor"] < limite_inferior) | (df["Valor"] > limite_superior)]

print("\n🚨 Outliers detectados:")
print(outliers)

# ============================================
# 3. Opciones de Tratamiento
# ============================================

# 🔹 Opción A: Eliminación de outliers
df_sin_outliers = df[(df["Valor"] >= limite_inferior) & (df["Valor"] <= limite_superior)]
print("\n✅ Datos sin outliers (eliminación):")
print(df_sin_outliers.describe())

# 🔹 Opción B: Transformación (logarítmica)
# (Solo funciona con valores positivos)
df["Log_Valor"] = np.log(df["Valor"] + abs(df["Valor"].min()) + 1)
print("\n🔄 Datos transformados con logaritmo:")
print(df[["Valor", "Log_Valor"]].head())

# 🔹 Opción C: Imputación (reemplazar por mediana)
mediana = df["Valor"].median()
df_imputado = df.copy()
df_imputado.loc[df_imputado["Valor"] > limite_superior, "Valor"] = mediana
df_imputado.loc[df_imputado["Valor"] < limite_inferior, "Valor"] = mediana

print("\n♻️ Datos con imputación (reemplazo por mediana):")
print(df_imputado.describe())

# ============================================
# 📌 Conclusión:
# - Detectamos outliers con boxplots y con IQR.
# - Mostramos varias formas de tratarlos (eliminación, transformación, imputación).
# - La elección depende del contexto y de la naturaleza de los datos.

"""
6.5 Análisis de Componentes Principales (PCA)

📌 El PCA (Principal Component Analysis) es una técnica estadística utilizada en Data Science
para reducir la dimensionalidad de un conjunto de datos mientras se conserva la mayor parte
de la variabilidad (información).

👉 ¿Por qué usamos PCA?
- Muchos datasets tienen variables altamente correlacionadas → redundancia.
- Más variables = modelos más complejos y lentos.
- Con PCA reducimos variables → hacemos los modelos más simples, rápidos y generalizables.
"""

# =====================================================
# 1. Importamos librerías necesarias
# =====================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =====================================================
# 2. Creamos un dataset de ejemplo
# =====================================================
# Para entender PCA, generamos datos artificiales con correlaciones
np.random.seed(42)
X1 = np.random.normal(5, 2, 100)      # Variable 1
X2 = X1 * 0.8 + np.random.normal(0,1,100)   # Variable 2 correlacionada con X1
X3 = np.random.normal(10, 5, 100)     # Variable 3 independiente

df = pd.DataFrame({
    "Var1": X1,
    "Var2": X2,
    "Var3": X3
})

print("\n=== Dataset original ===")
print(df.head())

# =====================================================
# 3. Estandarización de los datos
# =====================================================
"""
📌 IMPORTANTE:
PCA es sensible a la escala de las variables. 
Antes de aplicar PCA, se recomienda escalar los datos 
(media = 0, desviación estándar = 1).
"""
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

print("\n=== Datos estandarizados (primeras filas) ===")
print(scaled_data[:5])

# =====================================================
# 4. Aplicación de PCA
# =====================================================
"""
📌 Proceso de PCA:
1. Calcular matriz de covarianza.
2. Obtener valores propios y vectores propios.
3. Seleccionar componentes principales (PCs).
4. Proyectar los datos en estas PCs.

Con sklearn esto se simplifica enormemente.
"""
pca = PCA(n_components=2)   # Queremos reducir a 2 dimensiones
pca_result = pca.fit_transform(scaled_data)

# Convertimos resultado en DataFrame para analizar
df_pca = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"])
print("\n=== Dataset transformado con PCA ===")
print(df_pca.head())

# =====================================================
# 5. Varianza explicada por cada componente
# =====================================================
"""
📌 Interpretación:
- Cada componente explica un % de la varianza total.
- PC1 explica la mayor parte.
- PC2 explica la siguiente parte, y así sucesivamente.
"""
print("\n=== Varianza explicada por cada componente ===")
print(pca.explained_variance_ratio_)

# Visualizamos en gráfico
plt.figure(figsize=(6,4))
plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, alpha=0.7)
plt.xlabel("Componentes Principales")
plt.ylabel("Varianza Explicada")
plt.title("Varianza explicada por PCA")
plt.show()

# =====================================================
# 6. Visualización de los datos en el nuevo espacio PCA
# =====================================================
"""
📌 Ahora los datos están representados en un espacio de menor dimensión,
pero reteniendo la mayor parte de la información original.
"""
plt.figure(figsize=(6,6))
sns.scatterplot(x="PC1", y="PC2", data=df_pca, s=60, color="purple")
plt.title("Proyección de los datos en los 2 primeros Componentes Principales")
plt.show()

# =====================================================
# 7. Conclusión
# =====================================================
"""
✅ PCA es útil para:
- Reducir dimensionalidad.
- Eliminar redundancia de variables.
- Simplificar modelos.
- Mitigar riesgo de overfitting.
- Mejorar interpretabilidad.

En este ejemplo:
- Pasamos de 3 variables a 2 componentes principales.
- Aun así, mantenemos gran parte de la varianza (información).

Esto mismo se puede aplicar en datasets de alta dimensión
(imágenes, genética, sensores, finanzas, etc).
"""
"""
6.6 Ejemplos Prácticos

📌 En esta sección veremos ejemplos de:
1. Estadística descriptiva aplicada a datos de ingresos.
2. Preprocesamiento de datos (limpieza, transformación, integración, reducción).

👉 Importante:
La estadística descriptiva permite entender las características de un dataset
(medidas de tendencia central, dispersión, distribución).
El preprocesamiento asegura que los datos estén listos para aplicar modelos predictivos.
"""

# =====================================================
# 1. Importamos librerías necesarias
# =====================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

# =====================================================
# 2. Ejemplo de Estadística Descriptiva
# =====================================================
"""
🎯 Contexto:
Tenemos un conjunto de datos con información de ingresos anuales de una población.
Queremos entender su distribución con estadísticas descriptivas.
"""

# Generamos un dataset simulado de ingresos
np.random.seed(42)
ingresos = np.random.normal(50000, 15000, 200)  # media=50k, std=15k
ingresos = np.append(ingresos, [120000, 150000])  # agregamos outliers

df_ingresos = pd.DataFrame({"Ingreso_Anual": ingresos})

print("\n=== Primeras filas de los ingresos ===")
print(df_ingresos.head())

# Medidas de tendencia central
media = df_ingresos["Ingreso_Anual"].mean()
mediana = df_ingresos["Ingreso_Anual"].median()
moda = df_ingresos["Ingreso_Anual"].mode()[0]

print("\n=== Medidas de Tendencia Central ===")
print(f"Media: {media:.2f}, Mediana: {mediana:.2f}, Moda: {moda:.2f}")

# Medidas de dispersión
desviacion = df_ingresos["Ingreso_Anual"].std()
q1 = df_ingresos["Ingreso_Anual"].quantile(0.25)
q3 = df_ingresos["Ingreso_Anual"].quantile(0.75)
iqr = q3 - q1

print("\n=== Medidas de Dispersión ===")
print(f"Desviación Estándar: {desviacion:.2f}")
print(f"IQR (Rango Intercuartílico): {iqr:.2f}")

# Visualización: histograma y boxplot
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
sns.histplot(df_ingresos["Ingreso_Anual"], kde=True, bins=30, color="skyblue")
plt.axvline(media, color="red", linestyle="--", label="Media")
plt.axvline(mediana, color="green", linestyle="--", label="Mediana")
plt.legend()
plt.title("Distribución de Ingresos Anuales")

plt.subplot(1,2,2)
sns.boxplot(x=df_ingresos["Ingreso_Anual"], color="lightcoral")
plt.title("Boxplot de Ingresos Anuales")

plt.show()

# =====================================================
# 3. Ejemplo de Preprocesamiento de Datos
# =====================================================
"""
🎯 Contexto:
Dataset de clientes de un banco con:
- Edad
- Saldo de cuenta (con valores faltantes y outliers)
- Historial crediticio (texto)
- Tipo de cuenta

Queremos limpiar, transformar y preparar los datos.
"""

# Creamos dataset simulado
df_clientes = pd.DataFrame({
    "Edad": [25, 40, 35, 50, 29, np.nan, 60],
    "Saldo": [1000, 2000, -500, 999999, 3000, np.nan, 2500],
    "Historial_Credito": ["Bueno", "Malo", "Bueno", "Malo", "Bueno", "Regular", "Malo"],
    "Tipo_Cuenta": ["Ahorro", "Corriente", "Ahorro", "Ahorro", "Corriente", "Corriente", "Ahorro"]
})

print("\n=== Dataset original de clientes ===")
print(df_clientes)

# --- Limpieza de datos ---
# 1. Rellenamos valores faltantes de Edad con la media
df_clientes["Edad"].fillna(df_clientes["Edad"].mean(), inplace=True)

# 2. Rellenamos valores faltantes de Saldo con la mediana
df_clientes["Saldo"].fillna(df_clientes["Saldo"].median(), inplace=True)

# 3. Eliminamos outliers extremos en Saldo
q1 = df_clientes["Saldo"].quantile(0.25)
q3 = df_clientes["Saldo"].quantile(0.75)
iqr = q3 - q1
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr
df_clientes = df_clientes[(df_clientes["Saldo"] >= limite_inferior) & (df_clientes["Saldo"] <= limite_superior)]

print("\n=== Dataset después de limpieza ===")
print(df_clientes)

# --- Transformación de datos ---
"""
📌 One Hot Encoding:
Convertimos variables categóricas (texto) en binarias (0/1).
"""
encoder = OneHotEncoder(sparse_output=False)
encoded_vars = encoder.fit_transform(df_clientes[["Historial_Credito", "Tipo_Cuenta"]])

encoded_df = pd.DataFrame(encoded_vars, columns=encoder.get_feature_names_out())
df_final = pd.concat([df_clientes.reset_index(drop=True), encoded_df], axis=1)

print("\n=== Dataset transformado con One Hot Encoding ===")
print(df_final)

# --- Reducción de dimensionalidad con PCA ---
"""
📌 Aplicamos PCA en las variables numéricas (Edad, Saldo, variables binarias).
"""
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_final.drop(columns=["Historial_Credito", "Tipo_Cuenta"]))

df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2"])

print("\n=== Dataset reducido con PCA ===")
print(df_pca)

# Visualización de clientes en espacio PCA
plt.figure(figsize=(6,6))
sns.scatterplot(x="PC1", y="PC2", data=df_pca, s=80, color="purple")
plt.title("Clientes proyectados en 2 Componentes Principales (PCA)")
plt.show()

# =====================================================
# 4. Conclusión
# =====================================================
"""
✅ Estadística descriptiva → nos permitió entender distribución de ingresos, identificar outliers y medir dispersión.
✅ Preprocesamiento → limpiamos valores faltantes y outliers, transformamos variables categóricas, y reducimos dimensionalidad con PCA.

Esto refleja el flujo real de trabajo en Ciencia de Datos:
- Explorar
- Limpiar
- Transformar
- Preparar para modelar
"""


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


