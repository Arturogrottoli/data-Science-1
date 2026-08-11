**Materia:** Data Science I  
**Dataset:** `video_game_data.csv` (subir manualmente a Google Colab)  
**Notebook:** `Proyecto1.ipynb` (abrir en Google Colab)

> Documento generado a partir de todas las celdas del notebook (texto explicativo + código completo), en el mismo orden en que aparecen.

---

## Cómo dar este ejemplo en clase

Este es el **Ejemplo 2** de la Clase 10 (Repaso Final). A diferencia del Ejemplo 1 (que va directo a un problema de clasificación bien resuelto), este notebook tiene un valor pedagógico distinto: **muestra primero un camino que no funciona bien** (regresión sobre `Global_Sales`, con R² cercano a cero) **y después el mismo problema reformulado de una forma que sí funciona** (clasificación binaria de éxito comercial). Esa vuelta es en sí misma la lección más importante del ejemplo — en un proyecto real, el primer enfoque casi nunca es el bueno.

| Lo que se ve acá | De qué clase viene |
|---|---|
| Carga, `.shape`, `.info()`, `.describe()`, tipos de datos | Clase 03 — NumPy y Pandas |
| Duplicados, nulos, imputación (media/mediana/moda), outliers con boxplot | Clase 04 y Clase 06 — Limpieza y EDA |
| Histogramas, boxplots, scatterplots, análisis univariado/bivariado/multivariado | Clase 05 — Visualización |
| `LabelEncoder`, `StandardScaler`, `train_test_split` | Clase 04 y Clase 08 |
| Regresión Lineal, Árbol de Decisión, Random Forest — RMSE, MAE, R² | Clase 08 — Aprendizaje Supervisado (regresión) |
| `Pipeline` + `ColumnTransformer` + `OneHotEncoder` + `GridSearchCV` + `StratifiedKFold` + ROC-AUC | Clase 07 y Clase 08 — Reproducibilidad y clasificación |

**Por qué conviene leerlo en dos mitades:** la primera mitad del notebook (Bloques 1 y 2) resuelve el problema como **regresión** — predecir el número exacto de copias vendidas — y los tres modelos entrenados dan un R² cercano a cero. Eso **no es un error de código**, es una conclusión real: el éxito de un videojuego no se explica linealmente con estas variables. La segunda mitad (Bloque 3) reformula la pregunta como **clasificación binaria** — ¿superó el millón de copias o no? — y ahí sí el modelo tiene algo que decir. Mostrar ese contraste en vivo es más valioso que saltar directo a la parte que funciona.

**Sugerencia de recorrido en vivo** (no hace falta leer las 3 secciones completas palabra por palabra):
1. Arrancar por el "Abstracto" y "Contexto Comercial" — el gancho de negocio (Rockstar/GTA) y la lógica de *blockbuster* de la industria.
2. Mostrar rápido las "Hipótesis de trabajo" — es un buen ejemplo de cómo se arranca un análisis con preguntas concretas, no al revés.
3. Pasar por el EDA (univariado/bivariado/multivariado) mostrando 2 o 3 gráficos, sin detenerse en todos.
4. Detenerse en la sección "Por qué los modelos de regresión tienen R² cercano a cero" — es el momento pedagógico central de todo el ejemplo.
5. Cerrar con la sección de Clasificación Binaria: el cambio de umbral, el `Pipeline` correcto (contrastar con el `LabelEncoder` + `StandardScaler` sueltos de la primera mitad, que sí tienen leakage), y el gráfico de Feature Importances.

### Índice

- [Ficha del proyecto](#análisis-de-ventas-globales-de-videojuegos)
- [Abstracto](#abstracto)
- [Contexto Comercial y Analítico](#contexto-comercial-y-analítico)
- [Preguntas de Investigación e Hipótesis](#preguntas-de-investigación-e-hipótesis)
- [Objetivo](#objetivo)
- [Carga del Dataset](#carga-del-dataset)
- [Data Wrangling](#data-wrangling)
- [EDA (Univariado, Bivariado, Multivariado)](#eda)
- [Preprocesamiento de datos](#preprocesamiento-de-datos)
- [Feature Selection](#feature-selection)
- [Modelado (Regresión)](#modelado)
- [Por qué el R² da cercano a cero](#por-qué-los-modelos-de-regresión-tienen-r²-cercano-a-cero)
- [Conclusiones Finales (Regresión)](#conclusiones-finales)
- [Clasificación Binaria — Éxito Comercial](#clasificación-binaria--éxito-comercial-de-videojuegos)
- [Resumen de requisitos cumplidos](#resumen--requisitos-cumplidos-en-esta-sección)

---

# Análisis de Ventas Globales de Videojuegos
## ¿Qué hace que un videojuego sea un éxito comercial?

| Ficha del proyecto | |
|---|---|
| **Industria** | Entretenimiento digital · Mercado global de videojuegos |
| **Tipo de problema** | EDA + Regresión + Clasificación binaria |
| **Dataset** | 16.719 videojuegos · 16 variables · Años 1980–2020 |
| **Variable objetivo (regresión)** | `Global_Sales` (ventas globales en millones) |
| **Variable objetivo (clasificación)** | `EXITO_COMERCIAL` (1 si supera 1M de unidades) |
| **Herramientas** | Python · pandas · scikit-learn · seaborn · missingno |
| **Entorno** | Google Colab · Archivo `video_game_data.csv` |

## Abstracto

En 2021, el estudio Rockstar Games lanzó una remasterización de Grand Theft Auto que vendió millones de copias a pesar de recibir críticas mediocres. Ese mismo año, juegos muy valorados por la crítica quedaron en el olvido comercial. ¿Qué determina entonces que un videojuego venda?

Este proyecto parte de esa pregunta con un enfoque de datos. Usando un dataset de **16.719 videojuegos** con información de ventas por región, género, plataforma y puntuaciones de críticos y usuarios, exploramos qué variables están realmente asociadas al desempeño comercial de un título.

El trabajo tiene dos instancias:

- **Exploración y regresión:** entender la distribución de las ventas y probar si es posible predecir el número exacto de copias vendidas con los datos disponibles.
- **Clasificación binaria:** reformular el problema como una pregunta más acotada —¿va a vender más de un millón?— y construir un modelo que la responda con la metodología completa de machine learning supervisado.

**Conclusión anticipada:** predecir ventas exactas es difícil. Predecir si un juego va a ser un éxito comercial es más manejable, y el análisis muestra qué variables contribuyen más a esa distinción.

## Contexto Comercial y Analítico

### El mercado

El negocio de los videojuegos opera con una lógica de blockbusters: unos pocos títulos concentran la mayor parte de los ingresos del sector. Nintendo puede publicar 10 juegos en un año y uno solo de ellos —un Mario Kart, un Zelda— generar el 70% de sus ventas. Para el resto de la industria, la mayoría de los lanzamientos recupera apenas la inversión, y muchos ni eso.

Este fenómeno hace que la decisión de en qué plataforma lanzar, qué género elegir y cuánto invertir en producción sea crítica. Los estudios pequeños no tienen margen para equivocarse.

### El dataset

El archivo `video_game_data.csv` reúne datos de ventas físicas de **16.719 videojuegos** publicados entre 1980 y 2020. Las variables disponibles son:

| Grupo | Columnas |
|---|---|
| Identificación del juego | `Name`, `Platform`, `Year_of_Release`, `Genre`, `Publisher`, `Developer` |
| Ventas por región (millones) | `NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`, `Global_Sales` |
| Puntuaciones | `Critic_Score`, `Critic_Count`, `User_Score`, `User_Count` |
| Clasificación de edad | `Rating` (E, T, M, etc.) |

### Qué no captura este dataset

Antes de arrancar conviene ser honestos sobre los límites de los datos:

- Las ventas digitales (Steam, PlayStation Store, App Store) no están incluidas. Para juegos lanzados después de 2015, esto subestima enormemente el desempeño real.
- Muchos juegos no tienen `Critic_Score` porque salieron antes de Metacritic o porque son títulos de nicho que los medios ignoraron.
- Solo aparecen juegos que superaron las 10.000 copias físicas vendidas. El 90% de los juegos publicados no llega a ese número y directamente no existe en este dataset.

## Preguntas de Investigación e Hipótesis

### Lo que queremos entender

**1. ¿El género del juego define su techo de ventas?**
Intuimos que Action y Sports tienen más alcance masivo, pero puede que el éxito promedio sea similar entre géneros y que lo que varía sea la varianza (pocos juegos de RPG, pero los que salen bien venden muchísimo).

**2. ¿Vale la pena la opinión de los críticos?**
La correlación entre Critic_Score y Global_Sales no es obvia. Un juego puede tener 90 puntos en Metacritic y vender 500.000 copias, y otro puede tener 65 puntos y vender 10 millones gracias a su franquicia o campaña de marketing.

**3. ¿Las preferencias de mercado varían por región?**
Japón históricamente consume más RPGs y juegos de Nintendo portátil. Norteamérica compra más shooters y deportes. Europa está en el medio. ¿Los datos confirman estos estereotipos?

**4. ¿Se puede clasificar el éxito comercial?**
Si pasamos de "predecir las ventas exactas" a "predecir si supera el millón", ¿mejora la capacidad del modelo? ¿Qué variables son las más determinantes?

### Hipótesis de trabajo

| Hipótesis | Esperamos que... |
|---|---|
| **H1** | Action y Sports tengan las medianas de ventas más altas por género |
| **H2** | Critic_Score tenga correlación positiva con ventas, pero débil (R < 0.4) |
| **H3** | JP_Sales sea desproporcionadamente alta en plataformas Nintendo vs otras |
| **H4** | Random Forest supere a Regresión Logística en ROC-AUC para clasificar éxito |
| **H5** | Year_of_Release sea relevante: el mercado creció hasta ~2008 y después bajó (ventas físicas) |

## Objetivo

El trabajo está organizado en tres bloques que se construyen uno sobre el otro:

**Bloque 1 — Conocer los datos**
Antes de modelar, explorar. Ver cómo se distribuyen las ventas, qué variables tienen nulos y cómo tratarlos, identificar outliers y entender qué está diciendo cada columna. Un modelo entrenado sobre datos mal entendidos produce resultados que no sirven.

**Bloque 2 — Regresión (predecir ventas)**
Probar si `Global_Sales` puede estimarse a partir de las variables disponibles. Se van a entrenar tres modelos (Regresión Lineal, Árbol de Decisión, Random Forest) y evaluar con RMSE, MAE y R². El objetivo de este bloque no es necesariamente obtener un modelo bueno, sino entender por qué el problema es difícil.

**Bloque 3 — Clasificación binaria (predecir éxito)**
Reformular la pregunta: en lugar de predecir un número, predecir si el juego supera el millón de copias vendidas (`EXITO_COMERCIAL = 1`). Este bloque aplica la metodología completa:

```
Pipeline (preprocesado) → GridSearchCV + StratifiedKFold → ROC-AUC
→ Confusion Matrix → Curva ROC → Feature Importances
```

El resultado de este bloque es el que se evalúa como proyecto final.

## Carga del Dataset

### Librerías y configuración

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuracion visual del notebook
# style='ticks': ejes con marcas, sin grilla de fondo (mas limpio para presentacion)
# palette='colorblind': colores distinguibles por personas con daltonismo
sns.set_theme(style='ticks', palette='colorblind')
plt.rcParams['figure.dpi'] = 100  # resolucion de figuras en pantalla
```

### Lectura del dataset

```python
import pandas as pd

# Subir el archivo manualmente a Colab:
#   1. Panel izquierdo (icono de carpeta)
#   2. Arrastrar el archivo video_game_data.csv
#   3. El archivo queda en /content/ (mismo nivel que sample_data)
#
# Alternativa por codigo (abre selector de archivos del navegador):
#   from google.colab import files
#   files.upload()

df = pd.read_csv("/content/video_game_data.csv")
df.head()
```

### Análisis inicial

```python
# [REQUISITO - EDA] Dimensiones del dataset
df.shape
```

> El dataset utilizado contiene más de 2000 registros y más de 15 variables, cumpliendo con los requisitos del proyecto final.

```python
# shape: (filas, columnas) — primera verificacion de que el archivo cargo bien
print('Dimensiones del dataset:', df.shape)
print()

# info(): tipos de dato de cada columna y cuantos valores no nulos tiene
# Object = texto | float64/int64 = numerico | datetime64 = fecha
df.info()
print()

# describe(): estadisticos para columnas numericas
#   count = filas sin nulos | mean = promedio | std = dispersion
#   25%/50%/75% = cuartiles | min/max = extremos
df.describe()
```

### Descripción de variables

- **Name**: nombre del videojuego
- **Platform**: plataforma de lanzamiento
- **Year_of_Release**: año de lanzamiento
- **Genre**: género del videojuego
- **Publisher**: empresa publicadora
- **NA_Sales**: Ventas en América del Norte (millones).
- **EU_Sales**: Ventas en Europa (millones).
- **JP_Sales**: Ventas en Japón (millones).
- **Other_Sales**: Ventas en otros mercados.
- **Global_Sales**: ventas globales (millones de unidades)
- **Critic_Score**: puntaje otorgado por críticos
- **Critic_Count**: Cantidad de críticas profesionales registradas.
- **User_Score**: puntaje otorgado por usuarios
- **User_Count**: Cantidad de valoraciones de usuarios.
- **Developer**: Empresa desarrolladora del videojuego.
- **Rating**: Clasificación por edades del videojuego.

## Data Wrangling

### Duplicados

```python
df.duplicated().sum()
```

### Valores nulos

```python
# [REQUISITO - EDA] Conteo de valores nulos por columna
df.isna().sum()
```

**Tratamiento de valores nulos**

Se realiza una nueva revisión de valores nulos para poder continuar con la etapa de modelado. En la primera parte del proyecto solo se identificaron, mientras que en esta etapa se procederá a imputarlos para evitar errores al entrenar modelos de machine learning.

```python
# Convertir User Score a número
df["User_Score"] = pd.to_numeric(df["User_Score"], errors="coerce")
```

> Se transformó la variable a formato numérico para poder utilizarla correctamente en los modelos de machine learning.

```python
# Estrategia de imputacion de nulos:
#   Numericas con distribucion normal → media (Critic_Score, User_Score)
#   Numericas con outliers → mediana (Critic_Count, User_Count: pocos juegos con miles de reviews)
#   Year_of_Release → mediana (hay juegos sin fecha bien registrada)
#   Categoricas → moda (valor mas frecuente)
#   Developer con 'Unknown': indica que no se registro el desarrollador

# [REQUISITO - PREPROCESADO] Imputacion de valores nulos
# Relleno de variables numéricas
df["Critic_Score"] = df["Critic_Score"].fillna(df["Critic_Score"].mean())
df["User_Score"] = df["User_Score"].fillna(df["User_Score"].mean())
df["Critic_Count"] = df["Critic_Count"].fillna(df["Critic_Count"].median())
df["User_Count"] = df["User_Count"].fillna(df["User_Count"].median())
df["Year_of_Release"] = df["Year_of_Release"].fillna(df["Year_of_Release"].median())
# Relleno de variables categóricas
df["Rating"] = df["Rating"].fillna(df["Rating"].mode()[0])
df["Developer"] = df["Developer"].fillna("Unknown")
df["Genre"] = df["Genre"].fillna(df["Genre"].mode()[0])
df["Publisher"] = df["Publisher"].fillna(df["Publisher"].mode()[0])
```

> Se reemplazaron los valores faltantes utilizando: media para variables numéricas con distribución aproximadamente normal, mediana para variables con posibles outliers, y moda para variables categóricas. Se evitó el uso del parámetro `inplace` debido a futuras modificaciones en pandas que podrían afectar su funcionamiento.

### Outliers

```python
# Boxplot de ventas
sns.boxplot(y=df["Global_Sales"])
plt.show()
```

> A partir del boxplot se observa la presencia de valores atípicos en las ventas globales. Esto indica que existen videojuegos con ventas considerablemente superiores al promedio, lo cual es esperable en este mercado donde pocos títulos concentran gran parte del éxito comercial.

### Transformaciones

```python
# Convertir año a número
df["Year_of_Release"] = pd.to_numeric(df["Year_of_Release"], errors="coerce")
```

```python
# Eliminar espacios en nombres de columnas
df.columns = df.columns.str.strip()
```

## EDA

### Univariado

```python
# [REQUISITO - EDA - UNIVARIADO] Distribucion de la variable objetivo y conteo por categoria
# Histograma de ventas globales
plt.figure(figsize=(8,4))
sns.histplot(df["Global_Sales"], bins=30, kde=True, color='#8e44ad')
plt.title("Distribución de las ventas globales de videojuegos")
plt.xlabel("Ventas globales (millones)")
plt.ylabel("Frecuencia")
plt.show()
```

> La distribución de las ventas globales presenta una fuerte asimetría positiva. La mayoría de los videojuegos registra ventas bajas o moderadas, mientras que un número reducido alcanza ventas muy elevadas.

```python
# Conteo por género
sns.countplot(y=df["Genre"])
plt.title("Cantidad de videojuegos por género")
plt.xlabel("Cantidad de juegos")
plt.ylabel("Género")
plt.show()
```

> El gráfico muestra que algunos géneros cuentan con una mayor cantidad de títulos lanzados, lo que sugiere una mayor oferta en esos segmentos del mercado.

### Bivariado

```python
# [REQUISITO - EDA - BIVARIADO] Relacion entre variables numericas y la variable objetivo
# Ventas por género
sns.boxplot(x="Genre", y="Global_Sales", data=df, palette='viridis')
plt.xticks(rotation=45)
plt.title("Distribución de ventas globales por género")
plt.show()
```

> Se observan diferencias en las ventas globales entre los distintos géneros. Algunos géneros presentan medianas de ventas más altas y una mayor dispersión, lo que indica un mayor potencial comercial.

```python
# Puntaje de críticos vs ventas
sns.scatterplot(x="Critic_Score", y="Global_Sales", data=df, color='#e67e22', alpha=0.6)
plt.title("Relación entre ventas globales y puntaje de críticos")
plt.xlabel("Puntaje de críticos")
plt.ylabel("Ventas globales (millones)")
plt.show()
```

> El gráfico sugiere una relación positiva moderada entre el puntaje otorgado por los críticos y las ventas globales. Sin embargo, la dispersión de los puntos indica que el puntaje no es el único factor determinante del éxito comercial.

```python
# Ventas por plataforma
sns.scatterplot(x="User_Score", y="Global_Sales", data=df, color='#27ae60', alpha=0.6)
plt.title("Relación entre puntaje de usuarios y ventas globales")
plt.show()
```

> En este gráfico se analiza la relación entre la valoración de los usuarios y las ventas globales. Se observa una dispersión elevada, con algunos títulos que logran altas ventas incluso con puntajes moderados. Esto sugiere que la percepción del público tiene impacto en las ventas, aunque nuevamente no resulta un factor decisivo por sí solo.

### Multivariado

```python
# [REQUISITO - EDA - MULTIVARIADO] Analisis multivariado con hue
sns.scatterplot(
    data=df,
    x="Critic_Score",
    y="Global_Sales",
    hue="Genre",
    palette='tab10',
    alpha=0.6
)
plt.title("Ventas globales vs puntaje de críticos según género")
plt.xlabel("Puntaje de críticos")
plt.ylabel("Ventas globales (millones)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()
```

> Este análisis multivariado permite observar que la relación entre las ventas globales y el puntaje de los críticos varía según el género del videojuego. Algunos géneros logran altos niveles de ventas aun con puntuaciones moderadas, mientras que en otros la valoración parece tener un mayor impacto.

### Conclusiones del EDA

El análisis exploratorio evidencia que las ventas globales se concentran en un número reducido de videojuegos. Se observan diferencias significativas entre géneros y plataformas, así como una relación moderada entre las valoraciones de los críticos y las ventas, que no se mantiene uniforme en todos los casos.

## Preprocesamiento de datos

**Codificación de variables categóricas**

```python
# [NOTA] Codificacion de categoricas con LabelEncoder
# En la evaluacion final se requiere OneHotEncoder (para categoricas nominales)
# y TargetEncoder (para categoricas de alta cardinalidad)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["Platform"] = le.fit_transform(df["Platform"])
df["Genre"] = le.fit_transform(df["Genre"])
df["Publisher"] = le.fit_transform(df["Publisher"])
df["Rating"] = le.fit_transform(df["Rating"])
```

> ⚠️ **Nota para la clase:** este `LabelEncoder` sobre variables categóricas nominales (sin orden real) es exactamente el tipo de decisión que se corrige más abajo, en la sección de Clasificación Binaria, con `OneHotEncoder`. Vale la pena señalarlo en vivo — es un buen ejemplo de "funciona, pero no es la técnica correcta para este tipo de variable".

**Escalado de variables numéricas**

```python
# [REQUISITO - PREPROCESADO] Escalado de variables numericas con StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

numerical_cols = [
    "Critic_Score",
    "User_Score",
    "Critic_Count",
    "User_Count",
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales"
]

df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
```

## Feature Selection

**Selección de variable objetivo y variables independientes**

```python
# NOTA: La variable objetivo es Global_Sales (regresion continua)
# La evaluacion final requiere clasificacion binaria.
# Ver seccion 'CLASIFICACION BINARIA' al final del notebook.
X = df[["Critic_Score", "User_Score", "Year_of_Release"]]
y = df["Global_Sales"]
```

> Se seleccionó como variable objetivo "Global_Sales", ya que representa el rendimiento comercial del videojuego. Como variables predictoras se eligieron el puntaje de críticos, el puntaje de usuarios y el año de lanzamiento.

## Modelado

**Librerías necesarias para implementar los modelos**

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

**División de datos en conjuntos de entrenamiento y prueba**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)
```

**Predicción con conjunto de prueba**

### Modelo 1 — Linear Regression

```python
# [MODELO REGRESION 1] Linear Regression - predice ventas continuas
from sklearn.linear_model import LinearRegression

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)
```

**Métricas**

```python
# [METRICAS REGRESION] RMSE, MAE, R2 - metricas de regresion
print("Linear Regression")
print("RMSE:", mean_squared_error(y_test, y_pred_lr) ** 0.5)
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("R2:", r2_score(y_test, y_pred_lr))
```

### Modelo 2 — Decision Tree

```python
# [MODELO REGRESION 2] Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)

y_pred_dt = model_dt.predict(X_test)
```

**Métricas**

```python
# [METRICAS REGRESION] RMSE, MAE, R2
print("Decision Tree")
print("RMSE:", mean_squared_error(y_test, y_pred_dt) ** 0.5)
print("MAE:", mean_absolute_error(y_test, y_pred_dt))
print("R2:", r2_score(y_test, y_pred_dt))
```

### Modelo 3 — Random Forest

```python
# [MODELO REGRESION 3] Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)
```

**Métricas**

```python
# [METRICAS REGRESION] RMSE, MAE, R2
print("Random Forest")
print("RMSE:", mean_squared_error(y_test, y_pred_rf) ** 0.5)
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("R2:", r2_score(y_test, y_pred_rf))
```

**Conclusión sobre el modelado y las métricas**

### Por qué los modelos de regresión tienen R² cercano a cero?

Los tres modelos evaluados (Linear Regression, Decision Tree, Random Forest) presentan R² bajos. Esto no indica un error en el código, sino una característica del problema:

**El éxito de un videojuego no es lineal ni fácilmente predecible.**

El mercado de videojuegos sigue una distribución *long tail* extrema: unos pocos títulos (Wii Sports, Mario Kart) venden decenas de millones, mientras que el 80% de los juegos vende menos de 500.000 copias. Ninguna combinación de género, plataforma y puntuación de críticos explica completamente esa diferencia.

Factores que este dataset no captura pero que importan:
- Presupuesto de marketing del publisher
- Franquicia reconocida (Mario, FIFA, Call of Duty)
- Momento del lanzamiento (navidades, competencia ese mes)
- Boca a boca y viralidad en redes sociales (post-2010)

**Conclusión:** con estas variables, la regresión tiene un techo bajo. La clasificación binaria (exitoso / no exitoso) resulta más estable porque no intenta predecir el número exacto de ventas sino solo si supera un umbral.

## Conclusiones Finales

En este proyecto se realizó un análisis exploratorio de un dataset de ventas de videojuegos con el objetivo de comprender los factores que influyen en el éxito comercial de los títulos.

En la primera etapa se llevó a cabo un análisis exploratorio de datos (EDA), que permitió identificar patrones relevantes entre variables como el género, la plataforma, el puntaje de críticos y las ventas globales. Se observaron distribuciones asimétricas en las ventas y una alta dispersión, lo que sugiere que pocos títulos concentran gran parte del mercado.

En la segunda etapa se realizó el preprocesamiento de los datos, incluyendo:
- tratamiento de valores nulos mediante imputación (media, mediana y moda)
- conversión de variables a formatos adecuados
- codificación de variables categóricas
- normalización de variables numéricas

Posteriormente, se entrenaron tres modelos de regresión: Linear Regression, Decision Tree, Random Forest.

Al evaluar el desempeño mediante métricas como RMSE, MAE y R², se observó que:

- Linear Regression presentó el mejor desempeño general, con el mayor valor de R² y menor RMSE.
- Random Forest logró el menor MAE, aunque sin una mejora significativa en la capacidad explicativa del modelo.
- Decision Tree obtuvo un rendimiento inferior, incluso con un R² negativo, indicando baja capacidad predictiva.

En términos generales, todos los modelos presentaron valores de R² cercanos a cero, lo que indica que las variables seleccionadas no logran explicar adecuadamente la variabilidad de las ventas globales.

Esto sugiere que el problema es complejo y que el desempeño del modelo podría mejorar incorporando nuevas variables, realizando un mayor trabajo de feature engineering o utilizando técnicas más avanzadas.

Finalmente, se concluye que, dentro de los modelos evaluados, la regresión lineal resulta la opción más consistente para este conjunto de datos, aunque con limitaciones en su capacidad predictiva.

Como línea futura de trabajo, se podrían explorar técnicas de selección de variables más avanzadas y modelos más complejos para mejorar la capacidad predictiva.

---

# Clasificación Binaria — Éxito Comercial de Videojuegos

## ¿Por qué agregamos esta sección?

La parte anterior del notebook resuelve un problema de **regresión**: predecir el número exacto de ventas globales (un valor continuo como 3.5 millones).

La **evaluación final** requiere **clasificación binaria** (variable objetivo 0 o 1). Esta sección convierte el mismo dataset en un problema de clasificación:

> **¿Fue este videojuego un éxito comercial?**
> - Clase 1 (Éxito): `Global_Sales > 1.0 millón` de unidades
> - Clase 0 (No exitoso): `Global_Sales <= 1.0 millón`

### ¿Por qué 1.0 millón como umbral?

| Umbral | Razonamiento |
|---|---|
| **1.0M** (elegido) | Un juego que vende más de 1 millón es considerado exitoso en la industria. Representa aprox. el top 25% del dataset. |
| 0.5M | Demasiado permisivo — muchos juegos medianos quedarían como "exitosos" |
| 5.0M | Demasiado restrictivo — solo blockbusters, pocas muestras de clase 1 |

Este umbral es una **decisión de negocio**, no una decisión estadística. En un proyecto real, este umbral lo definiría el equipo junto con expertos del dominio.

### Librerías para la sección de clasificación

```python
# ══════════════════════════════════════════════════════════════════
# [REQUISITO - LIBRERIAS] Imports para la seccion de clasificacion
# ══════════════════════════════════════════════════════════════════
#
# Se importan librerias especificas de clasificacion binaria:
#   - missingno: visualizacion de nulos (EDA)
#   - Pipeline + ColumnTransformer: preprocesado sin data leakage
#   - SimpleImputer: imputa nulos (mediana para numericas, moda para categoricas)
#   - StandardScaler: normaliza variables numericas (requiere LogisticRegression)
#   - OneHotEncoder: convierte categorias en columnas 0/1
#   - LogisticRegression: modelo BASELINE (siempre el punto de comparacion)
#   - RandomForestClassifier: modelo principal (mas potente, no sensible a escala)
#   - StratifiedKFold: cross-validation con balance de clases garantizado
#   - GridSearchCV: busqueda sistematica de mejores hiperparametros
#   - roc_auc_score: metrica principal (ignora el desbalance de clases)
#   - classification_report: precision, recall, f1-score por clase
#   - confusion_matrix: las 4 combinaciones posibles de prediccion vs realidad
#   - roc_curve + auc: para graficar la curva ROC
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)

# Recargar el dataset desde cero (sin las transformaciones de regresion anteriores)
# La carga limpia garantiza que no heredamos el LabelEncoder ni el StandardScaler anterior
df_clf = pd.read_csv("/content/video_game_data.csv")
df_clf["User_Score"]       = pd.to_numeric(df_clf["User_Score"], errors="coerce")
df_clf["Year_of_Release"]  = pd.to_numeric(df_clf["Year_of_Release"], errors="coerce")
print(f"Dataset recargado: {df_clf.shape[0]:,} videojuegos, {df_clf.shape[1]} variables")
```

> 💡 **Punto para remarcar en clase:** recargar el dataset desde cero acá no es casual — es a propósito para no heredar el `LabelEncoder` ni el `StandardScaler` ya ajustados en la sección de regresión. Es un ejemplo simple pero real de cómo un estado "contaminado" de celdas anteriores puede filtrarse silenciosamente a un análisis posterior si no se controla.

### Visualización de missingness con `missingno`

```python
# ══════════════════════════════════════════════════════════════════
# [REQUISITO - EDA] VISUALIZACION DE MISSINGNESS CON MISSINGNO
# ══════════════════════════════════════════════════════════════════
#
# En el dataset de videojuegos hay variables con muchos nulos:
#   Critic_Score / Critic_Count → ~50% de nulos (muchos juegos no tienen resenas de criticos)
#   User_Score / User_Count     → ~45% de nulos (idem usuarios)
#   Developer / Rating          → ~40% de nulos
#
# Pregunta clave: ¿los nulos de Critic_Score y User_Score son MCAR o MNAR?
#   Hipotesis MNAR: los juegos mas viejos (antes de Metacritic, ~2000) no tienen puntuacion
#   Si esto es cierto → imputar con "sin puntuacion" es valido (no falsea la distribucion)
#
# msno.matrix(): permite ver si los nulos de Critic_Score y User_Score aparecen juntos
#   Si las columnas blancas se alinean en las mismas filas → esos son los juegos sin resena
#   Si no se alinean → los nulos son independientes (MCAR)
#
# msno.bar(): muestra el % de completitud de cada variable de un vistazo
#
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

msno.matrix(df_clf, ax=axes[0], sparkline=False, color=(0.85, 0.33, 0.10))
axes[0].set_title(
    "Patron de nulos por fila (msno.matrix)\n"
    "Negro = dato  |  Blanco = nulo  |  Alineacion = patron sistematico", fontsize=10)

msno.bar(df_clf, ax=axes[1], color=(0.85, 0.33, 0.10), fontsize=8)
axes[1].set_title(
    "Completitud por columna (msno.bar)\n"
    "Las barras cortas son las variables con mas nulos", fontsize=10)

plt.suptitle("[REQUISITO] Missingness — Video Games Sales Dataset", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# Tabla resumen con decision de imputacion para cada variable
nulos_pct = df_clf.isnull().mean() * 100
print("Porcentaje de nulos y estrategia de imputacion:")
for col in df_clf.columns:
    pct_val = nulos_pct[col]
    if pct_val > 40:
        estrategia = "imputar con mediana (numerica) o moda (categorica)"
    elif pct_val > 0:
        estrategia = "imputar con mediana o moda"
    else:
        estrategia = "completa, sin imputacion"
    if pct_val > 0:
        print(f"  {col:<20}: {pct_val:5.1f}% nulos → {estrategia}")
```

### Definición de la variable objetivo binaria

```python
# ══════════════════════════════════════════════════════════════════
# [REQUISITO - TARGET] Definicion de la Variable Objetivo Binaria
# ══════════════════════════════════════════════════════════════════
#
# Conversion de regresion → clasificacion binaria:
#   ANTES: predecir Global_Sales (numero continuo, ej: 3.2 millones)
#   AHORA: predecir si un juego supera 1 millon de ventas (0 o 1)
#
# Umbral elegido: 1.0 millon de unidades
#   Justificacion tecnica: percentil ~75 del dataset → mantiene clases razonablemente balanceadas
#   Justificacion de negocio: 1M de copias es el umbral de "exito" en la industria AAA
#
# Equivalente a problemas de churn:
#   Churn         : 0 = cliente sigue activo | 1 = cliente se fue
#   Exito de juego: 0 = ventas bajas/medias  | 1 = ventas altas (exito)
#
UMBRAL_EXITO = 1.0  # millones de unidades — decision de negocio
df_clf['EXITO_COMERCIAL'] = (df_clf['Global_Sales'] > UMBRAL_EXITO).astype(int)

conteo = df_clf['EXITO_COMERCIAL'].value_counts()
pct    = df_clf['EXITO_COMERCIAL'].value_counts(normalize=True) * 100

print(f"Umbral de exito: {UMBRAL_EXITO} millones de unidades vendidas globalmente")
print()
print(f"Clase 0 (ventas <= {UMBRAL_EXITO}M, no exitoso): {conteo.get(0,0):,} juegos ({pct.get(0,0):.1f}%)")
print(f"Clase 1 (ventas >  {UMBRAL_EXITO}M, exitoso):    {conteo.get(1,0):,} juegos ({pct.get(1,0):.1f}%)")
print()
print("Ejemplos de Clase 1 (exito comercial):")
print(df_clf[df_clf['EXITO_COMERCIAL'] == 1][['Name', 'Global_Sales', 'Genre', 'Platform']].head(5).to_string(index=False))
print()
print("Ejemplos de Clase 0 (sin exito comercial):")
print(df_clf[df_clf['EXITO_COMERCIAL'] == 0][['Name', 'Global_Sales', 'Genre', 'Platform']].head(5).to_string(index=False))
```

> Nota para la clase: exactamente la misma lógica de "convertir un continuo en 0/1 con un umbral de negocio" que se usó en el Ejemplo 1 para definir el churn de Veeqo (`live`/`implementation` → 1). Vale la pena señalar el paralelismo.

### Balance de clases

```python
# ══════════════════════════════════════════════════════════════════
# [REQUISITO - EDA] BALANCE DE CLASES
# ══════════════════════════════════════════════════════════════════
#
# En videojuegos, el desbalance tiene sentido de negocio:
#   La mayoria de los juegos vende poco (el mercado es long tail)
#   Solo los blockbusters (Nintendo, Activision, EA) superan el millon
#
# Para este modelo de clasificacion, el desbalance afecta el entrenamiento:
#   Si ignoramos el desbalance → el modelo aprende a predecir siempre "no exitoso"
#   Solucion: class_weight='balanced' ajusta los pesos de cada clase inversamente
#     a su frecuencia, haciendo que los errores en la clase minoritaria cuesten mas
#
# La metrica correcta con desbalance es ROC-AUC (no Accuracy):
#   ROC-AUC mide la separabilidad: ¿puede el modelo ordenar los juegos
#   de "mas probable de ser exitoso" a "menos probable"?
#
conteo = df_clf['EXITO_COMERCIAL'].value_counts()
pct    = df_clf['EXITO_COMERCIAL'].value_counts(normalize=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
labels_c = ['Clase 0\n(No exitoso)', 'Clase 1\n(Exitoso > 1M)']
colores  = ['#e67e22', '#2980b9']

axes[0].bar(labels_c, [conteo.get(0, 0), conteo.get(1, 0)],
            color=colores, edgecolor='black', linewidth=1.2)
axes[0].set_title("Balance de Clases — Conteo Absoluto", fontsize=11)
axes[0].set_ylabel("Cantidad de videojuegos")
for i, v in enumerate([conteo.get(0, 0), conteo.get(1, 0)]):
    axes[0].text(i, v + 50, f"{v:,}", ha='center', fontweight='bold')

axes[1].bar(labels_c, [pct.get(0, 0), pct.get(1, 0)],
            color=colores, edgecolor='black', linewidth=1.2)
axes[1].set_title("Balance de Clases — Porcentaje (%)", fontsize=11)
axes[1].set_ylabel("Porcentaje del total (%)")
for i, v in enumerate([pct.get(0, 0), pct.get(1, 0)]):
    axes[1].text(i, v + 0.5, f"{v:.1f}%", ha='center', fontweight='bold')

ratio = pct.get(0, 0) / max(pct.get(1, 0), 0.001)
plt.suptitle(
    f"[REQUISITO] Desbalance {ratio:.1f}:1  →  ROC-AUC + class_weight='balanced'",
    fontsize=10, style='italic', color='darkred')
plt.tight_layout()
plt.savefig("balance_clases_videojuegos.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"Conclusion: desbalance de {ratio:.1f}:1")
print(f"Accuracy maxima de un modelo tonto: {pct.get(0,0):.1f}% (siempre predice clase 0)")
print(f"ROC-AUC de un modelo tonto: 0.50 (equivale a tirar una moneda)")
```

### `Pipeline` + `GridSearchCV` + `StratifiedKFold`

```python
# ══════════════════════════════════════════════════════════════════
# [REQUISITO] SKLEARN PIPELINE + GRIDSEARCHCV + STRATIFIEDKFOLD
# ══════════════════════════════════════════════════════════════════
#
# ¿Por que usamos Pipeline aqui y no antes (en la seccion de regresion)?
# La seccion anterior del notebook aplico LabelEncoder y StandardScaler
# sobre TODO el dataset antes de dividirlo en train/test.
# Eso provoca DATA LEAKAGE: el escalado "vio" los datos de test.
#
# En este bloque usamos Pipeline para hacerlo correctamente:
#   1. fit(X_train)       → aprende mediana, escala, vocabulario OHE SOLO de train
#   2. transform(X_test)  → aplica lo aprendido a test sin ver sus estadisticas
#
# Diferencia OneHotEncoder vs LabelEncoder:
#   LabelEncoder: convierte 'Action'→0, 'Sports'→1, 'RPG'→2
#     Problema: el modelo asume que RPG(2) > Sports(1) > Action(0) en magnitud
#     Esto es INCORRECTO para categoricas nominales (no tienen orden)
#   OneHotEncoder: convierte cada categoria en una columna binaria 0/1
#     'Action' → [1,0,0] | 'Sports' → [0,1,0] | 'RPG' → [0,0,1]
#     No impone orden, el modelo puede aprender independientemente cada categoria
#

# ─────────────────────────────────────────────────────────────────
# Seleccion de features y target
# ─────────────────────────────────────────────────────────────────
num_feats = ['Critic_Score', 'User_Score', 'Critic_Count', 'User_Count', 'Year_of_Release']
cat_feats  = ['Genre', 'Rating']   # estas se codificaran con OneHotEncoder

X_clf = df_clf[num_feats + cat_feats].copy()
y_clf = df_clf['EXITO_COMERCIAL']

# stratify=y_clf → la division train/test mantiene el mismo ratio de clases
X_tr, X_te, y_tr, y_te = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42, stratify=y_clf)

print(f"Train: {X_tr.shape[0]:,} juegos | Test: {X_te.shape[0]:,} juegos")
print(f"Balance en train: {y_tr.value_counts(normalize=True).to_dict()}")

# ─────────────────────────────────────────────────────────────────
# Preprocesador
# ─────────────────────────────────────────────────────────────────
# Numericas: imputa nulos con mediana → normaliza (media=0, std=1)
# Categoricas: imputa con moda → codifica con OneHotEncoder
prepro = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imp',    SimpleImputer(strategy='median')),   # aprende mediana de train
        ('scaler', StandardScaler())                    # aprende escala de train
    ]), num_feats),
    ('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        # handle_unknown='ignore' → si aparece una categoria nueva en test, la ignora
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), cat_feats)
], remainder='drop')

# ─────────────────────────────────────────────────────────────────
# Pipelines completos
# ─────────────────────────────────────────────────────────────────
pipe_lr = Pipeline([
    ('prep',   prepro),
    # class_weight='balanced': pesa los errores en clase 1 N veces mas que en clase 0
    # max_iter=1000: suficiente para que el algoritmo converja con datos normalizados
    ('modelo', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

pipe_rf = Pipeline([
    ('prep',   prepro),
    # Random Forest no necesita StandardScaler, pero lo dejamos en el pipeline por consistencia
    # class_weight='balanced': idem LogisticRegression
    ('modelo', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1))
])

# ─────────────────────────────────────────────────────────────────
# StratifiedKFold + GridSearchCV
# ─────────────────────────────────────────────────────────────────
# n_splits=5 → 5 rondas; en cada una el 20% del train rota como validacion
cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid_rf = {
    # prefijo 'modelo__' → accede al paso 'modelo' dentro del Pipeline
    'modelo__n_estimators':     [100, 200],    # mas arboles = mas robusto pero mas lento
    'modelo__max_depth':        [5, 10, None], # None = arbol crece hasta hojas puras (riesgo de overfitting)
    'modelo__min_samples_leaf': [1, 5]         # hojas con >= 5 muestras → menos overfitting
}

print()
print("[GRIDSEARCHCV] Evaluando 12 combinaciones x 5 folds = 60 entrenamientos...")
gs = GridSearchCV(
    estimator=pipe_rf,
    param_grid=param_grid_rf,
    cv=cv_strat,        # garantiza balance en cada fold
    scoring='roc_auc',  # metrica de seleccion (correcta con desbalance)
    n_jobs=-1,          # paraleliza en todos los nucleos disponibles
    verbose=0
)
gs.fit(X_tr, y_tr)

print(f"Mejores hiperparametros: {gs.best_params_}")
print(f"ROC-AUC promedio en CV: {gs.best_score_:.4f}")
```

> ⚠️ **El contraste que vale la pena mostrar en vivo:** compará este bloque con la sección "Preprocesamiento de datos" de más arriba. Ahí, `LabelEncoder` y `StandardScaler` se ajustaron sobre el `df` completo, antes de separar train/test — data leakage real. Acá, todo el preprocesado vive adentro del `Pipeline`, así que `GridSearchCV` lo reajusta desde cero en cada fold, usando solo los datos de entrenamiento de ese fold. Es el mismo error que corrige el Ejemplo 1 en su propio contraste manual-vs-pipeline.

### Métricas finales, Confusion Matrix y Curva ROC

```python
# ══════════════════════════════════════════════════════════════════
# [REQUISITO] METRICAS FINALES + CONFUSION MATRIX + CURVA ROC
# ══════════════════════════════════════════════════════════════════
#
# Flujo de evaluacion:
#   1. GridSearchCV ya eligio el mejor modelo → lo usamos sobre X_te (nunca visto)
#   2. Comparamos contra el baseline (LogisticRegression)
#   3. Graficamos la Confusion Matrix y la Curva ROC para ambos modelos
#
# ¿Por que comparar con LogisticRegression como baseline?
#   Es el modelo mas simple de clasificacion binaria.
#   Si Random Forest no supera a LogisticRegression, algo esta mal
#   (datos insuficientes, features poco predictivas, o el problema es muy lineal).
#   La diferencia de AUC entre ambos muestra el "valor agregado" de la complejidad del RF.
#

# ─────────────────────────────────────────────────────────────────
# Predicciones
# ─────────────────────────────────────────────────────────────────
mejor_rf = gs.best_estimator_
y_pred_rf = mejor_rf.predict(X_te)
y_prob_rf = mejor_rf.predict_proba(X_te)[:, 1]  # probabilidad de ser exitoso

pipe_lr.fit(X_tr, y_tr)
y_pred_lr = pipe_lr.predict(X_te)
y_prob_lr = pipe_lr.predict_proba(X_te)[:, 1]

# ─────────────────────────────────────────────────────────────────
# Metricas en texto
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("[REQUISITO] METRICAS FINALES — Clasificacion Binaria de Exito Comercial")
print("=" * 65)
print()
print(f"  Random Forest (GridSearchCV) → ROC-AUC:   {roc_auc_score(y_te, y_prob_rf):.4f}")
print(f"  Logistic Regression (baseline)→ ROC-AUC:  {roc_auc_score(y_te, y_prob_lr):.4f}")
print()
print("[REQUISITO] Reporte de Clasificacion — Random Forest:")
print(classification_report(y_te, y_pred_rf, target_names=['No exitoso (0)', 'Exitoso (1)']))
print()
print("Como leer el reporte:")
print("  precision: de los que predijo como exitosos, ¿cuantos lo eran realmente?")
print("  recall:    de todos los exitosos reales, ¿cuantos los capturo el modelo?")
print("  f1-score:  media armonica de precision y recall (balance entre ambos)")

# ─────────────────────────────────────────────────────────────────
# Graficos: Confusion Matrix + Curva ROC
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
#   [0,0] = Verdaderos Negativos: predijo "fracaso", era fracaso    ✓
#   [0,1] = Falsos Positivos:     predijo "exito",   era fracaso    ✗ (riesgo de sobreinversion)
#   [1,0] = Falsos Negativos:     predijo "fracaso", era exito      ✗ (oportunidad perdida)
#   [1,1] = Verdaderos Positivos: predijo "exito",   era exito      ✓
cm = confusion_matrix(y_te, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['No exitoso (0)', 'Exitoso (1)'])
disp.plot(ax=axes[0], colorbar=False, cmap='Oranges')
axes[0].set_title("[REQUISITO] Confusion Matrix — Random Forest\n"
                  "Fila = Realidad  |  Columna = Prediccion del modelo", fontsize=11)

# Curva ROC — compara los dos modelos en el mismo grafico
fpr_rf, tpr_rf, _ = roc_curve(y_te, y_prob_rf)
fpr_lr, tpr_lr, _ = roc_curve(y_te, y_prob_lr)
auc_rf = auc(fpr_rf, tpr_rf)
auc_lr = auc(fpr_lr, tpr_lr)

axes[1].plot(fpr_rf, tpr_rf, color='#8e44ad', lw=2.5,
             label=f'Random Forest (AUC = {auc_rf:.3f})')
axes[1].plot(fpr_lr, tpr_lr, color='#27ae60', lw=2.0, linestyle='--',
             label=f'Logistic Regression (AUC = {auc_lr:.3f}) [baseline]')
axes[1].plot([0, 1], [0, 1], color='#7f8c8d', lw=1.5, linestyle=':',
             label='Clasificador aleatorio (AUC = 0.50)')
axes[1].fill_between(fpr_rf, tpr_rf, alpha=0.10, color='#8e44ad')
axes[1].set_xlabel("FPR — Tasa de Falsos Positivos")
axes[1].set_ylabel("TPR / Recall — Tasa de Verdaderos Positivos")
axes[1].set_title("[REQUISITO] Curva ROC — Comparacion de Modelos\n"
                  "Cuanto mas arriba-izquierda, mejor el modelo", fontsize=11)
axes[1].legend(loc='lower right', fontsize=10)

plt.suptitle("Evaluacion Visual — Clasificacion Binaria de Exito Comercial | Video Games",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("confusion_roc_videojuegos.png", dpi=150, bbox_inches='tight')
plt.show()
```

### Feature Importances — Interpretabilidad

```python
# ══════════════════════════════════════════════════════════════════
# [REQUISITO] FEATURE IMPORTANCES — Interpretabilidad
# ══════════════════════════════════════════════════════════════════
#
# ¿Que variables del videojuego predicen mejor su exito comercial?
#
# En este dataset, las features son:
#   Numericas: Critic_Score, User_Score, Critic_Count, User_Count, Year_of_Release
#   Categoricas (codificadas por OHE): Genre_Action, Genre_Sports, ..., Rating_E, Rating_M, ...
#
# El Pipeline transforma categoricas → columnas OHE antes de entrenar.
# Entonces tenemos mas features de las que pusimos originalmente.
# Necesitamos extraer los nombres del OHE para el grafico.
#
# Insight esperado en videojuegos:
#   Critic_Score suele ser importante (las resenas de metacritic influyen en ventas)
#   Year_of_Release puede ser importante (el mercado crecio mucho del 2000 al 2010)
#   Ciertos generos (Sports, Action) tienen sistematicamente mas ventas que otros
#
import matplotlib.pyplot as plt
import numpy as np

# Extraer el modelo RF del Pipeline
rf_fitted = mejor_rf.named_steps['modelo']

# Obtener nombres de columnas despues del OneHotEncoder
# El ColumnTransformer aplica primero las numericas y luego las categoricas (mismo orden que 'transformers')
ohe = mejor_rf.named_steps['prep'].named_transformers_['cat']['ohe']
ohe_names = ohe.get_feature_names_out(cat_feats)
all_feature_names = np.array(num_feats + list(ohe_names))

importancias = rf_fitted.feature_importances_

# Ordenar de mayor a menor y tomar las top 15
orden = np.argsort(importancias)[::-1]
top_n = min(15, len(orden))

# Colores degradados: verde mas importante → rojo menos importante
colores_fi = plt.cm.plasma(np.linspace(0.3, 0.9, top_n))[::-1]

fig, ax = plt.subplots(figsize=(10, 6))
# barh con orden invertido → la mas importante queda arriba del grafico
ax.barh(all_feature_names[orden[:top_n]][::-1],
        importancias[orden[:top_n]][::-1],
        color=colores_fi, edgecolor='black', linewidth=0.5)

ax.set_xlabel("Importancia Relativa (reduccion de impureza Gini)", fontsize=11)
ax.set_title(
    f"[REQUISITO] Feature Importances — Top {top_n} variables\n"
    "¿Que determina si un videojuego supera el millon de ventas?", fontsize=12)
plt.tight_layout()
plt.savefig("feature_importances_videojuegos.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"Top 5 factores que predicen el exito comercial de un videojuego:")
for name, imp in zip(all_feature_names[orden[:5]], importancias[orden[:5]]):
    print(f"  {name:<40} importancia: {imp:.4f}")
print()
print("Interpretacion de negocio:")
print("  Si Critic_Score lidera → las resenas de criticos son determinantes para las ventas")
print("  Si un genero (ej Genre_Sports) aparece alto → ese genero sistematicamente vende mas")
print("  Si Year_of_Release aparece → el mercado de la epoca importa mas que el juego en si")
```

## Resumen — Requisitos Cumplidos en esta Sección

| # | Requisito de la Evaluación Final | Celda | Estado |
|---|---|---|---|
| 1 | Dataset: >2000 filas, >15 variables | Lectura inicial | OK |
| 2 | EDA: dimensiones, tipos, estadísticos | Análisis inicial (arriba) | OK |
| 3 | EDA: visualización de nulos con `missingno` | Celda missingno | OK |
| 4 | EDA: balance de la variable objetivo | Celda balance de clases | OK |
| 5 | Target binario (clasificación) | Celda `EXITO_COMERCIAL` | OK |
| 6 | `SimpleImputer(median)` dentro de Pipeline | Celda Pipeline | OK |
| 7 | `StandardScaler` dentro de Pipeline | Celda Pipeline | OK |
| 8 | `OneHotEncoder` para categóricas | Celda Pipeline | OK |
| 9 | Baseline: `LogisticRegression` | Celda métricas | OK |
| 10 | Modelo principal: `RandomForestClassifier` | Celda Pipeline + métricas | OK |
| 11 | `StratifiedKFold(n_splits=5)` | Celda Pipeline | OK |
| 12 | `GridSearchCV(scoring='roc_auc')` | Celda Pipeline | OK |
| 13 | `ROC-AUC` como métrica de evaluación | Celda métricas | OK |
| 14 | `classification_report` | Celda métricas | OK |
| 15 | `confusion_matrix` graficada | Celda métricas | OK |
| 16 | Curva ROC graficada | Celda métricas | OK |
| 17 | Feature Importances graficadas | Celda importances | OK |

### Diferencia entre la sección de Regresión y esta sección

| Aspecto | Regresión (arriba) | Clasificación (esta sección) |
|---|---|---|
| Variable objetivo | `Global_Sales` (continua) | `EXITO_COMERCIAL` (0 o 1) |
| Tipo de problema | Regresión | Clasificación binaria |
| Modelos usados | LinearRegression, DT, RF Regressor | LogisticRegression, RF Classifier |
| Métricas | RMSE, MAE, R² | ROC-AUC, Precision, Recall, F1 |
| Preprocesado | `LabelEncoder` + `StandardScaler` sueltos (con leakage) | `Pipeline` con `SimpleImputer` + `OneHotEncoder` + `StandardScaler` (sin leakage) |
| Búsqueda de hiperparámetros | No | `GridSearchCV` + `StratifiedKFold` |

Ambas secciones son válidas como análisis del dataset. **La evaluación final requiere la sección de Clasificación Binaria.**
