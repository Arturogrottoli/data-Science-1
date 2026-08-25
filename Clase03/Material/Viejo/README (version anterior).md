# Clase 03: NumPy y Pandas

[Presentacion](https://docs.google.com/presentation/d/135XQdDjAvsoXtqDWhASGQ8-YWBFS5bDgxyJjfycykYs/edit?slide=id.g2204e13b0d5_2_631#slide=id.g2204e13b0d5_2_631)

---

### Contenidos
1. [Repaso Python — Clase 2](#1-repaso-python--clase-2)
2. [Pandas](#2-pandas)
   - [2.1 Series](#21-series)
   - [2.2 DataFrames](#22-dataframes)
   - [2.3 Selección e Indexación](#23-selección-e-indexación)
   - [2.4 Valores Ausentes (NaN)](#24-valores-ausentes-nan)
   - [2.5 Agrupación y Agregación](#25-agrupación-y-agregación)
   - [2.6 Transformación y Limpieza](#26-transformación-y-limpieza)
3. [Lectura y Escritura de Archivos](#3-lectura-y-escritura-de-archivos)
4. [Ejercicio: Análisis de Acciones](#4-ejercicio-análisis-de-acciones)
5. [NumPy](#5-numpy)
   - [5.1 Tipos y Atributos del ndarray](#51-tipos-y-atributos-del-ndarray)
   - [5.2 Creación de Arrays](#52-creación-de-arrays)
   - [5.3 Operaciones Vectorizadas](#53-operaciones-vectorizadas)
   - [5.4 Estadísticas con NumPy](#54-estadísticas-con-numpy)
   - [5.5 Álgebra Lineal](#55-álgebra-lineal)
   - [5.6 Reshape, Concatenación y Splitting](#56-reshape-concatenación-y-splitting)
6. [Cierre: NumPy + Pandas juntos](#6-cierre-numpy--pandas-juntos)

---

## 1. Repaso Python — Clase 2

Antes de arrancar con NumPy y Pandas, repasamos las estructuras de Python que vamos a usar constantemente:

```python
# Listas y diccionarios
lista = [1, 2, 3, 4]
diccionario = {"nombre": "Ana", "edad": 25}

# For y while
for item in lista:
    print(item)

# Funciones
def calcular_promedio(numeros):
    return sum(numeros) / len(numeros)
```

**¿Por qué importa el repaso?** NumPy y Pandas son extensiones naturales de estas estructuras: los arrays de NumPy mejoran las listas, y los DataFrames de Pandas mejoran los diccionarios de listas.

```python
# De lista a NumPy array
import numpy as np
temperaturas = [20, 22, 25, 23, 21]
arr = np.array(temperaturas)
print(arr * 9/5 + 32)   # vectorizado, sin bucle

# De diccionario a DataFrame
import pandas as pd
datos = {"nombres": ["Ana", "Luis"], "edades": [25, 30]}
df = pd.DataFrame(datos)
print(df[df["edades"] > 27])   # filtro simple
```

---

## 2. Pandas

### ¿Qué es Pandas?

Pandas es una librería de Python para manipular y analizar datos tabulares. Construida sobre NumPy, extiende sus capacidades con estructuras de datos flexibles y funciones optimizadas para análisis de datos del mundo real.

**¿Por qué Pandas y no Python solo?**

Con Python puro se puede hacer todo, pero rápidamente se vuelve engorroso: leer un CSV implica abrir el archivo, iterar línea por línea, separar por coma, convertir tipos... y eso antes de hacer cualquier análisis. Pandas resuelve eso con una sola línea (`pd.read_csv`). Además, opera sobre columnas enteras a la vez sin necesidad de bucles, lo que hace el código más corto, más legible y considerablemente más rápido cuando los datos crecen.

**Historia:** lanzada en 2008 por Wes McKinney en AQR Capital Management para análisis financiero. Su nombre viene de "Panel Data". Hoy es la herramienta estándar del ecosistema de Data Science en Python.

**Estructuras principales:**
- **Series**: array unidimensional con índice etiquetado
- **DataFrame**: tabla bidimensional — filas y columnas etiquetadas

**Por qué conviene sobre diccionarios y listas:**

| Tarea | Python puro | Pandas |
|-------|------------|--------|
| Cargar CSV | 10+ líneas | `pd.read_csv("archivo.csv")` |
| Promedio de columna | bucle manual | `df["col"].mean()` |
| Filtrar filas | list comprehension | `df[df["col"] > valor]` |
| Agrupar y agregar | diccionarios anidados | `df.groupby("col").mean()` |

```python
import pandas as pd
```

---

### 2.1 Series

Una Serie es un array unidimensional donde cada elemento tiene un índice asociado (número, texto o fecha).

```python
import pandas as pd

# Desde una lista
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s)

# Desde diccionario
data_dict = {"día1": 420, "día2": 380, "día3": 390}
serie_dict = pd.Series(data_dict)
print(serie_dict)

# Acceder a elementos
print(s['a'])           # por etiqueta
print(s.iloc[1])        # por posición

# Operaciones
print(s.sum())
print(s.mean())
print(s.max())
```

---

### 2.2 DataFrames

Un DataFrame es una tabla: filas y columnas, cada columna es una Serie.

```python
data = {
    'nombre': ['Ana', 'Luis', 'Juan', 'María'],
    'edad':   [23, 35, 29, 28],
    'ciudad': ['Córdoba', 'Buenos Aires', 'Rosario', 'Madrid'],
    'salario': [45000, 55000, 48000, 52000]
}

df = pd.DataFrame(data)
print(df)

# Información del DataFrame
print(f"Forma: {df.shape}")          # (4, 4)
print(df.dtypes)                      # tipos de cada columna
print(df.info())                      # resumen completo
print(df.describe())                  # estadísticas descriptivas
print(df.head(2))                     # primeras 2 filas
print(df.tail(2))                     # últimas 2 filas
```

---

### 2.3 Selección e Indexación

```python
# Seleccionar columna → Series
print(df['edad'])

# Seleccionar múltiples columnas
print(df[['nombre', 'edad']])

# Filtrar por condición
print(df[df['edad'] > 30])

# Filtros múltiples (& para AND, | para OR)
resultado = df[(df['edad'] >= 25) & (df['salario'] > 50000)]
print(resultado)

# loc: por etiqueta
print(df.loc[1])                         # fila con índice 1
print(df.loc[df['nombre'] == 'Ana'])     # filas donde nombre es Ana

# iloc: por posición numérica
print(df.iloc[0])       # primera fila
print(df.iloc[0:2])     # primeras 2 filas
print(df.iloc[0, 1])    # fila 0, columna 1

# Acceder a un valor específico
edad_ana = df.loc[df['nombre'] == 'Ana', 'edad'].iloc[0]
print(f"Edad de Ana: {edad_ana}")
```

---

### 2.4 Valores Ausentes (NaN)

En análisis de datos reales, siempre hay valores faltantes. Pandas los representa como `NaN` (Not a Number) y provee herramientas para manejarlos.

**¿Por qué aparecen?** Errores de carga, datos no disponibles, problemas de transmisión o simplemente que el dato no existía.

```python
import pandas as pd
import numpy as np

data_with_nan = {
    'nombre':      ['Ana', 'Luis', 'Juan', 'María', 'Pedro'],
    'edad':        [23, 35, np.nan, 28, 42],
    'ciudad':      ['Córdoba', np.nan, 'Rosario', 'Madrid', 'Barcelona'],
    'salario':     [45000, 55000, 48000, np.nan, 60000],
    'departamento': ['IT', 'Ventas', np.nan, 'Marketing', 'IT']
}
df_nan = pd.DataFrame(data_with_nan)

# Detectar valores ausentes
print(df_nan.isnull().sum())              # conteo por columna
print(df_nan.isnull().any().any())        # ¿hay algún NaN?

# Porcentaje de ausentes
pct = (df_nan.isnull().sum() / len(df_nan)) * 100
print(pct)

# Eliminar filas con NaN
df_clean = df_nan.dropna()               # filas con cualquier NaN
df_clean2 = df_nan.dropna(how='all')     # solo filas completamente vacías
df_clean3 = df_nan.dropna(axis=1)        # eliminar columnas con NaN

# Rellenar valores ausentes
df_nan['edad'] = df_nan['edad'].fillna(df_nan['edad'].mean())
df_nan['ciudad'] = df_nan['ciudad'].fillna('Desconocida')
df_nan['salario'] = df_nan['salario'].fillna(df_nan['salario'].median())

# Forward fill / backward fill (útil para series temporales)
df_ffill = df_nan.fillna(method='ffill')  # propagar valor anterior
df_bfill = df_nan.fillna(method='bfill')  # propagar valor siguiente

# Interpolación lineal
df_interp = df_nan.interpolate()
```

**Estrategia recomendada:**
- Variables numéricas continuas → rellenar con media o mediana
- Variables categóricas → rellenar con moda o "Desconocido"
- Si hay >40% de NaN en una columna → considerar eliminarla
- Si hay >40% de NaN en una fila → considerar eliminarla

---

### 2.5 Agrupación y Agregación

`groupby` divide el DataFrame por grupos, aplica una función y combina los resultados. Flujo: **Split → Apply → Combine**.

```python
ventas_data = {
    'vendedor': ['Ana', 'Luis', 'Juan', 'María', 'Ana', 'Luis', 'Juan', 'María'],
    'producto': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'ventas':   [100, 150, 200, 120, 180, 90, 220, 160],
    'region':   ['Norte', 'Sur', 'Norte', 'Sur', 'Norte', 'Sur', 'Norte', 'Sur'],
    'mes':      ['Enero', 'Enero', 'Enero', 'Enero', 'Febrero', 'Febrero', 'Febrero', 'Febrero']
}
df_ventas = pd.DataFrame(ventas_data)

# Agrupación simple
ventas_por_vendedor = df_ventas.groupby('vendedor')['ventas'].agg(['sum', 'mean', 'count'])
print(ventas_por_vendedor)

# Agrupación múltiple
multi = df_ventas.groupby(['vendedor', 'producto'])['ventas'].sum()
print(multi)

# Función de agregación personalizada
def rango(x):
    return x.max() - x.min()

stats = df_ventas.groupby('vendedor')['ventas'].agg(['sum', 'mean', 'std', rango])
print(stats)

# Pivot table
pivot = df_ventas.pivot_table(
    values='ventas',
    index='vendedor',
    columns='producto',
    aggfunc='sum',
    fill_value=0
)
print(pivot)

# Tabla de contingencia
crosstab = pd.crosstab(df_ventas['region'], df_ventas['producto'])
print(crosstab)

# Análisis temporal: agrupar por mes
ventas_mensuales = df_ventas.groupby('mes')['ventas'].sum()
crecimiento = ventas_mensuales.pct_change() * 100
print(crecimiento)
```

---

### 2.6 Transformación y Limpieza

```python
datos_sucios = {
    'nombre':         ['  Ana  ', 'Luis', '  Juan  ', 'María', 'Pedro'],
    'edad':           ['25', '30', '35', '28', '42'],
    'fecha_registro': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-01-25', '2023-04-05'],
    'puntuacion':     ['8.5', '7.2', '9.1', '6.8', '8.9'],
    'activo':         ['Sí', 'No', 'Sí', 'Sí', 'No']
}
df_sucio = pd.DataFrame(datos_sucios)

# Limpiar espacios en blanco
df_sucio['nombre'] = df_sucio['nombre'].str.strip()

# Convertir tipos
df_sucio['edad']           = pd.to_numeric(df_sucio['edad'])
df_sucio['puntuacion']     = pd.to_numeric(df_sucio['puntuacion'])
df_sucio['fecha_registro'] = pd.to_datetime(df_sucio['fecha_registro'])
df_sucio['activo']         = df_sucio['activo'].map({'Sí': True, 'No': False})

# Extraer información
df_sucio['año_registro']   = df_sucio['fecha_registro'].dt.year
df_sucio['dominio_email']  = df_sucio.get('email', pd.Series(dtype=str))

# Crear categorías con apply
def categorizar_edad(edad):
    if edad < 30:   return 'Joven'
    elif edad < 40: return 'Adulto'
    else:           return 'Senior'

df_sucio['categoria_edad'] = df_sucio['edad'].apply(categorizar_edad)

# Crear categorías con pd.cut
df_sucio['nivel_puntuacion'] = pd.cut(
    df_sucio['puntuacion'],
    bins=[0, 7, 8, 10],
    labels=['Bajo', 'Medio', 'Alto']
)

print(df_sucio.dtypes)
print(df_sucio.describe())
```

---

## 3. Lectura y Escritura de Archivos

Pandas puede leer y escribir en la mayoría de los formatos de datos usados en la industria: CSV, Excel, JSON, SQL, HTML.

### Leer CSV

```python
import pandas as pd

# Lectura básica
df = pd.read_csv('archivo.csv')

# Desde URL (GitHub, datasets públicos)
url = 'https://raw.githubusercontent.com/JJTorresDS/stocks-ds-edu/main/stocks.csv'
df_stocks = pd.read_csv(url)
print(df_stocks.head())
```

**Opciones más usadas:**

```python
pd.read_csv('archivo.csv',
    sep=';',                        # delimitador (por defecto coma)
    header=0,                       # fila que contiene los nombres de columna
    usecols=['col1', 'col3'],        # leer solo esas columnas
    index_col='id',                  # usar columna como índice
    parse_dates=['fecha'],           # parsear fechas automáticamente
    na_values=['N/A', '--', ''],     # qué valores tratar como NaN
    skiprows=2                       # saltear primeras 2 filas
)
```

### Leer Excel

```python
# Hoja específica
df = pd.read_excel('archivo.xlsx', sheet_name='Hoja1')

# Múltiples hojas
hojas = pd.read_excel('archivo.xlsx', sheet_name=['Hoja1', 'Hoja2'])
for nombre, df in hojas.items():
    print(nombre, df.shape)

# Todas las hojas
todas = pd.read_excel('archivo.xlsx', sheet_name=None)
print(list(todas.keys()))
```

### Leer desde otros formatos

```python
# JSON
df_json = pd.read_json('archivo.json')

# HTML (extrae tablas de páginas web)
tablas = pd.read_html('pagina.html')
df = tablas[0]

# SQL (requiere sqlalchemy)
# from sqlalchemy import create_engine
# engine = create_engine('sqlite:///mi_base.db')
# df = pd.read_sql('SELECT * FROM tabla', con=engine)
```

### Escritura

```python
df.to_csv('salida.csv', index=False)
df.to_excel('salida.xlsx', sheet_name='Datos', index=False)
df.to_json('salida.json', orient='records')
```

### Tabla resumen

| Formato | Función lectura | Función escritura |
|---------|----------------|-------------------|
| CSV | `pd.read_csv()` | `df.to_csv()` |
| Excel | `pd.read_excel()` | `df.to_excel()` |
| JSON | `pd.read_json()` | `df.to_json()` |
| HTML | `pd.read_html()` | `df.to_html()` |
| SQL | `pd.read_sql()` | `df.to_sql()` |

### Buenas prácticas al leer archivos

```python
import os

# Verificar que existe antes de leer
file_path = 'archivo.csv'
if os.path.exists(file_path):
    df = pd.read_csv(file_path)

# Manejo de errores
try:
    df = pd.read_csv('archivo.csv')
except FileNotFoundError:
    print("El archivo no existe")
except pd.errors.EmptyDataError:
    print("El archivo está vacío")

# Archivos muy grandes: leer por chunks
for chunk in pd.read_csv('archivo_grande.csv', chunksize=10000):
    procesar(chunk)
```

---

## 4. Ejercicio: Análisis de Acciones

```python
import pandas as pd
import numpy as np

# Cargar datos de stocks desde GitHub
url = 'https://raw.githubusercontent.com/JJTorresDS/stocks-ds-edu/main/stocks.csv'
stocks_df = pd.read_csv(url)
stocks_df['formatted_date'] = pd.to_datetime(stocks_df['formatted_date'])
stocks_df = stocks_df.set_index('formatted_date')

print(stocks_df.head())
print(f"Forma: {stocks_df.shape}")
print(f"Rango: {stocks_df.index.min()} → {stocks_df.index.max()}")
print(stocks_df.describe())
```

Actividades:
1. Calcular el rendimiento promedio mensual de cada acción
2. Identificar la acción con mayor volatilidad (desvío estándar de retornos)
3. Filtrar los días en que AAPL subió más de 2%
4. Crear una tabla pivot con precio promedio por mes y por acción

---

## 5. NumPy

### ¿Qué es NumPy?

NumPy (Numerical Python) es la librería base para computación numérica en Python. Creada en 2005, es la fundación sobre la que se construyen Pandas, scikit-learn, TensorFlow y SciPy.

Su estructura central es el **ndarray**: una grilla N-dimensional de elementos del mismo tipo, mucho más eficiente en memoria y velocidad que una lista de Python.

**¿Por qué es más rápido?**

| Aspecto | Lista Python | NumPy array |
|---------|-------------|-------------|
| Velocidad | Lenta (bucle en Python) | Rápida (C internamente) |
| Memoria | Mayor (objetos Python) | Menor (tipos nativos) |
| Operaciones matemáticas | Bucles explícitos | Vectorizadas (una línea) |
| Matrices y álgebra lineal | No soporta nativamente | Soporte completo |

```python
import numpy as np
```

---

### 5.1 Tipos y Atributos del ndarray

```python
import numpy as np

arr = np.zeros(10)

print(arr.ndim)       # 1 (dimensiones)
print(arr.shape)      # (10,)
print(arr.size)       # 10 (total de elementos)
print(arr.dtype)      # float64
print(arr.itemsize)   # 8 bytes por elemento
print(arr.nbytes)     # 80 bytes en total

# Tipos de datos disponibles
arr_int   = np.array([1, 2, 3, 4], dtype=np.int32)
arr_float = np.array([1.1, 2.2, 3.3], dtype=np.float64)
arr_bool  = np.array([True, False, True], dtype=np.bool_)

print(f"int:   {arr_int},   dtype: {arr_int.dtype}")
print(f"float: {arr_float}, dtype: {arr_float.dtype}")
print(f"bool:  {arr_bool},  dtype: {arr_bool.dtype}")

# Comparar memoria
arr_32 = np.array([1, 2, 3, 4], dtype=np.int32)
arr_64 = np.array([1, 2, 3, 4], dtype=np.int64)
print(f"int32: {arr_32.nbytes} bytes")
print(f"int64: {arr_64.nbytes} bytes")
```

---

### 5.2 Creación de Arrays

```python
import numpy as np

# Desde una lista
arr = np.array([1, 2, 3, 4, 5])

# Arrays predefinidos
np.zeros(5)           # [0. 0. 0. 0. 0.]
np.ones(5)            # [1. 1. 1. 1. 1.]
np.zeros((3, 4))      # matriz 3x4 de ceros
np.eye(3)             # matriz identidad 3x3

# Secuencias
np.arange(0, 10, 2)    # [0 2 4 6 8]
np.linspace(0, 1, 5)   # [0. 0.25 0.5 0.75 1.]

# Aleatorios
np.random.rand(5)         # valores uniformes [0, 1)
np.random.randn(5)        # distribución normal estándar
np.random.randint(0, 10, 5)  # enteros entre 0 y 9

# Matriz 2D
matriz = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(f"Shape: {matriz.shape}")   # (2, 3)
print(f"ndim: {matriz.ndim}")     # 2
```

---

### 5.3 Operaciones Vectorizadas

```python
import numpy as np

a = np.array([10, 20, 30])
b = np.array([1, 2, 3])

# Aritméticas (elemento a elemento)
print(a + b)          # [11 22 33]
print(a * 2)          # [20 40 60]
print(b ** 2)         # [1  4  9]

# Funciones matemáticas
print(np.sqrt(b))     # [1. 1.41 1.73]
print(np.exp(b))      # [e, e², e³]
print(np.log(b))      # logaritmo natural
print(np.abs([-1, -2, 3]))  # [1 2 3]

# Indexing y slicing
arr = np.array([10, 20, 30, 40, 50])
print(arr[0])         # 10
print(arr[-1])        # 50
print(arr[1:4])       # [20 30 40]
print(arr[::2])       # [10 30 50]

# Filtro con condición booleana
grandes = arr[arr > 25]
print(grandes)        # [30 40 50]
```

Comparativa de rendimiento:

```python
import time

size = 1_000_000
python_list = list(range(size))
numpy_array = np.array(range(size))

start = time.time()
result = [x * 2 for x in python_list]
print(f"Lista Python: {time.time() - start:.4f}s")

start = time.time()
result = numpy_array * 2
print(f"NumPy:        {time.time() - start:.4f}s")
```

---

### 5.4 Estadísticas con NumPy

```python
import numpy as np

data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

print(f"Media:    {np.mean(data)}")
print(f"Mediana:  {np.median(data)}")
print(f"Desvío:   {np.std(data):.2f}")
print(f"Varianza: {np.var(data):.2f}")
print(f"Mínimo:   {np.min(data)}")
print(f"Máximo:   {np.max(data)}")
print(f"Suma:     {np.sum(data)}")
print(f"Producto: {np.prod(data)}")

# Percentiles
print(np.percentile(data, [25, 50, 75]))

# Sobre matrices: axis=0 por columna, axis=1 por fila
matriz = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(np.mean(matriz, axis=0))  # promedio por columna: [2.5, 3.5, 4.5]
print(np.mean(matriz, axis=1))  # promedio por fila:    [2., 5.]

# Correlación entre arrays
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])
print(f"Correlación: {np.corrcoef(a, b)[0, 1]:.2f}")
```

---

### 5.5 Álgebra Lineal

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A.T)                    # Transpuesta
print(np.linalg.inv(A))       # Inversa
print(np.dot(A, B))           # Producto matricial
print(np.linalg.det(A))       # Determinante
print(np.trace(A))            # Traza

# Valores propios (eigenvalues)
valores, vectores = np.linalg.eig(A)
print(f"Eigenvalues: {valores}")

# Resolver sistema de ecuaciones lineales
# 2x + y = 5
# x + 3y = 6
coefs = np.array([[2, 1], [1, 3]])
terminos = np.array([5, 6])
solucion = np.linalg.solve(coefs, terminos)
print(f"x = {solucion[0]:.2f}, y = {solucion[1]:.2f}")

# Descomposición QR
Q, R = np.linalg.qr(A)
print(f"Q:\n{Q}")
print(f"R:\n{R}")
```

---

### 5.6 Reshape, Concatenación y Splitting

```python
import numpy as np

arr = np.arange(12)
print(arr)              # [ 0  1  2 ... 11]

# Reshape
matriz = arr.reshape(3, 4)
print(matriz)           # 3 filas, 4 columnas
print(matriz.reshape(2, 6))
print(matriz.reshape(-1))   # volver a 1D

# Concatenar
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.concatenate([a, b]))       # [1 2 3 4 5 6]

m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])
print(np.vstack([m1, m2]))   # apilar verticalmente
print(np.hstack([m1, m2]))   # apilar horizontalmente

# Splitting
arr = np.arange(9)
partes = np.array_split(arr, 3)
for p in partes:
    print(p)
```

---

## 6. Cierre: NumPy + Pandas juntos

En proyectos reales, NumPy y Pandas se usan en conjunto: Pandas para estructurar y limpiar datos, NumPy para los cálculos numéricos pesados.

```python
import pandas as pd
import numpy as np

# Cargar datos
url = 'https://raw.githubusercontent.com/JJTorresDS/stocks-ds-edu/main/stocks.csv'
df = pd.read_csv(url)
df['formatted_date'] = pd.to_datetime(df['formatted_date'])
df = df.set_index('formatted_date')

# Pandas: preparación
print(f"Forma: {df.shape}")
print(f"Columnas: {list(df.columns)}")

# NumPy: cálculos de rendimientos
prices = df[['MSFT', 'AMZN', 'AAPL']].values
returns = np.diff(prices, axis=0) / prices[:-1]

returns_df = pd.DataFrame(
    returns,
    index=df.index[1:],
    columns=['MSFT', 'AMZN', 'AAPL']
)

# Estadísticas con NumPy
for col in returns_df.columns:
    media = np.mean(returns_df[col])
    vol   = np.std(returns_df[col]) * np.sqrt(252)
    print(f"{col}: media={media:.4f}, volatilidad anual={vol:.4f}")

# Pandas: agrupar por mes
rendimientos_mensuales = returns_df.resample('M').apply(
    lambda x: np.prod(1 + x) - 1
)
print("\nRendimientos mensuales promedio:")
print(rendimientos_mensuales.mean())
```

**Regla práctica:** usá Pandas para estructurar y manipular datos, y NumPy cuando necesités cálculos matemáticos puros — son complementarios, no competidores.
