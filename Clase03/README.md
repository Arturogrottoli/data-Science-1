# 🧪 Clase: Introducción a NumPy y Pandas

[Presentacion](https://docs.google.com/presentation/d/135XQdDjAvsoXtqDWhASGQ8-YWBFS5bDgxyJjfycykYs/edit?slide=id.g2204e13b0d5_2_631#slide=id.g2204e13b0d5_2_631)

## 🎯 Objetivos de la clase

- Comprender la importancia de usar bibliotecas optimizadas como **NumPy** y **Pandas** en proyectos de ciencia de datos.
- Manipular estructuras de datos con **NumPy**: arrays, operaciones vectorizadas, álgebra lineal.
- Explorar los componentes fundamentales de **Pandas**: Series y DataFrames.
- Aplicar técnicas de indexación, selección y transformación de datos reales.

---

## 📌 ¿Por qué es importante NumPy?

En ciencia de datos, trabajamos con **grandes volúmenes de datos numéricos**. Las listas de Python funcionan bien, pero no están optimizadas para cálculos científicos.

**NumPy**:
- Permite realizar **operaciones vectorizadas** (sin bucles explícitos).
- Ofrece estructuras de datos eficientes como `ndarray` (arrays multidimensionales).
- Integra funciones de **álgebra lineal**, estadísticas y manipulación matemática.

✅ Usar NumPy puede significar mejoras de **10x a 100x en performance** comparado con listas nativas de Python.

---

## 📚 Parte 1: NumPy
[Video sobre Numpy](https://www.youtube.com/watch?v=cYm3DBG6KfI&t=16s)

## 3.1 Introducción a NumPy

### 🎯 **Teoría**

**¿Qué es NumPy?**
NumPy es una librería fundamental para el manejo de datos numéricos en Python, especialmente diseñada para realizar operaciones matemáticas y de álgebra lineal de manera eficiente.

**Historia y Evolución:**
- Fue creada en 2005 como una evolución de las bibliotecas Numeric y Numarray
- Su objetivo principal es optimizar el trabajo con grandes volúmenes de datos numéricos
- Permite a científicos de datos y desarrolladores manipular y analizar datos de manera más rápida y eficiente

**Estructura de Datos Principal:**
NumPy introduce una estructura de datos llamada **ndarray** (N-dimensional array), que es un array multidimensional optimizado para operaciones numéricas. Estos arrays son similares a las listas de Python, pero con la restricción de que todos los elementos deben ser del mismo tipo de dato, lo que permite:
- Almacenamiento más eficiente
- Operaciones más rápidas
- Mejor rendimiento con grandes conjuntos de datos

### 🔑 **Características Clave de NumPy**

1. **Arreglos multidimensionales (ndarrays)**: Permite la creación de arrays de una, dos o más dimensiones
2. **Operaciones matemáticas rápidas**: Las operaciones sobre ndarrays están altamente optimizadas
3. **Compatibilidad con otras librerías**: Es la base sobre la cual se construyen Pandas, SciPy y scikit-learn

### ✳️ **Importación de la librería**

```python
import numpy as np
````

### 💡 **Arrays en NumPy: Creación y Tipos**

**Tipos y Atributos**
En NumPy, los arrays (ndarrays) son estructuras de datos que solo pueden contener elementos de un mismo tipo. Esto es una de las principales diferencias con las listas en Python, que pueden almacenar elementos de diferentes tipos.

**Tipos de Datos Soportados:**
- `int` (enteros): Para representar números enteros
- `float` (números flotantes): Para representar números reales con decimales
- `bool` (booleanos): Para representar valores True o False
- `complex` (números complejos): Para representar números complejos
- `str` (cadenas de texto): Para representar datos textuales
- `object`: Para almacenar objetos arbitrarios
- `datetime` y `timedelta`: Para trabajar con fechas

**Atributos Importantes de los ndarrays:**
- `ndim`: Número de dimensiones del array
- `shape`: Tupla que indica el tamaño del array en cada dimensión
- `size`: Número total de elementos en el array
- `dtype`: Tipo de dato de los elementos del array
- `itemsize`: Tamaño en bytes de cada elemento
- `nbytes`: Tamaño total en bytes que ocupa el array en memoria

### 💡 **Ejemplos de Creación y Atributos**

```python
import numpy as np

# Crear un ndarray de ceros con 10 elementos
arr = np.zeros(10)

print("Atributos del array:")
print(f"ndim: {arr.ndim}")       # 1 (una dimensión)
print(f"shape: {arr.shape}")     # (10,) (10 elementos en una sola dimensión)
print(f"size: {arr.size}")       # 10 (10 elementos en total)
print(f"dtype: {arr.dtype}")     # float64 (tipo de dato de los elementos)
print(f"itemsize: {arr.itemsize}")   # 8 (cada elemento ocupa 8 bytes)
print(f"nbytes: {arr.nbytes}")   # 80 (10 elementos * 8 bytes por elemento)

# Crear arrays con diferentes tipos de datos
arr_int = np.array([1, 2, 3, 4], dtype=np.int32)
arr_float = np.array([1.1, 2.2, 3.3], dtype=np.float64)
arr_bool = np.array([True, False, True], dtype=np.bool_)

print(f"\nArray entero: {arr_int}, dtype: {arr_int.dtype}")
print(f"Array flotante: {arr_float}, dtype: {arr_float.dtype}")
print(f"Array booleano: {arr_bool}, dtype: {arr_bool.dtype}")
```

### 🔢 **Ejercicio 1: Crear vectores**

```python
# Crear un vector desde una lista
v = np.array([1, 2, 3, 4])
print(f"Vector: {v}")
print(f"Tipo: {type(v)}")
print(f"Shape: {v.shape}")
print(f"Dtype: {v.dtype}")

# Crear arrays con diferentes métodos
zeros = np.zeros(5)
ones = np.ones(5)
range_array = np.arange(0, 10, 2)  # De 0 a 10 con paso 2
linspace_array = np.linspace(0, 1, 5)  # 5 puntos entre 0 y 1

print(f"\nArray de ceros: {zeros}")
print(f"Array de unos: {ones}")
print(f"Array con arange: {range_array}")
print(f"Array con linspace: {linspace_array}")
```

### 🧪 **Ejercicios Prácticos de Creación**

**Ejercicio 1: Explorar atributos de arrays**
```python
# Crea diferentes tipos de arrays y explora sus atributos
arrays = [
    np.array([1, 2, 3, 4, 5]),
    np.zeros((3, 4)),
    np.ones((2, 3, 2)),
    np.random.rand(5, 5)
]

for i, arr in enumerate(arrays):
    print(f"\nArray {i+1}:")
    print(f"  Contenido: {arr}")
    print(f"  Dimensiones: {arr.ndim}")
    print(f"  Forma: {arr.shape}")
    print(f"  Tamaño total: {arr.size}")
    print(f"  Tipo de datos: {arr.dtype}")
```

**Ejercicio 2: Crear arrays con tipos específicos**
```python
# Crea arrays con tipos de datos específicos
int_array = np.array([1, 2, 3, 4], dtype=np.int32)
float_array = np.array([1.1, 2.2, 3.3], dtype=np.float32)
complex_array = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)

print(f"Array entero (32 bits): {int_array}, dtype: {int_array.dtype}")
print(f"Array flotante (32 bits): {float_array}, dtype: {float_array.dtype}")
print(f"Array complejo (64 bits): {complex_array}, dtype: {complex_array.dtype}")

# Comparar uso de memoria
print(f"\nUso de memoria:")
print(f"int_array: {int_array.nbytes} bytes")
print(f"float_array: {float_array.nbytes} bytes")
print(f"complex_array: {complex_array.nbytes} bytes")
```

### 🔁 **Ejercicio 2: Operaciones vectorizadas**

```python
a = np.array([10, 20, 30])
b = np.array([1, 2, 3])

# Suma elemento a elemento
print(f"Suma: {a + b}")

# Multiplicación escalar
print(f"Multiplicación escalar: {a * 2}")

# Potencia
print(f"Potencia: {b ** 2}")

# Otras operaciones vectorizadas
print(f"Raíz cuadrada: {np.sqrt(b)}")
print(f"Exponencial: {np.exp(b)}")
print(f"Logaritmo natural: {np.log(b)}")
print(f"Valor absoluto: {np.abs([-1, -2, -3])}")
```

### 🧪 **Ejercicios Prácticos de Operaciones**

**Ejercicio 1: Comparar rendimiento con listas de Python**
```python
import time

# Crear datos de prueba
size = 1000000
python_list = list(range(size))
numpy_array = np.array(range(size))

# Operación con lista de Python
start_time = time.time()
result_python = [x * 2 for x in python_list]
python_time = time.time() - start_time

# Operación con NumPy
start_time = time.time()
result_numpy = numpy_array * 2
numpy_time = time.time() - start_time

print(f"Tiempo con lista de Python: {python_time:.4f} segundos")
print(f"Tiempo con NumPy: {numpy_time:.4f} segundos")
print(f"NumPy es {python_time/numpy_time:.1f}x más rápido")
```

**Ejercicio 2: Operaciones estadísticas básicas**
```python
# Crear un array de datos
data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Estadísticas básicas
print(f"Datos: {data}")
print(f"Media: {np.mean(data)}")
print(f"Mediana: {np.median(data)}")
print(f"Desviación estándar: {np.std(data)}")
print(f"Varianza: {np.var(data)}")
print(f"Mínimo: {np.min(data)}")
print(f"Máximo: {np.max(data)}")
print(f"Suma: {np.sum(data)}")
print(f"Producto: {np.prod(data)}")
```

### 🧮 **Ejercicio 3: Matrices y álgebra lineal**

```python
# Matriz 2x2
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"Matriz A:\n{A}")
print(f"Matriz B:\n{B}")

# Transpuesta
print(f"\nTranspuesta de A:\n{A.T}")

# Inversa
print(f"\nInversa de A:\n{np.linalg.inv(A)}")

# Producto matricial
print(f"\nProducto matricial A × B:\n{np.dot(A, B)}")

# Determinante
print(f"\nDeterminante de A: {np.linalg.det(A)}")

# Valores propios
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nValores propios de A: {eigenvalues}")
print(f"Vectores propios de A:\n{eigenvectors}")
```

### 🧪 **Ejercicios Prácticos de Álgebra Lineal**

**Ejercicio 1: Resolver sistema de ecuaciones lineales**
```python
# Sistema de ecuaciones:
# 2x + y = 5
# x + 3y = 6

# Matriz de coeficientes
A = np.array([[2, 1], [1, 3]])
# Vector de términos independientes
b = np.array([5, 6])

# Resolver el sistema
x = np.linalg.solve(A, b)
print(f"Solución del sistema:")
print(f"x = {x[0]:.2f}")
print(f"y = {x[1]:.2f}")

# Verificar la solución
verification = np.dot(A, x)
print(f"\nVerificación:")
print(f"A × x = {verification}")
print(f"b = {b}")
```

**Ejercicio 2: Operaciones con matrices**
```python
# Crear matrices de ejemplo
matrix_3x3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
identity_3x3 = np.eye(3)
random_matrix = np.random.rand(3, 3)

print(f"Matriz 3x3:\n{matrix_3x3}")
print(f"\nMatriz identidad 3x3:\n{identity_3x3}")
print(f"\nMatriz aleatoria 3x3:\n{random_matrix}")

# Operaciones básicas
print(f"\nSuma de matrices:\n{matrix_3x3 + identity_3x3}")
print(f"\nMultiplicación elemento a elemento:\n{matrix_3x3 * identity_3x3}")
print(f"\nTraza de la matriz: {np.trace(matrix_3x3)}")
print(f"Rango de la matriz: {np.linalg.matrix_rank(matrix_3x3)}")
```

**Ejercicio 3: Decomposición de matrices**
```python
# Crear una matriz simétrica positiva definida
A = np.array([[4, 2], [2, 5]])

# Descomposición de Cholesky
L = np.linalg.cholesky(A)
print(f"Matriz original:\n{A}")
print(f"\nFactor de Cholesky L:\n{L}")
print(f"\nVerificación: L × L^T =\n{np.dot(L, L.T)}")

# Descomposición QR
Q, R = np.linalg.qr(A)
print(f"\nDescomposición QR:")
print(f"Q:\n{Q}")
print(f"R:\n{R}")
print(f"\nVerificación: Q × R =\n{np.dot(Q, R)}")
```

---

## 📚 Parte 2: Pandas

### ✳️ ¿Qué es Pandas?

Pandas es la **librería base para manipular datos en forma tabular** en Python. Provee dos estructuras fundamentales:

* `Series`: vector unidimensional con índice.
* `DataFrame`: tabla bidimensional con columnas e índices.

```python
import pandas as pd
```

---

### 📊 Ejercicio 1: Crear Series

```python
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s)
```

### 📊 Ejercicio 2: Crear DataFrames

```python
data = {
    'nombre': ['Ana', 'Luis', 'Juan'],
    'edad': [23, 35, 29],
    'ciudad': ['Córdoba', 'Buenos Aires', 'Rosario']
}

df = pd.DataFrame(data)
print(df)
```

---

### 🔍 Ejercicio 3: Selección e indexación

```python
# Seleccionar columna
print(df['edad'])

# Filtrar por condición
print(df[df['edad'] > 30])

# Acceder por etiqueta o posición
print(df.loc[1])   # Fila con índice 1
print(df.iloc[0])  # Primera fila
```

---

## 💬 Discusión guiada

* ¿Cuáles son las ventajas prácticas de usar NumPy frente a listas?
* ¿Por qué Pandas es más útil que un diccionario de listas?
* ¿Qué errores comunes hay al manipular DataFrames?

---

# 📝 Actividad práctica 

## Actividad Práctica: NumPy y Pandas en Python

## 1. NumPy: Manipulación de Arrays

**Creación de arrays básicos:**

```python
import numpy as np

# Crear un array simple
array_simple = np.array([1, 2, 3, 4, 5])
print("Array simple:", array_simple)

# Crear una matriz 2D
matriz_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("Matriz 2D:\n", matriz_2d)

# Crear arrays con valores específicos
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
print("Array de ceros:\n", zeros)
print("Array de unos:\n", ones)
```

**Operaciones matemáticas con arrays:**

```python
# Operaciones aritméticas básicas
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("Suma:", a + b)  # o np.add(a, b)
print("Resta:", a - b)  # o np.subtract(a, b)
print("Multiplicación:", a * b)  # o np.multiply(a, b)
print("División:", a / b)  # o np.divide(a, b)
print("Exponenciación:", a ** 2)  # o np.power(a, 2)

# Operaciones estadísticas
print("Suma de elementos:", np.sum(a))
print("Media:", np.mean(a))
print("Desviación estándar:", np.std(a))
```

## 2. Pandas como Herramienta de Analítica

### Series en Pandas

```python
import pandas as pd

# Crear una Serie desde una lista
serie_lista = pd.Series([10, 20, 30, 40])
print("Serie desde lista:\n", serie_lista)

# Crear una Serie desde un diccionario
datos_diccionario = {"día1": 420, "día2": 380, "día3": 390}
serie_dict = pd.Series(datos_diccionario)
print("Serie desde diccionario:\n", serie_dict)

# Selección específica con índice
serie_filtrada = pd.Series(datos_diccionario, index=["día1", "día2"])
print("Serie filtrada:\n", serie_filtrada)
```

### DataFrames en Pandas

```python
# Crear DataFrame desde diccionario
datos = {
    "Nombre": ["Juan", "María", "Pedro"],
    "Edad": [30, 25, 40],
    "Ciudad": ["Caracas", "Maracaibo", "Valencia"]
}
df = pd.DataFrame(datos)
print("DataFrame desde diccionario:\n", df)

# Acceder a datos del DataFrame
print("\nColumna de nombres:\n", df["Nombre"])
print("\nPrimeras filas:\n", df.head(2))
print("\nInformación del DataFrame:\n", df.info())
print("\nEstadísticas descriptivas:\n", df.describe())
```

### Operaciones Básicas con DataFrames

```python
# Filtrado de datos
mayores_30 = df[df["Edad"] > 30]
print("Personas mayores de 30:\n", mayores_30)

# Añadir nueva columna
df["Activo"] = [True, False, True]
print("\nDataFrame con nueva columna:\n", df)

# Operaciones en columnas
df["Edad_en_meses"] = df["Edad"] * 12
print("\nEdad en meses:\n", df)

# Agrupar y resumir datos
promedio_edad_por_ciudad = df.groupby("Ciudad")["Edad"].mean()
print("\nPromedio de edad por ciudad:\n", promedio_edad_por_ciudad)
```

## 3. Métodos de Lectura de Archivos con Pandas

### Lectura de archivos CSV

```python
# Leer archivo CSV local
# df_csv = pd.read_csv('datos.csv')

# Leer CSV desde URL
url_csv = 'https://raw.githubusercontent.com/JJTorresDS/stocks-ds-edu/main/stocks.csv'
df_stocks = pd.read_csv(url_csv)
print("Primeras filas del CSV:\n", df_stocks.head())

# Opciones de lectura CSV
df_csv_options = pd.read_csv(url_csv, 
                            # skiprows=1,         # Saltar filas
                            # usecols=['MSFT', 'AMZN'],  # Seleccionar columnas
                            index_col='formatted_date',  # Establecer columna como índice
                            parse_dates=['formatted_date'])  # Parsear fechas
print("\nCSV con opciones de lectura:\n", df_csv_options.head())
```

### Lectura de archivos JSON

```python
# Leer archivo JSON (ejemplo)
# Comentado para no generar errores si no existe el archivo
"""
url_json = 'https://raw.githubusercontent.com/tu_usuario/tu_repo/main/datos.json'
df_json = pd.read_json(url_json)
print("Datos del JSON:\n", df_json.head())

# Leer JSON con líneas múltiples
df_json_lines = pd.read_json(url_json, lines=True)
print("\nJSON con múltiples líneas:\n", df_json_lines.head())
"""
```

### Otros formatos de lectura

```python
# Leer Excel (comentado para evitar errores)
# df_excel = pd.read_excel('datos.xlsx', sheet_name='Hoja1')

# Leer SQL (ejemplo conceptual)
"""
from sqlalchemy import create_engine
engine = create_engine('sqlite:///mi_base.db')
df_sql = pd.read_sql('SELECT * FROM mi_tabla', con=engine)
"""

# Ejemplos de escritura de datos
df.to_csv('mi_dataframe.csv', index=False)
# df.to_excel('mi_dataframe.xlsx', index=False)
# df.to_json('mi_dataframe.json', orient='records')
```

## Ejercicio Práctico

Para consolidar los conocimientos adquiridos, se propone el siguiente ejercicio:

1. Cargar el archivo de stocks desde GitHub
2. Calcular el rendimiento promedio mensual de cada acción
3. Identificar la acción con mayor volatilidad
4. Crear un gráfico que muestre la evolución de las 3 acciones con mejor rendimiento

```python
# Solución parcial del ejercicio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos
url = 'https://raw.githubusercontent.com/JJTorresDS/stocks-ds-edu/main/stocks.csv'
stocks_df = pd.read_csv(url)
stocks_df['formatted_date'] = pd.to_datetime(stocks_df['formatted_date'])

# Establecer la fecha como índice
stocks_df = stocks_df.set_index('formatted_date')

# Las primeras filas del dataset
print(stocks_df.head())
```