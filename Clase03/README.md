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

### 🎯 **Resumen de NumPy**

**Ventajas principales:**
- **Rendimiento**: Operaciones vectorizadas 10-100x más rápidas que listas de Python
- **Memoria eficiente**: Arrays homogéneos con tipos de datos optimizados
- **Funcionalidad matemática**: Amplia biblioteca de funciones matemáticas y estadísticas
- **Interoperabilidad**: Base para otras librerías de ciencia de datos

**Casos de uso típicos:**
- Procesamiento de datos numéricos a gran escala
- Cálculos matemáticos y estadísticos
- Álgebra lineal y computación científica
- Manipulación de imágenes y señales
- Simulaciones y modelado numérico

---

## 📚 Parte 2: Pandas

### 🎯 **Teoría: Introducción a Pandas**

**¿Qué es Pandas?**
Pandas es una librería de Python diseñada para facilitar el manejo y análisis de datos. Construida sobre NumPy, Pandas extiende sus capacidades, proporcionando estructuras de datos y funciones avanzadas que permiten una manipulación más flexible y eficiente de datos tabulares y de series temporales.

**Historia y Evolución:**
- Lanzada inicialmente en 2008, Pandas se ha convertido en una herramienta indispensable para científicos de datos
- Es parte del proyecto NUMFOCUS, que apoya el desarrollo de herramientas de código abierto para ciencia de datos
- Su nombre proviene de "Panel Data", reflejando su capacidad para manejar datos estructurados

**Características Destacadas:**
- **Series y DataFrames**: Estructuras de datos principales para datos unidimensionales y bidimensionales
- **Operaciones de manipulación**: Filtrado, agregación y transformación de datos de manera intuitiva
- **Compatibilidad múltiple**: Lectura y escritura desde CSV, Excel, SQL, JSON y más formatos
- **Manejo de datos ausentes**: Herramientas robustas para detectar, gestionar y llenar valores faltantes

### ✳️ **Importación de la librería**

```python
import pandas as pd
```

---

### 📊 **Series y DataFrames: Estructuras Fundamentales**

**Series:**
Una Serie es una estructura de datos unidimensional que puede almacenar datos de cualquier tipo. Es similar a un array unidimensional de NumPy, pero con la ventaja adicional de que cada elemento tiene un índice asociado.

**Características de las Series:**
- **Índices**: Cada elemento tiene un índice asociado (numérico o texto)
- **Homogeneidad**: Todos los elementos deben ser del mismo tipo de datos
- **Funcionalidades**: Métodos incorporados para filtrado, agregación y operaciones aritméticas

### 📊 **Ejercicio 1: Crear Series**

```python
import pandas as pd

# Crear una Serie básica
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print("Serie básica:")
print(s)

# Crear Serie desde diccionario
data_dict = {"día1": 420, "día2": 380, "día3": 390}
serie_dict = pd.Series(data_dict)
print("\nSerie desde diccionario:")
print(serie_dict)

# Crear Serie con diferentes tipos de datos
serie_mixta = pd.Series([1, 2.5, "texto", True], index=['num1', 'num2', 'texto', 'booleano'])
print("\nSerie con diferentes tipos:")
print(serie_mixta)

# Acceder a elementos por índice
print(f"\nValor en 'a': {s['a']}")
print(f"Valor en posición 1: {s.iloc[1]}")

# Operaciones básicas
print(f"\nSuma de la serie: {s.sum()}")
print(f"Media de la serie: {s.mean()}")
print(f"Valor máximo: {s.max()}")
```

### 📊 **Ejercicio 2: Crear DataFrames**

**DataFrames:**
El DataFrame es una estructura de datos bidimensional que puede considerarse como una tabla, similar a una hoja de cálculo o una tabla en una base de datos. Un DataFrame está compuesto por múltiples Series, donde cada columna representa una Serie con su propio tipo de dato.

**Características de los DataFrames:**
- **Estructura tabular**: Filas y columnas para visualización y manipulación de datos
- **Índices en filas y columnas**: Acceso eficiente a datos por etiquetas
- **Manipulación de datos**: Funciones para selección, filtrado, agregación y fusión
- **Soporte para datos ausentes**: Manejo robusto de valores NaN

```python
# Crear DataFrame desde diccionario
data = {
    'nombre': ['Ana', 'Luis', 'Juan', 'María'],
    'edad': [23, 35, 29, 28],
    'ciudad': ['Córdoba', 'Buenos Aires', 'Rosario', 'Madrid'],
    'salario': [45000, 55000, 48000, 52000]
}

df = pd.DataFrame(data)
print("DataFrame básico:")
print(df)

# Crear DataFrame con índice personalizado
df_indexed = pd.DataFrame(data, index=['emp1', 'emp2', 'emp3', 'emp4'])
print("\nDataFrame con índice personalizado:")
print(df_indexed)

# Información del DataFrame
print(f"\nForma del DataFrame: {df.shape}")
print(f"Tipos de datos:\n{df.dtypes}")
print(f"Información general:")
print(df.info())

# Estadísticas descriptivas
print(f"\nEstadísticas descriptivas:")
print(df.describe())
```

---

### 🔍 **Ejercicio 3: Selección e indexación**

```python
# Seleccionar columna
print("Columna 'edad':")
print(df['edad'])

# Seleccionar múltiples columnas
print("\nColumnas 'nombre' y 'edad':")
print(df[['nombre', 'edad']])

# Filtrar por condición
print("\nPersonas mayores de 30 años:")
print(df[df['edad'] > 30])

# Filtros múltiples
print("\nPersonas entre 25 y 35 años con salario > 50000:")
print(df[(df['edad'] >= 25) & (df['edad'] <= 35) & (df['salario'] > 50000)])

# Acceder por etiqueta o posición
print("\nFila con índice 1 (loc):")
print(df.loc[1])

print("\nPrimera fila (iloc):")
print(df.iloc[0])

# Selección por posición
print("\nPrimeras 2 filas:")
print(df.iloc[0:2])

# Selección por etiquetas
print("\nFilas 'emp1' y 'emp3' (si usamos índice personalizado):")
print(df_indexed.loc[['emp1', 'emp3']])

# Acceder a valores específicos
print(f"\nEdad de Ana: {df.loc[df['nombre'] == 'Ana', 'edad'].iloc[0]}")
print(f"Salario de Luis: {df.loc[df['nombre'] == 'Luis', 'salario'].iloc[0]}")
```

### 🚨 **Manejo de Datos Ausentes (NaN)**

En el análisis de datos, es común encontrarse con conjuntos de datos incompletos donde faltan algunos valores. Pandas representa estos valores ausentes con `NaN` (Not a Number) y proporciona herramientas robustas para manejarlos.

**¿Por qué son importantes los datos ausentes?**
- Pueden surgir por errores en la recopilación de datos
- Problemas de transmisión o almacenamiento
- Datos simplemente no disponibles
- El manejo adecuado es crucial para la calidad del análisis

### 🔍 **Ejercicio 4: Detección y Manejo de Datos Ausentes**

```python
import pandas as pd
import numpy as np

# Crear un DataFrame con valores ausentes
data_with_nan = {
    'nombre': ['Ana', 'Luis', 'Juan', 'María', 'Pedro'],
    'edad': [23, 35, np.nan, 28, 42],
    'ciudad': ['Córdoba', np.nan, 'Rosario', 'Madrid', 'Barcelona'],
    'salario': [45000, 55000, 48000, np.nan, 60000],
    'departamento': ['IT', 'Ventas', np.nan, 'Marketing', 'IT']
}

df_nan = pd.DataFrame(data_with_nan)
print("DataFrame con valores ausentes:")
print(df_nan)

# 1. DETECCIÓN DE VALORES AUSENTES
print("\n=== DETECCIÓN DE VALORES AUSENTES ===")

# Detectar valores ausentes
print("Valores ausentes por columna:")
print(df_nan.isnull().sum())

print("\nMatriz de valores ausentes:")
print(df_nan.isnull())

# Verificar si hay valores ausentes en todo el DataFrame
print(f"\n¿Hay valores ausentes?: {df_nan.isnull().any().any()}")

# 2. ELIMINACIÓN DE VALORES AUSENTES
print("\n=== ELIMINACIÓN DE VALORES AUSENTES ===")

# Eliminar filas con cualquier valor ausente
df_clean_rows = df_nan.dropna()
print("DataFrame sin filas con valores ausentes:")
print(df_clean_rows)

# Eliminar solo filas donde TODAS las columnas tienen valores ausentes
df_clean_all = df_nan.dropna(how='all')
print("\nDataFrame eliminando solo filas completamente vacías:")
print(df_clean_all)

# Eliminar columnas con valores ausentes
df_clean_cols = df_nan.dropna(axis=1)
print("\nDataFrame sin columnas con valores ausentes:")
print(df_clean_cols)

# 3. RELLENO DE VALORES AUSENTES
print("\n=== RELLENO DE VALORES AUSENTES ===")

# Rellenar con valor constante
df_fill_constant = df_nan.fillna(0)
print("Rellenando con 0:")
print(df_fill_constant)

# Rellenar con valores específicos por columna
df_fill_specific = df_nan.fillna({
    'edad': df_nan['edad'].mean(),
    'ciudad': 'Desconocida',
    'salario': df_nan['salario'].median(),
    'departamento': 'Sin asignar'
})
print("\nRellenando con valores específicos:")
print(df_fill_specific)

# Rellenar con el valor anterior (forward fill)
df_ffill = df_nan.fillna(method='ffill')
print("\nRellenando con valor anterior (forward fill):")
print(df_ffill)

# Rellenar con el valor siguiente (backward fill)
df_bfill = df_nan.fillna(method='bfill')
print("\nRellenando con valor siguiente (backward fill):")
print(df_bfill)

# 4. INTERPOLACIÓN
print("\n=== INTERPOLACIÓN ===")

# Interpolación lineal (útil para series temporales)
df_interpolate = df_nan.interpolate()
print("Interpolación lineal:")
print(df_interpolate)

# 5. ANÁLISIS DE DATOS AUSENTES
print("\n=== ANÁLISIS DE DATOS AUSENTES ===")

# Porcentaje de valores ausentes por columna
missing_percentage = (df_nan.isnull().sum() / len(df_nan)) * 100
print("Porcentaje de valores ausentes por columna:")
for col, pct in missing_percentage.items():
    print(f"  {col}: {pct:.1f}%")

# Estrategia recomendada basada en el análisis
print("\nEstrategia recomendada:")
if missing_percentage['edad'] < 20:
    print("  - Edad: Rellenar con media/mediana")
else:
    print("  - Edad: Eliminar filas o usar técnicas avanzadas")

if missing_percentage['ciudad'] < 10:
    print("  - Ciudad: Rellenar con moda")
else:
    print("  - Ciudad: Crear categoría 'Desconocida'")
```

### 🧪 **Ejercicios Prácticos de Manejo de Datos Ausentes**

**Ejercicio 1: Análisis de dataset real con valores ausentes**
```python
# Crear dataset más complejo
np.random.seed(42)
n_rows = 100

data_complex = {
    'id': range(1, n_rows + 1),
    'edad': np.random.normal(35, 10, n_rows),
    'ingresos': np.random.exponential(50000, n_rows),
    'educacion': np.random.choice(['Primaria', 'Secundaria', 'Universidad', 'Postgrado'], n_rows),
    'satisfaccion': np.random.uniform(1, 10, n_rows)
}

# Introducir valores ausentes de manera realista
df_complex = pd.DataFrame(data_complex)

# Simular valores ausentes (5-15% por columna)
for col in ['edad', 'ingresos', 'educacion', 'satisfaccion']:
    mask = np.random.random(n_rows) < 0.1  # 10% de valores ausentes
    df_complex.loc[mask, col] = np.nan

print("Dataset con valores ausentes:")
print(df_complex.head(10))
print(f"\nValores ausentes por columna:")
print(df_complex.isnull().sum())

# Estrategia de limpieza
print("\n=== ESTRATEGIA DE LIMPIEZA ===")

# 1. Análisis inicial
print("1. Análisis inicial:")
print(f"   - Total de filas: {len(df_complex)}")
print(f"   - Filas con al menos un valor ausente: {df_complex.isnull().any(axis=1).sum()}")

# 2. Limpieza por columnas
print("\n2. Limpieza por columnas:")

# Edad: rellenar con mediana
df_complex['edad'] = df_complex['edad'].fillna(df_complex['edad'].median())
print("   - Edad: rellenada con mediana")

# Ingresos: rellenar con media
df_complex['ingresos'] = df_complex['ingresos'].fillna(df_complex['ingresos'].mean())
print("   - Ingresos: rellenados con media")

# Educación: rellenar con moda
education_mode = df_complex['educacion'].mode()[0]
df_complex['educacion'] = df_complex['educacion'].fillna(education_mode)
print(f"   - Educación: rellenada con moda ({education_mode})")

# Satisfacción: eliminar filas (pocos valores ausentes)
df_complex = df_complex.dropna(subset=['satisfaccion'])
print("   - Satisfacción: filas eliminadas")

print(f"\nDataset final: {len(df_complex)} filas")
print("Valores ausentes restantes:")
print(df_complex.isnull().sum())
```

### 🔧 **Operaciones Avanzadas con Pandas**

### 📊 **Ejercicio 5: Agrupación y Agregación**

```python
# Crear dataset para análisis de ventas
ventas_data = {
    'vendedor': ['Ana', 'Luis', 'Juan', 'María', 'Ana', 'Luis', 'Juan', 'María'],
    'producto': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'ventas': [100, 150, 200, 120, 180, 90, 220, 160],
    'region': ['Norte', 'Sur', 'Norte', 'Sur', 'Norte', 'Sur', 'Norte', 'Sur'],
    'mes': ['Enero', 'Enero', 'Enero', 'Enero', 'Febrero', 'Febrero', 'Febrero', 'Febrero']
}

df_ventas = pd.DataFrame(ventas_data)
print("Dataset de ventas:")
print(df_ventas)

# 1. AGRUPACIÓN SIMPLE
print("\n=== AGRUPACIÓN SIMPLE ===")

# Agrupar por vendedor y calcular estadísticas
ventas_por_vendedor = df_ventas.groupby('vendedor')['ventas'].agg(['sum', 'mean', 'count'])
print("Ventas por vendedor:")
print(ventas_por_vendedor)

# 2. AGRUPACIÓN MÚLTIPLE
print("\n=== AGRUPACIÓN MÚLTIPLE ===")

# Agrupar por vendedor y producto
ventas_vendedor_producto = df_ventas.groupby(['vendedor', 'producto'])['ventas'].sum()
print("Ventas por vendedor y producto:")
print(ventas_vendedor_producto)

# 3. FUNCIONES DE AGREGACIÓN PERSONALIZADAS
print("\n=== FUNCIONES PERSONALIZADAS ===")

def rango_ventas(x):
    return x.max() - x.min()

ventas_stats = df_ventas.groupby('vendedor')['ventas'].agg([
    'sum', 'mean', 'std', rango_ventas, 'count'
]).rename(columns={'rango_ventas': 'rango'})

print("Estadísticas completas por vendedor:")
print(ventas_stats)

# 4. PIVOT TABLES
print("\n=== PIVOT TABLES ===")

# Crear tabla pivote
pivot_ventas = df_ventas.pivot_table(
    values='ventas',
    index='vendedor',
    columns='producto',
    aggfunc='sum',
    fill_value=0
)
print("Tabla pivote - Ventas por vendedor y producto:")
print(pivot_ventas)

# 5. CROSS TABULATION
print("\n=== CROSS TABULATION ===")

# Tabla de contingencia
crosstab_region_producto = pd.crosstab(df_ventas['region'], df_ventas['producto'])
print("Distribución de productos por región:")
print(crosstab_region_producto)

# 6. ANÁLISIS TEMPORAL
print("\n=== ANÁLISIS TEMPORAL ===")

# Agrupar por mes y calcular totales
ventas_mensuales = df_ventas.groupby('mes')['ventas'].sum()
print("Ventas totales por mes:")
print(ventas_mensuales)

# Calcular crecimiento mensual
crecimiento_mensual = ventas_mensuales.pct_change() * 100
print("\nCrecimiento mensual (%):")
print(crecimiento_mensual)
```

### 🔄 **Ejercicio 6: Transformación y Limpieza de Datos**

```python
# Crear dataset con datos "sucios"
datos_sucios = {
    'nombre': ['  Ana  ', 'Luis', '  Juan  ', 'María', 'Pedro'],
    'edad': ['25', '30', '35', '28', '42'],
    'email': ['ana@email.com', 'luis@email.com', 'juan@email.com', 'maria@email.com', 'pedro@email.com'],
    'fecha_registro': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-01-25', '2023-04-05'],
    'puntuacion': ['8.5', '7.2', '9.1', '6.8', '8.9'],
    'activo': ['Sí', 'No', 'Sí', 'Sí', 'No']
}

df_sucio = pd.DataFrame(datos_sucios)
print("Dataset original (sucio):")
print(df_sucio)

# 1. LIMPIEZA DE TEXTO
print("\n=== LIMPIEZA DE TEXTO ===")

# Limpiar espacios en blanco
df_sucio['nombre'] = df_sucio['nombre'].str.strip()
print("Nombres limpios:")
print(df_sucio['nombre'])

# 2. CONVERSIÓN DE TIPOS
print("\n=== CONVERSIÓN DE TIPOS ===")

# Convertir edad a numérico
df_sucio['edad'] = pd.to_numeric(df_sucio['edad'])
print("Edad convertida a numérico:")
print(df_sucio['edad'])

# Convertir puntuación a float
df_sucio['puntuacion'] = pd.to_numeric(df_sucio['puntuacion'])
print("Puntuación convertida a float:")
print(df_sucio['puntuacion'])

# Convertir fecha a datetime
df_sucio['fecha_registro'] = pd.to_datetime(df_sucio['fecha_registro'])
print("Fecha convertida a datetime:")
print(df_sucio['fecha_registro'])

# Convertir activo a booleano
df_sucio['activo'] = df_sucio['activo'].map({'Sí': True, 'No': False})
print("Activo convertido a booleano:")
print(df_sucio['activo'])

# 3. EXTRACCIÓN DE INFORMACIÓN
print("\n=== EXTRACCIÓN DE INFORMACIÓN ===")

# Extraer dominio del email
df_sucio['dominio_email'] = df_sucio['email'].str.split('@').str[1]
print("Dominio del email:")
print(df_sucio['dominio_email'])

# Extraer año de registro
df_sucio['año_registro'] = df_sucio['fecha_registro'].dt.year
print("Año de registro:")
print(df_sucio['año_registro'])

# 4. CREACIÓN DE CATEGORÍAS
print("\n=== CREACIÓN DE CATEGORÍAS ===")

# Crear categoría de edad
def categorizar_edad(edad):
    if edad < 30:
        return 'Joven'
    elif edad < 40:
        return 'Adulto'
    else:
        return 'Senior'

df_sucio['categoria_edad'] = df_sucio['edad'].apply(categorizar_edad)
print("Categoría de edad:")
print(df_sucio['categoria_edad'])

# Crear categoría de puntuación
df_sucio['nivel_puntuacion'] = pd.cut(
    df_sucio['puntuacion'],
    bins=[0, 7, 8, 10],
    labels=['Bajo', 'Medio', 'Alto']
)
print("Nivel de puntuación:")
print(df_sucio['nivel_puntuacion'])

# 5. DATASET FINAL LIMPIO
print("\n=== DATASET FINAL LIMPIO ===")
print("Tipos de datos finales:")
print(df_sucio.dtypes)

print("\nDataset limpio:")
print(df_sucio)

# 6. RESUMEN ESTADÍSTICO
print("\n=== RESUMEN ESTADÍSTICO ===")
print("Estadísticas numéricas:")
print(df_sucio.describe())

print("\nDistribución por categorías:")
print("Categoría de edad:")
print(df_sucio['categoria_edad'].value_counts())

print("\nNivel de puntuación:")
print(df_sucio['nivel_puntuacion'].value_counts())
```

---

## 💬 **Discusión Guiada**

### 🤔 **Preguntas para Reflexión:**

**Sobre NumPy:**
- ¿Cuáles son las ventajas prácticas de usar NumPy frente a listas de Python?
- ¿En qué situaciones específicas notarías la diferencia de rendimiento?
- ¿Por qué es importante la homogeneidad de tipos en los arrays de NumPy?

**Sobre Pandas:**
- ¿Por qué Pandas es más útil que un diccionario de listas para análisis de datos?
- ¿Qué ventajas ofrece el sistema de indexación de Pandas?
- ¿Cuándo usarías Series vs DataFrame?

**Sobre Datos Ausentes:**
- ¿Qué estrategia usarías para manejar datos ausentes en un dataset de ventas?
- ¿Cómo decidirías entre eliminar vs rellenar valores faltantes?
- ¿Qué impacto pueden tener los datos ausentes en tus análisis?

**Sobre Integración:**
- ¿Cómo aprovechas las fortalezas de ambas librerías en un proyecto real?
- ¿Qué patrones de uso has identificado que funcionan mejor?
- ¿Cómo escalarías estas técnicas a datasets más grandes?

### 💡 **Errores Comunes y Mejores Prácticas:**

**Errores Comunes:**
- Usar bucles en lugar de operaciones vectorizadas
- No verificar tipos de datos antes de operaciones
- Ignorar valores ausentes sin analizarlos
- Confundir `loc` vs `iloc` en Pandas

**Mejores Prácticas:**
- Siempre verificar la forma y tipos de datos
- Usar operaciones vectorizadas cuando sea posible
- Documentar estrategias de manejo de datos ausentes
- Validar resultados después de transformaciones

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

---

## 🎯 **Cierre y Conclusión: NumPy y Pandas en el Ecosistema de Data Science**

### 📊 **Resumen Integrador: El Ecosistema Completo**

Hemos explorado las dos bibliotecas fundamentales que forman la base del ecosistema de ciencia de datos en Python:

**🔢 NumPy (Numerical Python):**
- **Propósito**: Computación numérica eficiente y operaciones matemáticas vectorizadas
- **Fortaleza**: Rendimiento optimizado para cálculos científicos y álgebra lineal
- **Casos de uso**: Procesamiento de datos numéricos, simulaciones, análisis estadístico

**📈 Pandas (Panel Data):**
- **Propósito**: Manipulación y análisis de datos estructurados en formato tabular
- **Fortaleza**: Flexibilidad para trabajar con datos heterogéneos y series temporales
- **Casos de uso**: Limpieza de datos, análisis exploratorio, preparación para machine learning

### 🚀 **Ejemplo Integrador: Análisis de Rendimiento de Acciones**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar datos de acciones
url = 'https://raw.githubusercontent.com/JJTorresDS/stocks-ds-edu/main/stocks.csv'
df = pd.read_csv(url)
df['formatted_date'] = pd.to_datetime(df['formatted_date'])
df = df.set_index('formatted_date')

# 1. USANDO PANDAS: Preparación y limpieza de datos
print("=== ANÁLISIS CON PANDAS ===")
print(f"Forma del dataset: {df.shape}")
print(f"Columnas disponibles: {list(df.columns)}")
print(f"Rango de fechas: {df.index.min()} a {df.index.max()}")

# Información básica del dataset
print("\nInformación del dataset:")
print(df.info())

# 2. USANDO NUMPY: Cálculos estadísticos avanzados
print("\n=== ANÁLISIS CON NUMPY ===")

# Convertir a arrays de NumPy para cálculos rápidos
prices_array = df[['MSFT', 'AMZN', 'AAPL']].values

# Calcular rendimientos diarios usando NumPy
returns = np.diff(prices_array, axis=0) / prices_array[:-1]
returns_df = pd.DataFrame(returns, 
                         index=df.index[1:], 
                         columns=['MSFT', 'AMZN', 'AAPL'])

# Estadísticas usando NumPy
print("Estadísticas de rendimientos diarios:")
for col in returns_df.columns:
    print(f"\n{col}:")
    print(f"  Media: {np.mean(returns_df[col]):.4f}")
    print(f"  Desv. Estándar: {np.std(returns_df[col]):.4f}")
    print(f"  Volatilidad anual: {np.std(returns_df[col]) * np.sqrt(252):.4f}")

# 3. INTEGRACIÓN: Análisis combinado
print("\n=== ANÁLISIS INTEGRADO ===")

# Usar Pandas para agrupar por mes y NumPy para cálculos
monthly_returns = returns_df.resample('M').apply(lambda x: np.prod(1 + x) - 1)

print("Rendimientos mensuales promedio:")
for col in monthly_returns.columns:
    avg_return = np.mean(monthly_returns[col])
    print(f"  {col}: {avg_return:.4f} ({avg_return*100:.2f}%)")

# 4. Visualización del análisis
plt.figure(figsize=(12, 8))

# Subplot 1: Precios históricos (Pandas)
plt.subplot(2, 2, 1)
df[['MSFT', 'AMZN', 'AAPL']].plot()
plt.title('Precios Históricos')
plt.ylabel('Precio ($)')
plt.legend()

# Subplot 2: Rendimientos diarios (NumPy + Pandas)
plt.subplot(2, 2, 2)
returns_df.plot()
plt.title('Rendimientos Diarios')
plt.ylabel('Rendimiento')
plt.legend()

# Subplot 3: Distribución de rendimientos (NumPy)
plt.subplot(2, 2, 3)
for col in returns_df.columns:
    plt.hist(returns_df[col].dropna(), bins=50, alpha=0.7, label=col)
plt.title('Distribución de Rendimientos')
plt.xlabel('Rendimiento')
plt.ylabel('Frecuencia')
plt.legend()

# Subplot 4: Correlación entre acciones (NumPy)
plt.subplot(2, 2, 4)
correlation_matrix = np.corrcoef(returns_df.dropna().T)
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(returns_df.columns)), returns_df.columns, rotation=45)
plt.yticks(range(len(returns_df.columns)), returns_df.columns)
plt.title('Matriz de Correlación')

plt.tight_layout()
plt.show()

# 5. Conclusiones del análisis
print("\n=== CONCLUSIONES DEL ANÁLISIS ===")
print("1. La integración de NumPy y Pandas permite:")
print("   - Pandas: Manejo eficiente de datos temporales y heterogéneos")
print("   - NumPy: Cálculos matemáticos rápidos y precisos")
print("   - Combinación: Análisis completo y visualización profesional")

print("\n2. Ventajas del ecosistema:")
print("   - Flexibilidad: Pandas para datos, NumPy para cálculos")
print("   - Rendimiento: Operaciones vectorizadas optimizadas")
print("   - Interoperabilidad: Fácil conversión entre estructuras")
print("   - Escalabilidad: Manejo eficiente de grandes volúmenes de datos")
```

### 🎓 **Lecciones Clave Aprendidas**

**🔧 NumPy:**
- Los arrays homogéneos permiten operaciones vectorizadas ultra-rápidas
- Las funciones matemáticas optimizadas son esenciales para cálculos científicos
- La interoperabilidad con otras librerías es fundamental en el ecosistema

**📊 Pandas:**
- Los DataFrames proporcionan una interfaz intuitiva para datos tabulares
- Las operaciones de indexación y filtrado son poderosas y expresivas
- La integración con NumPy permite lo mejor de ambos mundos

**🔄 Sinergia:**
- NumPy maneja la computación numérica pesada
- Pandas maneja la estructuración y manipulación de datos
- Juntos forman la base sólida para cualquier proyecto de data science

### 🚀 **Próximos Pasos en tu Journey de Data Science**

1. **Profundizar en NumPy**: Explorar álgebra lineal avanzada, broadcasting, y operaciones con arrays multidimensionales
2. **Dominar Pandas**: Aprender groupby avanzado, pivot tables, merging y joining de datasets
3. **Integración con otras librerías**: Conectar con Matplotlib/Seaborn para visualización, scikit-learn para machine learning
4. **Optimización**: Aprender técnicas de vectorización y evitar bucles cuando sea posible

### 💡 **Reflexión Final**

NumPy y Pandas no son solo herramientas, sino **fundamentos** del ecosistema de data science en Python. Su combinación te permite:

- **Transformar datos** de manera eficiente y expresiva
- **Realizar análisis** complejos con código simple y legible
- **Escalar** tus soluciones desde prototipos hasta sistemas de producción
- **Colaborar** con otros científicos de datos usando estándares de la industria

**Recuerda**: La maestría en estas bibliotecas te abrirá las puertas a todo el ecosistema de data science en Python. ¡Son tu base sólida para construir soluciones de datos impactantes! 🎯
```