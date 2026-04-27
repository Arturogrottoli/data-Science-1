# Clase 02: Fundamentos de Python para Ciencia de Datos

**[GUIA DE CLASE (PPTX)](https://docs.google.com/presentation/d/1jMeDrYaVZE7IYGxo6fF4dGYgT-KHFZJmzr-vAKvVGes/edit#slide=id.g2204f2a9531_0_0)**

---

### Contenidos
1. [¿Qué es Python?](#1-qué-es-python)
2. [Variables y Asignación](#2-variables-y-asignación)
3. [Tipos de Datos](#3-tipos-de-datos)
4. [Operadores](#4-operadores)
5. [Estructuras de Control: Condicionales](#5-estructuras-de-control-condicionales)
6. [Bucles](#6-bucles)
7. [Funciones](#7-funciones)
8. [NumPy: arreglos y operaciones vectorizadas](#8-numpy-arreglos-y-operaciones-vectorizadas)
9. [Visualización con Plotly Express](#9-visualización-con-plotly-express)
10. [Pandas: primera mirada](#10-pandas-primera-mirada)

---

## 1. ¿Qué es Python?

Python es un lenguaje de programación de alto nivel, interpretado y de tipado dinámico. Creado por Guido van Rossum en 1991, hoy es el lenguaje más usado en ciencia de datos, machine learning e inteligencia artificial.

**Características clave:**
- Sintaxis simple y legible, similar al inglés
- Tipado dinámico: no declarás el tipo de la variable
- Interpretado: se ejecuta línea a línea
- Multiplataforma: Windows, Mac, Linux
- Ecosistema enorme: pandas, NumPy, scikit-learn, Plotly, etc.

**¿Dónde vamos a trabajar?**

En esta materia usamos **Google Colab** — un entorno de notebooks en la nube que no requiere instalar nada. Cada celda puede contener código Python o texto Markdown.

```python
# Mi primer programa en Python
print("Hola, Data Science!")

# Python como calculadora
print(2 + 3)
print(10 / 3)
print(2 ** 8)  # 2 elevado a la 8
```

---

## 2. Variables y Asignación

Una variable es un nombre que apunta a un valor en memoria. En Python no hay que declarar el tipo — se infiere automáticamente.

```python
# Asignación básica
nombre = "María"
edad = 25
altura = 1.75
es_estudiante = True

# Asignación múltiple
x, y, z = 1, 2, 3
a = b = c = 0

# Ver el valor de una variable
print(nombre)
print(type(edad))   # <class 'int'>
```

**Reglas para nombres de variables:**
- Pueden contener letras, números y `_`
- No pueden empezar con número
- No pueden ser palabras reservadas (`if`, `for`, `class`, etc.)
- Se recomienda usar `snake_case`

```python
# Válidos
precio_por_unidad = 15.5
total_ventas_2024 = 100000

# Inválidos
# 2ventas = 100    # empieza con número
# precio-final = 5 # guión no permitido
```

### Expresiones y f-strings

```python
nombre = "Carlos"
edad = 30

# f-string: forma moderna de formatear texto
print(f"Me llamo {nombre} y tengo {edad} años")
print(f"El doble de mi edad es {edad * 2}")
```

### Ejercicio: Variables de un dataset

```python
# Creá variables para describir una acción de bolsa
ticker = "AAPL"
precio_actual = 189.50
variacion_diaria = 2.35
es_ganancia = variacion_diaria > 0

print(f"Acción: {ticker}")
print(f"Precio: ${precio_actual}")
print(f"Variación: {variacion_diaria:+.2f}")
print(f"¿Subió? {es_ganancia}")
```

---

## 3. Tipos de Datos

### Tipos básicos

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| `int` | Número entero | `42`, `-7`, `0` |
| `float` | Número decimal | `3.14`, `-0.5` |
| `str` | Texto (cadena) | `"hola"`, `'mundo'` |
| `bool` | Verdadero/Falso | `True`, `False` |

```python
entero = 42
decimal = 3.14159
texto = "Ciencia de Datos"
booleano = True

print(type(entero))    # <class 'int'>
print(type(decimal))   # <class 'float'>
print(type(texto))     # <class 'str'>
print(type(booleano))  # <class 'bool'>
```

### Conversión de tipos

```python
# str → int / float
numero_texto = "123"
numero = int(numero_texto)      # 123
precio = float("19.99")         # 19.99

# int → str
codigo = str(42)                # "42"

# Cuidado: esto genera error
# int("3.14")  → ValueError
int(float("3.14"))              # primero float, luego int → 3
```

### Tipos compuestos

```python
# Lista: ordenada, mutable
acciones = ["AAPL", "GOOG", "AMZN", "MSFT"]
acciones.append("TSLA")
print(acciones[0])   # AAPL

# Tupla: ordenada, inmutable
coordenadas = (40.7128, -74.0060)

# Diccionario: clave → valor
accion_info = {
    "ticker": "AAPL",
    "precio": 189.50,
    "sector": "Technology"
}
print(accion_info["precio"])  # 189.5

# Set: valores únicos
sectores = {"Technology", "Finance", "Health", "Technology"}
print(sectores)  # {'Technology', 'Finance', 'Health'}
```

---

## 4. Operadores

### Aritméticos

```python
a = 10
b = 3

print(a + b)    # 13   suma
print(a - b)    # 7    resta
print(a * b)    # 30   multiplicación
print(a / b)    # 3.33 división (siempre float)
print(a // b)   # 3    división entera
print(a % b)    # 1    módulo (resto)
print(a ** b)   # 1000 potencia
```

### Comparación y lógicos

```python
x = 15
print(x > 10)          # True
print(x == 15)         # True
print(x != 10)         # True
print(x >= 20)         # False

# Lógicos: and, or, not
print(x > 10 and x < 20)   # True
print(x > 20 or x == 15)   # True
print(not (x == 15))        # False
```

### Operadores de asignación compuesta

```python
precio = 100
precio += 10    # precio = precio + 10 → 110
precio *= 1.21  # aplica IVA
precio -= 5     # descuento
print(round(precio, 2))
```

### Ejercicio: Calcular rentabilidad

```python
precio_compra = 150.0
precio_venta = 189.5
cantidad = 10

ganancia_por_accion = precio_venta - precio_compra
ganancia_total = ganancia_por_accion * cantidad
rentabilidad_pct = (ganancia_por_accion / precio_compra) * 100

print(f"Ganancia por acción: ${ganancia_por_accion:.2f}")
print(f"Ganancia total: ${ganancia_total:.2f}")
print(f"Rentabilidad: {rentabilidad_pct:.1f}%")
```

---

## 5. Estructuras de Control: Condicionales

Los condicionales permiten ejecutar código solo si se cumple una condición.

```
if <condición>:
    # se ejecuta si condición es True
elif <otra condición>:
    # se ejecuta si la primera es False y esta es True
else:
    # se ejecuta si ninguna condición fue True
```

```python
precio = 189.50
precio_referencia = 180.0

if precio > precio_referencia * 1.10:
    recomendacion = "VENDER"
elif precio > precio_referencia:
    recomendacion = "MANTENER"
else:
    recomendacion = "COMPRAR"

print(f"Precio: ${precio} → {recomendacion}")
```

### Condicional anidado

```python
temperatura = 28
humedad = 85

if temperatura > 30:
    if humedad > 70:
        print("Muy bochornoso")
    else:
        print("Caluroso pero tolerable")
elif temperatura > 20:
    print("Agradable")
else:
    print("Frío")
```

### Ejercicio: Clasificar rendimiento de acciones

```python
def clasificar_accion(variacion_pct):
    if variacion_pct >= 5:
        return "Fuerte alza"
    elif variacion_pct >= 1:
        return "Alza moderada"
    elif variacion_pct > -1:
        return "Estable"
    elif variacion_pct > -5:
        return "Baja moderada"
    else:
        return "Fuerte caída"

variaciones = [7.2, 1.8, -0.3, -3.5, -6.1]

for v in variaciones:
    print(f"{v:+.1f}% → {clasificar_accion(v)}")
```

---

## 6. Bucles

### for: iterar sobre una secuencia

```python
acciones = ["AAPL", "GOOG", "AMZN"]

for ticker in acciones:
    print(f"Procesando: {ticker}")

# range(): generar secuencias numéricas
for i in range(5):          # 0, 1, 2, 3, 4
    print(i)

for i in range(1, 6):       # 1, 2, 3, 4, 5
    print(i)

for i in range(0, 10, 2):   # 0, 2, 4, 6, 8
    print(i)
```

### while: repetir mientras se cumpla una condición

```python
intentos = 0
maximo = 3

while intentos < maximo:
    print(f"Intento {intentos + 1}")
    intentos += 1

print("Fin del bucle")
```

### break y continue

```python
precios = [120, 145, 98, 200, 85, 160]

# break: salir del bucle al encontrar precio > 180
for precio in precios:
    if precio > 180:
        print(f"Precio muy alto encontrado: {precio}. Deteniendo.")
        break
    print(f"Precio OK: {precio}")

print("---")

# continue: saltear precios menores a 100
for precio in precios:
    if precio < 100:
        continue   # salta este y sigue al siguiente
    print(f"Precio válido: {precio}")
```

### Ejercicio colaborativo: análisis de precios históricos

```python
precios_historicos = [150, 155, 148, 162, 170, 165, 180, 175, 190, 185]

# Calcular la variación entre días consecutivos
variaciones = []
for i in range(1, len(precios_historicos)):
    variacion = precios_historicos[i] - precios_historicos[i - 1]
    variaciones.append(variacion)

dias_alza = sum(1 for v in variaciones if v > 0)
dias_baja = sum(1 for v in variaciones if v < 0)
max_subida = max(variaciones)
max_caida = min(variaciones)

print(f"Días de alza:  {dias_alza}")
print(f"Días de baja:  {dias_baja}")
print(f"Mayor subida:  +{max_subida}")
print(f"Mayor caída:   {max_caida}")
```

---

## 7. Funciones

Una función es un bloque de código con nombre que podemos reutilizar.

```python
def nombre_funcion(parametro1, parametro2):
    # código
    return resultado
```

```python
# Función básica
def saludar(nombre):
    return f"Hola, {nombre}!"

print(saludar("Ana"))

# Parámetro con valor por defecto
def calcular_iva(precio, tasa=0.21):
    return precio * (1 + tasa)

print(calcular_iva(100))         # 121.0
print(calcular_iva(100, 0.105))  # 110.5
```

### Funciones con múltiples retornos

```python
def estadisticas(numeros):
    return min(numeros), max(numeros), sum(numeros) / len(numeros)

minimo, maximo, promedio = estadisticas([10, 20, 30, 40, 50])
print(f"Min: {minimo}, Max: {maximo}, Promedio: {promedio}")
```

### Ejercicio: función para analizar una acción

```python
def analizar_accion(ticker, precios):
    """Recibe el nombre de una acción y una lista de precios históricos."""
    precio_inicial = precios[0]
    precio_final = precios[-1]
    rentabilidad = (precio_final - precio_inicial) / precio_inicial * 100
    volatilidad = max(precios) - min(precios)

    return {
        "ticker": ticker,
        "precio_inicial": precio_inicial,
        "precio_final": precio_final,
        "rentabilidad_pct": round(rentabilidad, 2),
        "volatilidad": volatilidad
    }

resultado = analizar_accion("AAPL", [150, 155, 162, 170, 189])
for clave, valor in resultado.items():
    print(f"{clave}: {valor}")
```

### *args y **kwargs (parámetros arbitrarios)

```python
def sumar(*numeros):
    return sum(numeros)

print(sumar(1, 2, 3))        # 6
print(sumar(10, 20, 30, 40)) # 100

def mostrar_info(**datos):
    for clave, valor in datos.items():
        print(f"{clave}: {valor}")

mostrar_info(nombre="GOOG", precio=175.3, sector="Tech")
```

---

## 8. NumPy: arreglos y operaciones vectorizadas

NumPy es la librería fundamental para computación numérica en Python. Permite trabajar con arreglos (arrays) de forma mucho más eficiente que con listas.

```python
import numpy as np
```

### Crear arreglos

```python
# Desde una lista
arr = np.array([1, 2, 3, 4, 5])
print(arr)           # [1 2 3 4 5]
print(arr.dtype)     # int64
print(arr.shape)     # (5,)

# Arreglos especiales
np.zeros(5)          # [0. 0. 0. 0. 0.]
np.ones(5)           # [1. 1. 1. 1. 1.]
np.arange(0, 10, 2)  # [0 2 4 6 8]
np.linspace(0, 1, 5) # [0.   0.25 0.5  0.75 1.  ]

# Arreglo 2D (matriz)
matriz = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(matriz.shape)  # (2, 3)
```

### Operaciones vectorizadas

Con NumPy, las operaciones se aplican elemento a elemento sin necesidad de bucles:

```python
precios = np.array([150.0, 155.0, 162.0, 170.0, 189.0])

# Operaciones sobre todo el arreglo
print(precios * 1.21)           # aplicar IVA a todos
print(precios - precios[0])     # diferencia respecto al primero
print(precios > 160)            # máscara booleana

# Estadísticas
print(f"Mínimo:  {precios.min()}")
print(f"Máximo:  {precios.max()}")
print(f"Promedio: {precios.mean():.2f}")
print(f"Desvío:  {precios.std():.2f}")
```

### Indexing y slicing

```python
arr = np.array([10, 20, 30, 40, 50])

print(arr[0])      # 10  (primer elemento)
print(arr[-1])     # 50  (último elemento)
print(arr[1:4])    # [20 30 40]
print(arr[::2])    # [10 30 50]  (cada dos)

# Filtrar con condición
grandes = arr[arr > 25]
print(grandes)     # [30 40 50]
```

---

## 9. Visualización con Plotly Express

Plotly Express es una librería de visualización interactiva. Con pocas líneas de código se crean gráficos publicables.

```python
import plotly.express as px
import pandas as pd
```

### Tipos de gráficos

| Función | Cuándo usarla |
|---------|---------------|
| `px.line()` | Evolución temporal, tendencias |
| `px.bar()` | Comparar categorías |
| `px.scatter()` | Relación entre dos variables numéricas |
| `px.histogram()` | Distribución de una variable |
| `px.box()` | Distribución + outliers por grupo |
| `px.pie()` | Proporciones (usar con moderación) |

### Gráfico de línea: precio histórico

```python
import plotly.express as px
import pandas as pd

fechas = pd.date_range("2024-01-01", periods=10, freq="W")
precios = [150, 155, 148, 162, 170, 165, 180, 175, 190, 185]

df = pd.DataFrame({"fecha": fechas, "precio": precios})

fig = px.line(
    df,
    x="fecha",
    y="precio",
    title="Evolución del precio de AAPL",
    labels={"precio": "Precio (USD)", "fecha": "Fecha"}
)
fig.show()
```

### Gráfico de barras: comparar sectores

```python
df_sectores = pd.DataFrame({
    "sector": ["Technology", "Finance", "Health", "Energy", "Consumer"],
    "rendimiento_pct": [18.5, 7.2, 12.1, -3.4, 5.8]
})

fig = px.bar(
    df_sectores,
    x="sector",
    y="rendimiento_pct",
    color="rendimiento_pct",
    color_continuous_scale="RdYlGn",
    title="Rendimiento por sector (YTD)"
)
fig.show()
```

### Gráfico de dispersión: precio vs volumen

```python
import numpy as np

np.random.seed(42)
n = 50
df_scatter = pd.DataFrame({
    "precio": np.random.uniform(50, 300, n),
    "volumen_millones": np.random.uniform(1, 100, n),
    "sector": np.random.choice(["Tech", "Finance", "Health"], n)
})

fig = px.scatter(
    df_scatter,
    x="precio",
    y="volumen_millones",
    color="sector",
    size="volumen_millones",
    title="Precio vs Volumen por sector",
    hover_data=["sector"]
)
fig.show()
```

### Histograma: distribución de retornos

```python
retornos_diarios = np.random.normal(0.001, 0.02, 252)  # 1 año de trading

fig = px.histogram(
    x=retornos_diarios,
    nbins=30,
    title="Distribución de retornos diarios",
    labels={"x": "Retorno diario"},
    color_discrete_sequence=["steelblue"]
)
fig.add_vline(x=0, line_dash="dash", line_color="red")
fig.show()
```

### Ejercicio colaborativo: dashboard de acciones

```python
import plotly.express as px
import pandas as pd

# Dataset de acciones ficticias
data = {
    "ticker": ["AAPL", "GOOG", "AMZN", "MSFT", "TSLA"] * 4,
    "trimestre": ["Q1", "Q1", "Q1", "Q1", "Q1",
                  "Q2", "Q2", "Q2", "Q2", "Q2",
                  "Q3", "Q3", "Q3", "Q3", "Q3",
                  "Q4", "Q4", "Q4", "Q4", "Q4"],
    "precio_cierre": [175, 140, 180, 370, 250,
                      185, 145, 185, 380, 265,
                      189, 150, 190, 390, 248,
                      195, 155, 200, 410, 260]
}
df = pd.DataFrame(data)

# Gráfico de líneas por empresa
fig = px.line(
    df,
    x="trimestre",
    y="precio_cierre",
    color="ticker",
    markers=True,
    title="Evolución de precios por trimestre"
)
fig.show()

# Promedio anual por empresa
resumen = df.groupby("ticker")["precio_cierre"].mean().reset_index()
resumen.columns = ["ticker", "precio_promedio"]

fig2 = px.bar(
    resumen,
    x="ticker",
    y="precio_promedio",
    color="ticker",
    title="Precio promedio anual por acción"
)
fig2.show()
```

---

## 10. Pandas: primera mirada

Pandas es la librería principal para manipular datos tabulares en Python. Trabaja con **DataFrames** (tablas) y **Series** (columnas).

```python
import pandas as pd
```

### Crear un DataFrame

```python
datos = {
    "ticker":  ["AAPL", "GOOG", "AMZN", "MSFT"],
    "precio":  [189.5,  155.0,  195.2,  410.0],
    "sector":  ["Tech",  "Tech", "Consumer", "Tech"],
    "pe_ratio": [28.5,   24.1,   65.3,    35.2]
}

df = pd.DataFrame(datos)
print(df)
```

### Exploración básica

```python
print(df.head())          # primeras 5 filas
print(df.tail(2))         # últimas 2 filas
print(df.shape)           # (filas, columnas)
print(df.dtypes)          # tipo de cada columna
print(df.describe())      # estadísticas descriptivas
print(df.isnull().sum())  # valores faltantes por columna
```

### Selección de datos

```python
# Seleccionar una columna → Series
print(df["precio"])
print(df["precio"].mean())

# Seleccionar varias columnas
print(df[["ticker", "precio"]])

# Filtrar filas por condición
tech = df[df["sector"] == "Tech"]
baratas = df[df["precio"] < 200]

# Ordenar
df_ordenado = df.sort_values("precio", ascending=False)
print(df_ordenado)
```

---

## Desafío integrador

Combiná todo lo visto para construir un análisis de acciones:

```python
import numpy as np
import pandas as pd
import plotly.express as px

# 1. Generar datos simulados
np.random.seed(42)
tickers = ["AAPL", "GOOG", "AMZN", "MSFT", "TSLA"]
fechas = pd.date_range("2024-01-01", periods=52, freq="W")

filas = []
for ticker in tickers:
    precio_base = np.random.uniform(100, 400)
    precios = precio_base + np.cumsum(np.random.normal(0, 5, 52))
    for i, fecha in enumerate(fechas):
        filas.append({"ticker": ticker, "fecha": fecha, "precio": round(precios[i], 2)})

df = pd.DataFrame(filas)

# 2. Función para calcular rentabilidad
def rentabilidad_accion(df, ticker):
    datos = df[df["ticker"] == ticker].sort_values("fecha")
    p0 = datos["precio"].iloc[0]
    pf = datos["precio"].iloc[-1]
    return round((pf - p0) / p0 * 100, 2)

print("Rentabilidad anual por acción:")
for t in tickers:
    r = rentabilidad_accion(df, t)
    emoji = "↑" if r > 0 else "↓"
    print(f"  {t}: {r:+.1f}% {emoji}")

# 3. Visualización
fig = px.line(
    df,
    x="fecha",
    y="precio",
    color="ticker",
    title="Evolución de precios — 2024"
)
fig.show()
```

---

## Recursos

- [Documentación oficial de Python](https://docs.python.org/3/)
- [NumPy — quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [Plotly Express — galería](https://plotly.com/python/plotly-express/)
- [Pandas — getting started](https://pandas.pydata.org/docs/getting_started/index.html)
- [Google Colab](https://colab.research.google.com/)
