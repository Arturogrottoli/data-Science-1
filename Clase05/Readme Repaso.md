# Readme Repaso — Clase 05: Visualizaciones Avanzadas en Data Science

Guia de clase para el profesor. Cada sección incluye qué decir, qué mostrar y qué ejecutar en el notebook.

---

## Antes de empezar — Encuadre de la clase

> "La clase de hoy tiene dos partes. En la primera hacemos un repaso de todo lo visto hasta ahora: variables, control de flujo, NumPy y Pandas. En la segunda parte arrancamos con el tema nuevo: visualizaciones avanzadas. Al final tienen una actividad práctica con un dataset real de tráfico aéreo."

Abrir el notebook: `Repaso_Data_Science_I_Fundamentos_para_la_Ciencia_de_Datos_.ipynb`

---

## PARTE 1 — Repaso Semanas 1 a 4

---

### Semana 1 — Variables y Tipos de Datos

**Qué decir:**

Una **variable** es un nombre que le ponemos a un espacio en la memoria de la computadora donde guardamos un valor. Cada vez que escribimos `edad = 25`, Python reserva un espacio en la RAM, guarda el número 25, y le cuelga la etiqueta `edad` para poder encontrarlo más tarde.

El **tipo de dato** determina qué cosas podemos hacerle a ese valor. No es lo mismo guardar el número `25` que el texto `"25"`: con el número podemos sumar, restar o comparar; con el texto solo podemos concatenar, buscar, dividir en partes. Python distingue cuatro tipos básicos:

| Tipo | Nombre técnico | Ejemplo | Para qué se usa |
|------|---------------|---------|-----------------|
| Entero | `int` | `25`, `-3`, `1000` | Conteos, edades, cantidades sin decimales |
| Decimal | `float` | `99.99`, `3.14`, `-0.5` | Precios, temperaturas, mediciones |
| Texto | `str` | `"Planta Alta"`, `"2026-01-01"` | Nombres, categorías, fechas como texto |
| Booleano | `bool` | `True`, `False` | Condiciones, flags de estado, activado/desactivado |

Las **f-strings** son la forma moderna de mezclar texto y variables. La `f` antes de las comillas le dice a Python que busque variables entre `{}` y las reemplace con su valor. Es equivalente a la función `format()` pero mucho más legible.

```python
nombre = "Planta Alta"
temperatura = 82.5
# Sin f-string (viejo estilo, más verbose):
print("El sensor " + nombre + " registró " + str(temperatura) + "°C.")
# Con f-string (estilo moderno):
print(f"El sensor {nombre} registró {temperatura}°C.")
```

Los **diccionarios** (`dict`) son la estructura de datos más cercana a cómo el mundo real organiza la información. Un sensor de temperatura no manda solo un número: manda su ID, la fecha y hora, la lectura y el estado. Un diccionario agrupa todo eso bajo un solo nombre usando pares **clave: valor**. Se accede a cada campo con su clave entre corchetes: `evento["temperatura"]`.

**Ejecutar:**
```python
edad = 25
precio = 99.99
nombre = "Planta Alta"
activo = True

print(f"El sensor {nombre} está activo? {activo}")

evento_maquina = {
    "sensor_id": "TERM-04",
    "fecha_hora": "2026-05-17 14:30:00",
    "temperatura": 82.5,
    "estado": "Operando"
}

print(f"El sensor {evento_maquina['sensor_id']} registró {evento_maquina['temperatura']}°C.")
```

**Qué mostrar en detalle:**
- Ejecutar `type(edad)` → `<class 'int'>`
- Ejecutar `type(precio)` → `<class 'float'>`
- Ejecutar `type(nombre)` → `<class 'str'>`
- Mostrar qué pasa si sumamos: `"25" + 5` → `TypeError`. El tipo importa.

**Preguntar a la clase:**

> ¿Por qué usamos un diccionario y no cuatro variables separadas (`sensor_id = "TERM-04"`, `fecha_hora = "2026-05-17"`, etc.)?

**Respuesta:** Porque agrupa toda la información de un solo evento bajo un único nombre. Si el sensor manda 1.000 registros por día, manejamos una lista de 1.000 diccionarios, no 4.000 variables sueltas flotando en el código. Además, si el día de mañana el sensor agrega un campo nuevo (por ejemplo `"humedad"`), solo hay que agregar una clave más al diccionario sin tocar el resto del código.

> ¿Qué diferencia hay entre `temperatura = 0` y `temperatura = None`?

**Respuesta:** `0` es un valor válido que dice "la temperatura medida es cero grados". `None` (o `np.nan` en NumPy) significa que no hay ningún dato: el sensor falló, no hubo medición, el campo está vacío. Son semánticamente muy distintos: un promedio de temperaturas que incluye un `0` se va a ver afectado; uno que ignora los `None` no.

---

### Semana 2 — Control de Flujo: `if/else` y `for`

**Qué decir:**

Hasta la Semana 1, los scripts ejecutaban cada línea de arriba hacia abajo, en orden, sin saltear nada ni repetir nada. El **control de flujo** le agrega inteligencia: el programa puede tomar caminos distintos según los datos, y puede repetir tareas automáticamente.

**El bloque `if/elif/else`** evalúa una condición (algo que da `True` o `False`) y decide qué código ejecutar. La condición puede usar los operadores de comparación `>`, `<`, `>=`, `<=`, `==` (igual), `!=` (distinto), y los operadores lógicos `and`, `or`, `not` para combinar condiciones.

```python
temperatura = 91.0

if temperatura > 90:
    print("PELIGRO CRÍTICO: apagar máquina")
elif temperatura > 80:
    print("ALERTA: temperatura elevada")
elif temperatura > 70:
    print("PRECAUCIÓN: monitorear")
else:
    print("NORMAL: todo bien")
```

Python evalúa cada condición de arriba hacia abajo y ejecuta el primer bloque que se cumpla. Si ninguna se cumple, ejecuta el `else`. Si no hay `else`, simplemente no hace nada.

**El bucle `for`** itera sobre cualquier colección de elementos (lista, rango, tupla, string, etc.) y ejecuta el mismo bloque de código para cada uno. Sin `for`, si queremos procesar 1.000 temperaturas, necesitaríamos escribir 1.000 líneas de código. Con `for`, escribimos el procesamiento una sola vez.

```python
# Sin for: inmanejable
print(historial[0] > 80)
print(historial[1] > 80)
# ... 998 líneas más

# Con for: escalable a cualquier cantidad de datos
for temp in historial:
    print(temp > 80)
```

La función `range(inicio, fin, paso)` genera una secuencia de números. Se usa cuando necesitamos iterar una cantidad fija de veces en lugar de sobre una lista existente:

```python
for i in range(5):        # 0, 1, 2, 3, 4
    print(i)

for i in range(0, 10, 2): # 0, 2, 4, 6, 8
    print(i)
```

El **patrón lista vacía + for + append** es uno de los más usados en data science: creamos una lista vacía, iteramos sobre los datos originales, y agregamos a la lista solo los que cumplen algún criterio.

**Ejecutar:**
```python
historial_temps = [72.0, 85.3, 91.0, 68.4, 88.9]
temps_altas = []

for temp in historial_temps:
    if temp > 80.0:
        temps_altas.append(temp)
    # else: pass  ← el else es opcional si no hay nada que hacer

print(f"Todas las mediciones: {historial_temps}")
print(f"Mediciones peligrosas (>80): {temps_altas}")
```

**Qué mostrar en detalle:**
- Agregar un `print(f"Evaluando {temp}...")` dentro del for para que vean cómo itera paso a paso.
- Mostrar que `append()` modifica la lista en el lugar: `temps_altas` crece con cada iteración.
- Mostrar la diferencia entre `=` (asignación) y `==` (comparación): `temp = 80` guarda el valor, `temp == 80` pregunta si son iguales.

**Preguntar a la clase:**

> ¿Qué pasa si sacamos el `else: pass`?

**Respuesta:** Absolutamente nada cambia. `pass` es una instrucción vacía que solo existe para que Python no tire error de sintaxis cuando un bloque debería tener al menos una línea. Si el bloque `else` no tiene código útil, se puede omitir directamente. En este ejemplo, cuando `temp <= 80.0` simplemente no hacemos nada, y eso es válido sin escribir `else: pass`.

> ¿Cómo modificarían el código para filtrar las temperaturas menores a 70?

**Respuesta:** Cambiar la condición de `temp > 80.0` por `temp < 70.0`. También habría que renombrar la lista `temps_altas` por algo como `temps_bajas` para que el nombre tenga sentido.

> ¿Qué pasa si la lista `historial_temps` está vacía?

**Respuesta:** El bucle `for` simplemente no ejecuta ninguna iteración. No da error. `temps_altas` queda como lista vacía `[]`. Esto es importante en producción: siempre hay que pensar qué pasa cuando no hay datos.

---

---

### Concepto previo — ¿Qué es una librería y por qué las usamos?

Antes de ver NumPy y Pandas, hay que entender qué es una **librería** y por qué existe.

**Analogía cotidiana:** Imaginate que te mudás a un departamento nuevo y necesitás un taladro. Tenés dos opciones:
1. Fabricar el taladro vos mismo: comprar el motor, las piezas, armarlo desde cero.
2. Ir al ferretería y comprarlo ya hecho.

Una **librería** es la ferretería. Alguien ya escribió ese código, lo probó, lo optimizó durante años, y lo empaquetó para que nosotros lo podamos usar con una sola línea: `import numpy`. No tenemos que saber cómo está construido por dentro; solo tenemos que saber cómo usarlo.

La línea `import numpy as np` hace dos cosas:
1. Le dice a Python "buscá el paquete llamado numpy que ya está instalado".
2. Le da el apodo `np` para no tener que escribir `numpy.` completo cada vez.

Sin esa línea, todas las funciones de NumPy son inaccesibles, como si el taladro estuviera en el depósito con llave. Con esa línea, todo el poder de NumPy queda disponible en el script.

Python solo ya trae muchas cosas útiles (`len`, `range`, `print`, `sorted`, etc.). Pero para matemática científica, tablas de datos, gráficos y machine learning, esas herramientas built-in no alcanzan. Las librerías extienden Python con funcionalidades especializadas que la comunidad científica fue construyendo durante décadas.

---

### Semana 3 — NumPy: Cálculos en Bloque

**Qué decir:**

**¿Por qué no alcanza Python puro para matemática con datos?**

Pensemos en un caso concreto: tenemos el sueldo de 50.000 empleados en una lista y queremos aplicarles un aumento del 12%. Con Python puro:

```python
sueldos = [45000, 62000, 38000, ...]  # 50.000 valores

sueldos_nuevos = []
for s in sueldos:
    sueldos_nuevos.append(s * 1.12)
```

Esto funciona, pero Python tiene que:
1. Leer la variable del bucle `s`.
2. Buscar el valor `1.12` en memoria.
3. Invocar la operación de multiplicación del intérprete de Python.
4. Crear un objeto Python nuevo con el resultado.
5. Llamar a `.append()` para agregarlo a la lista.
6. Repetir todo esto 50.000 veces.

Son al menos 6 pasos por elemento. Para 50.000 empleados, eso es 300.000 operaciones de overhead solo del intérprete.

**Analogía cotidiana:** Es como si en una fábrica de empanadas hubiera un solo operario que agarra una empanada, la rellena, la cierra, la lleva a la bandeja, vuelve, agarra la siguiente... una por una. Tarda horas.

NumPy es la línea de ensamblaje automatizada: las 50.000 empanadas pasan por la máquina al mismo tiempo. Un solo clic, resultado instantáneo.

```python
import numpy as np

sueldos = np.array([45000, 62000, 38000, ...])  # 50.000 valores
sueldos_nuevos = sueldos * 1.12  # una sola operación, 50.000 resultados
```

Internamente, NumPy pasa el array a rutinas escritas en **C compilado** (un lenguaje de bajísimo nivel que la CPU ejecuta directamente sin intérprete). El tiempo pasa de segundos a milisegundos.

Python puro es un lenguaje interpretado: ejecuta las instrucciones una por una, en tiempo real, con mucho overhead. Cuando queremos hacer la misma operación matemática sobre millones de datos, esto se vuelve lento.

**NumPy** (Numerical Python) soluciona esto con el **array**: una estructura de datos que almacena elementos del mismo tipo en bloques contiguos de memoria, y cuyas operaciones están implementadas en **C compilado**. El resultado es que NumPy puede ser entre 10 y 1.000 veces más rápido que un bucle `for` de Python para operaciones matemáticas.

La diferencia fundamental entre una lista de Python y un array de NumPy:

| Característica | Lista Python | Array NumPy |
|----------------|-------------|-------------|
| Tipos de datos | Puede mezclar int, str, float | Solo un tipo por array |
| Velocidad | Lenta (Python puro) | Muy rápida (C compilado) |
| Operaciones matemáticas | Una por una con `for` | Sobre todo el array a la vez |
| Memoria | Dispersa en RAM | Contigua y eficiente |

**Otro ejemplo cotidiano para explicar la diferencia:** Tenés que calcular el promedio de temperatura de 1 millón de sensores industriales. Con Python puro necesitás un `for` que suma cada valor uno por uno. Con NumPy escribís `datos.mean()` y el resultado sale en microsegundos porque NumPy usa instrucciones especiales de la CPU (SIMD) que calculan múltiples sumas en paralelo a nivel de hardware.

**Vectorización:** aplicar una operación a todo el array de una sola vez, sin bucles. `array * 1000` multiplica cada elemento por 1000 internamente en paralelo.

```python
import numpy as np

# Con lista Python (necesita for):
precios = [10.0, 25.5, 100.0, 5.25]
precios_ars_lista = []
for p in precios:
    precios_ars_lista.append(p * 1000)

# Con NumPy (vectorización, una línea):
precios_np = np.array([10.0, 25.5, 100.0, 5.25])
precios_ars_np = precios_np * 1000   # ← una operación, todos los elementos
```

**Funciones estadísticas** que NumPy calcula sobre el array completo de una sola vez:

```python
datos = np.array([72.0, 85.3, 91.0, 68.4, 88.9])

print(f"Suma:     {datos.sum()}")
print(f"Promedio: {datos.mean():.2f}")
print(f"Mínimo:   {datos.min()}")
print(f"Máximo:   {datos.max()}")
print(f"Desvío:   {datos.std():.2f}")
```

**Máscaras booleanas:** cuando escribimos `array > 80`, NumPy evalúa la condición para cada elemento y devuelve un nuevo array de `True`/`False`. Si usamos ese array de `True`/`False` como índice, obtenemos solo los elementos donde la condición fue verdadera. Todo sin escribir un `for`.

```python
temps = np.array([72.0, 85.3, 91.0, 68.4, 88.9])

mascara = temps > 80.0
print(mascara)               # [False  True  True False  True]

temps_altas = temps[mascara] # selecciona solo los True
print(temps_altas)           # [85.3 91.  88.9]

# Se puede hacer en una sola línea:
temps_altas = temps[temps > 80.0]
```

**Ejecutar:**
```python
import numpy as np

precios_usd = np.array([10.0, 25.5, 100.0, 5.25])
tipo_cambio = 1000

precios_ars = precios_usd * tipo_cambio
print(f"En USD:  {precios_usd}")
print(f"En ARS:  {precios_ars}")

print(f"\nPromedio USD: {precios_usd.mean():.2f}")
print(f"Máximo USD:   {precios_usd.max():.2f}")

precios_caros = precios_usd[precios_usd > 20.0]
print(f"\nMás de 20 USD: {precios_caros}")
```

**Qué mostrar en detalle:**
- Ejecutar `type(precios_usd)` → `<class 'numpy.ndarray'>`
- Ejecutar `precios_usd.dtype` → `float64` (NumPy eligió el tipo automáticamente)
- Mostrar `precios_usd > 20.0` antes de usarlo como índice para que vean el array de booleanos.
- Ejecutar `precios_usd.shape` → `(4,)` (un array 1D con 4 elementos).

**Preguntar a la clase:**

> ¿Por qué NumPy es más rápido que un bucle `for` de Python para operaciones matemáticas?

**Respuesta:** Porque las operaciones de NumPy están escritas en C compilado y se ejecutan directamente en la CPU sin el overhead del intérprete de Python. Además, los arrays almacenan los datos de forma contigua en memoria, lo que permite a la CPU cargarlos de forma eficiente en caché. Un bucle `for` de Python tiene que interpretar cada línea, crear objetos nuevos, manejar la memoria, etc. Para un millón de elementos, la diferencia puede ser de segundos vs. milisegundos.

> ¿Qué pasa si intentamos crear un array con tipos mezclados: `np.array([1, 2.5, "tres"])`?

**Respuesta:** NumPy va a convertir todos los elementos al tipo más general que los pueda representar. En este caso, va a convertir todo a `str` (`'<U32'` o similar), porque no puede representar texto como número, pero sí puede representar números como texto. El array resultante no sirve para operaciones matemáticas. NumPy no tira error, simplemente hace la conversión, lo cual puede generar bugs silenciosos.

---

### Semana 4 — Pandas: El Excel de Python

**Qué decir:**

**¿Por qué no alcanza Python puro (ni NumPy solo) para trabajar con tablas de datos?**

Imaginemos que tenemos una tabla de ventas con tres columnas: Producto (texto), Unidades (número), Precio (número). En Python puro, la única forma de representar esto es con una lista de listas o un diccionario de listas:

```python
# Python puro: funciona, pero es engorroso
datos = {
    "Producto":  ["Teclado", "Mouse", "Monitor"],
    "Unidades":  [10, 15, 5],
    "Precio":    [50, 25, 300]
}

# Para calcular el total de ventas de cada producto:
totales = []
for i in range(len(datos["Producto"])):
    totales.append(datos["Unidades"][i] * datos["Precio"][i])
print(totales)  # [500, 375, 1500]

# Para filtrar solo los productos con más de 8 unidades:
filtrados = []
for i in range(len(datos["Producto"])):
    if datos["Unidades"][i] > 8:
        filtrados.append(datos["Producto"][i])
print(filtrados)  # ["Teclado", "Mouse"]
```

Esto funciona, pero tiene problemas serios:
- Para cada operación hay que escribir un bucle completo.
- Si las columnas tienen distinto largo, todo explota silenciosamente.
- No hay forma simple de agrupar por categoría, de ordenar, de unir con otra tabla.
- No hay ningún manejo automático de valores faltantes.
- Si el dataset tiene 500 columnas y 1 millón de filas, este enfoque se vuelve imposible de manejar.

**NumPy tampoco alcanza**, porque un array de NumPy solo puede tener un tipo de dato. Si mezclamos texto (`"Teclado"`) con números (`50`), NumPy convierte todo a texto y perdemos la capacidad de hacer matemática. Las tablas reales siempre mezclan tipos.

**Analogía cotidiana:** Python puro para tablas es como llevar la contabilidad de una empresa en un cuaderno rayado: técnicamente funciona, pero es lento, propenso a errores y no escala. NumPy solo es como tener una calculadora científica muy rápida pero sin hojas de cálculo. **Pandas es Excel automatizable**: tiene las filas, las columnas, los tipos, el filtrado, el ordenamiento, el groupby... y además se puede programar y combinar con modelos de machine learning.

Con Pandas, lo que antes era un bucle de 6 líneas se convierte en una sola expresión:

```python
import pandas as pd

df = pd.DataFrame(datos)

# Columna calculada (sin for):
df["Total"] = df["Unidades"] * df["Precio"]

# Filtrado (sin for):
df_filtrado = df[df["Unidades"] > 8]

# Agrupación y suma (equivalente a GROUP BY de SQL):
df.groupby("Producto")["Total"].sum()
```

Pandas toma los arrays de NumPy y les agrega dos cosas fundamentales: **etiquetas en filas y columnas** (el índice y los nombres de columna), y la capacidad de manejar columnas con **tipos distintos** en la misma tabla. El resultado es el **DataFrame**: la estructura de datos central de todo el análisis de datos en Python.

Un DataFrame tiene:
- **Filas**: cada fila es una observación (un registro, un evento, un cliente, una transacción).
- **Columnas**: cada columna es una variable o característica (el nombre, la edad, el precio, la fecha).
- **Índice**: la etiqueta de cada fila. Por defecto es 0, 1, 2, 3... pero puede ser cualquier cosa (fechas, IDs, etc.).
- **dtype**: cada columna tiene su propio tipo de dato (int, float, object, datetime, etc.).

Las operaciones más comunes al trabajar con un DataFrame real:

```python
df.head()        # primeras 5 filas (para ver cómo son los datos)
df.tail()        # últimas 5 filas
df.shape         # (filas, columnas) — cuánto pesa el dataset
df.info()        # tipos de dato por columna y conteo de no-nulos
df.describe()    # estadísticas descriptivas de las columnas numéricas
df.isnull().sum()  # cuenta cuántos valores faltantes hay por columna
```

**`np.nan` vs `0` vs `None`:** `np.nan` es el valor especial de NumPy/Pandas para representar un dato ausente. Es diferente a cero (que es un valor válido). Pandas lo excluye automáticamente de los cálculos estadísticos (`.mean()`, `.sum()`, etc.), lo que es el comportamiento correcto. Siempre hay que saber cuántos NaN tiene cada columna antes de analizar.

**`fillna(valor)`:** rellena los NaN con el valor indicado. Estrategias comunes:
- `fillna(0)` → cuando el NaN significa "cero" (sin actividad, sin venta).
- `fillna(df["col"].mean())` → cuando queremos no distorsionar el promedio.
- `fillna(method="ffill")` → en series temporales: propaga el último valor conocido.

**Columnas calculadas:** se crea una nueva columna asignando el resultado de una operación sobre columnas existentes. Como NumPy por debajo, la operación se aplica fila por fila automáticamente sin `for`.

**`groupby`:** agrupa las filas que comparten el mismo valor en una columna, y luego aplica una función de agregación (sum, mean, count, max, etc.) a cada grupo. Es la operación más poderosa de Pandas para obtener resúmenes.

```python
# groupby en acción:
# 1. Agrupa las filas por la columna "Producto"
# 2. Para cada grupo, toma la columna "Total_Venta"
# 3. Suma todos los valores de ese grupo
df.groupby("Producto")["Total_Venta"].sum()
```

**Ejecutar:**
```python
import pandas as pd
import numpy as np

datos_ventas = {
    "Producto": ["Teclado", "Mouse", "Teclado", "Monitor", "Mouse"],
    "Unidades": [10, 15, np.nan, 5, 20],
    "Precio_Unitario": [50, 25, 50, 300, 25]
}

df = pd.DataFrame(datos_ventas)
print("Tabla original:")
print(df)
print(f"\nShape: {df.shape}")
print(f"\nNaN por columna:\n{df.isnull().sum()}")

df["Unidades"] = df["Unidades"].fillna(0)
df["Total_Venta"] = df["Unidades"] * df["Precio_Unitario"]
reporte = df.groupby("Producto")["Total_Venta"].sum()

print("\nTabla final:")
print(df)
print("\nTotal por producto:")
print(reporte)
```

**Qué mostrar en detalle:**
- Ejecutar `df.dtypes` antes y después del fillna para ver los tipos.
- Ejecutar `df.info()` para ver el resumen completo de la tabla.
- Mostrar `df.describe()` para ver las estadísticas descriptivas rápidas.
- Mostrar qué pasa sin `fillna`: `df["Unidades"] * df["Precio_Unitario"]` con NaN → la fila entera se convierte en NaN.

**Preguntar a la clase:**

> ¿Qué pasaría si en vez de `fillna(0)` usáramos `fillna(df["Unidades"].mean())`?

**Respuesta:** Completaríamos el NaN con el promedio de las otras unidades. En este caso, la media de `[10, 15, 5, 20]` es 12.5, así que la fila de Teclado faltante quedaría con 12.5 unidades. Esta estrategia es mejor cuando no tenemos razón para creer que el NaN significa "cero": simplemente no se registró el dato, pero el valor real probablemente estuvo cerca del promedio.

> ¿Qué diferencia hay entre `df["Producto"]` y `df[["Producto"]]`?

**Respuesta:** `df["Producto"]` (un solo par de corchetes) devuelve una **Serie** de Pandas: una columna individual, como un array con etiquetas. `df[["Producto"]]` (doble par de corchetes) devuelve un **DataFrame** con una sola columna. La diferencia importa cuando encadenamos operaciones: algunas funciones solo aceptan DataFrames, otras solo aceptan Series.

> ¿Qué hace `groupby` internamente?

**Respuesta:** Pandas recorre el DataFrame y separa las filas en grupos según el valor de la columna indicada. Luego aplica la función de agregación (`.sum()`, `.mean()`, etc.) dentro de cada grupo por separado y devuelve un resultado por grupo. Es el equivalente a un `GROUP BY` de SQL. Internamente usa hashing para asignar cada fila a su grupo de forma muy eficiente.

---

## PARTE 2 — Clase 05: Visualizaciones Avanzadas

Abrir el notebook: `Semana_5_Visualizaciones_Avanzadas_en_Data_Science_.ipynb`

---

## Bloque 1 — Arquitectura de Figuras y Diseño Avanzado

---

### Tema 1 — La Jerarquía de Matplotlib: Figure y Axes

**Qué decir:**
- Matplotlib funciona como un artista. Hay dos objetos principales:
  - **Figure**: el lienzo completo, el cuadro físico colgado en la pared.
  - **Axes**: la pintura individual dibujada sobre ese lienzo.
- Una Figure puede tener uno o muchos Axes (subgráficos).
- `plt.subplots(nrows, ncols)` devuelve el lienzo `fig` y la lista de Axes `axes`.

**Ejecutar:**
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ax1 = axes[0]
ax2 = axes[1]

ax1.plot([1, 2, 3], [10, 20, 30], color="blue", marker="o")
ax1.set_title("Pintura 1 (izquierda)")

ax2.bar(["A", "B", "C"], [5, 15, 10], color="orange")
ax2.set_title("Pintura 2 (derecha)")

fig.suptitle("Un solo lienzo con dos gráficos", fontsize=14)
plt.show()
```

**Qué remarcar:**
- Cuando tienen un solo subplot pueden usar `plt.plot()` directamente. Cuando tienen varios, usan `ax1.plot()`, `ax2.bar()`, etc. para controlar en cuál dibujan.

---

### Tema 2 — Seaborn: Axes-level vs Figure-level

**Qué decir:**
- Seaborn tiene dos tipos de funciones:
  - **Axes-level** (`sns.scatterplot`, `sns.boxplot`, `sns.histplot`): obedecen al Axes que el programador creó. Se les pasa `ax=mi_eje`.
  - **Figure-level** (`sns.displot`, `sns.relplot`, `sns.catplot`): crean su propio lienzo completo y lo administran solas. Devuelven un `FacetGrid`, no un Axes.
- La regla práctica: si querés integrar el gráfico con otros en el mismo subplot, usá Axes-level.

**Ejecutar:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")

# Axes-level: obedece al ax que creamos
fig, mi_eje = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=tips, x="total_bill", y="tip", ax=mi_eje, color="purple")
mi_eje.set_title("Axes-level: yo controlo el eje")
plt.show()

# Figure-level: hace lo suyo sola
g = sns.displot(data=tips, x="total_bill", col="time", kind="kde", fill=True)
plt.show()
```

---

### Tema 3 — GridSpec: Layouts Personalizados

**Qué decir:**
- Por defecto, `plt.subplots()` divide la figura en partes iguales.
- **GridSpec** permite romper esa simetría: que un gráfico ocupe el doble de espacio que otro.
- Útil cuando tenés un gráfico principal grande y uno secundario de resumen pequeño.

**Ejecutar:**
```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(8, 6))

# 2 filas, 1 columna. La segunda fila es 3 veces más alta que la primera.
gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 3], hspace=0.4)

ax_superior = fig.add_subplot(gs[0])
ax_inferior = fig.add_subplot(gs[1])

ax_superior.plot([1, 2, 4], [5, 4, 3], color="red")
ax_superior.set_title("Resumen (ratio 1)")

ax_inferior.scatter([1, 2, 3], [10, 40, 20], color="green", s=100)
ax_inferior.set_title("Detalle principal (ratio 3)")

plt.show()
```

---

## Bloque 2 — Series Temporales y Análisis Multivariado

---

### Tema 4 — Series Temporales: Por Qué Importa el Tipo `datetime`

**Qué decir:**
- Cuando pandas lee una columna de fechas desde un CSV, la lee como texto (`object`) por defecto.
- Si graficamos fechas como texto, el eje X las ordena **alfabéticamente** (Abril, Agosto, Diciembre, Enero...) y destruye la cronología.
- `pd.to_datetime()` convierte el texto en un tipo especial que pandas y matplotlib entienden como tiempo real.

**Ejecutar:**
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {'Año': ['2025', '2023', '2026', '2024'], 'Ventas': [50, 30, 60, 40]}
df = pd.DataFrame(data)

# INCORRECTO: orden de la tabla, no cronológico
plt.figure(figsize=(5, 3))
sns.lineplot(data=df, x="Año", y="Ventas", marker="o")
plt.title("Orden de tabla (INCORRECTO)")
plt.show()

# CORRECTO: convertimos a datetime
df['Año_Correcto'] = pd.to_datetime(df['Año'], format='%Y')
plt.figure(figsize=(5, 3))
sns.lineplot(data=df, x="Año_Correcto", y="Ventas", marker="o")
plt.title("Cronología correcta con datetime")
plt.show()
```

**Tip de profesor:** Mostrar ambos gráficos lado a lado para que el contraste sea visible de golpe.

---

### Tema 5 — Formatear el Eje de Fechas con `matplotlib.dates`

**Qué decir:**
- Cuando el eje X tiene fechas completas (año, mes, día), matplotlib a veces muestra el formato completo y queda ilegible.
- `mdates.DateFormatter('%Y')` le dice que solo muestre el año.
- `mdates.YearLocator()` le dice que ponga una marca por año.

**Ejecutar:**
```python
import matplotlib.dates as mdates

plt.figure(figsize=(10, 3))
sns.lineplot(data=df, x="Año_Correcto", y="Ventas", marker="o")

ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())

plt.title("Solo el año en el eje X")
plt.show()
```

---

### Tema 6 — Análisis Multivariado: Canales Estéticos

**Qué decir:**
- La pantalla es plana: solo tiene X e Y (2 dimensiones).
- Para analizar 3, 4 o más variables al mismo tiempo, mapeamos las variables extra a **canales estéticos**:
  - `hue` → color
  - `style` → forma del marcador o trazo
  - `size` → tamaño del punto
- Con estos tres canales podemos mostrar hasta 5 variables en un solo gráfico 2D.

**Ejecutar:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=tips,
    x="total_bill",  # Variable 1
    y="tip",         # Variable 2
    hue="smoker",    # Variable 3 (color)
    size="size",     # Variable 4 (tamaño)
    sizes=(20, 200)
)
plt.title("4 variables en 2 dimensiones")
plt.show()
```

---

### Tema 7 — Accesibilidad Cromática

**Qué decir:**
- Nunca confíen toda la información al color. Dos razones principales:
  1. **Daltonismo**: el 8% de los hombres no distingue rojo de verde.
  2. **Impresión en blanco y negro**: en un reporte impreso el color desaparece.
- La regla de oro: si cambiás el color, cambiá también la **forma** (`style`, `markers`).
- La paleta `colorblind` de Seaborn está diseñada para ser distinguible con daltonismo.

**Ejecutar:**
```python
flights = sns.load_dataset("flights")
df_meses = flights[flights['month'].isin(['Jan', 'Jul', 'Dec'])]
df_meses['month'] = df_meses['month'].cat.remove_unused_categories()

plt.figure(figsize=(8, 4))
sns.lineplot(
    data=df_meses,
    x="year",
    y="passengers",
    hue="month",
    style="month",       # forma distinta por mes
    markers=True,        # marcadores geométricos distintos
    palette="colorblind" # paleta apta para daltonismo
)
plt.title("Accesible: color + trazo + marcador")
plt.show()
```

**Preguntar a la clase:**
- ¿Por qué no alcanza con `hue` solo? — Porque si alguien imprime en blanco y negro o tiene daltonismo, todas las líneas se ven iguales.

---

## Bloque 3 — Interactividad y Reportes de Alta Calidad

---

### Tema 8 — Plotly Express: Gráficos Interactivos

**Qué decir:**
- Matplotlib y Seaborn generan imágenes estáticas: una foto del dato.
- **Plotly Express** genera gráficos basados en HTML y JavaScript.
- El resultado se puede abrir en el navegador y permite:
  - Hacer **zoom** en cualquier zona
  - Ver **tooltips** con el valor exacto al pasar el cursor
  - **Aislar** una categoría haciendo clic en la leyenda
- Mismo código, experiencia completamente distinta para el usuario final.

**Ejecutar:**
```python
import plotly.express as px
import seaborn as sns

penguins = sns.load_dataset("penguins")

fig = px.scatter(
    penguins,
    x="flipper_length_mm",
    y="body_mass_g",
    color="species",
    title="Interactivo: hover sobre los puntos, clic en la leyenda"
)
fig.show()
```

**Qué demostrar en vivo:**
1. Pasar el cursor sobre un punto → tooltip con valores exactos.
2. Clic en una especie en la leyenda → se aísla ese grupo.
3. Seleccionar una zona con el mouse → zoom automático.

---

### Tema 9 — Exportación: Rasterizado vs Vectorial

**Qué decir:**
- Cuando guardamos un gráfico, el formato define su comportamiento:
  - **Rasterizado** (`.png`, `.jpg`): guardamos píxeles. Si el `dpi` es bajo, se ve borroso al ampliar. Para presentaciones y web siempre usar `dpi=300`.
  - **Vectorial** (`.pdf`, `.svg`): guardamos la fórmula matemática de cada línea. Resolución infinita: podría imprimirse en un cartel de ruta sin pixelarse.
- Regla práctica: `.png` para PowerPoint y web, `.pdf` para documentos impresos o publicaciones.

**Ejecutar:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

penguins = sns.load_dataset("penguins")
sns.scatterplot(data=penguins, x="flipper_length_mm", y="body_mass_g")
plt.title("Gráfico para exportar")

# Rasterizado a alta resolución
plt.savefig("grafico_alta_resolucion.png", dpi=300)

# Vectorial
plt.savefig("grafico_vectorial.pdf")

plt.close()
print("Archivos guardados.")
```

---

### Tema 10 — `bbox_inches='tight'`: El Encuadre Perfecto

**Qué decir:**
- Por defecto, `savefig()` calcula el tamaño de la imagen basándose en el lienzo, pero "se olvida" de medir textos decorativos como títulos largos o etiquetas rotadas.
- El resultado: texto cortado en la imagen guardada, aunque se vea bien en pantalla.
- `bbox_inches='tight'` le ordena recalcular el encuadre incluyendo absolutamente todo el contenido.
- Es un parámetro barato que siempre conviene agregar.

**Ejecutar:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

penguins = sns.load_dataset("penguins")
sns.histplot(data=penguins, x="flipper_length_mm", color="teal")
plt.title("Este título ridículamente largo se suele cortar al exportar", fontsize=14, pad=20)
plt.xlabel("Largo de la Aleta (mm)", fontsize=14, labelpad=20)

# Sin tight: puede cortar
plt.savefig("grafico_mal_encuadrado.png")

# Con tight: perfecto
plt.savefig("grafico_perfecto.png", bbox_inches='tight')

plt.close()
print("Compará ambos archivos.")
```

**Qué demostrar:** Abrir ambas imágenes y comparar el corte del título.

---

## ACTIVIDAD PRÁCTICA — Tráfico Aéreo

Dataset: `vuelos_asientos_pasajeros.csv`

Los tres pasos de la actividad combinan todo lo visto en la clase:

| Paso | Tema que aplica |
|------|----------------|
| 1 — Histograma de vuelos diarios | Bloque 1: Figure + Axes + Axes-level |
| 2 — Evolución de pasajeros 2023 | Bloque 2: datetime + multivariado + accesibilidad |
| 3 — Interactivo + exportación PNG | Bloque 3: Plotly + savefig 300dpi + bbox_inches |

### Paso 1 — Distribución de vuelos

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")
df_vuelos = pd.read_csv("vuelos_asientos_pasajeros.csv")

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(data=df_vuelos, x="vuelos", ax=ax, bins=30, color="#4682B4", kde=True)
ax.set_title("Distribución de Vuelos Diarios", fontsize=12, fontweight='bold')
ax.set_xlabel("Cantidad de vuelos por día")
ax.set_ylabel("Frecuencia (días)")
plt.show()
```

### Paso 2 — Evolución temporal con accesibilidad

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_vuelos = pd.read_csv("vuelos_asientos_pasajeros.csv")
df_vuelos['indice_tiempo'] = pd.to_datetime(df_vuelos['indice_tiempo'])
df_2023 = df_vuelos[df_vuelos['indice_tiempo'].dt.year == 2023]

plt.figure(figsize=(12, 5))
sns.lineplot(
    data=df_2023,
    x="indice_tiempo",
    y="pasajeros",
    hue="clasificacion_vuelo",
    style="clasificacion_vuelo",
    markers=True,
    palette="colorblind"
)
plt.title("Evolución de Pasajeros 2023 — Cabotaje vs Internacional")
plt.xlabel("Fecha")
plt.ylabel("Pasajeros")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
```

### Paso 3 — Interactivo Plotly + exportación

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

df_vuelos = pd.read_csv("vuelos_asientos_pasajeros.csv")

# Gráfico interactivo
fig_interactiva = px.scatter(
    df_vuelos,
    x="asientos",
    y="pasajeros",
    color="clasificacion_vuelo",
    size="vuelos",
    title="Interactivo: Asientos vs Pasajeros",
    template="plotly_white"
)
fig_interactiva.show()

# Exportación de alta calidad
plt.figure(figsize=(7, 4.5))
sns.scatterplot(data=df_vuelos, x="asientos", y="pasajeros",
                hue="clasificacion_vuelo", size="vuelos")
plt.title("Ocupación: Asientos vs Pasajeros")
plt.savefig("reporte_vuelos_300dpi.png", dpi=300, bbox_inches='tight')
plt.close()
print("Exportado como PNG 300dpi.")
```

---

## Resumen de Conceptos Clave

| Concepto | Para qué sirve |
|----------|---------------|
| `Figure` / `Axes` | Jerarquía de Matplotlib: lienzo y gráfico |
| `plt.subplots(nrows, ncols)` | Crear grilla de gráficos |
| `GridSpec(height_ratios)` | Subgráficos con proporciones distintas |
| Axes-level vs Figure-level | Cuándo Seaborn obedece vs cuándo se adueña |
| `pd.to_datetime()` | Convertir texto a fecha para ordenamiento cronológico |
| `mdates.DateFormatter` | Formatear etiquetas del eje de fechas |
| `hue`, `style`, `size` | Canales estéticos para análisis multivariado |
| `palette="colorblind"` | Paleta accesible para daltonismo |
| `markers=True` | Respaldo de formas para impresión B&N |
| `plotly.express` | Gráficos interactivos (hover, zoom, leyenda) |
| `dpi=300` | Alta resolución para exportar PNG |
| `.pdf` / `.svg` | Exportación vectorial con resolución infinita |
| `bbox_inches='tight'` | Encuadre que no corta títulos ni etiquetas |

---

## Errores comunes que suelen aparecer

**"El eje X de mi gráfico temporal está desordenado"**
→ La columna de fechas sigue siendo texto. Aplicar `pd.to_datetime()` antes de graficar.

**"Mi histograma de Seaborn no aparece en el subplot que yo creé"**
→ Están usando una función Figure-level (`sns.displot`) en vez de Axes-level (`sns.histplot`). Cambiar la función y agregar `ax=ax`.

**"El título se corta en la imagen guardada"**
→ Falta `bbox_inches='tight'` en `plt.savefig()`.

**"Los gráficos de Plotly no se muestran en mi entorno"**
→ En Jupyter Lab instalar la extensión de Plotly o usar `fig.show(renderer="browser")`.
