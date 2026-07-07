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

## PARTE 2 — Visualizaciones Avanzadas en Data Science

---

### ¿Por qué importa diseñar bien un gráfico?

**Qué decir:**

Arrancar con esta pregunta para la clase:

> "¿Alguna vez vieron un gráfico en un diario, una presentación o una red social que los confundió más de lo que los ayudó? ¿O que les pareció convincente y después se dieron cuenta que estaba manipulado?"

La visualización de datos no es solo hacer gráficos bonitos. Es diseñar imágenes que comuniquen información de forma **clara, precisa y honesta**. Un gráfico mal diseñado puede generar decisiones equivocadas en una empresa, malinterpretar resultados científicos o directamente engañar a la audiencia.

En ciencia de datos, el 80% del trabajo es analizar y modelar los datos. Pero el 100% de ese trabajo se comunica a través de gráficos. Si el gráfico falla, el análisis falla también, aunque los números sean perfectos.

---

### 1 — Principios de Diseño de Visualizaciones

**Qué decir:**

El diseño visual no es subjetivo ni cuestión de gusto. Existe un conjunto de principios concretos basados en cómo el ojo y el cerebro humano procesan imágenes. El investigador Colin Ware documentó esto en *Information Visualization: Perception for Design*: la percepción visual tiene reglas, y los gráficos que las respetan se entienden más rápido y con menos errores.

---

#### 1.1 — Encuadre (Framing)

El **encuadre** define el espacio visual donde presentamos la información. La idea es eliminar todo lo que no agrega significado y enfocar la atención del observador en lo que importa.

**Reglas concretas:**
- Título claro que diga **qué** muestra el gráfico, no solo el nombre de las variables.
- Etiquetas en los ejes con la unidad de medida siempre visible (ej: "Temperatura (°C)", no solo "Temperatura").
- Sin elementos decorativos que no aporten: fondos con textura, sombras, bordes 3D, imágenes detrás del gráfico.
- El espacio en blanco es intencional: los márgenes dan respiro visual y facilitan la lectura.

**Ejemplo de título malo vs. bueno:**

| Malo | Bueno |
|------|-------|
| `temp_vs_fecha` | `Temperatura promedio diaria — Planta Alta (enero 2026)` |
| `ventas` | `Ventas totales por producto (en miles de pesos)` |
| `scatter_1` | `Relación entre precio y satisfacción del cliente` |

---

#### 1.2 — Jerarquía Visual

La **jerarquía visual** guía la mirada del observador. El ojo no mira un gráfico al azar: va primero a los elementos más grandes, más brillantes y con mayor contraste. Un buen diseño usa eso a propósito para que el dato más importante sea lo primero que se ve.

Las herramientas para crear jerarquía:

| Herramienta | Cómo se usa |
|-------------|-------------|
| **Tamaño** | El elemento más grande atrae primero la mirada |
| **Color** | Un elemento con color distinto al resto resalta inmediatamente |
| **Contraste** | Fondo claro + elemento oscuro, o viceversa |
| **Posición** | El ojo occidental lee de arriba a la izquierda hacia abajo a la derecha |
| **Grosor** | Una línea más gruesa o un borde más marcado señala importancia |

**Ejemplo práctico:** En un gráfico de barras con 12 meses, si el mes que nos interesa destacar (por ejemplo diciembre, el de mayor venta) se pinta de azul oscuro y los otros 11 de gris claro, la jerarquía visual hace que la mirada vaya directo a diciembre. No hay que leer nada: el mensaje es inmediato.

```python
import matplotlib.pyplot as plt

meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
         'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
ventas = [40, 38, 45, 42, 50, 48, 47, 53, 55, 60, 62, 80]

# Colorear solo el mes destacado
colores = ['#AAAAAA'] * 11 + ['#1a5276']  # 11 grises + 1 azul oscuro

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(meses, ventas, color=colores)
ax.set_title("Ventas mensuales — Diciembre rompe el récord", fontsize=13)
ax.set_ylabel("Ventas (miles $)")
plt.tight_layout()
plt.show()
```

---

#### 1.3 — Anotaciones

Las **anotaciones** son etiquetas, líneas de referencia, flechas o recuadros de texto que agregan contexto directamente sobre el gráfico. Cuando el dato por sí solo no se explica (por ejemplo, una caída repentina en una curva), una anotación le dice al observador "esto pasó por X motivo".

**Cuándo usar anotaciones:**
- Para señalar un evento específico en una serie temporal ("Inicio de pandemia").
- Para destacar el valor de un punto en un scatter plot.
- Para marcar un umbral o límite ("Objetivo de ventas: 50.000").
- Para nombrar directamente las líneas en lugar de usar una leyenda separada.

**Cuándo NO usarlas:** Cuando hay tantas que el gráfico se lee como un prospecto médico. Una anotación que señala todo no señala nada.

```python
import matplotlib.pyplot as plt

meses = list(range(1, 13))
ventas = [40, 38, 20, 42, 50, 48, 47, 53, 55, 60, 62, 80]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(meses, ventas, marker='o', color='#2c3e50', linewidth=2)

# Anotación de la caída en marzo
ax.annotate(
    'Caída por cierre\nde planta (Mar)',
    xy=(3, 20),              # punto donde apunta la flecha
    xytext=(5, 25),          # posición del texto
    arrowprops=dict(arrowstyle='->', color='red'),
    fontsize=9, color='red'
)

# Línea de referencia para el objetivo
ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='Objetivo mensual')
ax.legend()
ax.set_title("Ventas 2026 con contexto")
ax.set_xlabel("Mes")
ax.set_ylabel("Ventas (miles $)")
plt.tight_layout()
plt.show()
```

---

### 2 — Cómo Elegir el Tipo de Gráfico Correcto

**Qué decir:**

El error más común es elegir el gráfico por costumbre o por estética, no por el mensaje que queremos transmitir. La pregunta que siempre hay que hacerse antes de graficar es: **¿qué relación entre los datos quiero mostrar?**

Hay cuatro tipos de relaciones fundamentales, y cada una tiene sus gráficos:

| Relación que quiero mostrar | Gráficos adecuados |
|-----------------------------|-------------------|
| **Comparación** entre categorías | Barras verticales u horizontales, Lollipop |
| **Evolución** a lo largo del tiempo | Líneas, Área |
| **Distribución** de valores | Histograma, Boxplot, Violín |
| **Correlación** entre dos variables | Scatter plot, Heatmap |
| **Proporción** de un total | Barras apiladas (porcentaje), Treemap |
| **Composición** geográfica | Mapa de calor geográfico |

**Los gráficos más usados en Data Science, uno por uno:**

---

#### Gráfico de Barras — Comparar categorías

**Para qué sirve:** Comparar un mismo valor numérico entre categorías distintas. La pregunta que responde es: *"¿cuánto tiene cada uno?"*

**Ejemplos cotidianos:**
- Comparar las ventas mensuales de tres sucursales de un negocio: ¿cuál vendió más en julio?
- Comparar la cantidad de alumnas y alumnos aprobados por materia en una universidad.
- Ver cuántas personas votaron a cada candidato en una elección (las barras del canal 7 en la noche del escrutinio).
- Comparar el precio promedio del alquiler por barrio en CABA.

**Cuándo usarlo:**
- Pocas categorías (idealmente 3 a 15). Con más de 15 categorías las barras quedan muy delgadas y difíciles de leer.
- Cuando el orden de las barras no importa (si importara el tiempo, usarías líneas).

**Cuándo NO usarlo:**
- Si las categorías son momentos en el tiempo consecutivos → usá líneas en su lugar.
- Si tenés más de 20 categorías → usá barras horizontales o lollipop.

**Trampa común:** Empezar el eje Y en un valor distinto de 0. Si las barras van de 95% a 100%, empezar en 94% hace que la diferencia parezca enorme cuando en realidad es mínima. Las barras **siempre** deben empezar en 0.

> En esta clase lo usamos en: ejemplo del Bloque 1 con `ax2.bar(["A", "B", "C"], ...)`.

---

#### Gráfico de Barras Horizontales — Comparar categorías con nombres largos

**Para qué sirve:** Exactamente lo mismo que las barras verticales, pero rotadas 90°. La ventaja es que los nombres de las categorías en el eje Y tienen mucho más espacio para escribirse.

**Ejemplos cotidianos:**
- Ranking de los 20 países con mayor PIB: los nombres de los países caben bien en el eje Y.
- Comparar los partidos políticos por cantidad de bancas (los nombres de los partidos son largos).
- Listar las marcas de autos más vendidas en Argentina con sus unidades.

**Cuándo usarlo:**
- Cuando los nombres de las categorías son largos y se superpondrían en un gráfico vertical.
- Cuando querés que el observador sienta que está leyendo un "ranking" de arriba hacia abajo.

---

#### Gráfico de Líneas — Evolución en el tiempo

**Para qué sirve:** Mostrar cómo cambia un valor a lo largo del tiempo. La línea conecta los puntos para que el ojo vea la tendencia: ¿está subiendo? ¿bajando? ¿estable?

**Ejemplos cotidianos:**
- La curva de temperatura durante el día (8:00, 12:00, 16:00, 20:00 → picos y valles).
- El precio del dólar blue semana a semana durante el año.
- El peso de un paciente durante un tratamiento médico mes a mes.
- Los seguidores de una cuenta de Instagram semana a semana.
- La evolución de casos de COVID-19 por día durante la pandemia.

**Cuándo usarlo:**
- Cuando el eje X es **tiempo** o cualquier variable continua con orden natural.
- Cuando queremos que el observador vea la tendencia o el patrón temporal.
- Con múltiples líneas se puede comparar la evolución de varias categorías a la vez.

**Cuándo NO usarlo:**
- Si el eje X son categorías sin orden (países, sucursales, productos): la línea implica "continuidad entre categorías" que no existe → usá barras.
- Si solo hay 2 o 3 puntos de datos: la línea no tiene sentido con tan pocos puntos.

**Trampa común:** Conectar categorías que no son continuas. Si graficamos "lunes, miércoles, viernes" con una línea, estamos diciendo implícitamente que algo pasó "entre" esos días, cuando no hay datos.

> En esta clase lo usamos en: Bloque 2, análisis temporal de pasajeros 2023 con `sns.lineplot()`.

---

#### Histograma — Distribución de una variable numérica

**Para qué sirve:** Mostrar cómo se reparten los valores de una variable. La pregunta que responde es: *"¿en qué rango están concentrados la mayoría de los datos?"* No muestra categorías separadas — muestra la "forma" de los datos.

**Ejemplos cotidianos:**
- Las notas de un examen de 100 alumnos: ¿la mayoría sacó entre 6 y 8? ¿Hay muchos aplazados? ¿Hay un grupo de alumnos muy avanzados con notas de 10?
- Los sueldos de los empleados de una empresa: ¿están todos agrupados cerca del promedio o hay una gran diferencia entre los que ganan poco y los que ganan mucho?
- Los tiempos de espera en una guardia de emergencias: ¿la mayoría espera menos de 30 minutos? ¿Hay casos extremos de 4 horas?
- Las alturas de los jugadores de un club de básquet.
- Los tiempos de entrega de los pedidos de un e-commerce.

**Cómo leerlo:**
- **Distribución normal (campana de Gauss):** La mayoría está en el centro, pocos en los extremos. Ejemplo: alturas humanas.
- **Distribución sesgada a la derecha:** La mayoría tiene valores bajos pero hay algunos muy altos. Ejemplo: ingresos (pocos ricos, muchos de clase media/baja).
- **Distribución bimodal (dos picos):** Hay dos grupos bien diferenciados. Ejemplo: notas de un examen fácil para algunos y difícil para otros.

**Diferencia clave con barras:**

| | Barras | Histograma |
|---|---|---|
| Eje X | Categorías separadas (Prod A, Prod B...) | Una variable continua dividida en rangos |
| Pregunta | ¿Cuánto tiene cada uno? | ¿Cómo se distribuyen los valores? |
| Barra | Cada barra = una categoría | Cada barra = un rango de valores |

**Cuándo NO usarlo:**
- Para comparar categorías separadas → usá barras.
- Para datos muy escasos (menos de 30 valores): la distribución no se puede ver bien.

> En esta clase lo usamos en: Actividad práctica Paso 1, distribución de vuelos diarios con `sns.histplot()`.

---

#### Boxplot (Diagrama de Caja) — Resumen estadístico de una distribución

**Para qué sirve:** Mostrar el resumen estadístico completo de una variable (mínimo, Q1, mediana, Q3, máximo) en un solo símbolo compacto. Muy útil para comparar la distribución de una variable entre varios grupos.

**Cómo leerlo:**

```
     ┌──────────────┐
─────┤              ├─────    ← Bigotes: mínimo y máximo (sin outliers)
     │   Caja       │         ← Caja: entre Q1 (25%) y Q3 (75%) = el 50% central de los datos
     │   ═══════    │         ← Línea del medio: mediana (el valor de la mitad exacta)
     └──────────────┘
  ○  ← Puntos fuera de los bigotes: outliers (valores atípicos)
```

**Ejemplos cotidianos:**
- Comparar los sueldos de empleados por área en una empresa: ¿el área de sistemas tiene sueldos más altos que el área de administración? ¿Hay más dispersión en un área que en otra?
- Comparar los tiempos de entrega de tres servicios de mensajería: ¿cuál es más consistente (caja más pequeña)?
- Comparar las notas de un examen en tres turnos diferentes.
- En medicina: comparar la presión arterial de pacientes con tres tratamientos distintos.
- Comparar el tiempo de espera en cajeros de supermercados según el día de la semana.

**Cuándo usarlo:**
- Cuando querés comparar la distribución entre grupos.
- Cuando necesitás identificar outliers (valores extremos) de forma visual.
- Cuando tenés muchos datos y el histograma sería demasiado denso.

**Cuándo NO usarlo:**
- Si solo tenés un grupo (usa histograma en su lugar, que es más informativo).
- Si los datos son muy escasos (menos de 20 valores): el boxplot se vuelve engañoso.

**Lo que no se ve en un boxplot:** La distribución interna. Un boxplot con la misma mediana y los mismos bigotes puede representar una distribución normal, una bimodal o una distribución con todos los valores concentrados en un punto. Siempre combinar con un histograma o violinplot para ver la forma real.

---

#### Scatter Plot (Dispersión) — Relación entre dos variables numéricas

**Para qué sirve:** Mostrar si existe una relación entre dos variables numéricas. La pregunta que responde es: *"¿cuando una variable sube, la otra también sube (o baja)?"* Cada punto en el gráfico es una observación.

**Ejemplos cotidianos:**
- ¿Los alumnos que estudian más horas sacan mejores notas? (Eje X: horas de estudio, Eje Y: nota del examen)
- ¿Las casas con más metros cuadrados tienen mayor precio? (Eje X: m², Eje Y: precio)
- ¿Las personas que consumen más calorías tienen mayor peso? (Eje X: calorías diarias, Eje Y: kg)
- ¿Los días con más temperatura se vende más helado? (Eje X: temperatura °C, Eje Y: unidades vendidas)
- ¿Los autos más viejos consumen más nafta? (Eje X: año del auto, Eje Y: litros cada 100km)

**Cómo interpretar la nube de puntos:**
- **Nube en diagonal ascendente (↗):** Correlación positiva. Cuando una sube, la otra también.
- **Nube en diagonal descendente (↘):** Correlación negativa. Cuando una sube, la otra baja.
- **Nube circular/dispersa sin forma:** No hay correlación entre las variables.
- **Puntos alejados del grupo:** Outliers → observaciones atípicas que vale la pena investigar.

**Cuándo NO usarlo:**
- Si una de las variables es categórica → usá boxplot para ver la distribución por categoría.
- Si hay demasiados puntos superpuestos (overplotting) → usá hexbin o reducí el tamaño de los puntos con `alpha=0.3`.

> En esta clase lo usamos en: Actividad práctica Paso 3, asientos vs pasajeros por tipo de vuelo con `px.scatter()` interactivo.

---

#### Heatmap (Mapa de Calor) — Intensidad de un valor en una grilla

**Para qué sirve:** Mostrar el valor de una variable en la intersección de dos categorías, usando el color como canal de información. Permite ver patrones en grillas de datos de un golpe de vista.

**Ejemplos cotidianos:**
- **Horario de mayor actividad en una tienda:** Filas = días de la semana, Columnas = horas del día, Color = cantidad de clientes. De un vistazo se ve cuándo hay picos (rojo = muy ocupado, azul = tranquilo).
- **Mapa de calor del clima:** Filas = meses del año, Columnas = ciudades, Color = temperatura promedio. Se ve rápidamente qué ciudad es más calurosa en qué época.
- **Matriz de correlación entre variables:** Cada celda muestra qué tan relacionadas están dos variables del dataset (1 = correlación perfecta, 0 = sin correlación, -1 = correlación inversa). Fundamental antes de construir un modelo de machine learning.
- **Ventas por región y producto:** Filas = regiones del país, Columnas = productos, Color = unidades vendidas. Se ven los "huecos de mercado" de un vistazo.

**Cómo leerlo:** El color es el dato. Una escala de color típica va de azul oscuro (valor bajo) a rojo intenso (valor alto), o de blanco a un color saturado.

**Cuándo usarlo:**
- Cuando tenés datos en formato de grilla (filas × columnas = valor).
- Para matrices de correlación entre variables numéricas de un dataset.
- Para patrones temporales con dos dimensiones (hora × día, mes × año).

**Cuándo NO usarlo:**
- Para comparar valores exactos: el ojo es malo comparando matices de color con precisión.
- Si la grilla es muy grande: las celdas quedan tan chicas que los colores son ilegibles.

---

#### Lollipop — Alternativa visual a las barras

**Para qué sirve:** Exactamente lo mismo que las barras verticales u horizontales, pero con un diseño más liviano. En lugar de una barra sólida, muestra una línea delgada con un punto al final. El mensaje es idéntico pero el gráfico "pesa" visualmente menos.

**Ejemplos cotidianos:**
- Ranking de los 25 países con mayor esperanza de vida: con 25 barras sólidas el gráfico queda muy pesado; con lollipop se respira mejor.
- Comparar el NPS (Net Promoter Score) de 20 productos de una empresa.
- Listar los 30 empleados con mayor productividad en un mes.

**Cuándo usarlo:**
- Cuando hay muchas categorías (más de 15) y las barras sólidas saturan visualmente.
- Cuando queremos un diseño más moderno y limpio para una presentación ejecutiva.

**Diferencia visual con barras:** El lollipop elimina el "ruido visual" de las barras anchas. Esto le da más protagonismo a los valores extremos y hace que las diferencias entre categorías se lean más fácilmente. El contenido informativo es idéntico.

---

#### Gráfico de Torta / Pie Chart — Proporciones de un total

**Para qué sirve:** Mostrar cómo se divide un total (100%) entre categorías. La pregunta que responde es: *"¿qué parte del total representa cada categoría?"*

**Ejemplos cotidianos donde SÍ tiene sentido:**
- El presupuesto mensual de una familia dividido en categorías: 40% alquiler, 20% comida, 15% transporte, 25% otros. Con solo 4 categorías y diferencias grandes, la torta funciona.
- Cuota de mercado de 3 empresas que se reparten el 100%: si una tiene 60%, la segunda 30% y la tercera 10%, la torta lo muestra claramente.
- Distribución de sangre en una población: A (42%), O (44%), B (10%), AB (4%).

**Por qué los gráficos de torta tienen tan mala reputación:**

El ojo humano es MUCHO más preciso comparando **longitudes** (barras) que comparando **ángulos** (sectores de torta). Una diferencia de 5 puntos porcentuales entre dos barras es obvia. La misma diferencia entre dos sectores de torta es casi imperceptible.

Experimento: ¿cuál de estos sectores es más grande?

```
Sector A: 32%   Sector B: 29%   Sector C: 25%   Sector D: 14%
```

Con una torta es muy difícil saberlo sin leer los números. Con barras, la diferencia es inmediata.

**Cuándo NO usar la torta:**
- Si hay más de 5 o 6 categorías: los sectores se vuelven tan pequeños que son ilegibles.
- Si las proporciones son similares entre sí (ej: 28%, 26%, 24%, 22%): el ojo no puede distinguirlas.
- Si el mensaje principal es comparar valores entre categorías (en lugar de mostrar la parte del todo).

**La alternativa honesta:** Una barra horizontal apilada al 100% o barras simples ordenadas de mayor a menor comunican exactamente la misma información con mucha más precisión visual.

> En esta clase aparece en la sección de ética (Sección 3.2) como ejemplo de gráfico potencialmente manipulador, con el ejemplo de Apple/Google/Microsoft.

---

### 3 — Ética y Percepción Visual: Gráficos que Mienten

**Qué decir:**

Los gráficos son la forma más efectiva de comunicar datos, y también la forma más efectiva de manipularlos. Un gráfico puede presentar los mismos números y contar dos historias completamente distintas según cómo se diseñe. Esto es un problema ético serio en periodismo, política y ciencia.

Hay tres formas principales de manipular con visualizaciones:

---

#### 3.1 — Manipulación del eje Y (la más común)

Cuando el eje Y no empieza en cero, las diferencias entre barras se ven exageradas. La barra más alta puede parecer el doble de la más baja, cuando en realidad es solo un 2% mayor.

**Ejemplo concreto:** Cuota de mercado real de tres empresas tecnológicas:

```
Apple:     35%
Google:    33%
Microsoft: 32%
```

Estas tres empresas están prácticamente empatadas. Pero si graficamos con el eje Y entre 31% y 36%:
- Apple parece tener el triple de mercado que Microsoft.
- La diferencia real es 3 puntos porcentuales, pero visualmente parece enorme.

**Regla:** En gráficos de barras, el eje Y siempre debe empezar en 0 a menos que haya una justificación explícita y visible. En gráficos de líneas, es aceptable no empezar en 0 cuando los valores están todos en un rango acotado, pero hay que indicarlo claramente.

---

#### 3.2 — Gráficos de torta con demasiadas categorías o sectores manipulados

Los gráficos de torta son problemáticos porque el ojo humano es malo comparando ángulos. Si hay más de 4 o 5 categorías, los sectores se vuelven indistinguibles. Y si el sector que "importa" se saca hacia afuera (el efecto "pull"), parece más grande de lo que es aunque el ángulo sea el mismo.

**Código para demostrar el gráfico manipulado vs. el correcto:**

```python
import plotly.express as px
import pandas as pd

data = {
    'Compañía': ['Apple', 'Google', 'Microsoft', 'Otras'],
    'Cuota de Mercado (%)': [35, 15, 10, 40]
}
df = pd.DataFrame(data)

# Gráfico manipulado: Apple "separada" y con título que exagera su dominio
fig = px.pie(
    df,
    values='Cuota de Mercado (%)',
    names='Compañía',
    title='Cuota de Mercado Global de Tecnología (35% Apple)',
    hole=0.4,
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig.update_traces(
    textinfo='percent+label',
    hoverinfo='label+percent',
    textfont_size=14,
    pull=[0.15, 0, 0, 0]  # Apple separada para que parezca más dominante
)
fig.show()
```

**Qué señalar en el gráfico manipulado:**
- Apple tiene 35% pero el sector separado y el título que la nombra explícitamente da la impresión de que domina el mercado.
- "Otras" tiene 40% (más que Apple) pero está visualmente minimizado.
- El título ya implica un mensaje: "35% Apple" en lugar de "distribución equitativa".

**La versión honesta:**

```python
# Gráfico honesto: barras horizontales, todas iguales, eje desde 0
import matplotlib.pyplot as plt

empresas = ['Otras', 'Apple', 'Google', 'Microsoft']
cuotas = [40, 35, 15, 10]

fig, ax = plt.subplots(figsize=(7, 3))
ax.barh(empresas, cuotas, color='#5DADE2')
ax.set_xlabel('Cuota de mercado (%)')
ax.set_title('Cuota de mercado — sin destacar ninguna empresa')
ax.set_xlim(0, 50)  # eje desde 0
for i, v in enumerate(cuotas):
    ax.text(v + 0.5, i, f'{v}%', va='center')
plt.tight_layout()
plt.show()
```

**Qué remarcar:** El mismo dato, pero ahora "Otras" queda en primer lugar (más cuota que Apple). Ningún sector está separado. El mensaje es neutro. Los datos son idénticos.

---

#### 3.3 — Omisión selectiva de datos

Mostrar solo la parte del tiempo donde algo creció y omitir la caída anterior. Mostrar el promedio sin la distribución para ocultar que hay un grupo muy perjudicado. Usar datos sin normalizar para comparar cosas que no son comparables (ej: comparar ventas de una empresa grande con una chica en valores absolutos en lugar de porcentuales).

---

#### 3.4 — Accesibilidad: El daltonismo y la impresión en B&N

El **8% de los hombres** y el 0.5% de las mujeres tienen algún tipo de daltonismo. El tipo más común confunde rojo y verde. Si un gráfico usa solo color para diferenciar categorías (sin forma, sin patrón, sin etiqueta directa), esas personas no pueden leerlo.

**Reglas de accesibilidad:**
- Usar `palette="colorblind"` en Seaborn para paletas diseñadas científicamente.
- Respaldar el color con forma (`style="month", markers=True` en Seaborn).
- Nunca confiar solo en la diferencia rojo vs. verde para distinguir "bueno" vs. "malo".
- Verificar el gráfico en escala de grises antes de enviarlo (imprimir en B&N).

---

### 4 — Tipos de Gráfico y lo que Vamos a Usar en Semana 5

**Qué decir:**

> "Ahora que entendemos cuándo y por qué usar cada tipo de gráfico, y qué errores evitar, en la segunda parte de la clase vamos a ver cómo construir gráficos avanzados en Python. Estos son los tipos que vamos a usar:"

| Gráfico | Librería | Para qué lo vamos a usar |
|---------|----------|--------------------------|
| `histplot` con KDE | Seaborn | Distribución de vuelos diarios (actividad práctica) |
| `lineplot` multivariado | Seaborn | Evolución temporal de pasajeros con accesibilidad cromática |
| `scatterplot` con `hue` + `size` | Seaborn | Análisis multivariado: 4 variables en 2D |
| `scatter` interactivo | Plotly Express | Relación asientos vs. pasajeros con zoom y tooltips |
| Subplots con `GridSpec` | Matplotlib | Layout asimétrico: gráfico principal + resumen |
| Gráfico de líneas con `mdates` | Matplotlib + Seaborn | Eje de fechas formateado correctamente |
| Exportación `.png` 300dpi | Matplotlib | Reporte de alta resolución para presentación |

**Transición:** 

> "Antes de arrancar con el código, ¿preguntas sobre los principios de diseño o los errores éticos que vimos? El objetivo no es que se memoricen todo esto, sino que cuando hagan un gráfico se pregunten: ¿qué mensaje quiero transmitir? ¿Estoy usando el tipo correcto? ¿Estoy siendo honesto con los datos?"

---

---

## Código de la Clase — Abrir: `Semana_5_Visualizaciones_Avanzadas_en_Data_Science_.ipynb`

---

## Las Librerías que Vamos a Usar

Toda la clase gira alrededor de tres librerías de Python. Cada una resuelve un problema distinto, y en la práctica se combinan todo el tiempo (por ejemplo, Seaborn dibuja *sobre* un lienzo de Matplotlib, y Plotly se usa aparte para lo interactivo).

### Matplotlib

Es la librería de graficación original de Python, creada en 2003 por John Hunter (ver Tema 1 para la historia completa). Funciona a **bajo nivel**: cada elemento del gráfico —el lienzo, los ejes, cada línea, cada barra, cada etiqueta— es un objeto de Python que se crea y se configura a mano. Eso la hace la más flexible y también la más verbosa: se puede controlar absolutamente todo (posición exacta de una anotación, grosor de cada línea, tipografía de cada texto), pero hay que escribir más código para lograrlo.

La usamos porque:
- Genera **imágenes estáticas** (PNG, PDF, SVG) ideales para reportes, papers y presentaciones impresas.
- Es la librería que **todas las demás usan por debajo** — entender su lógica de `Figure`/`Axes` ayuda a entender Seaborn también.
- Da control total sobre el resultado final, algo clave para GridSpec (layouts asimétricos) y para exportación profesional (`dpi`, `bbox_inches`).

### Seaborn

Está construida **encima de Matplotlib** (no la reemplaza). La creó Michael Waskom en 2012 para reducir la cantidad de código necesaria al graficar datos tabulares (DataFrames de Pandas). En Matplotlib puro hay que decirle a mano qué graficar en X, qué en Y y de qué color cada categoría; en Seaborn se le pasa el DataFrame completo y los nombres de columnas: `sns.scatterplot(data=df, x="col1", y="col2", hue="categoria")` hace en una línea lo que en Matplotlib puro tomaría 5 o 10.

La usamos porque:
- Tiene **paletas de color y gráficos estadísticos ya resueltos** (boxplot, violinplot, distribuciones) que en Matplotlib habría que programar desde cero.
- Entiende de forma nativa columnas categóricas (`hue`, `style`, `size`) para análisis multivariado.
- Sigue siendo Matplotlib por debajo: cada gráfico de Seaborn devuelve un `Axes` de Matplotlib que se puede seguir personalizando con `ax.set_title()`, `ax.set_xlabel()`, etc.

### Plotly (Plotly Express)

Es la única de las tres que **no genera una imagen**, sino código HTML + JavaScript. El resultado se renderiza en el navegador (o en la celda del notebook) como un objeto interactivo: se puede hacer zoom, pasar el mouse sobre un punto para ver su valor exacto (tooltip), y hacer clic en la leyenda para ocultar categorías.

La usamos porque:
- Sirve para **exploración de datos**: poder hacer zoom e inspeccionar valores exactos ahorra tener que regraficar con distintos rangos cada vez que se quiere mirar un detalle.
- Es la mejor opción para **dashboards y reportes web**, donde el usuario final interactúa con el gráfico.
- No reemplaza a Matplotlib/Seaborn: para un documento impreso o un PDF, un gráfico interactivo no sirve de nada — ahí se sigue usando las otras dos.

**Resumen de cuándo usar cada una:**

| Necesito... | Uso |
|---|---|
| Un gráfico rápido y estándar (barras, líneas, dispersión) a partir de un DataFrame | Seaborn |
| Control total del layout (paneles asimétricos, anotaciones a medida, exportación en PDF/SVG) | Matplotlib |
| Que el usuario final pueda explorar el gráfico (zoom, tooltip, filtrar por leyenda) | Plotly |

---

## Bloque 0 — Principios de Diseño y Ética Visual en la Práctica

Este bloque no trae teoría nueva — es la ejecución en código de lo que ya se explicó en las Secciones 1, 2 y 3 de la Parte 2. Lo que sigue es el detalle línea por línea de los dos ejemplos, para no tener que improvisar la explicación en vivo.

### Ejemplo 1 — Encuadre, Jerarquía Visual y Anotaciones

**Qué compara:** el mismo dato (ventas por mes) graficado dos veces — una vez "como sale" y otra vez aplicando las reglas de diseño de la Sección 1.

**Línea por línea:**

```python
fig, (ax_sin_curar, ax_curado) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
```

- `plt.subplots(...)` crea el lienzo Y los ejes al mismo tiempo, y devuelve **dos cosas** — por eso hay dos nombres a la izquierda del `=`.
- `fig` es el lienzo completo (el objeto `Figure`) — el "cuadro físico" que contiene todo. Solo hay uno, aunque adentro tenga varios gráficos.
- `nrows=1, ncols=2` le pide a `subplots()` una grilla de 1 fila y 2 columnas → 2 gráficos lado a lado. Como son 2, `subplots()` devuelve una lista de 2 Axes en vez de un Axes solo.
- `(ax_sin_curar, ax_curado)`: en vez de recibir esa lista en una sola variable (`axes`) e indexarla después (`axes[0]`, `axes[1]`), acá se **desempaqueta directamente en dos variables**. Python permite esto porque a la derecha hay una lista de exactamente 2 elementos y a la izquierda hay 2 nombres entre paréntesis: el primer Axes de la lista va a `ax_sin_curar`, el segundo a `ax_curado`. Se usan esos nombres (y no `ax1`, `ax2`) a propósito — el nombre de la variable ya dice qué va a mostrar cada gráfico, así el código se entiende sin correrlo.
- `figsize=(12, 4)` es el tamaño del lienzo completo en pulgadas: 12 de ancho (para que entren los 2 gráficos con espacio) y 4 de alto.

```python
ax_sin_curar.bar(meses, ventas, color="steelblue")
ax_sin_curar.set_title("ventas")
```

- `ax_sin_curar.bar(...)` dibuja un gráfico de barras **dentro de ese Axes específico**, no en el lienzo entero. `meses` son las categorías del eje X, `ventas` las alturas de cada barra.
- `color="steelblue"` pinta **todas** las barras del mismo color — no hay ninguna decisión de jerarquía visual, es lo que sale por defecto si no se piensa el color.
- `set_title("ventas")` pone un título mínimo, sin decir la unidad (¿pesos? ¿unidades vendidas?) ni destacar ningún dato — el resultado de no aplicar encuadre.

```python
colores = ['#AAAAAA'] * 11 + ['#1a5276']
ax_curado.bar(meses, ventas, color=colores)
```

- Acá `color` ya no es un string único: es una **lista de 12 colores**, uno por barra. `bar()` acepta esto — si se le pasa una lista del mismo largo que los datos, pinta cada barra con su color correspondiente en vez de usar un solo color para todas.
- `['#AAAAAA'] * 11` repite el gris 11 veces (los primeros 11 meses) y `+ ['#1a5276']` agrega un azul oscuro al final (diciembre): esa es la jerarquía visual — el color le dice al ojo dónde mirar antes de leer ningún número.

```python
ax_curado.set_title("Ventas mensuales 2024 (miles de $)")
ax_curado.set_ylabel("Ventas (miles $)")
```

El título ahora es específico e **incluye la unidad** (encuadre: quien mira el gráfico no tiene que adivinar qué se está midiendo), y `set_ylabel()` refuerza lo mismo en el eje Y.

```python
ax_curado.annotate(
    'Récord del año',
    xy=(11, 80),
    xytext=(7, 68),
    arrowprops=dict(arrowstyle='->')
)
```

- `annotate()` dibuja un texto con una flecha que apunta a un punto específico del gráfico.
- `xy=(11, 80)` es la punta de la flecha: la posición del dato que se quiere señalar (mes en el índice 11 = diciembre; valor 80). Son coordenadas reales del gráfico, no píxeles de pantalla.
- `xytext=(7, 68)` es dónde se ubica el texto "Récord del año" — deliberadamente separado del dato para no taparlo con la flecha.
- `arrowprops=dict(arrowstyle='->')` dibuja la flecha que conecta el texto con el dato; sin este parámetro solo aparecería el texto suelto, sin flecha.

```python
plt.tight_layout()
plt.show()
```

`tight_layout()` ajusta el espacio entre los dos subgráficos para que el título de uno no se superponga con las etiquetas del otro, y `plt.show()` renderiza el lienzo completo con los 2 Axes ya dibujados.

**Qué decir:** el dato es idéntico en ambos gráficos. Lo único que cambió es texto (título con unidad), un color por barra en vez de uno solo para todas, y una anotación — tres decisiones de diseño, cero manipulación de los números.

---

### Ejemplo 2 — Ética Visual: el mismo dato, dos historias distintas

**Qué compara:** el pie chart de cuota de mercado tecnológico (código original del material de la materia) contra una versión honesta de los mismos datos.

**¿Por qué pandas acá, con solo 4 filas de datos?** `px.pie()` recibe un DataFrame y referencia las columnas por nombre (`values='Cuota de Mercado (%)'`, `names='Compañía'`). Se podría pasar directo dos listas sueltas, pero armar el `DataFrame` primero es el mismo patrón que se usa en el resto de la clase con Seaborn (`data=df, x="...", y="..."`) — conviene acostumbrarse desde acá aunque el dataset sea chico, porque con datos reales casi siempre van a venir ya en un DataFrame (de un `read_csv`, por ejemplo).

**¿Por qué Plotly Express y no Matplotlib para este ejemplo?** Porque lo que hace más persuasivo (y más fácil de manipular) a un gráfico de torta es justamente poder pasar el mouse por cada sector y ver el tooltip con el % exacto — eso es interactividad nativa de Plotly, que Matplotlib no da sin trabajo extra. Es el mismo motivo por el que la vamos a usar en el Bloque 3 para exploración de datos.

**Puntos clave del gráfico manipulado:**

- `hole=0.4` no manipula nada — es puramente estético (convierte la torta en "donut"). Se aclara para que no se confunda con la manipulación real.
- `title='Cuota de Mercado Global de Tecnología (35% Apple)'` es la primera manipulación: el título elige remarcar el número de Apple, dirigiendo la lectura antes de que la persona mire el gráfico.
- `pull=[0.15, 0, 0, 0]` es la segunda: separa físicamente el primer sector (Apple, por el orden en que aparece en `data`) del resto de la torta. Eso hace que el ojo lo perciba como más importante, aunque "Otras Compañías" (40%) sea matemáticamente mayor que Apple (35%).

**Puntos clave de la versión honesta:**

- `empresas` y `cuotas` están **ordenados de mayor a menor** — ninguna empresa está "elegida" al principio de la lista para separarla del resto.
- `ax.set_xlim(0, 50)` fuerza al eje X (los porcentajes) a arrancar en 0 — la regla de oro para que las barras sean comparables a simple vista.
- El bucle `for i, v in enumerate(cuotas): ax.text(...)` agrega el valor exacto al lado de cada barra, así no hace falta "leer" la escala del eje para saber el número real.

**Qué decir:** los datos de entrada (`data`, `empresas`/`cuotas`) son los mismos en los dos gráficos. Todo lo que cambió es texto, orden y un parámetro de separación — ninguno de los dos "inventa" números, pero uno presenta el dato para confundir y el otro para informar.

**¿Cuál de los dos es mejor y por qué?** El de barras. El pie chart no está mal hecho técnicamente — el problema es que `pull` + el título eligen y refuerzan una conclusión que el dato real no sostiene ("Apple domina" cuando en realidad "Otras Compañías" tiene más cuota, 40% vs 35%). Un gráfico honesto no es el más prolijo ni el más lindo: es el que un lector puede interpretar correctamente sin que quien lo hizo tenga que aclarar nada. El de barras gana porque, con el mismo dato, el orden por valor real, el eje desde 0 y el porcentaje explícito en cada barra garantizan que la conclusión que saca cualquiera que lo mire sea la correcta.

> Repaso completo de las 4 formas de manipular con gráficos (recorte de eje, tortas engañosas, omisión de datos, colores sesgados): **Sección 3** del Readme Repaso.

---

## Bloque 1 — Arquitectura de Figuras y Diseño Avanzado

---

### Tema 1 — La Jerarquía de Matplotlib: Figure y Axes

**Contexto teórico:**

Matplotlib fue creado en 2003 por John Hunter, un neurocientífico que necesitaba una herramienta de gráficos en Python similar a MATLAB. Su diseño principal es una **jerarquía de objetos** que refleja cómo funcionan los gráficos en el mundo real: primero existe el soporte físico (el lienzo), y encima se dibuja el contenido (el gráfico).

Esta jerarquía tiene dos niveles clave:

| Objeto | Nombre técnico | Qué contiene | Analogía |
|---|---|---|---|
| `Figure` | La figura completa | Todos los Axes, el fondo, el título general | El cuadro físico colgado en la pared |
| `Axes` | Un subgráfico | Los ejes X e Y, las líneas, las barras, el título, las etiquetas | Una pintura dentro del cuadro |

**Los dos estilos de programación en Matplotlib:**

**Estilo funcional (pyplot):** Se usa `plt.plot()`, `plt.title()`, `plt.xlabel()`. Matplotlib mantiene internamente un estado que dice "el gráfico activo es este", y todos los comandos de `plt.` afectan a ese gráfico activo. Es cómodo para hacer un solo gráfico rápido, pero cuando hay múltiples subgráficos, el "estado activo" es ambiguo y genera errores.

**Estilo orientado a objetos (OOP):** Se guardan el `Figure` y los `Axes` en variables (`fig` y `ax`), y se dibujan las cosas explícitamente en el eje correcto con `ax.plot()`, `ax.set_title()`. Esto es **la forma correcta para múltiples subgráficos**: no hay ambigüedad sobre dónde va cada cosa.

La función `plt.subplots(nrows, ncols)` devuelve AMBOS objetos a la vez:
- `fig` → el lienzo completo (solo hay uno)
- `axes` → lista de Axes (tantos como `nrows × ncols`)

Con `nrows=1, ncols=3`, axes es una lista `[ax1, ax2, ax3]` donde cada elemento es un gráfico independiente.

**Parámetros importantes de `plt.subplots()`:**

| Parámetro | Qué hace | Ejemplo |
|---|---|---|
| `nrows`, `ncols` | Número de filas y columnas de gráficos | `nrows=2, ncols=3` → 6 gráficos |
| `figsize=(w, h)` | Tamaño del lienzo en pulgadas | `(10, 4)` → ancho=10, alto=4 |
| `sharex=True` | Todos los gráficos comparten el eje X | Útil en series temporales |
| `sharey=True` | Todos los gráficos comparten el eje Y | Útil para comparar escalas |

**Parámetros visuales básicos (`color`, `marker`, `linewidth`, `s`):**

Estos parámetros aparecen en casi cualquier función de Matplotlib y no cambian de significado entre `plot()`, `bar()` o `scatter()`:

| Parámetro | Qué hace | Ejemplo |
|---|---|---|
| `color` | Color de la línea, barra o punto. Acepta nombres (`"blue"`), códigos hex (`"#1a5276"`) o abreviaturas (`"r"`, `"g"`, `"b"`) | `color="orange"` → barras naranjas |
| `marker` | Símbolo que se dibuja sobre cada punto de datos, además de la línea (parámetro de `plot()`) | `marker="o"` → círculo, `"s"` → cuadrado, `"^"` → triángulo |
| `linewidth` | Grosor de la línea en puntos tipográficos | `linewidth=2` → línea más gruesa que el default (1) |
| `s` | Tamaño de cada punto en un `scatter()` (marker size, en puntos al cuadrado) | `s=100` → puntos grandes |

Sin `marker`, `ax.plot()` dibuja solo la línea continua, sin ningún símbolo en los puntos de datos reales. Agregar `marker="o"` deja ver exactamente dónde están medidos los datos — importante con pocos puntos; con series de miles de puntos el marker satura el gráfico y conviene omitirlo.

**Diferencia crítica entre títulos:**

```python
ax.set_title("...")   # Título de ESE gráfico específico (aparece encima del Axes)
fig.suptitle("...")   # Título de TODA la figura (aparece por encima de todos los Axes)
```

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
- `fig.suptitle()` pone el título general por encima de todo. `ax.set_title()` pone el título de ese subplot específico.

---

### Tema 2 — Seaborn: Axes-level vs Figure-level

**Contexto teórico:**

Seaborn fue creado por Michael Waskom (2012) originalmente como una capa de estilos más bonitos encima de Matplotlib. Con el tiempo se convirtió en su propia librería con funciones propias. Esta historia dual explica por qué Seaborn tiene dos APIs completamente distintas:

**API Axes-level:** Diseñada para convivir con el modelo de objetos de Matplotlib. Cada función dibuja en un `Axes` existente. Acepta el parámetro `ax=` para saber exactamente dónde dibujar. Devuelve el mismo `Axes` de Matplotlib que recibió. Ejemplos: `sns.scatterplot()`, `sns.boxplot()`, `sns.histplot()`, `sns.lineplot()`, `sns.barplot()`.

**API Figure-level:** Diseñada para análisis exploratorio rápido. Cada función crea su propio `Figure` completo internamente. No acepta `ax=` porque no trabaja con Axes individuales sino con grillas de paneles. Devuelve un objeto `FacetGrid` (no un Axes). Ejemplos: `sns.displot()`, `sns.relplot()`, `sns.catplot()`, `sns.lmplot()`.

**¿Qué es un FacetGrid?**

Un FacetGrid es un objeto de Seaborn que maneja internamente una grilla de paneles. Cuando usamos `col="time"`, Seaborn crea automáticamente un panel por cada valor único de la columna `time`. El número de paneles es dinámico: si `time` tiene 3 valores, crea 3 paneles. Si tiene 10, crea 10.

Un FacetGrid NO es un Axes de Matplotlib. Tiene su propia API:
```python
g = sns.displot(...)
g.set(xlabel="Etiqueta X")           # cambiar etiquetas de todos los paneles
g.set_titles("{col_name}")            # cambiar títulos de los paneles
g.figure.suptitle("Título general")  # título de toda la figura
```

**Regla de oro para elegir:**

| Situación | Usar |
|---|---|
| Quiero un gráfico dentro de un subplot que yo controlo | Axes-level + `ax=mi_eje` |
| Quiero combinar con GridSpec o un layout personalizado | Axes-level |
| Quiero dividir los datos en paneles por categoría | Figure-level (`col=`, `row=`) |
| Quiero explorar rápidamente sin preocuparme por el layout | Figure-level |

> ⚠️ **Error frecuente:** Pasar `ax=` a `sns.displot()` o a cualquier función figure-level genera error. Figure-level no sabe trabajar con Axes individuales.

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

**Contexto teórico:**

`plt.subplots(nrows, ncols)` divide el lienzo en una grilla perfectamente simétrica: todos los gráficos tienen exactamente el mismo tamaño. Para muchos reportes esto no es ideal. Un dashboard profesional típico tiene un gráfico principal grande, un par de métricas pequeñas arriba, y una leyenda a la derecha.

`matplotlib.gridspec.GridSpec` resuelve este problema definiendo una **cuadrícula de proporciones** invisible sobre el lienzo. Cada gráfico luego se asigna a una posición (o a varias posiciones fusionadas) dentro de esa cuadrícula.

**El sistema de ratios:**

```python
gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 3])
```

Esto dice: "Divide el alto del lienzo en 4 partes (1+3). La fila 0 ocupa 1 parte. La fila 1 ocupa 3 partes." Las proporciones son relativas, no absolutas.

**Parámetros clave de GridSpec:**

| Parámetro | Qué controla | Ejemplo |
|---|---|---|
| `height_ratios` | Altura relativa de cada fila | `[1, 3]` → abajo es 3× más alto |
| `width_ratios` | Ancho relativo de cada columna | `[2, 1]` → izquierda es 2× más ancha |
| `hspace` | Espacio vertical entre gráficos | `0.4` = 40% del alto promedio |
| `wspace` | Espacio horizontal entre gráficos | `0.3` = 30% del ancho promedio |

**Asignación de posiciones:**

```python
ax_superior = fig.add_subplot(gs[0])    # slot 0 = primera fila completa
ax_inferior = fig.add_subplot(gs[1])    # slot 1 = segunda fila completa

# Con 2 filas y 2 columnas:
ax_a = fig.add_subplot(gs[0, 0])        # fila 0, columna 0
ax_b = fig.add_subplot(gs[0, 1])        # fila 0, columna 1
ax_c = fig.add_subplot(gs[1, :])        # fila 1, TODAS las columnas (gráfico ancho)
```

**Casos de uso reales:**
- Dashboard de monitoreo: métrica pequeña arriba (ratio 1) + gráfico detallado abajo (ratio 4)
- Reporte de análisis: histograma de distribución izquierda + boxplot derecha (ancho 2:1)
- Comparación temporal: gráfico principal + zoom de un período específico encuadrado arriba

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

**Contexto teórico:**

Internamente, Python y Pandas almacenan las fechas como **números enteros que representan los nanosegundos transcurridos desde el 1 de enero de 1970** (el llamado Unix epoch). Un datetime que diga "2024-07-15 14:30:00" es, por debajo, simplemente el número `1721051400000000000`.

Esta representación numérica es lo que permite que las fechas se **ordenen cronológicamente, se puedan restar entre sí, se puedan agrupar por mes o año**, y se puedan graficar en un eje continuo. Cuando Pandas lee una fecha como texto, ese número no existe: la fecha es solo una cadena de caracteres como `"ABC"`. No se puede ordenar cronológicamente, no se puede restar, no se puede usar `.dt.year`.

**¿Qué hace `pd.to_datetime()` internamente?**

1. Lee la cadena de texto (`"2024-07-15"`).
2. La interpreta según el `format` especificado (`'%Y-%m-%d'`).
3. Convierte el texto al número entero de nanosegundos desde el epoch.
4. Almacena ese número en la columna con tipo `datetime64[ns]`.

**Cadenas de formato — tabla completa:**

| Código | Significado | Texto de entrada |
|---|---|---|
| `%Y` | Año 4 dígitos | `"2024"` |
| `%m` | Mes 2 dígitos | `"07"` |
| `%d` | Día 2 dígitos | `"15"` |
| `%H` | Hora 24h | `"14"` |
| `%M` | Minutos | `"30"` |
| `%b` | Mes abreviado inglés | `"Jul"` |

Combinaciones comunes:
- Fecha argentina: `"15/07/2024"` → `format='%d/%m/%Y'`
- Fecha ISO: `"2024-07-15"` → `format='%Y-%m-%d'`
- Fecha con hora: `"2024-07-15 14:30:00"` → `format='%Y-%m-%d %H:%M:%S'`

**Verificación rápida:**
```python
df.dtypes                  # debe decir 'datetime64[ns]', no 'object'
df['fecha'].dt.year        # el accessor .dt solo funciona en columnas datetime
df['fecha'].dt.month       # extrae el mes
df['fecha'].dt.day_of_week # 0=lunes, 6=domingo
```

**Truco para fechas mal formateadas:**
```python
df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
# errors='coerce' → las fechas que no se puedan convertir quedan como NaT (Not a Time)
# en lugar de tirar un error y detener todo el programa
```

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

**Tip de profesor:** Mostrar ambos gráficos lado a lado para que el contraste sea visible de golpe. Ejecutar también `df.dtypes` después de la conversión para mostrar el cambio de tipo de `object` a `datetime64[ns]`.

---

### Tema 5 — Formatear el Eje de Fechas con `matplotlib.dates`

**Contexto teórico:**

Cuando el eje X tiene valores `datetime`, Matplotlib elige automáticamente cómo mostrar las etiquetas. El problema es que esa elección automática a veces genera etiquetas muy largas (`"2023-01-01 00:00:00"`) o demasiado densas (una etiqueta por cada punto de datos cuando hay 365 puntos en el año).

`matplotlib.dates` (`mdates`) es el módulo que controla los dos aspectos del eje de fechas:

**1. El Formatter — controla CÓMO SE MUESTRA cada etiqueta:**
```python
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
```
Controla el texto que aparece en cada marca del eje. Usa los mismos códigos de formato (`%Y`, `%m`, `%d`) que `pd.to_datetime()`.

Ejemplos de formatos:
- `'%Y'` → `"2024"` (solo año)
- `'%b %Y'` → `"Jul 2024"` (mes abreviado + año)
- `'%d/%m'` → `"15/07"` (día/mes)
- `'%Y-%m-%d'` → `"2024-07-15"` (formato ISO completo)

**2. El Locator — controla DÓNDE aparece cada marca:**
```python
ax.xaxis.set_major_locator(mdates.YearLocator())
```

| Locator | Frecuencia |
|---|---|
| `mdates.YearLocator()` | Una marca por año |
| `mdates.MonthLocator()` | Una marca por mes |
| `mdates.MonthLocator(bymonth=[1, 7])` | Solo enero y julio |
| `mdates.DayLocator(interval=7)` | Una marca cada 7 días |
| `mdates.HourLocator(interval=6)` | Una marca cada 6 horas |

**Cómo obtener el eje activo:**
```python
ax = plt.gca()   # "get current axes" → obtiene el Axes del gráfico activo
```
Necesitamos el objeto `ax` para poder llamar a `ax.xaxis.set_major_formatter()`. Si creamos el gráfico con OOP (`fig, ax = plt.subplots()`), ya tenemos `ax` directamente y no necesitamos `gca()`.

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

**Contexto teórico:**

Un gráfico 2D tiene exactamente dos ejes. Pero los datasets del mundo real tienen decenas de variables. La pregunta es: ¿cómo mostramos más de dos variables en una superficie plana?

La respuesta está en los **canales estéticos** o **atributos preatentivos**: propiedades visuales que el ojo humano procesa de forma automática e instantánea, sin necesidad de "leer" el gráfico conscientemente. El investigador Colin Ware (2004) demostró que el cerebro detecta estas propiedades en menos de 200 milisegundos, antes de que pueda razonar conscientemente sobre el gráfico.

**Los canales disponibles en Seaborn y su fuerza perceptual:**

| Canal | Parámetro | Tipo ideal de variable | Fuerza perceptual |
|---|---|---|---|
| Posición X | `x=` | Cualquiera | ★★★★★ Más precisa |
| Posición Y | `y=` | Cualquiera | ★★★★★ Más precisa |
| Color (hue) | `hue=` | Categórica | ★★★★ Muy buena |
| Tamaño | `size=` | Numérica continua | ★★★ Moderada |
| Forma/trazo | `style=` | Categórica (pocas) | ★★ Débil |

La posición es la más precisa porque el ojo puede comparar distancias con mucha exactitud. El color funciona bien para categorías pero el ojo es malo comparando tonos para valores numéricos continuos. El tamaño comunica una jerarquía pero es impreciso: es difícil determinar si un círculo es el doble o el triple que otro. La forma es la más débil y solo debe usarse como respaldo.

**Cuándo dejar de agregar canales:**

Con 4–5 variables cruzadas en un único gráfico, la carga cognitiva se vuelve muy alta. El observador tiene que recordar qué significa cada color, cada tamaño y cada forma simultáneamente. En esos casos es mejor dividir en múltiples paneles (`col=` en Seaborn) o usar una visualización interactiva con Plotly donde el hover tooltip muestre los detalles.

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

**Contexto teórico:**

El daltonismo (deficiencia de visión del color) afecta a **aproximadamente el 8% de los hombres y al 0.5% de las mujeres** en poblaciones de ascendencia europea. Los tipos más comunes son:

| Tipo | Qué afecta | Prevalencia en hombres |
|---|---|---|
| **Deuteranopía** | No puede distinguir rojo del verde | ~5% |
| **Protanopía** | Dificultad para percibir el rojo | ~2% |
| **Tritanopía** | Dificultad para percibir el azul | <0.1% |

El problema práctico: si usamos una línea roja para "malo" y una verde para "bueno" (la combinación más intuitiva), el 5–7% de los hombres en el salón no puede distinguirlas.

**La paleta colorblind de Seaborn** fue diseñada por los investigadores Masataka Okabe y Kei Ito (2008) específicamente para que sea distinguible por personas con los tipos más comunes de daltonismo. Usa naranja, azul claro, verde oscuro y violeta, que mantienen contraste incluso en deuteranopía.

**La regla del respaldo de forma:**

El color comunica una sola dimensión: el matiz. Si el color desaparece (impresión en B&N, proyector viejo, pantalla en modo económico), toda esa información se pierde. La solución es **siempre respaldar el color con una segunda dimensión visual**:

- En líneas: `style=` cambia el patrón del trazo (continuo, punteado, guionado).
- En puntos: `style=` + `markers=True` cambia la figura geométrica (○, □, △, ×).

El efecto es que el gráfico es legible por **tres caminos redundantes**: color, forma Y posición. Perder uno de los tres no destruye la comunicación.

**Paletas recomendadas en Seaborn:**

| Paleta | Descripción | Cuándo usar |
|---|---|---|
| `"colorblind"` | Paleta Okabe-Ito, diseñada para daltonismo | Primera opción siempre |
| `"muted"` | Colores desaturados, suaves | Reportes formales |
| `"deep"` | Colores saturados con buen contraste | Presentaciones con proyector |

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

**Contexto teórico:**

Matplotlib y Seaborn generan imágenes **rasterizadas** (mapas de píxeles) que se muestran como una foto estática. Una vez renderizada, no se puede interactuar con ella.

Plotly es una empresa de software que desarrolló una librería de visualización basada en **D3.js** (una librería de gráficos en JavaScript). En lugar de generar una imagen, genera **código HTML+JavaScript** que el navegador ejecuta de forma dinámica.

`plotly.express` (o `px`) es la API de alto nivel de Plotly: permite crear gráficos interactivos complejos con muy pocas líneas de código. Internamente, cada función de `px` genera un objeto `plotly.graph_objects.Figure` que contiene toda la definición del gráfico en formato JSON.

**Lo que puede hacer el usuario sin escribir código:**

- **Hover (tooltip):** Al pasar el cursor sobre cualquier punto aparece una etiqueta con los valores exactos de ese punto, el nombre de su categoría, y cualquier columna extra que hayamos incluido con `hover_data=`.
- **Zoom:** Arrastrando un rectángulo en el gráfico hace zoom en esa zona. Los ejes se reescalan automáticamente para mostrar solo esa región.
- **Pan (arrastrar):** Después de hacer zoom, se puede arrastrar el gráfico para moverlo.
- **Leyenda interactiva:** Clic en una categoría en la leyenda la oculta/muestra en el gráfico. Doble clic aísla esa categoría (oculta todas las demás).
- **Reset:** Doble clic en el área del gráfico vuelve al zoom original.

**Plotly Express vs Matplotlib/Seaborn — cuándo usar cada uno:**

| Situación | Herramienta |
|---|---|
| Exploración rápida de datos (uso propio) | Plotly Express |
| Dashboard en una app web | Plotly Express |
| Presentación interactiva en un notebook compartido | Plotly Express |
| Imagen para PDF, Word, PowerPoint | Matplotlib / Seaborn |
| Paper científico o tesis | Matplotlib / Seaborn |
| Reporte para impresión | Matplotlib / Seaborn |

> ⚠️ **Limitación importante:** Los gráficos de Plotly son interactivos en el navegador, pero **no se pueden exportar directamente como imagen** sin instalar la librería `kaleido`. Para reportes impresos, hay que recrear el gráfico con Matplotlib y exportarlo con `plt.savefig()`.

**Parámetros útiles de Plotly Express:**

```python
px.scatter(
    df,
    x="col_x",
    y="col_y",
    color="categoria",            # diferencia por color
    size="variable_numerica",     # tamaño del punto según valor
    hover_data=["col_extra"],     # agrega esta columna al tooltip
    labels={"col_x": "Eje X"},    # renombra ejes en el gráfico (no el DataFrame)
    template="plotly_white",      # tema: fondo blanco limpio
    title="Mi título"
)
```

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

**Contexto teórico:**

Cuando guardamos un gráfico a un archivo, la extensión determina fundamentalmente cómo se almacena la información visual. Hay dos tecnologías completamente distintas:

**Formatos Rasterizados (Mapas de bits):**

Almacenan el gráfico como una grilla rectangular de píxeles de color. Cada píxel es un número que indica su color exacto (en formato RGB: rojo, verde, azul).

- **Resolución fija:** Una vez guardado a 800×600 píxeles, ampliar la imagen la pixela porque no hay información entre los píxeles.
- **DPI (Dots Per Inch):** Define cuántos píxeles por pulgada tiene la imagen al imprimirse.
  - `dpi=72` → resolución de pantalla. Se ve bien en monitor, borroso al imprimir en papel.
  - `dpi=150` → impresión aceptable para uso interno.
  - `dpi=300` → **estándar profesional**. Nítido en cualquier tamaño de impresión en papel.
  - `dpi=600` → estándar de publicaciones científicas de alta calidad.

| Formato | Características | Cuándo usar |
|---|---|---|
| `.png` | Sin pérdida de calidad, fondo transparente posible | Presentaciones, web, correo |
| `.jpg` | Compresión con pérdida, fondo blanco siempre | Fotografías, web liviano |

**Formatos Vectoriales (Basados en fórmulas matemáticas):**

No almacenan píxeles. Almacenan las **instrucciones matemáticas** para dibujar cada elemento: "dibujá una línea desde el punto (x1,y1) hasta (x2,y2) con color #2c3e50 y grosor 2pt". Al mostrar el archivo, el software ejecuta esas instrucciones recalculándolas para el tamaño actual.

- **Resolución infinita:** Ampliar un vector al 1000% no pierde nitidez porque las instrucciones se recalculan con la nueva escala.
- **Editable:** Un archivo `.svg` se puede abrir en Inkscape o Illustrator y editar cada elemento individualmente.

| Formato | Características | Cuándo usar |
|---|---|---|
| `.pdf` | Estándar de documentos, soporte universal | Papers, tesis, reportes para imprimir |
| `.svg` | Formato web estándar, editable en editores gráficos | Web, presentaciones editables |

**Regla de decisión:**

> Si el gráfico va a un documento de texto (Word, PDF de tesis, informe técnico) → **PDF vectorial**.
> Si el gráfico va a una presentación, correo o web → **PNG a 300dpi**.

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

**Contexto teórico:**

Cuando Matplotlib calcula la imagen a guardar con `savefig()`, usa el `figsize` que se especificó al crear la figura para determinar el área de la imagen. El problema es que los **elementos decorativos** (títulos, etiquetas de ejes, leyendas) pueden quedar **fuera** del área `figsize` original, especialmente cuando:

- El título tiene muchas palabras o un `pad` extra.
- Las etiquetas del eje X están rotadas (`plt.xticks(rotation=45)`) y quedan más largas.
- La leyenda está posicionada fuera del gráfico.
- El `xlabel` tiene un `labelpad` grande.

`bbox_inches='tight'` le ordena a Matplotlib que **recalcule el bounding box** (área delimitadora) de la imagen justo antes de guardar, incluyendo absolutamente todo el contenido visual que exista fuera del área original.

**"Bounding box"** significa literalmente "caja que delimita". El bounding box tight es el rectángulo más pequeño posible que contiene todo el contenido del gráfico, desde el borde del título hasta el borde de la etiqueta del eje X, pasando por cualquier leyenda o anotación.

**`plt.tight_layout()` vs `bbox_inches='tight'` — son cosas distintas:**

| | `plt.tight_layout()` | `bbox_inches='tight'` |
|---|---|---|
| **Qué hace** | Ajusta el espaciado ENTRE subgráficos para que no se superpongan | Ajusta el ÁREA DE LA IMAGEN al guardar |
| **Cuándo actúa** | Al renderizar en pantalla | Solo al guardar con `savefig()` |
| **Para qué sirve** | Evitar que los títulos de un subplot tapen las etiquetas del otro | Evitar que el texto se corte en la imagen guardada |

En la práctica, se usan AMBOS juntos:
```python
plt.tight_layout()                                        # ajusta el layout interno
plt.savefig("reporte.png", dpi=300, bbox_inches='tight')  # encuadre perfecto al guardar
```

**Parámetros adicionales de `savefig()`:**

| Parámetro | Qué hace | Cuándo usarlo |
|---|---|---|
| `dpi=300` | Resolución de la imagen | Siempre para PNG de calidad |
| `bbox_inches='tight'` | Encuadre que no corta texto | Siempre, es gratuito |
| `transparent=True` | Fondo transparente (sin rectángulo blanco) | PNG sobre slides de color |
| `facecolor='white'` | Fondo blanco explícito | Cuando el tema del notebook tiene fondo oscuro |

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
