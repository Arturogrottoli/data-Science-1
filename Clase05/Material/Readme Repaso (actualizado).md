# Readme Repaso — Clase 05: Visualizaciones Avanzadas en Data Science

Guion de clase para el profesor. Cada sección incluye qué decir, qué mostrar y qué ejecutar en el notebook. La Parte 2 sigue el orden exacto de las filminas de `Clase05.html` (37 filminas) e interleava con el notebook `Semana_5_Visualizaciones_Avanzadas_en_Data_Science_ (actualizado).ipynb`. Cada vez que hay código, el texto marca explícitamente **👉 cuándo pasar a Colab** y **👉 cuándo volver a las filminas**, para que sea fácil ir alternando ventanas en vivo.

---

## Antes de empezar — Encuadre de la clase

> "La clase de hoy tiene dos partes. En la primera hacemos un repaso de todo lo visto hasta ahora: variables, control de flujo, NumPy y Pandas. En la segunda parte arrancamos con el tema nuevo: visualizaciones avanzadas, siguiendo las filminas y el notebook. Al final tienen una actividad práctica con un dataset real de tráfico aéreo, y una pre-entrega evaluable sobre el dataset de su propio proyecto."

Abrir tres archivos antes de arrancar, y dejarlos en pestañas separadas para poder alternar rápido:
- Repaso: `Repaso_Data_Science_I_Fundamentos_para_la_Ciencia_de_Datos_.ipynb`
- Filminas: `Clase05.html`
- Código de la Parte 2: `Semana_5_Visualizaciones_Avanzadas_en_Data_Science_ (actualizado).ipynb`

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

**Qué hacer con los NaN:** `fillna()` reemplaza cada NaN por un valor — la pregunta es CUÁL valor, y depende de qué signifique el hueco. No son tres pasos que se hacen juntos: son tres estrategias distintas, se elige UNA según el caso. Con los datos del ejemplo de abajo (`Unidades = [10, 15, NaN, 5, 20]`, falta la 3ª fila) esto es lo que haría cada una:

| Estrategia | Qué hace | Cuándo usarla | Resultado con este dataset |
|---|---|---|---|
| `fillna(0)` | Reemplaza el NaN por `0` | El hueco significa "no pasó nada" (sin ventas ese día) | La fila queda en `0` unidades |
| `fillna(df["col"].mean())` | Reemplaza el NaN por el promedio de la columna | No hay razón para pensar que fue cero, solo no se registró el dato | La fila queda en `12.5` (promedio de `10, 15, 5, 20`) |
| `fillna(method="ffill")` | Repite el último valor válido anterior | Series temporales, donde lo más probable es que el valor se mantuvo | La fila copia el valor de la fila de arriba (`15`) |

**El ejemplo de abajo usa `fillna(0)`** porque tratamos "Unidades faltante" como "no se vendió nada ese día" — es la más simple de las tres, y la que tiene sentido para este negocio en particular. Las otras dos quedan como referencia para cuando el contexto sea distinto (no se usan en el código de abajo).

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

df["Unidades"] = df["Unidades"].fillna(0)  # busca TODOS los NaN de esa columna y los reemplaza por 0;
                                            # sin esto, Unidades * Precio_Unitario daría NaN en esa fila.
                                            # Elegimos "0" acá: sin dato de unidades = no se vendió nada ese día
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

### Mapa rápido: Filminas ↔ Bloques del Notebook

| Filminas | Módulo (Clase05.html) | Bloque del notebook |
|---|---|---|
| 01 | Portada | — |
| 02–06 | Fundamentos de Matplotlib | Bloque 1 |
| 07–10 | Análisis Univariado: Histogramas y KDE | Bloque 2 |
| 11–15 | Seaborn: Distribución y Comparación | Bloque 3 |
| 16 | Break | — |
| 17–22 | Storytelling y Diseño de Dashboards | Bloque 4 |
| 23–24 | Arquitectura Visual y Regla de Lectura en "Z" | Bloque 5 (conceptual, sin código) |
| 25–29 | Principios de UI/UX y Jerarquía Visual | Bloque 6 |
| 30–32 | Práctica: Auditoría UX de Dashboards | Bloque 7 (sin código) |
| 33–36 | Pre-entrega: Limpieza y Documentación del Dataset | Bloque 8 |
| 37 | Cierre | — |

> Todo lo que quedó fuera del `Clase 05.docx` vigente (GridSpec, series temporales, análisis multivariado con `hue/style/size`, Plotly, formatos de exportación) sigue disponible en el **Anexo** del notebook — no tiene filmina asociada ni se pide en la Pre-entrega, pero está para quien quiera ir más allá.

**Qué decir al arrancar:**

> "Antes hacíamos gráficos como parte de otras clases. Hoy la visualización es el tema central: no solo cómo hacer un gráfico en Python, sino cómo diseñarlo para que comunique bien, y cómo armar un dashboard completo que un ejecutivo entienda en 3 segundos. Vamos a ir filmina por filmina, y en cada bloque les voy a avisar cuándo pasamos al notebook y cuándo volvemos a las filminas."

---

### BLOQUE 1 — Fundamentos de Matplotlib *(Filminas 02–06)*

**Filmina 02 — División de módulo.** Da el pie para el primer bloque: Figure, Axes, los dos estilos de trabajo, subplots y personalización. Es la filmina de transición — antes de entrar de lleno en Figure/Axes, conviene parar un minuto y dar la introducción general que sigue, porque todo lo que viene después (Matplotlib, Seaborn, Plotly) da por sentado que la clase entiende para qué sirve graficar en primer lugar.

#### Introducción — ¿Por Qué Graficar? ¿Qué es Matplotlib? *(no está en una filmina puntual, es el puente antes de arrancar)*

**Qué decir:**

**¿Por qué está bueno hacer gráficos?**

El cerebro humano no está diseñado para leer tablas de números — está diseñado para detectar patrones, formas y colores casi instantáneamente. Si le mostramos a alguien una tabla con 10.000 filas de ventas diarias, va a tardar minutos (u horas) en notar que las ventas cayeron a la mitad en marzo. Si le mostramos un gráfico de línea con esos mismos 10.000 datos, la caída se ve en menos de un segundo. Un gráfico no agrega información nueva a los datos — lo que hace es **traducir números a percepción visual**, que es el canal más rápido que tenemos para procesar información.

**¿Para qué nos sirven los gráficos, en la práctica?**

En el trabajo real de un/a Data Scientist, graficar cumple varios roles distintos, no uno solo:

- **Para explorar** (EDA — Análisis Exploratorio de Datos): antes de construir cualquier modelo, hay que "mirar" los datos — ver si hay outliers, si una variable está sesgada, si dos variables se relacionan. Esto casi nunca se hace leyendo la tabla cruda, se hace graficando.
- **Para detectar errores**: un valor cargado mal (por ejemplo, una edad de 300 años) salta a la vista en un histograma, pero puede pasar desapercibido en una tabla de 50.000 filas.
- **Para comunicar resultados**: un análisis técnico impecable no sirve de nada si la persona que toma la decisión (un gerente, un cliente) no lo entiende. El gráfico es el puente entre el análisis y la decisión.
- **Para convencer con evidencia**: mostrar "las ventas subieron 12% en la región norte" es una afirmación; mostrarlo en un gráfico de barras comparando regiones es una evidencia que se puede verificar de un vistazo.

Esta idea — graficar no es un paso decorativo al final, es una herramienta de análisis en sí misma — es el hilo conductor de toda la clase de hoy, y se repite con otras palabras en la Filmina 18 ("El Porqué de la Visualización") cuando lleguemos al Bloque 4.

**¿Qué es Matplotlib?**

Matplotlib es la librería base de visualización en el ecosistema de Python — la más antigua, la más usada, y la que está "por debajo" de casi todas las demás. Fue creada a principios de los 2000 buscando reproducir en Python las capacidades de gráficos de MATLAB (de ahí el nombre: Mat-plot-lib). Hoy es, junto con NumPy y Pandas, una de las tres librerías fundacionales de la ciencia de datos en Python.

Lo importante para entender la clase de hoy: **Seaborn (que vemos en el Bloque 3) no reemplaza a Matplotlib — está construido encima de Matplotlib.** Cuando alguien llama a `sns.boxplot()`, por detrás Seaborn termina llamando funciones de Matplotlib para dibujar. Aprender Matplotlib primero es lo que permite, más adelante, entender qué está pasando "por debajo" cuando se usa Seaborn, y poder personalizar un gráfico de Seaborn con comandos de Matplotlib cuando Seaborn solo no alcanza.

**¿Cómo se usa, en términos generales?**

El flujo de trabajo típico con Matplotlib, sin importar qué tan simple o complejo sea el gráfico, sigue siempre los mismos pasos:

1. **Importar la librería:** `import matplotlib.pyplot as plt` — por convención casi universal se importa con el alias `plt`.
2. **Crear el lienzo y el gráfico:** con `plt.subplots()`, que devuelve una `Figure` (el lienzo) y uno o más `Axes` (el gráfico en sí) — esto lo vamos a ver en detalle en la próxima filmina.
3. **Dibujar los datos:** con funciones como `ax.plot()` (líneas), `ax.bar()` (barras), `ax.scatter()` (dispersión), etc.
4. **Personalizar:** título, etiquetas de los ejes, leyenda, colores — para que el gráfico comunique, no solo que "esté" el dato.
5. **Mostrar o guardar:** `plt.show()` para verlo en pantalla, o `plt.savefig()` para exportarlo a un archivo.

Con este flujo general en la cabeza, ahora sí se puede entrar en el detalle de cada pieza — empezando por la diferencia entre `Figure`, `Axes` y `Axis` en la próxima filmina.

**Preguntar a la clase:**

> ¿Alguna vez tomaron una decisión (personal o de trabajo) mirando un gráfico en vez de una tabla de números? ¿Por qué el gráfico les resultó más fácil de leer?

---

#### Filmina 03 — Figure, Axes y Axis

**Qué decir (ampliando lo que dice la filmina):**

La filmina resume la diferencia en tres bullets cortos. Para que quede realmente claro y no se quede en una definición de memoria, conviene desarrollar cada uno por separado — no alcanza con leer la filmina, hay que explicar qué implica cada objeto en la práctica.

**`Figure` — el contenedor de nivel superior**

La `Figure` es el objeto más alto en la jerarquía: es literalmente el archivo que se termina generando cuando uno guarda un gráfico (`.png`, `.pdf`). No dibuja ningún dato por sí misma — su trabajo es puramente administrativo: define el tamaño total en pulgadas (`figsize`), el color de fondo de toda la ventana, la resolución al exportar (`dpi`), y actúa como el "dueño" de todos los gráficos que viven adentro de ella. Una analogía útil: la `Figure` es la hoja de papel en blanco, o el marco de una ventana — importa su tamaño y su borde, pero no lo que se dibuja adentro. En código, `fig` es la variable que casi siempre se usa solo dos veces: al crearla (`fig, ax = plt.subplots()`) y al guardarla (`fig.savefig(...)`) — el resto del trabajo se hace sobre `ax`.

**`Axes` — el área de dibujo, "el gráfico" propiamente dicho**

El `Axes` es donde realmente pasa todo: es el objeto que contiene los datos dibujados (las barras, los puntos, las líneas), el título, la leyenda, y los dos ejes (X e Y) con sus etiquetas. Cuando alguien dice coloquialmente "el gráfico", casi siempre se está refiriendo al `Axes`, no a la `Figure`. Es, con diferencia, el objeto con el que más se trabaja línea a línea: `ax.plot()`, `ax.set_title()`, `ax.legend()`, `ax.set_xlim()` — todos estos comandos le hablan al `Axes`. El nombre en inglés es plural ("Axes") porque técnicamente agrupa el conjunto de ejes X e Y que lo componen, pero en la práctica se usa para nombrar a **un** gráfico individual completo. Siguiendo la analogía de la hoja de papel: si la `Figure` es la hoja completa, cada `Axes` es un dibujo puntual hecho sobre esa hoja — y una hoja puede tener uno o varios dibujos.

**`Axis` — la regla graduada de un eje**

El `Axis` (sin la "e" al final, y en singular) es la pieza más chica de las tres: es solamente la línea numerada de UN eje — el eje X o el eje Y — con sus marcas (*ticks*) y sus etiquetas numéricas. Un `Axes` siempre tiene dos objetos `Axis` adentro (`ax.xaxis` y `ax.yaxis`). Rara vez se manipula el `Axis` directamente en el uso cotidiano — se hace sobre todo cuando hace falta un control muy fino del formato de los números o las fechas que aparecen en un eje (por ejemplo, mostrar fechas como "Ene 2024" en vez de "2024-01-01", algo que se ve en el Anexo del notebook con `matplotlib.dates`). Para el 95% de los casos de uso normales, alcanza con trabajar sobre `ax` (Axes) y nunca hace falta tocar `axis` directamente.

**La confusión más común, explicada con una regla práctica:** `Axes` (el objeto gráfico completo, con mayúscula inicial cuando se lo nombra como clase) no es lo mismo que `Axis` (un eje puntual). Casi todo lo que se personaliza en un gráfico — título, etiquetas de los ejes, límites, leyenda, color de fondo del gráfico — se hace llamando métodos sobre el objeto `ax` (Axes). Si en algún momento del código aparece la palabra `axis` a secas, casi seguro se está hablando de un eje individual (X o Y), no del gráfico completo.

Una `Figure` puede tener un solo `Axes` (un gráfico sencillo) o diez `Axes` (un dashboard completo) — este último caso lo vamos a ver en detalle en el Bloque 6.

Esta filmina es 100% teoría, todavía no hay código para correr — se sigue derecho a la Filmina 04.

**Preguntar a la clase:**

> Si armamos un dashboard con 4 gráficos en una sola imagen, ¿cuántas Figure y cuántos Axes hay?

**Respuesta:** Una sola `Figure` (la imagen final completa) y 4 `Axes` (uno por cada gráfico individual dentro de ella).

#### Filmina 04 — Pyplot vs. Orientado a Objetos

**Qué decir (ampliando lo que dice la filmina):**

La filmina muestra los dos bloques de código en paralelo. Antes de correrlos, conviene explicar el POR QUÉ de cada estilo por separado, no solo contrastarlos de pasada:

**Estilo Pyplot (implícito).** Se le habla directo al módulo `plt` (`plt.plot()`, `plt.title()`). Por detrás, Matplotlib mantiene la idea de "la figura activa" y "el Axes activo" — cada vez que se llama a una función de `plt`, Matplotlib busca cuál fue el último gráfico creado y aplica el comando ahí. Es un mecanismo cómodo cuando hay un solo gráfico en pantalla (no hay ambigüedad posible sobre "cuál es el activo"), pero se vuelve un problema en cuanto se crea un segundo `Axes`: `plt.title()` le va a poner el título al Axes que Matplotlib considera activo en ese momento, que no siempre es el que la persona que programa tiene en mente. Es la fuente número uno de bugs silenciosos para quien recién empieza con Matplotlib — el código no tira error, simplemente el título aparece en el gráfico "equivocado".

**Estilo Orientado a Objetos — OO (explícito, el recomendado).** Se crean variables con nombre propio (`fig`, `ax`) para la Figure y el Axes, y cada instrucción se dirige explícitamente a esa variable (`ax.plot()`, `ax.set_title()`). No existe el concepto de "objeto activo": cada línea de código dice exactamente a qué objeto le está hablando, sin importar cuántos Axes haya en la Figure. Esto es lo que permite, más adelante, iterar sobre una lista de Axes en un `for` (algo muy común al armar dashboards) sin perder el control de cuál es cuál.

En Data Science se recomienda casi siempre el estilo OO porque en cuanto se arma un panel comparativo (algo constante en EDA), el estilo Pyplot se vuelve un lío.

**Qué buscamos ver con este ejemplo:** el mismo gráfico de línea hecho de las dos formas, para comprobar en carne propia que con UN solo gráfico el resultado visual es idéntico — la diferencia entre los dos estilos no se nota todavía acá, recién se va a notar en la Filmina 05 cuando aparezca un segundo Axes.

👉 **Acá pasás a Colab.** Abrí `Semana_5_Visualizaciones_Avanzadas_en_Data_Science_ (actualizado).ipynb`, andá al **Bloque 1** y corré la celda del ejemplo "Estilo Pyplot vs. Estilo Orientado a Objetos".

**Ejecutar:**
```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# ── Estilo Pyplot: le hablamos directo a "plt", sin guardar ninguna referencia ──
plt.plot(x, y, color="steelblue", marker="o")
plt.title("Estilo Pyplot (implícito)")
plt.show()

# ── Estilo Orientado a Objetos: creamos fig y ax, y le hablamos a "ax" ──
fig, ax = plt.subplots()           # fig = la Figure completa; ax = el único Axes adentro
ax.plot(x, y, color="darkorange", marker="o")
ax.set_title("Estilo Orientado a Objetos (explícito)")
plt.show()
```

**Qué hace cada línea:**
- `import matplotlib.pyplot as plt`: trae el módulo de Matplotlib con el alias estándar `plt`. Sin esta línea, ningún comando de las siguientes existe.
- `x = [1, 2, 3, 4, 5]` / `y = [2, 3, 5, 7, 11]`: dos listas comunes de Python con los valores a graficar — Matplotlib no necesita NumPy ni Pandas para graficar algo simple, listas alcanzan.
- `plt.plot(x, y, color="steelblue", marker="o")`: dibuja la línea. Como no hay ningún `Axes` creado todavía, Matplotlib crea uno automáticamente "por detrás de escena" y lo marca como el activo. `marker="o"` agrega un círculo en cada punto de dato, además de la línea que los conecta.
- `plt.title(...)`: le pone título al Axes que Matplotlib considera activo en este momento — que es el que se acaba de crear en la línea anterior.
- `plt.show()`: renderiza y muestra en pantalla todo lo que se dibujó hasta acá.
- `fig, ax = plt.subplots()`: esta vez SÍ se crean explícitamente la Figure y el Axes, y quedan guardados en variables con nombre. A partir de esta línea, ya no hay "objeto activo" implícito — todo se hace hablándole directamente a `ax`.
- `ax.plot(...)`: idéntico a `plt.plot()` en el resultado visual, pero la diferencia es que este comando le habla puntualmente al objeto `ax`, no al "lo que sea que esté activo".
- `ax.set_title(...)`: pone el título específicamente en `ax` — no hay forma de que termine en otro Axes por error, porque no hay otro Axes en esta celda.

**Qué mostrar en detalle:**
- Con un solo gráfico, la diferencia visual es nula — hacer notar que el problema aparece recién con 2+ Axes (se ve en la próxima filmina).
- Mostrar `type(fig)` → `Figure`, `type(ax)` → `AxesSubplot`.

👉 **Volvés a las filminas, Filmina 05.**

#### Filmina 05 — `subplots()`: Varias Vistas en un Panel

**Qué decir (ampliando lo que dice la filmina):**

`plt.subplots(nrows, ncols)` crea de una vez la `Figure` y todos los `Axes` vacíos, organizados en una grilla, y los devuelve ya empaquetados: la Figure sola, y los Axes en una tupla (o un array de tuplas si la grilla tiene más de una fila). A partir de ahí, cada `ax.algo()` dibuja SOLO en ese Axes puntual — se termina la ambigüedad de no saber dónde está dibujando `plt.plot()`, que es justo el problema del estilo Pyplot que vimos en la filmina anterior: acá cada Axes tiene su propia variable, así que no hay "objeto activo" que confundir.

Esto es central para el **Análisis Exploratorio de Datos (EDA)**: casi siempre se necesita comparar dos vistas relacionadas en una sola imagen — la filmina menciona el ejemplo de un proyecto inmobiliario: un histograma de precios al lado de un scatter de metros² vs. precio. La razón de fondo es que el cerebro compara mucho mejor dos cosas que están una al lado de la otra, en la misma escala visual, que dos cosas que hay que ver en pestañas separadas o gráficos sueltos.

**Qué buscamos ver con este ejemplo:** dos series de datos distintas (ventas de enero y de febrero), cada una con el tipo de gráfico que mejor le queda (línea para ver evolución, barras para comparar valores puntuales), dibujadas una al lado de la otra en el mismo panel — y un título general para toda la Figure, además de los títulos individuales de cada Axes.

👉 **Acá pasás a Colab.** Seguís en el **Bloque 1**, corré la celda de "Varias Vistas en un Mismo Panel" (Enero vs. Febrero).

**Ejecutar:**
```python
import matplotlib.pyplot as plt

semanas = ["Sem 1", "Sem 2", "Sem 3", "Sem 4"]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ventas_enero = [10, 20, 15, 30]
ax1.plot(semanas, ventas_enero, color="blue", marker="o")
ax1.set_title("Ventas de Enero (miles $)")

ventas_febrero = [12, 18, 22, 25]
ax2.bar(semanas, ventas_febrero, color="orange")
ax2.set_title("Ventas de Febrero (miles $)")

fig.suptitle("Un solo panel (Figure) con dos gráficos (Axes): Enero vs Febrero", fontsize=14)
plt.show()
```

**Qué hace cada línea:**
- `semanas = [...]`: la lista compartida que va a servir de eje X para ambos gráficos — las mismas 4 semanas se usan en los dos paneles.
- `fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))`: pide una grilla de 1 fila y 2 columnas — dos Axes lado a lado. Como es una sola fila, `plt.subplots` devuelve los Axes en una tupla simple `(ax1, ax2)` que se desempaqueta directo en dos variables con nombre propio. `figsize=(10, 4)` fija el tamaño de la Figure completa en pulgadas (10 de ancho, 4 de alto) — sin esto, el tamaño por defecto suele quedar chico para dos gráficos lado a lado.
- `ventas_enero = [...]` y `ax1.plot(...)`: dibuja la serie de enero como línea en el Axes de la izquierda (`ax1`). `marker="o"` marca cada dato puntual sobre la línea.
- `ax1.set_title(...)`: título específico del gráfico de la izquierda — solo afecta a `ax1`.
- `ventas_febrero = [...]` y `ax2.bar(...)`: dibuja la serie de febrero como barras en el Axes de la derecha (`ax2`) — un tipo de gráfico distinto al de la izquierda, porque acá interesa comparar valores puntuales semana a semana, no tanto ver la tendencia continua.
- `ax2.set_title(...)`: título específico del gráfico de la derecha.
- `fig.suptitle(...)`: a diferencia de los dos anteriores, este comando le habla a la `fig` (la Figure completa), no a un Axes — pone un título que "corona" a los dos gráficos juntos, explicando qué relación hay entre ambos paneles.
- `plt.show()`: renderiza toda la Figure, con sus dos Axes y sus tres títulos (dos individuales + uno general).

**Qué mostrar en detalle:**
- `fig.suptitle()` vs `ax.set_title()`: el primero titula TODA la figura, el segundo titula un solo Axes. Mostrar la diferencia sacando uno de los dos.
- `(ax1, ax2) = plt.subplots(...)`: el desempaquetado automático de la tupla de Axes en dos variables con nombre.

👉 **Volvés a las filminas, Filmina 06.**

#### Filmina 06 — Personalización Completa

**Qué decir (ampliando lo que dice la filmina):**

La filmina lista los comandos (título, labels, leyenda, límites, tight_layout) como una tabla corta. Vale la pena remarcar el POR QUÉ de cada uno por separado, no solo el QUÉ: un gráfico sin etiquetas es solo una forma abstracta — no comunica nada por sí solo, aunque los datos que representa sean correctos.

- **`ax.set_title()`**: el nombre de la historia que se está contando. Un buen título dice qué se mide y, si corresponde, en qué período — no es solo un rótulo decorativo, es la primera (y a veces única) frase que alguien va a leer del gráfico.
- **`ax.set_xlabel()` / `ax.set_ylabel()`**: qué mide cada eje, en qué unidades. "Ingresos (USD)" es información concreta; "Ingresos" a secas obliga a quien lee a adivinar si son pesos, dólares o millones — una ambigüedad que en un reporte real puede llevar a una mala decisión.
- **`ax.legend()`**: obligatoria en cuanto hay más de una serie en el mismo Axes. Sin ella, dos líneas de distinto color no tienen forma de identificarse — la leyenda usa el texto que se le haya pasado en el parámetro `label=` de cada `plot()`.
- **`ax.set_xlim()` / `ax.set_ylim()`**: el rango visible de cada eje lo decide quien programa, no Matplotlib solo — por defecto, Matplotlib ajusta los límites para que "entren" justo los datos, lo cual puede exagerar visualmente variaciones chicas. Esto se vuelve crítico más adelante cuando hablemos de ejes truncados en el Bloque 4: fijar el límite inferior en `0` es, muchas veces, una decisión ética, no solo estética.
- **`fig.tight_layout()`**: el salvavidas cuando los subplots se pisan entre sí — recalcula automáticamente los espacios entre gráficos, títulos y etiquetas para que nada quede cortado o superpuesto.

**Qué buscamos ver con este ejemplo:** dos series de datos en el mismo Axes (Producto A y B), con todos los elementos de personalización juntos en un solo gráfico — título con negrita, etiquetas de ejes con unidades, leyenda, grilla suave, y un límite de eje Y fijado a propósito.

👉 **Acá pasás a Colab.** Seguís en el **Bloque 1**, última celda: "Personalización Completa" (ventas Producto A vs. B).

**Ejecutar:**
```python
import matplotlib.pyplot as plt

años = [2020, 2021, 2022, 2023, 2024]
ventas_A = [50, 55, 63, 70, 90]
ventas_B = [48, 52, 58, 65, 68]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(años, ventas_A, label='Producto A', color='blue', linewidth=2)
ax.plot(años, ventas_B, label='Producto B', color='orange', linestyle='--')

ax.set_title("Evolución de Ventas Anuales", fontsize=14, fontweight='bold')
ax.set_xlabel("Año")
ax.set_ylabel("Ventas (Miles de Unidades)")
ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.show()
```

**Qué hace cada línea:**
- `años`, `ventas_A`, `ventas_B`: tres listas paralelas — el mismo índice en cada una corresponde al mismo año.
- `fig, ax = plt.subplots(figsize=(8, 5))`: un solo Axes esta vez (no hace falta grilla), con tamaño 8x5 pulgadas.
- `ax.plot(años, ventas_A, label='Producto A', color='blue', linewidth=2)`: dibuja la primera línea. El parámetro `label=` no se ve en el gráfico todavía — es el texto que va a usar `ax.legend()` más abajo. `linewidth=2` engrosa la línea para que se distinga bien.
- `ax.plot(años, ventas_B, label='Producto B', color='orange', linestyle='--')`: segunda línea en el mismo Axes — `linestyle='--'` la dibuja punteada, una forma extra (además del color) de diferenciarla, útil para accesibilidad (lo vamos a retomar en el Bloque 6).
- `ax.set_title(..., fontsize=14, fontweight='bold')`: título con tamaño de fuente explícito y en negrita, para que jerárquicamente destaque por sobre las etiquetas de los ejes.
- `ax.set_xlabel(...)` / `ax.set_ylabel(...)`: las unidades quedan explícitas en el propio texto de la etiqueta ("Miles de Unidades"), no hay que adivinarlas.
- `ax.legend()`: sin argumentos — Matplotlib arma la leyenda automáticamente usando los `label=` que se pasaron en cada `plot()`.
- `ax.grid(True, linestyle=':', alpha=0.6)`: activa una grilla de fondo, punteada (`linestyle=':'`) y semi-transparente (`alpha=0.6`) para que ayude a leer valores sin dominar visualmente al dato.
- `ax.set_ylim(0, 100)`: fuerza el eje Y a arrancar en 0 — decisión consciente, no dejada al azar de Matplotlib.
- `plt.tight_layout()`: ajusta espacios finales antes de mostrar.
- `plt.show()`: renderiza todo el gráfico final.

**Qué mostrar en detalle — Errores comunes** (esto no está en la filmina, agregarlo de palabra): confundir `Axes` con `Axis`; no guardar `fig, ax` (se pierde control fino apenas hay más de un gráfico); el "gráfico de espagueti" (10 líneas de colores en un solo Axes — mejor separar en subplots); olvidar las unidades del eje.

👉 **Volvés a las filminas, Filmina 07 (división de módulo).**

---

### BLOQUE 2 — Análisis Univariado: Histogramas y KDE *(Filminas 07–10)*

**Filmina 07 — División de módulo.**

#### Filmina 08 — El Histograma y la Cantidad de Bins

**Qué decir (ampliando lo que dice la filmina):**

Antes de buscar relaciones entre variables, el primer paso de todo análisis es mirar cada variable **por separado**. La filmina lista cuatro preguntas — vale la pena desarrollar cada una, porque son literalmente la checklist mental que hay que recorrer frente a cualquier variable numérica nueva:

- **¿Cuál es el valor más común?** — dónde se concentra la mayor densidad de datos, el "centro de gravedad" de la variable.
- **¿Están muy dispersos o se concentran en un solo punto?** — qué tan ancha es la distribución: una variable puede tener el mismo promedio y comportarse muy distinto según qué tan esparcidos estén sus valores alrededor de ese promedio.
- **¿Hay valores extraños (outliers) que se alejan del resto?** — datos que podrían ser errores de carga, o podrían ser el caso más interesante del dataset (un cliente excepcional, un evento atípico real).
- **¿La distribución es simétrica o tiene "cola" hacia un lado?** — esto determina, entre otras cosas, si conviene usar la media o la mediana como resumen, y qué tipo de gráfico la representa mejor (lo vamos a ver en la tabla de la Filmina 10).

Esta checklist va a reaparecer, con las mismas cuatro preguntas, en el Bloque 8 (Pre-entrega) — es el punto de partida obligado de cualquier EDA.

El **histograma** divide el rango de una variable numérica en intervalos (*bins*) y cuenta cuántos datos caen en cada uno: el eje X muestra los intervalos, el eje Y la cantidad de datos que cayó en cada uno. La cantidad de bins cambia completamente el mensaje, y esto la filmina lo dice pero sin ejemplo visual — conviene remarcarlo oralmente con el mecanismo detrás de cada caso:
- **Pocos bins:** cada barra agrupa un rango muy amplio de valores, promediando internamente toda la variación que hay dentro de ese rango — el resultado es una imagen demasiado general que puede ocultar, por ejemplo, que la distribución en realidad tiene dos picos separados (dos subgrupos mezclados).
- **Muchos bins:** cada barra representa un rango tan angosto que empieza a reflejar el ruido aleatorio de la muestra, no la forma real de la población — el gráfico se vuelve una serie de picos aislados sin patrón reconocible, casi como ver el "grano" de los datos en vez de su forma.

No hay un número "correcto" universal — hay que probar un par de valores y quedarse con el que mejor cuenta la historia sin mentir ni esconder. El código de esta idea se corre recién en la próxima filmina, junto con el KDE.

#### Filminas 09–10 — KDE y Lectura de la Forma

**Qué decir (ampliando lo que dicen las filminas):**

El **KDE** (*Kernel Density Estimation*, estimación de densidad por kernel) resuelve el problema de "cuántos bins elegir" de otra manera: en vez de agrupar en bloques rígidos, coloca una pequeña curva suave (una "colina", técnicamente un *kernel*) centrada exactamente en cada observación individual, y después suma todas esas colinas superpuestas. El resultado es una única curva continua que muestra la forma subyacente de los datos sin depender de dónde exactamente se cortan los bins — no hay una decisión arbitraria de ancho de intervalo escondida atrás.

Con el histograma y el KDE en mano, se puede leer la tabla que trae la Filmina 10 — vale la pena explicar cada fila con su implicación práctica, no solo leerla:

| Patrón | Qué significa | Por qué importa |
|---|---|---|
| **Simetría** | Izquierda espejo de la derecha (Campana de Gauss). | La media y la mediana coinciden — cualquiera de las dos resume bien a la variable. |
| **Sesgo a la derecha** | Cola larga hacia valores altos (salarios, precios de casas). | La media queda "arrastrada" hacia arriba por los valores extremos — la mediana suele representar mejor al dato típico. |
| **Sesgo a la izquierda** | Cola larga hacia valores bajos. | Caso simétricamente opuesto al anterior — menos común en variables de negocio, pero aparece en variables acotadas por arriba (ej. calificaciones, porcentajes). |
| **Multimodalidad** | Dos o más "jorobas" → hay subgrupos mezclados en la misma variable (ej. alturas de hombres y mujeres juntas). | Es una señal de que conviene segmentar la variable antes de analizarla — tratarla como un solo grupo homogéneo esconde la estructura real. |

**Qué buscamos ver con este ejemplo:** el mismo dato (`total_bill`, el monto de la cuenta en el dataset de propinas) graficado tres veces con distinta cantidad de bins, para comparar en vivo el efecto de "pocos bins / muchos bins / cantidad razonable + KDE" que se explicó en la Filmina 08.

👉 **Acá pasás a Colab.** Abrí el **Bloque 2** y corré la celda con los 3 subplots de bins (5, 60, 20+KDE).

**Ejecutar:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

sns.histplot(data=tips, x="total_bill", bins=5, ax=axes[0])
axes[0].set_title("Muy pocos bins (5): imagen demasiado general")

sns.histplot(data=tips, x="total_bill", bins=60, ax=axes[1])
axes[1].set_title("Demasiados bins (60): ruidoso")

sns.histplot(data=tips, x="total_bill", bins=20, kde=True, ax=axes[2], color="teal")
axes[2].set_title("20 bins + KDE: forma clara")

plt.tight_layout()
plt.show()
```

**Qué hace cada línea:**
- `tips = sns.load_dataset("tips")`: carga el dataset de ejemplo "propinas de restaurante" que trae Seaborn incorporado — no requiere ningún archivo local, se descarga (o se lee de caché) automáticamente.
- `fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))`: crea 3 Axes en una fila — uno para cada versión del mismo histograma. `axes` acá es un array de 3 elementos (`axes[0]`, `axes[1]`, `axes[2]`), no una tupla de 2 como en el ejemplo anterior de Enero/Febrero.
- `sns.histplot(data=tips, x="total_bill", bins=5, ax=axes[0])`: dibuja el histograma con solo 5 bins, en el primer Axes. `data=tips` le pasa el DataFrame completo, y `x="total_bill"` le dice qué columna graficar — Seaborn se encarga de contar cuántas filas caen en cada uno de los 5 intervalos.
- `axes[0].set_title(...)`: título descriptivo de qué decisión se tomó en ese panel puntual.
- `sns.histplot(..., bins=60, ax=axes[1])`: mismo dato, mismo tipo de gráfico, pero con 60 bins — el único parámetro que cambia es `bins`, para que la comparación sea limpia.
- `sns.histplot(..., bins=20, kde=True, ax=axes[2], color="teal")`: acá se agrega `kde=True`, que le pide a Seaborn que superponga la curva KDE sobre las barras del histograma, además de usar una cantidad de bins intermedia (20) que ya se ve razonable.
- `plt.tight_layout()`: evita que los 3 títulos y las 3 etiquetas de eje se pisen entre paneles.
- `plt.show()`: renderiza los 3 gráficos juntos, para poder compararlos de un vistazo.

**Qué mostrar en detalle:** `total_bill` (el monto de la cuenta) tiene sesgo a la derecha — la mayoría de las cuentas son bajas/medias, con una cola de cuentas altas. Es el mismo patrón que salarios o precios de casas, tal como dice la tabla de la Filmina 10.

**Preguntar a la clase:**

> ¿Por qué un histograma NO es lo mismo que un gráfico de barras?

**Respuesta:** El histograma es para variables numéricas continuas: las barras se tocan porque no hay huecos entre un número y el siguiente. El gráfico de barras es para variables categóricas (países, marcas): ahí sí tiene sentido dejar espacio entre barras, porque no hay continuidad numérica entre categorías.

👉 **Volvés a las filminas, Filmina 11 (división de módulo).**

---

### BLOQUE 3 — Seaborn: Distribución y Comparación entre Grupos *(Filminas 11–15)*

**Filmina 11 — División de módulo.**

#### Filmina 12 — ¿Por Qué Seaborn?

**Qué decir (ampliando lo que dice la filmina):**

Seaborn **no es "mejor"** que Matplotlib — está construido ENCIMA de Matplotlib, pensado específicamente para graficar DataFrames de Pandas de forma rápida. Internamente, cuando se llama a una función de Seaborn, esa función termina invocando comandos de Matplotlib para efectivamente dibujar — Seaborn es una capa de conveniencia, no un motor de dibujo alternativo. La diferencia práctica está en el nivel de abstracción: en Matplotlib puro hay que filtrar cada grupo a mano (con un `for` o con máscaras booleanas) y dibujarlo por separado, iteración por iteración; en Seaborn se le pasa el DataFrame completo y los nombres de las columnas, y la librería arma sola los colores, la leyenda y el estilo.

- **Seaborn** → ideal para EDA: histogramas, boxplots, correlaciones, scatterplots con categorías — resuelve en una línea lo que en Matplotlib puro son varias.
- **Matplotlib** → cuando hace falta personalizar mucho un gráfico puntual, o construir un tipo de visualización que Seaborn no tiene resuelto de fábrica (como el ejemplo de GridSpec del Anexo).

**Esto no está en la filmina, pero conviene agregarlo, porque genera errores confusos si no se entiende:** Seaborn tiene dos "modos" de funcionar, y mezclarlos es una fuente común de bugs.

- **Funciones Axes-level** (`scatterplot`, `boxplot`, `histplot`, entre otras): aceptan el parámetro `ax=` y dibujan exactamente en el Axes que se les indique — se integran perfecto dentro de un layout de subplots armado a mano con Matplotlib, tal como se vio en el Bloque 1.
- **Funciones Figure-level** (`displot`, `relplot`, `catplot`): arman su propia `Figure` desde cero y devuelven un objeto `FacetGrid`, no un `Axes` — no aceptan `ax=` porque no tiene sentido pedirles que dibujen "dentro de" algo que ya existe. Son ideales para exploración rápida, o para separar automáticamente en paneles por categoría usando el parámetro `col=`.

**Regla práctica:** si en algún momento aparece un error al pasarle `ax=` a una función de Seaborn, lo primero para revisar es si esa función es Figure-level — ese parámetro simplemente no existe para ellas.

**Qué buscamos ver con este ejemplo:** primero, el mismo scatter plot coloreado por género hecho con Matplotlib puro (filtrando a mano) y con Seaborn (una sola línea), para comparar la diferencia de esfuerzo. Después, un ejemplo de cada modalidad de Seaborn (Axes-level obedeciendo a un `ax=` propio, y Figure-level armando su panel automático por categoría).

👉 **Acá pasás a Colab.** Abrí el **Bloque 3** y corré las dos primeras celdas: "Matplotlib puro vs. Seaborn" y "Axes-level vs. Figure-level".

**Ejecutar:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")

# ── SIN Seaborn: hay que filtrar y dibujar grupo por grupo a mano ──
fig, ax = plt.subplots(figsize=(6, 4))
for genero, color in [("Male", "steelblue"), ("Female", "orange")]:
    subset = tips[tips["sex"] == genero]
    ax.scatter(subset["total_bill"], subset["tip"], color=color, label=genero)
ax.legend()
ax.set_title("Con Matplotlib puro: hay que filtrar y dibujar grupo por grupo")
plt.show()

# ── CON Seaborn: una sola línea hace lo mismo ──
plt.figure(figsize=(6, 4))
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="sex")
plt.title("Con Seaborn: mismo resultado en 1 línea")
plt.show()
```
```python
# Axes-level: obedece al ax= que le pasamos
fig, mi_eje = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=tips, x="total_bill", y="tip", ax=mi_eje, color="purple")
mi_eje.set_title("Yo obedezco al 'ax' que creó el programador")
plt.show()

# Figure-level: arma su propia Figure, no acepta ax=
g = sns.displot(data=tips, x="total_bill", col="time", kind="kde", fill=True)
plt.show()
```

**Qué hace cada línea:**
- `tips = sns.load_dataset("tips")`: carga el dataset de propinas — se reutiliza en casi todo este Bloque 3.
- `for genero, color in [("Male", "steelblue"), ("Female", "orange")]:`: recorre una lista de tuplas (género, color), dos iteraciones en total — una por cada valor posible de la columna `sex`.
- `subset = tips[tips["sex"] == genero]`: dentro del `for`, filtra el DataFrame completo quedándose solo con las filas de ese género — esto es exactamente el trabajo manual que Seaborn hace solo con `hue=`.
- `ax.scatter(subset["total_bill"], subset["tip"], color=color, label=genero)`: dibuja los puntos de ESE subconjunto, con SU color y SU etiqueta — se ejecuta una vez por cada vuelta del `for`, así que en total dibuja dos capas de puntos superpuestas en el mismo Axes.
- `ax.legend()`: arma la leyenda usando los `label=` acumulados en las dos llamadas a `scatter()`.
- `sns.scatterplot(data=tips, x="total_bill", y="tip", hue="sex")`: hace exactamente lo mismo que las 5 líneas anteriores, pero en una sola línea — `hue="sex"` le indica a Seaborn que separe, coloree y arme la leyenda automáticamente por esa columna.
- `fig, mi_eje = plt.subplots(figsize=(6, 4))` + `sns.scatterplot(..., ax=mi_eje, ...)`: acá se crea el Axes explícitamente con Matplotlib primero, y luego se le pide a Seaborn que dibuje adentro de ese `ax` puntual — comportamiento Axes-level.
- `g = sns.displot(data=tips, x="total_bill", col="time", kind="kde", fill=True)`: no hay ningún `plt.subplots()` previo — `displot` arma su propia Figure sola. `col="time"` le pide que separe automáticamente en un panel por cada valor de la columna `time` (Lunch/Dinner), sin que el programador arme esos subplots a mano. `kind="kde"` pide curvas de densidad en vez de barras, y `fill=True` rellena el área bajo la curva.

**Qué mostrar en detalle:** `hue="sex"` reemplaza el `for` completo — Seaborn separa, colorea y arma la leyenda automáticamente.

👉 **Volvés a las filminas, Filminas 13–14.**

#### Filminas 13–14 — Boxplot y Violinplot

**Qué decir (ampliando lo que dicen las filminas):**

Un **Boxplot** resume una distribución completa en 5 números clave, y vale la pena explicar cada uno con lo que representa:

- **Mínimo**: el valor más bajo del grupo (sin contar los outliers, que quedan marcados aparte como puntos sueltos).
- **Q1 (25%)**: el valor por debajo del cual queda el 25% de los datos — el borde inferior de la caja.
- **Mediana**: el valor central — la mitad de los datos está por debajo, la otra mitad por arriba. Se dibuja como una línea dentro de la caja.
- **Q3 (75%)**: el valor por debajo del cual queda el 75% de los datos — el borde superior de la caja.
- **Máximo**: el valor más alto del grupo (sin contar outliers).

La **caja** en sí (entre Q1 y Q3) es el rango intercuartílico (IQR) — el 50% central de los datos, la zona donde "vive" la mayoría del grupo. Los **bigotes** se extienden hasta el mínimo y el máximo "normales"; los puntos que quedan afuera de los bigotes son los **outliers**. Un matiz importante que no está en la filmina, pero conviene decirlo en voz alta: un outlier no siempre es un error de carga — puede ser el cliente más valioso de la empresa, o el caso más interesante del dataset. El boxplot solo lo señala; la decisión de qué hacer con él es de quien analiza.

Un **Violinplot** combina el boxplot con una curva KDE (la misma técnica de suavizado que vimos en el Bloque 2): el ancho del violín en cada altura representa la frecuencia de datos en ese valor. La utilidad concreta: dos grupos pueden tener exactamente la misma mediana y los mismos cuartiles — es decir, boxplots visualmente idénticos — pero formas de distribución completamente distintas (uno concentrado en el centro, otro con dos picos). El boxplot solo no puede distinguir esos dos casos; el violín sí, porque muestra la densidad completa.

- Usar **Boxplot** si la audiencia no es técnica, o si solo importa comunicar variabilidad y valores extremos de forma rápida.
- Usar **Violinplot** si hace falta la forma precisa de la distribución y hay suficientes registros — con pocos datos, el KDE del violín suaviza de más y da una falsa sensación de continuidad que los datos reales no tienen.

**Qué buscamos ver con este ejemplo:** la misma variable (`total_bill`) comparada entre dos categorías (`time`: Lunch vs. Dinner), primero resumida en 5 números (boxplot) y después con su forma de densidad completa (violinplot), una al lado de la otra para poder comparar directamente qué información aporta cada una.

👉 **Acá pasás a Colab.** Seguís en el **Bloque 3**, celda de "Boxplot y Violinplot".

**Ejecutar:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 4.5))

sns.boxplot(data=tips, x="time", y="total_bill", ax=axes[0])
axes[0].set_title("Boxplot: resumen por categoría")

sns.violinplot(data=tips, x="time", y="total_bill", ax=axes[1])
axes[1].set_title("Violinplot: boxplot + forma")

plt.tight_layout()
plt.show()
```

**Qué hace cada línea:**
- `fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 4.5))`: dos Axes lado a lado, uno para cada tipo de gráfico.
- `sns.boxplot(data=tips, x="time", y="total_bill", ax=axes[0])`: `x="time"` es la variable categórica (define cuántas cajas se dibujan, una por valor único de `time`), `y="total_bill"` es la variable numérica cuya distribución se resume dentro de cada caja.
- `sns.violinplot(data=tips, x="time", y="total_bill", ax=axes[1])`: mismos parámetros `x`/`y` que el boxplot — es intencional, para que la comparación entre ambos gráficos sea directa (mismos datos, mismo agrupamiento, distinta representación).
- `plt.tight_layout()` / `plt.show()`: ajusta espacios y renderiza el panel completo.

👉 **Volvés a las filminas, Filmina 15.**

#### Filmina 15 — Heatmap de Correlación y Estrategia de Elección

**Qué decir (ampliando lo que dice la filmina):**

Un **heatmap** convierte una tabla de números en una cuadrícula de colores — cada celda de la tabla se pinta según su valor, en vez de mostrarse como texto. El uso más común en ciencia de datos: visualizar la **matriz de correlación**, que resume con el coeficiente de Pearson (un número entre -1 y 1) qué tan relacionadas están todas las parejas de variables numéricas de un dataset, de una sola vez.

- **Cerca de 1** (color intenso en un extremo): las variables se mueven juntas — cuando una sube, la otra también (ej. metros² y precio de una casa).
- **Cerca de -1** (color intenso en el extremo opuesto): las variables se mueven en direcciones contrarias — cuando una sube, la otra baja (ej. peso del auto y eficiencia de combustible).
- **Cerca de 0** (color neutro, en el medio de la escala): no hay una relación lineal clara entre esas dos variables.

La ventaja de verlo como heatmap en vez de como tabla de números: con 10 o 20 variables, la tabla de correlaciones tiene cientos de celdas — leerla número por número es lento y propenso a error. El color permite detectar de un vistazo cuáles son las relaciones fuertes (celdas muy oscuras o muy claras) sin tener que leer cada valor.

La filmina cierra este bloque con la tabla de estrategia — la síntesis de todo lo visto en el Bloque 3:

| Pregunta | Gráfico |
|---|---|
| ¿Outliers o dispersión en un grupo? | Boxplot |
| ¿Forma y densidad de la distribución? | Violinplot |
| ¿Qué variables están relacionadas? | Heatmap |

**Qué buscamos ver con este ejemplo:** la matriz de correlación de las tres columnas numéricas del dataset de propinas, coloreada y con el número exacto escrito adentro de cada celda — para poder leer tanto el color (impresión rápida) como el valor preciso (verificación).

👉 **Acá pasás a Colab.** Última celda del **Bloque 3**: "Heatmap de correlación".

**Ejecutar:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")
matriz_corr = tips[["total_bill", "tip", "size"]].corr()

plt.figure(figsize=(5, 4))
sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Heatmap de correlación: total_bill, tip, size")
plt.show()
```

**Qué hace cada línea:**
- `tips[["total_bill", "tip", "size"]]`: selecciona solo las 3 columnas numéricas del DataFrame — nota el doble corchete, que devuelve un DataFrame (no una Serie) con esas 3 columnas.
- `.corr()`: el método de Pandas que calcula la matriz de correlación de Pearson entre todas las parejas de columnas del DataFrame — el resultado es una tabla cuadrada de 3x3, con un `1.0` en la diagonal (cada variable correlaciona perfecto consigo misma).
- `sns.heatmap(matriz_corr, ...)`: recibe directamente esa tabla de correlaciones y la pinta como cuadrícula de colores.
- `annot=True`: le pide a Seaborn que escriba el número de cada celda encima del color — sin esto, solo se vería el color, sin el valor exacto.
- `cmap="coolwarm"`: elige la paleta de colores — "coolwarm" es una paleta divergente (fríos para un extremo, cálidos para el otro), ideal para correlaciones porque tienen un punto medio significativo (el 0).
- `vmin=-1, vmax=1`: fija el rango de la escala de colores exactamente entre -1 y 1 — sin esto, Seaborn ajustaría la escala a los valores mínimo y máximo presentes en ESTA matriz puntual, lo que haría que el mismo color represente cosas distintas en gráficos diferentes.

**Qué mostrar en detalle — Errores comunes** (agregar de palabra, no está en la filmina): saturar el heatmap con 50 variables (filtrar primero las relevantes); violinplots sobre grupos de 5 datos (engañoso, mejor un boxplot); no ordenar las categorías del boxplot de mayor a menor mediana (reduce la carga cognitiva de quien lee).

👉 **Volvés a las filminas, Filmina 16 (Break).**

---

### ☕ Break *(Filmina 16)*

10 minutos.

---

### BLOQUE 4 — Storytelling y Diseño de Dashboards *(Filminas 17–22)*

**Filmina 17 — División de módulo.** Acá cambia el eje de la clase: hasta ahora era "cómo se hace un gráfico técnicamente correcto"; de acá en adelante es "cómo se diseña para que comunique la verdad, de forma clara, en el menor tiempo posible".

#### Filmina 18 — El Porqué de la Visualización

**Qué decir (ampliando lo que dice la filmina):**

**Analogía del mapa vs. la lista de direcciones** (la filmina la nombra, conviene desarrollarla en voz alta): una lista de direcciones ("gire a la izquierda, camine 200m, cruce el puente") es precisa, pero no da contexto — es exactamente lo que es una tabla de datos cruda. Un mapa, en cambio, muestra de un vistazo no solo dónde está el destino, sino qué hay alrededor, qué rutas alternativas existen y cuán lejos se está — es lo que hace una buena visualización con los datos: no agrega información nueva, pero la organiza espacialmente para que se entienda de un vistazo.

**Función cognitiva** (esto amplía el "bolt" de la filmina, conviene explicarlo con el mecanismo detrás): leer una tabla es un proceso secuencial — el cerebro tiene que leer fila por fila, comparar mentalmente, y recordar el valor anterior para notar una diferencia. Un gráfico, en cambio, aprovecha la percepción visual "subatentiva": el sistema visual humano detecta diferencias de posición, color o tamaño de forma automática y paralela, sin esfuerzo consciente. Ejemplo para dar en vivo: en 1.000 transacciones bancarias, encontrar una transacción sospechosa leyendo la tabla lleva minutos; en un scatter plot, un punto que está mil veces más arriba que el resto se ve en milisegundos. Esa es la esencia de por qué graficar reduce la carga cognitiva: libera espacio mental para el análisis, en vez de gastarlo en el simple hecho de leer.

**Nota para el docente:** esta filmina no tiene código propio. Antes de seguir a la Filmina 19, es un buen momento para bajar la teoría a tierra con dos ejemplos aplicados del notebook que, aunque no tienen una filmina 1 a 1, sirven de "banco de pruebas" para todo lo que se explica en las Filminas 18 a 22.

**Qué buscamos ver con el Ejemplo 1:** el mismo dato de ventas mensuales graficado dos veces — una vez sin ninguna decisión de diseño (`ax.bar()` a secas), y otra vez aplicando tres decisiones concretas (encuadre, jerarquía visual, anotación) para que se note en carne propia la diferencia entre "graficar el dato" y "comunicar el dato".

👉 **Acá pasás a Colab** (adelantándote un poco al hilo de las filminas, a propósito). Abrí el **Bloque 4** y corré "Ejemplo 1 — Encuadre, Jerarquía Visual y Anotaciones" y "Ejemplo 2 — Ética Visual".

**Ejecutar (Ejemplo 1):**
```python
import matplotlib.pyplot as plt

meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
ventas = [40, 38, 45, 42, 50, 48, 47, 53, 55, 60, 62, 80]

fig, (ax_sin_diseno, ax_con_diseno) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

ax_sin_diseno.bar(meses, ventas, color="steelblue")
ax_sin_diseno.set_title("ventas")

colores = ['#AAAAAA'] * 11 + ['#1a5276']
ax_con_diseno.bar(meses, ventas, color=colores)
ax_con_diseno.set_title("Ventas mensuales 2024 (miles de $)")
ax_con_diseno.set_ylabel("Ventas (miles $)")
ax_con_diseno.annotate('Récord del año', xy=(11, 80), xytext=(7, 68), arrowprops=dict(arrowstyle='->'))

plt.tight_layout()
plt.show()
```

**Qué hace cada línea:**
- `meses` / `ventas`: las mismas 12 barras se van a dibujar dos veces, con distinto tratamiento visual — así la comparación es sobre el diseño, no sobre el dato.
- `ax_sin_diseno.bar(meses, ventas, color="steelblue")`: un solo color para las 12 barras, sin ningún criterio — todas "pesan" lo mismo visualmente.
- `ax_sin_diseno.set_title("ventas")`: título en minúscula, sin unidad — no dice si son pesos, dólares, ni qué mes destacar. Esto es intencional: es el "gráfico sin pensarlo" del que habla la filmina.
- `colores = ['#AAAAAA'] * 11 + ['#1a5276']`: arma una lista de 12 colores — 11 veces gris (`'#AAAAAA'` repetido) y al final un azul oscuro (`'#1a5276'`) para el mes 12 (diciembre). Es la implementación concreta de la **jerarquía visual**: un solo color distinto basta para que el ojo vaya directo ahí.
- `ax_con_diseno.bar(meses, ventas, color=colores)`: a diferencia de la primera barra, acá `color=` recibe una LISTA de 12 colores en vez de uno solo — Matplotlib pinta cada barra con el color que le corresponde según su posición en la lista.
- `ax_con_diseno.set_title(...)` y `.set_ylabel(...)`: título específico con la unidad explícita ("miles de $") — esto es el **encuadre**.
- `ax_con_diseno.annotate(...)`: agrega la **anotación**. `xy=(11, 80)` es la punta de la flecha, en coordenadas reales del gráfico (mes índice 11 = diciembre, valor 80). `xytext=(7, 68)` es dónde va el texto, separado del dato para no taparlo. `arrowprops=dict(arrowstyle='->')` es lo que efectivamente dibuja la flecha — sin este parámetro solo aparecería el texto suelto, sin conexión visual al dato.

**Qué decir sobre este ejemplo:** mismos datos, dos gráficos. A la izquierda, `ax.bar()` a secas — no dice unidad ni qué mirar. A la derecha, tres decisiones de diseño encima: título con unidad (**encuadre**), un color distinto para el mes que importa (**jerarquía visual**), y una flecha que explica por qué (**anotación**).

**Qué buscamos ver con el Ejemplo 2:** el mismo dato de cuota de mercado representado de dos formas — una que manipula sutilmente la percepción (torta con un sector "separado" y un título sesgado) y otra que lo muestra sin ningún truco (barras ordenadas, eje desde 0) — para poder discutir en vivo por qué la segunda es más honesta aunque la primera "se vea" más atractiva.

**Ejecutar (Ejemplo 2):**
```python
import plotly.express as px
import pandas as pd

data = {'Compañía': ['Apple', 'Google', 'Microsoft', 'Otras Compañías'],
        'Cuota de Mercado (%)': [35, 15, 10, 40]}
df = pd.DataFrame(data)

fig = px.pie(df, values='Cuota de Mercado (%)', names='Compañía',
             title='Cuota de Mercado Global de Tecnología (35% Apple)',
             hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
fig.update_traces(textinfo='percent+label', pull=[0.15, 0, 0, 0])
fig.show()
```
```python
import matplotlib.pyplot as plt

empresas = ['Otras Compañías', 'Apple', 'Google', 'Microsoft']
cuotas = [40, 35, 15, 10]

fig, ax = plt.subplots(figsize=(7, 3))
ax.barh(empresas, cuotas, color='#5DADE2')
ax.set_xlabel('Cuota de mercado (%)')
ax.set_title('Cuota de mercado — sin destacar ninguna empresa')
ax.set_xlim(0, 50)

for i, v in enumerate(cuotas):
    ax.text(v + 0.5, i, f'{v}%', va='center')

plt.tight_layout()
plt.show()
```

**Qué hace cada línea:**
- `data = {...}` / `df = pd.DataFrame(data)`: arma un DataFrame de 4 filas con la compañía y su cuota — Plotly Express trabaja mejor recibiendo un DataFrame y los nombres de columna, en vez de listas sueltas.
- `fig = px.pie(df, values='Cuota de Mercado (%)', names='Compañía', title=..., hole=0.4, color_discrete_sequence=...)`: crea el gráfico de torta. `values=` indica qué columna define el tamaño de cada sector, `names=` qué columna los etiqueta. El `title` ya está redactado para sesgar la lectura hacia Apple. `hole=0.4` le da forma de "dona" en vez de torta completa (estética, no afecta el problema ético). `color_discrete_sequence` fija una paleta pastel.
- `fig.update_traces(textinfo='percent+label', pull=[0.15, 0, 0, 0])`: `textinfo='percent+label'` muestra el porcentaje y el nombre dentro de cada sector. **`pull=[0.15, 0, 0, 0]`** es la línea clave de la manipulación: separa el primer sector (Apple, primero en la lista `data`) un 15% del resto del gráfico — visualmente lo hace parecer más grande y más importante de lo que su porcentaje real indica.
- `fig.show()`: renderiza el gráfico interactivo.
- `empresas` / `cuotas`: en la segunda versión, los mismos 4 valores pero **ordenados de mayor a menor** cuota — "Otras Compañías" (40%, el valor más alto) queda primero.
- `ax.barh(empresas, cuotas, color='#5DADE2')`: `barh` (horizontal bar) en vez de `bar` — con nombres de compañía largos, las barras horizontales son más legibles que las verticales. Un solo color para las 4 barras: nadie se destaca artificialmente.
- `ax.set_xlim(0, 50)`: fuerza el eje X (acá el eje de los valores, porque son barras horizontales) a arrancar en 0 — la regla de honestidad para gráficos de barras que se vio en el Bloque 1.
- `for i, v in enumerate(cuotas): ax.text(v + 0.5, i, f'{v}%', va='center')`: recorre cada valor de `cuotas` con su índice `i`, y escribe el porcentaje exacto como texto al lado de cada barra (`v + 0.5` para separarlo un poco de la punta de la barra) — así no hace falta "leer" la escala del eje para saber el valor exacto de cada una.

**Qué decir sobre este ejemplo:** la primera versión (torta con Plotly) separa el sector de Apple y titula "35% Apple" — visualmente parece que domina. La segunda versión (barras horizontales) ordena por valor real: "Otras Compañías" (40%) queda primera. Los números son idénticos — lo que cambia es el diseño. **¿Cuál es mejor? El de barras.** No porque el pie chart esté mal hecho técnicamente, sino porque `pull` + el título elegido comunican una conclusión que los datos no sostienen. Un gráfico honesto no es el más lindo: es el que se interpreta bien sin ayuda de quien lo hizo.

👉 **Volvés a las filminas, Filmina 19.**

#### Filminas 19–21 — Pilares, Dashboards, Semáforo de Color y Casos Reales

**Qué decir (ampliando lo que dicen las filminas):**

**Cuatro pilares del análisis visual** (Filmina 19) — vale la pena desarrollar cada uno con un ejemplo concreto además de la tabla:

| Pilar | Pregunta | Gráfico típico | Ejemplo concreto |
|---|---|---|---|
| **Distribución** | ¿Cómo se reparten mis datos? | Histograma, KDE | Todo el Bloque 2 de hoy: bins, sesgo, multimodalidad |
| **Relación** | ¿Depende X de Y? | Scatter, Heatmap | El heatmap de correlación del Bloque 3 |
| **Evolución temporal** | ¿Crecemos, decrecemos, estancamos? | Línea | Series temporales — se profundiza en el Anexo del notebook |
| **Composición** | ¿Qué % viene de cada parte? | Barras apiladas, torta | El propio ejemplo de cuota de mercado que acabamos de ver |

**Data Storytelling** (agregar de palabra, no está desarrollado en la filmina): todo análisis visual bien contado sigue una estructura narrativa, igual que un relato. **Inicio**: da el contexto ("las ventas fluctuaron este año"). **Nudo**: señala dónde está el problema o la oportunidad ("pero en la región norte, la caída es drástica"). **Desenlace**: propone la acción a seguir ("hay que investigar al nuevo competidor en esa zona"). Un gráfico aislado, sin esta estructura alrededor, es solo un dato suelto — el storytelling es lo que lo convierte en una conclusión accionable.

**De gráficos sueltos a dashboards** (Filmina 20): la analogía del tablero de un auto es literal — el velocímetro, el nivel de combustible y la temperatura del motor son tres indicadores independientes que, combinados en un solo panel, le permiten al conductor tomar una decisión (frenar, parar a cargar nafta) sin tener que consultar tres pantallas separadas. Un dashboard de negocio cumple la misma función: reúne varias visualizaciones para dar una imagen completa de una situación de un vistazo. **Regla de oro:** un dashboard efectivo se limita a un máximo de 5 visualizaciones principales — más que eso empieza a generar "ruido visual" que confunde en vez de aclarar, exactamente lo opuesto de lo que se busca.

**Semáforo cognitivo del color** (Filmina 21) — la filmina da la clasificación, conviene remarcar el criterio detrás de cada tipo:
- **Secuenciales**: para variables numéricas que tienen un orden natural (azul claro→oscuro para ingresos bajos→altos) — el color crece o decrece junto con el valor.
- **Divergentes**: para variables con un punto medio significativo (rojo/blanco/azul para temperatura bajo cero/en cero/sobre cero) — el mismo esquema de color que se usó en el heatmap de correlación del Bloque 3, donde el punto medio era la correlación 0.
- **Categóricos**: para grupos sin orden inherente entre sí — acá lo importante no es la progresión del color, sino respetar las convenciones culturales (rojo = pérdida/alerta, verde = ganancia/ok). Usarlos al revés confunde al 100% de la audiencia, porque la asociación cultural es más fuerte que cualquier leyenda que se agregue.

**Casos reales que trae la filmina** — vale la pena leerlos completos porque son el "para qué sirve esto en un trabajo real": **Retail** (heatmaps de recorrido de clientes sobre el plano de una tienda — no solo muestran qué se vende, sino qué se mira pero no se compra, permitiendo mover productos de baja rotación a zonas de alto tráfico visual: +15% de ventas sin cambiar precios). **Salud pública** (gráficos de áreas apiladas para ver la capacidad hospitalaria disponible frente a la ocupada, permitiendo ver el "punto de quiebre" antes de que ocurra). **Amazon** (una caída súbita en el gráfico de "añadir al carrito" delata un bug técnico en el botón en cuestión de segundos, no que la gente dejó de querer los productos). **Finanzas personales** (donuts de gastos — un caso donde el pie chart, criticado en contextos técnicos, funciona bien para un usuario casual que solo necesita entender "la mitad de mi plata se va en alquiler").

Estas tres filminas son teóricas, sin código propio — el "banco de pruebas" ya se corrió antes (Filmina 18). Se sigue derecho a la Filmina 22.

**Preguntar a la clase:**

> Si tuvieras que explicarle a un gerente por qué las ventas bajaron el último mes en 10 segundos, ¿qué gráfico usarías y por qué?

#### Filmina 22 — Errores Comunes y Trampas Visuales

**Qué decir (ampliando lo que dice la filmina):**

La filmina trae la tabla de 3 trampas — vale la pena desarrollar cada una, no solo la del eje truncado que se demuestra en código:

- **Eje Y truncado**: empezar el eje vertical en un número distinto de cero para exagerar una diferencia. El mecanismo es puramente geométrico: si el eje arranca en 995 en vez de 0, una diferencia real de 12 unidades sobre una base de 1000 (1.2%) ocupa visualmente todo el alto del gráfico, como si fuera una diferencia del 50%. Mantener siempre el cero como base, salvo razón estadística muy fuerte — y si la hay, aclararlo explícitamente en el gráfico.
- **Pie chart con demasiadas categorías**: el problema no es estético, es perceptual — el ojo humano es objetivamente malo comparando ángulos y áreas (a diferencia de comparar longitudes, que es el atributo pre-atentivo más preciso, visto en la Filmina 20). Con más de 5-6 categorías, distinguir si una "tajada" de 7% es mayor que una de 6% se vuelve casi imposible. La alternativa casi siempre mejor: un gráfico de barras ordenado, donde comparar longitudes es trivial para el cerebro.
- **Exceso de "Data-Ink"** (concepto de Edward Tufte): llenar el gráfico de bordes gruesos, sombras 3D, fondos de colores y líneas de grilla pesadas. Todo lo que no sea directamente el dato es ruido visual que compite por la atención. Se retoma con un ejemplo de código en el Bloque 6.

La más fácil de demostrar en vivo con código es el **eje Y truncado**.

**Qué buscamos ver con este ejemplo:** los mismos 4 valores de ingresos trimestrales (con una variación real de apenas ~1.2%) graficados dos veces con distinto límite de eje Y — para comprobar en vivo cómo la sola elección de `set_ylim()` puede convertir un cambio insignificante en algo que parece un salto dramático.

👉 **Acá pasás a Colab.** Seguís en el **Bloque 4**, celda de "El Eje Y Truncado".

**Ejecutar:**
```python
import matplotlib.pyplot as plt

trimestres = ["Q1", "Q2", "Q3", "Q4"]
ingresos = [1000, 1005, 1008, 1012]  # variación real ~1.2%

fig, (ax_truncado, ax_honesto) = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))

ax_truncado.bar(trimestres, ingresos, color="#E74C3C")
ax_truncado.set_ylim(995, 1015)  # la trampa
ax_truncado.set_title("Eje truncado: parece un salto enorme")

ax_honesto.bar(trimestres, ingresos, color="#27AE60")
ax_honesto.set_ylim(0, 1200)
ax_honesto.set_title("Eje desde 0: la variación real (~1.2%)")

plt.tight_layout()
plt.show()
```

**Qué hace cada línea:**
- `ingresos = [1000, 1005, 1008, 1012]`: los mismos 4 valores se usan en ambos gráficos — la única diferencia entre los dos paneles va a ser el límite del eje Y, nada del dato cambia.
- `ax_truncado.bar(trimestres, ingresos, color="#E74C3C")`: barras rojas — un color de alerta, casi como anticipando visualmente el efecto engañoso.
- `ax_truncado.set_ylim(995, 1015)`: acá está la trampa exacta — el eje Y va de 995 a 1015, un rango de apenas 20 unidades. Como los valores (1000 a 1012) ocupan casi todo ese rango angosto, las diferencias entre barras se ven enormes.
- `ax_honesto.set_ylim(0, 1200)`: el mismo tipo de gráfico, pero con el eje Y arrancando en 0 y llegando bien por encima del valor máximo — en esta escala, la diferencia real entre trimestres (apenas 12 unidades sobre 1000) se ve tan chica como realmente es.

**Qué mostrar en detalle:** mismos 4 números en los dos gráficos — el de la izquierda "grita" una tendencia dramática que, en la escala real (derecha), es casi imperceptible. Las otras dos trampas de la tabla (pie chart con muchas categorías, exceso de Data-Ink) se retoman en el Bloque 6 con el ejemplo de chartjunk.

👉 **Volvés a las filminas, Filmina 23 (división de módulo).**

---

### BLOQUE 5 — Arquitectura Visual y la Regla de Lectura en "Z" *(Filminas 23–24)*

**Filmina 23 — División de módulo.**

#### Filmina 24 — El Patrón de Lectura en "Z"

**Qué decir (ampliando lo que dice la filmina):**

Un dashboard (en Excel o cualquier herramienta de BI) es una **interfaz de usuario**, no una colección de gráficos insertados uno al lado del otro sin criterio. La diferencia es importante: una interfaz de usuario está diseñada pensando en cómo la persona que la usa se mueve por ella, no solo en qué información contiene. El objetivo es reducir la carga cognitiva: el usuario debe poder extraer los insights más críticos en los primeros 3 segundos, sin tener que "buscar" dónde está lo importante — esto conecta directo con la "regla de oro" de 5 visualizaciones que vimos en la Filmina 20: un dashboard con demasiados elementos vuelve imposible cumplir ese objetivo de 3 segundos.

En culturas occidentales leemos de izquierda a derecha y de arriba hacia abajo — por eso, en pantallas o documentos sin mucho texto continuo (como un dashboard, a diferencia de un libro), el ojo tiende a escanear la información siguiendo un recorrido en forma de la letra **"Z"**: primero una pasada horizontal arriba, después una diagonal hacia abajo, después otra pasada horizontal abajo. Diseñar un dashboard significa, literalmente, poner cada elemento en el punto de ese recorrido que le corresponde según su importancia:

| Posición | Qué colocar | Por qué ahí |
|---|---|---|
| **1. Arriba-Izquierda** (Ancla visual) | KPIs principales: Ventas Totales, Margen, ROI | Es el primer punto que toca el ojo en el recorrido — tiene que ser lo más importante de todo el reporte, sin excepción. |
| **2. Arriba-Derecha** (Contexto y control) | Slicers, línea de tiempo (Timeline) | Es el segundo punto del recorrido — acá van los elementos que permiten filtrar o ajustar la vista antes de bajar al detalle. |
| **3. Abajo-Izquierda** (Profundidad analítica) | Tendencias en el tiempo, comparativas clave | El ojo vuelve a la izquierda después de la diagonal — es el lugar natural para el "cómo llegamos hasta acá" (gráficos de líneas o barras). |
| **4. Abajo-Derecha** (Punto de salida) | Tablas de detalle, top 10, formato condicional | El último punto del recorrido — el lugar del detalle fino, para quien quiere auditar los números exactos antes de cerrar el reporte. |

Recorrido recomendado: **KPIs → Controles/Slicers → Gráficos de Tendencia → Tablas de Detalle**. Ocultar las líneas de cuadrícula por defecto de Excel y usar el tamaño de las celdas como un sistema de grillas propio ayuda a que el ojo siga ese recorrido sin distracciones — de lo contrario, la cuadrícula nativa de la planilla compite visualmente con el diseño intencional del dashboard.

**Este bloque es 100% conceptual — no hay celda de código en el notebook para correr acá.** Se aplica al diseñar dashboards en Excel/BI, no en Python. Se sigue derecho a la Filmina 25.

**Preguntar a la clase:**

> Si tuvieran que armar un dashboard de ventas con 4 elementos (un KPI de facturación total, un filtro de fecha, un gráfico de tendencia mensual y una tabla de top clientes), ¿dónde pondrían cada uno siguiendo la "Z"?

---

### BLOQUE 6 — Principios de UI/UX y Jerarquía Visual en Dashboards *(Filminas 25–29)*

**Filmina 25 — División de módulo.**

#### Filminas 26–27 — Jerarquía Visual y sus Pilares

**Qué decir (ampliando lo que dicen las filminas):**

**Analogía de la cabina de avión** (Filmina 26): cientos de indicadores compitiendo por la atención al mismo tiempo. Un piloto entrenado sabe exactamente dónde mirar en cada momento de vuelo, porque su entrenamiento le dio una jerarquía mental de qué instrumento revisar primero; un ejecutivo con 5 minutos antes de una junta, enfrentado a un dashboard sin esa jerarquía, no tiene ese entrenamiento — y el diseño tiene que suplirlo. La jerarquía visual es exactamente eso: el sistema que le dice al usuario, sin que tenga que aprenderlo de memoria, "mirá esto primero, esto después, esto solo si necesitás el detalle". Sin jerarquía, todos los elementos "pesan" lo mismo visualmente — el resultado es ruido: nada se destaca, todo compite. Esto conecta con un principio de psicología cognitiva que menciona la filmina: el cerebro es perezoso por naturaleza (en el buen sentido — es eficiente) y busca atajos para no gastar energía de más; un diseño con jerarquía clara aprovecha esa tendencia en vez de pelear contra ella.

Los 4 pilares que trae la Filmina 27 — desarrollando cada uno con su mecanismo:

- **Patrón F/Z**: ya lo vimos en detalle en el Bloque 5 con la letra "Z" — acá se aplica la misma lógica de recorrido visual: zona superior izquierda para los KPIs críticos, zona media para tendencias, zona inferior o derecha para el detalle.
- **Tamaño y peso**: lo más grande se percibe automáticamente como lo más importante — es un atributo pre-atentivo (se procesa sin esfuerzo consciente, como vimos en el Bloque 4). El matiz que agrega la filmina: en Excel es común abusar de esto, haciendo un KPI gigante que ocupa media pantalla. La recomendación es que sea "lo suficientemente grande para destacar, pero lo suficientemente chico para permitir contexto" — el tamaño debe usarse con intención, no al máximo posible. También recomienda fuentes Sans Serif (como Segoe UI o Aptos) para los números, porque son más limpias y legibles en pantallas digitales que las fuentes con serifas.
- **Color con propósito**: la idea central es que en un reporte analítico el color debe ser funcional, no decorativo. Colores semánticos (rojo=alerta, verde=cumplimiento, ámbar=atención) comunican estado sin necesidad de leer texto. Colores neutros (grises) para ejes, cuadrículas y etiquetas de datos — el gris reduce el ruido visual de fondo y hace que los colores funcionales, cuando aparecen, resalten más por contraste.
- **Espacio en blanco**: no es espacio desperdiciado — es el "aire" que permite a los elementos respirar. Agrupar demasiados gráficos sin separación entre ellos crea una mancha visual indistinguible, donde el ojo no sabe dónde termina un elemento y empieza el siguiente. El espacio en blanco cumple una función activa: agrupa conceptos relacionados (poniéndolos cerca) y separa secciones distintas (poniendo distancia entre ellas).

Estas dos filminas son teóricas — el código llega en la próxima, con el ejemplo de chartjunk que baja el pilar de "espacio en blanco / color con propósito" a código real.

#### Chartjunk vs. Gráfico Limpio *(código que ilustra las Filminas 27 y 29)*

**Qué decir (esto no está desarrollado como ejemplo en las filminas, pero conecta directo con la tabla de Filmina 29):**

El mismo principio de "menos es más" que se explicó en la Filmina 27 se puede aplicar y ver en código real con Matplotlib: cada borde, sombra o línea de más es *ruido* que compite con el dato. Edward Tufte (el autor que acuñó el término "chartjunk", literalmente "basura de gráfico") propuso el concepto de **Data-to-Ink Ratio** (proporción de tinta dedicada a los datos vs. tinta dedicada a la decoración): la regla práctica es que si se puede quitar un elemento visual y el dato se sigue entendiendo igual de bien, hay que quitarlo, porque ese elemento no estaba aportando información, solo ocupando espacio y atención. Esta es también la tercera trampa que había quedado pendiente de la Filmina 22 (Exceso de Data-Ink) — se retoma acá porque recién ahora tenemos las herramientas de código (`spines`, `grid`) para demostrarla en vivo.

**Qué buscamos ver con este ejemplo:** los mismos 5 valores por región, graficados dos veces — una versión "cargada" (bordes gruesos, colores random sin significado, grilla pesada) y otra minimalista (un solo color, sin bordes innecesarios, grilla apenas visible) — para comprobar que ambas comunican exactamente el mismo dato, pero una lo hace con mucho menos "ruido" visual.

👉 **Acá pasás a Colab.** Abrí el **Bloque 6** y corré la celda "Chartjunk vs. gráfico limpio".

**Ejecutar:**
```python
import matplotlib.pyplot as plt

categorias = ["Norte", "Sur", "Este", "Oeste", "Centro"]
valores = [82, 65, 90, 58, 74]
colores_random = ["#E74C3C", "#3498DB", "#F1C40F", "#9B59B6", "#1ABC9C"]

fig, (ax_chartjunk, ax_limpio) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5))

ax_chartjunk.bar(categorias, valores, color=colores_random, edgecolor="black", linewidth=2.5)
ax_chartjunk.set_title("Chartjunk: mucho ruido, poco dato")
ax_chartjunk.grid(True, linewidth=1.5, color="black")
for spine in ax_chartjunk.spines.values():
    spine.set_linewidth(2.5)

ax_limpio.bar(categorias, valores, color="#4C72B0")
ax_limpio.set_title("Data-Ink Ratio alto: solo lo necesario")
ax_limpio.spines["top"].set_visible(False)
ax_limpio.spines["right"].set_visible(False)
ax_limpio.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
```

**Qué hace cada línea:**
- `colores_random = [...]`: 5 colores muy distintos entre sí, uno por barra, sin ningún criterio semántico (no representan categorías con significado propio) — es intencionalmente "ruido de color".
- `ax_chartjunk.bar(categorias, valores, color=colores_random, edgecolor="black", linewidth=2.5)`: además del color random por barra, `edgecolor="black"` le agrega un borde negro a cada barra, y `linewidth=2.5` lo hace grueso — un elemento visual más que no aporta información sobre el dato en sí.
- `ax_chartjunk.grid(True, linewidth=1.5, color="black")`: activa una grilla de fondo gruesa y negra — compite directamente con las barras por la atención, en vez de ayudar sutilmente a leer valores.
- `for spine in ax_chartjunk.spines.values(): spine.set_linewidth(2.5)`: `spines` son los cuatro bordes del área del gráfico (arriba, abajo, izquierda, derecha) — este bucle los recorre todos y les pone un grosor de línea de 2.5, engrosando el marco completo del Axes.
- `ax_limpio.bar(categorias, valores, color="#4C72B0")`: un solo color para las 5 barras, sin bordes adicionales — el color no está comunicando ninguna categoría distinta, así que no hace falta variarlo.
- `ax_limpio.spines["top"].set_visible(False)` y `["right"].set_visible(False)`: apaga específicamente los bordes de arriba y de la derecha del Axes — son los dos bordes que casi nunca aportan información (el de abajo y el de la izquierda sí sirven, porque son los propios ejes X e Y).
- `ax_limpio.grid(axis="y", alpha=0.3)`: grilla solo en el eje Y (no en X, porque acá son categorías, no tiene sentido una grilla vertical) y con `alpha=0.3` (30% de opacidad) — suficiente para ayudar a leer valores aproximados, sin dominar visualmente al dato.

👉 **Volvés a las filminas, Filmina 28.**

#### Filminas 28–29 — Arquitectura de la Información, Anti-Patrones y Accesibilidad

**Qué decir (ampliando lo que dicen las filminas):**

Un dashboard ejecutivo es como una conversación (Filmina 28) — vale la pena desarrollar cada uno de los 4 niveles con el porqué de su orden: **El Titular (KPIs)** responde "¿cómo vamos?" — es lo primero porque es la conclusión que cualquiera necesita antes que ningún detalle. **El Contexto (Tendencias)** responde "¿cómo llegamos hasta acá?" — le da al titular una dimensión temporal, mostrando si la situación actual es una mejora, un empeoramiento o algo estable. **El Diagnóstico (Desgloses)** responde "¿quién o qué causó esto?" — desglosa el número general en sus partes (por región, por producto, por centro de costo) para encontrar dónde está el problema u oportunidad puntual. **El Detalle (Tablas)** responde "mostrame los datos exactos" — es el último nivel, para quien necesita auditar o verificar un número específico.

*Ejemplo real que trae la filmina — dashboard de un Controller revisando el cierre mensual:* Nivel 1: cuatro tarjetas grandes arriba con EBITDA, Ingresos, Costos y % de Desviación (el titular). Nivel 2: un gráfico de cascada (Waterfall) que explica el paso del presupuesto al real (el contexto). Nivel 3: un gráfico de barras con el top 5 de centros de costo que se excedieron (el diagnóstico). Nivel 4: una tabla dinámica filtrable con las facturas específicas (el detalle). Los 4 niveles de la conversación, aplicados en orden.

**Anti-patrones** (Filmina 29) — cada uno con su mecanismo y su corrección concreta:

| Anti-patrón | Qué pasa | Solución |
|---|---|---|
| **Efecto Árbol de Navidad** | Demasiados colores vibrantes compitiendo entre sí — el usuario no sabe qué es una alerta real y qué es solo decoración. | Usar una paleta monocromática o de colores similares para los datos estándar, y reservar los colores brillantes (rojo, naranja) exclusivamente para las excepciones o metas incumplidas. |
| **Chartjunk** (Tufte) | Bordes gruesos, sombras 3D, líneas de división muy oscuras — visto en el ejemplo de código de la Filmina 27. | Maximizar el Data-to-Ink Ratio: si se puede quitar algo y el dato se sigue entendiendo, quitarlo. |
| **Dato Huérfano** | "$1.000.000" sin ningún contexto no significa nada por sí solo — ¿es bueno o malo? ¿es más que el mes pasado? | Todo KPI debe venir acompañado de una comparación (vs. Target, vs. Año Anterior) o una micro-tendencia (un Sparkline al lado del número). |

**Accesibilidad** (también Filmina 29): aproximadamente el 8% de los hombres y el 0.5% de las mujeres tiene alguna deficiencia en la percepción del color — el tipo más común confunde rojo y verde, que son justamente los colores semánticos más usados en dashboards (alerta/cumplimiento). Depender solo de ese semáforo rojo/verde deja afuera a una parte real de la audiencia. La corrección tiene dos caminos que se pueden combinar: agregar un ícono adicional (una flecha hacia abajo junto al rojo, por ejemplo) para que el significado no dependa solo del color, o usar una paleta ya diseñada para ser distinguible por personas con daltonismo, como `palette="colorblind"` en Seaborn — que es exactamente lo que hace el código de esta filmina.

**Qué buscamos ver con este ejemplo:** una misma serie de datos (pasajeros por mes, para 3 meses distintos) graficada con tres capas de accesibilidad superpuestas — color distinto, trazo de línea distinto, y marcador geométrico distinto por categoría — de forma que la información siga siendo legible aunque una persona no pueda distinguir los colores, o aunque el gráfico se imprima en blanco y negro.

👉 **Acá pasás a Colab.** Última celda del **Bloque 6**: "Accesibilidad Cromática".

**Ejecutar:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

flights = sns.load_dataset("flights")
df_meses = flights[flights['month'].isin(['Jan', 'Jul', 'Dec'])]
df_meses['month'] = df_meses['month'].cat.remove_unused_categories()

plt.figure(figsize=(8, 4))
sns.lineplot(data=df_meses, x="year", y="passengers", hue="month", style="month",
             markers=True, palette="colorblind")
plt.title("Accesible: Diferentes colores, trazos y figuras")
plt.show()
```

**Qué hace cada línea:**
- `flights = sns.load_dataset("flights")`: carga el dataset de pasajeros aéreos por mes que trae Seaborn — tiene una fila por combinación de año y mes, con la cantidad de pasajeros.
- `df_meses = flights[flights['month'].isin(['Jan', 'Jul', 'Dec'])]`: filtra el DataFrame, quedándose solo con enero, julio y diciembre — con los 12 meses juntos, el gráfico se saturaría de líneas; con 3, se puede ver el efecto de accesibilidad con claridad.
- `df_meses['month'] = df_meses['month'].cat.remove_unused_categories()`: la columna `month` es de tipo categórico en el dataset original, con los 12 meses como categorías posibles. Después del filtro, quedan solo 3 meses con datos, pero la columna "recuerda" las 12 categorías originales — este método limpia esa memoria, dejando solo las 3 categorías realmente presentes (evita que Seaborn intente reservar colores/estilos para meses que ya no están en los datos).
- `sns.lineplot(..., hue="month", style="month", markers=True, palette="colorblind")`: acá están las tres capas de accesibilidad juntas — `hue="month"` da un color distinto por mes, `style="month"` da un tipo de trazo distinto por mes (sólido, punteado, etc.), y `markers=True` agrega un marcador geométrico distinto en cada punto de dato. `palette="colorblind"` reemplaza la paleta de colores default por una diseñada específicamente para ser distinguible con daltonismo.

👉 **Volvés a las filminas, Filmina 30 (división de módulo).**

---

### BLOQUE 7 — Práctica: Auditoría UX de Dashboards *(Filminas 30–32, sin código)*

**Filmina 30 — División de módulo.**

#### Filmina 31 — Ejercicio 1: Auditoría y Rediseño

**Qué decir (ampliando lo que dice la filmina):** para esta práctica no se usa código, sino capacidad de análisis crítico — el objetivo del Bloque 7 es que la clase aplique, sin ayuda de la computadora, todo lo que se vino explicando desde el Bloque 4 en adelante. El escenario que describe la filmina: un gráfico de pastel con 15 categorías en colores muy similares (la trampa de "pie chart con demasiadas categorías" de la Filmina 22), títulos en Comic Sans tamaño 24 en azul brillante (una violación directa del pilar "tamaño y peso" y "color con propósito" de la Filmina 27), y un KPI de "Ventas" en la esquina inferior derecha sin ningún comparativo (el anti-patrón "Dato Huérfano" de la Filmina 29) — conviene leer el escenario completo en voz alta antes de dar tiempo a resolver, porque cada detalle que menciona corresponde a un concepto puntual ya visto.

**Consigna, desarrollada:**
1. **Identificación**: enumerar al menos 4 fallos de jerarquía visual o UX presentes en el escenario — la idea es que cada fallo identificado se pueda nombrar con el vocabulario técnico visto en la clase (chartjunk, dato huérfano, eje truncado, etc.), no solo describirlo en palabras propias.
2. **Propuesta de layout**: describir cómo se reorganizarían los elementos usando el patrón de lectura en "F" (variante del patrón "Z" para pantallas con más densidad de información) — qué va arriba a la izquierda, qué se mueve al final.
3. **Guía de estilo**: definir una paleta de colores profesional, con un máximo de 3 colores principales más grises, explicando la función analítica de cada uno (cuál es semántico, cuál es neutro).
4. **Contextualización**: proponer qué métrica de comparación agregarle al KPI de ventas para que deje de ser un dato huérfano (¿vs. el mes anterior? ¿vs. un objetivo?).

Sin código — no hay transición a Colab en este bloque, este ejercicio se resuelve en papel o en un documento de texto.

#### Filmina 32 — Ejercicio 2: Caso "Dashboard Comercial - Q3"

**Qué decir (ampliando lo que dice la filmina):** este segundo ejercicio cambia el formato — en vez de un escenario abstracto, es un caso de estudio con un rol concreto: sos el nuevo analista de datos de una empresa de retail. El gerente de ventas te envía el dashboard actual (construido en Excel) que usa el equipo directivo, con el comentario textual: *"la información está ahí, pero en las reuniones tardamos 10 minutos en entender cómo nos fue en el mes"* — esta frase es la pista central del ejercicio: no es un problema de datos faltantes, es un problema de diseño que hace lento el acceso a la información que sí está.

**Consigna, desarrollada:**
- **Diagnóstico**: identificar al menos 3 violaciones graves a los principios de UI/UX y al patrón de lectura en "Z" — conectar explícitamente con lo visto en la Filmina 24 (el patrón Z en sí) y las Filminas 26-29 (jerarquía visual, pilares del diseño, anti-patrones).
- **Reestructuración**: proponer cómo redistribuir 4 elementos concretos — tabla dinámica, gráfico circular, slicers, tarjetas de KPI — usando correctamente el modelo en "Z" (KPIs arriba-izquierda, controles arriba-derecha, tendencias abajo-izquierda, detalle abajo-derecha).

**Preguntar a la clase:** dejar que un par de estudiantes propongan su diagnóstico en voz alta antes de mostrar la respuesta esperada — este bloque funciona mejor como discusión abierta que como exposición. Una forma de dinamizarlo: pedir que cada estudiante mencione UN solo fallo distinto al que ya mencionó un compañero, para ir construyendo la lista entre todos en vez de que una sola persona la complete sola.

👉 **Volvés a las filminas, Filmina 33 (división de módulo).**

---

### BLOQUE 8 — Pre-entrega: Limpieza y Documentación del Dataset *(Filminas 33–36)*

**Filmina 33 — División de módulo.** Este es el cierre evaluable de la clase. **Importante:** esta consigna es la oficial de la plataforma — reemplaza a una versión anterior de este mismo bloque que hablaba de "EDA Visual" (gráficos). No confundir tampoco con la **Pre-entrega 4** de Clase 04 ("Estructura Inicial del Dataset"): son dos checkpoints relacionados pero distintos — Clase 04 ya sanea la estructura básica (nulos, tipos, filtros); esta pre-entrega de Clase 05 profundiza la limpieza (imputación con criterio, deduplicación, conversión de fechas), suma agregaciones de negocio con `groupby`, y agrega el requisito de documentar todo en un repositorio de GitHub con `README.md`.

#### Filmina 34 — Inicialización, Carga y Diagnóstico

**Qué decir (ampliando lo que dice la filmina):**

**Paso 1 — Inicialización:** antes de escribir una sola línea de análisis, el proyecto necesita un repositorio en GitHub con un notebook (`.ipynb`) o script (`.py`) y el dataset original — o, si el dataset pesa mucho, un link a la fuente en vez del archivo pesado directamente en el repo. Esto no es un detalle administrativo menor: un repositorio bien inicializado desde el principio es lo que permite que el trabajo sea reproducible y compartible, en vez de vivir solo en la computadora de quien lo hizo.

**Paso 2 — Carga y Diagnóstico:** `read_csv()` para traer los datos, y tres comandos de diagnóstico que ya se vieron en la Pre-entrega 4 de Clase 04 pero que acá se aplican con más profundidad: `.info()` (nulos y `dtypes` por columna), `.describe()` (rangos y estadísticas de las variables numéricas), `.isnull().sum()` (el conteo exacto de nulos por columna, para saber con precisión qué hay que resolver antes de seguir).

**Qué buscamos ver con este ejemplo:** el diagnóstico completo de un dataset real de altas de clientes de una plataforma de e-commerce (`ecommerce_clientes.csv`, 2.589 filas) — con nulos y duplicados genuinos, no simulados, para que la limpieza que viene en la próxima filmina tenga sentido real.

👉 **Acá pasás a Colab.** Abrí el **Bloque 8** y corré la celda de "Carga y Diagnóstico".

**Ejecutar:**
```python
import pandas as pd

df = pd.read_csv("ecommerce_clientes.csv")

print(f"Dimensiones: {df.shape}")
df.info()

print("\nValores nulos por columna:")
print(df.isnull().sum())

print("\nEstadísticas descriptivas (columnas numéricas):")
print(df.describe())
```

**Qué hace cada línea:**
- `df = pd.read_csv("ecommerce_clientes.csv")`: carga el dataset — 9 columnas: `COMPANY_ID`, `COMPANY_CREATED_AT`, `SUBSCRIPTION_STATUS`, `PRODUCT`, `SELLER_TYPE`, `COUNTRY`, `EMAIL`, `ACTIVE_USERS_L_28D`, `ALL_USERS_COUNT`.
- `df.shape`: dimensiones del dataset — cuántas filas (clientes) y columnas (variables) hay antes de tocar nada.
- `df.info()`: recorre columna por columna mostrando cuántos valores no nulos tiene cada una y de qué `dtype` es — acá aparece el primer problema real: `COMPANY_CREATED_AT` es `object` (texto), cuando debería ser una fecha.
- `df.isnull().sum()`: el conteo exacto de nulos por columna — muestra que `SELLER_TYPE` y `ALL_USERS_COUNT` tienen huecos importantes, y que `ACTIVE_USERS_L_28D` también tiene algunos.
- `df.describe()`: estadísticas de las columnas numéricas (`COMPANY_ID`, `ACTIVE_USERS_L_28D`, `ALL_USERS_COUNT`) — da una primera idea de los rangos antes de decidir cómo imputar.

**Qué mostrar en detalle:** tres problemas reales conviven en este dataset — nulos en `SELLER_TYPE` (categórica), nulos en `ACTIVE_USERS_L_28D` y `ALL_USERS_COUNT` (numéricas, pero con significados distintos), y una fecha guardada como texto. Cada uno se resuelve con una estrategia diferente en la próxima filmina — no hay una receta única para "limpiar nulos".

👉 **Volvés a las filminas, Filmina 35.**

#### Filmina 35 — Limpieza Rigurosa y Agregaciones de Negocio

**Qué decir (ampliando lo que dice la filmina):**

**Paso 3 — Limpieza Rigurosa**, con sus tres técnicas desarrolladas por separado:
- **Imputación de valores faltantes**: no es una sola técnica, son tres estrategias distintas según qué signifique el hueco — **media** (para numéricas donde el valor faltante probablemente esté cerca del promedio), **moda** (para categóricas, el valor más frecuente), o **eliminación estratégica** (cuando no hay ninguna imputación razonable y es mejor descartar la fila).
- **Eliminación de duplicados**: no alcanza con `duplicated()` sin argumentos (que compara la fila completa) — en datos reales, dos registros del mismo cliente casi nunca son idénticos en todas las columnas, pero sí comparten un identificador único como el email.
- **Conversión de tipos**: las fechas que llegan como texto no se pueden ordenar cronológicamente ni operar con ellas hasta convertirlas con `pd.to_datetime()` — el mismo problema que se vio con `indice_tiempo` en el dataset de vuelos, ahora aplicado a `COMPANY_CREATED_AT`.

**Paso 4 — Agregaciones de Negocio:** al menos 3 `groupby` que respondan preguntas concretas — no agregaciones al azar, sino preguntas que un negocio real haría: ¿en qué país están concentrados los clientes?, ¿qué producto genera más uso activo?, ¿cómo se reparte el tipo de vendedor según el estado de la suscripción?

**Qué buscamos ver con este ejemplo:** resolver los tres problemas diagnosticados en la filmina anterior, cada uno con la técnica que le corresponde — y después responder tres preguntas de negocio distintas sobre el dataset ya limpio.

👉 **Acá pasás a Colab.** Seguís en el **Bloque 8**, celdas de "Limpieza Rigurosa" y "Agregaciones de Negocio".

**Ejecutar (Limpieza Rigurosa):**
```python
import pandas as pd

df["SELLER_TYPE"] = df["SELLER_TYPE"].fillna(df["SELLER_TYPE"].mode()[0])
df["ACTIVE_USERS_L_28D"] = df["ACTIVE_USERS_L_28D"].fillna(0)
df = df.dropna(subset=["ALL_USERS_COUNT"])
df = df.reset_index(drop=True)

duplicados = df["EMAIL"].duplicated().sum()
print(f"Emails duplicados antes de limpiar: {duplicados}")
df = df.drop_duplicates(subset=["EMAIL"], keep="first").reset_index(drop=True)
print(f"Filas después de eliminar duplicados: {len(df)}")

df["COMPANY_CREATED_AT"] = pd.to_datetime(df["COMPANY_CREATED_AT"])
print(f"\ndtype de COMPANY_CREATED_AT después de convertir: {df['COMPANY_CREATED_AT'].dtype}")
```

**Qué hace cada línea:**
- `df["SELLER_TYPE"].fillna(df["SELLER_TYPE"].mode()[0])`: `.mode()` devuelve una Serie (puede haber más de un valor empatado como más frecuente), por eso `[0]` toma el primero — imputar una variable categórica con un promedio no tendría sentido, por eso acá se usa la moda.
- `df["ACTIVE_USERS_L_28D"].fillna(0)`: acá el nulo probablemente significa "sin actividad registrada en ese período", no "dato desconocido" — es una decisión de negocio, no una regla mecánica aplicable a cualquier columna numérica.
- `df.dropna(subset=["ALL_USERS_COUNT"])`: a diferencia de las dos anteriores, acá se decide eliminar directamente las filas — no hay una imputación razonable si no se sabe si esa empresa activó usuarios o no.
- `df.reset_index(drop=True)`: después de cada operación que quita filas, el índice queda con "huecos" (0, 1, 3, 7...) — sin este paso, ese índice discontinuo puede romper operaciones posteriores que asuman una secuencia consecutiva. Es exactamente el error que la filmina marca como "común a evitar".
- `df["EMAIL"].duplicated().sum()`: cuenta cuántos emails están repetidos — a diferencia de `df.duplicated()` sin argumentos, que solo detectaría filas 100% idénticas en todas las columnas.
- `df.drop_duplicates(subset=["EMAIL"], keep="first")`: elimina las filas con email repetido, quedándose con la primera aparición de cada una.
- `pd.to_datetime(df["COMPANY_CREATED_AT"])`: convierte el texto a un tipo de dato fecha real — sin esto, cualquier análisis de antigüedad o tendencia temporal daría resultados incorrectos, porque Pandas ordenaría las fechas alfabéticamente en vez de cronológicamente.

**Ejecutar (Agregaciones de Negocio):**
```python
print("1) Cantidad de clientes por país:")
print(df.groupby("COUNTRY")["COMPANY_ID"].count())

print("\n2) Promedio de usuarios activos (últimos 28 días) por producto:")
print(df.groupby("PRODUCT")["ACTIVE_USERS_L_28D"].mean().round(1).sort_values(ascending=False))

print("\n3) Cantidad de clientes por estado de suscripción y tipo de vendedor:")
print(df.groupby(["SUBSCRIPTION_STATUS", "SELLER_TYPE"])["COMPANY_ID"].count())
```

**Qué hace cada línea:**
- `df.groupby("COUNTRY")["COMPANY_ID"].count()`: agrupa por país y cuenta cuántos `COMPANY_ID` hay en cada grupo — responde "¿dónde está concentrada la base de clientes?".
- `df.groupby("PRODUCT")["ACTIVE_USERS_L_28D"].mean().round(1).sort_values(ascending=False)`: agrupa por producto, promedia el uso activo reciente, redondea a 1 decimal y ordena de mayor a menor — responde "¿qué producto genera más uso real, no solo más altas?".
- `df.groupby(["SUBSCRIPTION_STATUS", "SELLER_TYPE"])["COMPANY_ID"].count()`: agrupa por DOS columnas a la vez (una lista en vez de un solo nombre) — responde una pregunta que un `groupby` de una sola columna no puede responder: cómo se reparte el tipo de vendedor dentro de cada estado de suscripción.

**Qué mostrar en detalle — Errores comunes** (la filmina los trae, conviene desarrollarlos): **olvidar `reset_index()`** tras eliminar filas — el ejemplo de arriba lo aplica explícitamente después de cada `dropna`. **No verificar la consistencia de tipos antes de calcular** — intentar promediar una columna que todavía es texto tira error o, peor, da un resultado sin sentido si Pandas logra forzar la conversión. **Subir archivos de datos extremadamente pesados** — si el dataset del proyecto pesa más de 50MB, va un link a la fuente en el repositorio, no el archivo en sí.

👉 **Volvés a las filminas, Filmina 36.**

#### Filmina 36 — Documentación y Entregable

**Qué decir (ampliando lo que dice la filmina):**

**Paso 5 — Documentación:** el `README.md` del repositorio tiene que explicar, en texto llano (no en código), dos cosas: de dónde salen los datos, y qué decisiones de limpieza se tomaron y por qué. No alcanza con que el código funcione — sin esta documentación, cualquier persona que abra el repositorio ve las transformaciones pero no entiende el criterio detrás de cada una, que es justamente lo que se evalúa en este checkpoint.

**Qué buscamos ver con este ejemplo:** cómo se vería, en la práctica, la sección de documentación del `README.md` — aplicada a las decisiones de limpieza reales que se tomaron en la Filmina 35.

👉 **Acá pasás a Colab.** Última celda del **Bloque 8**: el ejemplo de documentación.

> **Origen de los datos:** extracto de altas de clientes de una plataforma de e-commerce (Company, Product, Subscription Status, Seller Type, Country, Email, actividad de usuarios).
>
> **Decisiones de limpieza:** `SELLER_TYPE` nulo se imputó con la moda porque representa una clasificación faltante, no una ausencia real de tipo. `ACTIVE_USERS_L_28D` nulo se completó con 0 porque significa "sin actividad registrada". Las filas sin `ALL_USERS_COUNT` se eliminaron porque no permiten evaluar adopción del producto. Se eliminaron duplicados por `EMAIL`, quedándose con el primer registro de cada cliente. `COMPANY_CREATED_AT` se convirtió a tipo fecha para poder analizar antigüedad y tendencias temporales.

**Entregable:** un **repositorio en GitHub** (no un solo archivo) que contenga: el notebook o script con la carga, el diagnóstico y la limpieza aplicada; el dataset original (o un link a la fuente si pesa más de 50MB); al menos 3 `groupby` de agregación de negocio; y un `README.md` que documente el origen de los datos y las decisiones de limpieza tomadas.

👉 **Volvés a las filminas, Filmina 37 (cierre).**

---

### Cierre *(Filmina 37)*

¿Dudas? ¿Consultas? Momento para preguntas antes de pasar a la actividad práctica.

---

## ACTIVIDAD PRÁCTICA INTEGRADORA — Tráfico Aéreo

Dataset: `vuelos_asientos_pasajeros.csv` (columnas: `indice_tiempo`, `clasificacion_vuelo`, `pasajeros`, `asientos`, `vuelos`). Esta actividad no tiene filminas propias — es 100% Colab, al final del notebook.

### Paso 1 — Arquitectura y Distribución *(Bloques 1 y 2)*

**Consigna:** cargar el CSV, crear una estructura Figure/Axes, graficar un histograma de `vuelos` con 30 bins, color `#4682B4`.

**Qué buscamos ver:** el primer contacto con el dataset real de la actividad — cuántos vuelos por día son "normales" y si la distribución tiene alguna forma particular (sesgo, outliers) antes de avanzar a comparaciones más complejas.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")
df_vuelos = pd.read_csv("vuelos_asientos_pasajeros.csv")

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(data=df_vuelos, x="vuelos", ax=ax, bins=30, color="#4682B4", kde=True)
ax.set_title("Distribución de la Cantidad de Vuelos Diarios", fontsize=12, fontweight='bold')
ax.set_xlabel("Cantidad de Vuelos por Día", fontsize=10)
ax.set_ylabel("Frecuencia (Días)", fontsize=10)
plt.show()
```

**Qué hace cada línea:**
- `sns.set_theme(style="whitegrid")`: fija un estilo visual global de Seaborn para toda la sesión — todos los gráficos que se dibujen después de esta línea van a compartir el mismo fondo con grilla suave, sin tener que repetirlo en cada celda.
- `df_vuelos = pd.read_csv(...)`: carga el dataset real de tráfico aéreo — se va a reutilizar en los 3 pasos de esta actividad.
- `fig, ax = plt.subplots(figsize=(8, 4))`: la estructura explícita de Figure/Axes que pide la consigna, en vez de usar `plt.figure()` a secas.
- `sns.histplot(data=df_vuelos, x="vuelos", ax=ax, bins=30, color="#4682B4", kde=True)`: histograma de la columna `vuelos` con 30 bins (el número exacto que pide la consigna) y curva KDE superpuesta — la misma técnica vista en el Bloque 2, ahora aplicada a un dataset real en vez de al dataset de ejemplo de propinas.
- `ax.set_title(...)`, `ax.set_xlabel(...)`, `ax.set_ylabel(...)`: la personalización obligatoria vista en la Filmina 06 — sin esto, no queda claro qué mide cada eje.

### Paso 2 — Comparación de Grupos y Relación entre Variables *(Bloque 3)*

**Consigna:** comparar `pasajeros` entre `clasificacion_vuelo` (Cabotaje vs. Internacional) con Boxplot y Violinplot, y un Heatmap de correlación entre `pasajeros`, `asientos`, `vuelos`.

**Qué buscamos ver:** tres gráficos del Bloque 3 aplicados juntos al mismo dataset — si el tipo de vuelo (Cabotaje/Internacional) hace diferencia en la cantidad de pasajeros, y si las tres variables numéricas del dataset están relacionadas entre sí.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_vuelos = pd.read_csv("vuelos_asientos_pasajeros.csv")

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4.5))

sns.boxplot(data=df_vuelos, x="clasificacion_vuelo", y="pasajeros", ax=axes[0], palette="colorblind")
axes[0].set_title("Boxplot: Pasajeros por Tipo de Vuelo")

sns.violinplot(data=df_vuelos, x="clasificacion_vuelo", y="pasajeros", ax=axes[1], palette="colorblind")
axes[1].set_title("Violinplot: Forma de la Distribución")

matriz_corr = df_vuelos[["pasajeros", "asientos", "vuelos"]].corr()
sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[2])
axes[2].set_title("Correlación entre Variables Numéricas")

plt.tight_layout()
plt.show()
```

**Qué hace cada línea:**
- `fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4.5))`: tres Axes en una fila, uno para cada gráfico — el mismo patrón de subplots del Bloque 1, con `axes[0]`, `axes[1]`, `axes[2]`.
- `sns.boxplot(..., x="clasificacion_vuelo", y="pasajeros", ..., palette="colorblind")`: `x` es la variable categórica (Cabotaje/Internacional, define cuántas cajas hay), `y` es la variable numérica que se resume — igual que en el Bloque 3, pero acá además se usa `palette="colorblind"` para que el resultado sea accesible desde el vamos, aplicando lo visto en el Bloque 6.
- `sns.violinplot(...)`: mismos parámetros `x`/`y` que el boxplot, para poder comparar directamente qué información extra aporta la forma completa de la densidad.
- `matriz_corr = df_vuelos[["pasajeros", "asientos", "vuelos"]].corr()`: selecciona las 3 columnas numéricas y calcula su matriz de correlación — igual que en el Bloque 3, pero acá con las variables reales del dataset de vuelos.
- `sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[2])`: dibuja esa matriz como cuadrícula de colores en el tercer Axes, con los mismos parámetros vistos en la Filmina 15 (`annot=True` para ver el número exacto, `vmin`/`vmax` fijos para que la escala de color sea consistente).

### Paso 3 — Mini-EDA con Storytelling *(Bloques 4-6: aplicando jerarquía visual)*

**Consigna:** elegir 2 gráficos (uno de distribución, uno de relación) y escribir una interpretación de 2-3 líneas debajo de cada uno — practicando el hábito de acompañar todo gráfico con una conclusión escrita, no solo mostrarlo.

**Qué buscamos ver:** cerrar la actividad integradora aplicando los principios de storytelling y jerarquía visual vistos después del break (Bloques 4-6) sobre el dataset de vuelos — el gráfico técnicamente correcto MÁS la interpretación en palabras, el mismo hábito que se retoma más adelante en el curso al preparar reportes.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_vuelos = pd.read_csv("vuelos_asientos_pasajeros.csv")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5))

sns.histplot(data=df_vuelos, x="asientos", kde=True, ax=axes[0], color="#5DADE2")
axes[0].set_title("Distribución de Asientos Ofertados")

sns.scatterplot(data=df_vuelos, x="asientos", y="pasajeros", hue="clasificacion_vuelo",
                 palette="colorblind", ax=axes[1])
axes[1].set_title("Asientos Ofertados vs. Pasajeros Reales")

plt.tight_layout()
plt.show()
```

**Qué hace cada línea:**
- `sns.histplot(data=df_vuelos, x="asientos", kde=True, ax=axes[0], color="#5DADE2")`: el gráfico de "distribución" pedido por la consigna — la forma en que se reparten los asientos ofertados día a día.
- `sns.scatterplot(data=df_vuelos, x="asientos", y="pasajeros", hue="clasificacion_vuelo", palette="colorblind", ax=axes[1])`: el gráfico de "relación" pedido por la consigna — cada punto es un día, su posición muestra la relación entre asientos ofertados y pasajeros reales, y el color (accesible) distingue el tipo de vuelo.

**Comentar en clase:** los asientos ofertados se concentran en un rango medio con cola hacia valores altos (vuelos internacionales de mayor porte); existe relación positiva entre asientos y pasajeros en ambas clasificaciones, con más dispersión en Cabotaje.

💡 Para quien quiera ir más allá: el dataset tiene una columna de fecha (`indice_tiempo`) y se presta para practicar series temporales, Plotly interactivo, o exportación en alta resolución — todo eso está en el **Anexo** del notebook.

---

## ANEXO del Notebook — Herramientas Complementarias (fuera del temario actual)

El notebook `Semana_5_Visualizaciones_Avanzadas_en_Data_Science_ (actualizado).ipynb` termina con un Anexo que **no tiene filmina asociada** ni se pide en la Pre-entrega, pero conserva técnicas reales y útiles de una versión anterior del programa:

- **A. GridSpec:** layouts de subplots con tamaños desiguales (mini-dashboards con un gráfico grande y uno chico).
- **B. Series Temporales:** conversión a `datetime`, `resample()` y `rolling()` para suavizar series ruidosas.
- **C. Análisis Multivariado:** canales estéticos `hue`/`style`/`size` para cruzar 4 variables en un gráfico 2D.
- **D. Plotly Express:** scatter interactivo con tooltip, zoom y leyenda clickeable.
- **E. Formatos de Exportación:** rasterizado (PNG/JPG) vs. vectorial (PDF/SVG), DPI, y `bbox_inches='tight'`.

**Qué decir si sobra tiempo:** "esto quedó fuera del programa actual, pero si les interesa profundizar en dashboards con series temporales o gráficos interactivos, está documentado al final del notebook con el mismo nivel de detalle que vimos hoy."

---

## Resumen de Conceptos Clave

| Concepto | Una línea |
|---|---|
| Figure / Axes / Axis | El lienzo completo / cada gráfico individual / la línea graduada de un eje |
| Pyplot vs. OO | Hablarle a `plt` (implícito, ambiguo) vs. a `fig`/`ax` (explícito, recomendado) |
| Histograma / KDE | Barras por bins / curva suave — ambos muestran la forma de una variable |
| Boxplot / Violinplot | 5 números + outliers / boxplot + forma completa de la densidad |
| Heatmap de correlación | Cuadrícula de colores para ver qué variables se mueven juntas |
| Encuadre / Jerarquía / Anotación | Título con unidad / color que destaca lo importante / flecha que explica |
| Eje Y truncado | La trampa más común para exagerar una diferencia mínima |
| Regla de lectura en "Z" | KPIs arriba-izq. → controles arriba-der. → tendencias abajo-izq. → detalle abajo-der. |
| Chartjunk / Data-Ink Ratio | Ruido visual que no aporta información / maximizar lo que sí aporta |
| Pre-entrega: Limpieza y Documentación | Repo GitHub con carga, diagnóstico, limpieza, 3 `groupby` y `README.md` — no un PDF |

## Errores comunes que suelen aparecer

- Usar `plt.title()` en vez de `ax.set_title()` cuando hay más de un Axes — el título termina en el gráfico equivocado.
- Elegir la cantidad de bins de un histograma "porque sí", sin probar un par de valores.
- Usar un violinplot sobre un grupo con muy pocos datos (falsa sensación de suavidad).
- Truncar el eje Y sin aclararlo — aunque no sea intencional, el efecto engañoso es el mismo.
- Armar un dashboard con más de 5-6 visualizaciones principales — genera ruido en vez de claridad.
- Usar rojo/verde como único diferenciador de categorías, sin respaldo de forma o trazo.
- En la Pre-entrega: olvidar `reset_index()` tras eliminar filas, imputar sin pensar en qué significa el nulo, o entregar el notebook sin `README.md` documentando las decisiones.
