# Readme Repaso — Clase 05: Visualizaciones Avanzadas en Data Science

Guion de clase para el profesor. Cada sección incluye qué decir, qué mostrar y qué ejecutar en el notebook. La Parte 2 sigue el orden exacto de las filminas de `Clase05.html` (37 filminas) e interleava con el notebook `Semana_5_Visualizaciones_Avanzadas_en_Data_Science_ (actualizado al docx nuevo).ipynb`. Cada vez que hay código, el texto marca explícitamente **👉 cuándo pasar a Colab** y **👉 cuándo volver a las filminas**, para que sea fácil ir alternando ventanas en vivo.

---

## Antes de empezar — Encuadre de la clase

> "La clase de hoy tiene dos partes. En la primera hacemos un repaso de todo lo visto hasta ahora: variables, control de flujo, NumPy y Pandas. En la segunda parte arrancamos con el tema nuevo: visualizaciones avanzadas, siguiendo las filminas y el notebook. Al final tienen una actividad práctica con un dataset real de tráfico aéreo, y una pre-entrega evaluable sobre el dataset de su propio proyecto."

Abrir tres archivos antes de arrancar, y dejarlos en pestañas separadas para poder alternar rápido:
- Repaso: `Repaso_Data_Science_I_Fundamentos_para_la_Ciencia_de_Datos_.ipynb`
- Filminas: `Clase05.html`
- Código de la Parte 2: `Semana_5_Visualizaciones_Avanzadas_en_Data_Science_ (actualizado al docx nuevo).ipynb`

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
| 33–36 | Pre-entrega: EDA Visual del Proyecto | Bloque 8 |
| 37 | Cierre | — |

> Todo lo que quedó fuera del `Clase 05.docx` vigente (GridSpec, series temporales, análisis multivariado con `hue/style/size`, Plotly, formatos de exportación) sigue disponible en el **Anexo** del notebook — no tiene filmina asociada ni se pide en la Pre-entrega, pero está para quien quiera ir más allá.

**Qué decir al arrancar:**

> "Antes hacíamos gráficos como parte de otras clases. Hoy la visualización es el tema central: no solo cómo hacer un gráfico en Python, sino cómo diseñarlo para que comunique bien, y cómo armar un dashboard completo que un ejecutivo entienda en 3 segundos. Vamos a ir filmina por filmina, y en cada bloque les voy a avisar cuándo pasamos al notebook y cuándo volvemos a las filminas."

---

### BLOQUE 1 — Fundamentos de Matplotlib *(Filminas 02–06)*

**Filmina 02 — División de módulo.** Da el pie para el primer bloque: Figure, Axes, los dos estilos de trabajo, subplots y personalización. Sin código todavía — es la filmina de transición, se pasa rápido.

#### Filmina 03 — Figure, Axes y Axis

**Qué decir (ampliando lo que dice la filmina):**

La filmina resume la diferencia en tres bullets. Para que quede realmente claro, conviene explicarlo con la analogía completa: Matplotlib separa tres conceptos que se confunden todo el tiempo al empezar.

- **`Figure`**: es la hoja de papel completa. El objeto de más alto nivel — lo que termina siendo la imagen final que se muestra o se guarda. Puede estar vacía, o tener adentro uno o cien gráficos.
- **`Axes`**: es cada gráfico individual dentro de esa hoja. Ojo: en inglés "Axes" es plural, pero acá se usa para referirse a UN gráfico completo (con su título, su leyenda, sus datos). Es el objeto con el que más se trabaja.
- **`Axis`** (sin la "e" del final): es solamente la línea graduada de un eje — el X o el Y — con sus marcas (*ticks*) y sus números.

**Analogía que no está en la filmina:** una `Figure` es un álbum de fotos. Cada `Axes` es una foto individual dentro del álbum. El `Axis` es apenas el marco numerado de esa foto (los números en el borde). La regla práctica: casi todo lo que uno personaliza (título, etiquetas, límites, leyenda) se hace sobre el objeto `ax` (Axes), no sobre `axis`.

Una `Figure` puede tener un solo `Axes` (un gráfico sencillo) o diez `Axes` (un dashboard completo) — este último caso lo vamos a ver en detalle en el Bloque 6.

Esta filmina es 100% teoría, todavía no hay código para correr — se sigue derecho a la Filmina 04.

**Preguntar a la clase:**

> Si armamos un dashboard con 4 gráficos en una sola imagen, ¿cuántas Figure y cuántos Axes hay?

**Respuesta:** Una sola `Figure` (la imagen final completa) y 4 `Axes` (uno por cada gráfico individual dentro de ella).

#### Filmina 04 — Pyplot vs. Orientado a Objetos

**Qué decir (ampliando lo que dice la filmina):**

La filmina muestra los dos bloques de código en paralelo. Antes de correrlos, conviene explicar el POR QUÉ:

- **Estilo Pyplot (implícito):** se le habla directo al módulo `plt` (`plt.plot()`, `plt.title()`). Matplotlib asume que el comando aplica al último gráfico que se creó. Es rápido para un gráfico suelto, pero se vuelve ambiguo apenas hay dos o más gráficos en pantalla — es como dar instrucciones sin decir a quién.
- **Estilo Orientado a Objetos (explícito, recomendado):** se crean variables `fig` y `ax` y se les habla directamente a ellas (`ax.plot()`, `ax.set_title()`). No hay ambigüedad posible: cada instrucción le habla a un objeto puntual.

En Data Science se recomienda casi siempre el estilo OO porque en cuanto se arma un panel comparativo (algo constante en EDA), el estilo Pyplot se vuelve un lío.

👉 **Acá pasás a Colab.** Abrí `Semana_5_Visualizaciones_Avanzadas_en_Data_Science_ (actualizado al docx nuevo).ipynb`, andá al **Bloque 1** y corré la celda del ejemplo "Estilo Pyplot vs. Estilo Orientado a Objetos".

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

**Qué mostrar en detalle:**
- Con un solo gráfico, la diferencia visual es nula — hacer notar que el problema aparece recién con 2+ Axes (se ve en la próxima filmina).
- Mostrar `type(fig)` → `Figure`, `type(ax)` → `AxesSubplot`.

👉 **Volvés a las filminas, Filmina 05.**

#### Filmina 05 — `subplots()`: Varias Vistas en un Panel

**Qué decir (ampliando lo que dice la filmina):**

`plt.subplots(nrows, ncols)` crea de una vez la `Figure` y todos los `Axes` vacíos, organizados en una grilla. A partir de ahí, cada `ax.algo()` dibuja SOLO en ese Axes puntual — se termina la ambigüedad de no saber dónde está dibujando `plt.plot()`, que es justo el problema del estilo Pyplot que vimos en la filmina anterior.

Esto es central para el **Análisis Exploratorio de Datos (EDA)**: casi siempre se necesita comparar dos vistas relacionadas en una sola imagen — la filmina menciona el ejemplo de un proyecto inmobiliario: un histograma de precios al lado de un scatter de metros² vs. precio.

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

**Qué mostrar en detalle:**
- `fig.suptitle()` vs `ax.set_title()`: el primero titula TODA la figura, el segundo titula un solo Axes. Mostrar la diferencia sacando uno de los dos.
- `(ax1, ax2) = plt.subplots(...)`: el desempaquetado automático de la tupla de Axes en dos variables con nombre.

👉 **Volvés a las filminas, Filmina 06.**

#### Filmina 06 — Personalización Completa

**Qué decir (ampliando lo que dice la filmina):**

La filmina lista los comandos (título, labels, leyenda, límites, tight_layout) como una tabla. Vale la pena remarcar el POR QUÉ de cada uno, no solo el QUÉ: un gráfico sin etiquetas es solo una forma abstracta — no comunica nada por sí solo.

- `ax.set_title()`: el nombre de la historia que se está contando.
- `ax.set_xlabel()` / `ax.set_ylabel()`: qué mide cada eje, en qué unidades. "Ingresos (USD)" es información; "Ingresos" a secas no dice si son pesos, dólares o millones.
- `ax.legend()`: obligatoria en cuanto hay más de una serie en el mismo Axes.
- `ax.set_xlim()` / `ax.set_ylim()`: el rango visible lo decide quien programa, no Matplotlib solo — esto se vuelve crítico más adelante cuando hablemos de ejes truncados en el Bloque 4.
- `fig.tight_layout()`: el salvavidas cuando los subplots se pisan entre sí.

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

**Qué mostrar en detalle — Errores comunes** (esto no está en la filmina, agregarlo de palabra): confundir `Axes` con `Axis`; no guardar `fig, ax` (se pierde control fino apenas hay más de un gráfico); el "gráfico de espagueti" (10 líneas de colores en un solo Axes — mejor separar en subplots); olvidar las unidades del eje.

👉 **Volvés a las filminas, Filmina 07 (división de módulo).**

---

### BLOQUE 2 — Análisis Univariado: Histogramas y KDE *(Filminas 07–10)*

**Filmina 07 — División de módulo.**

#### Filmina 08 — El Histograma y la Cantidad de Bins

**Qué decir (ampliando lo que dice la filmina):**

Antes de buscar relaciones entre variables, el primer paso de todo análisis es mirar cada variable **por separado**, respondiendo cuatro preguntas: ¿cuál es el valor más común?, ¿están dispersos o concentrados?, ¿hay outliers?, ¿es simétrica o tiene cola hacia un lado? La filmina lista estas preguntas — vale la pena que la clase las tenga anotadas porque van a reaparecer en el Bloque 8 (Pre-entrega).

El **histograma** divide el rango de una variable numérica en intervalos (*bins*) y cuenta cuántos datos caen en cada uno. La cantidad de bins cambia completamente el mensaje, y esto la filmina lo dice pero sin ejemplo visual — conviene remarcarlo oralmente:
- **Pocos bins:** imagen demasiado general — puede ocultar, por ejemplo, que la distribución tiene dos picos.
- **Muchos bins:** el gráfico se vuelve ruidoso, como una serie de picos aislados sin patrón reconocible.

No hay un número "correcto" universal — hay que probar un par de valores y quedarse con el que mejor cuenta la historia sin mentir ni esconder. El código de esta idea se corre recién en la próxima filmina, junto con el KDE.

#### Filminas 09–10 — KDE y Lectura de la Forma

**Qué decir (ampliando lo que dicen las filminas):**

El **KDE** (*Kernel Density Estimation*, estimación de densidad por kernel) reemplaza los bloques rígidos del histograma por pequeñas "colinas" suaves puestas sobre cada observación, que después se suman todas. El resultado es una curva continua que muestra la forma subyacente de los datos sin depender de la elección arbitraria de bins.

Con el histograma y el KDE en mano, se puede leer la tabla que trae la Filmina 10:

| Patrón | Qué significa |
|---|---|
| **Simetría** | Izquierda espejo de la derecha (Campana de Gauss). |
| **Sesgo a la derecha** | Cola larga hacia valores altos (salarios, precios de casas). |
| **Sesgo a la izquierda** | Cola larga hacia valores bajos. |
| **Multimodalidad** | Dos o más "jorobas" → hay subgrupos mezclados en la misma variable (ej. alturas de hombres y mujeres juntas). |

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

Seaborn **no es "mejor"** que Matplotlib — está construido ENCIMA de Matplotlib, pensado específicamente para graficar DataFrames de Pandas de forma rápida. La diferencia práctica: en Matplotlib puro hay que filtrar cada grupo a mano y dibujarlo por separado; en Seaborn se le pasa el DataFrame completo y los nombres de columnas, y arma solo colores, leyenda y estilo.

- **Seaborn** → ideal para EDA: histogramas, boxplots, correlaciones, scatterplots con categorías.
- **Matplotlib** → cuando hace falta personalizar mucho un gráfico puntual, o algo que Seaborn no tiene resuelto.

**Esto no está en la filmina, pero conviene agregarlo:** Seaborn también tiene dos "modos" de funcionar: funciones **Axes-level** (`scatterplot`, `boxplot`, `histplot`) que aceptan `ax=` y dibujan donde se les indica, y funciones **Figure-level** (`displot`, `relplot`, `catplot`) que arman su propia Figure y devuelven un `FacetGrid` — ideales para exploración rápida o para separar en paneles por categoría con `col=`. Pasarle `ax=` a una función figure-level tira error.

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

**Qué mostrar en detalle:** `hue="sex"` reemplaza el `for` completo — Seaborn separa, colorea y arma la leyenda automáticamente.

👉 **Volvés a las filminas, Filminas 13–14.**

#### Filminas 13–14 — Boxplot y Violinplot

**Qué decir (ampliando lo que dicen las filminas):**

Un **Boxplot** resume una distribución en 5 números: **Mínimo**, **Q1 (25%)**, **Mediana**, **Q3 (75%)** y **Máximo**. La caja es el rango intercuartílico (IQR — el 50% central de los datos). Los puntos fuera de los "bigotes" son outliers — ojo, no siempre son un error de carga: pueden ser el cliente más valioso de la empresa (esto no está en la filmina, pero es un buen disparador de discusión).

Un **Violinplot** combina el boxplot con una curva KDE: el ancho del violín representa la frecuencia. Dos grupos pueden tener exactamente la misma mediana y los mismos cuartiles (boxplots idénticos) pero formas de distribución completamente distintas — el violín revela eso que el boxplot solo no puede mostrar.

- Usar **Boxplot** si la audiencia no es técnica, o solo importa variabilidad y extremos.
- Usar **Violinplot** si hace falta la forma precisa y hay suficientes registros (con pocos datos, el violín miente dando una falsa sensación de suavidad).

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

👉 **Volvés a las filminas, Filmina 15.**

#### Filmina 15 — Heatmap de Correlación y Estrategia de Elección

**Qué decir (ampliando lo que dice la filmina):**

Un **heatmap** convierte una tabla de números en una cuadrícula de colores. El uso más común: visualizar la **matriz de correlación** (coeficiente de Pearson, entre -1 y 1).

- **Cerca de 1** (color intenso): las variables se mueven juntas (metros² y precio).
- **Cerca de -1** (color intenso opuesto): direcciones contrarias (peso del auto y eficiencia de combustible).
- **Cerca de 0** (color neutro): sin relación clara.

La filmina cierra este bloque con la tabla de estrategia:

| Pregunta | Gráfico |
|---|---|
| ¿Outliers o dispersión en un grupo? | Boxplot |
| ¿Forma y densidad de la distribución? | Violinplot |
| ¿Qué variables están relacionadas? | Heatmap |

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

**Analogía del mapa vs. la lista de direcciones** (la filmina la nombra, conviene desarrollarla en voz alta): una lista de direcciones es precisa, pero no da contexto — como una tabla de datos cruda. Un mapa muestra qué hay alrededor y cuán lejos se está del destino — como una buena visualización.

**Función cognitiva** (esto amplía el "bolt" de la filmina): leer una tabla es secuencial (fila por fila, comparando mentalmente). Un gráfico permite que la percepción detecte patrones en milisegundos. Ejemplo para dar en vivo: en 1.000 transacciones bancarias, encontrar una sospechosa a mano lleva minutos; en un scatter, un punto disparado del resto se ve al instante.

**Nota para el docente:** esta filmina no tiene código propio. Antes de seguir a la Filmina 19, es un buen momento para bajar la teoría a tierra con dos ejemplos aplicados del notebook que, aunque no tienen una filmina 1 a 1, sirven de "banco de pruebas" para todo lo que se explica en las Filminas 18 a 22.

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

**Qué decir sobre este ejemplo:** mismos datos, dos gráficos. A la izquierda, `ax.bar()` a secas — no dice unidad ni qué mirar. A la derecha, tres decisiones de diseño encima: título con unidad (**encuadre**), un color distinto para el mes que importa (**jerarquía visual**), y una flecha que explica por qué (**anotación**).

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

**Qué decir sobre este ejemplo:** la primera versión (torta con Plotly) separa el sector de Apple y titula "35% Apple" — visualmente parece que domina. La segunda versión (barras horizontales) ordena por valor real: "Otras Compañías" (40%) queda primera. Los números son idénticos — lo que cambia es el diseño. **¿Cuál es mejor? El de barras.** No porque el pie chart esté mal hecho técnicamente, sino porque `pull` + el título elegido comunican una conclusión que los datos no sostienen. Un gráfico honesto no es el más lindo: es el que se interpreta bien sin ayuda de quien lo hizo.

👉 **Volvés a las filminas, Filmina 19.**

#### Filminas 19–21 — Pilares, Dashboards, Semáforo de Color y Casos Reales

**Qué decir (ampliando lo que dicen las filminas):**

**Cuatro pilares del análisis visual** (Filmina 19):

| Pilar | Pregunta | Gráfico típico |
|---|---|---|
| Distribución | ¿Cómo se reparten mis datos? | Histograma, KDE |
| Relación | ¿Depende X de Y? | Scatter, Heatmap |
| Evolución temporal | ¿Crecemos, decrecemos, estancamos? | Línea |
| Composición | ¿Qué % viene de cada parte? | Barras apiladas |

**Data Storytelling** (agregar de palabra, no está desarrollado en la filmina): todo análisis visual tiene Inicio (contexto: "las ventas fluctuaron"), Nudo (el problema: "cae fuerte en la región norte") y Desenlace (la acción: "investigar al nuevo competidor").

**De gráficos sueltos a dashboards** (Filmina 20): la analogía del tablero de un auto — velocímetro, combustible, temperatura del motor combinados para que el conductor actúe rápido. **Regla de oro:** máximo 5 visualizaciones principales por dashboard.

**Semáforo cognitivo del color y casos reales** (Filmina 21): secuenciales (variables numéricas, azul claro→oscuro), divergentes (punto medio crítico, rojo/blanco/azul), categóricos (grupos sin orden, pero respetando convenciones — rojo = pérdida, no al revés). Casos reales que trae la filmina: Retail (heatmaps de recorrido de clientes, +15% ventas reubicando productos), Salud pública (áreas apiladas de capacidad hospitalaria), Amazon (caída súbita en "añadir al carrito" delata un bug en segundos), Finanzas personales (donuts de gastos).

Estas tres filminas son teóricas, sin código propio — el "banco de pruebas" ya se corrió antes (Filmina 18). Se sigue derecho a la Filmina 22.

**Preguntar a la clase:**

> Si tuvieras que explicarle a un gerente por qué las ventas bajaron el último mes en 10 segundos, ¿qué gráfico usarías y por qué?

#### Filmina 22 — Errores Comunes y Trampas Visuales

**Qué decir (ampliando lo que dice la filmina):**

La filmina trae la tabla de 3 trampas (Eje Y truncado, Pie chart con muchas categorías, Exceso de Data-Ink). La más fácil de demostrar en vivo es el **eje Y truncado**: empezar el eje vertical en un número distinto de cero para exagerar una diferencia. Puede hacer que un crecimiento del 1% parezca del 50%. Mantener siempre el cero como base, salvo razón estadística muy fuerte — y si la hay, aclararlo explícitamente en el gráfico.

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

**Qué mostrar en detalle:** mismos 4 números en los dos gráficos — el de la izquierda "grita" una tendencia dramática que, en la escala real (derecha), es casi imperceptible. Las otras dos trampas de la tabla (pie chart con muchas categorías, exceso de Data-Ink) se retoman en el Bloque 6 con el ejemplo de chartjunk.

👉 **Volvés a las filminas, Filmina 23 (división de módulo).**

---

### BLOQUE 5 — Arquitectura Visual y la Regla de Lectura en "Z" *(Filminas 23–24)*

**Filmina 23 — División de módulo.**

#### Filmina 24 — El Patrón de Lectura en "Z"

**Qué decir (ampliando lo que dice la filmina):**

Un dashboard (en Excel o cualquier herramienta de BI) es una **interfaz de usuario**, no una colección de gráficos insertados. El objetivo es reducir la carga cognitiva: el usuario debe extraer los insights más críticos en los primeros 3 segundos — esto conecta directo con la "regla de oro" de 5 visualizaciones que vimos en la Filmina 20.

En culturas occidentales leemos de izquierda a derecha y de arriba hacia abajo. En pantallas sin mucho texto continuo, el ojo escanea siguiendo un recorrido en forma de la letra **"Z"**, tal como muestra la tabla de la filmina:

| Posición | Qué colocar |
|---|---|
| 1. Arriba-Izquierda (Ancla visual) | KPIs principales: Ventas Totales, Margen, ROI |
| 2. Arriba-Derecha (Contexto y control) | Slicers, línea de tiempo |
| 3. Abajo-Izquierda (Profundidad analítica) | Tendencias, comparativas clave |
| 4. Abajo-Derecha (Punto de salida) | Tablas de detalle, top 10 |

Recorrido recomendado: **KPIs → Controles/Slicers → Gráficos de Tendencia → Tablas de Detalle**.

**Este bloque es 100% conceptual — no hay celda de código en el notebook para correr acá.** Se aplica al diseñar dashboards en Excel/BI, no en Python. Se sigue derecho a la Filmina 25.

**Preguntar a la clase:**

> Si tuvieran que armar un dashboard de ventas con 4 elementos (un KPI de facturación total, un filtro de fecha, un gráfico de tendencia mensual y una tabla de top clientes), ¿dónde pondrían cada uno siguiendo la "Z"?

---

### BLOQUE 6 — Principios de UI/UX y Jerarquía Visual en Dashboards *(Filminas 25–29)*

**Filmina 25 — División de módulo.**

#### Filminas 26–27 — Jerarquía Visual y sus Pilares

**Qué decir (ampliando lo que dicen las filminas):**

**Analogía de la cabina de avión** (Filmina 26): cientos de indicadores compitiendo por la atención. Un piloto entrenado sabe dónde mirar; un ejecutivo con 5 minutos antes de una junta, no. La jerarquía visual es el sistema que le dice al usuario "mirá esto primero, esto después, esto solo si necesitás el detalle". Sin jerarquía, todo "pesa" igual — el resultado es ruido visual. El cerebro es perezoso por naturaleza: busca atajos.

Los 4 pilares que trae la Filmina 27:

| Pilar | Idea clave |
|---|---|
| Patrón F/Z | Zona superior izq. = KPIs críticos. Zona media = tendencias. Zona inferior/derecha = detalle. |
| Tamaño y peso | Lo más grande se percibe como más importante. Fuentes Sans Serif para números. |
| Color con propósito | Semánticos (rojo=alerta, verde=cumplimiento, ámbar=atención). Grises para ejes/etiquetas. |
| Espacio en blanco | No es espacio desperdiciado: agrupa y separa conceptos. |

Estas dos filminas son teóricas — el código llega en la próxima, con el ejemplo de chartjunk que baja el pilar de "espacio en blanco / color con propósito" a código real.

#### Chartjunk vs. Gráfico Limpio *(código que ilustra las Filminas 27 y 29)*

**Qué decir (esto no está desarrollado como ejemplo en las filminas, pero conecta directo con la tabla de Filmina 29):**

El mismo principio de "menos es más" se aplica a un gráfico de Matplotlib: cada borde, sombra o línea de más es *ruido* que compite con el dato. Edward Tufte llama a esto maximizar el **Data-to-Ink Ratio**: si se puede quitar un elemento y el dato se sigue entendiendo igual de bien, hay que quitarlo. Esta es también la tercera trampa que había quedado pendiente de la Filmina 22 (Exceso de Data-Ink).

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

👉 **Volvés a las filminas, Filmina 28.**

#### Filminas 28–29 — Arquitectura de la Información, Anti-Patrones y Accesibilidad

**Qué decir (ampliando lo que dicen las filminas):**

Un dashboard ejecutivo es como una conversación (Filmina 28): **El Titular (KPIs)** — "¿cómo vamos?"; **El Contexto (Tendencias)** — "¿cómo llegamos hasta acá?"; **El Diagnóstico (Desgloses)** — "¿quién o qué lo causó?"; **El Detalle (Tablas)** — "mostrame los datos exactos".

*Ejemplo real que trae la filmina — dashboard de un Controller:* Nivel 1: EBITDA/Ingresos/Costos/Desviación. Nivel 2: gráfico de cascada. Nivel 3: top 5 centros de costo excedidos. Nivel 4: tabla dinámica filtrable.

**Anti-patrones** (Filmina 29):

| Anti-patrón | Solución |
|---|---|
| Efecto Árbol de Navidad | Demasiados colores vibrantes compitiendo — usar paleta monocromática, reservar el brillante para excepciones |
| Chartjunk | Bordes/sombras/grids pesados — visto en el ejemplo de la Filmina 27 |
| Dato Huérfano | "$1.000.000" sin contexto — acompañar con comparativo (vs. Target, vs. Año Anterior) |

**Accesibilidad** (también Filmina 29): ~8% de los hombres tiene algún tipo de daltonismo (el más común: confunde rojo y verde). Respaldar siempre el color con `style=` y usar `palette="colorblind"`.

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

👉 **Volvés a las filminas, Filmina 30 (división de módulo).**

---

### BLOQUE 7 — Práctica: Auditoría UX de Dashboards *(Filminas 30–32, sin código)*

**Filmina 30 — División de módulo.**

#### Filmina 31 — Ejercicio 1: Auditoría y Rediseño

**Qué decir (ampliando lo que dice la filmina):** para esta práctica no se usa código, sino capacidad de análisis crítico. El escenario: un gráfico de pastel con 15 categorías en colores similares, títulos en Comic Sans azul brillante, y un KPI de "Ventas" sin comparativo — la filmina resume la consigna, conviene leerla en voz alta completa antes de dar tiempo a resolver.

**Consigna:** identificar 4 fallos de jerarquía visual/UX, proponer un layout con el patrón "F", definir una guía de estilo (máx. 3 colores + grises), y contextualizar el KPI con una métrica de comparación.

Sin código — no hay transición a Colab en este bloque.

#### Filmina 32 — Ejercicio 2: Caso "Dashboard Comercial - Q3"

**Qué decir (ampliando lo que dice la filmina):** sos el nuevo analista de una empresa de retail. El gerente dice: *"la información está ahí, pero tardamos 10 minutos en entenderla"*. Diagnóstico: al menos 3 violaciones a UI/UX y al patrón "Z" (conectar con lo visto en las Filminas 24 y 26-29). Reestructuración: redistribuir tabla dinámica, gráfico circular, slicers y tarjetas de KPI con el modelo "Z".

**Preguntar a la clase:** dejar que un par de estudiantes propongan su diagnóstico en voz alta antes de mostrar la respuesta esperada — este bloque funciona mejor como discusión abierta que como exposición.

👉 **Volvés a las filminas, Filmina 33 (división de módulo).**

---

### BLOQUE 8 — Pre-entrega: EDA Visual del Dataset del Proyecto *(Filminas 33–36)*

**Filmina 33 — División de módulo.** Este es el cierre evaluable de la clase.

#### Filmina 34 — Dónde Estamos y Qué Sigue

**Qué decir (ampliando lo que dice la filmina):**

Hasta acá el proyecto ya recorrió: configuración del entorno, carga y saneamiento del dataset (Módulo 4), transformación con Pandas — la filmina lo resume en 3 bullets. Vale la pena remarcar la frase clave que sí trae la filmina: este checkpoint es el **puente hacia el análisis predictivo**. Las relaciones que se descubran hoy deciden qué variables se usan en los modelos de Machine Learning más adelante — no es un ejercicio aislado de visualización, es la base del próximo módulo del curso.

Sin código en esta filmina — se sigue a la 35.

#### Filmina 35 — Qué Construir

**Qué decir (ampliando lo que dice la filmina):**

Sin exportar todos los gráficos posibles, seleccionar solo los que aporten valor:
- **Univariado:** ≥2 gráficos de distribución (histogramas con KDE) — esto es literalmente el Bloque 2 de hoy.
- **Bivariado/Multivariado:** ≥3 gráficos de relación o comparación (boxplots, scatter, heatmap de correlación) — el Bloque 3 de hoy.
- **Interpretación:** 2-3 líneas por gráfico, explicando qué se ve y qué conclusión preliminar se puede sacar — esto es nuevo, no se pidió en ningún bloque anterior, conviene remarcarlo.

**Errores a evitar** (la filmina los trae en el pie de página, conviene leerlos completos): el "vómito de gráficos" (20 sin explicación — mejor 5 bien analizados), ejes sin escala ni etiqueta, ignorar la distribución antes de analizar correlación.

#### Filmina 36 — El Entregable

**Qué decir (ampliando lo que dice la filmina):**

Un único archivo **PDF**: portada (título + nombre), resumen del dataset (1 párrafo), mínimo 5 visualizaciones (≥1 histograma/KDE, ≥1 boxplot/violinplot, ≥1 scatter/heatmap), y análisis por gráfico. **No se entrega código** en este paso — se evalúa la visualización y la comunicación. Nombre sugerido: `EDA_Visual_Apellido_Nombre.pdf`.

👉 **Acá pasás a Colab.** Última celda del **Bloque 8**: el mini-ejemplo de cómo se ve una de las 5 visualizaciones pedidas, aplicado al dataset de vuelos.

**Ejecutar:**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df_vuelos = pd.read_csv("vuelos_asientos_pasajeros.csv")

fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(data=df_vuelos, x="pasajeros", kde=True, ax=ax, color="#4682B4")
ax.set_title("Distribución de Pasajeros Diarios", fontsize=13, fontweight="bold")
ax.set_xlabel("Cantidad de Pasajeros por Día")
ax.set_ylabel("Frecuencia (Días)")
plt.tight_layout()
plt.show()
```

> **Interpretación de ejemplo (así se escribe en el PDF final):** la distribución de pasajeros diarios muestra una concentración principal entre valores medios, con una cola hacia la derecha — hay un grupo más chico de días con tráfico excepcionalmente alto que conviene revisar por separado antes de correlacionar esta variable con otras.

👉 **Volvés a las filminas, Filmina 37 (cierre).**

---

### Cierre *(Filmina 37)*

¿Dudas? ¿Consultas? Momento para preguntas antes de pasar a la actividad práctica.

---

## ACTIVIDAD PRÁCTICA INTEGRADORA — Tráfico Aéreo

Dataset: `vuelos_asientos_pasajeros.csv` (columnas: `indice_tiempo`, `clasificacion_vuelo`, `pasajeros`, `asientos`, `vuelos`). Esta actividad no tiene filminas propias — es 100% Colab, al final del notebook.

### Paso 1 — Arquitectura y Distribución *(Bloques 1 y 2)*

**Consigna:** cargar el CSV, crear una estructura Figure/Axes, graficar un histograma de `vuelos` con 30 bins, color `#4682B4`.

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

### Paso 2 — Comparación de Grupos y Relación entre Variables *(Bloque 3)*

**Consigna:** comparar `pasajeros` entre `clasificacion_vuelo` (Cabotaje vs. Internacional) con Boxplot y Violinplot, y un Heatmap de correlación entre `pasajeros`, `asientos`, `vuelos`.

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

### Paso 3 — Mini-EDA con Storytelling *(Bloque 8 — ensayo de la Pre-entrega)*

**Consigna:** elegir 2 gráficos (uno de distribución, uno de relación) y escribir una interpretación de 2-3 líneas debajo de cada uno — es un ensayo directo del formato de la Pre-entrega, aplicado al dataset de vuelos en vez del dataset propio.

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

**Comentar en clase:** los asientos ofertados se concentran en un rango medio con cola hacia valores altos (vuelos internacionales de mayor porte); existe relación positiva entre asientos y pasajeros en ambas clasificaciones, con más dispersión en Cabotaje.

💡 Para quien quiera ir más allá: el dataset tiene una columna de fecha (`indice_tiempo`) y se presta para practicar series temporales, Plotly interactivo, o exportación en alta resolución — todo eso está en el **Anexo** del notebook.

---

## ANEXO del Notebook — Herramientas Complementarias (fuera del temario actual)

El notebook `Semana_5_Visualizaciones_Avanzadas_en_Data_Science_ (actualizado al docx nuevo).ipynb` termina con un Anexo que **no tiene filmina asociada** ni se pide en la Pre-entrega, pero conserva técnicas reales y útiles de una versión anterior del programa:

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
| Pre-entrega EDA Visual | PDF con ≥5 gráficos e interpretación — el puente hacia Machine Learning |

## Errores comunes que suelen aparecer

- Usar `plt.title()` en vez de `ax.set_title()` cuando hay más de un Axes — el título termina en el gráfico equivocado.
- Elegir la cantidad de bins de un histograma "porque sí", sin probar un par de valores.
- Usar un violinplot sobre un grupo con muy pocos datos (falsa sensación de suavidad).
- Truncar el eje Y sin aclararlo — aunque no sea intencional, el efecto engañoso es el mismo.
- Armar un dashboard con más de 5-6 visualizaciones principales — genera ruido en vez de claridad.
- Usar rojo/verde como único diferenciador de categorías, sin respaldo de forma o trazo.
- En la Pre-entrega: entregar código en vez de PDF, o gráficos sin interpretación escrita.
