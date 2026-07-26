# Clase 03: NumPy y Pandas — Guía Completa para el Docente

Esta guía es el **libreto de apoyo para dictar la Clase 03**. Reúne, en un solo lugar y con más profundidad de la que entra en una diapositiva, toda la teoría y todos los ejemplos que aparecen en:

- **`Clase 03.pdf`** — el material teórico oficial de la unidad (9 secciones: de Arrays y Vectorización hasta la Inspección Inicial de datos).
- **`Clase03.html`** — las diapositivas que se proyectan en clase (43 filminas).
- **`Clase 03.ipynb`** — el notebook con todos los ejemplos ejecutables y comentados, con teoría ampliada en celdas Markdown.

La idea es la misma que en la guía de Clase 02: si en medio de la clase te falla la memoria sobre un detalle (¿por qué `reshape` tira error si no coinciden las dimensiones?, ¿cuál era la regla de broadcasting?, ¿`agg` o `transform`?), lo encuentres acá explicado con más contexto del que alcanza a mostrar una filmina.

Todos los ejemplos de código usan `stocks.csv` (en esta misma carpeta) como dataset real de referencia: precios mensuales de 14 acciones entre 2016 y 2021.

---

## Índice

- [Mapa rápido de la clase](#0-mapa-rápido-de-la-clase)
- [Sobre el Dataset: `stocks.csv`](#sobre-el-dataset-stockscsv)

0. [Bloque 0 — Repaso de la Clase Anterior](#bloque-0--repaso-de-la-clase-anterior-fundamentos-de-python)
1. [Módulo 1 — Arrays Multidimensionales y Vectorización en NumPy](#módulo-1--arrays-multidimensionales-y-vectorización-en-numpy)
2. [Módulo 2 — Broadcasting y Operaciones sobre Matrices](#módulo-2--broadcasting-y-operaciones-sobre-matrices)
3. [Módulo 3 — Álgebra Lineal con NumPy](#módulo-3--álgebra-lineal-con-numpy)
4. [Módulo 4 — NumPy y Pandas: la Relación entre el Cálculo y la Estructura de Datos](#módulo-4--numpy-y-pandas-la-relación-entre-el-cálculo-y-la-estructura-de-datos)
5. [Módulo 5 — Introducción a Pandas: Series y DataFrames](#módulo-5--introducción-a-pandas-series-y-dataframes)
6. [Módulo 6 — Preprocesamiento de Datos](#módulo-6--preprocesamiento-de-datos)
7. [Módulo 7 — Integración, Agregación y Preprocesamiento Avanzado](#módulo-7--integración-agregación-y-preprocesamiento-avanzado)
8. [Módulo 8 — La Sinergia de Datos: NumPy y Pandas en Profundidad](#módulo-8--la-sinergia-de-datos-numpy-y-pandas-en-profundidad)
9. [Módulo 9 — Inspección Inicial de Datos y Pre-Entrega](#módulo-9--inspección-inicial-de-datos-y-pre-entrega)

---

## 0. Mapa rápido de la clase

Para ir siguiendo la clase en paralelo con `Clase03.html` (43 filminas) sin perderte: qué filmina corresponde a qué sección de esta guía y del notebook.

| # | Módulo | Slides | Notebook (`Clase 03.ipynb`) | Idea central |
|---|---|---|---|---|
| — | Portada | 01 | Celda de título | Presentación de la clase |
| 0 | Repaso de la Clase Anterior | *(sin slides — repaso previo)* | Bloque 0 | Variables, condicionales, `for` y funciones, frescos para contrastar con la vectorización |
| 1 | Arrays Multidimensionales y Vectorización | 02–09 | Módulo 1 | El `ndarray`, su homogeneidad, y por qué reemplaza al `for` |
| 2 | Broadcasting y Operaciones sobre Matrices | 10–15 | Módulo 2 | Operar arrays de distinta forma sin bucles ni copias de más |
| 3 | Álgebra Lineal con NumPy | 16–19 | Módulo 3 | Producto punto, sistemas de ecuaciones, diagnóstico matricial |
| — | Break del Coder | 20 | — | Corte de ~10 minutos |
| 4 | NumPy y Pandas: la Relación | 21–23 | Módulo 4 | Dos librerías, dos especialidades — y por qué Pandas está construido sobre NumPy |
| 5 | Series y DataFrames | 24–28 | Módulo 5 | Las dos estructuras de Pandas, con etiquetas en vez de posiciones |
| 6 | Preprocesamiento de Datos | 29–31 | Módulo 6 | Eliminación vs. imputación, y la regla de oro contra el Data Leakage |
| 7 | Integración, Agregación y Preprocesamiento Avanzado | 32–36 | Módulo 7 | `merge`, `agg` vs. `transform`, outliers y escalamiento |
| 8 | La Sinergia de Datos en Profundidad | 37–39 | Módulo 8 | Un caso end-to-end combinando las dos librerías, y el mito del bucle `for` |
| 9 | Inspección Inicial y Pre-Entrega | 40–42 | Módulo 9 | `head`/`info`/`describe` + la consigna del Checkpoint |
| — | ¿Dudas? | 43 | — | Cierre y preguntas |

---

## Sobre el Dataset: `stocks.csv`

Antes de meternos con el código, vale la pena aclarar en clase qué es exactamente lo que estamos abriendo, porque a simple vista es "una tabla de números" y sin este contexto no dice nada.

**Estructura del archivo:**
- **71 filas**, una por mes, desde **enero de 2016** hasta **noviembre de 2021**.
- **Columna `formatted_date`**: el primer día de cada mes (frecuencia mensual, no diaria).
- **Las otras 14 columnas**: cada una es el **ticker** (código bursátil) de una empresa que cotiza en bolsa, y el valor numérico es el **precio de su acción ese mes, en dólares (USD)**.

**¿Qué tipo de "precio" es exactamente?** El archivo no lo aclara con un nombre de columna explícito (no dice "Close" ni "Adj Close"), pero por dos señales podemos inferir con bastante confianza que es el **precio de cierre ajustado (Adjusted Close)**:
1. La cantidad de decimales (`106.33214569091797`) es el patrón típico de datos descargados con la librería `yfinance` de Yahoo Finance, que reporta así el Adjusted Close.
2. El valor de MCD en enero de 2016 (~106) es más bajo que el precio de cierre "de pizarra" real de esa fecha (~118) — exactamente el efecto esperado del ajuste retroactivo por dividendos pagados, que es lo que distingue al Adjusted Close del precio de cierre nominal.

**Por qué importa usar el precio ajustado (para mencionar en clase)**: el precio de cierre nominal no refleja el dinero que efectivamente ganó o perdió un inversor, porque no cuenta los dividendos cobrados en el camino. El Adjusted Close sí, y por eso es el estándar en análisis financiero para comparar el rendimiento real de una acción a través del tiempo.

**Qué empresa es cada ticker:**

| Ticker | Empresa | Rubro |
|---|---|---|
| MCD | McDonald's | Consumo — restaurantes |
| SBUX | Starbucks | Consumo — restaurantes |
| GOOG | Alphabet (Google) | Tecnología |
| AMZN | Amazon | Consumo / Tecnología — e-commerce |
| MSFT | Microsoft | Tecnología |
| JPM | JPMorgan Chase | Finanzas — banca |
| BAC | Bank of America | Finanzas — banca |
| C | Citigroup | Finanzas — banca |
| MAR | Marriott | Turismo — hotelería |
| HLT | Hilton | Turismo — hotelería |
| RCL | Royal Caribbean | Turismo — cruceros |
| V | Visa | Finanzas — medios de pago |
| MA | Mastercard | Finanzas — medios de pago |
| PYPL | PayPal | Finanzas — pagos digitales |

**Fuente**: el archivo viene del repositorio público [`JJTorresDS/stocks-ds-edu`](https://raw.githubusercontent.com/JJTorresDS/stocks-ds-edu/main/stocks.csv) en GitHub, usado como dataset estándar de práctica en este curso.

---

## Bloque 0 — Repaso de la Clase Anterior (Fundamentos de Python)

**Contexto para abrir la clase**: antes de tocar NumPy, conviene una vuelta relámpago por los cuatro pilares de Python que ya vimos en la Clase 02 — **variables**, **condicionales**, **ciclo `for`** y **funciones** — reunidos en un solo ejemplo. No es contenido nuevo: es el punto de apoyo que vamos a necesitar en la próxima sección, cuando comparemos "resolver esto con un `for`" contra "resolverlo con NumPy en una línea".

### Variables (repaso)

Una variable es la etiqueta que le ponemos a un dato en memoria. Python decide el tipo automáticamente al momento de la asignación (**tipado dinámico**).

```python
producto = "Auriculares"
precio = 45.90
stock = 12
en_oferta = True

print(f"{producto} -> tipo: {type(producto)}")
print(f"{precio} -> tipo: {type(precio)}")
```

**Línea por línea:**
- `producto = "Auriculares"` → asigna un `str` (texto entre comillas).
- `precio = 45.90` → asigna un `float` (número con decimales).
- `stock = 12` → asigna un `int` (número entero).
- `en_oferta = True` → asigna un `bool` (solo puede ser `True` o `False`).
- Los dos `print(f"...")` arman un string con `f""` que inserta el valor de la variable y, con `type(...)`, el tipo que Python le asignó automáticamente.

**Y las listas también son variables**: hasta acá `producto`, `precio`, `stock` y `en_oferta` son **escalares** — un solo valor cada una. Python también tiene **colecciones**, que agrupan muchos escalares en una sola variable; la más simple es la **lista**. Definimos acá `precios_lista`, que vamos a reutilizar tal cual en el `for` y en la función de más abajo — así queda claro desde el principio que no es un dato nuevo que aparece de la nada.

```python
precios_lista = [15, 45, 120, 300, 80, 500, 20]
print(f"{precios_lista} -> tipo: {type(precios_lista)}")
print(f"Primer precio: {precios_lista[0]}")
print(f"Cantidad de precios: {len(precios_lista)}")
```

**Línea por línea:**
- `precios_lista = [...]` → crea una **lista**, una colección ordenada que agrupa 7 escalares (`int`) en una sola variable.
- `type(precios_lista)` → confirma que el tipo es `list`.
- `precios_lista[0]` → **indexación**: accede al elemento en la posición 0 (el primero), no al valor "0".
- `len(precios_lista)` → cuenta cuántos elementos tiene la lista (acá, 7).

### Condicionales (repaso)

`if` / `elif` / `else` bifurcan el camino de ejecución según una condición booleana. Para que se sienta "en vivo", pedimos el stock por teclado con `input()` — así cada vez que se corre la celda con un valor distinto, cambia el resultado. También combinamos condiciones con `and` / `or`, como una regla de negocio real: sin stock **siempre** es alerta; con poco stock, solo es alerta si el producto **además** está en oferta (mayor demanda esperada).

```python
stock = int(input("Ingresá el stock actual: "))

if stock == 0 or (stock < 5 and en_oferta):
    estado = "Atención: revisar reposición"
else:
    estado = "Stock ok"

print(f"{producto}: {estado}")
```

**Línea por línea:**
- `input("...")` → pausa la ejecución y pide un valor por teclado; siempre devuelve **texto**, por eso `int(...)` lo convierte a número.
- `if stock == 0 or (...)`: → se cumple si el stock es 0, **o** si se cumple lo de adentro del paréntesis. `or` alcanza con que una de las dos partes sea verdadera.
- `(stock < 5 and en_oferta)` → esta parte necesita **ambas** condiciones a la vez: poco stock **y** que esté en oferta.
- `else: estado = "Stock ok"` → se ejecuta si ninguna condición del `if` se cumplió.
- El `print` final muestra el producto junto con el resultado de la decisión.

### Ciclo for (repaso)

`for` recorre una colección elemento por elemento — en este caso, la misma `precios_lista` que ya definimos arriba. **Este es el protagonista del contraste que viene**: todo lo que hoy resolvemos con un `for` sobre una lista, en NumPy lo vamos a resolver sin ningún bucle explícito, operando sobre el array completo de una sola vez.

```python
for p in precios_lista:
    print(f"Precio: {p}")
```

**Línea por línea:**
- `for p in precios_lista:` → recorre `precios_lista` elemento por elemento; en cada vuelta, `p` toma el valor de un elemento distinto.
- `print(f"Precio: {p}")` → está **indentado dentro** del `for`, así que se ejecuta una vez por cada elemento de la lista (7 veces en total).

### Función que reúne todo

Una función combina variables, condicionales y bucles en un bloque reutilizable. El ejemplo de referencia — clasificar una lista de precios en categorías de negocio — es el mismo problema que vamos a retomar con NumPy en el próximo módulo, para poder comparar código y enfoque lado a lado.

```python
def clasificar_precios(precios):
    """Recorre una lista de precios y cuenta cuántos caen en cada categoría de negocio."""
    resumen = {"Económico": 0, "Medio": 0, "Premium": 0}

    for precio in precios:
        if precio < 50:
            categoria = "Económico"
        elif precio < 200:
            categoria = "Medio"
        else:
            categoria = "Premium"
        resumen[categoria] += 1

    return resumen


reporte = clasificar_precios(precios_lista)
print(reporte)
# {'Económico': 3, 'Medio': 2, 'Premium': 2}
```

**Línea por línea:**
- `def clasificar_precios(precios):` → define la función; `precios` es el **parámetro**, el molde que va a recibir la lista real.
- El docstring (`"""..."""`) documenta qué hace la función — buena práctica, no afecta la ejecución.
- `resumen = {...}` → un diccionario que arranca en 0 para cada categoría; es la variable donde vamos a ir contando.
- `for precio in precios:` → recorre cada precio de la lista que se le pasó a la función.
- `if / elif / else` → decide a qué categoría pertenece ese precio puntual, según su valor.
- `resumen[categoria] += 1` → suma 1 al contador de la categoría que correspondió (equivale a `resumen[categoria] = resumen[categoria] + 1`).
- `return resumen` → termina la función y entrega el diccionario completo, para poder guardarlo en una variable y reutilizarlo.
- `reporte = clasificar_precios(precios_lista)` → **llama** a la función pasándole `precios_lista` como **argumento** (el valor real que ocupa el lugar del parámetro `precios`).

**Por qué este bloque va primero**: cuando en la próxima sección resolvamos el mismo tipo de transformación con `precios_array * 1.21` (sin `for`, sin `if` explícito por elemento), el salto de un enfoque al otro se entiende mucho mejor si el enfoque "de toda la vida" está fresco y no es una abstracción lejana.

---

## Módulo 1 — Arrays Multidimensionales y Vectorización en NumPy

### ¿Qué es NumPy, y por qué es tan buena para la parte matemática?

**NumPy (Numerical Python) es una librería de Python** — no es parte del lenguaje "de fábrica": hay que instalarla (`pip install numpy`) e importarla en cada notebook con `import numpy as np`. Por dentro, la mayor parte de NumPy **no está escrita en Python sino en C**: cuando llamás a una función de NumPy, Python le pasa el trabajo pesado a rutinas compiladas, muchísimo más rápidas que un bucle interpretado línea por línea.

**Lo que la hace tan buena para matemática, específicamente:**

1. **Homogeneidad de tipo**: un array de NumPy exige que todos sus elementos sean del mismo tipo (todos `int64`, todos `float64`, etc.). Al saberlo de antemano, NumPy reserva un bloque de memoria **contiguo** y de tamaño exacto — nada de "adivinar" qué tipo es cada elemento en cada operación, como sí pasa con una lista de Python (que puede mezclar tipos).
2. **Vectorización**: gracias a esa memoria contigua y homogénea, una operación como `array * 2` no itera elemento por elemento en Python — le pide al procesador que aplique la multiplicación a todo el bloque de una sola vez. Resultado: operaciones hasta **100 veces más rápidas** que el equivalente con `for`.
3. **Sintaxis matemática nativa**: `matriz_a @ matriz_b`, `array ** 2`, `np.sqrt(array)` — se escribe álgebra casi como en el papel, sin tener que "abrir" la colección elemento por elemento.

**Por qué importa en Data Science**: NumPy es el cimiento silencioso de casi todo el ecosistema — **Pandas**, **Matplotlib** y **Scikit-Learn** están construidos sobre NumPy y lo usan para todos sus cálculos internos. Aprender NumPy no es un desvío, es la base.

### El ndarray: la regla de la homogeneidad

El objeto central de NumPy es el `ndarray` (n-dimensional array). Se parece a una lista de Python, pero con una regla de oro que una lista no tiene: **todos los elementos deben ser del mismo tipo**.

```python
import numpy as np

lista_python = [1, "dos", 3.0, True]      # una lista mezcla tipos sin problema
array_numpy = np.array([1, 2, 3.0, 4])    # NumPy homogeneiza: todo termina en float64

print(array_numpy.dtype)   # float64 -> NumPy convirtió los enteros a flotante
```

**Línea por línea:**
- `import numpy as np` → carga la librería NumPy y le da el alias `np` (convención universal, siempre se usa así).
- `lista_python = [1, "dos", 3.0, True]` → una lista de Python acepta un `int`, un `str`, un `float` y un `bool` sin quejarse.
- `np.array([1, 2, 3.0, 4])` → `np.array()` convierte esa lista en un `ndarray`; como hay un `3.0` (float) en el medio, NumPy "sube" **todos** los elementos a `float64` para mantenerlos homogéneos.
- `array_numpy.dtype` → el atributo que revela de qué tipo son, por dentro, los elementos del array.

### Dimensiones, forma y tipo: `ndim`, `shape`, `dtype`

| Concepto | Qué responde | Ejemplo |
|---|---|---|
| `ndim` | ¿Cuántos "ejes" tiene el array? | Un vector tiene `ndim = 1`, una matriz `ndim = 2` |
| `shape` | ¿Cuántos elementos hay en cada eje? | `(71, 14)` → 71 filas, 14 columnas |
| `dtype` | ¿De qué tipo son los elementos? | `float64`, `int32`, `bool` |

- **0D (Escalar)**: un solo número, ej. `5`.
- **1D (Vector)**: una fila de números, como una lista simple.
- **2D (Matriz)**: filas y columnas — la forma típica de una tabla de datos.
- **3D (Tensor)**: un "cubo" de datos — varias matrices apiladas (ej. una imagen a color: alto × ancho × canales de color).

### Creación de arrays

```python
import numpy as np

# Desde una lista de Python
mi_array = np.array([1, 2, 3, 4])

# Arrays "preparados" para guardar resultados
np.zeros((3, 4))     # matriz 3x4 llena de ceros
np.ones(5)            # [1. 1. 1. 1. 1.]

# Secuencias numéricas rápidas (como range(), pero devuelve un array)
np.arange(0, 10, 2)   # [0 2 4 6 8]
```

**Línea por línea:**
- `np.array([1, 2, 3, 4])` → toma una lista ya existente y la convierte en `ndarray`.
- `np.zeros((3, 4))` → la tupla `(3, 4)` indica **3 filas, 4 columnas**; crea esa matriz llena de ceros, lista para "rellenar" con resultados después.
- `np.ones(5)` → un solo número (sin tupla) crea un array 1D de 5 elementos, todos en `1`.
- `np.arange(0, 10, 2)` → genera una secuencia desde `0` hasta `10` **sin incluir el 10**, saltando de a `2`: `[0, 2, 4, 6, 8]`.

### Ejemplo real: cargando `stocks.csv` como matriz de NumPy

`stocks.csv` (en esta misma carpeta) tiene precios mensuales de 14 acciones (McDonald's, Starbucks, Google, Amazon, Microsoft, bancos, hoteles, tarjetas...) entre 2016 y 2021. Es un archivo de texto plano con una primera columna de **fechas** (texto) y 14 columnas de **precios** (números).

```python
import numpy as np

# Cargamos SOLO las columnas numéricas (excluimos la columna de fecha, que es texto)
# skip_header=1 salta la fila de encabezados; usecols=range(1, 15) toma las 14 columnas de precios
precios_matriz = np.genfromtxt("stocks.csv", delimiter=",", skip_header=1, usecols=range(1, 15))

print(f"Shape: {precios_matriz.shape}")   # (71, 14) -> 71 meses, 14 acciones
print(f"ndim:  {precios_matriz.ndim}")    # 2
print(f"dtype: {precios_matriz.dtype}")   # float64
```

**Línea por línea:**
- `np.genfromtxt("stocks.csv", ...)` → lee el archivo de texto y arma un `ndarray` numérico a partir de él.
- `delimiter=","` → indica que las columnas están separadas por comas (formato CSV estándar).
- `skip_header=1` → salta la primera fila del archivo (los nombres de columna), que son texto y no números.
- `usecols=range(1, 15)` → `range(1, 15)` genera los índices `1, 2, ..., 14`: toma solo esas 14 columnas de precios, dejando afuera la columna 0 (la fecha).
- `.shape`, `.ndim`, `.dtype` → los tres atributos que ya vimos: forma, cantidad de ejes y tipo de dato del array resultante.

**Un detalle importante para remarcar acá**: tuvimos que **excluir la columna de fechas** con `usecols`. Si intentáramos meterla en el array, NumPy — fiel a la regla de homogeneidad — convertiría **todos** los precios a texto para poder incluir las fechas, y perderíamos la capacidad de hacer cuentas. Esta limitación es exactamente el motivo por el que en el Módulo 5 vamos a introducir **Pandas**: necesitamos una herramienta que sí pueda mezclar fechas, texto y números en la misma tabla sin perder la parte matemática de NumPy por debajo.

### Reshape y operaciones elemento a elemento

`reshape` reorganiza un array sin tocar sus datos — el único requisito es que la cantidad total de elementos coincida.

```python
# Tomamos solo la columna de MSFT (columna índice 4) como vector 1D
msft = precios_matriz[:, 4]
print(msft.shape)          # (71,)

# La transformamos en una matriz de 71 filas x 1 columna
msft_columna = msft.reshape(71, 1)
print(msft_columna.shape)  # (71, 1)

# Operaciones elemento a elemento: ocurren "posición a posición", sin bucles
precio_en_pesos = msft * 1000          # simulando un tipo de cambio fijo
diferencia_mensual = msft[1:] - msft[:-1]   # variación mes a mes
```

**Línea por línea:**
- `precios_matriz[:, 4]` → `[:, 4]` se lee "todas las filas, columna con índice 4"; extrae la columna de MSFT como un vector 1D.
- `msft.reshape(71, 1)` → reorganiza esos 71 números en una matriz de 71 filas y 1 columna; `71 × 1 = 71` coincide con el total original, así que es válido.
- `msft * 1000` → multiplica **cada uno** de los 71 valores por 1000, sin escribir ningún bucle.
- `msft[1:] - msft[:-1]` → `msft[1:]` es "del segundo elemento en adelante"; `msft[:-1]` es "del primero al anteúltimo". Restar ambos alinea cada mes con el anterior y da la variación mes a mes.

### Vectorización frente a bucles: retomando el Bloque 0

En el Bloque 0 resolvimos "clasificar una lista de precios" con un `for` y un `if`. Ahora resolvamos un problema del mismo estilo — **aplicar un 21% de aumento a todos los precios de MSFT** — de las dos formas, para comparar:

```python
import time

msft_lista = list(msft)   # lo mismo, pero como lista de Python

# --- Enfoque tradicional: bucle for ---
inicio = time.time()
msft_aumentado_lista = []
for precio in msft_lista:
    msft_aumentado_lista.append(precio * 1.21)
tiempo_for = time.time() - inicio

# --- Enfoque NumPy: vectorizado ---
inicio = time.time()
msft_aumentado_array = msft * 1.21
tiempo_numpy = time.time() - inicio

print(f"Bucle for : {tiempo_for:.6f}s")
print(f"NumPy     : {tiempo_numpy:.6f}s")
```

**Línea por línea:**
- `list(msft)` → convierte el array de NumPy de vuelta a una lista de Python, para poder comparar "en igualdad de condiciones".
- `time.time()` antes y después de cada bloque → mide cuánto tiempo real pasó entre ambas llamadas; la resta da la duración.
- El bucle `for` crea una lista vacía y va agregando (`.append()`) cada precio ya aumentado, uno por uno.
- `msft * 1.21` → la misma transformación, pero aplicada a los 71 elementos de una sola vez.
- `:.6f` en el `print` → formatea el número con 6 decimales, para que se note la diferencia de tiempo aunque sea mínima.

Con 71 elementos la diferencia de tiempo es imperceptible — pero la diferencia de **código** ya es contundente: 4 líneas contra 1. Si `msft_lista` tuviera 10 millones de elementos (como el ejemplo de la tienda global del PDF), la diferencia de tiempo también se volvería contundente.

### Errores comunes y buenas prácticas

| Error | Qué pasa | Cómo evitarlo |
|---|---|---|
| Mezclar tipos | `np.array([1, 2, "3"])` convierte **todo** a texto | Verificá siempre `.dtype` antes de operar |
| Confundir `shape` con `len()` | En una matriz `(71, 14)`, `len()` solo devuelve `71` (filas) | Para la estructura completa, usá siempre `.shape` |
| `reshape` incompatible | `precios_matriz.reshape(5, 5)` falla si no hay exactamente 25 elementos | El producto de las nuevas dimensiones debe igualar el total de elementos originales |
| Bucles innecesarios | Escribir un `for` sobre un array para algo que ya tiene función vectorizada | Antes de iterar, preguntate: "¿esto ya existe en NumPy?" |

---

## Módulo 2 — Broadcasting y Operaciones sobre Matrices

**Contexto para abrir el módulo**: en el Módulo 1 vimos operaciones entre arrays del mismo tamaño. En el mundo real casi nunca es así — rara vez trabajamos con arreglos de tamaño exacto. Acá entra el verdadero "superpoder" de NumPy: el Broadcasting.

### La Matriz de Datos: filas, columnas e indexación

En Data Science, la información casi siempre se organiza de forma tabular — una **Matriz de Datos**:

- **Filas (eje 0)**: cada observación. En `precios_matriz` (Módulo 1), cada fila es **un mes**.
- **Columnas (eje 1)**: cada variable. Cada columna es **una acción** (MCD, SBUX, GOOG...).
- **Indexación base 0**: `precios_matriz[2, 1]` es la **tercera fila, segunda columna** → el precio de SBUX en el tercer mes registrado.

```python
print(precios_matriz[2, 1])   # precio de SBUX (columna 1) en el mes con índice 2
```

**Línea por línea:**
- `precios_matriz[2, 1]` → indexación 2D: el primer número (`2`) es la fila, el segundo (`1`) es la columna. Devuelve un único valor: el precio de SBUX en el tercer mes registrado (índice 2, porque se cuenta desde 0).

### Broadcasting: el superpoder de NumPy

El **Broadcasting** es el conjunto de reglas que le permite a NumPy operar entre arrays de **distinta forma**, sin escribir un bucle y sin copiar datos de más en memoria. NumPy compara las formas **de derecha a izquierda**: dos dimensiones son compatibles si son iguales o si una de ellas es `1`.

**Caso real: centrar los precios restando la media de cada acción.** `precios_matriz` tiene shape `(71, 14)`; el vector de medias por columna tiene shape `(14,)`. NumPy "estira" el vector de 14 medias para cubrir las 71 filas, sin crear 71 copias del vector.

```python
medias_por_accion = precios_matriz.mean(axis=0)     # shape (14,) -> una media por columna
print(f"Shape de las medias: {medias_por_accion.shape}")

precios_centrados = precios_matriz - medias_por_accion   # broadcasting: (71,14) - (14,) -> (71,14)
print(precios_centrados[0])   # cuánto se desvía cada acción de su propia media, en el primer mes
```

**Línea por línea:**
- `precios_matriz.mean(axis=0)` → `axis=0` significa "promediar recorriendo las filas", es decir, calcular un promedio **por columna**; el resultado es un vector de 14 medias, una por acción.
- `precios_matriz - medias_por_accion` → resta un vector `(14,)` a una matriz `(71,14)`. Por broadcasting, NumPy "estira" el vector para restarlo a cada una de las 71 filas, sin copiarlo 71 veces en memoria.
- `precios_centrados[0]` → la primera fila del resultado: cuánto se aleja cada acción de su propio promedio, en el primer mes del dataset.

### Multiplicación elemento a elemento (`*`) vs. Transposición (`.T`)

- **`*` (elemento a elemento)**: multiplica posición a posición — (0,0) con (0,0), (0,1) con (0,1)... Debe cumplir las reglas de broadcasting, **no** es álgebra lineal.
- **`.T` (transposición)**: gira la matriz — filas pasan a ser columnas. `precios_matriz` es `(71, 14)`; `precios_matriz.T` es `(14, 71)`. No copia datos, solo cambia cómo se *leen* — es prácticamente gratis en rendimiento, y es la herramienta clave para alinear dimensiones antes de una multiplicación matricial.

```python
volatilidad = precios_matriz.std(axis=0)           # desvío estándar por acción, shape (14,)
precios_normalizados = precios_centrados / volatilidad   # * y / también son elemento a elemento

print(precios_matriz.T.shape)   # (14, 71) -> ahora cada fila es una acción, cada columna un mes
```

**Línea por línea:**
- `precios_matriz.std(axis=0)` → desvío estándar por columna (por acción), igual lógica que `.mean(axis=0)` antes.
- `precios_centrados / volatilidad` → divide cada columna de `precios_centrados` por su propio desvío; es otra operación elemento a elemento con broadcasting, no álgebra lineal.
- `precios_matriz.T` → `.T` transpone la matriz: intercambia filas por columnas. No mueve datos en memoria, solo cambia cómo se los "lee", así que es prácticamente instantáneo.

### Multiplicación Matricial (`@`): un vector de pesos de portafolio

A diferencia de `*`, el operador `@` (o `np.dot`) sigue la regla clásica del álgebra lineal: para multiplicar una matriz `A` por una matriz `B`, **el número de columnas de A debe igualar el número de filas de B**.

**Caso real**: si armamos un portafolio con un peso (proporción invertida) por cada una de las 14 acciones, `precios_matriz @ pesos` nos da el **valor del portafolio en cada uno de los 71 meses**, en una sola operación.

```python
# Un peso igual para las 14 acciones (suman 1 entre todas)
pesos = np.ones(14) / 14
print(f"Shape de pesos: {pesos.shape}")             # (14,)

valor_portafolio = precios_matriz @ pesos            # (71,14) @ (14,) -> (71,)
print(f"Shape del resultado: {valor_portafolio.shape}")
print(valor_portafolio[:5])   # valor del portafolio en los primeros 5 meses
```

**Línea por línea:**
- `np.ones(14) / 14` → crea un array de 14 unos y divide **cada uno** por 14: da 14 pesos iguales, cada uno de aproximadamente `0.0714`, que suman 1 entre todos.
- `precios_matriz @ pesos` → multiplicación matricial: por cada mes (fila), multiplica cada precio por su peso y suma los 14 resultados — el mismo cálculo que un producto punto, repetido para las 71 filas a la vez. `(71,14) @ (14,)` da como resultado un vector `(71,)`.
- `valor_portafolio[:5]` → los primeros 5 valores del vector resultante, uno por cada uno de los primeros 5 meses.

### Errores comunes de broadcasting

```python
vector_mal_dimensionado = np.array([1, 2, 3])   # shape (3,), pero precios_matriz tiene 14 columnas

# precios_matriz - vector_mal_dimensionado
# ValueError: operands could not be broadcast together with shapes (71,14) (3,)
```

**Línea por línea:**
- `np.array([1, 2, 3])` → un vector de shape `(3,)`. Comparado de derecha a izquierda contra `(71, 14)`, el `3` no coincide con el `14` ni es `1`, así que la regla de broadcasting no se cumple.
- La resta comentada es justamente la que **fallaría**: la dejamos comentada para mostrar el error sin frenar la ejecución del resto del notebook.

| Error | Causa | Solución |
|---|---|---|
| `could not be broadcast together` | Las formas no son iguales ni una de ellas es 1 (comparando de derecha a izquierda) | Verificar `.shape` de ambos arrays antes de operar |
| `shapes not aligned` (con `@`) | Las columnas de A no igualan las filas de B | Usar `.T` para transponer el que corresponda: `A @ B.T` |
| Confundir `*` con `@` | `*` es elemento a elemento; `@` combina filas y columnas (álgebra lineal) | Preguntarse: "¿quiero ajustar cada valor, o combinar variables?" |

---

## Módulo 3 — Álgebra Lineal con NumPy

**Contexto para abrir el módulo**: en Ciencia de Datos, los algoritmos rara vez procesan datos de forma aislada. Este módulo es el puente directo hacia la Regresión Lineal y las Redes Neuronales, que por dentro no son otra cosa que multiplicaciones matriciales y sistemas de ecuaciones resueltos a alta velocidad.

### Producto punto (Dot Product)

El producto punto toma dos vectores de la **misma longitud** y devuelve un único **escalar**: multiplica los elementos correspondientes y suma los resultados.

**Caso real**: si tenemos el retorno promedio de cada acción y los pesos del portafolio del Módulo 2, el producto punto entre ambos vectores nos da el **retorno esperado del portafolio completo**, en una sola cuenta.

```python
retornos_mensuales = (precios_matriz[1:] - precios_matriz[:-1]) / precios_matriz[:-1]  # % de cambio mes a mes
retorno_promedio_por_accion = retornos_mensuales.mean(axis=0)   # shape (14,)

retorno_esperado_portafolio = np.dot(retorno_promedio_por_accion, pesos)   # escalar
print(f"Retorno mensual esperado del portafolio: {retorno_esperado_portafolio:.4%}")
```

**Línea por línea:**
- `(precios_matriz[1:] - precios_matriz[:-1]) / precios_matriz[:-1]` → la fórmula del % de cambio: `(precio actual − precio anterior) / precio anterior`, calculada para los 71 meses y las 14 acciones a la vez.
- `retornos_mensuales.mean(axis=0)` → promedio por columna: el retorno promedio histórico de cada una de las 14 acciones.
- `np.dot(retorno_promedio_por_accion, pesos)` → producto punto: multiplica cada retorno promedio por su peso correspondiente y suma los 14 resultados en un único número (el retorno esperado del portafolio completo).
- `:.4%` → formatea el número como porcentaje con 4 decimales.

En NumPy moderno se prefiere el operador infijo `@` sobre `np.dot()` por legibilidad — para vectores 1D ambos hacen exactamente lo mismo.

### Sistemas de Ecuaciones Lineales: `np.linalg.solve`

Muchos problemas se reducen a resolver $Ax = b$: $A$ son los coeficientes conocidos, $b$ los términos independientes, $x$ las incógnitas que buscamos.

**Calcular la inversa de A ($x = A^{-1}b$) es ineficiente y propenso a errores de redondeo.** NumPy usa descomposición LU internamente con `np.linalg.solve`, mucho más rápida y estable — y solo funciona si $A$ es cuadrada y **no singular** (determinante distinto de cero).

**Caso real**: queremos comprar una combinación de acciones de MCD y SBUX tal que el total sean **100 acciones** y el valor total sea de **10.000 USD**, usando los precios reales del primer mes del dataset.

```python
precio_mcd = precios_matriz[0, 0]    # MCD, mes 0
precio_sbux = precios_matriz[0, 1]   # SBUX, mes 0

# x + y = 100                 (cantidad total de acciones)
# precio_mcd*x + precio_sbux*y = 10000   (valor total en USD)
A = np.array([[1, 1],
              [precio_mcd, precio_sbux]])
b = np.array([100, 10000])

x_mcd, y_sbux = np.linalg.solve(A, b)
print(f"Acciones de MCD:  {x_mcd:.1f}")
print(f"Acciones de SBUX: {y_sbux:.1f}")
```

**Línea por línea:**
- `precios_matriz[0, 0]` y `[0, 1]` → los precios de MCD y SBUX en el mes 0 (el primero del dataset).
- `A = np.array([[1, 1], [precio_mcd, precio_sbux]])` → la matriz de coeficientes: la primera fila representa la ecuación "cantidad total = 100"; la segunda, "valor total = 10.000".
- `b = np.array([100, 10000])` → el vector de resultados de cada ecuación, en el mismo orden que las filas de `A`.
- `np.linalg.solve(A, b)` → resuelve el sistema $Ax = b$ y devuelve un array con las dos incógnitas; lo "desempaquetamos" directo en `x_mcd, y_sbux`.
- `:.1f` → redondea el resultado a 1 decimal para mostrarlo más prolijo.

### Diagnóstico y Propiedades Matriciales (`np.linalg`)

| Función | Propiedad matemática | Utilidad en Data Science |
|---|---|---|
| `np.linalg.det(A)` | Determinante: si es 0, la matriz es singular (no invertible) | Diagnóstico de colinealidad entre variables |
| `np.linalg.norm(v)` | Norma: magnitud geométrica de un vector | Distancias, similitud coseno, algoritmos como KNN |
| `np.linalg.eig(A)` | Eigenvalues / eigenvectors: direcciones que solo se escalan, no rotan | Reducción de dimensionalidad (PCA) |

**Caso real**: la matriz de correlación entre las 14 acciones es cuadrada (14×14) — perfecta para diagnosticarla con `np.linalg`.

```python
correlaciones = np.corrcoef(precios_matriz.T)   # matriz 14x14: correlación entre cada par de acciones
print(f"Shape: {correlaciones.shape}")

determinante = np.linalg.det(correlaciones)
print(f"Determinante: {determinante:.6f}")   # muy cercano a 0 -> hay acciones fuertemente correlacionadas

valores_propios, vectores_propios = np.linalg.eig(correlaciones)
print(f"Eigenvalues: {valores_propios.round(2)}")
```

**Línea por línea:**
- `np.corrcoef(precios_matriz.T)` → calcula la correlación entre cada par de columnas; se usa `.T` porque `corrcoef` espera una **fila por variable**, no por observación como está `precios_matriz` originalmente.
- `correlaciones.shape` → confirma que el resultado es una matriz cuadrada `(14, 14)`, una fila y columna por cada acción.
- `np.linalg.det(correlaciones)` → calcula el determinante de esa matriz cuadrada.
- `np.linalg.eig(correlaciones)` → devuelve **dos** resultados a la vez (eigenvalues y eigenvectors), que se desempaquetan en dos variables.
- `.round(2)` → redondea cada valor del array a 2 decimales antes de mostrarlo.

**Por qué esto importa para lo que viene**: la ecuación normal de la Regresión Lineal ($\beta = (X^TX)^{-1}X^Ty$) y el Análisis de Componentes Principales (PCA) se apoyan exactamente en estas operaciones — transposición, multiplicación matricial, sistemas lineales y eigendecomposition — todas resueltas por NumPy en milisegundos.

---

## Módulo 4 — NumPy y Pandas: la Relación entre el Cálculo y la Estructura de Datos

### Dos librerías de Python, dos especialidades

**Tan importante como saber usar cada una es tener clarísimo que ambas son librerías de Python** — ninguna viene instalada por defecto, ambas se instalan (`pip install numpy pandas`) y se importan explícitamente. No son "modos" distintos de Python: son herramientas externas, cada una diseñada para resolver un problema distinto.

- **NumPy** existe para la **parte matemática**: números homogéneos, memoria contigua, cálculos vectorizados a velocidad de C. Ya lo vimos en los Módulos 1 a 3.
- **Pandas** existe para la **parte de tablas**: datos del mundo real con nombres de columna, tipos mixtos (texto + números + fechas en la misma tabla) y valores faltantes. Es la pieza que nos faltaba desde el Módulo 1, cuando tuvimos que **excluir la columna de fechas** de `stocks.csv` para poder cargarla en un `ndarray`.

**¿Por qué Pandas es tan buena para la parte de tablas, específicamente?**

1. **Etiquetas en vez de posiciones**: en NumPy, si querés la columna de precios tenés que acordarte que es la columna índice `4`. En Pandas simplemente pedís `df["MSFT"]`.
2. **Tipos heterogéneos por columna**: cada columna de un DataFrame puede tener su propio `dtype` — una de fechas, otras de texto, otras de números — todas conviviendo en la misma tabla sin forzar una conversión global (algo que a NumPy, por su regla de homogeneidad, le resulta imposible).
3. **Datos faltantes de primera clase**: Pandas tiene herramientas integradas (`isnull()`, `dropna()`, `fillna()`) para detectar y tratar esos huecos que **siempre** aparecen en datos reales — algo que en NumPy hay que resolver "a mano".
4. **Ingesta de archivos reales**: `pd.read_csv()`, `pd.read_excel()`, `pd.read_json()`... Pandas lee directamente los formatos que se usan en la industria; NumPy, en el mejor de los casos, solo lee bloques numéricos puros (como vimos con `np.genfromtxt` en el Módulo 1).

**El punto clave que conecta ambas**: Pandas **está construido sobre NumPy**. Por dentro, cada columna de un DataFrame es, literalmente, un array de NumPy con una etiqueta pegada encima. Cuando Pandas suma dos columnas, le pide el cálculo a NumPy y después le vuelve a poner las etiquetas al resultado.

### Comparativa: ¿cuándo usar cuál?

| Característica | NumPy | Pandas |
|---|---|---|
| Estructura principal | `ndarray` (matriz) | `DataFrame` (tabla) |
| Tipo de datos | Homogéneo (todo números) | Heterogéneo (mezcla de tipos por columna) |
| Identificación | Solo por posición (`[2, 1]`) | Por posición **y** por nombre (`df["MSFT"]`) |
| Uso ideal | Álgebra lineal, cálculos pesados | Leer archivos, limpiar datos, EDA |
| Celdas vacías | Difíciles de manejar | Diseñado para detectarlas y tratarlas |

> **"NumPy trabaja con números; Pandas trabaja con datos."** En el 90% de los proyectos reales usamos las dos: Pandas para traer y ordenar la información, NumPy (por debajo, casi siempre sin que lo notemos) para hacer las cuentas.

### Demostración: la misma tabla, dos herramientas

```python
import numpy as np
import pandas as pd

# Con NumPy (Módulo 1): tuvimos que EXCLUIR la columna de fecha
precios_matriz = np.genfromtxt("stocks.csv", delimiter=",", skip_header=1, usecols=range(1, 15))
print(type(precios_matriz))   # <class 'numpy.ndarray'>

# Con Pandas: la tabla completa, fechas incluidas, sin perder nada
df_stocks = pd.read_csv("stocks.csv")
print(df_stocks.dtypes.head())   # formatted_date: object, MCD: float64, SBUX: float64...

# La prueba de que Pandas está construido sobre NumPy:
columna_msft = df_stocks["MSFT"]
print(type(columna_msft))          # <class 'pandas.core.series.Series'>
print(type(columna_msft.values))   # <class 'numpy.ndarray'> -> ¡por dentro, sigue siendo un array!
```

**Línea por línea:**
- `np.genfromtxt(...)` → la misma carga del Módulo 1: solo columnas numéricas, sin la fecha.
- `pd.read_csv("stocks.csv")` → carga la tabla **completa**, dejando que Pandas maneje la mezcla de texto (fecha) y números sin quejarse.
- `df_stocks.dtypes.head()` → `.dtypes` lista el tipo de cada columna; `.head()` muestra solo las primeras (para no imprimir las 15).
- `df_stocks["MSFT"]` → seleccionar una columna con corchete simple devuelve una **Serie**.
- `columna_msft.values` → extrae los datos "en crudo" de la Serie, sin la etiqueta del índice — y ese resultado es, literalmente, un `numpy.ndarray`.

---

## Módulo 5 — Introducción a Pandas: Series y DataFrames

**Contexto para abrir el módulo**: si trabajar con listas y diccionarios se siente como organizar una biblioteca sin estantes, Pandas es el sistema de estanterías profesional que estábamos esperando. Acá vemos cómo organiza la información en sus dos estructuras fundamentales.

### La Serie: el bloque de construcción unidimensional

Una **Serie** es como una lista de Python, pero "con esteroides": además de los **valores**, tiene un **índice** — una etiqueta para cada valor. Si no le asignás uno, Pandas pone `0, 1, 2...` por defecto.

```python
precios_msft = df_stocks["MSFT"]        # esto ya es una Serie
print(type(precios_msft))                # pandas.core.series.Series
print(precios_msft.index[:5])            # RangeIndex: 0, 1, 2, 3, 4... (por defecto)

# Una Serie creada "a mano", con índice con nombre propio (no numérico)
stock_por_sector = pd.Series(
    {"Tecnología": 3, "Consumo": 4, "Finanzas": 3, "Turismo": 3},
)
print(stock_por_sector["Consumo"])        # acceso directo por etiqueta, sin saber la posición
```

**Línea por línea:**
- `df_stocks["MSFT"]` → seleccionar una columna de un DataFrame con `[]` devuelve una Serie.
- `precios_msft.index[:5]` → los primeros 5 valores del índice; como no le asignamos ninguno explícito, Pandas puso números correlativos (`RangeIndex`).
- `pd.Series({...})` → cuando se crea una Serie a partir de un diccionario, las **llaves** se convierten automáticamente en el índice y los **valores**, en los datos.
- `stock_por_sector["Consumo"]` → accede al valor por su etiqueta (`"Consumo"`), sin necesidad de saber en qué posición numérica está.

### El DataFrame: la tabla bidimensional

El **DataFrame** es la hoja de cálculo completa — la estructura más usada en Ciencia de Datos. Se puede pensar como un **diccionario de Series** que comparten el mismo índice de filas.

**Componentes clave:**
- **Datos**: la información en celdas.
- **Índice de filas**: identifica cada registro (por defecto, `0, 1, 2...`, pero puede ser cualquier otra cosa — como una fecha).
- **Columnas**: los nombres de cada variable.

Ahora cargamos `stocks.csv` "bien hecho": convertimos la columna de fechas a tipo fecha real (`parse_dates`) y la usamos como **índice** de filas (`set_index`), en vez de dejarla como una columna de texto más.

```python
df_stocks = pd.read_csv("stocks.csv", parse_dates=["formatted_date"])
df_stocks = df_stocks.set_index("formatted_date")

print(df_stocks.index[:3])     # DatetimeIndex: ahora las fechas son el índice, no una columna
print(df_stocks.columns)        # Index(['MCD', 'SBUX', 'GOOG', ...], dtype='object')
print(df_stocks.shape)          # (71, 14) -> mismo tamaño que precios_matriz, pero con etiquetas
```

**Línea por línea:**
- `parse_dates=["formatted_date"]` → le dice a `read_csv` que convierta esa columna a tipo fecha real (`datetime64`) durante la carga, en vez de dejarla como texto.
- `df_stocks.set_index("formatted_date")` → mueve la columna de fechas para que pase a ser el **índice** de filas; por eso hay que reasignar el resultado a `df_stocks`.
- `df_stocks.index[:3]` → las primeras 3 etiquetas del nuevo índice (ahora son fechas, no números).
- `df_stocks.columns` → los nombres de las 14 columnas que quedaron (ya sin la fecha, que pasó a ser índice).
- `df_stocks.shape` → sigue siendo `(71, 14)`: mismas dimensiones que `precios_matriz`, pero acá con nombres y fechas en vez de solo números.

### Creación de Series y DataFrames desde listas y diccionarios

No siempre partimos de un archivo — a veces armamos estas estructuras a mano, desde objetos que ya conocemos.

```python
# Desde una lista: la forma más simple, para una Serie
precios_ejemplo = pd.Series([15.0, 45.0, 120.0], name="Precio")

# Desde un diccionario: el método más común para un DataFrame
# (las llaves se convierten naturalmente en nombres de columna)
sectores_dict = {
    "ticker": ["MCD", "SBUX", "GOOG", "AMZN", "MSFT", "JPM", "BAC"],
    "sector": ["Consumo", "Consumo", "Tecnología", "Consumo", "Tecnología", "Finanzas", "Finanzas"],
}
df_sectores = pd.DataFrame(sectores_dict)
print(df_sectores)
```

**Línea por línea:**
- `pd.Series([15.0, 45.0, 120.0], name="Precio")` → una Serie desde una lista simple; `name=` le pone nombre a la Serie completa (útil si después se convierte en columna de un DataFrame).
- `sectores_dict = {...}` → un diccionario donde cada llave (`"ticker"`, `"sector"`) va a convertirse en el nombre de una columna, y cada lista, en los valores de esa columna.
- `pd.DataFrame(sectores_dict)` → arma la tabla; Pandas alinea automáticamente los elementos por posición dentro de cada lista (el primer ticker con el primer sector, y así sucesivamente).

Vamos a reutilizar `df_sectores` en el Módulo 7 para mostrar cómo **combinar** (`merge`) dos tablas relacionadas.

### Errores comunes: dimensión, índice y tipos

| Confusión | Aclaración |
|---|---|
| `df["MSFT"]` vs. `df[["MSFT"]]` | `df["MSFT"]` devuelve una **Serie**; `df[["MSFT", "AAPL"]]` (con doble corchete) devuelve un **DataFrame**, aunque sea de una sola columna. |
| "El índice siempre es un número" | Falso — acá el índice de `df_stocks` son fechas. Un índice con sentido de negocio facilita muchísimo la búsqueda de datos puntuales. |
| "Si mezclo texto y número, Pandas se rompe" | No se rompe, pero si una columna numérica trae **un solo** valor de texto (ej. `"N/D"`), Pandas convierte **toda la columna** a `object` y no vas a poder calcular hasta limpiarla. |

---

## Módulo 6 — Preprocesamiento de Datos

### El fenómeno de los datos ausentes

En entornos productivos, los datos faltantes (`NaN` — *Not a Number*) son una constante: fallas de sensores, pipelines interrumpidos, registros incompletos. `stocks.csv` viene limpio, así que para practicar vamos a **simular** el problema sobre una copia — algo muy común al enseñar o probar un pipeline de limpieza.

```python
df_sucio = df_stocks.copy()

# Introducimos NaN a propósito para simular datos reales incompletos
df_sucio.iloc[3, 2] = np.nan     # un NaN en la fila 3
df_sucio.iloc[7, 0] = np.nan     # primer NaN en la fila 7
df_sucio.iloc[7, 5] = np.nan     # segundo NaN en la MISMA fila 7
df_sucio.iloc[20, 1] = np.nan    # un NaN en la fila 20

print(df_sucio.isnull().sum())          # conteo de NaN por columna
print(f"Total de NaN: {df_sucio.isnull().sum().sum()}")
```

**Línea por línea:**
- `df_stocks.copy()` → crea una copia **independiente**; sin `.copy()`, `df_sucio` apuntaría a la misma tabla que `df_stocks` y modificar una modificaría la otra.
- `df_sucio.iloc[3, 2] = np.nan` → `iloc[fila, columna]` accede por **posición numérica**; le asigna `NaN` a esa celda puntual.
- `df_sucio.isnull()` → devuelve una tabla del mismo tamaño, pero con `True`/`False` según si cada celda es nula.
- `.sum()` → sobre esa tabla booleana, suma por columna (`True` vale 1, `False` vale 0): da el conteo de nulos por columna.
- `.sum().sum()` → el segundo `.sum()` suma esos conteos de columna en un solo número: el total de nulos del DataFrame.

### Eliminación vs. Imputación

| Estrategia | Ventaja | Riesgo |
|---|---|---|
| **Eliminación por filas** | No inventa datos; mantiene la certeza de los registros restantes | Reduce la muestra; puede sesgarla si los datos no faltan al azar |
| **Imputación por la media** | Preserva el tamaño de la muestra; computacionalmente eficiente | Subestima la varianza original y altera correlaciones con otras variables |

### El algoritmo de decisión y la Regla de Oro

Regla aplicada fila por fila: si tiene **más de 1** NaN, se elimina el registro completo; si tiene **exactamente 1**, se imputa con la media de esa columna.

**Regla de oro (para evitar Data Leakage)**: las medias de cada columna se calculan **antes** de eliminar ninguna fila, usando todos los valores presentes. Si calculáramos la media después de eliminar filas, estaríamos alterando la distribución original de la variable e invalidando el valor imputado.

```python
# 1) Medias ANTES de eliminar nada (regla de oro)
medias_columnas = df_sucio.mean(numeric_only=True)

# 2) Clasificamos cada fila según su cantidad de NaN
nan_por_fila = df_sucio.isnull().sum(axis=1)
filas_a_eliminar = nan_por_fila[nan_por_fila > 1].index
filas_a_imputar = nan_por_fila[nan_por_fila == 1].index

# 3) Aplicamos la regla
df_limpio = df_sucio.drop(index=filas_a_eliminar)
df_limpio = df_limpio.fillna(medias_columnas)

print(f"Filas eliminadas (>1 NaN): {len(filas_a_eliminar)}")
print(f"Filas imputadas (==1 NaN): {len(filas_a_imputar)}")
print(f"NaN restantes: {df_limpio.isnull().sum().sum()}")   # 0
```

**Línea por línea:**
- `df_sucio.mean(numeric_only=True)` → calcula el promedio de cada columna numérica; `numeric_only=True` evita que Pandas intente promediar columnas de texto.
- `df_sucio.isnull().sum(axis=1)` → acá `axis=1` cambia el sentido: suma **a lo largo de las columnas**, es decir, cuenta cuántos `NaN` tiene cada **fila**.
- `nan_por_fila[nan_por_fila > 1]` → filtro booleano: se queda solo con las filas cuyo conteo de nulos es mayor a 1; `.index` extrae las etiquetas de esas filas.
- `df_sucio.drop(index=filas_a_eliminar)` → elimina del DataFrame las filas cuyas etiquetas están en esa lista.
- `df_limpio.fillna(medias_columnas)` → rellena cualquier `NaN` restante con la media de su propia columna (calculada en el primer paso, antes de eliminar nada).
- `len(filas_a_eliminar)` → cuenta cuántas etiquetas quedaron en ese índice filtrado.

---

## Módulo 7 — Integración, Agregación y Preprocesamiento Avanzado

**Contexto para abrir el módulo**: en la práctica profesional, la información casi nunca vive en una sola tabla prolija — está distribuida en varias fuentes relacionadas. Este módulo cubre cómo combinarlas sin romper la integridad de los datos, y cómo resumirlas en métricas de negocio.

### Combinar tablas: claves y tipos de join

La información real casi nunca vive en una sola tabla. `stocks.csv` tiene precios, pero no sectores — eso está en `df_sectores` (Módulo 5). Para combinarlas, primero convertimos `df_stocks` de formato **ancho** (una columna por ticker) a formato **largo** (una fila por combinación fecha-ticker) con `melt`, y después unimos con `merge` usando `ticker` como clave.

```python
df_largo = df_stocks.reset_index().melt(
    id_vars="formatted_date", var_name="ticker", value_name="precio"
)
print(df_largo.head())   # formatted_date | ticker | precio

df_con_sector = pd.merge(
    left=df_largo,
    right=df_sectores,
    on="ticker",
    how="left",             # conservamos TODOS los precios, tengan sector o no
    validate="many_to_one",  # error si "ticker" estuviera duplicado en df_sectores
    indicator=True,
)
print(df_con_sector["_merge"].value_counts())
```

**Línea por línea:**
- `df_stocks.reset_index()` → vuelve a poner la fecha como columna normal (se lo pedimos porque `melt` necesita que las columnas a "derretir" no sean el índice).
- `.melt(id_vars="formatted_date", var_name="ticker", value_name="precio")` → `id_vars` es la columna que se mantiene fija; el resto de las columnas (los 14 tickers) se "apilan" en dos columnas nuevas: `ticker` (con el nombre de cada columna original) y `precio` (con su valor).
- `pd.merge(left=..., right=..., on="ticker", how="left", ...)` → une `df_largo` con `df_sectores` usando `"ticker"` como clave común; `how="left"` conserva todas las filas de `df_largo` aunque no tengan sector.
- `validate="many_to_one"` → verificación de seguridad: lanza un error si `"ticker"` estuviera duplicado en `df_sectores` (evita un producto cartesiano accidental).
- `indicator=True` → agrega la columna `_merge`, que etiqueta cada fila como `both`, `left_only` o `right_only`.
- `.value_counts()` → cuenta cuántas filas cayeron en cada una de esas categorías.

`df_sectores` solo tiene 7 de los 14 tickers — por eso el `left join` deja `NaN` en `sector` para el resto, y la columna `_merge` (gracias a `indicator=True`) nos deja **auditar** exactamente cuáles.

| Tipo de Join | Comportamiento |
|---|---|
| **Inner** | Solo las filas cuya clave está en ambas tablas (intersección) |
| **Left** | Todas las filas de la izquierda; sin coincidencia, `NaN` |
| **Right** | Simétrico al Left; se prefiere reordenar y usar Left |
| **Outer** | Todas las filas de ambas tablas (unión), con `NaN` donde falte |

### Split-Apply-Combine: `agg()` vs. `transform()`

`groupby` divide el DataFrame en subgrupos, aplica una función y combina los resultados.

- **`.agg()`**: **reduce** — un grupo de 100 filas se convierte en 1 fila con el estadístico.
- **`.transform()`**: **preserva la forma** — devuelve un valor por cada fila original, proyectando el resultado del grupo hacia atrás. Es la base de la normalización intragrupo.

```python
df_con_sector = df_con_sector.dropna(subset=["sector"])   # nos quedamos con las que sí tienen sector

# agg(): reduce a un resumen por sector
resumen_sector = df_con_sector.groupby("sector")["precio"].agg(
    precio_promedio="mean", precio_max="max", n_registros="count"
)
print(resumen_sector)

# transform(): Z-score, pero DENTRO de cada sector (no global)
df_con_sector["precio_z_sector"] = df_con_sector.groupby("sector")["precio"].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

**Línea por línea:**
- `dropna(subset=["sector"])` → `subset=["sector"]` hace que solo se eliminen filas con `NaN` **en esa columna puntual**, no en cualquier otra.
- `df_con_sector.groupby("sector")["precio"]` → agrupa las filas por sector y selecciona la columna `precio` dentro de cada grupo.
- `.agg(precio_promedio="mean", precio_max="max", n_registros="count")` → cada `nombre="función"` calcula un estadístico distinto y lo guarda en una columna con ese nombre — tres preguntas de negocio resueltas en una sola línea.
- `.transform(lambda x: (x - x.mean()) / x.std())` → a diferencia de `.agg()`, `transform` devuelve **un valor por cada fila original** (no una fila por grupo); acá, `x` es la Serie de precios de un solo sector, y la fórmula calcula el z-score de cada precio **dentro de su propio sector**.

### Outliers y escalamiento: Winsorización, Z-Score y Robust Scaling

En vez de eliminar valores extremos (perdiendo información), la **Winsorización** les pone un tope en un percentil bajo y otro alto — típicamente P1 y P99.

```python
p1, p99 = np.percentile(df_con_sector["precio"], [1, 99])
precio_winsorizado = np.clip(df_con_sector["precio"], p1, p99)   # los extremos se "achatan" al tope
```

**Línea por línea:**
- `np.percentile(df_con_sector["precio"], [1, 99])` → calcula, en una sola llamada, los valores que dejan el 1% y el 99% de los datos por debajo; se desempaquetan en `p1` y `p99`.
- `np.clip(serie, p1, p99)` → cualquier valor menor a `p1` se sube a `p1`, y cualquier valor mayor a `p99` se baja a `p99` — el resto de los valores queda intacto.

Para llevar variables a la misma escala (imprescindible en KNN, regresiones regularizadas o redes neuronales):

```python
# Z-Score: sensible a outliers (media y desvío se distorsionan con valores extremos)
z_score = (df_con_sector["precio"] - df_con_sector["precio"].mean()) / df_con_sector["precio"].std()

# Robust Scaling: usa mediana e IQR, inmune a outliers extremos
mediana = df_con_sector["precio"].median()
iqr = df_con_sector["precio"].quantile(0.75) - df_con_sector["precio"].quantile(0.25)
precio_robusto = (df_con_sector["precio"] - mediana) / iqr
```

**Línea por línea:**
- `z_score = (precio - media) / desvío` → la fórmula clásica de estandarización: centra los datos en 0 y los escala a una desviación estándar de 1.
- `df_con_sector["precio"].median()` → el valor central de la columna (menos sensible a outliers que la media).
- `.quantile(0.75) - .quantile(0.25)` → el Rango Intercuartílico (IQR): la diferencia entre el percentil 75 y el percentil 25.
- `precio_robusto = (precio - mediana) / iqr` → la misma idea que el z-score, pero con estadísticos robustos en vez de media y desvío estándar.

---

## Módulo 8 — La Sinergia de Datos: NumPy y Pandas en Profundidad

### El motor y la carrocería, en un caso completo

Retomando el Módulo 4: **NumPy es el motor** (cálculo puro, homogéneo, veloz) y **Pandas es la carrocería** (etiquetas, tipos mixtos, manejo de nulos) — y por dentro, cada columna de un DataFrame **es** un array de NumPy. En un proyecto real, casi siempre se usan las dos en la misma celda: Pandas prepara y limpia, NumPy hace la cuenta pesada.

**Caso real: calcular la volatilidad de un portafolio.** La fórmula de la varianza de un portafolio es $\sigma_p^2 = w^T \Sigma w$, donde $w$ es el vector de pesos y $\Sigma$ la matriz de covarianzas entre activos — pura álgebra lineal, la misma herramienta del Módulo 3.

```python
# Pandas: preparar los datos (retornos porcentuales, maneja el primer NaN solo)
retornos = df_stocks.pct_change().dropna()

# NumPy: la cuenta pesada (covarianza + álgebra lineal)
matriz_covarianza = np.cov(retornos.values.T)   # retornos.values -> ya es un ndarray
pesos = np.ones(14) / 14

varianza_portafolio = pesos @ matriz_covarianza @ pesos   # w^T Σ w
volatilidad_portafolio = np.sqrt(varianza_portafolio)

print(f"Volatilidad mensual del portafolio: {volatilidad_portafolio:.4%}")
```

**Línea por línea:**
- `df_stocks.pct_change()` → calcula el % de cambio entre cada fila y la anterior, columna por columna; la primera fila siempre da `NaN` (no tiene fila anterior con la cual compararse).
- `.dropna()` → elimina esa primera fila con `NaN`.
- `np.cov(retornos.values.T)` → calcula la matriz de covarianzas; se transpone con `.T` porque `np.cov` espera una fila por variable (acción), no por observación (mes).
- `pesos @ matriz_covarianza @ pesos` → dos multiplicaciones matriciales encadenadas: primero `matriz_covarianza @ pesos`, y el resultado se vuelve a multiplicar por `pesos` — la fórmula $w^T\Sigma w$ de la varianza del portafolio.
- `np.sqrt(varianza_portafolio)` → la volatilidad es, por definición, la raíz cuadrada de la varianza.

### El mito del bucle `for` sobre un DataFrame

**Error frecuente**: recorrer un DataFrame fila por fila con `for` y `.iloc[i]` para calcular algo que ya tiene una operación vectorizada. Es lento y destruye la ventaja competitiva de las dos librerías.

```python
# Mal: recorrer fila por fila
totales = []
for i in range(len(df_stocks)):
    totales.append(df_stocks.iloc[i].sum())

# Bien: vectorizado, con la misma API de Pandas (que delega en NumPy por debajo)
totales_vectorizado = df_stocks.sum(axis=1)
```

**Línea por línea:**
- `totales = []` → lista vacía donde vamos a ir acumulando un resultado por fila.
- `for i in range(len(df_stocks)):` → genera un índice numérico `0, 1, 2...` por cada fila del DataFrame.
- `df_stocks.iloc[i].sum()` → `iloc[i]` trae la fila `i` completa; `.sum()` la suma. El problema no es esta suma (que sí es rápida), sino el bucle de Python que la llama fila por fila.
- `df_stocks.sum(axis=1)` → `axis=1` suma a lo largo de las columnas, **una vez por fila**, pero todo resuelto internamente por Pandas/NumPy, sin que Python itere.

### Aplicaciones reales por industria

- **Finanzas** (nuestro propio `stocks.csv`): Pandas nació en el sector financiero para manejar series temporales de precios; calcular medias móviles o volatilidad toma dos líneas.
- **Retail**: unir tablas de "Ventas" e "Inventario" por ID de producto (el mismo patrón de `merge` del Módulo 7) para anticipar faltantes de stock.
- **Investigación científica**: NumPy es el estándar para procesar imágenes (matrices de píxeles) o señales — cálculos que en Python puro tardarían días.

---

## Módulo 9 — Inspección Inicial de Datos y Pre-Entrega

### `head()`, `info()`, `describe()`: el perfilado inicial

Antes de cualquier cálculo o gráfico, un Data Scientist le "toma el pulso" al dataset. Tres funciones cubren el 90% del diagnóstico inicial:

- **`df.head(n)`**: primeras `n` filas — ¿se cargaron bien las columnas?
- **`df.info()`**: nombre de cada columna, cantidad de valores no nulos y `dtype` — el comando más importante del diagnóstico.
- **`df.describe()`**: resumen estadístico (`count`, `mean`, `std`, `min`, percentiles, `max`) de las columnas numéricas.

```python
print(df_stocks.head())
print(df_stocks.shape)          # (71, 14)
df_stocks.info()
print(df_stocks.describe())
print(df_stocks.isnull().sum()) # en este dataset, todo en cero: no hay nulos reales
```

**Línea por línea:**
- `df_stocks.head()` → sin argumento, muestra las primeras 5 filas por defecto.
- `df_stocks.shape` → es un **atributo** (sin paréntesis): la tupla `(filas, columnas)`.
- `df_stocks.info()` → imprime, para cada columna, su nombre, cuántos valores no nulos tiene y su `dtype`.
- `df_stocks.describe()` → calcula `count`, `mean`, `std`, `min`, los percentiles 25/50/75% y `max` de cada columna numérica.
- `df_stocks.isnull().sum()` → la misma combinación que en el Módulo 6: máscara booleana + suma por columna.

**Cómo leer un `describe()` real**: en `df_stocks.describe()`, comparar `min` y `max` de cada acción contra lo que sabés del mundo real es la primera línea de defensa contra errores de carga — un precio de acción negativo, por ejemplo, sería una alerta inmediata (no es el caso acá, pero es el hábito a construir).

### Pre-Entrega: Checkpoint — Estructura Inicial del Dataset

| Bloque | Contenido |
|---|---|
| **1. Carga e Inspección** | `pd.read_csv`, `.head()` para inspección visual, `.shape` e `.info()` para volumen y tipos. |
| **2. Perfilado Inicial** | Total de valores nulos por columna (`isnull().sum()`) y estadísticas descriptivas con `.describe()`. |
| **3. Saneamiento y Selección** | Al menos 3 filtros booleanos para recortar el dataset (ej. `df[df["MSFT"] > 100]`); eliminar alguna columna innecesaria. |
| **4. Reflexión** | Celda Markdown: ¿qué problemas encontraste en los datos? ¿cuáles van a ser tus variables clave? |

**Entregable**: un PDF con el Jupyter Notebook exportado (código + resultados + comentarios), incluyendo el diagnóstico de nulos, `dtypes`, `describe()` y los filtros aplicados. Nombre sugerido: `Apellido_Nombre_Checkpoint1.pdf`.

### `.apply()`: aplicar funciones propias a un DataFrame

Hasta acá vimos operaciones vectorizadas nativas (`df["col"] * 2`, `.mean()`, `.groupby().transform()`) y, en el Bloque 0, funciones `def` con `if/elif/else`. **`.apply()` es el puente entre las dos cosas**: te deja correr **tu propia función de Python** — con toda la lógica condicional que necesites — sobre cada valor de una columna o cada fila de un DataFrame.

```python
# Sobre una columna (Serie): la función recibe UN VALOR por vez
df["MSFT"].apply(mi_funcion)

# Sobre el DataFrame completo, fila por fila: la función recibe LA FILA completa (axis=1)
df.apply(mi_funcion, axis=1)
```

**Línea por línea:**
- `df["MSFT"].apply(mi_funcion)` → recorre la Serie `df["MSFT"]` y le pasa **cada valor individual** a `mi_funcion`, uno por uno.
- `df.apply(mi_funcion, axis=1)` → `axis=1` cambia el comportamiento: ahora `mi_funcion` recibe **la fila entera** (como una Serie), lo que permite reglas que combinan varias columnas a la vez.

**Con una función `def` (para lógica de negocio con `if/elif/else`):**

```python
def categorizar_precio(precio):
    if precio < 100:
        return "Bajo"
    elif precio < 250:
        return "Medio"
    else:
        return "Alto"

df_stocks["MSFT_categoria"] = df_stocks["MSFT"].apply(categorizar_precio)
```

**Línea por línea:**
- `def categorizar_precio(precio):` → recibe un **único** valor numérico (un precio puntual) cada vez que se la llama.
- `if / elif / else` → la misma estructura condicional del Bloque 0, ahora aplicada a un precio de acción en vez de un precio de producto.
- `df_stocks["MSFT"].apply(categorizar_precio)` → llama a la función una vez por cada valor de la columna `MSFT`, y el resultado se guarda en una columna nueva.

**Con una `lambda` (para una transformación de una sola línea):**

```python
df_stocks["GOOG_miles"] = df_stocks["GOOG"].apply(lambda x: round(x / 1000, 2))
```

**Línea por línea:**
- `lambda x: round(x / 1000, 2)` → función anónima de una sola línea: recibe `x` (un precio), lo divide por 1000 y lo redondea a 2 decimales.
- `.apply(lambda ...)` → aplica esa mini función a cada valor de la columna `GOOG`, igual que hicimos antes con `categorizar_precio`, pero sin necesidad de darle nombre a la función.

**¿Por qué no usarlo siempre, si es tan cómodo?** Por dentro, `.apply()` **sí itera** fila por fila o valor por valor — no es vectorización real, es "el mito del bucle `for`" del Módulo 8 con otro disfraz. Es más lento que `df["col"] * 2` sobre datasets grandes. La regla práctica: **si existe una operación vectorizada que resuelve lo mismo, se prefiere esa**; `.apply()` se reserva para cuando la lógica es condicional o demasiado específica como para expresarla con operadores matemáticos directos — exactamente el caso de una función `def` con `if/elif/else`.

### Ejemplo resuelto: cómo encarar la Segunda Pre-Entrega

La consigna de la Segunda Pre-Entrega (Ingesta, Radiografía y Funciones de Transformación) pide 4 bloques. Así se resuelven los cuatro, de punta a punta, sobre `stocks.csv` — es el mismo patrón que vas a aplicar sobre el dataset que elegiste en tu Pre-entrega 1.

```python
# ==========================================
# BLOQUE 1: INGESTA DE DATOS Y PRIMER VISTAZO
# ==========================================
import pandas as pd

df = pd.read_csv("stocks.csv", parse_dates=["formatted_date"])
display(df.head())

# ==========================================
# BLOQUE 2: RADIOGRAFÍA TÉCNICA
# ==========================================
print(f"Dimensiones del dataset: {df.shape}")
df.info()
display(df.describe())

# Hallazgos (irían en una celda Markdown en el notebook real):
# 1) MSFT va de ~45 a ~330 USD: rango amplio, coherente con el crecimiento
#    tecnológico entre 2016 y 2021.
# 2) RCL (Royal Caribbean) tiene una desviación estándar alta relativa a su media:
#    coherente con el shock del turismo/cruceros durante 2020.

# ==========================================
# BLOQUE 3: DIAGNÓSTICO DE DATOS FALTANTES
# ==========================================
porcentaje_nulos = (df.isnull().mean() * 100).sort_values(ascending=False)
display(porcentaje_nulos)
# En stocks.csv da 0% en todas las columnas porque el dataset viene limpio;
# con un dataset real casi siempre vas a ver porcentajes distintos de cero acá.

# ==========================================
# BLOQUE 4: LÓGICA ALGORÍTMICA Y FUNCIONES PERSONALIZADAS
# ==========================================
# Función formal (def + if/elif/else): regla de negocio sobre MSFT
def categorizar_precio_msft(precio):
    if precio < 100:
        return "Bajo"
    elif precio < 250:
        return "Medio"
    else:
        return "Alto"

df["MSFT_categoria"] = df["MSFT"].apply(categorizar_precio_msft)

# Función express (lambda): transformación rápida sobre GOOG
df["GOOG_miles"] = df["GOOG"].apply(lambda x: round(x / 1000, 2))

display(df[["MSFT", "MSFT_categoria", "GOOG", "GOOG_miles"]].head())
```

**Línea por línea:**
- **Bloque 1**: `pd.read_csv(..., parse_dates=[...])` carga el archivo y convierte la fecha; `display(df.head())` renderiza las primeras 5 filas con estilo (mejor que `print()` en un notebook).
- **Bloque 2**: `df.shape` da las dimensiones; `df.info()` lista tipos y no-nulos; `df.describe()` da el resumen estadístico — los tres comandos de radiografía técnica que pide la consigna.
- Los comentarios de "Hallazgos" son el equivalente en código de lo que en el notebook real iría en una **celda Markdown aparte**, como pide el bloque 2 de la consigna.
- **Bloque 3**: `df.isnull().mean() * 100` da el porcentaje de nulos por columna; `.sort_values(ascending=False)` lo ordena de mayor a menor, tal como pide la consigna.
- **Bloque 4**: la función `categorizar_precio_msft` (con `if/elif/else`) y la `lambda` sobre `GOOG` son las dos funciones que pide la consigna, y ambas se aplican con `.apply()` para crear columnas nuevas.
- El `display()` final selecciona 4 columnas a la vez (doble corchete `[[...]]`) para comparar el valor original contra el transformado, en la misma tabla.

Con esto, los 4 bloques de la consigna quedan resueltos: ingesta con vistazo estilizado, radiografía técnica con hallazgos comentados, ranking de nulos, y las dos funciones (formal + lambda) aplicadas al dataset con `.apply()`.

---

## Material de la clase

| Archivo | Qué es |
|---|---|
| `Clase 03.pdf` | Material teórico oficial de la unidad (fuente original de esta guía). |
| `Clase03.html` | Diapositivas para proyectar en clase (43 filminas). Abrir en el navegador; navegación con flechas del teclado o los botones inferiores. |
| `Clase 03.ipynb` | Notebook con teoría ampliada + todos los ejemplos ejecutables. Es el material que se comparte con los alumnos. |
| `stocks.csv` | Dataset real de precios de acciones (14 tickers, 2016–2021), usado como hilo conductor en todos los módulos. |
| `Material/` | Carpeta con recursos adicionales. |

**Cómo usar esta guía durante la clase**: cada sección sigue el mismo orden que las diapositivas y el notebook — se puede ir alternando entre proyectar la filmina/notebook correspondiente y volver acá si hace falta más contexto, una analogía o recordar un detalle técnico.
