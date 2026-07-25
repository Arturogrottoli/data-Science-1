# Clase 03: NumPy y Pandas — Guía Completa para el Docente

Esta guía es el **libreto de apoyo para dictar la Clase 03**. Reúne, en un solo lugar y con más profundidad de la que entra en una diapositiva, toda la teoría y todos los ejemplos que aparecen en:

- **`Clase 03.pdf`** — el material teórico oficial de la unidad (9 secciones: de Arrays y Vectorización hasta la Inspección Inicial de datos).
- **`Clase03.html`** — las diapositivas que se proyectan en clase (40 filminas).
- **`Clase 03.ipynb`** — el notebook con todos los ejemplos ejecutables y comentados, con teoría ampliada en celdas Markdown.

La idea es la misma que en la guía de Clase 02: si en medio de la clase te falla la memoria sobre un detalle (¿por qué `reshape` tira error si no coinciden las dimensiones?, ¿cuál era la regla de broadcasting?, ¿`agg` o `transform`?), lo encuentres acá explicado con más contexto del que alcanza a mostrar una filmina.

Todos los ejemplos de código usan `stocks.csv` (en esta misma carpeta) como dataset real de referencia: precios mensuales de 14 acciones entre 2016 y 2021.

---

## Índice

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

**Y las listas también son variables**: hasta acá `producto`, `precio`, `stock` y `en_oferta` son **escalares** — un solo valor cada una. Python también tiene **colecciones**, que agrupan muchos escalares en una sola variable; la más simple es la **lista**. Definimos acá `precios_lista`, que vamos a reutilizar tal cual en el `for` y en la función de más abajo — así queda claro desde el principio que no es un dato nuevo que aparece de la nada.

```python
precios_lista = [15, 45, 120, 300, 80, 500, 20]
print(f"{precios_lista} -> tipo: {type(precios_lista)}")
print(f"Primer precio: {precios_lista[0]}")
print(f"Cantidad de precios: {len(precios_lista)}")
```

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

### Ciclo for (repaso)

`for` recorre una colección elemento por elemento — en este caso, la misma `precios_lista` que ya definimos arriba. **Este es el protagonista del contraste que viene**: todo lo que hoy resolvemos con un `for` sobre una lista, en NumPy lo vamos a resolver sin ningún bucle explícito, operando sobre el array completo de una sola vez.

```python
for p in precios_lista:
    print(f"Precio: {p}")
```

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

### La Matriz de Datos: filas, columnas e indexación

En Data Science, la información casi siempre se organiza de forma tabular — una **Matriz de Datos**:

- **Filas (eje 0)**: cada observación. En `precios_matriz` (Módulo 1), cada fila es **un mes**.
- **Columnas (eje 1)**: cada variable. Cada columna es **una acción** (MCD, SBUX, GOOG...).
- **Indexación base 0**: `precios_matriz[2, 1]` es la **tercera fila, segunda columna** → el precio de SBUX en el tercer mes registrado.

```python
print(precios_matriz[2, 1])   # precio de SBUX (columna 1) en el mes con índice 2
```

### Broadcasting: el superpoder de NumPy

El **Broadcasting** es el conjunto de reglas que le permite a NumPy operar entre arrays de **distinta forma**, sin escribir un bucle y sin copiar datos de más en memoria. NumPy compara las formas **de derecha a izquierda**: dos dimensiones son compatibles si son iguales o si una de ellas es `1`.

**Caso real: centrar los precios restando la media de cada acción.** `precios_matriz` tiene shape `(71, 14)`; el vector de medias por columna tiene shape `(14,)`. NumPy "estira" el vector de 14 medias para cubrir las 71 filas, sin crear 71 copias del vector.

```python
medias_por_accion = precios_matriz.mean(axis=0)     # shape (14,) -> una media por columna
print(f"Shape de las medias: {medias_por_accion.shape}")

precios_centrados = precios_matriz - medias_por_accion   # broadcasting: (71,14) - (14,) -> (71,14)
print(precios_centrados[0])   # cuánto se desvía cada acción de su propia media, en el primer mes
```

### Multiplicación elemento a elemento (`*`) vs. Transposición (`.T`)

- **`*` (elemento a elemento)**: multiplica posición a posición — (0,0) con (0,0), (0,1) con (0,1)... Debe cumplir las reglas de broadcasting, **no** es álgebra lineal.
- **`.T` (transposición)**: gira la matriz — filas pasan a ser columnas. `precios_matriz` es `(71, 14)`; `precios_matriz.T` es `(14, 71)`. No copia datos, solo cambia cómo se *leen* — es prácticamente gratis en rendimiento, y es la herramienta clave para alinear dimensiones antes de una multiplicación matricial.

```python
volatilidad = precios_matriz.std(axis=0)           # desvío estándar por acción, shape (14,)
precios_normalizados = precios_centrados / volatilidad   # * y / también son elemento a elemento

print(precios_matriz.T.shape)   # (14, 71) -> ahora cada fila es una acción, cada columna un mes
```

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

### Errores comunes de broadcasting

```python
vector_mal_dimensionado = np.array([1, 2, 3])   # shape (3,), pero precios_matriz tiene 14 columnas

# precios_matriz - vector_mal_dimensionado
# ValueError: operands could not be broadcast together with shapes (71,14) (3,)
```

| Error | Causa | Solución |
|---|---|---|
| `could not be broadcast together` | Las formas no son iguales ni una de ellas es 1 (comparando de derecha a izquierda) | Verificar `.shape` de ambos arrays antes de operar |
| `shapes not aligned` (con `@`) | Las columnas de A no igualan las filas de B | Usar `.T` para transponer el que corresponda: `A @ B.T` |
| Confundir `*` con `@` | `*` es elemento a elemento; `@` combina filas y columnas (álgebra lineal) | Preguntarse: "¿quiero ajustar cada valor, o combinar variables?" |

---

## Módulo 3 — Álgebra Lineal con NumPy

### Producto punto (Dot Product)

El producto punto toma dos vectores de la **misma longitud** y devuelve un único **escalar**: multiplica los elementos correspondientes y suma los resultados.

**Caso real**: si tenemos el retorno promedio de cada acción y los pesos del portafolio del Módulo 2, el producto punto entre ambos vectores nos da el **retorno esperado del portafolio completo**, en una sola cuenta.

```python
retornos_mensuales = (precios_matriz[1:] - precios_matriz[:-1]) / precios_matriz[:-1]  # % de cambio mes a mes
retorno_promedio_por_accion = retornos_mensuales.mean(axis=0)   # shape (14,)

retorno_esperado_portafolio = np.dot(retorno_promedio_por_accion, pesos)   # escalar
print(f"Retorno mensual esperado del portafolio: {retorno_esperado_portafolio:.4%}")
```

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

---

## Módulo 5 — Introducción a Pandas: Series y DataFrames

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

---

## Módulo 7 — Integración, Agregación y Preprocesamiento Avanzado

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

### Outliers y escalamiento: Winsorización, Z-Score y Robust Scaling

En vez de eliminar valores extremos (perdiendo información), la **Winsorización** les pone un tope en un percentil bajo y otro alto — típicamente P1 y P99.

```python
p1, p99 = np.percentile(df_con_sector["precio"], [1, 99])
precio_winsorizado = np.clip(df_con_sector["precio"], p1, p99)   # los extremos se "achatan" al tope
```

Para llevar variables a la misma escala (imprescindible en KNN, regresiones regularizadas o redes neuronales):

```python
# Z-Score: sensible a outliers (media y desvío se distorsionan con valores extremos)
z_score = (df_con_sector["precio"] - df_con_sector["precio"].mean()) / df_con_sector["precio"].std()

# Robust Scaling: usa mediana e IQR, inmune a outliers extremos
mediana = df_con_sector["precio"].median()
iqr = df_con_sector["precio"].quantile(0.75) - df_con_sector["precio"].quantile(0.25)
precio_robusto = (df_con_sector["precio"] - mediana) / iqr
```

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

**Cómo leer un `describe()` real**: en `df_stocks.describe()`, comparar `min` y `max` de cada acción contra lo que sabés del mundo real es la primera línea de defensa contra errores de carga — un precio de acción negativo, por ejemplo, sería una alerta inmediata (no es el caso acá, pero es el hábito a construir).

### Pre-Entrega: Checkpoint — Estructura Inicial del Dataset

| Bloque | Contenido |
|---|---|
| **1. Carga e Inspección** | `pd.read_csv`, `.head()` para inspección visual, `.shape` e `.info()` para volumen y tipos. |
| **2. Perfilado Inicial** | Total de valores nulos por columna (`isnull().sum()`) y estadísticas descriptivas con `.describe()`. |
| **3. Saneamiento y Selección** | Al menos 3 filtros booleanos para recortar el dataset (ej. `df[df["MSFT"] > 100]`); eliminar alguna columna innecesaria. |
| **4. Reflexión** | Celda Markdown: ¿qué problemas encontraste en los datos? ¿cuáles van a ser tus variables clave? |

**Entregable**: un PDF con el Jupyter Notebook exportado (código + resultados + comentarios), incluyendo el diagnóstico de nulos, `dtypes`, `describe()` y los filtros aplicados. Nombre sugerido: `Apellido_Nombre_Checkpoint1.pdf`.

---

## Material de la clase

| Archivo | Qué es |
|---|---|
| `Clase 03.pdf` | Material teórico oficial de la unidad (fuente original de esta guía). |
| `Clase03.html` | Diapositivas para proyectar en clase (40 filminas). Abrir en el navegador; navegación con flechas del teclado o los botones inferiores. |
| `Clase 03.ipynb` | Notebook con teoría ampliada + todos los ejemplos ejecutables. Es el material que se comparte con los alumnos. |
| `stocks.csv` | Dataset real de precios de acciones (14 tickers, 2016–2021), usado como hilo conductor en todos los módulos. |
| `Material/` | Carpeta con recursos adicionales. |

**Cómo usar esta guía durante la clase**: cada sección sigue el mismo orden que las diapositivas y el notebook — se puede ir alternando entre proyectar la filmina/notebook correspondiente y volver acá si hace falta más contexto, una analogía o recordar un detalle técnico.
