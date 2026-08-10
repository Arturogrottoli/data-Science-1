# Clase 04: Manipulación de Datos con Pandas — Guía Completa para el Docente

Esta guía es el **libreto de apoyo para dictar la Clase 04**. Reúne, en un solo lugar y con más profundidad de la que entra en una diapositiva, la teoría y los ejemplos que acompañan a:

- **`Clase 04.pdf` / `Clase 04.docx`** — el material teórico oficial de la unidad (5 secciones: Valores Faltantes y Duplicados, Transformaciones con `map`/`apply`, GroupBy y Pivot Tables, Fechas y Resampling, y Manipulación de Datos — síntesis + Pre-Entrega).
- **`Clase04.html`** — las diapositivas que se proyectan en clase (41 filminas, Módulos 1 a 5).
- **`Clase 04.ipynb`** — el notebook con todos los ejemplos ejecutables.

Antes de los 5 módulos nuevos hay un **Módulo 0 de repaso**, pensado como puente: retoma comandos de Pandas que ya se usaron en la Clase 03 (Módulos 5 a 8: Series/DataFrame, Preprocesamiento, Integración/Agregación, Sinergia NumPy-Pandas) pero que en esa clase no llegaron a explicarse con el detalle que merecen — `.loc`/`.iloc`, filtrado booleano combinado, `value_counts`, `sort_values`, `rename`, `drop`, un `groupby` simple. Sirve para nivelar antes de sumar contenido nuevo.

---

## Índice

- [Sobre el Dataset: `fifa_world_cup_2026_player_performance.csv`](#sobre-el-dataset-fifa_world_cup_2026_player_performancecsv)
- [Módulo 0 — Repaso: NumPy Relámpago y Comandos Clásicos de Pandas](#módulo-0--repaso-numpy-relámpago-y-comandos-clásicos-de-pandas)
- [Módulo 1 — Valores Faltantes y Duplicados](#módulo-1--valores-faltantes-y-duplicados)
- [Módulo 2 — Transformaciones con `map` y `apply`](#módulo-2--transformaciones-con-map-y-apply)
- [Módulo 3 — Agrupar, Resumir y Comparar: GroupBy y Pivot Tables](#módulo-3--agrupar-resumir-y-comparar-groupby-y-pivot-tables)
- [Módulo 4 — Fechas, Series Temporales y Resampling](#módulo-4--fechas-series-temporales-y-resampling)
- [Módulo 5 — Manipulación de Datos: Pandas (Síntesis)](#módulo-5--manipulación-de-datos-pandas-síntesis)
- [Módulo 6 — Pre-Entrega 4: Estructura Inicial del Dataset del Proyecto](#módulo-6--pre-entrega-4-estructura-inicial-del-dataset-del-proyecto)

---

## Mapa rápido de la clase

Para seguir la clase en paralelo con `Clase04.html` (41 filminas) sin perderte:

| # | Módulo | Slides | Notebook (`Clase 04.ipynb`) | Idea central |
|---|---|---|---|---|
| 0 | Repaso: NumPy relámpago y Pandas clásico | *(sin slides — repaso previo)* | Módulo 0 | Puente con la Clase 03: `.loc`/`.iloc`, filtros combinados, `value_counts`, `sort_values`, `rename`, `drop`, `groupby` simple |
| 1 | Valores Faltantes y Duplicados | 02–09 | Módulo 1 | El lenguaje de la ausencia, detección, cuantificación, `duplicated()`/`drop_duplicates()`, eliminar vs. imputar |
| 2 | Transformaciones con `map` y `apply` | 10–15 | Módulo 2 | Traducir etiquetas y calcular indicadores sin bucles |
| 3 | Agrupar, Resumir y Comparar | 16–23 | Módulo 3 | Split-Apply-Combine, `agg()` múltiple, `groupby` con varias columnas, `pivot_table` |
| — | Break del Coder | 24 | — | Corte de ~10 minutos |
| 4 | Fechas, Series Temporales y Resampling | 25–29 | Módulo 4 | `to_datetime`, índice temporal, `resample()` |
| 5 | Manipulación de Datos: Pandas (Síntesis) | 30–37 | Módulo 5 | Origen de Pandas, pipeline profesional completo, `merge` vs. `concat`, errores comunes |
| 6 | Pre-Entrega 4: Estructura Inicial del Dataset | 38–41 | Módulo 6 | Consigna evaluable: carga, `.shape`, `.info()`, `.describe()`, 3 filtros booleanos de saneamiento |

---

## Sobre el Dataset: `fifa_world_cup_2026_player_performance.csv`

A partir de esta clase cambiamos de dataset: dejamos `stocks.csv` (Clase 03, series financieras) y pasamos a un dataset deportivo, con una estructura muy distinta — más filas, más columnas, y una mezcla mucho más rica de texto, categorías y números. Esa variedad es justamente lo que hace falta para practicar `groupby`, `pivot_table` y fechas más adelante en esta clase.

**Fuente**: descargado de Kaggle — [`rauffauzanrambe/fifa-world-cup-2026-player-performance-dataset`](https://www.kaggle.com/datasets/rauffauzanrambe/fifa-world-cup-2026-player-performance-dataset).

**Estructura del archivo:**
- **54.600 filas**. Cada fila es una **aparición de un jugador en un partido** (no un jugador único): si Messi jugó 5 partidos, aparece 5 veces, una por partido.
- **75 columnas**, que se agrupan en tres familias:
  - **Datos del jugador** (no cambian partido a partido): `player_id`, `player_name`, `age`, `nationality`, `team`, `position`, `height_cm`, `market_value_eur`, `club_name`...
  - **Datos del partido**: `match_id`, `match_date`, `stadium`, `opponent_team`, `tournament_stage`, `match_result` (`W`/`D`/`L`)...
  - **Métricas de rendimiento en ese partido**: `goals`, `assists`, `shots`, `pass_accuracy`, `tackles`, `minutes_played`, `player_rating`...
- **1.248 jugadores únicos**, **48 equipos**, **1.050 partidos únicos**. La relación entre estos números no es arbitraria: 1.050 partidos × ~26 convocados por plantel (titulares + banco de ambos equipos) da aproximadamente las 54.600 filas totales — la aritmética que explica por qué el dataset tiene ese tamaño exacto.
- **`position`** tiene 4 valores: `Defender` (18.900 filas), `Midfielder` (16.800), `Forward` (12.600), `Goalkeeper` (6.300).
- **0 valores nulos y 0 filas duplicadas** en todo el archivo — a diferencia de `stocks.csv`, acá no hace falta ni siquiera simular problemas para el Módulo 1 de esta clase (duplicados), porque otras columnas sí van a tener inconsistencias que vamos a explotar ahí.

**Un hallazgo real para abrir la clase**: la columna `player_rating` va de `0` a `9.4`, pero el percentil 25 es exactamente `0`. ¿Por qué tantos ceros? No es un error de carga: **`player_rating == 0` ocurre exactamente en las mismas 23.042 filas donde `minutes_played == 0`** — jugadores convocados a un partido que no llegaron a jugar (suplentes no utilizados). Es un buen ejemplo de que un `describe()` raro no siempre es un dato roto: a veces hay que cruzarlo con otra columna para entender la causa, antes de decidir si se filtra, se imputa o se deja como está.

---

## Módulo 0 — Repaso: NumPy Relámpago y Comandos Clásicos de Pandas

**Por qué existe este módulo**: en la Clase 03 se cubrió NumPy en profundidad (Módulos 1 a 4) y Pandas desde cero (Módulos 5 a 8: Series/DataFrame, Preprocesamiento, Integración/Agregación, Sinergia NumPy-Pandas). Pero en el ritmo de esa clase quedaron varios comandos de uso diario sin el espacio que merecían — `.loc`/`.iloc`, combinar varios filtros booleanos, `value_counts`, `sort_values`, `rename`, `drop`. Antes de sumar contenido nuevo (duplicados, `map`/`apply`, `groupby` avanzado, fechas), este módulo repasa y completa esa base, ahora sobre el dataset de esta clase.

Este módulo no tiene filminas propias — es puente de repaso, para dar antes de arrancar con la Filmina 02 (Módulo 1) de `Clase04.html`.

### 0.1 NumPy — Repaso relámpago

Muy breve, porque NumPy ya se vio a fondo en la Clase 03. Solo el recordatorio esencial antes de meternos de lleno en Pandas: una columna de un DataFrame se puede convertir en un `ndarray` con `.to_numpy()` (o el más viejo `.values`), y a partir de ahí todas las operaciones son **vectorizadas** — sin `for`.

🎯 **Qué mostramos acá:** convertir una columna del DataFrame en un array de NumPy, transformarla sin bucles (pasar de euros a millones) y calcular estadísticas con funciones de NumPy (`.mean()`, `.max()`, `np.percentile`).

👉 **En Colab:**
```python
import numpy as np
import pandas as pd

df = pd.read_csv("fifa_world_cup_2026_player_performance.csv")

valores_mercado = df["market_value_eur"].to_numpy()   # de Serie de Pandas a ndarray de NumPy
print(type(valores_mercado))                            # numpy.ndarray

valores_en_millones = valores_mercado / 1_000_000        # vectorizado: sin for
print(f"Promedio: {valores_en_millones.mean():.2f}M | Máximo: {valores_en_millones.max():.2f}M")
print(f"Percentil 90: {np.percentile(valores_en_millones, 90):.2f}M")
```

**Línea por línea:**
- `df["market_value_eur"].to_numpy()` → toma la Serie (columna) y devuelve el `ndarray` que tiene por dentro, sin las etiquetas de índice de Pandas. Es casi gratis en tiempo de ejecución: como una Serie de Pandas ya usa un array de NumPy como almacenamiento interno, `to_numpy()` no recalcula nada, solo "destapa" la estructura que ya estaba ahí.
- `valores_mercado / 1_000_000` → división vectorizada: cada uno de los 54.600 valores se divide por un millón en una sola operación, sin recorrer la lista.
- `.mean()`, `.max()` → métodos de NumPy sobre el array resultante.
- `np.percentile(valores_en_millones, 90)` → valor de mercado por debajo del cual está el 90% de las apariciones jugador-partido; el 10% restante son los jugadores más valiosos del torneo. A diferencia de `.median()` (fijo en el percentil 50), `np.percentile` acepta cualquier corte — útil para definir umbrales de negocio ("el top 10% más caro") sin tener que ordenar el array a mano.

### 0.2 Pandas — Series y DataFrame: lo esencial (repaso)

Recordatorio rápido de la Clase 03, Módulo 5, ahora sobre el dataset nuevo: `read_csv` carga el archivo, `.shape`/`.info()`/`.head()` son el primer vistazo, y `df["col"]` (un corchete) devuelve una **Serie** mientras que `df[["col1", "col2"]]` (doble corchete) devuelve un **DataFrame**.

👉 **En Colab:**
```python
df = pd.read_csv("fifa_world_cup_2026_player_performance.csv")

print(df.shape)     # (54600, 75) -> 54.600 apariciones jugador-partido, 75 columnas
df.info()             # tipos de dato y nulos por columna
display(df.head(3))

nombres = df["player_name"]                          # un corchete -> Serie
ficha = df[["player_name", "team", "position"]]      # doble corchete -> DataFrame
print(type(nombres), type(ficha))
```

**Línea por línea:**
- `pd.read_csv(...)` → carga el CSV completo en un DataFrame llamado `df`.
- `df.shape` → tupla `(filas, columnas)`; acá `(54600, 75)`.
- `df.info()` → lista las 75 columnas con su tipo de dato (`Dtype`) y cuántos valores no nulos tiene cada una.
- `df["player_name"]` → **un** corchete selecciona una sola columna como Serie (1D, con índice).
- `df[["player_name", "team", "position"]]` → **doble** corchete (una lista de nombres adentro) selecciona varias columnas como DataFrame (2D). El corchete externo es el operador de selección de `df`; el interno es la lista de Python con los nombres — por eso "doble corchete" en realidad son dos cosas distintas, no un capricho de sintaxis.

### 0.3 Selección con `.loc[]` y `.iloc[]`

Este es uno de los comandos que en la Clase 03 no se llegó a desarrollar del todo. Son las dos formas "serias" de seleccionar filas y columnas a la vez:
- **`.iloc[filas, columnas]`**: selección **por posición** (números enteros, como el índice de una lista). `df.iloc[0:5, 0:4]` es "las primeras 5 filas, las primeras 4 columnas", sin importar cómo se llamen.
- **`.loc[filas, columnas]`**: selección **por etiqueta** (nombre del índice y nombre de columna). Se combina naturalmente con un filtro booleano en la parte de "filas".

🎯 **Qué mostramos acá:** `.iloc` para recortar un bloque del DataFrame por posición, y `.loc` para combinar un filtro con una selección de columnas por nombre — el patrón que más se usa en la práctica.

👉 **En Colab:**
```python
primeras_filas = df.iloc[0:5, 0:4]                 # posición: filas 0-4, columnas 0-3
print(primeras_filas)

goleadores = df.loc[df["goals"] > 2, ["player_name", "team", "goals"]]   # etiqueta: filtro + columnas por nombre
print(goleadores)
```

**Línea por línea:**
- `df.iloc[0:5, 0:4]` → `0:5` son las filas con posición 0 a 4 (5 filas), `0:4` las columnas con posición 0 a 3 (4 columnas); ambos por número, no por nombre. A diferencia del slicing de listas de Python, en `.iloc` el límite superior también es exclusivo — `0:5` trae 5 filas, no 6 —, así que en ese punto se comporta igual que una lista común.
- `df["goals"] > 2` → una Serie de booleanos (`True`/`False`), una por fila, según si esa aparición tuvo más de 2 goles.
- `df.loc[condición, ["player_name", "team", "goals"]]` → `.loc` recibe **primero** el filtro (qué filas) y **después** la lista de columnas por nombre (qué columnas); el resultado son las filas donde la condición dio `True`, mostrando solo esas tres columnas. Con `.loc`, a diferencia de `.iloc`, si el segundo argumento fuera un slice de etiquetas (`"col_a":"col_c"`) el límite superior sería **inclusivo** — otra diferencia entre ambos que suele confundir al principio.

### 0.4 Filtrado booleano con múltiples condiciones

También quedó corto en la Clase 03: cómo combinar **más de un** filtro a la vez. La regla clave: sobre Series de Pandas se usan `&` (y), `|` (o) y `~` (no) — **nunca** `and`/`or`/`not` de Python puro — y cada condición individual va entre paréntesis.

🎯 **Qué mostramos acá:** tres filtros sobre el hallazgo de la sección "Sobre el Dataset" (jugadores que no llegaron a jugar) y sobre las columnas de posición y valor de mercado, combinando `&` y `~`.

👉 **En Colab:**
```python
jugaron = df[df["minutes_played"] > 0]                                              # excluye los 23.042 suplentes no utilizados
delanteros_caros = df[(df["position"] == "Forward") & (df["market_value_eur"] > 50_000_000)]
no_finales = df[~(df["tournament_stage"] == "Final")]                               # ~ invierte la condición

print(f"Apariciones con minutos jugados: {len(jugaron)}")
print(f"Delanteros con valor > 50M€: {len(delanteros_caros)}")
print(f"Apariciones fuera de la Final: {len(no_finales)}")
```

**Línea por línea:**
- `df[df["minutes_played"] > 0]` → filtro simple: se queda con las filas donde la condición es `True`.
- `(df["position"] == "Forward") & (df["market_value_eur"] > 50_000_000)` → dos condiciones combinadas con `&` (Y lógico); cada una entre paréntesis, obligatorio para que Python respete el orden de evaluación. Sin los paréntesis, `&` tiene **mayor precedencia** que `==`/`>`, así que Python intentaría evaluar `market_value_eur > (50_000_000 & df["position"])` primero y el código rompería con un error de tipos.
- `~(df["tournament_stage"] == "Final")` → `~` niega la condición completa: se queda con todo lo que **no** es la Final (equivalente a `!=`, pero útil cuando la condición ya es compleja). La diferencia real aparece con condiciones combinadas: `~((a) & (b))` no es lo mismo que `(~a) & (~b)` — son las leyes de De Morgan, y en Pandas se aplican igual que en álgebra booleana.

### 0.5 Explorar categorías y ordenar: `value_counts()`, `unique()`, `nunique()`, `sort_values()`

Cuatro comandos "clásicos" para explorar una columna categórica o para ordenar por una numérica — de uso constante y que tampoco se habían explicado en detalle:
- **`value_counts()`**: cuenta cuántas veces aparece cada valor distinto de una columna, de mayor a menor.
- **`unique()`**: devuelve el array de valores distintos (sin contarlos).
- **`nunique()`**: devuelve **cuántos** valores distintos hay (un solo número).
- **`sort_values("columna", ascending=False)`**: reordena todo el DataFrame según una columna.

👉 **En Colab:**
```python
print(df["position"].value_counts())        # Defender 18900, Midfielder 16800, Forward 12600, Goalkeeper 6300
print(df["team"].nunique())                   # 48 equipos distintos

top_valor = df.sort_values("market_value_eur", ascending=False).head(5)
print(top_valor[["player_name", "team", "market_value_eur"]])
```

**Línea por línea:**
- `df["position"].value_counts()` → tabla de frecuencias: cada posición y cuántas apariciones jugador-partido tiene. Por defecto ordena de mayor a menor y **excluye** los `NaN` de ese conteo; `value_counts(normalize=True)` da directamente proporciones en vez de conteos absolutos, útil cuando se quiere comparar contra un dataset de otro tamaño.
- `df["team"].nunique()` → cuenta los 48 equipos distintos en la columna `team`, sin listarlos. Por dentro es casi equivalente a `len(df["team"].unique())`, pero `nunique()` no arma la lista completa en memoria, así que es la opción más liviana cuando solo interesa el número.
- `df.sort_values("market_value_eur", ascending=False)` → reordena **todo** el DataFrame de mayor a menor valor de mercado; `ascending=False` es necesario porque el default es ascendente.
- `.head(5)` → se queda con las primeras 5 filas del resultado ya ordenado: los 5 jugadores-partido con mayor valor de mercado.

### 0.6 Preprocesamiento: nulos, `rename()` y `drop()`

Repaso del Módulo 6 de Clase 03 (nulos), más `rename()` — el comando que realmente faltaba (cero apariciones en toda la guía de Clase 03) — reforzado junto con `drop()`, que sí se usó en Clase 03 pero conviene repasar en el mismo bloque por lo seguido que se usan juntos en la práctica.

👉 **En Colab:**
```python
print(df.isnull().sum().sum())    # 0 -> este dataset viene sin nulos

df_renombrado = df.rename(columns={"goals": "goles", "assists": "asistencias"})   # renombrar para trabajar en español
df_renombrado = df_renombrado.drop(columns=["jersey_number"])                      # sacar una columna que no vamos a usar

print(df_renombrado.columns[:5].tolist())
```

**Línea por línea:**
- `df.isnull().sum().sum()` → el primer `.sum()` cuenta nulos por columna, el segundo suma esos totales en un único número: `0` en este dataset.
- `df.rename(columns={...})` → recibe un diccionario `{"nombre viejo": "nombre nuevo"}`; devuelve un DataFrame **nuevo** con las columnas renombradas, sin tocar el original — por eso lo guardamos en `df_renombrado` en vez de sobreescribir `df` (los módulos siguientes de esta guía siguen usando los nombres originales `goals`/`assists`). El mismo método sirve para renombrar el **índice** en vez de columnas, pasando `index={...}` — o, con `axis=1` en vez del diccionario por `columns=`, aplicar una función a todos los nombres a la vez (por ejemplo, `df.rename(columns=str.lower)` para pasar todo a minúsculas de un saque).
- `df_renombrado.drop(columns=["jersey_number"])` → `columns=[...]` elimina columnas completas (para eliminar filas se usa `index=[...]`, visto ya en la Clase 03).

### 0.7 Agregación básica con `groupby` (puente al Módulo 3 de esta clase)

Repaso corto del `.agg()` de la Clase 03, Módulo 7, con el caso más simple posible: **una** columna para agrupar y **una** métrica. El `groupby` con múltiples agregaciones a la vez y las `pivot_table` (la versión "en cruz" del mismo concepto) se profundizan en el Módulo 3 de esta clase — acá solo repasamos la base para que ese salto no sea desde cero.

👉 **En Colab:**
```python
valor_por_posicion = df.groupby("position")["market_value_eur"].mean().sort_values(ascending=False)
print((valor_por_posicion / 1_000_000).round(2))
```

**Línea por línea:**
- `df.groupby("position")` → agrupa las 54.600 filas en 4 grupos, uno por posición. Por sí sola, esta línea todavía no calcula nada: devuelve un objeto `DataFrameGroupBy` que solo guarda "el plan" de agrupamiento; el cálculo real recién ocurre cuando se encadena una agregación como `.mean()`.
- `["market_value_eur"].mean()` → dentro de cada grupo, calcula el promedio de esa única columna.
- `.sort_values(ascending=False)` → ordena los 4 resultados de mayor a menor.
- El resultado real: **Forward** (≈27,2M€) > **Midfielder** (≈23,0M€) > **Defender** (≈15,4M€) > **Goalkeeper** (≈12,2M€) — coincide con la lógica del mercado de pases, donde los puestos ofensivos suelen cotizar más caro.

### 0.8 Sinergia NumPy + Pandas (repaso breve)

Cierre del módulo, repasando la idea central del Módulo 8 de Clase 03: Pandas está construido sobre NumPy, y conviene resolver con operaciones vectorizadas antes que con un `for` o incluso antes que con `.apply()` (que se introduce recién en el Módulo 2 de esta clase). `np.where(condición, si_true, si_false)` es el ejemplo típico: una versión vectorizada de un `if/else` aplicado a toda una columna de una sola vez.

👉 **En Colab:**
```python
df["tuvo_gol"] = np.where(df["goals"] > 0, "Sí", "No")   # if/else vectorizado, sin for ni apply
print(df["tuvo_gol"].value_counts())
```

**Línea por línea:**
- `np.where(df["goals"] > 0, "Sí", "No")` → evalúa la condición para las 54.600 filas a la vez; donde es `True` pone `"Sí"`, donde es `False` pone `"No"`. Devuelve un `ndarray`, que Pandas acepta directamente como nueva columna. `np.where` admite anidarse (`np.where(cond1, valor1, np.where(cond2, valor2, valor3))`) para simular un `if/elif/else` completo sin bucles, aunque para más de 2-3 niveles suele ser más legible `pd.cut()` o una función con `.apply()`.
- `df["tuvo_gol"] = ...` → crea la columna nueva `tuvo_gol` asignando ese array.
- `.value_counts()` → confirma cuántas apariciones jugador-partido terminaron con al menos un gol.

> **Con esto cerramos el repaso.** A partir del Módulo 1 (`Clase04.html`, Filmina 02 en adelante) el contenido es nuevo: duplicados, `map`/`apply` en profundidad, `groupby`/`pivot_table` avanzado, fechas y resampling, y la síntesis final con la Pre-Entrega de esta clase.

---

## Módulo 1 — Valores Faltantes y Duplicados

**Contexto**: `fifa_world_cup_2026_player_performance.csv` viene sin nulos y sin filas duplicadas — perfecto para un primer análisis, pero inútil para practicar limpieza. Igual que hicimos con `stocks.csv` en la Clase 03, **simulamos** ambos problemas sobre una copia (`df_sucio`), como si un par de fallas reales de captura hubieran ocurrido: un partido cargado dos veces por el sistema, y algunas métricas que no se pudieron registrar en vivo.

### El lenguaje de la ausencia: NaN, None, Null y vacío *(Filmina 03)*

Cuatro términos que se usan casi como sinónimos, pero no son lo mismo:
- **NaN** (*Not a Number*): el estándar técnico de Pandas/NumPy para faltantes numéricos. Para la computadora es, por dentro, un `float`.
- **None**: el "vacío" nativo de Python. Al cargar un DataFrame, Pandas suele convertirlo en `NaN` automáticamente.
- **Null**: término genérico de SQL; en Pandas se usa "nulo" y "faltante" casi indistintamente.
- **Cadena vacía (`""`) ≠ Faltante**: es texto válido sin caracteres — Pandas **no** la trata como nulo.

Un detalle que sorprende la primera vez: `np.nan == np.nan` da **`False`**. Por eso no se puede filtrar nulos con `==`; existen `isna()`/`isnull()`, hechas específicamente para esto. La razón técnica viene del estándar IEEE 754 para números de punto flotante: `NaN` está definido como "no comparable ni siquiera consigo mismo", precisamente para que cualquier operación que involucre un dato faltante se propague como faltante en vez de devolver silenciosamente un resultado falso.

Otro efecto colateral de que `NaN` sea siempre `float`: una columna de enteros (`int64`) que originalmente no tenía nulos, si en algún momento se le inserta uno, Pandas la **convierte entera a `float64`** — porque el tipo entero clásico de NumPy no tiene forma de representar "faltante". Es una causa frecuente del error "¿por qué mi columna de edades ahora tiene decimales?".

🎯 **Qué mostramos acá:** el comportamiento contraintuitivo de `NaN` contra sí mismo, y cómo `None` se convierte en `NaN` apenas entra a un DataFrame.

👉 **En Colab:**
```python
import pandas as pd
import numpy as np

print(np.nan == np.nan)   # False -> NaN nunca es igual a sí mismo

data = {"Edad": [25, np.nan, 30], "Ciudad": ["Madrid", "Bogotá", None]}
df_ejemplo = pd.DataFrame(data)
print(df_ejemplo.isna())   # None también se detecta como nulo
```

**Línea por línea:**
- `np.nan == np.nan` → comparar `NaN` con `NaN` da `False`: es la razón técnica por la que existen `isna()`/`isnull()` en vez de comparar con `==`.
- `data = {...}` → un diccionario con un `np.nan` explícito (numérico) y un `None` (genérico de Python) en la misma estructura.
- `df_ejemplo.isna()` → máscara booleana; confirma que Pandas trata a **ambos**, `NaN` y `None`, como faltantes.

### ¿Por qué aparecen los valores faltantes? *(Filmina 04)*

Cuatro causas típicas, y por qué importa distinguirlas antes de limpiar:
1. **Captura incompleta**: un sensor falló, o alguien prefirió no responder un campo.
2. **Errores de integración**: al unir dos tablas (ej. Clientes y Compras), un cliente sin compras queda con "monto" vacío — y **eso es información válida**, no un error.
3. **Datos no aplicables**: "nombre de la mascota" vacío en alguien sin mascota — vacío por lógica, no por falla.
4. **Diseño intencional**: el proceso de recolección cambió con el tiempo y un campo no se pedía antes.

**Principio de oro**: nunca limpiar sin entender el contexto. Borrar todas las filas con "descuento aplicado" nulo borraría a todos los clientes que pagaron el precio completo — el nulo, ahí, *es* el dato.

**En nuestro dataset real** hay un caso análogo, ya visto en el Módulo 0: `player_rating == 0` no es una falla de captura, es información válida (el jugador no jugó ese partido). Si alguien "limpiara" esas filas sin entender el contexto, estaría borrando exactamente a los suplentes no utilizados — un dato de convocatoria, no basura.

### Detección de valores faltantes: `info()`, `isna()`, `isnull()` *(Filmina 05)*

`info()` da el panorama general (columna por columna, cuántos valores "Non-Null" hay); `isna()`/`isnull()` (son alias, hacen lo mismo — `isnull()` existe por compatibilidad con la terminología de R/SQL, ninguna es "la versión vieja") devuelven la máscara booleana completa — útil para filtrar, inútil para "mirar" en un dataset de miles de filas.

🎯 **Qué mostramos acá:** simular la falla de captura sobre una copia del dataset real, y confirmarla con `info()`.

👉 **En Colab:**
```python
df_sucio = df.copy()

# Simulamos métricas que no se pudieron registrar en vivo (falla de captura real)
df_sucio.loc[50, "player_rating"] = np.nan
df_sucio.loc[120, "pass_accuracy"] = np.nan
df_sucio.loc[120, "distance_covered_km"] = np.nan   # dos nulos en la MISMA fila
df_sucio.loc[300, "nationality"] = np.nan

df_sucio.info()
```

**Línea por línea:**
- `df.copy()` → trabajamos sobre una copia; nunca se simulan fallas sobre el DataFrame original. Sin `.copy()`, `df_sucio = df` no crea una tabla nueva, solo una segunda etiqueta apuntando a los mismos datos en memoria — modificar `df_sucio` modificaría también `df` (el mismo problema de "Copias vs. Vistas" que se retoma en el Módulo 5).
- `df_sucio.loc[50, "player_rating"] = np.nan` → `.loc[fila, columna]` asigna un valor puntual; acá lo usamos para "romper" una celda a propósito.
- La fila `120` recibe **dos** nulos en columnas distintas — así después la Filmina 06 (cuantificación) tiene un caso real de "más de un problema en la misma fila".
- `df_sucio.info()` → confirma la baja: las columnas tocadas ahora muestran menos valores `Non-Null` que el resto.

### Cuantificación: ¿qué tan grave es el problema? *(Filmina 06)*

Una tabla, no una ciencia exacta, pero sirve como regla de arranque:

| % de Nulos | Estrategia sugerida |
|---|---|
| Menos del 5% | Suele ser seguro eliminar las filas o imputar con métodos sencillos. |
| Entre 5% y 30% | Requiere pensar una estrategia de imputación más sofisticada. |
| Más del 50% | A veces es mejor descartar la columna entera: más "ruido" que información real. |

Como `True` vale `1` y `False` vale `0`, `df.isna().sum()` cuenta nulos por columna sin necesidad de un `if`. Dividido por `len(df)` y multiplicado por 100, da el porcentaje — la métrica que realmente importa (50 nulos en 60 filas es un desastre; 50 nulos en 54.600, no). Ese mismo truco (`True`/`False` tratados como `1`/`0`) es lo que permite usar `.sum()` sobre cualquier máscara booleana para contar cuántos `True` hay — el mecanismo detrás de casi todos los "conteo de filas que cumplen tal condición" que se ven en la práctica.

👉 **En Colab:**
```python
nulos_totales = df_sucio.isna().sum()
porcentaje_nulos = (nulos_totales / len(df_sucio) * 100).round(4)

print(porcentaje_nulos[porcentaje_nulos > 0])   # solo las columnas afectadas
```

**Línea por línea:**
- `df_sucio.isna().sum()` → cuenta de nulos por columna (el `sum()` de una columna de booleanos cuenta los `True`).
- `nulos_totales / len(df_sucio) * 100` → convierte el conteo absoluto en porcentaje sobre el total de filas (54.600).
- `porcentaje_nulos[porcentaje_nulos > 0]` → filtro booleano (repaso del Módulo 0) para mostrar solo las columnas realmente afectadas, en vez de una lista de 74 columnas en cero.
- Con solo 4 celdas tocadas sobre 54.600 filas, cada porcentaje da bien por debajo del 5% — según la tabla, seguro para eliminar o imputar con un método simple.

### Identificación de duplicados: `duplicated()` y `drop_duplicates()` *(Filmina 07)*

- **`duplicated()`**: marca `True` las filas que **ya aparecieron antes**; por defecto compara **todos** los valores de la fila. La primera aparición de cada valor repetido queda marcada `False` — no son "las dos", es solo la copia extra.
- **`subset=["columna"]`**: busca duplicados mirando solo una columna (por ejemplo, un ID que debería ser único).
- **`drop_duplicates()`**: limpia el DataFrame; `keep="first"` o `keep="last"` decide cuál copia conservar. Existe una tercera opción, `keep=False`, que descarta **todas** las copias (incluida la primera) — útil cuando un duplicado es tan sospechoso que ni siquiera se confía en cuál de las dos versiones es la "buena".

🎯 **Qué mostramos acá:** simulamos el error típico de un sistema que registra el mismo partido dos veces (duplicado exacto de fila), lo detectamos y lo eliminamos.

👉 **En Colab:**
```python
# Simulamos que el sistema cargó por duplicado las apariciones de las filas 100 a 102
df_sucio = pd.concat([df_sucio, df_sucio.iloc[100:103]], ignore_index=True)

print(f"Duplicados exactos: {df_sucio.duplicated().sum()}")   # 3

df_limpio = df_sucio.drop_duplicates(keep="first")
print(f"Filas antes: {len(df_sucio)} | Filas después: {len(df_limpio)}")
```

**Línea por línea:**
- `pd.concat([df_sucio, df_sucio.iloc[100:103]], ignore_index=True)` → le pega **al final** una copia de las filas 100 a 102, simulando que el sistema las cargó dos veces; `ignore_index=True` genera un índice nuevo y correlativo (si no, quedarían índices repetidos).
- `df_sucio.duplicated().sum()` → cuenta cuántas filas son copia exacta de una anterior: da `3`, las que acabamos de duplicar.
- `df_sucio.drop_duplicates(keep="first")` → elimina las copias, conservando la primera aparición de cada una.

### Criterios de limpieza: ¿eliminar o imputar? *(Filmina 08)*

- **Eliminar (`dropna()`)**: la opción más drástica. Válida cuando sobran datos y perder un 2% no afecta estadísticamente, o cuando falta justo la columna "etiqueta" que se quiere predecir. Por defecto `dropna()` elimina una fila si tiene **al menos un** `NaN` en cualquier columna; el parámetro `thresh=N` cambia esa regla a "eliminar solo si tiene menos de `N` valores no nulos", útil cuando una fila puede tolerar 1 o 2 huecos sin volverse inútil.
- **Imputar (rellenar)**: **media** si la distribución es normal y sin outliers; **mediana** si hay valores extremos (más robusta, porque no la mueve un valor absurdo); **moda** para columnas categóricas; **valor constante** (`"Desconocido"`, `0`) cuando conviene conservar la fila sin inventar un número. La mediana es más robusta justamente porque es un estadístico de **orden** (el valor del medio), no de magnitud — un outlier corre la media pero no necesariamente cambia cuál es el valor "del medio".

👉 **En Colab:**
```python
media_rating = df_sucio["player_rating"].mean()
df_sucio["player_rating"] = df_sucio["player_rating"].fillna(media_rating)   # numérica -> media

moda_nacionalidad = df_sucio["nationality"].mode()[0]
df_sucio["nationality"] = df_sucio["nationality"].fillna(moda_nacionalidad)   # categórica -> moda

df_sucio = df_sucio.dropna(subset=["pass_accuracy", "distance_covered_km"])   # sin sustituto razonable -> eliminar
```

**Línea por línea:**
- `df_sucio["player_rating"].mean()` → calcula el promedio **ignorando** el `NaN` (comportamiento por defecto de Pandas), y se usa para imputar.
- `.fillna(media_rating)` → reemplaza únicamente los `NaN` de esa columna por el valor dado; el resto de los datos queda intacto.
- `.mode()[0]` → la moda puede devolver más de un valor si hay empate; `[0]` toma el primero como criterio simple.
- `dropna(subset=["pass_accuracy", "distance_covered_km"])` → elimina solo las filas con nulo en **esas** columnas puntuales (no en todo el DataFrame), porque ahí no hay un sustituto razonable (inventar la distancia recorrida por un jugador sería fabricar el dato).

### El dilema de los duplicados: no todos se borran *(Filmina 09)*

No todo duplicado es un error:
- **Identidad** (misma persona, mismo producto, mismo segundo exacto): error del sistema → **borrar**.
- **Eventos** (mismo cliente comprando el mismo artículo dos días seguidos): son dos eventos reales → **conservar**.

La pregunta clave antes de borrar: ¿hay un timestamp o un ID único de transacción que distinga un evento real de un error de carga? En nuestro caso, cada fila ya tiene `match_id` + `player_id`: dos filas con la misma combinación **sí** son un error (un jugador no puede tener dos aparaciones distintas en el mismo partido), a diferencia de un cliente comprando dos veces el mismo producto en días distintos.

> **Mapa mental del módulo**: explorá (`info()`, `isna().sum()`) → contextualizá (¿error o realidad del negocio?) → medí en porcentajes, no en absolutos → decidí (`dropna()`, imputar, o descartar la columna) → deduplicá según las claves de negocio que correspondan (acá, `match_id` + `player_id`).

---

## Módulo 2 — Transformaciones con `map` y `apply`

**Contexto**: con los datos ya limpios (Módulo 1), el siguiente paso es transformarlos — traducir códigos a etiquetas legibles, y calcular columnas nuevas a partir de reglas de negocio, sin escribir un solo `for`.

### ¿Por qué transformar datos? *(Filmina 11)*

Para la máquina, `"ESP"`, `"esp"` y `"España"` son tres categorías distintas. Transformar sirve para tres cosas:
- **Estandarizar**: unificar criterios (todas las variantes de país pasan a un único valor).
- **Enriquecer**: crear columnas derivadas (calcular el IVA a partir del precio).
- **Categorizar**: traducir números a etiquetas de negocio ("Gasto Alto", "Gasto Bajo").

### El método `map()`: la tabla de traducción *(Filmina 12)*

`.map()` se usa sobre una **Series** (una sola columna) y es ideal cuando hay una correspondencia clara, uno a uno — como un diccionario de traducción. **Error común**: si un valor de la columna no está como clave en el diccionario, `map()` lo convierte en `NaN` — hay que cubrir todos los casos posibles. `.map()` también acepta una función (no solo un diccionario) o incluso otra Series como argumento — pero en la práctica, el 90% de los usos es con un diccionario chico de traducción, que es además el caso donde `.map()` es más rápido que el equivalente con `.apply()`.

🎯 **Qué mostramos acá:** traducir la columna `preferred_foot` (`"Left"`/`"Right"`) a etiquetas en español, con un diccionario de dos entradas.

👉 **En Colab:**
```python
diccionario_pie = {"Left": "Izquierdo", "Right": "Derecho"}

df["pie_habil"] = df["preferred_foot"].map(diccionario_pie)
print(df["pie_habil"].value_counts())   # Derecho: 40.656 | Izquierdo: 13.944
```

**Línea por línea:**
- `diccionario_pie = {...}` → mapa de traducción uno a uno: cada valor posible de `preferred_foot` tiene su equivalente en español.
- `df["preferred_foot"].map(diccionario_pie)` → recorre la Series y reemplaza cada valor por su traducción según el diccionario; como cubrimos **las dos únicas** categorías que existen en la columna (`Left`/`Right`), no queda ningún `NaN`.
- `.value_counts()` → confirma el resultado y de paso valida que no se coló ningún valor inesperado.

### El método `apply()`: flexibilidad total *(Filmina 13)*

- **Sobre una Series**: transforma una columna con lógica más compleja que un simple mapeo (una función, no solo un diccionario).
- **Sobre un DataFrame con `axis=1`**: le pasa a la función la **fila completa** — ahí es donde brilla para reglas de negocio que combinan varias columnas a la vez. El default es `axis=0`, que le pasaría a la función **columna por columna completa** (útil para aplicar algo como `.sum()` propio a cada columna) — es fácil olvidarse el `axis=1` y obtener un resultado sin sentido, sin que Pandas avise con un error.

🎯 **Qué mostramos acá:** `.apply()` con una `lambda` sobre una sola columna, y `.apply(axis=1)` con una función `def` que combina dos columnas para una regla de negocio real: identificar apariciones de "riesgo disciplinario" (jugador que además de cometer varias faltas, ya fue amonestado en ese partido).

👉 **En Colab:**
```python
df["equipo_mayuscula"] = df["team"].apply(lambda x: x.upper())   # transformación simple con lambda

def calcular_riesgo(fila):
    if fila["fouls_committed"] >= 2 and fila["yellow_cards"] == 1:
        return "Riesgo disciplinario"
    return "Bajo"

df["riesgo"] = df.apply(calcular_riesgo, axis=1)   # axis=1 -> la función recibe la FILA completa
print(df["riesgo"].value_counts())   # Riesgo disciplinario: 540
```

**Línea por línea:**
- `df["team"].apply(lambda x: x.upper())` → aplica la función a **cada valor** de la columna; acá una `lambda` de una línea alcanza porque la lógica es simple (pasar a mayúsculas).
- `def calcular_riesgo(fila):` → una función que recibe una **fila entera** (una especie de diccionario de columna→valor), no un solo valor.
- `fila["fouls_committed"] >= 2 and fila["yellow_cards"] == 1` → acá sí se usa `and` de Python normal (no `&`), porque `fila["..."]` es un único valor escalar, no una Series completa.
- `df.apply(calcular_riesgo, axis=1)` → `axis=1` es la clave: le dice a `apply` que le pase **la fila** a la función, no columna por columna (que sería `axis=0`, el default). Por dentro, `apply(axis=1)` arma una Serie nueva por cada fila antes de pasarla a la función — por eso suele ser la variante más lenta de `.apply()`: para reglas simples que combinan pocas columnas, muchas veces conviene una condición vectorizada con `&`/`np.where` en vez de esta forma.
- El resultado real: 540 apariciones jugador-partido caen en "Riesgo disciplinario" sobre 54.600 — una regla que un `.map()` no podría resolver, porque depende de **dos** columnas a la vez.

### Diferencias clave: `map`, `apply` y `applymap` *(Filmina 14)*

| Método | Se aplica sobre | Uso principal |
|---|---|---|
| `map` | Series | Sustitución simple, con un diccionario. |
| `apply` | Series o DataFrame | Funciones más complejas, o lógica entre varias columnas (`axis=1`). |
| `applymap` | DataFrame completo | Aplicar una función a **todas** las celdas a la vez (ej. redondear todos los decimales). |

Un detalle de versiones: en Pandas moderno, `applymap` está marcado como *deprecated* a favor de `df.apply(func)` (que, cuando `func` se aplica elemento a elemento, hace exactamente lo mismo) — se lo menciona acá porque todavía aparece en muchísimo código y documentación en producción.

### Recomendaciones de oro para transformar *(Filmina 15)*

1. **Priorizá la vectorización**: si se puede resolver con `df["A"] + df["B"]` (como en el Módulo 0 con `np.where`), no uses `apply` — las operaciones directas son mucho más rápidas.
2. **Validá después de transformar**: `df.head()` y `df["columna"].value_counts()` confirman que la transformación hizo lo esperado (como hicimos arriba con `pie_habil` y `riesgo`).
3. **Cuidado con los nulos**: una función personalizada puede fallar con un `NaN` inesperado (por ejemplo, `.upper()` sobre un valor nulo). Conviene limpiar (Módulo 1) antes de transformar.

---

## Módulo 3 — Agrupar, Resumir y Comparar: GroupBy y Pivot Tables

**Contexto**: en el Módulo 0 ya usamos un `groupby` con una sola columna y una sola métrica. Acá profundizamos: varias agregaciones a la vez, agrupar por más de una columna, y la versión "en cruz" del mismo concepto — la `pivot_table`.

### De lo micro a lo macro *(Filmina 17)*

Una fila de nuestro dataset (un jugador, en un partido, con sus estadísticas) es un dato **micro**, aislado. Agrupar responde preguntas de negocio — "¿qué posición metió más goles?", "¿cómo varía el rendimiento según la instancia del torneo?" — con suma, promedio, conteo o máximo/mínimo. Sin agrupación, los datos son ruido; con ella, son información.

### El flujo Split-Apply-Combine *(Filmina 18)*

Concepto de **Hadley Wickham**, el modelo mental detrás de cada `groupby`:
1. **Split (dividir)**: Pandas separa el DataFrame en "cajas", una por cada valor de la columna agrupadora (ej. cada posición).
2. **Apply (aplicar)**: dentro de cada caja, se aplica una operación (sumar los goles de los `Forward`, y así con cada posición).
3. **Combine (combinar)**: Pandas une los resultados de cada caja en una tabla resumen, mucho más chica que la original.

Vale la pena notar que "Apply" acá es el concepto general (aplicar una operación a cada grupo), no necesariamente el método `.apply()` de Pandas del Módulo 2 — con un `groupby`, ese paso casi siempre se resuelve con `.mean()`, `.sum()` o `.agg()`, que son vectorizados y mucho más rápidos que pasarle una función propia con `.apply()` grupo por grupo.

### Selección de columnas y agregaciones comunes *(Filmina 19)*

**Regla de oro**: seleccioná la columna numérica **antes** de aplicar el resumen (no tiene sentido promediar una columna de texto). `.mean()`, `.sum()`, `.count()`, `.median()`, `.min()`, `.max()` son las agregaciones más usadas. Si se omite la selección de columna (`df.groupby("position").mean()` a secas), Pandas moderno directamente **descarta** las columnas no numéricas del resultado en vez de tirar error — conviene seleccionar explícitamente para no terminar con un resumen de columnas que no se pidieron.

### Agregaciones múltiples con `agg()` *(Filmina 20)*

A veces hace falta el total **y** el promedio a la vez, para comparar volumen contra rendimiento. `.agg([...])` recibe una lista de funciones y las aplica todas de una sola vez.

🎯 **Qué mostramos acá:** tres preguntas de negocio sobre los goles por posición, respondidas en una sola línea: total de goles, promedio por aparición, y cantidad de apariciones (para poner el promedio en contexto).

👉 **En Colab:**
```python
resumen_goles = df.groupby("position")["goals"].agg(["sum", "mean", "count"])
print(resumen_goles.round(2))
```

**Línea por línea:**
- `df.groupby("position")["goals"]` → agrupa las 54.600 filas en 4 posiciones, y selecciona la columna `goals` dentro de cada grupo.
- `.agg(["sum", "mean", "count"])` → aplica las tres funciones a la vez; el resultado es una tabla con una columna por cada una. `.agg()` también acepta un diccionario (`{"columna": ["sum", "mean"]}`) para pedir agregaciones distintas por columna cuando se resumen varias a la vez, y permite pasarle nombres propios a cada resultado (`.agg(total=("goals", "sum"))`) en vez de quedarse con `sum`/`mean` como nombre de columna.
- El resultado real: **Forward** suma 1.805 goles (promedio 0.14 por aparición) sobre 12.600 apariciones; **Defender** suma 353 (0.02) sobre 18.900; **Midfielder** 866 (0.05) sobre 16.800; **Goalkeeper** 0 goles, como es esperable. El `count` es lo que evita una lectura ingenua del promedio: los `Forward` tienen menos apariciones que los `Defender`, pero convierten muchas más veces por aparición.

### Agrupación por múltiples columnas *(Filmina 21)*

Pasar una **lista** de columnas al `groupby` crea una jerarquía — más nivel de detalle. Útil para detectar patrones que una sola columna esconde (ej. el rendimiento no es igual en un partido de Grupos que en una Final, y esa diferencia puede variar según la posición).

👉 **En Colab:**
```python
rating_detallado = df.groupby(["tournament_stage", "position"])["player_rating"].mean().round(2)
print(rating_detallado.head(8))
```

**Línea por línea:**
- `df.groupby(["tournament_stage", "position"])` → la lista `[...]` agrupa primero por instancia del torneo, y dentro de cada instancia, por posición — una jerarquía de dos niveles. El orden de la lista importa: agrupar por `["position", "tournament_stage"]` da los mismos números, pero organizados con la jerarquía invertida.
- `["player_rating"].mean()` → el promedio de rating dentro de cada combinación instancia+posición.
- El resultado queda con un **índice múltiple** (`MultiIndex`): cada fila combina una instancia y una posición. Para acceder a un valor puntual se encadenan las etiquetas de cada nivel — `resultado.loc["Final", "Forward"]` — y `.reset_index()` "aplana" ese índice de vuelta a columnas normales cuando conviene filtrar o graficar.

### Pivot Tables: el poder de las tablas dinámicas *(Filmina 22)*

Una `pivot_table` es la misma idea que un `groupby` de dos columnas, pero mostrada como una **matriz 2D** en vez de una lista vertical — igual que una tabla dinámica de Excel:
- **`index`**: la columna que va en las filas.
- **`columns`**: la columna que va en la parte superior.
- **`values`**: la columna numérica a resumir.
- **`aggfunc`**: la operación (por defecto, el promedio).

Una combinación `index`/`columns` que no tiene datos queda como `NaN` en la matriz — a diferencia de un `groupby` de dos columnas, que directamente **no genera esa fila** si esa combinación no existe. Es una diferencia útil: la `pivot_table` deja "a la vista" los huecos de la matriz completa, mientras que el `groupby` los oculta.

🎯 **Qué mostramos acá:** la misma información del punto anterior (rating promedio por instancia y posición), pero en formato cruzado — mucho más fácil de leer de un vistazo.

👉 **En Colab:**
```python
tabla_rating = df.pivot_table(
    index="tournament_stage", columns="position", values="player_rating", aggfunc="mean"
).round(2)
print(tabla_rating)
```

**Línea por línea:**
- `index="tournament_stage"` → cada instancia del torneo (Group Stage, Round of 32... Final) se convierte en una fila.
- `columns="position"` → cada posición se convierte en una columna: el cruce da una matriz de 7 filas × 4 columnas.
- `values="player_rating", aggfunc="mean"` → cada celda de la matriz es el promedio de `player_rating` para esa combinación puntual de instancia y posición.

**Un hallazgo real, mirando la tabla resultante**: los arqueros (`Goalkeeper`) tienen un rating promedio de ~2.0–2.1 en todas las instancias, muy por debajo de las demás posiciones (~3.7–4.0). No es que jueguen peor: es que **el 66,7% de las apariciones de arquero son de suplentes que no llegaron a jugar** (rating en 0, el hallazgo del Módulo 0), contra ~39% en el resto de las posiciones — porque cada plantel lleva 2 o 3 arqueros al torneo, pero solo uno juega por partido. Antes de comparar posiciones por rating, conviene filtrar `df[df["minutes_played"] > 0]` (Módulo 0) para no mezclar quien jugó con quien no.

### Errores comunes y mejores prácticas de GroupBy *(Filmina 23)*

| Error | Cómo evitarlo |
|---|---|
| **Confundir `count()` con `sum()`** | `count()` dice cuántas filas hay; `sum()` suma sus valores. ¿La pregunta es "cuántos" o "cuánto"? `count()`, además, solo cuenta valores **no nulos** — para contar filas sin importar nulos existe `.size()`, que suele confundirse con `count()` por el nombre parecido. |
| **Olvidar los nulos** | Pandas los ignora por defecto en `mean()`/`sum()` — engañoso si falta gran parte de una columna (por eso el Módulo 1 va **antes** que este). |
| **No resetear el índice** | La columna agrupada pasa a ser el índice del resultado. `.reset_index()` la devuelve a columna normal, necesario para volver a filtrar o graficar. |

```python
resumen_reseteado = df.groupby("position")["goals"].sum().reset_index()
```

---

## Módulo 4 — Fechas, Series Temporales y Resampling

**Contexto**: `match_date` viene como texto (`"2026-06-11"`) al cargar el CSV. El torneo completo va del **11 de junio al 31 de julio de 2026** (51 fechas distintas con partidos) — suficiente rango para ver de verdad qué hace un `resample()`.

### Conversión de texto a Datetime *(Filmina 26)*

Al cargar un CSV, las fechas se leen como texto simple (`object`), no como fechas reales. `pd.to_datetime()` convierte ese texto en objetos de fecha reales, con toda la lógica del calendario adentro (meses con distinta cantidad de días, años bisiestos, etc.) — algo que un `string` no sabe manejar por sí solo. Por defecto, `to_datetime()` infiere el formato fila por fila (lo cual puede ser lento en datasets grandes); si todas las fechas comparten el mismo formato, pasar `format="%Y-%m-%d"` explícito evita la inferencia y acelera bastante la conversión — además de eliminar la ambigüedad clásica entre `"01/02/2026"` como 1 de febrero o 2 de enero.

👉 **En Colab:**
```python
df["match_date"] = pd.to_datetime(df["match_date"])
print(df["match_date"].dtype)   # datetime64[...], ya no object
```

**Línea por línea:**
- `pd.to_datetime(df["match_date"])` → interpreta cada valor de texto como una fecha real y devuelve una Series de tipo fecha.
- `df["match_date"].dtype` → confirma el cambio: pasa de `object` (texto) a un `datetime64` real (la resolución exacta — `ns`, `us`— depende de la versión de Pandas, pero en cualquier caso ya es un tipo de fecha, no texto).

### El índice temporal: tu mejor aliado *(Filmina 27)*

La mejor práctica es convertir la columna de fechas en el **índice** del DataFrame. Eso habilita acceder a datos por período (`df["2026-06"]` trae todo junio) y es el requisito para poder hacer **Resampling** — internamente, un `DatetimeIndex` sabe ordenar y "binear" fechas en bloques de calendario (semanas, meses) de forma nativa, algo que un índice numérico común no puede hacer.

👉 **En Colab:**
```python
df_temporal = df.set_index("match_date").sort_index()   # sort_index: fechas en orden antes de operar

junio = df_temporal.loc["2026-06"]
print(f"Apariciones jugador-partido en junio: {len(junio)}")
```

**Línea por línea:**
- `df.set_index("match_date")` → la columna de fechas deja de ser una columna y pasa a ser el índice del DataFrame.
- `.sort_index()` → ordena las filas por fecha; imprescindible antes de cualquier operación temporal (ver "Errores comunes" más abajo). Con el índice ordenado, Pandas puede aplicar **búsqueda binaria** para localizar un rango de fechas en vez de recorrer fila por fila — es también una cuestión de performance, no solo de corrección.
- `df_temporal.loc["2026-06"]` → acceso por período: con un índice de fechas, `.loc["2026-06"]` trae directamente "todo junio de 2026", sin necesidad de armar un filtro booleano con `>=`/`<=`.

### Resampling: cambiando el "zoom" de tus datos *(Filmina 28)*

`resample()` es un `groupby` especializado en tiempo: agrupa automáticamente por período de calendario.

| Código | Frecuencia |
|---|---|
| `D` | Diario |
| `W` | Semanal |
| `ME` | Fin de mes (*Month End*) |
| `YE` | Fin de año (*Year End*) |

**Downsampling**: reducir la frecuencia (diario → semanal), resumiendo con una estadística. **Upsampling**: aumentar la frecuencia (mensual → diario), decidiendo cómo rellenar los huecos nuevos que aparecen (típicamente con `.ffill()` — repetir el último valor conocido — o `.interpolate()` para estimar un valor intermedio).

🎯 **Qué mostramos acá:** downsampling de partido-por-partido a semana-por-semana, contando partidos únicos y sumando goles — dos preguntas de negocio distintas sobre la misma serie temporal.

👉 **En Colab:**
```python
partidos_por_semana = (
    df_temporal.drop_duplicates("match_id")   # cada partido cuenta una sola vez, no una por jugador
    .resample("W")
    .size()
)
print(partidos_por_semana.head())

goles_por_semana = df_temporal.resample("W")["goals"].sum()
print(goles_por_semana.head())
```

**Línea por línea:**
- `drop_duplicates("match_id")` → antes de contar partidos, sacamos las filas repetidas por jugador (cada partido tiene ~22 filas, una por jugador); sin este paso, "partidos por semana" en realidad contaría apariciones.
- `.resample("W")` → agrupa el índice temporal en bloques semanales, igual que un `groupby` agruparía por categoría. Por defecto, cada semana se etiqueta con su **domingo** (fin de semana ISO); `resample("W-MON")` cambiaría el corte de semana a los lunes, si el negocio lo necesita.
- `.size()` → cuenta cuántas filas (partidos únicos) cayeron en cada semana.
- `df_temporal.resample("W")["goals"].sum()` → misma lógica de agrupación semanal, pero ahora sumando los goles de **todas** las apariciones (no hace falta deduplicar: cada gol de cada jugador debe contarse).
- La primera semana completa (que arranca el 15/06) tiene 448 goles sobre 153 partidos — casi 3 goles por partido, un dato que solo aparece al agrupar por tiempo.

### Errores comunes al trabajar con fechas *(Filmina 29)*

| Error | Cómo evitarlo |
|---|---|
| **No ordenar los datos** | Usar `.sort_index()` antes de remuestrear; un índice desordenado da resultados impredecibles. |
| **Operar con texto** | Sumar o comparar fechas sin convertir a `datetime` primero lanza un error de tipos — por eso el Módulo 4 empieza con `pd.to_datetime()`. |
| **Confundir sumas con promedios** | ¿La pregunta es el total de goles de la semana (`.sum()`) o el promedio de goles por partido (`.mean()`)? Cada una responde algo distinto. |

---

## Módulo 5 — Manipulación de Datos: Pandas (Síntesis)

**Contexto**: cierre conceptual de la clase — de dónde viene Pandas, cómo se conecta todo lo visto en un pipeline profesional, y las dos piezas que faltaban: combinar tablas (`merge`/`concat`) y los errores más comunes incluso entre gente con experiencia.

### ¿Por qué existe Pandas? *(Filmina 31)*

Pandas fue creada por **Wes McKinney** en el sector financiero, hacia 2008. Los analistas necesitaban la flexibilidad de Python con la estructura de una base de datos — el nombre viene de **"Panel Data"**, un término de econometría. La ventaja frente a una hoja de cálculo: en Excel hay que repetir clics y fórmulas con cada archivo nuevo; en Pandas se escribe el "recetario" una sola vez, y funciona igual con 100 filas o con 1.000.000, sin errores humanos de por medio.

### Los pilares de Pandas: DataFrame y Series *(Filmina 32)*

- **DataFrame**: estructura bidimensional. Eje 0 (filas) = registros/observaciones; eje 1 (columnas) = variables/atributos. La distinción entre `axis=0` y `axis=1` que aparece en `.drop()`, `.apply()` o `.sum()` es exactamente esta: `axis=0` "recorre hacia abajo" (opera columna por columna, a través de las filas), `axis=1` "recorre hacia el costado" (opera fila por fila, a través de las columnas).
- **Series**: una sola columna, unidimensional, **con índice**. Error común: pensarla como una simple lista — una lista de valores no sabe a qué fila pertenece cada uno; una Series sí, gracias a sus etiquetas.

### El flujo de trabajo profesional (pipeline) *(Filmina 33)*

Las tres fases que ya recorrimos en esta clase, ahora nombradas como proceso:
1. **Ingesta e Inspección**: `head()`/`tail()` para una primera probada, `info()` para tipos y nulos por columna (Módulo 0 y 1).
2. **Limpieza**: decidir entre `dropna()` (borrar) o `fillna()` (rellenar sin perder información) (Módulo 1).
3. **Transformación y Vectorización**: `df["precio"] * 0.9` en vez de un `for` — miles de veces más rápido, apoyado en NumPy (Módulo 0, 2 y el "mito del `for`" de la Clase 03).

En la práctica, estas tres fases rara vez son un pasaje único de principio a fin: es normal volver de la Transformación a la Limpieza al descubrir un problema nuevo (una columna transformada que revela outliers, por ejemplo) — el pipeline es más un ciclo iterativo que una línea recta.

### Agregación: Split-Apply-Combine en la industria *(Filmina 34)*

El mismo flujo del Módulo 3 (separar, aplicar, combinar) es el que usan firmas como **J.P. Morgan** o **Goldman Sachs** para analizar miles de transacciones por segundo y obtener promedios diarios o máximos históricos por activo financiero — la escala cambia, la lógica es exactamente la misma.

### Combinación de fuentes: Merge vs. Concat *(Filmina 35)*

- **`merge`**: como el `JOIN` de SQL. Se busca una columna en común y se fusionan las tablas **lateralmente** — agrega columnas nuevas sobre los mismos registros.
- **`concat`**: como apilar hojas de papel. Se pone una tabla debajo de otra — agrega **filas** nuevas del mismo tipo. También sirve para apilar **columnas** en vez de filas, con `axis=1` — menos común, pero útil cuando dos tablas ya comparten el mismo índice y solo hace falta pegarlas una al lado de la otra.

🎯 **Qué mostramos acá:** un `merge` real, agregando la confederación de cada equipo a partir de una tabla de referencia chica (a propósito, incompleta — para mostrar qué pasa con los equipos que no aparecen); y un `concat`, dividiendo el dataset en dos mitades cronológicas y volviendo a unirlas.

👉 **En Colab:**
```python
# Tabla de referencia chica, A PROPÓSITO incompleta (solo 10 de los 48 equipos)
df_confederaciones = pd.DataFrame({
    "team": ["Argentina", "Brazil", "France", "Spain", "Germany",
             "Japan", "Morocco", "Mexico", "Nigeria", "Qatar"],
    "confederacion": ["CONMEBOL", "CONMEBOL", "UEFA", "UEFA", "UEFA",
                       "AFC", "CAF", "CONCACAF", "CAF", "AFC"],
})

df_con_confederacion = pd.merge(df, df_confederaciones, on="team", how="left")
print(df_con_confederacion["confederacion"].isna().sum())   # equipos sin match en la tabla chica

# concat: partimos el dataset en dos mitades cronológicas y las volvemos a unir
primera_mitad = df[df["match_date"] < "2026-07-01"]
segunda_mitad = df[df["match_date"] >= "2026-07-01"]
df_reunido = pd.concat([primera_mitad, segunda_mitad])
print(len(df_reunido) == len(df))   # True -> concat no perdió ni duplicó filas
```

**Línea por línea:**
- `df_confederaciones` → una tabla de referencia mínima, con solo 10 de los 48 equipos — a propósito, para que el `merge` deje algo sin matchear.
- `pd.merge(df, df_confederaciones, on="team", how="left")` → `on="team"` es la columna en común; `how="left"` conserva **todas** las filas de `df`, tengan o no confederación asignada. Existen también `how="right"` (prioriza la tabla de la derecha), `how="inner"` (solo lo que matchea en ambas — el default) y `how="outer"` (todo, de las dos tablas, con `NaN` donde falte); la elección depende de qué DataFrame es "la base" que no se puede perder.
- `df_con_confederacion["confederacion"].isna().sum()` → cuenta cuántas apariciones quedaron con `NaN` en `confederacion`: son las de los 38 equipos que no estaban en la tabla chica — el mismo patrón de `how="left"` que se vio con `merge` en la Clase 03.
- `primera_mitad` / `segunda_mitad` → dos recortes del mismo DataFrame por fecha, sin superposición.
- `pd.concat([primera_mitad, segunda_mitad])` → apila una tabla debajo de la otra; como no hay filas repetidas ni perdidas, el total coincide exactamente con el original.

### Errores comunes y mejores prácticas *(Filmina 36)*

| Error | Detalle |
|---|---|
| **La trampa del índice** | No es una columna normal, es el sistema de direcciones de las filas. Decidí si conservarlo al hacer `.reset_index()`. |
| **Copias vs. Vistas** | Filtrar a veces da una "ventana" a la tabla original, no una copia independiente (Pandas avisa con `SettingWithCopyWarning`). Usá `.copy()` explícito si vas a modificar el resultado de un filtro. |
| **No tratar nulos al inicio** | Sin gestionar los `NaN` (Módulo 1), los cálculos estadísticos pueden salir sesgados o directamente fallar. |
| **Alineación automática** | Al sumar dos Series, Pandas alinea por **etiqueta** de índice, no por posición física — una ventaja una vez que se entiende, pero confunde a quien viene de listas de Python. Si dos Series tienen índices que no coinciden del todo, el resultado trae `NaN` en las etiquetas que no matchean en ambas — Pandas nunca "adivina" alineando por posición como haría una lista. |

```python
# Patrón seguro: .copy() explícito antes de modificar un recorte
delanteros = df[df["position"] == "Forward"].copy()
delanteros["goles_por_90"] = delanteros["goals"] / (delanteros["minutes_played"] / 90)
```

### Aplicaciones en la industria real *(Filmina 37)*

- **Comercio electrónico**: detectar outliers en precios que podrían indicar errores en la carga de productos.
- **Logística**: calcular tiempos de entrega promedio uniendo tablas de rutas, tráfico y conductores (`merge`).
- **Salud**: limpiar registros de pacientes — unificar formatos de fechas y manejar datos clínicos faltantes en ensayos (Módulos 1 y 4, aplicados a un dominio distinto).

---

## Módulo 6 — Pre-Entrega 4: Estructura Inicial del Dataset del Proyecto

**Origen de esta consigna**: es el **Checkpoint 1 / Pre-entrega 4** oficial de la plataforma ("Pre-entrega: Estructura inicial del dataset del proyecto"). Nace como cierre del módulo de NumPy y Pandas (Clase 03), pero como en esa clase no llegamos a ver todo el contenido necesario, se corrió a Clase 04 — por eso vive acá y no en la guía de Clase 03. **No confundir con la Pre-entrega de Clase 05** ("EDA Visual del Dataset del Proyecto"): son dos checkpoints distintos y consecutivos — esta (Pre-entrega 4) sanea la estructura del dataset; la de Clase 05 lo visualiza una vez ya saneado.

**Contexto y posición en la cadena**: en Nivelación y Fundamentos de Python preparaste el entorno y la lógica de programación; en este primer checkpoint aplicás Pandas y NumPy para "darle una identidad" a tu propio Proyecto Final. Lo que construyas hoy no es el análisis definitivo — es la base saneada y explorada sobre la que en el próximo módulo vas a aplicar visualización, y más adelante, Machine Learning.

**Objetivo de aprendizaje**: consolidar la capacidad de cargar un dataset real, inspeccionar su estructura interna, validar tipos de datos y hacer un saneamiento inicial con funciones de Pandas y NumPy.

**Qué construir**: un informe de inspección inicial en un Jupyter Notebook, que muestre el proceso de carga y el "diagnóstico" de tu dataset. El objetivo no es tener datos perfectos todavía — es tener un mapa claro de qué datos tenés y qué problemas (nulos, tipos erróneos) vas a tener que resolver.

**Temas a cubrir en la entrega:**

| Bloque | Qué incluye |
|---|---|
| **Carga del Dataset** | `read_csv` (o similar) para traer tus datos al entorno. |
| **Estructura Base** | Reporte de filas y columnas (`.shape`). |
| **Diagnóstico Estructural** | `.info()` para identificar columnas con nulos y `dtypes` que no coincidan con la realidad (ej. números que Pandas lee como texto). |
| **Resumen Numérico** | `.describe()` para entender los rangos de tus variables. |
| **Saneamiento Inicial** | Al menos **3 filtros o selecciones booleanas** que "recorten" o limpien el dataset, y eliminar al menos una columna que no aporte al problema de negocio. |

**Errores comunes a evitar:**
- **Ignorar los tipos de datos**: es muy común intentar sumar una columna y que tire error porque Pandas la leyó como `object` (texto). Revisá siempre `.info()`.
- **No guardar los cambios**: filtrar con `df[df["edad"] > 18]` no modifica el original a menos que lo reasignes (`df = df[df["edad"] > 18]`).

**Conexión con el entregable final**: este dataset "saneado" es la materia prima para el resto del curso — en el próximo módulo se usa el mismo archivo para gráficos con Matplotlib y Seaborn. Una estructura sólida hoy evita que las visualizaciones de mañana "mientan".

**Entregable**: un PDF con la exportación del notebook (código, resultados y comentarios visibles), incluyendo diagnóstico de nulos, `dtypes`, `describe()` y los filtros realizados. Nombre sugerido: `Apellido_Nombre_Checkpoint1.pdf` (en Jupyter: *File → Download as → PDF via HTML*, o imprimiendo la página como PDF).

Ejemplo completo abajo, sobre `fifa_world_cup_2026_player_performance.csv` — la misma estructura que vas a usar con el dataset de tu propio proyecto.

### Preparación

Tené tu dataset (CSV o Excel) en la misma carpeta donde vas a crear el notebook — evita rutas absolutas que solo funcionan en tu computadora.

### Carga e Inspección

👉 **En Colab:**
```python
import pandas as pd

df = pd.read_csv("fifa_world_cup_2026_player_performance.csv")
display(df.head())

print(f"Dimensiones: {df.shape}")   # (54600, 75)
df.info()
```

**Línea por línea:**
- `pd.read_csv(...)` → carga el dataset completo en `df`, la variable que vamos a usar en todo el checkpoint.
- `df.head()` → inspección visual: ¿las columnas se ven como esperabas, sin corrimientos ni texto pegado? Un problema típico que `.head()` deja ver de inmediato es un CSV mal delimitado — si el separador real es `;` y se cargó como `,`, todo el contenido aparece amontonado en una sola columna.
- `df.shape` → volumen de datos: 54.600 filas, 75 columnas. Es una **tupla**, no una función — por eso se escribe `df.shape` sin paréntesis, a diferencia de `df.info()`.
- `df.info()` → columna por columna, cuántos valores no nulos tiene y de qué `dtype` es. Acá conviene revisar con ojo crítico: `market_value_eur` es `int64` (correcto, es un número), `match_date` es `object` porque todavía no la convertimos a fecha (eso pasa recién en el Módulo 4) — nada aparece como texto por error, a diferencia de un dataset real donde un símbolo de moneda o una coma de miles puede "ensuciar" una columna numérica y forzarla a `object`.

### Perfilado Inicial

👉 **En Colab:**
```python
nulos_por_columna = df.isnull().sum()
print(f"Total de valores nulos: {nulos_por_columna.sum()}")

display(df.describe())
```

**Línea por línea:**
- `df.isnull().sum()` → nulos por columna; `.sum()` sobre eso da el total del dataset — acá da `0`, este archivo viene sin nulos.
- `df.describe()` → estadísticas (`count`, `mean`, `std`, `min`, percentiles, `max`) de cada columna numérica. Por defecto **ignora** las columnas de texto/categóricas; para incluirlas (con `count`, `unique`, `top`, `freq` en vez de las estadísticas numéricas) hace falta `df.describe(include="all")`. Mirando la tabla real: `age` va de 17 a 39 años (rango de jugadores de un Mundial, tiene sentido); `market_value_eur` va de ~529 mil a 200 millones de euros, con una media (~20M) muy por debajo del máximo — señal de que unas pocas estrellas empujan el promedio hacia arriba.

### Saneamiento y Selección

👉 **En Colab:**
```python
jugaron = df[df["minutes_played"] > 0]                                              # filtro 1: excluye suplentes que no jugaron
delanteros_caros = df[(df["position"] == "Forward") & (df["market_value_eur"] > 50_000_000)]  # filtro 2: condición combinada
fase_eliminatoria = df[df["tournament_stage"] != "Group Stage"]                     # filtro 3: todo lo que no es fase de grupos

df_reducido = df.drop(columns=["jersey_number"])   # eliminamos una columna que no aporta al análisis

print(f"Apariciones con minutos jugados: {len(jugaron)}")
print(f"Delanteros con valor > 50M€: {len(delanteros_caros)}")
print(f"Apariciones en fase eliminatoria: {len(fase_eliminatoria)}")
```

**Línea por línea:**
- `df["minutes_played"] > 0` → primer filtro: descarta las apariciones de jugadores convocados que no llegaron a jugar (el hallazgo del Módulo 0).
- `(df["position"] == "Forward") & (df["market_value_eur"] > 50_000_000)` → segundo filtro, con `&` combinando dos condiciones sobre Series (nunca `and` acá).
- `df["tournament_stage"] != "Group Stage"` → tercer filtro: todo lo que es instancia eliminatoria (Round of 32 en adelante). Sería equivalente escribirlo como `~(df["tournament_stage"] == "Group Stage")`, pero `!=` es más directo cuando la condición es una sola comparación, sin necesidad de negar un paréntesis.
- `df.drop(columns=["jersey_number"])` → elimina una columna irrelevante para un análisis de rendimiento (el número de camiseta no aporta nada analíticamente); devuelve un DataFrame nuevo, por eso se reasigna a `df_reducido` — si no reasignás, el original queda intacto (ver "Errores comunes a evitar").

### Reflexión

> **¿Qué problemas encontré en los datos?** Ninguno grave: `fifa_world_cup_2026_player_performance.csv` viene con los 75 `dtypes` correctos y sin valores nulos — la única transformación pendiente es convertir `match_date` de texto a fecha real, que se resuelve en el Módulo 4. El mayor punto de atención no es de calidad de datos sino de interpretación: cada fila es una *aparición* jugador-partido, no un jugador único, así que cualquier conteo tiene que aclarar sobre qué unidad está parado.
>
> **¿Cuáles van a ser mis variables clave?** `player_rating` y `market_value_eur` para medir rendimiento y valor; `position` y `tournament_stage` para segmentar comparaciones; `minutes_played` como filtro obligatorio antes de calcular cualquier promedio de rendimiento, para no mezclar titulares con suplentes que no jugaron.

### Exportación

En Jupyter: *File → Download as → PDF via HTML* (o imprimir la página como PDF). Nombre del archivo: `Apellido_Nombre_Checkpoint1.pdf`.
