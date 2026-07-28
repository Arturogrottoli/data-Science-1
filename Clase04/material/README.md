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
- [Módulo 6 — Pre-Entrega: Limpieza y Análisis Exploratorio](#módulo-6--pre-entrega-limpieza-y-análisis-exploratorio)

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
| 6 | Pre-Entrega: Limpieza y Análisis Exploratorio | 38–41 | Módulo 6 | Consigna evaluable: nulos, duplicados, análisis de impacto e informe |

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
- **1.248 jugadores únicos**, **48 equipos**, **1.050 partidos únicos**.
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
- `df["market_value_eur"].to_numpy()` → toma la Serie (columna) y devuelve el `ndarray` que tiene por dentro, sin las etiquetas de índice de Pandas.
- `valores_mercado / 1_000_000` → división vectorizada: cada uno de los 54.600 valores se divide por un millón en una sola operación, sin recorrer la lista.
- `.mean()`, `.max()` → métodos de NumPy sobre el array resultante.
- `np.percentile(valores_en_millones, 90)` → valor de mercado por debajo del cual está el 90% de las apariciones jugador-partido; el 10% restante son los jugadores más valiosos del torneo.

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
- `df[["player_name", "team", "position"]]` → **doble** corchete (una lista de nombres adentro) selecciona varias columnas como DataFrame (2D).

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
- `df.iloc[0:5, 0:4]` → `0:5` son las filas con posición 0 a 4 (5 filas), `0:4` las columnas con posición 0 a 3 (4 columnas); ambos por número, no por nombre.
- `df["goals"] > 2` → una Serie de booleanos (`True`/`False`), una por fila, según si esa aparición tuvo más de 2 goles.
- `df.loc[condición, ["player_name", "team", "goals"]]` → `.loc` recibe **primero** el filtro (qué filas) y **después** la lista de columnas por nombre (qué columnas); el resultado son las filas donde la condición dio `True`, mostrando solo esas tres columnas.

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
- `(df["position"] == "Forward") & (df["market_value_eur"] > 50_000_000)` → dos condiciones combinadas con `&` (Y lógico); cada una entre paréntesis, obligatorio para que Python respete el orden de evaluación.
- `~(df["tournament_stage"] == "Final")` → `~` niega la condición completa: se queda con todo lo que **no** es la Final (equivalente a `!=`, pero útil cuando la condición ya es compleja).

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
- `df["position"].value_counts()` → tabla de frecuencias: cada posición y cuántas apariciones jugador-partido tiene.
- `df["team"].nunique()` → cuenta los 48 equipos distintos en la columna `team`, sin listarlos.
- `df.sort_values("market_value_eur", ascending=False)` → reordena **todo** el DataFrame de mayor a menor valor de mercado; `ascending=False` es necesario porque el default es ascendente.
- `.head(5)` → se queda con las primeras 5 filas del resultado ya ordenado: los 5 jugadores-partido con mayor valor de mercado.

### 0.6 Preprocesamiento: nulos, `rename()` y `drop()`

Repaso del Módulo 6 de Clase 03 (nulos), más dos comandos clásicos que faltaban: `rename()` para ponerle otro nombre a una columna sin tocar los datos, y `drop()` para sacar filas o columnas completas.

👉 **En Colab:**
```python
print(df.isnull().sum().sum())    # 0 -> este dataset viene sin nulos

df = df.rename(columns={"goals": "goles", "assists": "asistencias"})   # renombrar para trabajar en español
df = df.drop(columns=["jersey_number"])                                 # sacar una columna que no vamos a usar

print(df.columns[:5].tolist())
```

**Línea por línea:**
- `df.isnull().sum().sum()` → el primer `.sum()` cuenta nulos por columna, el segundo suma esos totales en un único número: `0` en este dataset.
- `df.rename(columns={...})` → recibe un diccionario `{"nombre viejo": "nombre nuevo"}`; devuelve un DataFrame nuevo con las columnas renombradas (no modifica `df` in place salvo que se reasigne, como acá).
- `df.drop(columns=["jersey_number"])` → `columns=[...]` elimina columnas completas (para eliminar filas se usa `index=[...]`, visto ya en la Clase 03).

### 0.7 Agregación básica con `groupby` (puente al Módulo 3 de esta clase)

Repaso corto del `.agg()` de la Clase 03, Módulo 7, con el caso más simple posible: **una** columna para agrupar y **una** métrica. El `groupby` con múltiples agregaciones a la vez y las `pivot_table` (la versión "en cruz" del mismo concepto) se profundizan en el Módulo 3 de esta clase — acá solo repasamos la base para que ese salto no sea desde cero.

👉 **En Colab:**
```python
valor_por_posicion = df.groupby("position")["market_value_eur"].mean().sort_values(ascending=False)
print((valor_por_posicion / 1_000_000).round(2))
```

**Línea por línea:**
- `df.groupby("position")` → agrupa las 54.600 filas en 4 grupos, uno por posición.
- `["market_value_eur"].mean()` → dentro de cada grupo, calcula el promedio de esa única columna.
- `.sort_values(ascending=False)` → ordena los 4 resultados de mayor a menor.
- El resultado real: **Forward** (≈27,2M€) > **Midfielder** (≈23,0M€) > **Defender** (≈15,4M€) > **Goalkeeper** (≈12,2M€) — coincide con la lógica del mercado de pases, donde los puestos ofensivos suelen cotizar más caro.

### 0.8 Sinergia NumPy + Pandas (repaso breve)

Cierre del módulo, repasando la idea central del Módulo 8 de Clase 03: Pandas está construido sobre NumPy, y conviene resolver con operaciones vectorizadas antes que con un `for` o incluso antes que con `.apply()` (que se introduce recién en el Módulo 2 de esta clase). `np.where(condición, si_true, si_false)` es el ejemplo típico: una versión vectorizada de un `if/else` aplicado a toda una columna de una sola vez.

👉 **En Colab:**
```python
df["tuvo_gol"] = np.where(df["goles"] > 0, "Sí", "No")   # if/else vectorizado, sin for ni apply
print(df["tuvo_gol"].value_counts())
```

**Línea por línea:**
- `np.where(df["goles"] > 0, "Sí", "No")` → evalúa la condición para las 54.600 filas a la vez; donde es `True` pone `"Sí"`, donde es `False` pone `"No"`. Devuelve un `ndarray`, que Pandas acepta directamente como nueva columna.
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

Un detalle que sorprende la primera vez: `np.nan == np.nan` da **`False`**. Por eso no se puede filtrar nulos con `==`; existen `isna()`/`isnull()`, hechas específicamente para esto.

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

`info()` da el panorama general (columna por columna, cuántos valores "Non-Null" hay); `isna()`/`isnull()` (son alias, hacen lo mismo) devuelven la máscara booleana completa — útil para filtrar, inútil para "mirar" en un dataset de miles de filas.

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
- `df.copy()` → trabajamos sobre una copia; nunca se simulan fallas sobre el DataFrame original.
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

Como `True` vale `1` y `False` vale `0`, `df.isna().sum()` cuenta nulos por columna sin necesidad de un `if`. Dividido por `len(df)` y multiplicado por 100, da el porcentaje — la métrica que realmente importa (50 nulos en 60 filas es un desastre; 50 nulos en 54.600, no).

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

- **`duplicated()`**: marca `True` las filas que **ya aparecieron antes**; por defecto compara **todos** los valores de la fila.
- **`subset=["columna"]`**: busca duplicados mirando solo una columna (por ejemplo, un ID que debería ser único).
- **`drop_duplicates()`**: limpia el DataFrame; `keep="first"` o `keep="last"` decide cuál copia conservar.

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

- **Eliminar (`dropna()`)**: la opción más drástica. Válida cuando sobran datos y perder un 2% no afecta estadísticamente, o cuando falta justo la columna "etiqueta" que se quiere predecir.
- **Imputar (rellenar)**: **media** si la distribución es normal y sin outliers; **mediana** si hay valores extremos (más robusta); **moda** para columnas categóricas; **valor constante** (`"Desconocido"`, `0`) cuando conviene conservar la fila sin inventar un número.

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
