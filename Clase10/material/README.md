# Clase 10: Repaso Final — Guía Completa para el Docente

Esta guía acompaña a `Clase10.html` (29 filminas: portada + 3 por cada una de las Clases 01 a 09 + cierre). Es la última clase del curso, así que el objetivo no es enseñar contenido nuevo sino **reactivar** lo ya visto, con suficiente profundidad como para responder preguntas si alguien pide un repaso puntual de algún tema.

Cada filmina de "Temas" (tiles) trae una frase corta por concepto — acá se explica cada una con más contexto del que entra en una tarjeta. Cada filmina de "profundidad" (tabla o código) tiene su propio desarrollo.

> Para un repaso más detallado de cualquier clase puntual, cada `ClaseN` tiene su propia guía docente en `material/README.md` (o `README.md` en la raíz de la carpeta) — esta guía es un repaso rápido, no reemplaza a esas.

---

## Mapa rápido: Filminas ↔ Clases

| Filminas | Clase | Eje central |
|---|---|---|
| 01 | — | Portada |
| 02–04 | Clase 01 | Transformación Digital e Industria 4.0 |
| 05–07 | Clase 02 | Fundamentos de Python |
| 08–10 | Clase 03 | NumPy y Pandas |
| 11–13 | Clase 04 | Manipulación de Datos con Pandas |
| 14–16 | Clase 05 | Visualización de Datos |
| 17–19 | Clase 06 | EDA y Estadística Descriptiva |
| 20–22 | Clase 07 | Pipelines Reproducibles y Casos de ML |
| 23–25 | Clase 08 | Aprendizaje Supervisado en Práctica |
| 26–28 | Clase 09 | Aprendizaje No Supervisado |
| 29 | — | Cierre |

**Qué decir al arrancar:**

> "Hoy no vemos contenido nuevo. Recorremos las nueve clases que ya dimos, clase por clase, para que quede todo conectado antes de cerrar la cursada. Si en algún punto tienen dudas puntuales de alguna clase, este es el momento de preguntar."

---

## Filmina 01 — Portada

Sin contenido propio — es el título de toda la clase: "Repaso Final: de la Industria 4.0 al Aprendizaje No Supervisado". Vale la pena remarcar en voz alta el arco completo que se recorrió: se empezó hablando de fábricas y sensores (Industria 4.0) y se terminó entrenando modelos que agrupan clientes sin decirles cómo (Aprendizaje No Supervisado) — todo ese camino se apoya en las mismas herramientas de base (Python, NumPy, Pandas) aprendidas en las primeras clases.

---

## CLASE 01 — Transformación Digital e Industria 4.0 *(Filminas 02–04)*

**Filmina 02 — División.** Da el pie: por qué los datos son el activo central de la Industria 4.0.

### Filmina 03 — Lo que vimos

**Qué decir (ampliando cada tile):**

- **9 Elementos de la Industria 4.0:** la clase original los presentó como las piezas que arman la fábrica y la empresa digital de hoy. Vale la pena nombrar algunos concretos para que no quede abstracto: **IoT** (sensores en una máquina que miden temperatura o vibración en tiempo real), **Big Data** (esos sensores generan más datos por minuto de los que un humano podría revisar), **Cloud Computing** (donde se procesa y guarda todo eso — el mismo Google Colab que usamos es, en chico, esta idea), **Ciberseguridad** (proteger esos datos y sistemas conectados), **Robótica Avanzada** (cobots — robots colaborativos — que trabajan junto a operarios humanos, no en jaulas separadas como antes), y **Manufactura Aditiva** (impresión 3D). No hace falta recitar los 9 de memoria; lo que importa es la idea de fondo: la Industria 4.0 no es una sola tecnología, es la combinación de varias funcionando juntas, y todas terminan generando o consumiendo datos.
- **Ciclo de Vida de un Proyecto de Data Science:** el mapa que se repitió, en distintas formas, en cada clase siguiente. Sus etapas clásicas son: **Definición del problema de negocio** (¿qué pregunta queremos responder?), **Recolección de datos** (¿de dónde salen?), **Limpieza y preprocesamiento** (Clases 03, 04 y 06), **Análisis exploratorio y visualización** (Clase 05), **Modelado** (Clases 08 y 09), y **Comunicación de resultados / toma de decisión** (volver al negocio con una respuesta). Vale la pena señalar que este ciclo es literalmente la estructura de los Ejemplos 1 y 2 de esta misma clase (Clase 10): arrancan con una pregunta de negocio ("¿qué empresas se quedan en Veeqo?", "¿qué hace exitoso a un videojuego?") y terminan con un modelo evaluado y una recomendación.
- **Entorno de Trabajo: Google Colab:** el lugar físico (virtual) donde se programó toda la cursada. La ventaja concreta frente a instalar Python en la propia computadora: no hay que configurar nada, el notebook corre en un servidor de Google con las librerías más usadas (pandas, numpy, sklearn) ya instaladas, se puede acceder desde cualquier computadora con navegador, y es gratis para el uso que le dimos. La contracara es que los archivos que se suben (como los CSV de los ejemplos) no persisten entre sesiones a menos que se guarden en Drive — por eso en los dos Ejemplos de Clase 10 el primer paso siempre es volver a subir el dataset.
- **Tipos de Datos en Python:** simples (`int` para enteros, `float` para decimales, `str` para texto, `bool` para verdadero/falso) y estructurados (`list` para colecciones ordenadas y modificables, `tuple` para colecciones ordenadas e inmutables, `dict` para pares clave-valor, `set` para colecciones sin duplicados). Es el primer vocabulario técnico del curso — antes de poder decir "esta columna tiene valores nulos" hace falta poder decir "esta variable es de tipo X". Se profundiza mucho más en la Clase 02 (Filmina 06), pero la primera exposición fue acá.

**Nota:** los datos son el activo central de la Industria 4.0 — todo lo que vino después en el curso (Python, NumPy, Pandas, visualización, ML) construye sobre esa idea de que el dato, bien tratado, genera valor. Una fábrica con sensores que no analiza esos datos tiene la misma información que una sin sensores; el valor está en el análisis, no en la sola recolección.

### Filmina 04 — Las 4 Revoluciones Industriales

**Qué decir:**

| Era | Motor del cambio |
|---|---|
| **1.0** — Fines s. XVIII-XIX | Máquina de vapor y mecanización |
| **2.0** — Fines s. XIX-XX | Electricidad y producción en masa |
| **3.0** — 2ª mitad s. XX | Electrónica y automatización |
| **4.0** — Siglo XXI (hoy) | Datos, IoT e Inteligencia Artificial |

Cada revolución no reemplazó del todo a la anterior — se le sumó. Hoy conviven máquinas eléctricas (2.0), líneas automatizadas (3.0) y sensores conectados generando datos en tiempo real (4.0) en la misma fábrica. La Industria 4.0 es, en el fondo, la capa de datos que se le agrega a todo lo que ya existía.

**Preguntar a la clase:** ¿qué ejemplo de su vida cotidiana o laboral encaja en cada una de las 4 revoluciones?

---

## CLASE 02 — Fundamentos de Python *(Filminas 05–07)*

**Filmina 05 — División.** El vocabulario básico de programación que todo lo demás en el curso da por sabido.

### Filmina 06 — Lo que vimos

**Qué decir (ampliando cada tile):**

- **Variables y Tipos:** escalares (`int`, `float`, `str`, `bool`) y los operadores básicos (aritméticos `+ - * /`, de comparación `> < ==`, lógicos `and or not`). Una variable es simplemente un nombre que apunta a un valor guardado en memoria — `edad = 25` reserva espacio, guarda el 25, y le cuelga la etiqueta `edad`. El tipo importa porque determina qué operaciones son válidas: `"25" + 5` tira error porque no se puede sumar texto con número, aunque a simple vista "se vean parecidos". Es el punto de partida literal del curso — sin esto, nada de lo que sigue (ni una columna de un DataFrame, que no es otra cosa que muchas variables del mismo tipo) tiene sentido.
- **Colecciones a Fondo:** listas (`[1, 2, 3]`, ordenadas y mutables — se pueden modificar después de creadas), tuplas (`(1, 2, 3)`, ordenadas pero inmutables — una vez creadas no cambian), y diccionarios (`{"nombre": "Ana", "edad": 30}`, pares clave-valor, se accede por nombre en vez de por posición). La elección de cuál usar depende de tres preguntas: ¿los datos necesitan mantener un orden?, ¿se van a modificar después?, ¿tiene sentido buscarlos por una clave con nombre (como en un diccionario) en vez de por posición numérica? Un registro de sensor con `{"id": "TERM-04", "temperatura": 82.5}` es un uso típico de diccionario — accedés a `evento["temperatura"]` sin tener que recordar en qué posición quedó ese dato.
- **Control de Flujo:** `if/elif/else` evalúa una condición y decide qué bloque ejecutar; `for` recorre una colección elemento por elemento; `while` repite mientras una condición siga siendo verdadera. El patrón más usado en todo el curso — lista vacía + `for` + `if` + `append()` — apareció acá por primera vez: se arranca con una lista vacía, se recorre la colección original, y se van agregando solo los elementos que cumplen una condición. Ese mismo patrón reaparece disfrazado en Pandas (`df[df["columna"] > valor]` es la versión vectorizada de un `for` con `if` adentro), en limpieza de datos, y hasta en los bloques de imputación de los Ejemplos de Clase 10.
- **Funciones:** bloques de código reutilizables, definidos con `def nombre(parametros):`, que reciben datos y devuelven un resultado con `return`. Evitan repetir lógica (si hay que calcular lo mismo en 5 lugares del notebook, se escribe una vez y se llama 5 veces) y son la base de cómo se organiza cualquier notebook profesional. De hecho, los dos Ejemplos de esta Clase 10 llevan esta idea a su máxima expresión: un `Pipeline` de scikit-learn no es más que una cadena de funciones (imputar, escalar, codificar, entrenar) empaquetadas para ejecutarse en orden, una sola vez, de forma reproducible.

### Filmina 07 — Todo junto, en un ejemplo chico

**Qué decir:**

```python
temperaturas = [22.5, 24.0, 30.1, 18.3]

alertas = []
for temp in temperaturas:
    if temp > 28:
        alertas.append(temp)

def resumen(lista):
    return f"{len(lista)} alertas"

print(resumen(alertas))
```

Este ejemplo condensa los cuatro conceptos de la filmina anterior en 8 líneas: una **lista** guarda la colección de datos, el **`for` + `if`** filtra sin escribir una línea por cada temperatura, y la **función** empaqueta la lógica para poder reutilizarla con cualquier otra lista. Este patrón — lista vacía + `for` + `append` — vuelve a aparecer todo el curso, hasta en Pandas (donde después se reemplaza por operaciones vectorizadas, ver Clase 03).

---

## CLASE 03 — NumPy y Pandas *(Filminas 08–10)*

**Filmina 08 — División.** Vectorización y la estructura de datos central de la ciencia de datos en Python.

### Filmina 09 — Lo que vimos

**Qué decir (ampliando cada tile):**

- **Vectorización y Broadcasting:** operar un array entero de una sola vez, sin bucles — `array * 1000` multiplica cada elemento en paralelo, por debajo, en C compilado, en vez de que Python interprete una multiplicación a la vez con un `for`. Para 50.000 elementos, la diferencia entre un `for` de Python puro y una operación vectorizada de NumPy puede ser de segundos contra milisegundos. **Broadcasting** es lo que permite combinar arrays de distinto tamaño sin igualar las formas a mano — por ejemplo, sumarle un único número a un array de 100 elementos: NumPy "expande" automáticamente ese número para que la operación tenga sentido elemento por elemento, sin que el programador tenga que repetirlo 100 veces.
- **Álgebra Lineal:** operaciones matriciales con NumPy — productos de matrices (`@` o `np.dot`), transposiciones (`.T`), determinantes e inversas. No se usó todos los días de forma explícita en el curso, pero es la base matemática que hace funcionar por dentro a modelos como la Regresión Lineal (que en el fondo resuelve un sistema de ecuaciones matriciales) o las redes neuronales (que son, en esencia, multiplicaciones de matrices encadenadas).
- **Series y DataFrames:** la estructura central de Pandas. Una **Serie** es como una columna con índice — una lista de valores, cada uno con una etiqueta (por defecto 0, 1, 2... pero puede ser una fecha o un ID). Un **DataFrame** es una tabla de Series que comparten el mismo índice — filas y columnas, como una hoja de cálculo. La diferencia clave con un array de NumPy: cada columna de un DataFrame puede tener su propio tipo de dato (`int64`, `float64`, `object` para texto, `datetime64` para fechas), mientras que un array de NumPy solo admite un único tipo para toda la estructura — por eso Pandas es la herramienta para tablas reales (que casi siempre mezclan texto y números) y NumPy es la herramienta para cálculo numérico puro.
- **Preprocesamiento e Integración:** los primeros pasos de limpieza y agregación de datos — filtrar filas, crear columnas calculadas, combinar información de más de una fuente. Todavía sin la profundidad de la Clase 04 (que dedica una clase entera a nulos, `map`/`apply` y `groupby`), pero ya con la idea de fondo instalada: los datos crudos casi nunca están listos para analizar tal como llegan, y ese trabajo de preparación es, en la práctica, la mayor parte del tiempo de cualquier proyecto real — más que el modelado en sí.

### Filmina 10 — Lista de Python vs. Array de NumPy

**Qué decir:**

| Característica | Lista Python | Array NumPy |
|---|---|---|
| Tipos de datos | Puede mezclar int, str, float | Solo un tipo por array |
| Velocidad | Lenta (Python puro) | Muy rápida (C compilado) |
| Operaciones matemáticas | Una por una, con `for` | Sobre todo el array a la vez |

Para un millón de elementos, la diferencia puede ser de segundos vs. milisegundos. Por eso NumPy es la base de Pandas, Scikit-Learn y prácticamente todo el ecosistema de ciencia de datos en Python — cuando en la Clase 08 se entrena un Random Forest sobre miles de filas en segundos, es este mecanismo el que lo hace posible por debajo.

**Preguntar a la clase:** ¿por qué convertir una lista de Python a array de NumPy antes de hacer cálculos matemáticos sobre muchos datos?

---

## CLASE 04 — Manipulación de Datos con Pandas *(Filminas 11–13)*

**Filmina 11 — División.** Limpiar, transformar y agrupar — la base de cualquier análisis confiable.

### Filmina 12 — Lo que vimos

**Qué decir (ampliando cada tile):**

- **Nulos y Duplicados:** detección con `.isnull().sum()` (cuántos huecos hay por columna) y `.duplicated().sum()` (cuántas filas están repetidas), y la decisión — no automática — de eliminar o imputar cada caso. Esa decisión depende de qué signifique el hueco: `fillna(0)` si "sin dato" equivale a "no pasó nada" (por ejemplo, sin ventas ese día), `fillna(media/mediana)` si el dato faltante probablemente esté cerca del resto de los valores, o eliminar la fila directamente si no hay ninguna imputación razonable. Esta misma decisión reaparece en los Ejemplos 1 y 2 de esta clase: en ambos hay una tabla explícita de "qué estrategia de imputación usar para cada columna, y por qué" — no es una regla única para todo el dataset, es columna por columna.
- **`map` y `apply`:** `map()` traduce valores de una columna usando un diccionario o una función simple (por ejemplo, convertir códigos de país en nombres completos); `apply()` calcula una columna nueva a partir de una función más compleja, que puede usar una o varias columnas existentes a la vez. Ambos reemplazan lo que en Python puro sería un `for` recorriendo fila por fila — es la versión Pandas del patrón de vectorización visto en la Clase 03: en vez de decirle a Python "hacé esto para cada fila", se le dice "aplicá esta transformación a toda la columna de una vez".
- **GroupBy y Pivot Tables:** el patrón **Split-Apply-Combine** — se **separa** (Split) el DataFrame en grupos según el valor de una columna (por ejemplo, todas las ventas del mismo producto juntas), se **aplica** (Apply) una función a cada grupo por separado (sumar, promediar, contar), y se **combinan** (Combine) los resultados en una tabla resumen final. `df.groupby("Producto")["Ventas"].sum()` hace las tres cosas en una sola línea. Una Pivot Table es la misma idea pero mostrada como una tabla cruzada, con una variable en filas y otra en columnas — el equivalente a una tabla dinámica de Excel. Es la operación más usada para pasar de datos crudos (una fila por transacción) a un resumen de negocio (un número por categoría).
- **Fechas y Resampling:** `pd.to_datetime()` convierte una columna de texto (`"2024-01-15"`) a un tipo de dato fecha real (`datetime64`), y a partir de ahí se puede usar esa columna como índice temporal y aplicar `resample("W")` o `resample("M")` para agrupar automáticamente por semana o por mes, calculando por ejemplo el promedio de cada período. Sin esa conversión previa, Pandas ordena las fechas como si fueran texto común (alfabéticamente: "2023" antes que "2024" está bien, pero "10" antes que "9" ya rompe la cronología) — un error que aparece explícitamente en los dos Ejemplos de Clase 10, donde `COMPANY_CREATED_AT` y `Year_of_Release` llegan como texto y hay que recordar convertirlos antes de usarlos en cualquier análisis temporal.

### Filmina 13 — `groupby` en acción

**Qué decir:**

```python
# 1. Agrupa las filas por "Producto"
# 2. Toma la columna "Total_Venta"
# 3. Suma todos los valores de cada grupo
df.groupby("Producto")["Total_Venta"].sum()
```

Es el equivalente al `GROUP BY` de SQL, y la operación más usada para pasar de una tabla de transacciones sueltas a un resumen accionable: ventas por región, promedio por categoría, conteo por tipo de cliente. Con esto se cerró la Pre-Entrega 4 (Clase 04): carga, diagnóstico y primer saneamiento del dataset del proyecto final — el mismo dataset que después se usa en la Clase 05 para visualizar, y en la Clase 08 para modelar.

---

## CLASE 05 — Visualización de Datos *(Filminas 14–16)*

**Filmina 14 — División.** De Matplotlib a un dashboard que se entiende en 3 segundos.

### Filmina 15 — Lo que vimos

**Qué decir (ampliando cada tile):**

- **Matplotlib:** el modelo de objetos `Figure` (el lienzo completo — lo que termina siendo el archivo exportado), `Axes` (cada gráfico individual dentro de esa hoja, con su título, su leyenda, sus ejes) y `Axis` (la línea graduada de un solo eje — X o Y — con sus marcas y números). El estilo recomendado es crear `fig, ax = plt.subplots()` y hablarle explícitamente a `ax` (`ax.plot()`, `ax.set_title()`), en vez del estilo implícito `plt.plot()` que se confunde apenas hay más de un gráfico en pantalla. Es la librería base sobre la que está construida Seaborn — entenderla primero es lo que permite después personalizar un gráfico de Seaborn con comandos de Matplotlib cuando Seaborn solo no alcanza.
- **Seaborn:** histogramas (`sns.histplot`), boxplots (`sns.boxplot`), violinplots (`sns.violinplot`, boxplot + curva de densidad combinados) y heatmaps de correlación (`sns.heatmap`), resueltos en una línea de código en vez de las varias que requeriría Matplotlib puro. Se integra directamente con DataFrames de Pandas — se le pasa el DataFrame y los nombres de columna, y Seaborn arma solo los colores, la leyenda y el estilo. Los dos Ejemplos de esta Clase 10 usan Seaborn para todo el bloque de EDA: distribución de ventas, boxplots por categoría, scatterplots de puntaje vs. ventas.
- **Storytelling y Dashboards:** **encuadre** (un título con la unidad, no "ventas" a secas sino "Ventas mensuales (miles de $)"), **jerarquía visual** (usar el color para destacar el dato que importa, dejando el resto en un tono neutro), **ética de los gráficos** (el eje Y siempre debería arrancar en cero, salvo razón estadística muy fuerte — truncarlo exagera visualmente diferencias mínimas), y la **regla de oro** de máximo 5 visualizaciones por dashboard — más que eso genera ruido visual en vez de claridad.
- **UI/UX:** la **regla de lectura en "Z"** — en culturas occidentales el ojo escanea una pantalla siguiendo ese recorrido, así que los KPIs principales van arriba-izquierda (lo primero que se ve), los controles/filtros arriba-derecha, las tendencias abajo-izquierda, y el detalle (tablas) abajo-derecha. Y los principios de jerarquía visual: color con propósito (semántico — rojo alerta, verde cumplimiento — no decorativo), espacio en blanco (agrupa y separa, no es espacio desperdiciado), y tamaño/peso (lo más grande se percibe como más importante, sin abusar).

### Filmina 16 — ¿Qué Gráfico Usar?

**Qué decir:**

| Pregunta | Gráfico |
|---|---|
| ¿Cómo se reparten mis datos? | Histograma / KDE |
| ¿Outliers o dispersión en un grupo? | Boxplot |
| ¿Qué variables están relacionadas? | Scatter / Heatmap |

Un gráfico técnicamente correcto no alcanza — tiene que comunicar de forma honesta y clara. Los dos Ejemplos de esta Clase 10 aplican exactamente esta tabla: histogramas y boxplots en la etapa de EDA, scatterplots en el análisis bivariado, y heatmaps de correlación o de nulos (`missingno`) antes de modelar.

---

## CLASE 06 — EDA y Estadística Descriptiva *(Filminas 17–19)*

**Filmina 17 — División.** Entender la forma de los datos antes de modelarlos.

### Filmina 18 — Lo que vimos

**Qué decir (ampliando cada tile):**

- **Limpieza e Integración:** detección y tratamiento de valores faltantes y outliers, y combinar información que viene de más de una fuente. Es el paso que, en la práctica, ocupa más tiempo que el modelado en cualquier proyecto real — los dos Ejemplos de Clase 10 dedican secciones enteras solo a esto antes de entrenar el primer modelo.
- **Tendencia Central y Dispersión:** la **media** (promedio, sensible a valores extremos), la **mediana** (el valor del medio, resistente a extremos), la **moda** (el valor más frecuente, la única que sirve para categóricas), y el **desvío estándar** (qué tan dispersos están los datos alrededor de la media — un desvío chico significa datos concentrados, uno grande significa datos esparcidos). Son las cuatro formas más comunes de resumir una variable completa con un solo número, y elegir cuál usar depende de la forma de la distribución (ver Filmina 19).
- **Distribuciones:** la **Normal** (forma de campana, simétrica, la mayoría de los valores cerca del promedio — alturas de personas, por ejemplo) y la **Uniforme** (todos los valores tienen la misma probabilidad, sin un pico central — como tirar un dado). Reconocerlas con histogramas y diagramas de densidad (KDE, una curva suave que estima la forma sin depender de cuántos "bins" se eligieron) es el primer paso antes de decidir qué medida de tendencia central o qué transformación aplicar — no tiene sentido usar la media como resumen de una distribución muy sesgada.
- **Normalización y Escalado:** llevar variables a la misma escala antes de compararlas o de usarlas en un modelo sensible a la magnitud. El problema concreto: si una variable va de 0 a 1 (por ejemplo, una tasa) y otra va de 0 a 100.000 (por ejemplo, un ingreso), un modelo como KNN o Regresión Logística le va a dar más peso a la segunda solo por su escala numérica, no porque sea más importante. El `StandardScaler` (que resta la media y divide por el desvío estándar, dejando cada variable con media 0 y desvío 1) que aparece en los Ejemplos 1 y 2 de esta clase resuelve exactamente este problema.

### Filmina 19 — Media vs. Mediana: ¿Cuál Usar?

**Qué decir:**

| Medida | Cuándo conviene |
|---|---|
| **Media** | Distribución simétrica, sin valores extremos que la distorsionen. |
| **Mediana** | Hay outliers o la distribución tiene sesgo (ej. salarios, ventas muy altas puntuales). |

La mediana resiste mejor los valores extremos — la elección correcta depende de la forma real de la distribución, no de una regla fija. Es exactamente el criterio que se aplica en el Ejemplo 2 de esta clase (videojuegos): `Critic_Score` y `User_Score` se imputan con media (distribución más simétrica), mientras que `Critic_Count` y `User_Count` se imputan con mediana (unos pocos juegos tienen miles de reseñas, y eso arrastraría el promedio).

**Preguntar a la clase:** si una columna de "años de antigüedad de un empleado" tiene un outlier de un fundador con 40 años en la empresa, ¿conviene imputar los nulos con la media o con la mediana del resto del equipo?

---

## CLASE 07 — Pipelines Reproducibles y Casos de ML *(Filminas 20–22)*

**Filmina 20 — División.** Un análisis que no se puede reproducir no sirve en un entorno profesional.

### Filmina 21 — Lo que vimos

**Qué decir (ampliando cada tile):**

- **Principios de Reproducibilidad:** qué hace que un pipeline se pueda repetir con el mismo resultado siempre, en cualquier computadora. Los tres pilares: **semillas fijas** (`random_state=42` en cada split y cada modelo — sin esto, cada corrida daría números ligeramente distintos porque los procesos "aleatorios" de la computadora en realidad son pseudoaleatorios y dependen de un punto de partida), **sin pasos manuales** (nada de "ejecutar esta celda antes que esa otra a mano" — el notebook tiene que correr de arriba a abajo con "Run all"), y **no depender del orden de ejecución** (cada celda no debería asumir que otra ya corrió antes fuera de su propio orden natural).
- **Pipeline End-to-End:** las etapas completas de un proyecto de datos, de punta a punta: **ingesta** (¿de dónde vienen los datos — un CSV, una API, una base de datos?), **transformación** (limpieza, cálculo de columnas nuevas, agregaciones), y la idea de que cada etapa debería poder ejecutarse sola y producir siempre la misma salida a partir de la misma entrada — eso es lo que la vuelve "componible" con las etapas siguientes, sin sorpresas.
- **ML en la Industria:** casos de uso reales de Machine Learning aplicados a procesos productivos — **mantenimiento predictivo** (anticipar la falla de una máquina antes de que ocurra, a partir de datos de sensores, en vez de esperar a que se rompa), **control de calidad** (detectar automáticamente productos defectuosos en una línea de producción), y **optimización logística** (predecir demanda para no sobre-stockear ni quedarse sin producto). Los dos Ejemplos de Clase 10 son, en el fondo, casos de uso de este mismo tipo pero en SaaS B2B (predecir qué clientes se van a quedar) y en entretenimiento (predecir qué videojuego va a vender).

### Filmina 22 — ¿Qué Hace Reproducible un Pipeline?

**Qué decir:**

- Corre completo con "Run all", sin celdas ejecutadas a mano fuera de orden.
- Usa semillas fijas (`random_state`) para que el resultado no cambie entre corridas.
- No depende de variables sueltas fuera de las funciones.
- Documenta de dónde salen los datos y qué transformaciones se aplicaron.

Este mismo criterio reaparece en cada Pre-Entrega del curso — y en los dos Ejemplos de esta Clase 10: ambos usan `random_state=42` en cada split y cada modelo, ambos documentan el origen del dataset, y ambos son notebooks que corren de punta a punta sin intervención manual. Un notebook que solo corre "en la compu de quien lo hizo" no está terminado.

---

## CLASE 08 — Aprendizaje Supervisado en Práctica *(Filminas 23–25)*

**Filmina 23 — División.** De un dataset limpio a un modelo entrenado y validado correctamente.

### Filmina 24 — Lo que vimos

**Qué decir (ampliando cada tile):**

- **Árbol de Decisión y KNN:** dos formas muy distintas de clasificar. El **Árbol de Decisión** aprende reglas jerárquicas del tipo "si `ALL_USERS_COUNT` es mayor a 5, mirá `ACTIVE_CHANNELS`; si no, clasificá como churn" — es fácil de visualizar e interpretar, como un diagrama de flujo. **KNN** (K-Nearest Neighbors, "los K vecinos más cercanos") no aprende reglas explícitas: para clasificar un caso nuevo, busca los K casos más parecidos ya conocidos (según distancia en el espacio de variables) y vota por la clase mayoritaria entre ellos — por eso es tan sensible a la escala de las variables (de ahí la importancia del `StandardScaler` visto en Clase 06).
- **Regresión Logística y Lineal:** la **Regresión Lineal** predice un número continuo (por ejemplo, "cuánto va a vender un videojuego en millones"); la **Regresión Logística**, a pesar del nombre, predice una probabilidad de pertenecer a una clase (por ejemplo, "70% de probabilidad de que este cliente haga churn") y con un umbral se convierte en 0 o 1. El Ejemplo 2 de esta clase muestra ambos casos sobre el mismo dataset de videojuegos — primero regresión (predecir ventas exactas, con R² cercano a cero: el problema es demasiado difícil para estas variables) y después clasificación (predecir éxito/no éxito con un umbral de 1 millón de copias, con resultados bastante mejores). Es un buen ejemplo real de que a veces reformular la pregunta es mejor estrategia que agregar más variables.
- **Random Forest:** muchos árboles de decisión entrenados sobre subconjuntos distintos de datos y variables (un "bosque"), donde cada árbol vota su predicción y se toma la mayoría (clasificación) o el promedio (regresión). Es más robusto que un árbol solo y menos propenso a sobreajustar, porque los errores individuales de cada árbol tienden a cancelarse entre sí. Además, permite calcular **Feature Importances**: qué tan seguido cada variable fue la que más ayudó a separar las clases en todo el bosque — en el Ejemplo 1 (Veeqo), `ALL_USERS_COUNT` resultó ser, por lejos, la variable más importante para predecir qué empresas se quedan.
- **Cross-Validation:** validar el modelo en varios splits distintos de los datos (no en uno solo) para confiar en que la métrica obtenida no fue solo buena o mala suerte con una partición particular de train/test. `StratifiedKFold` (usado en los dos Ejemplos) divide los datos en, por ejemplo, 5 partes ("folds"), entrena 5 veces usando 4 partes para entrenar y 1 para validar (rotando cuál es la de validación), y además garantiza que cada partición mantenga la misma proporción de clases que el dataset completo — crítico cuando las clases están desbalanceadas, como en ambos Ejemplos (0.82% de clientes `live` en Veeqo, ~25% de juegos "exitosos").

### Filmina 25 — El Error Más Común: Data Leakage

**Qué decir:**

```python
# La clave: fit SOLO sobre train,
# transform aplicado a ambos
scaler.fit(X_train)
X_train_esc = scaler.transform(X_train)
X_test_esc = scaler.transform(X_test)
```

Si se ajusta el `scaler` (o cualquier preprocesamiento) sobre todo el dataset antes de dividir, el modelo "espía" información del test — la métrica final queda inflada y miente sobre el rendimiento real. Este es, literalmente, el defecto que tiene la primera mitad del Ejemplo 2 de Clase 10 (`LabelEncoder` y `StandardScaler` ajustados sobre el DataFrame completo) y que se corrige explícitamente en la segunda mitad, con un `Pipeline` que ajusta cada paso solo sobre el train de cada fold de `GridSearchCV`. Vale la pena mostrar ese contraste en vivo si hay tiempo — es la lección más importante de toda la Clase 08, vista en código real.

---

## CLASE 09 — Aprendizaje No Supervisado *(Filminas 26–28)*

**Filmina 26 — División.** Sin etiquetas: encontrar estructura en los datos en vez de predecir.

### Filmina 27 — Lo que vimos

**Qué decir (ampliando cada tile):**

- **Reglas de Asociación:** el algoritmo **Apriori** encuentra qué productos o eventos aparecen juntos con frecuencia — el ejemplo clásico es "quienes compran pañales también compran cerveza" (canasta de compra en retail). Se mide con tres métricas: **soporte** (qué tan frecuente es esa combinación en todo el dataset), **confianza** (dado que alguien compró A, ¿qué probabilidad hay de que compre B?), y **lift** (¿esa relación es más fuerte que la que habría por puro azar?). No predice nada sobre un caso individual como los modelos supervisados — encuentra patrones de co-ocurrencia mirando el dataset completo.
- **K-Means:** agrupar los datos en `k` clusters según similitud — cada punto queda asignado al cluster cuyo "centro" (centroide) tiene más cerca, y los centroides se van recalculando hasta que el agrupamiento se estabiliza. La pregunta difícil es cómo elegir `k` con criterio en vez de adivinarlo: el **método del codo** grafica qué tan compactos quedan los clusters para distintos valores de `k` y busca el punto donde agregar un cluster más deja de mejorar mucho las cosas; el **silhouette score** mide qué tan bien separado está cada punto de los clusters vecinos, con un número entre -1 y 1.
- **Jerárquico y DBSCAN:** el clustering **Jerárquico** no necesita definir `k` de antemano — va fusionando (o dividiendo) grupos de a poco y arma un **dendrograma**, un árbol que se puede "cortar" a la altura que se quiera para obtener más o menos clusters. **DBSCAN** agrupa por densidad (zonas donde hay muchos puntos juntos forman un cluster, zonas dispersas no) y tiene una ventaja que ni K-Means ni el jerárquico tienen: detecta **ruido/outliers** como su propia categoría, en vez de forzarlos a pertenecer a algún cluster igual.
- **PCA** (Análisis de Componentes Principales): reducir muchas variables correlacionadas a unas pocas "componentes" nuevas que conservan la mayor parte de la información original — por ejemplo, pasar de 30 variables a 2 componentes que explican el 85% de la variabilidad de los datos. Es útil tanto para **visualizar** en 2D datos que tienen demasiadas dimensiones para graficar directamente, como para **simplificar** antes de modelar (menos variables, menos ruido, entrenamiento más rápido) — aunque las componentes resultantes ya no tienen un significado tan directo como las variables originales.

### Filmina 28 — Tres Grandes Tipos de Problemas

**Qué decir:**

| Tipo | Objetivo | Ejemplo |
|---|---|---|
| Clustering | Agrupar datos similares | Segmentar clientes |
| Reducción de Dimensionalidad | Simplificar variables | Visualizar en 2D con PCA |
| Reglas de Asociación | Encontrar relaciones frecuentes | Qué se compra junto en retail |

Sin etiquetas, el objetivo cambia de "predecir una respuesta conocida" (Clase 08) a "encontrar patrones ocultos" — es otra forma de pensar el problema, no una versión incompleta del supervisado. Vale la pena cerrar señalando que ambos mundos (supervisado y no supervisado) conviven en un proyecto real: por ejemplo, se podría usar K-Means para segmentar clientes de Veeqo (Ejemplo 1 de esta clase) antes de entrenar un modelo de churn distinto para cada segmento.

**Preguntar a la clase:** de los dos Ejemplos de hoy (Veeqo y videojuegos), ¿qué problema de clustering o reglas de asociación se podría plantear sobre esos mismos datasets, además de lo que ya se hizo?

---

## Filmina 29 — Cierre

"De Python puro a los dos grandes mundos del Machine Learning" — el resumen de todo el recorrido: empezar con variables y `for` loops (Clase 02), pasar por la manipulación y visualización de datos (Clases 03 a 06), la reproducibilidad como estándar profesional (Clase 07), y terminar sabiendo cuándo un problema tiene etiquetas (Clase 08) y cuándo no (Clase 09). Es el momento de abrir la clase a preguntas de cualquiera de las nueve clases antes de pasar a los dos Ejemplos integradores.
