# Clase 06 — Estadística y Preprocesamiento: Fundamentos para el Análisis de Datos

**Curso de Data Science I · Clase 06** — el checkpoint de la primera Pre-entrega del proyecto.

Esta guía sigue el **orden exacto de las 53 filminas** de `Clase06.html`, organizadas en 7 módulos. No es un resumen de lo que ya dice cada filmina — la idea es que la filmina sea el disparador visual (el título, la tabla, el tile) y que el texto de acá sea el material **adicional** para decir en voz alta: ejemplos numéricos resueltos, preguntas para tirarle al grupo, conexiones con otras clases, y matices que no entran en una diapositiva. Al final hay una guía del notebook (`Clase_6_Fundamentos_de_Ciencia_de_Datos_.ipynb`) y el detalle de la Pre-entrega evaluada.

> **Por qué esta clase importa más que las anteriores**: es la primera vez que lo que se enseña se convierte directamente en una nota. Todo lo que sigue —tendencia central, dispersión, escalado, limpieza— no es teoría suelta: es exactamente lo que hace falta para completar el informe de "Tienda TechWorld" al final de la clase.

---

## Objetivos de la clase

1. Elegir la medida de tendencia central (media, mediana, moda) y de dispersión (rango, varianza, desviación estándar) correcta según la forma de los datos.
2. Aplicar Normalización (Min-Max) y Estandarización (Z-Score), evitando la Fuga de Información (Data Leakage) al dividir train/test.
3. Distinguir población de muestra, y variables cualitativas de cuantitativas, para elegir el preprocesamiento correcto de cada una.
4. Construir tablas de frecuencia (absoluta y relativa) y usarlas para detectar sesgos de muestra.
5. Crear variables derivadas con criterio de negocio (Feature Engineering: ratios, variables temporales, binning).
6. Completar la Pre-entrega "Tienda TechWorld": clasificar variables, detectar errores de calidad y proponer su limpieza sobre un CSV crudo.

---

## Filmina 01 — Portada

Apertura de la clase. El título ya anticipa el arco completo del día: empezamos con números que resumen datos (Módulo 01-02) y terminamos con un dataset sucio real que hay que diagnosticar y limpiar (Pre-entrega). Vale la pena decirlo así de entrada — todo lo que se enseña hoy converge en el informe que se entrega al final.

---

# Módulo 01 — Estadística Descriptiva (Filminas 02-07)

## Filmina 02 — División de Módulo

Divisor de sección. El gancho narrativo: sos el analista de una app de delivery y el gerente pregunta "¿cuánto tardamos en promedio en entregar un pedido?". Parece una pregunta simple, pero la respuesta correcta depende de qué medida elijas.

## Filmina 03 — Medidas de Tendencia Central: Media, Mediana y Moda

**Ejemplo numérico para resolver en el pizarrón (no está en la filmina)**: tomá estos 8 tiempos de entrega en minutos: `18, 19, 19, 20, 21, 22, 23, 180`.

- **Media** = (18+19+19+20+21+22+23+180) / 8 = 322 / 8 = **40.25 min**.
- **Mediana** = ordenados, el promedio de los dos centrales (20 y 21) = **20.5 min**.
- **Moda** = **19 min** (es el único que se repite).

Un solo pedido con tormenta (180 min) sobre 8 casos ya duplicó la media respecto a la mediana. Con este ejemplo a mano, la pregunta "¿le pasarías al gerente el número 40 o el 20?" se contesta sola.

**Pregunta para tirar a la clase**: ¿tiene sentido calcular la "moda" del tiempo de entrega si nunca se repite exactamente el mismo minuto con decimales? — No, y ahí está el límite práctico de la moda en variables continuas: para que sirva hay que agruparla en intervalos (bins) primero, exactamente lo que hace un histograma (Clase 05). Es un buen anticipo de por qué el Bloque 0 del notebook repasa histogramas antes de seguir.

## Filmina 04 — Medidas de Dispersión: Rango, Varianza y Desviación Estándar

**Ejemplo numérico clásico para resolver en el pizarrón**: datos `2, 4, 4, 4, 5, 5, 7, 9` (media = 5).

| Dato | Distancia a la media | Distancia² |
|---|---|---|
| 2 | −3 | 9 |
| 4 | −1 | 1 |
| 4 | −1 | 1 |
| 4 | −1 | 1 |
| 5 | 0 | 0 |
| 5 | 0 | 0 |
| 7 | 2 | 4 |
| 9 | 4 | 16 |

Suma de distancias² = 32. Varianza = 32 / 8 = **4**. Desviación estándar = √4 = **2**.

**Trampa técnica que no está en la filmina, y que sí aparece en la práctica**: ¿dividís por `n` u por `n − 1`? Dividir por `n` (usado arriba) es la **varianza poblacional**; dividir por `n − 1` es la **varianza muestral** (corrección de Bessel), y es la que casi siempre corresponde porque casi siempre trabajamos con una muestra, no con la población completa. Este detalle importa en la práctica porque `numpy` y `pandas` **no coinciden por defecto**: `np.std()` usa `ddof=0` (poblacional) y `df['col'].std()` de pandas usa `ddof=1` (muestral) — es una fuente real de bugs cuando se mezclan ambas librerías sobre los mismos datos y los resultados no calzan.

## Filmina 05 — Lectura Conjunta: Centro y Dispersión

**Ejemplo numérico del Coeficiente de Variación (no está en la filmina)**: un elefante pesa en promedio 4.000 kg con desviación de 400 kg → CV = 400/4.000 = **10%**. Un ratón pesa en promedio 20 g con desviación de 4 g → CV = 4/20 = **20%**. En términos absolutos el elefante "varía más" (400 kg vs. 4 g), pero en términos *relativos* el ratón es el doble de variable — el CV es el que revela esto, y es la razón por la que se usa para comparar variabilidad entre grupos de magnitudes muy distintas (sucursales chicas vs. grandes, por ejemplo).

## Filmina 06 — Aplicaciones Reales y Errores Comunes

Los tres casos de la filmina (Fintech, Streaming, Manufactura) ya se explican solos con la tabla. Un cuarto caso para sumar en voz alta, porque conecta hacia adelante con el Módulo 02: **Educación** — dos exámenes con distinta dificultad no se pueden comparar por la nota cruda (un 8 en un examen fácil no es lo mismo que un 8 en uno difícil); se estandariza cada nota con su propia media y desviación del curso (Z-Score) antes de comparar. Es el mismo problema del "campo de juego nivelado" que va a aparecer formalmente en la Filmina 09, aplicado a un contexto no numérico-financiero.

## Filmina 07 — Práctica (no entregable): Estadística Descriptiva sobre tus Propios Datos

**Qué mirar al corregir (no está en la filmina)**: el error más común no es de cálculo, es de **consistencia** — alumnos que calculan la desviación estándar con `numpy` en un punto del trabajo y con `pandas` en otro, y les dan números ligeramente distintos por el tema de `ddof` explicado en la Filmina 04, sin darse cuenta de por qué. Vale la pena preguntarles explícitamente qué librería usaron para cada métrica.

---

# Módulo 02 — Normalización y Estandarización (Filminas 08-15)

## Filmina 08 — División de Módulo

El objetivo del módulo en una frase: poner a todos los datos en un "campo de juego nivelado", sin importar sus unidades originales.

## Filmina 09 — El Problema de la Escala: ¿Por Qué Escalar?

**Cálculo concreto para mostrar por qué domina la variable de rango más grande (no está en la filmina)**: dos personas, Edad y Salario. Persona A: Edad 30, Salario \$50.000. Persona B: Edad 25, Salario \$51.000. Distancia euclidiana sin escalar: √((30−25)² + (50.000−51.000)²) = √(25 + 1.000.000) ≈ **1000.01**. La diferencia de 5 años en la edad prácticamente no mueve la aguja — el algoritmo de K-Nearest Neighbors terminaría clasificando a estas dos personas como "distintas" basándose casi enteramente en el salario, ignorando la edad por completo.

## Filmina 10 — Normalización (Min-Max Scaling)

**Fórmula**: `X_norm = (X − X_min) / (X_max − X_min)`

**Caso borde que no está en la filmina**: ¿qué pasa si en el conjunto de test aparece un valor más alto que el `X_max` visto en el train? El resultado normalizado da **mayor a 1** (o menor a 0, si es más chico que el mínimo del train) — no es un error, es la señal esperada de que el escalador se ajustó correctamente solo con el train (esto se retoma con un ejemplo de código real en la Filmina 13 y en el notebook). Otro caso borde: si `X_max == X_min` en el train (una columna constante), la fórmula divide por cero — hay que filtrar esas columnas antes de escalar.

## Filmina 11 — Estandarización (Z-Score Scaling)

**Fórmula**: `X_std = (X − μ) / σ`

**Regla práctica que no está en la filmina**: en una distribución normal, aproximadamente el 68% de los datos cae entre Z = −1 y Z = +1, el 95% entre −2 y +2, y el 99.7% entre −3 y +3 (la "regla empírica" o 68-95-99.7). Por eso **|Z| > 3** es uno de los criterios numéricos más usados para marcar un dato como outlier — vale la pena anotarlo acá porque va a reaparecer, sin la fórmula, cuando se hable de outliers en el Módulo 06 (Filmina 39).

## Filmina 12 — Comparativa: ¿Cuándo Usar Cada Una?

**Una tercera opción que no aparece en la tabla de la filmina**: `RobustScaler` (usa la mediana y el IQR en vez de la media y la desviación estándar). Ni Min-Max ni Z-Score son ideales cuando hay outliers fuertes — Min-Max los deja dominar el rango entero, y Z-Score los deja inflar la desviación estándar y "aplastar" al resto de los datos cerca de 0. `RobustScaler` esquiva ambos problemas porque la mediana y el IQR, por definición, no se mueven con los extremos (la misma robustez de la Filmina 03).

## Filmina 13 — El Peligro de la "Fuga de Información" (Data Leakage)

**Demostración numérica de por qué importa (no está en la filmina)**: supongamos un dataset con un mínimo de \$0 (típico si hay algún registro con el salario faltante cargado como 0) y un outlier de test con salario \$1.000.000, mientras que el 95% del train tiene salarios entre \$20.000 y \$50.000. Si ajustás el `MinMaxScaler` sobre **todo** el dataset (con fuga), un salario de \$40.000 se normaliza como (40.000−0)/(1.000.000−0) = **0.04** — casi cero, aplastado por el outlier que ni siquiera pertenece al train. Si ajustás **solo con el train** (rango \$20.000–\$50.000, sin ese outlier), el mismo \$40.000 se normaliza como (40.000−20.000)/(50.000−20.000) ≈ **0.67** — un número totalmente distinto, y el correcto. El modelo que entrena con la primera versión "vio" indirectamente la existencia de un outlier de test antes de tiempo, y eso es exactamente la fuga de información.

**Conexión hacia adelante**: en `scikit-learn`, la forma de blindarse contra este error en cross-validation es usar `Pipeline` (cada fold ajusta su propio escalador desde cero, sin fuga) — se retoma con código real en Clase 08.

## Filmina 14 — Errores Comunes en el Escalado

**Un orden de operaciones que no está explícito en la filmina, pero que conviene remarcar**: la secuencia correcta es **outliers → split train/test → escalar**, no al revés. Si escalás antes de tratar los outliers, el outlier distorsiona los parámetros del escalador (Filmina 13 lo muestra con números); si escalás antes de dividir train/test, hay fuga de información. El orden importa tanto como cada paso individual.

## Filmina 15 — Práctica (no entregable): Escalado en Python

**Qué mirar al corregir (no está en la filmina)**: el error de ejecución más común no es conceptual, es de sintaxis — llamar `.fit_transform()` sobre el test en vez de `.transform()`. Vale la pena revisar ese detalle línea por línea en el código de cada alumno, porque el resultado numérico *parece* razonable igual (números entre 0 y 1), así que el error no salta a la vista solo con mirar el output.

---

# Módulo 03 — Estadística y Preprocesamiento: Fundamentos (Filminas 16-23)

## Filmina 16 — División de Módulo

El gancho: "Garbage in, garbage out" — si los datos de entrada son ruidosos, el resultado será erróneo, sin importar qué tan sofisticado sea el modelo que viene después.

## Filmina 17 — El Puente entre los Datos Crudos y el Conocimiento

**Conexión con Clase 01 que no está en la filmina**: la división entre "preprocesamiento" (fontanería) y "estadística" (interpretación) es, en la práctica de una empresa real, la misma división de roles vista en la Filmina 06 de Clase 01 — el **Data Engineer** hace el preprocesamiento a gran escala (pipelines que limpian datos antes de que lleguen a nadie más), y el **Data Scientist** hace la interpretación estadística sobre esos datos ya limpios. En este curso una sola persona hace ambos roles, pero vale la pena que el grupo identifique en qué "sombrero" está en cada paso.

## Filmina 18 — Población, Muestra y Tipos de Variables

**Matiz estadístico que no está en la filmina**: con variables **ordinales** (bajo/medio/alto) es tentador calcular un "promedio" asignándoles números (1, 2, 3) — pero estrictamente esa operación asume que la distancia entre "bajo" y "medio" es igual a la distancia entre "medio" y "alto", algo que casi nunca es cierto en la realidad (la satisfacción "alta" puede estar mucho más lejos de la "media" de lo que la "media" está de la "baja"). Por eso para ordinales suele ser más honesto reportar la **moda** o la mediana de los rangos, no la media aritmética.

## Filmina 19 — El Flujo de Trabajo del Preprocesamiento

**El código real detrás de cada tile (no está en la filmina, pero conviene tenerlo a mano)**: valores faltantes → `.fillna()` / `.dropna()`; ruido/outliers → filtro por IQR o por límites de negocio (`df[df['edad'] < 120]`); integración → `pd.merge()` / `pd.concat()`; transformación → `pd.cut()` para binning, `.resample()` para agregación temporal. Cada uno de estos cuatro verbos se usa literalmente en el notebook de hoy (Bloque 1 y siguientes).

## Filmina 20 — Ejemplo del Mundo Real: el Caso del E-commerce

**La línea de código que soluciona el problema de "Madrid"/"madrid"/"MADRID" (no está en la filmina)**: `df['ciudad'] = df['ciudad'].str.lower().str.strip()` — pasar todo a minúscula y sacar espacios en blanco sobrantes resuelve de una sola vez dos de los tres problemas de esa columna. El tercer problema (el gasto "500)" con un error de tipeo) necesita una limpieza más específica, generalmente con una expresión regular o una revisión manual si son pocos casos.

## Filmina 21 — Frecuencias y el Riesgo del Sesgo de Muestra

**Un segundo caso de sesgo de muestra, distinto al de la filmina**: una encuesta de "salida" (exit poll) hecha en la puerta de un supermercado premium mide satisfacción de "los clientes" — pero por construcción excluye a toda la gente que no puede pagar ese supermercado y compra en otro lado. La frecuencia relativa calculada sobre esa muestra puede estar perfectamente bien hecha matemáticamente, y aun así no significar nada sobre "los consumidores" en general.

## Filmina 22 — Errores Comunes y Mejores Prácticas

**Un error de "suciedad invisible" que no está en la filmina**: dos archivos con distinta codificación de caracteres (UTF-8 vs. Latin-1) pueden hacer que "Córdoba" se lea como "CÃ³rdoba" en uno de ellos — Python los va a tratar como dos categorías **totalmente distintas** aunque a simple vista, mirando el archivo en Excel, puede que ni se note el problema. Es de los bugs más difíciles de diagnosticar porque no tira error, solo infla silenciosamente el conteo de categorías únicas.

## Filmina 23 — Práctica (no entregable): Diagnóstico de una App de Fitness

**Qué mirar al corregir (no está en la filmina)**: la trampa de esta práctica es que "Edad = 0" y "Edad = 200" son errores de **naturaleza distinta** aunque ambos sean "valores raros" — el 0 probablemente es un campo vacío que el sistema completó con un valor por defecto (Filmina 06, Módulo 01: la moda revelando un "valor por defecto"), mientras que el 200 es casi seguro un error de tipeo (¿20 con un cero de más?). Vale la pena que el alumno distinga ambos casos en vez de tratarlos con la misma técnica.

---

# Módulo 04 — El Arte de Ordenar el Caos (Filminas 25-30)

*(Filmina 24 es el Break — 10 minutos.)*

## Filmina 25 — División de Módulo

Reencuadre después del break: de una montaña de "datos en bruto" a una historia coherente sobre ellos — la metáfora es la biblioteca con los libros tirados en el piso.

## Filmina 26 — GIGO: el Flujo del Científico de Datos en 5 Pasos

**Nombre técnico que no está en la filmina**: estos 5 pasos son, en esencia, una versión simplificada de **CRISP-DM** (Cross-Industry Standard Process for Data Mining), la metodología estándar de la industria para proyectos de datos. Vale la pena nombrarla explícitamente — es un término que un alumno puede buscar después, o mencionar en una entrevista de trabajo, y que va a reconocer si en el futuro trabaja con equipos que la usan formalmente.

## Filmina 27 — La Tabla de Frecuencias: el Primer Puente

**Un segundo ejemplo numérico, de marketing, para reforzar el mismo punto (no está en la filmina)**: un anuncio A tiene 100 clics sobre 100.000 impresiones (CTR = 0.1%); un anuncio B tiene 10 clics sobre 500 impresiones (CTR = 2%). En frecuencia absoluta, A parece "mejor" (100 clics contra 10). En frecuencia relativa, B es 20 veces más efectivo. Es el mismo patrón del caso de la fábrica, en un dominio completamente distinto — sirve para que el grupo vea que no es una coincidencia del ejemplo de la filmina, sino un patrón general.

## Filmina 28 — Visualización: el Modelo del "Zoom"

**Conexión con la jerarquía de una organización, que no está en la filmina**: los tres niveles de zoom coinciden, en la práctica, con quién los consume en una empresa — un analista trabaja con la **Tabla** (el detalle), un gerente medio mira el **Gráfico** (la forma general), y un director o CEO solo necesita la **Medida** (un número: "¿estamos bien o mal?"). El mismo dato se presenta distinto según el nivel de la organización que lo va a leer.

## Filmina 29 — Errores Comunes: el Engaño de la Media y el Contexto

**Los números concretos del "engaño de la media" (la filmina lo describe sin cifras)**: 9 personas que ganan \$30.000 cada una, más 1 persona que gana \$970.000. Media = (9×30.000 + 970.000) / 10 = 1.240.000 / 10 = **\$124.000**. Ninguna de las 10 personas gana cerca de \$124.000 — 9 de ellas ganan menos de un cuarto de esa cifra. Es el ejemplo perfecto para mostrar, con números reales en el pizarrón, por qué la media sola puede mentir.

## Filmina 30 — Práctica (no entregable): el Reporte del Café Popular

**Qué mirar al corregir (no está en la filmina)**: el punto más sutil del ejercicio no es corregir "CAFE AMERICANO" vs. "Café Americano" (eso se ve a simple vista) — es decidir qué hacer con el croissant de \$45 al calcular el "precio promedio de las bebidas". Si el alumno lo incluye en el promedio de bebidas sin comentario, es una señal de que no distinguió que el croissant ni siquiera es de la misma categoría de producto — no es un outlier a tratar, es un dato mal clasificado desde el origen.

---

# Módulo 05 — Feature Engineering (Filminas 31-34)

## Filmina 31 — División de Módulo

El gancho: "limpiar la mesa no es lo mismo que cocinar" — después de limpiar los datos, el Feature Engineering es donde entra la creatividad y el conocimiento de negocio.

## Filmina 32 — ¿Qué es el Feature Engineering?

**El cálculo completo del ejemplo de Capacidad de Ahorro (la filmina lo deja conceptual, sin números)**: Ingresos Mensuales = \$80.000, Gastos Mensuales = \$65.000 → Capacidad de Ahorro = \$15.000, o dicho como ratio, 18.75% del ingreso. Comparar a dos personas por "Capacidad de Ahorro en %" es mucho más justo que compararlas por ingreso bruto — alguien que gana \$200.000 y gasta \$195.000 tiene *menos* capacidad de ahorro real que alguien que gana \$80.000 y gasta \$65.000, aunque su ingreso bruto sea más del doble.

## Filmina 33 — Ratios, Variables Temporales y Binning

**Una cuarta técnica que no está en los tiles de la filmina**: la **interacción entre variables** — multiplicar o combinar dos columnas para capturar un efecto conjunto que ninguna de las dos captura sola. Ejemplo: `Precio × Cantidad = Monto Total de la Venta` no es una simple cuenta contable, es una feature que le permite a un modelo de detección de fraude ver transacciones "raras" (mucha cantidad a precio muy bajo) que pasarían desapercibidas mirando precio o cantidad por separado.

## Filmina 34 — La Regla de Oro: el Contexto es Rey

**El peligro opuesto, que no está en la filmina**: no solo hay riesgo de crear *demasiadas* variables sin sentido — también existe el riesgo de crear una variable que **filtra información del futuro** (leakage a través de features), no del split train/test como en la Filmina 13, sino de la definición misma de la columna. Ejemplo clásico: para predecir si un cliente va a cancelar su suscripción (*churn*), crear la variable "días desde que canceló" — esa columna literalmente contiene la respuesta que se quiere predecir, camuflada como un "feature" más. Es un error más sutil que "crear ruido al azar", y más peligroso porque el modelo va a dar resultados espectacularmente buenos en la validación y va a fallar por completo en producción.

---

# Módulo 06 — Consolidación en la Práctica (Filminas 35-42)

## Filmina 35 — División de Módulo

Reencuadre: de entender promedios y desviaciones a normalizar, detectar outliers y razonar con probabilidad — se cierra el círculo de todo lo visto hasta acá.

## Filmina 36 — Práctica (no entregable): Auditoría de Datos e Interpretación

Escenario introductorio del módulo: ventas de una sucursal (muestra) para decidir la estrategia de compras de toda la región (población). *(Nota: en la filmina esta práctica aparece resumida en el cierre del módulo — filmina 42 — pero conviene anticipar el escenario acá para que el grupo lo tenga presente en las filminas siguientes.)*

## Filmina 37 — El Flujo Completo: de la Recogida a la Inferencia

**Lo que "Inferencia" va a significar formalmente más adelante (no está en la filmina)**: acá se usa la palabra de forma intuitiva ("predecir sobre el grupo más grande"), pero en unidades futuras se va a formalizar con intervalos de confianza y pruebas de hipótesis — herramientas que cuantifican *cuán seguros* podemos estar de que un patrón visto en la muestra realmente existe en la población, y no es puro azar de qué 100 personas encuestamos. Vale la pena decir esto para que el grupo sepa que "inferencia" no termina hoy.

## Filmina 38 — Población, Muestra e Individuo: el Riesgo del Sesgo

**Un segundo caso de sesgo de muestra, clásico de la historia de la estadística (no está en la filmina)**: durante la Segunda Guerra Mundial, la fuerza aérea quería reforzar el blindaje de los aviones que volvían de combate, en las zonas con más impactos de bala. El estadístico Abraham Wald hizo notar el error: los aviones que analizaban eran los que **volvieron** — los impactos que importaban eran los de los aviones que **no volvieron**, en las zonas *sin* impactos visibles en la muestra. Es el "sesgo de supervivencia" (*survivorship bias*), y es el mismo mecanismo del caso de e-commerce de la Filmina 17: la muestra excluye sistemáticamente a un grupo, y ese grupo excluido es justo el que tenía la información más importante.

## Filmina 39 — Gestión de Outliers: ¿Borrar o Investigar?

**La fórmula concreta detrás del boxplot (no está en la filmina)**: un valor se marca como outlier si cae por debajo de `Q1 − 1.5 × IQR` o por encima de `Q3 + 1.5 × IQR`. El multiplicador 1.5 es una convención (no una ley matemática) — algunos análisis usan 3 para ser más permisivos y marcar solo los outliers "extremos". Esta es la misma regla que corre, en código real, en el notebook de la clase.

## Filmina 40 — Lidiar con la Incertidumbre: Introducción a la Probabilidad

**La conexión que cierra el módulo, y que no está explícita en la filmina**: la probabilidad, en su definición frecuentista, **es** el límite de la frecuencia relativa (Filmina 21 y 27) cuando el número de observaciones crece — si una moneda sale cara en el 50% de miles de tiradas, decimos que "la probabilidad de cara es 0.5" precisamente por eso. Frecuencia relativa y probabilidad no son dos temas separados del módulo: la segunda es la primera, llevada al límite.

## Filmina 41 — Errores Comunes: Tabla de Trampas Frecuentes

**Un ejemplo concreto de correlación sin causalidad, distinto al genérico de la filmina**: las ventas de helado y las muertes por ahogamiento están fuertemente correlacionadas mes a mes — pero uno no causa el otro. La variable oculta es el calor: en verano la gente compra más helado *y* nada más (y por lo tanto se ahoga más). Es el ejemplo de manual para esta trampa, y vale la pena tenerlo memorizado porque es más contundente que decir solo "correlación no implica causalidad" en abstracto.

## Filmina 42 — Práctica (no entregable): Auditoría de Datos e Interpretación

**La resolución numérica del caso de Precio Unitario (no está en la filmina — útil tenerla a mano, sin regalarla antes de que el grupo lo intente)**: datos `[10, 12, 11, 15, 2, 500, 13, 11]`. Media = 574/8 = **71.75**. Mediana (ordenados: `2, 10, 11, 11, 12, 13, 15, 500`) = (11+12)/2 = **11.5**. La diferencia entre 71.75 y 11.5 es la evidencia numérica de que el 500 (probable error de tipeo, ¿faltó un punto decimal?) y el 2 (¿faltó un dígito?) están distorsionando cualquier lectura basada en la media.

---

# Módulo 07 — Pre-entrega: El Puente entre los Datos y las Decisiones (Filminas 43-53)

## Filmina 43 — División de Módulo

El anuncio: esta es la última parada teórica antes del entregable evaluado. Todo lo visto en los seis módulos anteriores converge acá.

## Filmina 44 — La Estadística como Ciencia de la Incertidumbre

**Un matiz de vocabulario que no está en la filmina, útil si alguien pregunta**: existen dos grandes escuelas para pensar la probabilidad y la inferencia — la **frecuentista** (la probabilidad es un límite de frecuencias observadas, Filmina 40) y la **bayesiana** (la probabilidad es un grado de creencia que se actualiza con nueva evidencia). Este curso trabaja mayormente con el enfoque frecuentista, pero vale la pena que el grupo sepa que existe el otro, porque va a aparecer en biblioteca y papers como `pymc` o "A/B testing bayesiano".

## Filmina 45 — Preprocesamiento: la Cocina de los Datos

**Un framework que no está en la filmina, y que organiza todo el módulo**: la industria suele hablar de **6 dimensiones de calidad de datos**: Completitud (¿faltan valores?), Precisión (¿los valores son correctos?), Consistencia (¿el mismo dato se representa igual en todos lados?), Actualidad (¿está desactualizado?), Validez (¿respeta el formato/rango esperado?) y Unicidad (¿hay duplicados?). Cada error visto en la clase de hoy —nulos, "Madrid"/"madrid", edades de 200 años, duplicados— encaja en una de estas seis categorías; puede ser útil cerrarlo así antes de entrar a la Pre-entrega.

## Filmina 46 — Variables y Frecuencias: el Lenguaje de la Medición

**El nombre técnico del fenómeno del caso Ciudad A/B (no está en la filmina)**: cuando una conclusión cambia (o se revierte) según si mirás el dato agregado o desagregado, el fenómeno general se llama **Paradoja de Simpson**. El caso de la filmina es una versión simplificada — vale la pena nombrar la paradoja porque es un patrón que reaparece constantemente en análisis por grupos (por ejemplo, comparar dos tratamientos médicos sin tener en cuenta la gravedad de cada paciente).

## Filmina 47 — El Rol Crítico de la Probabilidad

**La decisión que no está en la filmina, pero que es el problema real detrás del "0.98"**: ¿en qué umbral de probabilidad el sistema decide "es spam" y lo manda a la carpeta de correo no deseado? Bajar el umbral (marcar como spam con 0.5 de probabilidad) atrapa más spam real pero también manda más correos legítimos por error; subirlo (recién a partir de 0.95) es más conservador pero deja pasar más spam. Este es el trade-off entre **Precision** y **Recall** — se retoma con métricas y fórmulas concretas en Clase 08.

## Filmina 48 — Errores Comunes en Preprocesamiento y Estadística

**El resultado concreto de promediar códigos postales (la filmina lo menciona sin resolverlo)**: promediar el código postal 1000 con el 3000 da 2000 — un código que puede corresponder a un barrio geográficamente lejano de ambos, o directamente no existir. La operación es matemáticamente válida (son números) pero semánticamente absurda (el código postal es una etiqueta, no una cantidad) — el mismo error, en el fondo, que promediar variables ordinales de la Filmina 18.

## Filmina 49 — Aplicaciones en la Industria Real

**Un cuarto caso, de Recursos Humanos, que no está en la filmina**: comparar el desempeño de empleados evaluados por managers distintos es injusto sin normalizar — un manager estricto que califica en promedio 6/10 y uno permisivo que califica en promedio 9/10 hacen que sus equipos parezcan de calidad distinta aunque no lo sean. Normalizar (o estandarizar) el puntaje de cada empleado respecto al promedio y desviación de *su propio* manager es la misma lógica de la Filmina 06 (Educación) aplicada a *people analytics*.

## Filmina 50 — Síntesis: Puntos Clave para Recordar

**Dos preguntas rápidas para hacerle al grupo antes de pasar a la Pre-entrega (no están en la filmina)**: (1) "Si la media y la mediana de una columna son muy distintas, ¿qué dos explicaciones vimos hoy que lo pueden causar?" (outliers, o una tendencia/sesgo estructural en los datos — Clase 07 va a sumar una tercera: una serie temporal con tendencia). (2) "¿Por qué no se puede ajustar el escalador con todo el dataset?" — si alguien no responde con seguridad "fuga de información", conviene volver un momento a la Filmina 13 antes de seguir.

## Filminas 51-53 — Entregable: Pre-entrega "Tienda TechWorld"

Ver la sección **Pre-entrega**, más abajo, con el detalle completo del dataset y de las 4 consignas.

## Filmina 53 (última) — ¿Dudas? ¿Consultas?

Cierre de la clase — espacio abierto antes de que el grupo se ponga a trabajar en la Pre-entrega.

---

## Guía del Notebook (`Clase_6_Fundamentos_de_Ciencia_de_Datos_.ipynb`)

El notebook aplica el contenido de la clase sobre un dataset real de más de 1 millón de registros de vuelos en Argentina (`base_microdatos.csv`, conectividad aérea — datos.gob.ar). Va en el mismo orden lógico que la filmina, con una salvedad: agrupa varios módulos teóricos en un mismo bloque de código cuando comparten el mismo dataset y el mismo paso del flujo.

**Mapa rápido:**

| Bloque | Contenido | Módulo de la filmina que aplica |
|---|---|---|
| **Bloque 0** | Repaso de Clase 05: `fig, ax`, histograma+KDE, boxplot, heatmap — con un mini dataset sintético antes de tocar el dataset real. | Repaso, no es contenido nuevo |
| **Paso 0** | Diagnóstico automático de calidad: nulos y duplicados, con alertas si superan un umbral de negocio. | Módulo 03 (Filmina 19) |
| **Bloque 1** | Limpieza (duplicados, imputación por tipo de columna) e Integración (variable derivada `vuelo_escala` con `pd.cut`, un caso de Binning). | Módulo 03 (Filminas 19-20) |
| **Frecuencias Absolutas y Relativas** | Tabla de frecuencias sobre `clase_vuelo` con `.value_counts()` y `.value_counts(normalize=True)`. | Módulo 03 (Filmina 21) |
| **Bloque 2** | Media, mediana, desviación estándar e IQR sobre `pasajeros`; boxplot por clase de vuelo. | Módulo 01 (Filminas 03-05) |
| **Bloque 3** | Histograma+KDE y heatmap de correlación de Pearson sobre variables operativas del vuelo. | Repaso de Clase 05, aplicado sobre el dataset nuevo |
| **Feature Engineering: Ratios y Proporciones** | Variable derivada `factor_ocupacion` (pasajeros/asientos) y su Binning en Baja/Media/Alta ocupación. | Módulo 05 (Filmina 33) |
| **Bloque 4** | `LabelEncoder` + `StandardScaler` + PCA a 2 componentes. | Módulo 02 (estandarización) — LabelEncoder y PCA se adelantan de la próxima unidad. |
| **Normalización (Min-Max Scaling)** | `MinMaxScaler` sobre `pasajeros`/`asientos`, con `train_test_split` **antes** del `fit`. | Módulo 02 (Filminas 10, 13) |
| **Entregable: Pre-entrega "Tienda TechWorld"** | Consigna completa + mini dataset ilustrativo para explorar. | Módulo 07 (Filminas 51-53) |

**Nota pedagógica**: el Bloque 4 es el único punto del notebook que se adelanta a contenido de la próxima unidad (codificación de categóricas y PCA). Si se quiere mantener el notebook estrictamente dentro del temario de hoy, se puede correr igual como demostración de "hacia dónde vamos", aclarando explícitamente que `LabelEncoder` y PCA se explican recién en la próxima clase.

### Bloque 0 — Repaso de la Clase Anterior (Clase 05)

**Por qué arranca acá el notebook, y no directo con el dataset de la clase**: los Bloques 2 y 3 van a usar `sns.boxplot`, `sns.histplot` y `sns.heatmap` sin volver a explicarlos — se asume que ya se saben. Este bloque existe para refrescarlos en dos minutos, sobre un dataset chico e inventado, antes de que aparezcan "en serio" sobre 1 millón de filas.

**Qué decir mientras corre:**

- Se genera un `demo` de 500 filas con una variable `tiempo_entrega` que tiene 480 valores normales (media 20, desvío 4) y 20 valores "de tormenta" (media 180, desvío 15) — la misma historia de la Filmina 03, ahora con datos de verdad para graficar.
- `fig, axes = plt.subplots(1, 3, figsize=(15, 4))` crea **una Figure con tres Axes en fila** — el repaso explícito del modelo Figure/Axes: no son tres gráficos sueltos, son tres `Axes` que viven dentro de la misma `fig` y se referencian por posición (`axes[0]`, `axes[1]`, `axes[2]`).
- **Panel 1 — Histograma + KDE**: mostrar cómo la curva KDE deja ver los 20 valores de tormenta como una segunda "joroba" chiquita a la derecha, casi invisible en los bins del histograma solo.
- **Panel 2 — Boxplot**: los 20 valores de tormenta van a aparecer como puntos sueltos por encima del bigote superior — el mismo umbral `Q3 + 1.5×IQR` de la Filmina 39, ahora visible en un gráfico real.
- **Panel 3 — Heatmap de correlación**: con solo dos columnas el heatmap es mínimo a propósito — alcanza para recordar que el rango de colores va de -1 a 1 y que la diagonal siempre da 1.

**Para profundizar si alguien pregunta por qué generamos datos en vez de usar el dataset real ya**: porque el dataset real (`base_microdatos.csv`) todavía no está cargado ni limpio en este punto del notebook (eso pasa recién en el Paso 0 y el Bloque 1) — el repaso necesita ser autocontenido para poder correrse de forma aislada, sin depender de pasos posteriores.

### Paso 0 y Bloque 1 — Exploración, Limpieza e Integración

El Paso 0 corre una auditoría automática (`isnull().sum()`, `duplicated().sum()`) y dispara alertas si una columna supera 50.000 nulos — acá aparecen `origen_provincia` y `destino_provincia` con más de 220.000 faltantes cada una. El Bloque 1 no borra esas filas de una: primero **deduce la causa** cruzando los nulos con `origen_pais != 'Argentina'` (son vuelos internacionales, que lógicamente no tienen provincia argentina), y recién ahí decide cómo imputar cada columna según su tipo. Cierra con la creación de `vuelo_escala` (`pd.cut` sobre `pasajeros`) — el primer caso de Binning del notebook, antes incluso de que la filmina lo nombre formalmente en el Módulo 05.

**Qué decir**: este es el ejemplo perfecto para instalar la idea de "no borres antes de investigar" — un analista junior ve 220.000 nulos y piensa en eliminarlos; el código demuestra que, cruzando una segunda columna, esos nulos tenían una explicación de negocio perfectamente válida.

### Frecuencias Absolutas y Relativas

Construye la tabla de frecuencias de `clase_vuelo` con `value_counts()` (absoluta) y `value_counts(normalize=True) * 100` (relativa), y las junta en un solo DataFrame. **Qué decir**: es el ejemplo de la Filmina 21 pero con datos reales — remarcar que mirar solo el conteo absoluto no permitiría comparar esta distribución con la de otro país o período sin saber cuántos vuelos totales hubo en cada uno.

### Bloque 2 y Bloque 3 — Estadística Descriptiva y Distribuciones

El Bloque 2 calcula media, mediana, desviación estándar e IQR sobre `pasajeros`, y deduce automáticamente si hay sesgo comparando media vs. mediana (en este dataset, media > mediana → sesgo a la derecha, Filmina 05). El Bloque 3 retoma el histograma+KDE y el heatmap del Bloque 0, ahora sobre `pasajeros`, `asientos` y `vuelos` del dataset real, y calcula el coeficiente de Pearson entre pasajeros y asientos. **Qué decir**: es el momento de cerrar el círculo con el Bloque 0 — "esto es exactamente lo mismo que repasamos al principio, con datos reales en vez del ejemplo inventado".

### Feature Engineering: Ratios y Proporciones

Crea `factor_ocupacion = pasajeros / asientos` (con `.replace(0, np.nan)` para evitar división por cero) y lo agrupa en Baja/Media/Alta con `pd.cut` — el mismo patrón de Binning que `vuelo_escala` en el Bloque 1, ahora aplicado a un **ratio** en vez de a una variable cruda. **Qué decir**: `pasajeros` por sí sola no dice si un vuelo es rentable; el ratio sí — es la variable que realmente le importaría a un analista de revenue management.

### Bloque 4 y Normalización (Min-Max Scaling)

El Bloque 4 codifica categóricas con `LabelEncoder`, estandariza con `StandardScaler` y reduce a 2 componentes con PCA — remarcar la salvedad de la nota pedagógica de arriba (LabelEncoder y PCA son de la próxima unidad). El bloque de Normalización que sigue es el que corresponde de lleno al temario de hoy: aplica `MinMaxScaler` sobre `pasajeros`/`asientos`, pero **primero divide en train/test con `train_test_split` y recién ajusta el escalador sobre el train** — es la puesta en práctica exacta de la regla de oro de la Filmina 13, a diferencia del Bloque 4 (que sí ajusta sobre todo el dataset, por simplicidad, ya que el objetivo ahí es solo mostrar PCA).

### Entregable: Pre-entrega "Tienda TechWorld"

Última celda del notebook. La consigna completa está desarrollada en la sección **Pre-entrega**, más abajo — acá el notebook solo arma un mini `DataFrame` de 10 filas con la misma "suciedad" descripta (fechas en dos formatos, "Mouse"/"ratón"/"MOUSE ", precios con `"N/A"`, edades negativas o mayores a 150) para que el estudiante pueda explorarla con `.info()` antes de escribir el informe.

**Qué decir, y remarcarlo con claridad**: esta celda **no resuelve el ejercicio** — es a propósito. La clasificación de variables, la detección de errores y la propuesta de limpieza son el trabajo que tiene que hacer el estudiante; el notebook solo le da el material crudo para mirar.

---

## Pre-entrega: "Tienda TechWorld"

✅ **Entregable evaluado del Módulo.** Esta es la que se corrige y suma al proyecto — todas las demás prácticas de la clase son guiadas y no evaluables.

### El dataset

Un CSV crudo con las ventas del último mes de una tienda de electrónica:

| Columna | Problema detectado |
|---|---|
| `ID_Transaccion` | Número correlativo — sin problemas. |
| `Fecha` | Formatos mixtos: `DD/MM/AAAA` y `MM/DD/AAAA`. |
| `Producto` | `"Mouse"`, `"ratón"`, `"MOUSE "` — mismo producto, tres etiquetas distintas. |
| `Precio` | Valores numéricos, pero algunos campos son `"N/A"`. |
| `Edad_Cliente` | Algunos valores negativos o superiores a 150. |

### Qué tiene que presentar el estudiante

1. **Clasificación de variables**: cuáles son cualitativas y cuáles cuantitativas.
2. **Detección de errores**: al menos 3 problemas de "suciedad" encontrados.
3. **Propuesta de limpieza**: qué técnica aplicaría por cada error (normalización de etiquetas, imputación, eliminación de atípicos).
4. **Muestra vs. Población**: si la tienda tiene 50 sucursales pero solo se entregaron datos de la sucursal central, por qué esa muestra podría estar sesgada para una predicción de ventas nacional.

**Formato de entrega**: documento (PDF o Word) — no es un notebook.

---

## Síntesis y Conexión Final

Toda la clase se puede resumir en una idea: **antes de modelar, hay que poder confiar en el dato**. La estadística descriptiva (Módulo 01) da el vocabulario para resumir una variable; el escalado (Módulo 02) nivela el campo de juego entre variables de distinta magnitud; el preprocesamiento (Módulos 03-04 y 06) es el trabajo de limpieza que hace que esas medidas signifiquen algo real; el Feature Engineering (Módulo 05) es donde el conocimiento de negocio se convierte en una variable que un modelo puede usar; y todo eso confluye en la Pre-entrega, donde el estudiante tiene que demostrar que puede diagnosticar un dataset sucio real sin que nadie le diga paso a paso qué hacer.

En la próxima unidad se retoma esto para codificar variables categóricas (`LabelEncoder`/`OneHotEncoder`) y aplicar reducción de dimensionalidad (PCA) — ambas ya asomadas, a propósito, en el Bloque 4 del notebook de hoy.
