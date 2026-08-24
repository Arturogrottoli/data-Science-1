# Clase 06 — Estadística y Preprocesamiento: Fundamentos para el Análisis de Datos

**Curso de Data Science I · Clase 06** — el checkpoint de la primera Pre-entrega del proyecto.

Esta guía sigue el **orden exacto de las 53 filminas** de `Clase06.html`, organizadas en 7 módulos. Cada sección está etiquetada con la filmina a la que corresponde, para poder ir mostrando la diapositiva y leyendo/ampliando en paralelo — el contenido de acá suele ser más profundo que lo que dice literalmente la filmina. Al final hay una guía del notebook (`Clase_6_Fundamentos_de_Ciencia_de_Datos_.ipynb`) y el detalle de la Pre-entrega evaluada.

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

**Qué decir:**

- **Media**: suma de todos los valores dividido la cantidad. Es la más conocida, pero extremadamente sensible a outliers. Si la mayoría de los pedidos llegan en 20 minutos pero un par de veces por semana (tormenta, falta de repartidores) tardan 180, la media dice "45 minutos" — una cifra que no representa la experiencia de casi nadie.
- **Mediana**: el valor que queda justo en el centro al ordenar los datos. Es la medida más robusta ante extremos — ideal para salarios, tiempos de entrega, cualquier variable con cola larga.
- **Moda**: el valor que más se repite. Es la única de las tres que también sirve para variables categóricas (el color de auto más vendido). Un dataset puede ser bimodal, o no tener moda si nada se repite.

*Regla práctica para instalar*: si media y mediana son parecidas, la distribución es razonablemente simétrica; si difieren mucho, hay sesgo (Filmina 05).

## Filmina 04 — Medidas de Dispersión: Rango, Varianza y Desviación Estándar

**Qué decir:**

- **Rango**: máximo − mínimo. Solo usa dos datos — no dice nada de lo que pasa en el medio.
- **Varianza**: promedio de las distancias al cuadrado respecto a la media. El problema: al elevar al cuadrado, las unidades también cambian (si medís metros, la varianza da metros²), lo que la hace difícil de interpretar a simple vista.
- **Desviación estándar**: raíz cuadrada de la varianza — vuelve a las unidades originales. Es la "distancia típica" a la que están los datos respecto a la media.

*Analogía del Río A y Río B (usarla en el pizarrón)*: ambos ríos tienen 1.5 m de profundidad **media**. El Río A siempre mide 1.5 m; el Río B tiene zonas de 0.1 m y pozos de 4 m. Conocer solo el promedio y cruzar el Río B a ciegas puede ser fatal — la dispersión es la que te salva.

## Filmina 05 — Lectura Conjunta: Centro y Dispersión

**Qué decir:**

- **Media ≈ Mediana** → distribución simétrica (campana).
- **Media > Mediana** → valores grandes "tiran" la media hacia arriba (sesgo a la derecha) — típico de salarios o precios de vivienda.
- **Media < Mediana** → valores pequeños tiran hacia abajo (sesgo a la izquierda).
- **Coeficiente de Variación**: comparar la desviación estándar del peso de un elefante contra la de un ratón es injusto — el elefante "gana" siempre porque es más grande. El CV normaliza esa comparación y permite comparar manzanas con naranjas.

## Filmina 06 — Aplicaciones Reales y Errores Comunes

**Qué decir (tres casos de industria):**

- **Fintech**: la mediana del monto de préstamos evita que "grandes préstamos corporativos" distorsionen el perfil del cliente típico.
- **Streaming**: la moda revela géneros populares; media y desviación del "tiempo de visionado" detectan cuentas compartidas (comportamiento anómalo).
- **Manufactura**: una desviación estándar alta en el diámetro de un tornillo es un problema de calidad, aunque la media sea perfecta — el objetivo es la consistencia, no solo el promedio.

**Errores a marcar**: usar solo la media (reportar siempre ambas), olvidar que la varianza cambia de unidades, e ignorar que la moda puede revelar un valor "por defecto" del sistema (como el 999 para datos faltantes).

## Filmina 07 — Práctica (no entregable): Estadística Descriptiva sobre tus Propios Datos

Ejercicio de calentamiento, no entregable: elegir una columna numérica de un dataset ya trabajado (precios, tiempos de envío, montos de venta), calcular las 7 métricas del módulo, redactar una interpretación por métrica, y comparar media vs. mediana para detectar sesgo. Sirve como puente directo al Bloque 2 del notebook.

---

# Módulo 02 — Normalización y Estandarización (Filminas 08-15)

## Filmina 08 — División de Módulo

El objetivo del módulo en una frase: poner a todos los datos en un "campo de juego nivelado", sin importar sus unidades originales.

## Filmina 09 — El Problema de la Escala: ¿Por Qué Escalar?

**Qué decir:**

- *Analogía*: un lanzador de pesas llega a 20 metros; un saltador de altura, a 2.30 metros. El 20 es mayor que el 2.30, pero eso no dice nada sobre quién es mejor atleta — están en escalas distintas.
- En Data Science pasa lo mismo: si el Ingreso Mensual es $5.000 y la Edad es 30, un modelo puede "pensar" que el ingreso importa 166 veces más que la edad, solo porque el número es más grande.
- **Dónde impacta concretamente**: algoritmos de **distancia** (K-Nearest Neighbors) — una variable con rango 0-1.000.000 domina totalmente a otra de rango 0-1; y algoritmos de **gradiente** (Regresión Logística, Redes Neuronales) — escalas dispares hacen que el entrenamiento "rebote" y tarde más, o no converja.

## Filmina 10 — Normalización (Min-Max Scaling)

**Fórmula**: `X_norm = (X − X_min) / (X_max − X_min)`

**Qué decir**: comprime todos los valores de una columna a un rango fijo [0, 1]. El mínimo se vuelve 0, el máximo se vuelve 1. *Ejemplo en el pizarrón*: velocidades de procesador [2.0, 3.0, 4.0, 5.0] GHz → mínimo 2.0, máximo 5.0 → el valor 3.5 se normaliza como 0.5. Ideal cuando hay límites claros y conocidos (píxeles de 0 a 255, redes neuronales).

## Filmina 11 — Estandarización (Z-Score Scaling)

**Fórmula**: `X_std = (X − μ) / σ`

**Qué decir**: no obliga a los datos a un rango fijo — los centra para que tengan media 0 y desviación estándar 1. *Ejemplo*: estatura con media 170 cm y desviación 10 cm → una persona de 180 cm tiene Z = +1; de 160 cm, Z = −1; de 170 cm, Z = 0. Es la técnica preferida cuando los datos son normales o el algoritmo asume datos centrados (Regresión Lineal, PCA).

## Filmina 12 — Comparativa: ¿Cuándo Usar Cada Una?

Tabla comparativa: rango de salida, media/desviación resultante, sensibilidad a outliers (Min-Max es muy sensible: un solo extremo cambia todo el rango; Z-Score es más robusta) y cuándo elegir cada una. **Punto para remarcar**: en la práctica profesional, la estandarización suele ser el punto de partida por defecto.

## Filmina 13 — El Peligro de la "Fuga de Información" (Data Leakage)

**Qué decir — la regla de oro, palabra por palabra:**

1. Dividir primero los datos en **train** y **test**.
2. Ajustar (`fit`) el escalador usando **solo** el train.
3. Transformar train y test con esos valores ya calculados.

*Analogía*: es como estudiar para un examen mirando las preguntas del examen (test) para decidir qué estudiar (train) — la nota sale artificialmente alta, pero no aprendiste nada realmente. Este es el error más crítico y más común entre quienes recién empiezan.

## Filmina 14 — Errores Comunes en el Escalado

Tabla de tres errores: creer que el escalado elimina outliers (un millonario en un dataset de sueldos medios queda en 1.0 y aplasta al resto cerca de 0.0001 — hay que tratar los outliers **antes** de escalar); escalar variables categóricas (Género 0/1 o ID_Cliente no tienen magnitud real); y pensar que el escalado cambia la "forma" de la distribución (si los datos tienen forma de "L", seguirán con forma de "L" después de estandarizar, solo que centrados en cero).

## Filmina 15 — Práctica (no entregable): Escalado en Python

Puente directo al notebook: elegir dos variables con escalas muy distintas, dividir en train/test **antes** de transformar, aplicar `MinMaxScaler` y `StandardScaler` con `.fit()` solo en train, y verificar con `.describe()`. Error a remarcar una vez más: nunca `.fit_transform()` sobre el test.

---

# Módulo 03 — Estadística y Preprocesamiento: Fundamentos (Filminas 16-23)

## Filmina 16 — División de Módulo

El gancho: "Garbage in, garbage out" — si los datos de entrada son ruidosos, el resultado será erróneo, sin importar qué tan sofisticado sea el modelo que viene después.

## Filmina 17 — El Puente entre los Datos Crudos y el Conocimiento

**Qué decir:**

- *Analogía del restaurante saludable*: 500 personas dejaron sus opiniones en papelitos sueltos — algunos con edad, otros no, precios en distintas monedas. Antes de decidir si el menú debe ser más barato, hay que limpiar esos papeles (preprocesamiento) y resumir la información útil (estadística).
- **Preprocesamiento** = la "fontanería": que los datos sean consistentes, completos y estén en el formato adecuado.
- **Estadística** = la interpretación: entender qué dice esa masa de datos ya limpia sobre la realidad.
- *Caso concreto*: un empleado con salario "0" (dato faltante mal cargado) y el CEO con "1.000.000" — el promedio simple no representa a nadie en la oficina. El preprocesamiento detecta el "0" como error; la estadística advierte que el millón distorsiona la realidad.

## Filmina 18 — Población, Muestra y Tipos de Variables

Tabla de referencia rápida: **Población** (todos los elementos a estudiar) vs. **Muestra** (subconjunto representativo — casi siempre se trabaja con muestras). Luego los cuatro tipos de variable: **Cualitativas Nominales** (sin orden: color de ojos), **Cualitativas Ordinales** (con orden: bajo/medio/alto), **Cuantitativas Discretas** (conteos: número de hijos) y **Cuantitativas Continuas** (cualquier valor en un rango: peso, altura).

## Filmina 19 — El Flujo de Trabajo del Preprocesamiento

Cuatro tiles: **valores faltantes** (¿eliminar o imputar con media/mediana?), **detección de ruido** (errores de dedo evidentes, como una edad de 250 años), **integración** (unificar fuentes distintas en una sola estructura) y **transformación** (normalización, agregación de datos diarios a mensuales).

## Filmina 20 — Ejemplo del Mundo Real: el Caso del E-commerce

Tabla con tres clientes (A, B, C) donde aparecen los tres problemas típicos a la vez: edad faltante, ciudad escrita con mayúsculas/minúsculas distintas ("Madrid"/"madrid"/"MADRID"), y un gasto con error de captura evidente. Sirve para que el grupo practique identificar los tres tipos de "suciedad" en un solo ejemplo — casi un anticipo en miniatura de la Pre-entrega.

## Filmina 21 — Frecuencias y el Riesgo del Sesgo de Muestra

**Qué decir:**

- **Frecuencia absoluta**: cuántas veces aparece un dato (10 personas prefieren el iPhone).
- **Frecuencia relativa**: qué porcentaje del total representa (esas 10 personas son el 20%).
- *Caso de sesgo*: si la muestra de "usuarios de tecnología" es 90% menores de 20 años, las conclusiones sobre facilidad de uso van a estar sesgadas y no van a representar a la población general — por más que la frecuencia esté bien calculada.

## Filmina 22 — Errores Comunes y Mejores Prácticas

Tabla de tres errores clásicos (confundir media con mediana en salarios desiguales, ignorar datos faltantes sin preguntar por qué faltan, no estandarizar formatos de texto) con su mejor práctica correspondiente. Cierre: visualizar antes de calcular, documentar los cambios, y mantener el escepticismo profesional — si un dato parece demasiado bueno para ser verdad, probablemente sea un error.

## Filmina 23 — Práctica (no entregable): Diagnóstico de una App de Fitness

Escenario con 100 usuarios y cinco columnas problemáticas (Edad con "0" y un "200", Ciudad con tres variantes de escritura, Suscripción con mayúsculas inconsistentes). Pide clasificar las variables, decidir media vs. mediana ante outliers de peso, y proponer qué hacer con 15 registros sin dato de peso.

---

# Módulo 04 — El Arte de Ordenar el Caos (Filminas 25-30)

*(Filmina 24 es el Break — 10 minutos.)*

## Filmina 25 — División de Módulo

Reencuadre después del break: de una montaña de "datos en bruto" a una historia coherente sobre ellos — la metáfora es la biblioteca con los libros tirados en el piso.

## Filmina 26 — GIGO: el Flujo del Científico de Datos en 5 Pasos

**Qué decir**: Garbage In, Garbage Out. Los cinco pasos, en orden: **Preguntar** (¿cuál es el perfil de nuestro cliente ideal?) → **Recolectar** (encuestas, bases de datos, sensores) → **Preprocesar** (eliminar ruido y estructurar) → **Representar** (tablas y gráficos) → **Interpretar** (extraer conclusiones). Es el mismo flujo que después se ve, aplicado, en el notebook.

## Filmina 27 — La Tabla de Frecuencias: el Primer Puente

**Caso de la fábrica (usarlo en el pizarrón)**: 10 fallas en el Turno Mañana y 10 fallas en el Turno Noche — a simple vista (frecuencia absoluta) parecen iguales. Pero el Turno Mañana produjo 1.000 piezas y el Turno Noche solo 100: la tasa de falla real es 1% vs. 10%. Moraleja: el preprocesamiento estadístico permite comparar "manzanas con manzanas" usando proporciones, no totales brutos.

## Filmina 28 — Visualización: el Modelo del "Zoom"

**Qué decir**: la **Tabla** es zoom máximo (cada dato individual), el **Gráfico** es zoom medio (la forma del conjunto: barras, histogramas, pie charts) y la **Medida de tendencia central** es zoom alejado (solo el punto central). Un gráfico no es decoración — es una herramienta de preprocesamiento visual que ayuda a detectar patrones que las tablas ocultan.

## Filmina 29 — Errores Comunes: el Engaño de la Media y el Contexto

Tres trampas: el engaño de la media (9 personas que ganan poco y 1 que gana mucho — el promedio no describe al "típico"), confundir frecuencia absoluta con relativa (los totales engañan si los grupos tienen tamaños distintos), e ignorar el contexto de negocio (un pico de ventas el 14 de febrero no es un error de datos, es San Valentín — investigar antes de "limpiar").

## Filmina 30 — Práctica (no entregable): el Reporte del Café Popular

Datos de ventas de una cafetería con fallos de registro reales: nombres de categoría inconsistentes ("CAFE AMERICANO" vs. "Café Americano"), un dato faltante, y un valor atípico (un croissant a $45 entre bebidas de $2-3.50). Pide armar la tabla de frecuencias corrigiendo antes los nombres, y explicar cómo el croissant afecta el precio promedio.

---

# Módulo 05 — Feature Engineering (Filminas 31-34)

## Filmina 31 — División de Módulo

El gancho: "limpiar la mesa no es lo mismo que cocinar" — después de limpiar los datos, el Feature Engineering es donde entra la creatividad y el conocimiento de negocio.

## Filmina 32 — ¿Qué es el Feature Engineering?

**Qué decir**: usar conocimiento de negocio para crear nuevas variables a partir de las existentes, con el fin de que los algoritmos entiendan mejor el problema. *Ejemplo*: Ingresos y Gastos Mensuales por separado son útiles, pero **Capacidad de Ahorro** (Ingresos − Gastos) es mucho más informativa para un modelo de scoring crediticio. *Ejemplo de delivery*: el dato crudo `hora_pedido = 14:00` se convierte en `es_hora_pico = 1`, porque sabemos que entre las 13:00 y 15:00 la demanda colapsa.

## Filmina 33 — Ratios, Variables Temporales y Binning

Tres tiles: **Ratios y Proporciones** ("Ventas por Metro Cuadrado" en vez de "Ventas Totales"; CTR en marketing), **Variables Temporales** (recencia, estacionalidad, antigüedad del cliente) y **Binning** (convertir Edad continua en categorías como "Generación Z" o "Millennials").

## Filmina 34 — La Regla de Oro: el Contexto es Rey

**Qué decir**: crear cientos de variables al azar es un error común de principiantes — más no es mejor. Una variable sin lógica de negocio suele ser "ruido" que confunde al modelo. Antes de crear una columna, la pregunta clave: si fuera un experto en este negocio, ¿este dato me ayudaría a decidir algo?

---

# Módulo 06 — Consolidación en la Práctica (Filminas 35-42)

## Filmina 35 — División de Módulo

Reencuadre: de entender promedios y desviaciones a normalizar, detectar outliers y razonar con probabilidad — se cierra el círculo de todo lo visto hasta acá.

## Filmina 36 — Práctica (no entregable): Auditoría de Datos e Interpretación

Escenario introductorio del módulo: ventas de una sucursal (muestra) para decidir la estrategia de compras de toda la región (población). *(Nota: en la filmina esta práctica aparece resumida en el cierre del módulo — filmina 41 — pero conviene anticipar el escenario acá para que el grupo lo tenga presente en las filminas siguientes.)*

## Filmina 37 — El Flujo Completo: de la Recogida a la Inferencia

Cuatro tiles retomando el flujo GIGO ya visto (Filmina 26), ahora cerrando el círculo con **Inferencia**: usar lo aprendido en la muestra para predecir sobre el grupo más grande. *Refuerzo*: sin preprocesamiento, la estadística descriptiva puede mentir — un sueldo mal tipeado como "1.000.000.000" dispara el promedio artificialmente.

## Filmina 38 — Población, Muestra e Individuo: el Riesgo del Sesgo

**Qué decir**: se agrega el concepto de **Individuo** (cada elemento único dentro de la población o muestra) a los ya vistos de Población y Muestra. *Caso de industria*: un e-commerce encuesta sobre su nuevo diseño solo a quienes compraron hoy — esa muestra está sesgada porque ignora a quienes entraron y se fueron frustrados por el diseño mismo.

## Filmina 39 — Gestión de Outliers: ¿Borrar o Investigar?

**Qué decir**: no siempre hay que borrar los outliers. A veces es un error de digitación (se borra); otras veces es un caso de fraude bancario o una oportunidad de mercado única (se investiga). El **Boxplot** es la herramienta visual para detectarlos antes de decidir qué hacer con ellos.

## Filmina 40 — Lidiar con la Incertidumbre: Introducción a la Probabilidad

**Qué decir**: la probabilidad va de 0 (imposible) a 1 (seguro), con 0.5 como el caso de lanzar una moneda. Un **experimento aleatorio** es un proceso donde no sabemos el resultado exacto. *Analogía del pronóstico del tiempo*: cuando el meteorólogo dice 90% de probabilidad de lluvia, no está "adivinando" — analizó datos históricos (estadística descriptiva) de días similares (muestra) y llegó a una conclusión lógica (inferencia).

## Filmina 41 — Errores Comunes: Tabla de Trampas Frecuentes

Tabla de cuatro errores con su forma correcta: ignorar valores faltantes, confundir correlación con causalidad, no normalizar antes de comparar, y confundir descriptiva con inferencia (afirmar que "toda la población" se comporta así por una muestra chica).

## Filmina 42 — Práctica (no entregable): Auditoría de Datos e Interpretación

El ejercicio completo: definir población y muestra en el caso de la cadena de supermercados; detectar anomalías en una lista de precios `[10, 12, 11, 15, 2, 500, 13, 11]`; calcular frecuencia absoluta y relativa de compradores de productos orgánicos; y explicar por qué Edad (18-90) y Gasto Total (500-50.000) no pueden ir directo a un modelo sin normalizar.

---

# Módulo 07 — Pre-entrega: El Puente entre los Datos y las Decisiones (Filminas 43-53)

## Filmina 43 — División de Módulo

El anuncio: esta es la última parada teórica antes del entregable evaluado. Todo lo visto en los seis módulos anteriores converge acá.

## Filmina 44 — La Estadística como Ciencia de la Incertidumbre

**Qué decir:**

- *Analogía de la escena del crimen*: papeles sueltos, fotos borrosas, testimonios contradictorios — datos ruidosos y potencialmente engañosos. La estadística recoge, organiza, resume e interpreta datos para decidir cuando no tenemos toda la información.
- *Analogía de la sopa*: no hace falta tomarse toda la olla (población) para saber si está bien de sal — basta una cucharada bien mezclada (muestra). Pero si se prueba solo la superficie sin revolver, la conclusión será errónea — el preprocesamiento es, en parte, asegurar que esa "mezcla" sea la adecuada antes de probarla.

## Filmina 45 — Preprocesamiento: la Cocina de los Datos

**Qué decir**: los datos son "sucios" por naturaleza — valores faltantes (clientes que no completaron su edad), ruido y atípicos (un sensor que marca 500°C en una oficina), inconsistencias ("España", "Esp", "España " en la misma columna). La línea de ensamblaje completa: Dato Bruto → Limpieza → Organización → Representación → Interpretación.

## Filmina 46 — Variables y Frecuencias: el Lenguaje de la Medición

**Caso Ciudad A vs. Ciudad B (usarlo en el pizarrón)**: 500 compras en la Ciudad A (con 1.000.000 de habitantes → 0.05%) vs. 50 compras en la Ciudad B (con 100 habitantes → 50%). La frecuencia relativa cuenta la historia real: el producto es un éxito rotundo en B, no en A — el número absoluto solo. engañaba.

## Filmina 47 — El Rol Crítico de la Probabilidad

**Qué decir**: 0 es imposible, 1 es seguro, 0.5 es lanzar una moneda. *Ejemplo del filtro de spam*: el modelo no dice "esto es basura" — dice "hay 0.98 de probabilidad de que sea basura". Esa graduación es la que permite diseñar sistemas que solo actúan cuando la confianza es suficientemente alta.

## Filmina 48 — Errores Comunes en Preprocesamiento y Estadística

Cuatro errores: pensar que "más datos" es "mejor análisis" (un millón de registros mal recolectados solo hace el error más grande y costoso), confundir muestra con población (una encuesta de Twitter no representa a todo un país), ignorar outliers sin pensar, y tratar variables categóricas como numéricas (promediar códigos postales no tiene sentido).

## Filmina 49 — Aplicaciones en la Industria Real

Tres casos: **Streaming** (normaliza tiempo visto en celular vs. TV 4K, trata los faltantes de series abandonadas), **Políticas Públicas** (datos censales mal tabulados pueden hacer que un hospital se construya en el lugar equivocado), **Finanzas** (sin normalizar montos, el modelo de detección de fraude falla).

## Filmina 50 — Síntesis: Puntos Clave para Recordar

Cierre teórico del módulo — buen momento para una pausa y preguntas antes de pasar a la Pre-entrega. Los cuatro puntos: representatividad de la muestra, tratamiento distinto según tipo de variable, frecuencia relativa por sobre absoluta, y el preprocesamiento como ciclo continuo.

## Filminas 51-53 — Entregable: Pre-entrega "Tienda TechWorld"

Ver la sección **Pre-entrega**, más abajo, con el detalle completo del dataset y de las 4 consignas.

## Filmina 53 (última) — ¿Dudas? ¿Consultas?

Cierre de la clase — espacio abierto antes de que el grupo se ponga a trabajar en la Pre-entrega.

---

## Guía del Notebook (`Clase_6_Fundamentos_de_Ciencia_de_Datos_.ipynb`)

El notebook aplica el contenido de la clase sobre un dataset real de más de 1 millón de registros de vuelos en Argentina (`base_microdatos.csv`, conectividad aérea — datos.gob.ar). Va en el mismo orden lógico que la filmina, con una salvedad: agrupa varios módulos teóricos en un mismo bloque de código cuando comparten el mismo dataset y el mismo paso del flujo.

| Bloque | Contenido | Módulo de la filmina que aplica |
|---|---|---|
| **Bloque 0** | Repaso de Clase 05: `fig, ax`, histograma+KDE, boxplot, heatmap — con un mini dataset sintético antes de tocar el dataset real. | Repaso, no es contenido nuevo |
| **Paso 0** | Diagnóstico automático de calidad: nulos y duplicados, con alertas si superan un umbral de negocio. | Módulo 03 (Filmina 19) |
| **Bloque 1** | Limpieza (duplicados, imputación por tipo de columna) e Integración (variable derivada `vuelo_escala` con `pd.cut`, un caso de Binning). | Módulo 03 (Filminas 19-20) |
| **Frecuencias Absolutas y Relativas** | Tabla de frecuencias sobre `clase_vuelo` con `.value_counts()` y `.value_counts(normalize=True)`. | Módulo 03 (Filmina 21) |
| **Bloque 2** | Media, mediana, desviación estándar e IQR sobre `pasajeros`; boxplot por clase de vuelo. | Módulo 01 (Filminas 03-05) |
| **Bloque 3** | Histograma+KDE y heatmap de correlación de Pearson sobre variables operativas del vuelo. | Repaso de Clase 05, aplicado sobre el dataset nuevo |
| **Feature Engineering: Ratios y Proporciones** | Variable derivada `factor_ocupacion` (pasajeros/asientos) y su Binning en Baja/Media/Alta ocupación. | Módulo 05 (Filmina 33) |
| **Bloque 4** | `LabelEncoder` + `StandardScaler` + PCA a 2 componentes. | Módulo 02 (estandarización) — LabelEncoder y PCA se adelantan de la próxima unidad, a tener en cuenta si se quiere acotar el bloque a lo estrictamente visto hoy. |
| **Normalización (Min-Max Scaling)** | `MinMaxScaler` sobre `pasajeros`/`asientos`, con `train_test_split` **antes** del `fit` — el ejemplo correcto de la regla de oro contra la Fuga de Información. | Módulo 02 (Filminas 10, 13) |
| **Entregable: Pre-entrega "Tienda TechWorld"** | Consigna completa + un mini dataset ilustrativo con la misma "suciedad" descripta, para explorar antes de escribir el informe (no resuelve el ejercicio). | Módulo 07 (Filminas 51-53) |

**Nota pedagógica**: el Bloque 4 es el único punto del notebook que se adelanta a contenido de la próxima unidad (codificación de categóricas y PCA). Si se quiere mantener el notebook estrictamente dentro del temario de hoy, se puede correr igual como demostración de "hacia dónde vamos", aclarando explícitamente que `LabelEncoder` y PCA se explican recién en la próxima clase.

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
