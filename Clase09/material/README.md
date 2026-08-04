# Clase 09: Aprendizaje No Supervisado — Guía Completa para el Docente

Esta guía es el **libreto de apoyo para dictar la Clase 09**. Reúne, en un solo lugar y con más profundidad de la que entra en una diapositiva, toda la teoría que aparece en:

- **`Clase 09_teoria.pdf`** — el material teórico oficial de la unidad (6 secciones: de la introducción al aprendizaje no supervisado hasta el panorama comparativo de métodos).
- **`Clase09.html`** — las diapositivas que se proyectan en clase (36 filminas).

> **Estado de esta guía**: por ahora cubre teoría (PDF + filminas) y el repaso de aprendizaje supervisado como puente desde la Clase 08. Todavía no incorpora el notebook de la clase ni un dataset real propio — eso queda para una próxima iteración.

---

## Índice

- [Mapa rápido de la clase](#mapa-rápido-de-la-clase)
- [Módulo 0 — Repaso: Aprendizaje Supervisado](#módulo-0--repaso-aprendizaje-supervisado-puente-desde-la-clase-08)
- [Módulo 1 — ¿Qué es el Aprendizaje No Supervisado?](#módulo-1--qué-es-el-aprendizaje-no-supervisado)
- [Módulo 2 — Reglas de Asociación](#módulo-2--reglas-de-asociación)
- [Módulo 3 — K-Means y la Elección de k](#módulo-3--k-means-y-la-elección-de-k)
- [Módulo 4 — Clustering Jerárquico y DBSCAN](#módulo-4--clustering-jerárquico-y-dbscan)
- [Módulo 5 — PCA: Reducción de Dimensionalidad](#módulo-5--pca-reducción-de-dimensionalidad)
- [Módulo 6 — Panorama de Métodos (Síntesis)](#módulo-6--panorama-de-métodos-síntesis)

---

## Mapa rápido de la clase

Para seguir la clase en paralelo con `Clase09.html` (36 filminas) sin perderte:

| # | Módulo | Slides | Idea central |
|---|---|---|---|
| 0 | Repaso: Aprendizaje Supervisado | *(sin slides — repaso previo, puente desde Clase 08)* | Contraste con lo que ya sabés: datos etiquetados, `f(X) → y`, para entender por qué hoy no hay `y` |
| — | Portada | 01 | Presentación de la clase |
| 1 | ¿Qué es el Aprendizaje No Supervisado? | 02–06 | Sin etiquetas: clustering, reducción de dimensionalidad, reglas de asociación |
| 2 | Reglas de Asociación | 07–10 | Apriori, FP-Growth, y las métricas support/confidence/lift |
| 3 | K-Means y la Elección de k | 11–17 | El algoritmo de clustering más usado, y cómo elegir bien su parámetro clave |
| — | Break del Coder | 18 | Corte de ~10 minutos |
| 4 | Clustering Jerárquico y DBSCAN | 19–24 | Dendrogramas, linkage, densidad, ruido |
| 5 | PCA: Reducción de Dimensionalidad | 25–30 | Covarianza, eigenvectores/eigenvalores, varianza explicada |
| 6 | Panorama de Métodos (Síntesis) | 31–35 | Comparación de las 5 técnicas + demo real de PCA mejorando un modelo |
| — | ¿Dudas? | 36 | Cierre y preguntas |

---

## Módulo 0 — Repaso Express: Aprendizaje Supervisado (puente desde la Clase 08)

**Por qué este módulo**: la Clase 08 cerró el bloque de aprendizaje supervisado. Antes de arrancar con la Clase 09, conviene un repaso corto — no para volver a enseñarlo, sino para que el contraste con lo de hoy quede bien marcado.

### ¿Qué es Machine Learning, en general?

Antes de meternos en "supervisado" puntualmente, vale la pena bajar un escalón más y aclarar la idea más general en la que se apoya toda esta clase. Programar una computadora "a la vieja usanza" significa escribirle instrucciones explícitas para cada caso: "si pasa esto, hacé esto otro". El problema es que hay tareas donde armar esas reglas a mano es imposible — nadie puede escribir manualmente todas las reglas que distinguen una foto de un gato de una foto de un perro, o que dicen exactamente cuándo un mail es spam.

**Machine Learning (aprendizaje automático)** resuelve eso de otra manera: en vez de programarle las reglas a la máquina, se le muestran muchos ejemplos y se deja que ella misma **encuentre los patrones** y arme sus propias reglas internas, a base de repetición y ajuste. Es parecido a cómo un chico aprende a reconocer animales: nadie le da una lista de reglas escritas ("un perro tiene 4 patas, pelo, y ladra") — simplemente ve muchos perros distintos, y con el tiempo su cerebro arma solo el patrón que le permite reconocer un perro nuevo que nunca vio.

Dentro de Machine Learning hay distintas formas de "dejar que la máquina aprenda sola", según qué tipo de datos y qué tipo de ayuda se le da durante ese aprendizaje. La que ya se vio en la Clase 08 —y la que se repasa a continuación— es el **aprendizaje supervisado**; la que arranca hoy es el **aprendizaje no supervisado**, con una diferencia central que se explica más abajo.

### ¿Qué es el aprendizaje supervisado?

La idea de fondo es muy parecida a cómo aprende una persona con ejemplos resueltos: si querés aprender a distinguir mails de spam, lo más fácil es que alguien te muestre miles de mails **ya marcados** como "spam" o "no spam", y con el tiempo empezás a notar patrones (ciertas palabras, remitentes raros, exceso de mayúsculas) que te ayudan a clasificar un mail nuevo que nunca viste. Eso es exactamente lo que hace un modelo de aprendizaje supervisado: se le muestran muchos ejemplos donde la respuesta correcta **ya se conoce**, y el modelo va ajustando sus parámetros internos hasta encontrar una regla (una función matemática) que relacione los datos de entrada con esa respuesta. Una vez entrenado, se usa esa regla para predecir la respuesta de casos **nuevos**, donde no se conoce de antemano.

En la notación que se usa en la jerga de Machine Learning: a las variables de entrada (edad, ingresos, tamaño de un tumor, cantidad de habitaciones de una casa...) se las llama `X`; a la respuesta que se quiere predecir (spam o no, precio de la casa) se la llama `y`. Entrenar un modelo supervisado es, ni más ni menos, buscar una función `f` tal que `f(X)` se parezca lo más posible a `y`, usando los ejemplos históricos donde ambas cosas ya se conocen.

Existen dos grandes familias, según qué tipo de dato es `y`:

| | Clasificación | Regresión |
|---|:---:|:---:|
| **`y` es...** | Una categoría | Un número |
| **Ejemplo** | ¿Tumor maligno? | ¿Precio de la vivienda? |
| **Métricas** | Accuracy, F1, AUC-ROC | MAE, RMSE, R² |

- **Clasificación**: la respuesta que se quiere predecir es una **etiqueta**, elegida entre un grupo cerrado de opciones — "maligno" o "benigno", "spam" o "no spam". No hay término medio: la predicción es una de esas categorías, no un número.
- **Regresión**: la respuesta que se quiere predecir es un **número** que puede tomar cualquier valor — el precio de una casa (podría ser $150.234 o $150.987, cualquier cifra), la temperatura de mañana. Acá sí hay término medio: el modelo puede acertar "más o menos", no es todo o nada.

**Cómo se mide si un modelo de clasificación es bueno** — con un ejemplo concreto: supongamos que se evalúa el modelo sobre 100 tumores donde ya se sabe la respuesta real (90 benignos, 10 malignos).
- **Accuracy** es lo más simple de entender: de esos 100 casos, ¿en cuántos acertó el modelo (dijo "maligno" cuando era maligno, o "benigno" cuando era benigno)? Si acertó en 92, el Accuracy es 92%.
- El problema de quedarse solo con Accuracy: si el modelo fuera tan vago que dijera **siempre** "benigno", sin analizar nada, igual acertaría en los 90 casos benignos y solo fallaría en los 10 malignos — un Accuracy del 90%, que suena bien pero es un modelo completamente inútil (nunca detecta un tumor maligno, que es justo el caso grave que importa detectar).
- Por eso existe **F1**: en vez de mirar el total de aciertos, mira específicamente qué tan bien detecta los casos positivos (los malignos, en este ejemplo) sin ni inventar demasiados falsos positivos ni dejar pasar demasiados casos reales. Un modelo que siempre dice "benigno" tendría un F1 muy bajo, aunque su Accuracy sea alto — por eso F1 es la métrica preferida cuando hay pocos casos de la categoría que más importa detectar (como suele pasar con enfermedades, fraude, etc.).
- Una tercera métrica que se mencionó en la Clase 08 es **AUC-ROC**: la mayoría de los modelos de clasificación en realidad no dicen "sí" o "no" en seco, sino que calculan una **probabilidad** (75% de probabilidad de ser maligno) y después esa probabilidad se convierte en "sí"/"no" según algún umbral (por ejemplo, "sí" si supera 50%). AUC-ROC mide qué tan bien el modelo **ordena** los casos por esa probabilidad — si comparás un caso maligno real contra uno benigno real, tomados al azar, ¿qué tan seguido el modelo le puso mayor probabilidad de "maligno" al que realmente lo era? Un valor de 1 sería un modelo perfecto ordenando casos; un valor de 0,5 sería lo mismo que tirar una moneda al aire, sin ningún poder de discriminación real.

**Cómo se mide si un modelo de regresión es bueno** — con otro ejemplo: un modelo que predice precios de casas.
- **MAE** (Error Absoluto Medio): agarra la diferencia entre lo que predijo el modelo y el precio real de cada casa, y promedia esas diferencias (sin importar si se equivocó "de más" o "de menos"). Si el MAE da $10.000, quiere decir que, en promedio, el modelo se equivoca por $10.000 en cada predicción — un número fácil de interpretar porque está en la misma unidad (dólares) que lo que se está prediciendo.
- **RMSE**: muy parecido al MAE, pero antes de promediar los errores los eleva al cuadrado (y al final saca la raíz cuadrada del resultado). El efecto práctico: un error grande pesa mucho más que varios errores chicos — un modelo que casi siempre acierta bien pero se equivoca feo en un par de casas raras va a tener un RMSE bastante peor que su MAE, mientras que un modelo con errores parejos y moderados va a tener MAE y RMSE parecidos entre sí.
- **R²**: en vez de dar un error en dólares, da un número entre 0 y 1 (a veces se explica como porcentaje) que responde "¿qué tan bien el modelo explica por qué el precio de cada casa es el que es?". Un R² de 1 sería un modelo perfecto (acierta el precio exacto siempre); un R² de 0 significa que el modelo no es mejor que simplemente decir siempre "el precio promedio de todas las casas", sin mirar ninguna variable en particular.

### Los modelos que se vieron en la Clase 08

Estos cinco modelos son las herramientas concretas con las que se resuelven los problemas de clasificación y regresión. Repasarlos uno por uno, con una idea intuitiva de cómo funciona cada uno:

- **Regresión Lineal**: el modelo más simple de todos — busca la "mejor línea recta" (o, con más de una variable de entrada, el mejor plano) que pase lo más cerca posible de todos los puntos de entrenamiento. Ejemplo: predecir el precio de una casa a partir de sus metros cuadrados — a más metros cuadrados, más precio, y la Regresión Lineal encuentra la relación numérica exacta ("cada metro cuadrado extra suma, en promedio, tantos dólares"). Es un modelo de **regresión** (predice un número), muy fácil de interpretar, pero limitado cuando la relación entre las variables no es una línea recta.
- **Árbol de Decisión**: funciona como un juego de "20 preguntas" — va haciendo preguntas de sí/no sobre los datos ("¿el ingreso es mayor a $50.000?", "¿tiene más de 30 años?"), y según las respuestas va bajando por ramas del árbol hasta llegar a una predicción final en una "hoja". Se puede usar tanto para clasificación ("¿el cliente paga el préstamo o no?") como para regresión ("¿cuánto va a gastar este cliente?"). Su gran ventaja es que es muy fácil de visualizar y explicar — literalmente se puede dibujar el árbol de preguntas y mostrárselo a alguien sin conocimientos técnicos.
- **Random Forest**: en vez de confiar en un único Árbol de Decisión (que puede memorizar demasiado los datos de entrenamiento y funcionar mal con datos nuevos), Random Forest entrena **muchos** árboles distintos — cada uno viendo una porción distinta, al azar, de los datos y de las variables — y después promedia (en regresión) o vota por mayoría (en clasificación) las predicciones de todos ellos. La idea es la misma que "preguntarle a un grupo de expertos en vez de a uno solo": el resultado grupal suele ser más confiable que el de un único árbol, porque los errores individuales de cada árbol tienden a cancelarse entre sí.
- **Regresión Logística**: a pesar del nombre (que confunde a todo el mundo la primera vez), **no es un modelo de regresión sino de clasificación**. Se usa para predecir la probabilidad de que algo pertenezca a una categoría — por ejemplo, la probabilidad de que un tumor sea maligno, entre 0% y 100% — y después esa probabilidad se convierte en una predicción final ("maligno" si la probabilidad supera 50%, por ejemplo). El nombre viene de que matemáticamente usa una función llamada "logística" para convertir un cálculo interno en un número entre 0 y 1.
- **KNN (K-Nearest Neighbors, "K vecinos más cercanos")**: la idea más intuitiva de las cinco — para predecir la categoría (o el valor) de un caso nuevo, mira cuáles son los `K` casos **ya conocidos** más parecidos a él (los "vecinos más cercanos", midiendo distancia entre sus variables), y les copia la respuesta mayoritaria. Ejemplo: para adivinar si a alguien le va a gustar una película, KNN mira a los `K` usuarios con gustos más parecidos a los suyos, y se fija qué opinaron ellos de esa película. No necesita "entrenarse" en el sentido tradicional — simplemente guarda todos los datos y compara en el momento de predecir.

### Buenas prácticas: evitar el Data Leakage

Uno de los errores más peligrosos (porque no siempre se nota) en Machine Learning es el ***Data Leakage*** ("fuga de datos"): que información del conjunto de **test** (los datos que se supone el modelo nunca vio, usados solo para evaluar qué tan bien predice) se "filtre" de alguna forma hacia el proceso de entrenamiento. Cuando eso pasa, el modelo parece funcionar excelente durante la evaluación, pero en la vida real (con datos genuinamente nuevos) rinde mucho peor — porque en el fondo "hizo trampa" viendo pistas que no debería haber visto.

Un ejemplo concreto de cómo ocurre sin querer: si se calcula el promedio y el desvío estándar de una columna usando **todo** el dataset (entrenamiento + test juntos) para escalar los datos, y **después** se separa en train/test, el modelo ya "vio" información estadística de los datos de test (su promedio, su dispersión) antes de ser evaluado con ellos. Es una fuga sutil, fácil de cometer sin darse cuenta, y por eso la Clase 08 insistió en dos herramientas concretas para evitarla:

- **`StandardScaler`**: la herramienta que reescala las variables numéricas para que todas queden en una escala comparable (en general, restando el promedio y dividiendo por el desvío estándar, de forma que la variable termine con promedio 0 y desvío 1). Es necesario porque muchos modelos (KNN es el caso más claro, ya que mide distancias) se ven distorsionados si una variable está en una escala mucho más grande que otra — por ejemplo, "ingresos" en miles de dólares vs. "edad" en años: sin escalar, la variable "ingresos" dominaría por completo cualquier cálculo de distancia o similitud, aunque "edad" fuera igual de importante para el problema.
- **`Pipeline`**: en vez de escalar los datos como un paso "suelto" antes de entrenar, un `Pipeline` de scikit-learn encadena todos los pasos (escalado, y después el modelo) en un único objeto. La ventaja concreta: cuando se usa `Pipeline` correctamente (ajustando el escalador **solo** con los datos de entrenamiento, nunca con los de test), es mucho más difícil cometer el error de fuga de datos por accidente — el `Pipeline` fuerza a que cada paso se aplique en el orden correcto, sin mezclar información de test dentro del entrenamiento.

**`train_test_split` con `stratify`**: antes de entrenar cualquier modelo, se separa el dataset en dos partes — una porción (típicamente 70-80%) para **entrenar** el modelo, y el resto para **evaluarlo** con datos que no vio durante el entrenamiento (simulando qué tan bien funcionaría con casos reales nuevos). El parámetro `stratify` es importante cuando se trabaja con clasificación y las categorías están desbalanceadas — por ejemplo, si solo el 5% de los tumores del dataset son malignos, un split al azar podría (por mala suerte) dejar casi ningún caso maligno en el conjunto de test, haciendo que la evaluación no sea representativa. `stratify=y` le asegura al split que mantenga la misma proporción de cada categoría (5% maligno / 95% benigno) tanto en entrenamiento como en test.

### Validación: por qué un solo split no alcanza

Confiar en un único `train_test_split` tiene un problema: el resultado de la evaluación depende, en parte, de **qué** casos cayeron por azar en el conjunto de test — con otro split distinto (otros casos al azar), la métrica final podría salir un poco distinta, mejor o peor, sin que el modelo en sí haya cambiado. Para tener una medida más confiable y menos dependiente de la suerte del split, se usa la **validación cruzada** (*cross-validation*):

- **`StratifiedKFold`**: en vez de partir el dataset en un solo par entrenamiento/test, lo divide en `K` partes iguales (folds) — por ejemplo, 5 partes. El proceso entrena y evalúa el modelo `K` veces distintas: en cada vuelta, usa una parte distinta como test y las `K-1` restantes como entrenamiento. Al final, se tienen `K` mediciones de la métrica elegida, no una sola. El "Stratified" (estratificado) es lo mismo que el `stratify` de antes, pero aplicado a cada una de las `K` partes: garantiza que cada fold mantenga la misma proporción de categorías que el dataset completo.
- **`cross_val_score`**: la función de scikit-learn que automatiza todo ese proceso — entrena y evalúa el modelo las `K` veces, y devuelve las `K` métricas resultantes, listas para promediar. En vez de reportar un único número ("el modelo tuvo 85% de Accuracy"), la buena práctica es reportar el promedio **y** la dispersión de esas `K` mediciones ("85% ± 3%") — un desvío chico entre folds indica que el modelo es estable y confiable; un desvío grande es una señal de alerta de que el resultado depende mucho de qué datos le tocaron, y que probablemente no generalice bien a casos nuevos.

### Lo que cambia hoy

El aprendizaje no supervisado parte de datos **sin `y`** — sin una respuesta correcta conocida de antemano. El objetivo deja de ser predecir y pasa a ser **descubrir estructura**: qué observaciones se parecen entre sí (clustering), cómo simplificar muchas variables en pocas sin perder lo esencial (reducción de dimensionalidad), o qué patrones se repiten con frecuencia (reglas de asociación) — los tres frentes que recorre el resto de esta clase.

---

## Módulo 1 — ¿Qué es el Aprendizaje No Supervisado?

### Apertura del módulo *(Filmina 02)*

Esta filmina es la divisoria que abre el Módulo 1 — el título en pantalla ("¿Qué es el Aprendizaje No Supervisado? — Definición, tipos de problemas, ejemplos industriales y flujo típico de trabajo") funciona como el "índice hablado" de los próximos 15-20 minutos de clase. Antes de avanzar a la Filmina 03, es el momento de instalar oralmente, en una frase cada una y sin apoyarte todavía en la próxima diapositiva, las dos definiciones generales que van a servir de ancla durante el resto de la clase:

- **Aprendizaje supervisado** (lo que se cerró en la Clase 08): a partir de datos históricos donde **cada** ejemplo trae una respuesta ya conocida (una etiqueta), el modelo aprende una función que relaciona las variables de entrada con esa respuesta, con el objetivo de predecir la respuesta de casos nuevos donde todavía no se conoce. Es aprender "con la solución del libro al lado".
- **Aprendizaje no supervisado** (lo que arranca ahora): a partir de datos donde **ningún** ejemplo trae una respuesta conocida, el modelo busca regularidades, agrupamientos o estructuras internas — no para predecir un valor puntual, sino para describir cómo están organizados los datos por sí mismos. Es aprender "sin la solución del libro", encontrando el patrón a fuerza de mirar los datos.

Una forma de presentar el contraste en clase, con un ejemplo cotidiano: un supervisado es como aprender a distinguir perros de gatos porque alguien te mostró miles de fotos ya etiquetadas "perro"/"gato"; un no supervisado es como que te den una pila de miles de fotos de animales sin ningún cartel, y tengas que agruparlas vos mismo por similitud, sin que nadie te haya dicho de antemano cuántos grupos hay ni cómo se llaman. El resultado del segundo ejercicio puede coincidir con "perros" y "gatos" — pero el algoritmo llegó ahí solo por semejanza visual, no porque alguien le haya enseñado esas categorías.

Conviene remarcar en voz alta, antes de pasar a la Filmina 03, los cuatro bloques que anuncia esta diapositiva y que se van a recorrer en orden: (1) una definición formal de qué es el aprendizaje no supervisado, (2) los tres tipos de problemas que lo componen, (3) ejemplos concretos de la industria, y (4) el flujo de trabajo típico que se va a repetir, con variaciones, en cada módulo siguiente de la clase.

### Definición y diferencias con el aprendizaje supervisado *(Filmina 03)*

El aprendizaje no supervisado es un conjunto de técnicas de Machine Learning que buscan identificar estructuras, patrones o relaciones en datos que **no cuentan con etiquetas o respuestas conocidas**. A diferencia del aprendizaje supervisado (Módulo 0), donde el modelo aprende a partir de ejemplos con etiquetas, acá el objetivo es descubrir información oculta sin guía explícita.

**Para desarrollar antes de mostrar la tabla comparativa**: en la Clase 08 el flujo siempre fue el mismo — separar `X` (variables) de `y` (la respuesta a predecir), entrenar un modelo que aprenda esa relación, y medir qué tan bien predice sobre datos nuevos. Ese flujo depende por completo de que `y` exista y esté bien etiquetada — conseguir ese etiquetado en la vida real casi siempre implica un costo (alguien tuvo que revisar cada transacción y marcarla "fraude"/"no fraude", cada imagen y marcarla "tumor"/"no tumor"). El aprendizaje no supervisado nace, en parte, como respuesta a ese costo: la enorme mayoría de los datos que genera cualquier empresa **no tienen etiqueta**, y etiquetarlos a mano no siempre es viable en tiempo o presupuesto. Estas técnicas permiten extraer valor de esos datos "tal como vienen", sin la etapa previa de etiquetado.

Otra forma de plantear la diferencia, útil para la clase: en el aprendizaje supervisado el científico de datos sabe de antemano **qué pregunta** está respondiendo el modelo ("¿es spam?", "¿cuánto va a costar?"). En el no supervisado, muchas veces ni siquiera se sabe con precisión qué se va a encontrar — el algoritmo puede revelar una segmentación de clientes que nadie había considerado, o una relación entre productos que el equipo de marketing no había notado. Por eso al aprendizaje no supervisado también se lo asocia con el **análisis exploratorio**: se usa tanto para resolver un problema puntual como para "conocer" un dataset nuevo antes de decidir qué hacer con él.

| Característica | Aprendizaje Supervisado | Aprendizaje No Supervisado |
|---|---|---|
| Datos de entrada | Con etiquetas o respuestas | Sin etiquetas |
| Objetivo | Predecir o clasificar | Encontrar patrones o estructuras |
| Ejemplos de problemas | Clasificación, regresión | Clustering, reducción de dimensionalidad, reglas de asociación |

**Un matiz que vale la pena mencionar en clase** (aunque se profundiza en cursos más avanzados): la frontera entre ambos mundos no siempre es absoluta. Existen enfoques intermedios — el aprendizaje **semi-supervisado** (una pequeña porción de datos etiquetados, mucha data sin etiquetar) y el aprendizaje **autosupervisado** (el propio dataset genera sus etiquetas, por ejemplo tapando parte de una imagen y pidiéndole al modelo que la reconstruya). No forman parte del temario de hoy, pero saber que existen ayuda a entender que "supervisado vs. no supervisado" es más un espectro que una dicotomía cerrada.

### Tres grandes tipos de problemas *(Filmina 04)*

1. **Clustering (agrupamiento)**: agrupa datos similares en clusters. Ejemplo: segmentar clientes según comportamiento de compra.
2. **Reducción de dimensionalidad**: simplifica datos complejos con muchas variables a representaciones más manejables. Ejemplo: usar PCA para visualizar datos en 2D o 3D.
3. **Reglas de asociación**: encuentra relaciones frecuentes entre variables. Ejemplo: identificar productos que se compran juntos en retail.

Estas tres categorías se exploran en detalle en los Módulos 2 a 5 de esta clase, cada una con sus algoritmos y métricas propias.

**Material para desarrollar cada punto en clase, antes de pasar a la Filmina 05:**

- **Clustering** responde a la pregunta *"¿quién se parece a quién?"*. No hay un número de grupos predefinido de antemano (salvo que el algoritmo lo pida como parámetro, como K-Means) — el propio proceso de agrupar es el resultado que se busca. Vale la pena anticipar acá que en esta clase se van a ver **tres** algoritmos distintos de clustering (K-Means, Jerárquico, DBSCAN), y que ninguno es "el mejor" en términos absolutos: cada uno asume cosas distintas sobre la forma de los grupos, y esa es la razón por la que hace falta conocer más de uno.
- **Reducción de dimensionalidad** responde a *"¿puedo decir lo mismo con menos variables?"*. Es fácil de subestimar si nunca se trabajó con un dataset de verdad ancho — pero es común encontrar tablas con cientos de columnas (encuestas, datos genómicos, sensores IoT), donde ni siquiera es posible graficar todas las relaciones a la vez. PCA, que se ve en el Módulo 5, es la técnica de referencia acá — pero el concepto general ("comprimir información sin perder lo esencial") es más amplio que un solo algoritmo.
- **Reglas de asociación** responde a *"¿qué suele pasar junto con qué?"*. A diferencia de las otras dos, no trabaja con "puntos en un espacio" sino con **transacciones** (listas de ítems que ocurrieron juntos) — es la técnica más ligada al mundo del retail y el e-commerce de las tres, y la única que no requiere que los datos sean numéricos para funcionar.

Un ejercicio útil para la clase: para cada uno de los tres tipos, pedirle al grupo un ejemplo propio (no el que ya está en la filmina) de un problema de su día a día que encajaría en esa categoría — ayuda a consolidar la diferencia antes de entrar en el detalle técnico de cada algoritmo.

### Ejemplos de aplicación en la industria *(Filmina 05)*

- **Retail y E-commerce**: segmentación de clientes para campañas personalizadas, análisis de cesta de la compra con reglas de asociación.
- **Tecnología y Big Data**: detección de anomalías en redes, agrupamiento de documentos o imágenes.
- **Analítica de negocios**: reducción de variables para simplificar reportes y visualizaciones.

Estos ejemplos muestran cómo el aprendizaje no supervisado ayuda a extraer valor de datos sin necesidad de etiquetas previas, facilitando la toma de decisiones basada en patrones reales — en retail, por ejemplo, saber cómo se agrupan los clientes o qué productos se compran juntos puede mejorar significativamente las estrategias de marketing y ventas.

**Para ampliar cada rubro con más detalle antes de la filmina:**

- **Retail y E-commerce**: además de la segmentación y la cesta de compra, el no supervisado se usa para detectar **fraude de devoluciones** (agrupando patrones de compra-devolución atípicos) y para el **diseño de layout de tiendas físicas** — qué productos ubicar cerca de cuáles, a partir de qué se compra junto en la práctica, no de la intuición del gerente.
- **Tecnología y Big Data**: en ciberseguridad, la detección de anomalías en redes es en esencia un problema de clustering "al revés" — en vez de buscar el grupo al que pertenece un punto, se busca a los puntos que **no** encajan bien en ningún grupo (muy cerca del concepto de "ruido" que va a aparecer con DBSCAN en el Módulo 4). En NLP, agrupar documentos por similitud de contenido es la base de los sistemas de recomendación de artículos o noticias.
- **Analítica de negocios**: cuando un dashboard tiene 40 métricas y nadie sabe cuáles mirar primero, reducir dimensionalidad ayuda a identificar qué puñado de "meta-indicadores" resume la mayor parte de la variabilidad del negocio — un uso de PCA orientado a la comunicación con gerencia, no solo al preprocesamiento técnico.

Un cuarto sector que vale la pena mencionar aunque no esté explícito en la filmina: **salud**, donde el clustering se usa para descubrir subtipos de una enfermedad (pacientes que responden de forma distinta a un mismo tratamiento) sin que existiera antes una clasificación clínica formal para esos subgrupos.

### Flujo típico de trabajo *(Filmina 06)*

1. **Recolección y preparación de datos**: limpieza, selección y escalado de variables.
2. **Selección del método adecuado**: según el problema y el tipo de datos.
3. **Aplicación del algoritmo**: ejecución y ajuste de parámetros.
4. **Evaluación y validación**: métricas específicas para medir la calidad de agrupamientos o representaciones.
5. **Interpretación y uso de resultados**: integración en procesos de negocio o análisis posteriores.

Este flujo es la base para las prácticas y análisis de toda la clase — cambia el algoritmo módulo a módulo, no la lógica del proceso.

**Desarrollo de cada paso, para presentar antes de la filmina:**

1. **Recolección y preparación**: acá es donde más se apoya esta clase en la Clase 03/04 (Pandas) — sin datos limpios y bien tipados, ningún algoritmo de esta clase da resultados confiables. El **escalado** merece mención aparte: casi todas las técnicas de hoy (K-Means, Jerárquico, DBSCAN, PCA) miden distancias o varianza, y una variable en una escala mucho mayor que las demás (ingresos en miles vs. edad en años) puede dominar el resultado por completo si no se estandariza antes.
2. **Selección del método**: no existe "el" algoritmo de aprendizaje no supervisado — la elección depende de si se conoce de antemano cuántos grupos se esperan, si los datos tienen ruido, si las relaciones son lineales o no. Este paso es, en buena medida, el contenido de los Módulos 3, 4 y 5 de hoy.
3. **Aplicación del algoritmo**: a diferencia del supervisado, acá casi siempre hay al menos un **hiperparámetro crítico** que hay que decidir antes de correr el modelo (el `k` de K-Means, el `eps` de DBSCAN, el número de componentes de PCA) — y a diferencia también del supervisado, muchas veces no hay una única respuesta "correcta" para ese parámetro.
4. **Evaluación y validación**: sin `y`, no se puede usar Accuracy ni R². Por eso el Módulo 3 introduce el **coeficiente silhouette** y el **método del codo** — las métricas propias de este mundo, que evalúan qué tan bien separados y compactos quedaron los grupos, en vez de comparar contra una respuesta conocida.
5. **Interpretación y uso**: el paso que más distingue a esta rama del Machine Learning. Un modelo supervisado "sabe" si acertó (comparando contra `y`); un modelo no supervisado nunca sabe si el agrupamiento que encontró "tiene sentido" para el negocio — esa interpretación siempre requiere a una persona que conozca el dominio, mirando los grupos resultantes y poniéndoles nombre y sentido.

---

## Módulo 2 — Reglas de Asociación

**Contexto**: ¿alguna vez te preguntaste cómo las tiendas en línea saben qué productos recomendarte juntos? Las reglas de asociación son la técnica detrás de eso — descubrir patrones frecuentes en grandes conjuntos de transacciones.

### Apertura del módulo *(Filmina 07)*

Esta filmina divisoria anuncia el segundo bloque temático de la clase, con el subtítulo "Apriori, FP-Growth y las métricas support, confidence y lift". A diferencia del Módulo 1 (que fue conceptual, sin algoritmos concretos), acá arranca el primero de los cinco algoritmos específicos que se recorren hoy.

**Contexto para presentar antes de entrar al contenido**: las reglas de asociación son, históricamente, una de las aplicaciones de Machine Learning más ligadas al negocio de retail — nacieron en los años 90 a partir del llamado *"market basket analysis"* (análisis de la canasta de mercado), motivado por una pregunta muy concreta de las cadenas de supermercados: "si un cliente ya puso tal producto en el carrito, ¿qué otro producto tiene sentido sugerirle?". El caso más citado (aunque parcialmente mítico y discutido en su veracidad exacta) es el de "pañales y cerveza": un análisis de canasta habría encontrado que los viernes por la tarde, los clientes que compraban pañales también compraban cerveza con mayor frecuencia de la esperada — la hipótesis de negocio fue que padres jóvenes, encargados de comprar pañales para el fin de semana, aprovechaban la salida para comprarse también una cerveza. Se use o no ese ejemplo puntual, ilustra bien la idea central del módulo: encontrar relaciones **que nadie pidió explícitamente buscar**, pero que aparecen solas al mirar el volumen de transacciones.

Antes de pasar a la Filmina 08, conviene aclarar que este módulo trabaja con un tipo de dato distinto al resto de la clase: no son "puntos en un espacio" con coordenadas numéricas (como sí lo van a ser en K-Means, Jerárquico, DBSCAN o PCA), sino **listas de ítems por transacción** — un formato de datos categórico y desordenado, más parecido a una lista de compras que a una tabla de números.

### Apriori y FP-Growth *(Filmina 08)*

- **Apriori**: método clásico para encontrar conjuntos frecuentes de ítems. Genera candidatos de conjuntos y evalúa su frecuencia, descartando los que no cumplen un umbral mínimo (*support*). Intuitivo y fácil de implementar, pero computacionalmente costoso en bases grandes por la generación masiva de candidatos.
- **FP-Growth** (*Frequent Pattern Growth*): más eficiente, evita generar candidatos explícitos. Construye una estructura llamada **árbol FP** que compacta la información de las transacciones y extrae patrones frecuentes directamente. Más rápido y escalable que Apriori, aunque su implementación es más compleja.

**Para profundizar antes de mostrar la filmina:**

El nombre "Apriori" viene de un principio muy intuitivo, conocido como la **propiedad Apriori** o **propiedad de monotonía**: *si un conjunto de ítems es frecuente, entonces todos sus subconjuntos también son frecuentes*. Dicho al revés (que es como realmente se usa): *si un conjunto pequeño de ítems ya es poco frecuente, cualquier conjunto más grande que lo contenga también lo va a ser* — no hace falta ni probarlo. Esa propiedad es lo que le permite al algoritmo "podar" candidatos sin evaluarlos todos: arranca calculando el *support* de ítems individuales, descarta los infrecuentes, y solo combina los que sobrevivieron para formar pares; de los pares que sobreviven arma tríos, y así sucesivamente. Sin esta poda, el número de combinaciones posibles de ítems crece exponencialmente con el tamaño del catálogo, y se vuelve intratable incluso para un supermercado mediano (unos pocos miles de productos ya generan millones de combinaciones posibles).

FP-Growth ataca el mismo problema desde otro ángulo: en vez de generar y descartar candidatos (el paso más costoso de Apriori), comprime **todas** las transacciones en una única estructura de árbol (el árbol FP), donde los caminos compartidos entre transacciones parecidas se superponen. Una vez construido el árbol, extraer los conjuntos frecuentes es un recorrido sobre esa estructura, sin volver a generar combinaciones desde cero. Es el motivo por el que FP-Growth es el algoritmo preferido en la industria cuando el catálogo de productos es grande (miles o decenas de miles de ítems) — Apriori sigue siendo el más usado en contextos educativos y en catálogos chicos, precisamente porque su lógica es mucho más fácil de explicar y depurar paso a paso.

### Métricas clave: support, confidence y lift *(Filmina 09)*

| Métrica | Definición | Interpretación |
|---|---|---|
| **Support** | `P(A ∩ B)` — proporción de transacciones que contienen A **y** B | Indica la frecuencia con la que ocurre la regla en el conjunto de datos |
| **Confidence** | `support(A∩B) / support(A)` — probabilidad de que B ocurra dado que ocurrió A | Indica la fuerza de la regla, condicionada a A |
| **Lift** | `confidence(A→B) / support(B)` | Valores > 1 sugieren una relación positiva real entre A y B, no azar |

**Nota clave**: una regla con alto *support* y *confidence* es frecuente y confiable, pero el *lift* es el que dice si la asociación es significativa o simplemente casual. Una regla con alto *support* pero bajo *lift* puede no ser interesante, porque la asociación podría ser casual.

**Para desarrollar cada métrica en detalle, antes del ejemplo de código:**

- **Support** responde "¿qué tan común es esta combinación en general?". Es la métrica más básica de las tres, y también la que se usa como filtro inicial: antes de calcular *confidence* o *lift* de nada, Apriori descarta directamente los conjuntos con *support* por debajo de un umbral mínimo (`min_support`), porque una regla que ocurre en el 0,001% de las transacciones rara vez es útil para una decisión de negocio, sea cual sea su fuerza de asociación.
- **Confidence** responde "dado que ya pasó A, ¿qué tan seguido pasa B también?". Es una probabilidad condicional — matemáticamente idéntica a `P(B|A)` en estadística — y por eso **no es simétrica**: `confidence(pan → manteca)` y `confidence(manteca → pan)` casi nunca dan el mismo número, porque dependen de qué tan frecuente es cada ítem por separado. Es un error común de quien recién empieza con reglas de asociación asumir que la flecha "no importa" — sí importa, y mucho.
- **Lift** responde la pregunta más sutil de las tres: "¿A y B aparecen juntos más de lo que aparecerían si fueran totalmente independientes entre sí?". Un lift de exactamente 1 significa que no hay ninguna relación — A y B ocurrirían juntos esa misma cantidad de veces aunque no tuvieran nada que ver el uno con el otro, solo por pura probabilidad de que ambos son frecuentes por separado. Por eso el lift es la métrica que de verdad filtra el ruido estadístico: *support* y *confidence* pueden estar altos simplemente porque uno de los dos ítems es muy popular (como se ve en el ejemplo de código de abajo, con `leche`), y solo el lift lo deja en evidencia.

🎯 **Ejemplo**: calcular las tres métricas a mano, sobre una canasta de compras chica — sin librerías especializadas, para ver exactamente qué hay detrás de cada fórmula.

```python
# 10 transacciones de ejemplo (cada lista es la compra de un cliente)
transacciones = [
    ["pan", "leche", "manteca"],
    ["pan", "leche"],
    ["leche", "huevos"],
    ["pan", "manteca", "cafe"],
    ["pan", "leche", "manteca", "huevos"],
    ["leche", "cafe"],
    ["pan", "leche", "manteca"],
    ["pan", "cafe"],
    ["leche", "huevos", "cafe"],
    ["pan", "leche", "huevos"],
]
n = len(transacciones)

def support(itemset):
    itemset = set(itemset)
    return sum(1 for t in transacciones if itemset.issubset(t)) / n

def regla(a, b):
    sup_a, sup_b = support([a]), support([b])
    sup_ab = support([a, b])
    confidence = sup_ab / sup_a
    lift = confidence / sup_b
    print(f"{a} -> {b}: support={sup_ab:.2f}, confidence={confidence:.2f}, lift={lift:.2f}")

regla("pan", "manteca")
regla("pan", "leche")
```

**Línea por línea:**
- `support(itemset)` → `issubset(t)` chequea si **todos** los ítems del conjunto están en la transacción `t`; contar cuántas transacciones cumplen eso, dividido por el total, es exactamente la definición de *support*.
- `regla(a, b)` → aplica las tres fórmulas de la tabla de arriba en orden: primero los *supports* individuales y conjunto, después *confidence* (`sup_ab / sup_a`), después *lift* (`confidence / sup_b`).
- **Resultado real**: `pan -> manteca` da `support=0.40, confidence=0.57, lift=1.43` — lift > 1, asociación real. `pan -> leche` da `support=0.50, confidence=0.71, lift=0.89` — a pesar de tener *support* y *confidence* más altos que la regla anterior, el lift menor a 1 revela que la asociación es más débil de lo que parece: `leche` es tan frecuente por sí sola (80% de las transacciones) que aparece junto con casi cualquier cosa, sin que eso signifique una relación real con `pan`.

### ¿Cuándo usar reglas de asociación en retail? *(Filmina 10)*

- Para descubrir productos que se compran juntos y diseñar promociones cruzadas.
- Para optimizar la disposición de productos en tiendas físicas o virtuales.
- Para personalizar recomendaciones en plataformas de e-commerce.

**Importante**: estas reglas transforman datos transaccionales en insights accionables, pero expresan **co-ocurrencia, no causalidad** — un *support* y *confidence* altos no prueban que A "cause" B.

**Para cerrar el módulo con más contexto de aplicación:**

- **Promociones cruzadas**: una vez identificada una regla fuerte (alto *lift*), la decisión de negocio típica no es necesariamente "poner ambos productos en oferta juntos" — muchas veces es lo contrario: si A y B ya se compran juntos naturalmente, tiene más sentido poner en oferta solo uno de los dos (el de menor margen) para atraer al cliente, sabiendo que probablemente compre el otro a precio completo. Es un matiz que conviene discutir en clase, porque muestra que la regla de asociación es un insumo para la decisión, no la decisión en sí misma.
- **Disposición de productos**: en supermercados físicos, productos con alto *lift* a veces se colocan **lejos** uno del otro a propósito, no cerca — para que el cliente tenga que recorrer más pasillos (y estar expuesto a más productos) en el trayecto entre uno y otro. Es la misma lógica de las reglas de asociación puesta al servicio de un objetivo distinto (maximizar exposición) en vez de la comodidad de compra.
- **Recomendaciones en e-commerce**: los sistemas de "quienes compraron esto también compraron..." de sitios como Amazon o Mercado Libre son, en su forma más simple, reglas de asociación calculadas sobre millones de transacciones — aunque en producción suelen combinarse con técnicas más sofisticadas de sistemas de recomendación (filtrado colaborativo, embeddings) para mejorar la personalización.

**El punto de cierre más importante para remarcar**: co-ocurrencia no es causalidad. Que `pan` y `manteca` tengan un lift alto no prueba que comprar pan **cause** comprar manteca — podría haber una tercera variable en común (ambos se compran más los fines de semana, por ejemplo) que explique la asociación sin que exista una relación causal directa entre los dos productos. Es el mismo principio de "correlación no implica causalidad" que aparece en estadística general, aplicado al mundo de las transacciones.

---

## Módulo 3 — K-Means y la Elección de k

**Contexto**: ¿cómo agrupar datos sin etiquetas? K-Means es el algoritmo más usado de clustering — divide un conjunto de datos en grupos naturales basándose en similitud.

### Apertura del módulo *(Filmina 11)*

La divisoria de este módulo trae el subtítulo "El algoritmo de clustering más usado, y cómo elegir bien su parámetro clave" — y es, en términos de duración, el módulo más largo de la clase (7 filminas), lo cual tiene sentido: K-Means es probablemente el algoritmo de aprendizaje no supervisado más usado en la industria, por su simplicidad conceptual y su bajo costo computacional.

**Para presentar antes del contenido técnico**: conviene retomar acá, en voz alta, la definición general de clustering del Módulo 1 ("agrupar datos similares en clusters") y anticipar que K-Means la resuelve con una idea muy visual: imaginar que cada cluster tiene un "centro de gravedad" (el centroide), y que cada punto del dataset "cae" naturalmente hacia el centro más cercano. Es una buena metáfora para instalar antes de entrar en el detalle algorítmico de la Filmina 12, porque todo el resto del módulo (los 4 pasos, los problemas de convergencia, la elección de k) gira alrededor de esa única idea: minimizar qué tan lejos está, en promedio, cada punto de su centro asignado.

### Qué es y cómo funciona *(Filmina 12)*

K-Means es un **algoritmo de partición**: divide un conjunto de datos en `k` grupos (clusters) según la similitud de sus características. El objetivo es minimizar la suma de las distancias entre cada punto y el **centroide** (promedio) de su cluster asignado. Se apoya en las métricas de distancia (Euclidiana, Manhattan, Coseno) que ya se usaron en clases anteriores para definir "similitud".

**Para ampliar antes de mostrar la filmina**: el nombre completo del algoritmo, "K-Means" (K-Medias), ya describe su mecánica — la "K" es la cantidad de grupos a formar, y "Means" (medias) es literalmente cómo se calcula cada centroide: el promedio de todos los puntos que pertenecen a ese cluster en un momento dado. Formalmente, el algoritmo minimiza una función llamada **inercia** o **WCSS** (que se retoma en la Filmina 15): la suma, sobre todos los puntos, de la distancia al cuadrado entre cada punto y el centroide de su cluster. Elevar al cuadrado la distancia (en vez de usarla directa) tiene una razón matemática concreta: penaliza mucho más fuerte a los puntos lejanos que a los cercanos, lo que empuja al algoritmo a formar grupos compactos en vez de tolerar unos pocos puntos muy alejados de su centro.

Sobre las métricas de distancia: K-Means usa por defecto la distancia **Euclidiana** (la "línea recta" entre dos puntos, el teorema de Pitágoras aplicado a más de dos dimensiones) — es la que mejor encaja con la definición de centroide como promedio aritmético. Usar Manhattan (la suma de diferencias absolutas, como moverse en cuadras de una ciudad) o Coseno (el ángulo entre dos vectores, típico en texto) requeriría, estrictamente, variantes del algoritmo (K-Medoids es la alternativa más conocida cuando se necesita otra métrica de distancia).

### Los 4 pasos del algoritmo *(Filmina 13)*

1. **Inicialización**: se eligen `k` centroides iniciales — al azar o con **k-means++** para mejorar la convergencia.
2. **Asignación**: cada punto se asigna al cluster cuyo centroide esté más cerca (distancia Euclidiana, típicamente).
3. **Actualización**: se recalculan los centroides como el promedio de los puntos asignados a cada cluster.
4. **Repetición**: se repiten Asignación y Actualización hasta que las asignaciones no cambien o se alcance un número máximo de iteraciones.

**Desarrollo paso a paso, para acompañar la animación de la filmina en vivo:**

Este algoritmo también se conoce como **"Lloyd's algorithm"** en la literatura técnica, y es un buen ejemplo de un procedimiento **iterativo**: no calcula la respuesta de una vez, sino que la va refinando en rondas sucesivas, cada una un poco mejor que la anterior. Vale la pena remarcar en clase que los pasos 2 y 3 son, en esencia, un ciclo de "adivinar y corregir": el paso 2 (Asignación) responde "con los centroides que tengo ahora, ¿cuál es la mejor partición posible?"; el paso 3 (Actualización) responde "con esta partición, ¿cuáles son los mejores centroides posibles?". Cada ronda del ciclo garantiza matemáticamente que el WCSS total **nunca aumenta** — por eso el algoritmo siempre termina convergiendo (ver Filmina 14), aunque no siempre al mejor resultado posible.

Sobre la Inicialización: la opción "al azar" simplemente elige `k` puntos cualquiera del dataset como primeros centroides — es simple pero puede arrancar en una posición muy mala. **k-means++** (el default en la implementación de scikit-learn) es más inteligente: elige el primer centroide al azar, y cada centroide siguiente lo elige con una probabilidad proporcional a qué tan lejos está de los centroides ya elegidos — favoreciendo que los `k` puntos de arranque queden bien repartidos por el espacio de datos, en vez de agrupados por casualidad en una sola zona.

### Convergencia, inicialización y problemas comunes *(Filmina 14)*

- K-Means **siempre converge**, pero a un **mínimo local**, no necesariamente al óptimo global.
- La inicialización de los centroides afecta la calidad y velocidad de convergencia; **k-means++** ayuda a elegir centroides iniciales más representativos, reduciendo la probabilidad de resultados pobres.
- **Outliers**: pueden distorsionar los centroides y afectar la agrupación.
- **Formas no esféricas**: K-Means asume clusters convexos y de tamaño similar; no funciona bien con formas arbitrarias.

**Para desarrollar cada punto con más profundidad:**

- **Mínimo local vs. global**: como el resultado final depende de dónde arrancaron los centroides, correr K-Means dos veces con inicializaciones distintas puede dar dos particiones **distintas**, ambas "válidas" en el sentido de que el algoritmo convergió correctamente en las dos, pero una puede ser mejor que la otra. La solución práctica que usa scikit-learn (y que aparece en el ejemplo de código de la Filmina 16, con el parámetro `n_init=10`) es correr el algoritmo completo varias veces con distintas inicializaciones al azar, y quedarse con el resultado que dio el WCSS más bajo de todos los intentos.
- **Sensibilidad a outliers**: como el centroide es un **promedio**, un solo punto muy alejado del resto puede "arrastrar" el centroide entero hacia él, distorsionando la posición de todo el cluster — el mismo fenómeno por el que la media aritmética es sensible a valores extremos (visto en clases anteriores de estadística descriptiva). Es una de las razones por las que suele convenir revisar y tratar outliers **antes** de correr K-Means, no después.
- **Formas no esféricas**: como K-Means asigna cada punto según distancia al centroide más cercano, la "frontera" natural entre dos clusters siempre termina siendo una línea recta (o un plano, en más dimensiones) — geométricamente, solo puede separar bien grupos que tengan forma redondeada y tamaño parecido. Con clusters alargados, en forma de luna, o de tamaños muy distintos entre sí, K-Means directamente separa mal — y ese es exactamente el problema que resuelve DBSCAN, que se ve en el Módulo 4.

### Elegir k: método del codo (Elbow Method) *(Filmina 15)*

Para cada valor de `k` se calcula el **WCSS** (*Within-Cluster Sum of Squares*): la suma de las distancias al cuadrado entre cada punto y el centroide de su cluster. Un WCSS más bajo indica clusters más compactos.

Se grafica WCSS en función de `k` — la curva baja a medida que `k` crece, porque agrupar en más clusters siempre reduce la distancia interna. El objetivo es identificar el punto donde la tasa de disminución se frena notablemente, formando un **"codo"**: a partir de ahí, agregar más clusters no mejora significativamente la calidad de la agrupación. Balancea complejidad del modelo (muchos clusters) contra calidad de la agrupación (pocos clusters, cada uno con sentido) — evitando tanto el subajuste como el sobreajuste.

**Para ampliar antes de mostrar el gráfico**: vale la pena mencionar el caso extremo para que la lógica quede clara — si `k` fuera igual a la cantidad total de puntos del dataset, cada punto sería su propio cluster, y el WCSS daría exactamente `0` (cada punto coincide con su propio centroide). Ese extremo es matemáticamente "perfecto" pero completamente inútil para el negocio: no agrupa nada. El método del codo es, en el fondo, una forma visual de encontrar el compromiso entre ese extremo inútil (`k` = cantidad de puntos, WCSS = 0) y el otro extremo igual de inútil (`k` = 1, todo en un solo grupo, WCSS máximo). Conviene aclarar también que la ubicación del "codo" no siempre es tan clara como en el ejemplo de esta clase — en datasets reales, la curva a veces baja de forma más gradual, sin un quiebre visualmente obvio, y ahí es donde el coeficiente silhouette (Filmina 16) aporta una segunda opinión más cuantitativa.

### Elegir k: coeficiente silhouette *(Filmina 16)*

Para cada punto, compara su **cohesión** (distancia promedio a los demás puntos de su propio cluster) contra su **separación** (distancia promedio al cluster más cercano al que no pertenece). El resultado es un valor entre **-1 y 1**:

- Cerca de **1**: el punto está muy bien asignado a su cluster.
- Cerca de **-1**: el punto probablemente está mal asignado, y encajaría mejor en otro cluster.

Se calcula el promedio del coeficiente para todos los puntos, para cada `k` candidato, y se elige el `k` que **maximiza** ese promedio — el que da los clusters más definidos y separados. También sirve para detectar outliers: puntos con coeficiente cercano a -1 son candidatos a estar mal asignados.

**Para desarrollar el mecanismo con más detalle antes del ejemplo de código:**

El coeficiente silhouette de un punto se calcula, formalmente, como `(b - a) / max(a, b)`, donde `a` es la distancia promedio del punto a los demás puntos de **su propio** cluster (la cohesión — cuanto más chica, mejor) y `b` es la distancia promedio a los puntos del cluster **vecino más cercano** al que no pertenece (la separación — cuanto más grande, mejor). Un valor cercano a **0** (no solo los extremos -1 y 1) también es informativo: significa que el punto está prácticamente sobre el límite entre dos clusters, ni claramente adentro de uno ni del otro — una zona ambigua que suele señalar que, en esa región del espacio, tal vez `k` no está bien elegido.

A diferencia del método del codo (que es una lectura visual, algo subjetiva, de dónde "se frena" una curva), el silhouette da un **número único y objetivo** para comparar entre valores de `k` — por eso en la práctica se suelen usar los dos métodos en conjunto: el codo da una intuición rápida, y el silhouette confirma (o contradice) esa intuición con un criterio cuantitativo. Cuando ambos coinciden en el mismo `k` (como en el ejemplo de código de abajo, donde los dos señalan `k=4`), la elección queda mucho más respaldada que si se hubiera usado un solo criterio.

🎯 **Ejemplo**: generar datos sintéticos con 4 grupos conocidos de antemano, "olvidarnos" de ese número, y recuperarlo con el método del codo y el silhouette.

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Datos sintéticos con 4 centros conocidos (en la práctica, no lo sabríamos)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.1, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# Método del codo: WCSS para k de 1 a 8
wcss = []
for k in range(1, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)   # inertia_ = WCSS de ese modelo

# Coeficiente silhouette: no se calcula para k=1 (no hay "otro cluster" con quien comparar)
mejores = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    mejores.append((k, silhouette_score(X_scaled, labels)))

mejor_k = max(mejores, key=lambda par: par[1])[0]
print(f"Mejor k según silhouette: {mejor_k}")

# Modelo final con el k elegido
kmeans_final = KMeans(n_clusters=mejor_k, n_init=10, random_state=42)
etiquetas = kmeans_final.fit_predict(X_scaled)
```

**Línea por línea:**
- `make_blobs(n_samples=300, centers=4, ...)` → genera 300 puntos repartidos en 4 grupos con forma esférica — el escenario "ideal" para K-Means.
- `km.inertia_` → atributo de scikit-learn que ya trae calculado el WCSS del modelo ajustado; no hace falta calcularlo a mano.
- `silhouette_score(X_scaled, labels)` → recibe los datos y las etiquetas de cluster que asignó el modelo, y devuelve el promedio del coeficiente silhouette de todos los puntos.
- `max(mejores, key=lambda par: par[1])` → de la lista de tuplas `(k, silhouette)`, se queda con la que tiene el silhouette más alto.
- **Resultado real**: el WCSS cae de 600 (`k=1`) a 74.6 (`k=3`) y a 20.9 (`k=4`) — ahí está el "codo", porque de `k=4` en adelante la mejora es marginal (18.7, 16.6, 14.7...). El silhouette confirma lo mismo de otra forma: da su valor más alto (0.778) exactamente en `k=4` — el mismo número de centros que usamos para generar los datos, recuperado sin haberlo usado en ningún momento del cálculo.

### Aplicación práctica y relevancia en la industria *(Filmina 17)*

- **Retail**: segmentar clientes por frecuencia de compra, monto gastado y preferencia de categorías, para diseñar campañas de marketing personalizadas.
- **Finanzas**: identificar grupos de clientes con perfiles de riesgo similares, mejorando la gestión de cartera y la detección de fraudes.
- **Imágenes**: segmentación de regiones en análisis de imágenes, y sistemas de recomendación personalizados.

La correcta elección de `k` evita tanto la **sobresegmentación** (demasiados clusters que complican la interpretación) como la **subsegmentación** (pocos clusters que ocultan diferencias importantes).

**Para cerrar el módulo con ejemplos más desarrollados:**

- **Retail**: la técnica de segmentación de clientes más citada en la industria es el análisis **RFM** (*Recency, Frequency, Monetary* — hace cuánto compró por última vez, con qué frecuencia compra, cuánto gasta en total), que se calcula con Pandas (`groupby`/`agg`, contenido de la Clase 04) y después se pasa como input a K-Means para formar los segmentos finales — un ejemplo concreto de cómo se conectan las herramientas de las últimas clases entre sí.
- **Finanzas**: además de detección de fraude, K-Means se usa en gestión de portafolios para agrupar activos financieros con comportamientos de precio similares (una forma de diversificación basada en datos, en vez de en la clasificación tradicional por sector o industria).
- **Imágenes**: un uso muy visual para mostrar en clase es la **cuantización de color** — aplicar K-Means sobre los píxeles de una imagen (donde cada píxel es un punto en un espacio de 3 dimensiones: rojo, verde, azul) reduce la imagen a solo `k` colores distintos, el color de cada centroide. Es una forma concreta y visual de "ver" cómo trabaja el algoritmo, sin necesitar imaginar puntos abstractos en un gráfico.

**Sobre la sobresegmentación y subsegmentación**: en términos de negocio, la sobresegmentación tiene un costo operativo real — si marketing tiene que diseñar 15 campañas distintas para 15 microsegmentos de clientes, el costo de gestionar esa complejidad puede superar el beneficio de la personalización. La subsegmentación, en cambio, tiene un costo de oportunidad: agrupar en pocos clusters muy amplios puede esconder un segmento pequeño pero muy rentable dentro de un grupo más grande y menos interesante. No existe una regla matemática que resuelva esta tensión — el método del codo y el silhouette dan candidatos razonables de `k`, pero la decisión final casi siempre involucra also una restricción práctica del negocio (cuántos segmentos puede gestionar realmente el equipo de marketing, por ejemplo).

---

## Módulo 4 — Clustering Jerárquico y DBSCAN

**Contexto**: dos alternativas a K-Means, para cuando no querés (o no podés) definir `k` de antemano, o cuando tus datos tienen ruido y formas irregulares.

### Apertura del módulo, después del Break *(Filmina 19)*

Esta divisoria llega justo después del corte de 10 minutos (Filmina 18) — conviene arrancar retomando brevemente dónde había quedado la clase antes del break: K-Means resuelve bien el clustering cuando los grupos son razonablemente esféricos, de tamaño parecido, y se conoce (o se puede estimar) el número `k` de antemano. Este módulo presenta **dos alternativas** que relajan, cada una a su manera, esas mismas condiciones.

**Para presentar antes de entrar al contenido**: es útil anticipar la pregunta que motiva a ambos algoritmos, aunque la resuelven de formas completamente distintas: *"¿qué hago cuando no sé cuántos grupos hay, o cuando mis grupos no tienen forma de círculo?"*. El clustering **jerárquico** responde con una idea de "no elegir un solo k, sino construir todas las particiones posibles a la vez, y decidir después". **DBSCAN** responde con una idea distinta: en vez de definir clusters por cercanía a un centro (como K-Means) o por una jerarquía de fusiones (como el jerárquico), los define por **densidad** — dónde hay muchos puntos juntos versus dónde hay pocos. Instalar esta distinción de entrada ayuda a que el resto del módulo se entienda como "dos soluciones a problemas parecidos, con lógicas de fondo distintas" y no como una lista de algoritmos sueltos.

### Clustering jerárquico: aglomerativo y divisivo *(Filmina 20)*

El clustering jerárquico construye una jerarquía de clusters, sin necesidad de definir el número de clusters de antemano:

- **Aglomerativo** (el más usado en la práctica): comienza con cada punto como un cluster individual, y fusiona iterativamente los dos clusters más parecidos, hasta que todos quedan combinados en uno solo.
- **Divisivo**: el enfoque inverso — parte de un único cluster con todos los datos, y lo va dividiendo progresivamente.

El resultado se visualiza en un **dendrograma**: un diagrama en forma de árbol donde cada hoja es un punto individual, y la altura donde dos clusters se unen indica su grado de disimilitud (cuanto más abajo se unen, más similares son). Cortar el dendrograma a distintas alturas da distintos números de clusters, sin tener que volver a correr el algoritmo.

**Para desarrollar antes del ejemplo de código:**

El enfoque **aglomerativo** ("bottom-up", de abajo hacia arriba) es, con mucha diferencia, el más usado en la práctica frente al divisivo ("top-down") — la razón es principalmente de costo computacional: en cada paso, el aglomerativo solo necesita encontrar el par de clusters más parecido entre los que ya existen y fusionarlos, mientras que el divisivo necesitaría evaluar **todas** las formas posibles de partir un cluster grande en dos, un problema combinatorio mucho más costoso. Por eso, cuando en la práctica se habla de "clustering jerárquico" sin más aclaración, casi siempre se refiere al aglomerativo.

La gran ventaja pedagógica del dendrograma es que muestra **toda la estructura de agrupamiento posible** en una sola imagen — desde `k=1` (todo en la raíz del árbol) hasta `k = cantidad de puntos` (cada hoja individual). Elegir el número de clusters se convierte, visualmente, en elegir a qué altura "cortar" el árbol con una línea horizontal: cuantos más nodos verticales cruce esa línea, más clusters resultan. Es una diferencia de fondo respecto a K-Means, donde `k` hay que decidirlo **antes** de correr el algoritmo (con el codo o el silhouette del Módulo 3) — acá se puede correr el algoritmo una sola vez y decidir `k` **después**, mirando el árbol completo.

🎯 **Ejemplo del PDF**: construir un dendrograma sobre un dataset sintético de clientes (Ingresos, Gasto Mensual, Edad), usando el método de linkage `ward` (minimiza la varianza dentro de los clusters).

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# 1) Dataset de ejemplo con variables continuas
np.random.seed(42)
data = pd.DataFrame({
    "Ingresos": np.random.randint(20000, 150000, 50),
    "Gasto_Mensual": np.random.randint(5000, 40000, 50),
    "Edad": np.random.randint(18, 70, 50),
})

# 2) Estandarización: obligatoria antes de medir distancias
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 3) Linkage con método Ward (minimiza la varianza dentro de los clusters)
Z = linkage(data_scaled, method="ward")

# 4) Dendrograma
plt.figure(figsize=(12, 7))
plt.title("Dendrograma - Clustering Jerárquico")
dendrogram(Z, leaf_rotation=90, leaf_font_size=10)
plt.show()
```

**Línea por línea:**
- `StandardScaler().fit_transform(data)` → estandariza las 3 columnas (media 0, desvío 1); imprescindible porque `Ingresos` y `Edad` tienen escalas completamente distintas, y sin escalar, `Ingresos` dominaría por completo el cálculo de distancias.
- `linkage(data_scaled, method="ward")` → calcula, paso a paso, qué par de clusters fusionar en cada nivel; el resultado `Z` es la estructura que describe todo el árbol de fusiones.
- `dendrogram(Z, ...)` → dibuja el árbol; `leaf_rotation=90` rota las etiquetas del eje X para que no se superpongan.

### Parámetro clave: el linkage *(Filmina 21)*

El método de linkage determina cómo se mide la distancia entre dos clusters para decidir si conviene fusionarlos:

| Linkage | Criterio |
|---|---|
| **Single** | Distancia mínima entre puntos de dos clusters |
| **Complete** | Distancia máxima entre puntos de dos clusters |
| **Average** | Promedio de todas las distancias entre pares de puntos |

Cada criterio afecta la forma y el tamaño de los clusters resultantes — no hay una elección "correcta" universal, depende de la estructura de los datos.

**Para profundizar cada criterio antes de mostrar la tabla:**

- **Single linkage** (también llamado "vecino más cercano"): como usa la distancia **mínima**, alcanza con que dos clusters tengan **un solo par** de puntos muy cercanos para que se fusionen — aunque el resto de los puntos de ambos clusters estén lejos entre sí. Esto le permite detectar clusters de forma alargada o irregular, pero lo vuelve propenso a un problema conocido como *"chaining"* (encadenamiento): una fila de puntos equiespaciados puede terminar uniendo dos grupos que, intuitivamente, deberían quedar separados, solo porque hay un "puente" de puntos intermedios.
- **Complete linkage** (o "vecino más lejano"): usa la distancia **máxima**, así que exige que **todos** los puntos de ambos clusters estén razonablemente cerca antes de fusionarlos — tiende a formar clusters más compactos y de tamaño parecido entre sí, el opuesto casi exacto de single linkage.
- **Average linkage**: un punto intermedio entre los dos anteriores, promediando todas las distancias par a par — suele ser una opción "segura" cuando no hay una razón clara para preferir uno de los dos extremos.

El ejemplo de código de la filmina anterior usó un cuarto criterio, **Ward**, que no aparece en esta tabla del PDF pero es el más usado en la práctica con datos numéricos: en vez de basarse directamente en distancias entre puntos, fusiona en cada paso el par de clusters que produce el **menor incremento posible en la varianza interna total** — conceptualmente, es el mismo objetivo que minimiza K-Means (WCSS), pero aplicado paso a paso dentro de la lógica jerárquica.

### DBSCAN: clustering basado en densidad *(Filminas 22–23)*

**DBSCAN** (*Density-Based Spatial Clustering of Applications with Noise*) identifica clusters como regiones **densas** separadas por regiones de baja densidad, y detecta puntos aislados como **ruido** en vez de forzarlos a pertenecer a algún cluster.

Dos parámetros clave:
- **`eps`** (épsilon): el radio máximo para considerar a dos puntos "vecinos".
- **`min_samples`**: la cantidad mínima de puntos que tiene que haber en ese radio para considerar la zona "densa".

Tres tipos de puntos:
- **Core point**: tiene al menos `min_samples` vecinos dentro de su radio `eps` — es el corazón de un cluster denso.
- **Border point**: está dentro del radio `eps` de un core point, pero no tiene suficientes vecinos propios para ser core.
- **Noise point**: no es ni core ni border — queda marcado con `label = -1`, fuera de cualquier cluster.

DBSCAN es especialmente útil para detectar clusters de **forma arbitraria** (no solo esféricos, a diferencia de K-Means) y manejar ruido explícitamente.

**Para desarrollar el mecanismo con más profundidad, antes del ejemplo de código:**

El nombre completo, *Density-Based Spatial Clustering of Applications with Noise*, ya resume la idea central: en vez de preguntarse "¿a qué centro está más cerca este punto?" (la pregunta de K-Means), DBSCAN se pregunta **"¿este punto está en una zona densamente poblada?"**. Un cluster, para DBSCAN, no es más que una región conectada de puntos densos: si el punto A es vecino denso del punto B, y B es vecino denso de C, entonces A y C terminan en el mismo cluster aunque A y C no sean vecinos directos entre sí — es un criterio de conectividad "en cadena" (transitivo), muy distinto a la idea de "cercanía a un centro único" de K-Means.

Los dos parámetros son las dos preguntas que hay que responder para definir "denso": `eps` responde *"¿qué tan cerca hay que estar para contar como vecino?"*, y `min_samples` responde *"¿cuántos vecinos hacen falta para considerar la zona densa?"*. Ajustar estos dos números cambia radicalmente el resultado: un `eps` muy chico deja casi todo como ruido (porque casi nada tiene suficientes vecinos tan cerca); un `eps` muy grande termina fusionando clusters que deberían quedar separados (porque "casi todo" pasa a ser vecino de "casi todo"). Por eso la Filmina 22 trae la técnica del **k-distance plot** (que se ve en el ejemplo de código): una forma sistemática de estimar un buen valor de `eps` a partir de los propios datos, en vez de adivinarlo a prueba y error.

Sobre los tres tipos de punto: la distinción entre **core** y **border** es sutil pero importante — un border point sí forma parte de un cluster (queda "adentro" de la región densa por estar cerca de un core point), pero no tiene la densidad suficiente **por sí mismo** como para ser considerado el corazón de esa densidad. Es la diferencia entre "vivir en un barrio poblado" (border) y "ser, vos mismo, uno de los puntos que hace que el barrio esté poblado" (core). Solo el **noise point** queda completamente afuera de cualquier cluster — y a diferencia de K-Means, donde **todo** punto es forzado a pertenecer a algún cluster (incluso un outlier extremo), en DBSCAN el ruido es un resultado legítimo y esperado, no un error.

🎯 **Ejemplo del PDF**: generar un dataset sintético con formas no convexas ("lunas"), blobs densos y ruido disperso, usar un **k-distance plot** para estimar `eps`, y correr DBSCAN.

```python
import numpy as np
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

# Dataset de ejemplo: "lunas" (no convexas) + blobs (densos) + ruido disperso
np.random.seed(42)
X1, _ = make_moons(n_samples=300, noise=0.08)
X2, _ = make_blobs(n_samples=150, centers=[(3, 3), (6, -1)], cluster_std=[0.3, 0.6])
ruido = np.random.uniform(low=-3, high=8, size=(60, 2))
X = np.vstack([X1, X2, ruido])
X_scaled = StandardScaler().fit_transform(X)

# k-distance plot: la distancia al k-ésimo vecino de cada punto, ordenada
# El "codo" de esta curva es un buen candidato para eps
k = 4  # suele usarse min_samples o min_samples - 1
nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
distancias, _ = nbrs.kneighbors(X_scaled)
k_distancias = np.sort(distancias[:, -1])

# DBSCAN con eps estimado a partir del codo del gráfico anterior
db = DBSCAN(eps=0.20, min_samples=4)
labels = db.fit_predict(X_scaled)   # label == -1 -> ruido

n_clusters = len(set(labels) - {-1})
n_ruido = list(labels).count(-1)
print(f"Clusters encontrados: {n_clusters} | Puntos de ruido: {n_ruido}")
```

**Línea por línea:**
- `make_moons(...)` y `make_blobs(...)` → generan datos sintéticos con dos formas bien distintas: semicírculos entrelazados (no convexos, el punto débil de K-Means) y grupos densos y compactos.
- `NearestNeighbors(n_neighbors=k).fit(...).kneighbors(...)` → para cada punto, calcula la distancia a sus `k` vecinos más cercanos; nos quedamos con la distancia al último (`[:, -1]`).
- `np.sort(distancias[:, -1])` → ordena esas distancias de menor a mayor; graficada, esta curva muestra un "codo" que es el valor recomendado para `eps`.
- `DBSCAN(eps=0.20, min_samples=4).fit_predict(X_scaled)` → corre el algoritmo; devuelve un array de etiquetas, una por punto, donde `-1` es ruido.
- Corriendo este ejemplo en la práctica: **4 clusters** detectados (las dos lunas y los dos blobs) y **60 puntos** marcados como ruido — exactamente los que se generaron como ruido disperso a propósito.

### Comparación: Jerárquico vs. Particional vs. DBSCAN *(Filmina 24)*

| Característica | Jerárquico | Particional (K-Means) | DBSCAN |
|---|---|---|---|
| **Forma de clusters** | Jerarquía flexible | Convexa, esférica | Arbitraria |
| **Número de clusters** | No requiere definirlo | Requiere definir `k` | No requiere definirlo |
| **Manejo de ruido** | No explícito | No explícito | Sí, lo detecta |
| **Parámetros clave** | Linkage | Número de clusters (`k`) | `eps`, `min_samples` |

La elección del método depende del tipo de datos, la forma esperada de los clusters y la presencia de ruido — el clustering jerárquico es útil para **explorar** la estructura antes de decidir un número de clusters; DBSCAN es ideal para detectar grupos irregulares y manejar ruido (por ejemplo, zonas de alta concentración de clientes en un análisis geoespacial).

**Guía práctica para cerrar el módulo, útil como resumen para dictar de memoria:**

- Si el dataset es **grande** (cientos de miles de puntos o más): K-Means, por lejos el más rápido de los tres — el clustering jerárquico tiene un costo computacional que crece muy rápido con la cantidad de puntos (calcular y actualizar distancias entre todos los pares), lo que lo vuelve poco práctico a esa escala.
- Si **no se sabe cuántos grupos hay** y se quiere explorar visualmente antes de decidir: clustering jerárquico, por el dendrograma.
- Si los datos tienen **ruido real** (sensores con lecturas erróneas, usuarios anómalos, fraude) que no debería forzarse a ningún cluster: DBSCAN, el único de los tres pensado explícitamente para separar señal de ruido.
- Si los clusters esperados tienen **formas irregulares** (no convexas, tamaños muy distintos): DBSCAN; K-Means y, en menor medida, el jerárquico con linkage `complete`/`ward`, tienden a fallar en ese escenario.

Un caso de uso muy citado en clase para DBSCAN es el análisis geoespacial: agrupar coordenadas GPS de usuarios o eventos para encontrar "zonas calientes" de actividad (por ejemplo, dónde se concentran los pedidos de una app de delivery en determinado horario) — un escenario donde el número de zonas no se conoce de antemano, y donde puntos aislados (un pedido en una zona rural sin actividad alrededor) deberían quedar como ruido, no forzados dentro de la zona caliente más cercana.

---

## Módulo 5 — PCA: Reducción de Dimensionalidad

**Contexto**: ¿cómo simplificar un dataset con decenas o cientos de variables sin perder lo esencial? El Análisis de Componentes Principales (PCA) es la técnica fundamental para reducir dimensionalidad, facilitando la visualización y el análisis.

### Apertura del módulo *(Filmina 25)*

Esta divisoria trae el subtítulo "Covarianza, eigenvectores/eigenvalores, varianza explicada" — y anuncia el módulo más matemático de la clase. A diferencia de los tres módulos de clustering (donde la matemática de fondo se puede dejar bastante implícita y trabajar con la intuición geométrica de "puntos que se agrupan"), PCA requiere presentar un mínimo de álgebra lineal para que las filminas siguientes tengan sentido.

**Para presentar antes del contenido técnico**: conviene arrancar retomando la Filmina 04 (Módulo 1), donde la reducción de dimensionalidad se definió como "simplificar datos complejos con muchas variables a representaciones más manejables". PCA es la técnica de referencia para resolver ese problema, y su lógica se puede resumir en una sola idea, sin fórmulas todavía: encontrar las direcciones **nuevas** (no necesariamente las variables originales) a lo largo de las cuales los datos varían más — porque ahí es donde vive la mayor parte de la información. Es una buena analogía para instalar acá: sacarle una foto a una escultura 3D desde el ángulo que muestra más detalle en una sola imagen 2D, en vez de desde un ángulo que la aplana y esconde su forma. PCA busca, matemáticamente, ese "mejor ángulo" para los datos.

### Covarianza: la relación entre variables *(Filmina 26)*

La covarianza mide cómo varían **juntas** dos variables: si ambas tienden a subir o bajar a la vez, es positiva; si una sube mientras la otra baja, es negativa.

$$Cov(X,Y) = E[(X - \mu_X)(Y - \mu_Y)]$$

En PCA, la **matriz de covarianza** resume esas relaciones entre **todas** las variables del dataset a la vez, y es la base para identificar las direcciones de mayor variabilidad.

**Para desarrollar antes de mostrar la fórmula:**

Vale la pena recordar primero qué es la **varianza** (el caso particular de covarianza de una variable consigo misma, `Cov(X,X)`) antes de saltar a la covarianza entre dos variables distintas: la varianza mide qué tan dispersos están los valores de una única variable respecto a su propio promedio. La covarianza extiende esa misma idea a un **par** de variables: en vez de preguntar "¿qué tan lejos está cada valor de X de su propio promedio?", pregunta "¿cuándo X se aleja de su promedio hacia arriba, Y también tiende a alejarse del suyo hacia arriba (covarianza positiva), hacia abajo (covarianza negativa), o no hay ningún patrón (covarianza cercana a cero)?".

La razón por la que PCA construye una **matriz** de covarianza (y no solo un número) es que un dataset real casi nunca tiene 2 variables, sino muchas — la matriz de covarianza es simplemente la tabla que junta, en una sola estructura, la covarianza de **cada par posible** de variables (más la varianza de cada una consigo misma, en la diagonal). Para 3 variables, es una matriz de 3×3; para 75 variables (como el dataset de la Clase 04), sería una matriz de 75×75. Esa matriz completa es el punto de partida matemático de todo lo que sigue en el módulo: PCA busca, dentro de esa matriz, las direcciones donde la variabilidad conjunta es máxima.

### Eigenvectores y eigenvalores: direcciones y magnitudes *(Filmina 27)*

Un **eigenvector** es un vector que, al aplicarle una transformación lineal (como la matriz de covarianza), solo cambia en magnitud, no en dirección. El factor por el que cambia esa magnitud es el **eigenvalor** correspondiente.

En PCA:
- Los **eigenvectores** de la matriz de covarianza son las **Componentes Principales** — las nuevas direcciones ortogonales sobre las que se proyectan los datos.
- Los **eigenvalores** indican la **varianza** que explica cada componente — un eigenvalor alto significa que esa dirección captura mucha variabilidad de los datos.

**Para desarrollar el concepto de eigenvector/eigenvalor con más profundidad, antes del ejemplo de código:**

Es probablemente el concepto más abstracto de toda la clase, y merece una explicación intuitiva antes de la definición formal. Pensá en la matriz de covarianza como una transformación que "estira" el espacio en distintas direcciones, más en las direcciones donde los datos varían más, menos donde varían poco. La mayoría de los vectores, al pasar por esa transformación, no solo cambian de tamaño sino también de **dirección** — apuntan "torcido" respecto a como apuntaban antes. Los eigenvectores son la excepción: son las pocas direcciones especiales que la transformación **no tuerce**, solo estira o encoge a lo largo de esa misma dirección. Por eso son las direcciones "naturales" de esa matriz — los ejes a lo largo de los cuales tiene sentido describir cómo varían los datos.

En el contexto específico de PCA, el eigenvector con el eigenvalor más alto es literalmente la dirección de **máxima varianza** posible en los datos — la "primera Componente Principal". El segundo eigenvector (siempre ortogonal, es decir perpendicular, al primero) es la dirección de máxima varianza que queda **después** de descontar la que ya explicó el primero. Y así sucesivamente: cada componente principal explica la mayor variabilidad posible que las componentes anteriores todavía no explicaron. Esa es la propiedad que hace útil a PCA para reducir dimensionalidad: las primeras componentes concentran la mayor parte de la información, así que se pueden descartar las últimas (las que explican poca varianza) sin perder demasiado.

🎯 **Ejemplo**: calcular la matriz de covarianza y sus eigenvalores "a mano" con NumPy, y confirmar que da exactamente lo mismo que el `PCA` de scikit-learn — para que quede claro que no es magia, es álgebra lineal.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Dataset sintético: x2 correlacionada con x1 a propósito, x3 independiente (ruido)
np.random.seed(42)
n = 200
x1 = np.random.normal(0, 1, n)
x2 = x1 * 0.9 + np.random.normal(0, 0.3, n)
x3 = np.random.normal(0, 1, n)
datos_escalados = StandardScaler().fit_transform(np.column_stack([x1, x2, x3]))

# Matriz de covarianza "a mano"
matriz_cov = np.cov(datos_escalados.T)
print(matriz_cov.round(2))

# Eigenvalores y eigenvectores (eigh: para matrices simétricas, como la de covarianza)
autovalores, autovectores = np.linalg.eigh(matriz_cov)
autovalores = np.sort(autovalores)[::-1]                       # orden de mayor a menor
varianza_explicada = autovalores / autovalores.sum() * 100
print(f"Varianza explicada (a mano): {varianza_explicada.round(1)}")

# Confirmación con PCA de scikit-learn
pca = PCA().fit(datos_escalados)
print(f"Varianza explicada (sklearn): {(pca.explained_variance_ratio_ * 100).round(1)}")
```

**Línea por línea:**
- `x2 = x1 * 0.9 + ruido` → construye a propósito una variable fuertemente correlacionada con `x1`, para que el ejemplo tenga una dirección de varianza claramente dominante.
- `np.cov(datos_escalados.T)` → la matriz de covarianza 3×3; en el resultado real, la covarianza entre `x1` y `x2` da `0.95` (muy alta), mientras que `x3` queda casi en `0` con las otras dos.
- `np.linalg.eigh(...)` → variante de `eig` pensada para matrices **simétricas** (la de covarianza siempre lo es); a diferencia de `eig`, devuelve los autovalores ya como números reales, sin parte imaginaria residual.
- **Resultado real**: la varianza explicada da `[66.1%, 32.1%, 1.8%]` calculada a mano, y **exactamente los mismos tres números** con `PCA()` de scikit-learn — el primer componente concentra dos tercios de toda la variabilidad, justamente porque resume la relación compartida entre `x1` y `x2`.

### Varianza explicada y selección de componentes *(Filmina 28)*

La suma de todos los eigenvalores es la varianza total de los datos. La varianza explicada por cada componente es el porcentaje que representa su eigenvalor respecto a esa suma total. Esto ayuda a decidir cuántos componentes conservar:

- Conservar los primeros componentes que expliquen un porcentaje significativo (ej. 90%) de la varianza acumulada.
- Un gráfico de codo (misma lógica que en K-Means) ayuda a ver dónde agregar más componentes deja de aportar varianza relevante.
- En algunos casos conviene priorizar **menos** componentes para simplificar el modelo, aunque se pierda algo de varianza — es una decisión de compromiso, no una regla fija.

**Para desarrollar antes de la filmina:**

Vale la pena remarcar el paralelismo explícito con el método del codo de K-Means (Módulo 3, Filmina 15): en los dos casos se grafica una curva (WCSS en un caso, varianza explicada acumulada en el otro) en función de un número entero que hay que elegir (`k` clusters, o cantidad de componentes), y en los dos casos se busca el punto donde agregar "una unidad más" deja de aportar una mejora proporcional. Es el mismo patrón de decisión — "¿cuánta complejidad adicional se justifica por la mejora que trae?" — aplicado a dos problemas distintos.

Un umbral común en la práctica profesional es conservar los componentes que expliquen el **95%** de la varianza acumulada (a veces 90%, según qué tan crítico sea no perder información) — pero ese número no es una ley matemática, es una convención razonable. Si el objetivo final es solo **visualizar** los datos, casi siempre se usan exactamente 2 o 3 componentes, sin importar qué porcentaje de varianza expliquen — porque el límite ahí no es estadístico, es que un gráfico no puede tener más de 3 ejes.

### Limitaciones de PCA *(Filmina 29)*

- **Linealidad**: PCA solo captura relaciones **lineales** entre variables; con estructuras no lineales complejas, puede no ser suficiente.
- **Escalado**: es sensible a la escala de las variables — por eso es común normalizar o estandarizar los datos antes de aplicarlo (igual que en clustering).
- **Interpretabilidad**: las componentes principales son combinaciones lineales de las variables originales, lo que puede dificultar su interpretación directa frente a un público no técnico.

**Para ampliar cada limitación con más ejemplos:**

- **Linealidad**: el ejemplo clásico para ilustrar esta limitación en clase es un dataset con forma de espiral o de "S" en el espacio — PCA, al buscar solo direcciones **rectas** de máxima varianza, no puede "desenroscar" esa estructura y termina proyectando puntos que estaban lejos en la espiral original muy cerca entre sí en el resultado. Para esos casos existen alternativas no lineales (t-SNE, UMAP, autoencoders) que quedan fuera del temario de hoy, pero vale la pena que quien pregunte sepa que existen.
- **Escalado**: si no se estandariza antes, una variable con valores en millones (como `market_value_eur` del dataset de la Clase 04) tendría una varianza numéricamente gigantesca comparada con una variable en unidades chicas (como `age`) — y como PCA busca **maximizar varianza**, terminaría armando la primera componente casi exclusivamente a partir de esa única variable de escala grande, ignorando de hecho a todas las demás. Es la misma razón por la que el escalado es obligatorio en K-Means y DBSCAN, aplicada acá a un problema distinto (varianza en vez de distancia).
- **Interpretabilidad**: cuando la primera componente principal resulta ser, por ejemplo, `0.6 × ingresos + 0.5 × gasto_mensual - 0.3 × edad + ...`, explicarle a un directorio "qué es" esa componente en términos de negocio no es trivial — a diferencia de una variable original como "edad", que se entiende sin esfuerzo. Por eso, en contextos donde la explicabilidad ante un público no técnico es prioritaria, a veces se prefiere sacrificar algo de la reducción de dimensionalidad y quedarse con un subconjunto de variables originales, más fáciles de comunicar aunque menos eficientes matemáticamente.

### Aplicación práctica y relevancia en la industria *(Filmina 30)*

- **Visualización**: reducir dimensiones a 2 o 3 para graficar y detectar patrones o segmentos de clientes a simple vista.
- **Preprocesamiento**: simplificar datos antes de aplicar clustering o clasificación, mejorando el rendimiento y reduciendo ruido.

Por ejemplo, un analista puede usar PCA para transformar variables de comportamiento de compra en componentes principales que resumen tendencias clave, facilitando la segmentación de clientes — y entender la varianza explicada permite justificar cuántos componentes usar, balanceando precisión y simplicidad.

**Para cerrar el módulo con más contexto de uso:**

- **Visualización**: un flujo de trabajo muy habitual en la práctica es aplicar PCA para reducir un dataset de muchas variables a 2 componentes, graficar esos 2 componentes en un scatter plot, y **después** colorear cada punto según el cluster que le asignó K-Means (Módulo 3) — combinando las dos técnicas de la clase para poder "ver" en un gráfico 2D una segmentación que en realidad vive en un espacio de muchas más dimensiones, imposible de graficar directamente.
- **Preprocesamiento**: además de mejorar rendimiento (como se ve en el Módulo 6, con la demo de PCA + KNN), reducir dimensionalidad antes de clustering también ayuda a esquivar la llamada **"maldición de la dimensionalidad"** — un fenómeno donde, en espacios de muchísimas dimensiones, la noción misma de "distancia" empieza a perder sentido (todos los puntos terminan pareciendo casi igual de lejos unos de otros), lo que degrada la calidad de algoritmos como K-Means o DBSCAN que dependen exactamente de medir distancias.
- Un tercer uso, no mencionado explícitamente en la filmina pero común en la industria: la **compresión de datos** — guardar solo las primeras componentes principales de un dataset (en vez de todas las variables originales) para ahorrar espacio de almacenamiento, aceptando una pérdida controlada de información a cambio.

---

## Módulo 6 — Panorama de Métodos (Síntesis)

**Contexto**: cierre conceptual de la clase — comparar las cinco técnicas vistas, entender sus límites, y ver PCA mejorando el rendimiento de un modelo real, no solo en teoría.

### Apertura del módulo de cierre *(Filmina 31)*

La última divisoria de la clase trae el subtítulo "Tu superpoder analítico: consolidando el flujo de trabajo profesional completo" — y funciona como el cierre conceptual de las casi dos horas de clase. A esta altura ya se recorrieron cinco algoritmos concretos (Apriori/FP-Growth, K-Means, Jerárquico, DBSCAN, PCA); este módulo no agrega un sexto algoritmo, sino que da un paso atrás para mirarlos **a todos juntos**.

**Para presentar antes del contenido**: es un buen momento para pedirle al grupo, antes de mostrar ninguna tabla, que intente recordar de memoria los cinco algoritmos vistos y a qué familia pertenece cada uno (clustering: K-Means, Jerárquico, DBSCAN; reducción de dimensionalidad: PCA; reglas de asociación: Apriori/FP-Growth) — es un buen chequeo rápido de qué quedó instalado de la clase antes de pasar al repaso formal de las Filminas 32 a 35. También es el momento de anticipar que el módulo cierra con algo distinto a las clases anteriores: una demostración con números reales de que la elección de técnica (PCA en este caso) no es solo una cuestión teórica, sino que **cambia el resultado de un modelo posterior** de forma medible.

### Decisiones de diseño y parámetros clave *(Filmina 32)*

| Técnica | Parámetros clave | Consideración principal |
|---|---|---|
| **K-Means** | Número de clusters `k` | Elegir `k` adecuado; sensible a valores atípicos |
| **Clustering jerárquico** | Método de linkage (single, complete...) | Interpretación del dendrograma; escalabilidad |
| **DBSCAN** | `eps`, `min_samples` | Detecta ruido; adecuado para formas arbitrarias |
| **PCA** | Número de componentes a conservar | Balance entre reducción y pérdida de información |
| **Apriori** | Soporte mínimo, confianza mínima | Controla cantidad y calidad de reglas generadas |

**Para desarrollar esta tabla en clase, columna por columna:**

Vale la pena remarcar un patrón que atraviesa las cinco filas: **todas** las técnicas de hoy tienen al menos un hiperparámetro que hay que decidir a mano antes de correr el algoritmo, y en **ninguno** de los cinco casos existe una fórmula única que lo calcule automáticamente — solo heurísticas (el codo, el silhouette, el k-distance plot, el umbral de varianza explicada) que ayudan a acercarse a un buen valor. Es una diferencia de fondo respecto al aprendizaje supervisado de la Clase 08, donde muchos hiperparámetros se pueden ajustar de forma más sistemática con `GridSearchCV` comparando contra una métrica objetiva como Accuracy — acá, al no existir una `y` contra la cual medir "qué tan bien salió", la elección de parámetros conserva siempre un componente de criterio humano.

También vale la pena conectar la columna "Consideración principal" con lo ya visto: la sensibilidad de K-Means a valores atípicos (Filmina 14), la escalabilidad limitada del jerárquico con datasets grandes (Filmina 20), la capacidad de DBSCAN de manejar formas arbitrarias (Filmina 22-23), el balance de PCA entre reducción y pérdida de información (Filmina 28), y el control de calidad de reglas de Apriori vía soporte/confianza (Filmina 09) — esta tabla es, en esencia, un resumen de una idea clave por módulo, y sirve como buena guía de repaso rápido antes de un examen o de aplicar estas técnicas en un proyecto real.

### Limitaciones y supuestos básicos *(Filmina 33)*

- El **clustering** asume que la similitud/diferencia entre puntos es significativa y que los datos pueden agruparse con claridad.
- **PCA** asume relaciones lineales y que la varianza es una medida adecuada de "información".
- Las **reglas de asociación** requieren datos transaccionales y pueden generar muchas reglas irrelevantes sin filtros adecuados.

**Para reflexionar en clase**: ¿qué pasaría si aplicás K-Means a datos con clusters de formas muy irregulares? ¿O PCA a datos con relaciones fuertemente no lineales? (Spoiler: en ambos casos, conviene DBSCAN o técnicas no lineales en vez de forzar el método "de siempre".)

**Para ampliar cada supuesto antes de la filmina:**

El hilo conductor de esta filmina es que **ninguna técnica de hoy funciona "a ciegas"** — cada una parte de un supuesto sobre cómo son los datos, y cuando ese supuesto no se cumple, el resultado puede ser engañoso sin que el algoritmo avise del error. El clustering, por ejemplo, siempre va a devolver **algún** agrupamiento, incluso si se le pasan datos generados completamente al azar sin ninguna estructura real — el algoritmo no tiene forma de "darse cuenta" de que no había nada que agrupar, y es responsabilidad de quien lo usa evaluar (con silhouette, por ejemplo) si el resultado tiene sentido real o es ruido estadístico disfrazado de grupos.

Sobre PCA: además de asumir linealidad (ya visto en la Filmina 29), asume que **más varianza significa más información relevante** — un supuesto razonable en la mayoría de los casos, pero que puede fallar si, por ejemplo, una variable tiene mucha varianza justamente por errores de medición (ruido de sensor) y no por señal real; en ese escenario, PCA podría terminar priorizando una dirección que en realidad es puro ruido.

Sobre reglas de asociación: generar reglas sin ningún filtro de soporte/confianza mínimos en un catálogo grande puede producir literalmente millones de reglas técnicamente válidas pero comercialmente inútiles (asociaciones triviales, coincidencias estadísticas) — el criterio de negocio para decidir los umbrales mínimos es tan importante como el algoritmo en sí.

### Aplicaciones prácticas por escenario *(Filmina 34)*

- **Clustering**: segmentación de clientes, detección de fraude agrupando comportamientos atípicos, análisis de patrones en sensores industriales.
- **PCA**: visualización de datos complejos, reducción de ruido antes de un modelo supervisado, compresión de datos para almacenamiento eficiente.
- **Reglas de asociación**: productos que se compran juntos, optimización de layout de tienda, análisis de comportamiento de compra.

En la práctica, la elección depende del contexto de negocio: en un e-commerce con datos ruidosos y clusters de forma compleja, DBSCAN suele ganarle a K-Means.

**Para cerrar con un caso integrador, combinando varias técnicas de la clase:**

Un flujo de trabajo realista en una empresa de e-commerce podría combinar **las tres familias en una sola cadena de análisis**: primero, PCA para reducir docenas de variables de comportamiento de cada cliente (frecuencia de compra, categorías preferidas, monto gastado, dispositivo usado, horario de navegación...) a un puñado de componentes principales que resuman lo esencial; segundo, K-Means o DBSCAN sobre esas componentes reducidas para segmentar a los clientes en grupos con comportamientos similares (más rápido y con mejores resultados que clusterizar sobre las variables originales sin reducir, por la maldición de la dimensionalidad mencionada en el Módulo 5); y tercero, reglas de asociación aplicadas **dentro de cada segmento** por separado, para encontrar patrones de compra específicos de cada grupo de clientes, en vez de patrones genéricos que mezclan comportamientos muy distintos entre sí. Es un buen ejemplo para cerrar la clase mostrando que estas técnicas no compiten entre sí — se combinan.

### Demostración: PCA mejorando un modelo real *(Filmina 35)*

El PDF cierra con un ejemplo didáctico controlado que demuestra, con números, que PCA puede **mejorar** el rendimiento de un modelo — no solo "comprimir" datos:

- Dataset de **cáncer de mama** (scikit-learn, 30 *features* reales).
- Se agregan **300 columnas de ruido** (features irrelevantes) a propósito, simulando un escenario de alta dimensionalidad con mucha señal desperdiciada.
- Clasificador **KNN**, elegido por ser sensible al *curse of dimensionality* (empeora notablemente con muchas features irrelevantes).
- Se compara accuracy **sin PCA** vs. **con PCA** (reducción fuerte: 330 → 30 componentes).

**Para presentar el diseño del experimento antes de correr el código:**

Vale la pena explicar por qué el experimento está armado exactamente así, porque el diseño es parte de lo que hace convincente la demostración. El dataset de cáncer de mama (`load_breast_cancer`) ya viene con 30 variables reales y significativas (medidas de núcleos celulares). Agregarle 300 columnas de **ruido gaussiano puro** — números aleatorios sin ninguna relación con si el tumor es maligno o benigno — simula, de forma controlada y medible, algo que pasa todo el tiempo en datasets reales: una tabla con muchas columnas donde solo una fracción de ellas realmente importa para el problema, y el resto es "ruido" (variables mal elegidas, redundantes, o simplemente irrelevantes para la pregunta puntual que se está resolviendo).

La elección de **KNN** como clasificador no es casual: KNN clasifica un punto nuevo mirando literalmente qué tan cerca está de sus vecinos ya clasificados — y esa noción de "cerca" se calcula con distancia sobre **todas** las columnas por igual, ruido incluido. Con 300 columnas de ruido contra solo 30 de señal real, la distancia entre dos puntos queda dominada casi por completo por coincidencias aleatorias en las columnas de ruido, y el vecino "más cercano" deja de ser realmente el más parecido en términos clínicos. Es la manifestación concreta de la maldición de la dimensionalidad mencionada en el Módulo 5. Un modelo como Random Forest, en cambio, sería mucho menos sensible a este mismo experimento — porque puede aprender a ignorar variables irrelevantes; la elección de KNN está pensada a propósito para que el efecto de PCA se note con claridad.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import numpy as np

SEED = 42
np.random.seed(SEED)

# Dataset real + 300 columnas de ruido gaussiano añadidas a propósito
datos = load_breast_cancer()
X_real, y = datos.data, datos.target
X_ruido = np.random.normal(loc=0.0, scale=1.0, size=(X_real.shape[0], 300))
X = np.hstack([X_real, X_ruido])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)

# Baseline SIN PCA: KNN directo sobre las 330 columnas (30 reales + 300 de ruido)
pipe_sin_pca = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
pipe_sin_pca.fit(X_train, y_train)
acc_sin_pca = accuracy_score(y_test, pipe_sin_pca.predict(X_test))

# CON PCA: reducción fuerte (330 -> 30 componentes) antes del mismo KNN
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

pca = PCA(n_components=30, random_state=SEED)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca = pca.transform(X_test_s)

knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
acc_con_pca = accuracy_score(y_test, knn_pca.predict(X_test_pca))

print(f"Accuracy SIN PCA: {acc_sin_pca:.4f}")
print(f"Accuracy CON PCA: {acc_con_pca:.4f}")
```

**Línea por línea:**
- `np.random.normal(..., size=(X_real.shape[0], 300))` → genera 300 columnas de ruido gaussiano puro, sin ninguna relación con `y`; se concatenan a las 30 columnas reales con `np.hstack`.
- `train_test_split(..., stratify=y)` → `stratify=y` mantiene la misma proporción de clases (maligno/benigno) en train y test — clave en clasificación, ya visto en la Clase 08.
- `make_pipeline(StandardScaler(), KNeighborsClassifier(...))` → el mismo patrón de `Pipeline` de la Clase 08: escala y clasifica en un solo paso, evitando Data Leakage.
- `pca.fit_transform(X_train_s)` / `pca.transform(X_test_s)` → **regla de oro** (la misma que en imputación/escalado): el PCA se ajusta (`fit`) solo con datos de entrenamiento, y se aplica (`transform`) a ambos conjuntos — nunca se ajusta sobre test.
- **Resultado real, corriendo este código**: `Accuracy SIN PCA: 0.8531` vs. `Accuracy CON PCA: 0.9161` — una mejora de más de 6 puntos porcentuales. La razón: KNN mide distancias, y con 300 columnas de ruido esas distancias quedan "contaminadas"; PCA concentra la señal real en pocas componentes y descarta gran parte del ruido, mejorando la relación señal/ruido que ve el clasificador.

> **Con esto cierra la clase.** El panorama completo: cinco técnicas (K-Means, Jerárquico, DBSCAN, PCA, Apriori), cada una con su caso de uso, sus parámetros y sus límites — y una prueba concreta de que elegir bien la técnica de preprocesamiento (PCA) puede ser la diferencia entre un modelo mediocre y uno bueno, incluso antes de tocar el algoritmo de predicción en sí.
