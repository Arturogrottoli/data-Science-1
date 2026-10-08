# Clase 09: Aprendizaje No Supervisado — Guía Completa para el Docente

Esta guía es el **libreto de apoyo para dictar la Clase 09**. Reúne, en un solo lugar y con más profundidad de la que entra en una diapositiva, toda la teoría que aparece en:

- **`Clase 09_fixed.docx`** — el material teórico oficial de la unidad (reemplaza a `Clase 09_teoria.pdf`, que queda obsoleto).
- **`Clase09.html`** — las diapositivas que se proyectan en clase (36 filminas).

> **Estado de esta guía**: actualizada para seguir el nuevo `Clase 09_fixed.docx`. Se sacaron Reglas de Asociación (Apriori/FP-Growth), Clustering Jerárquico y la base matemática de PCA (covarianza/eigenvectores) porque el docx nuevo ya no los trae; se agregó Customer Profiling, Ética y Sesgos, y la Pre-entrega, que sí trae. El notebook de la clase (`Clase_9.ipynb`) todavía refleja la estructura vieja — queda pendiente para una próxima iteración.

---

## Índice

- [Mapa rápido de la clase](#mapa-rápido-de-la-clase)
- [Módulo 0 — Repaso: Aprendizaje Supervisado](#módulo-0--repaso-aprendizaje-supervisado-puente-desde-la-clase-08)
- [Módulo 1 — ¿Qué es el Aprendizaje No Supervisado?](#módulo-1--qué-es-el-aprendizaje-no-supervisado)
- [Módulo 2 — K-Means y la Elección de k](#módulo-2--k-means-y-la-elección-de-k)
- [Módulo 3 — DBSCAN: Clustering Basado en Densidad](#módulo-3--dbscan-clustering-basado-en-densidad)
- [Módulo 4 — PCA: Reducción de Dimensionalidad](#módulo-4--pca-reducción-de-dimensionalidad)
- [Módulo 5 — Panorama de Métodos (Síntesis)](#módulo-5--panorama-de-métodos-síntesis)
- [Módulo 6 — Customer Profiling](#módulo-6--customer-profiling)
- [Módulo 7 — Ética, Sesgos y Cierre](#módulo-7--ética-sesgos-y-cierre)
- [Pre-entrega: Aprendizaje No Supervisado](#pre-entrega-aprendizaje-no-supervisado)

---

## Mapa rápido de la clase

Para seguir la clase en paralelo con `Clase09.html` (36 filminas) sin perderte:

| # | Módulo | Slides | Idea central |
|---|---|---|---|
| — | Portada | 01 | Presentación de la clase |
| 0 | Repaso: Aprendizaje Supervisado | 02–04 | Solo los nombres — la explicación completa vive en esta guía, no en la filmina |
| 1 | ¿Qué es el Aprendizaje No Supervisado? | 05–09 | Sin etiquetas: clustering, reducción de dimensionalidad, detección de anomalías |
| 2 | K-Means y la Elección de k | 10–16 | El algoritmo de clustering más usado, y cómo elegir bien su parámetro clave |
| — | Break del Coder | 17 | Corte de ~10 minutos |
| 3 | DBSCAN: Clustering Basado en Densidad | 18–21 | Densidad, ruido, y comparación directa contra K-Means |
| 4 | PCA: Reducción de Dimensionalidad | 22–25 | Varianza explicada, limitaciones, aplicación en la industria |
| 5 | Panorama de Métodos (Síntesis) | 26–30 | Comparación de K-Means/DBSCAN/PCA + demo real de PCA mejorando un modelo |
| 6 | Customer Profiling | 31–32 | Traducir clústeres matemáticos en perfiles de cliente accionables |
| 7 | Ética, Sesgos y Cierre | 33–35 | Interpretación responsable, sin Ground Truth, y la Pre-entrega del módulo |
| — | ¿Dudas? | 36 | Cierre y preguntas |

---

## Módulo 0 — Repaso Express: Aprendizaje Supervisado (puente desde la Clase 08)

**Por qué este módulo**: la Clase 08 cerró el bloque de aprendizaje supervisado. Antes de arrancar con la Clase 09, conviene un repaso corto — no para volver a enseñarlo, sino para que el contraste con lo de hoy quede bien marcado.

### ¿Qué es Machine Learning, en general? *(Filmina 02)*

Antes de meternos en "supervisado" puntualmente, vale la pena bajar un escalón más y aclarar la idea más general en la que se apoya toda esta clase. Programar una computadora "a la vieja usanza" significa escribirle instrucciones explícitas para cada caso: "si pasa esto, hacé esto otro". El problema es que hay tareas donde armar esas reglas a mano es imposible — nadie puede escribir manualmente todas las reglas que distinguen una foto de un gato de una foto de un perro, o que dicen exactamente cuándo un mail es spam.

**Machine Learning (aprendizaje automático)** resuelve eso de otra manera: en vez de programarle las reglas a la máquina, se le muestran muchos ejemplos y se deja que ella misma **encuentre los patrones** y arme sus propias reglas internas, a base de repetición y ajuste. Es parecido a cómo un chico aprende a reconocer animales: nadie le da una lista de reglas escritas ("un perro tiene 4 patas, pelo, y ladra") — simplemente ve muchos perros distintos, y con el tiempo su cerebro arma solo el patrón que le permite reconocer un perro nuevo que nunca vio.

Dentro de Machine Learning hay distintas formas de "dejar que la máquina aprenda sola", según qué tipo de datos y qué tipo de ayuda se le da durante ese aprendizaje. La que ya se vio en la Clase 08 —y la que se repasa a continuación— es el **aprendizaje supervisado**; la que arranca hoy es el **aprendizaje no supervisado**, con una diferencia central que se explica más abajo.

### ¿Qué es el aprendizaje supervisado? *(Filmina 02)*

La idea de fondo es muy parecida a cómo aprende una persona con ejemplos resueltos: si querés aprender a distinguir mails de spam, lo más fácil es que alguien te muestre miles de mails **ya marcados** como "spam" o "no spam", y con el tiempo empezás a notar patrones (ciertas palabras, remitentes raros, exceso de mayúsculas) que te ayudan a clasificar un mail nuevo que nunca viste. Eso es exactamente lo que hace un modelo de aprendizaje supervisado: se le muestran muchos ejemplos donde la respuesta correcta **ya se conoce**, y el modelo va ajustando sus parámetros internos hasta encontrar una regla (una función matemática) que relacione los datos de entrada con esa respuesta. Una vez entrenado, se usa esa regla para predecir la respuesta de casos **nuevos**, donde no se conoce de antemano.

En la notación que se usa en la jerga de Machine Learning: a las variables de entrada (edad, ingresos, antigüedad laboral, cantidad de habitaciones de una casa...) se las llama `X`; a la respuesta que se quiere predecir (spam o no, precio de la casa) se la llama `y`. Entrenar un modelo supervisado es, ni más ni menos, buscar una función `f` tal que `f(X)` se parezca lo más posible a `y`, usando los ejemplos históricos donde ambas cosas ya se conocen.

Existen dos grandes familias, según qué tipo de dato es `y`:

| | Clasificación | Regresión |
|---|:---:|:---:|
| **`y` es...** | Una categoría | Un número |
| **Ejemplo** | ¿El cliente paga el préstamo? | ¿Precio de la vivienda? |
| **Métricas** | Accuracy, F1, AUC-ROC | MAE, RMSE, R² |

- **Clasificación**: la respuesta que se quiere predecir es una **etiqueta**, elegida entre un grupo cerrado de opciones — "paga" o "no paga", "spam" o "no spam". No hay término medio: la predicción es una de esas categorías, no un número.
- **Regresión**: la respuesta que se quiere predecir es un **número** que puede tomar cualquier valor — el precio de una casa (podría ser $150.234 o $150.987, cualquier cifra), la temperatura de mañana. Acá sí hay término medio: el modelo puede acertar "más o menos", no es todo o nada.

**Cómo se mide si un modelo de clasificación es bueno** — con un ejemplo concreto: un banco evalúa el modelo sobre 100 clientes a los que ya les prestó dinero en el pasado, así que ya se sabe qué pasó realmente con cada uno (90 pagaron a tiempo, 10 no pagaron / entraron en mora).
- **Accuracy** es lo más simple de entender: de esos 100 casos, ¿en cuántos acertó el modelo (predijo "no paga" cuando efectivamente no pagó, o "paga" cuando efectivamente pagó)? Si acertó en 92, el Accuracy es 92%.
- El problema de quedarse solo con Accuracy: si el modelo fuera tan vago que dijera **siempre** "va a pagar", sin analizar nada, igual acertaría en los 90 clientes que sí pagaron y solo fallaría en los 10 que no — un Accuracy del 90%, que suena bien pero es un modelo completamente inútil para el banco (nunca detecta a un cliente riesgoso, que es justo el caso que importa detectar antes de prestarle plata).
- Por eso existe **F1**, que en realidad combina dos métricas más chicas y específicas. Sigamos con el ejemplo: supongamos que el modelo marca a **12 clientes** como "riesgosos" (predijo que no van a pagar). De esos 12, después se descubre que **8 realmente no pagaron** y **4 sí pagaron** (el modelo se equivocó con ellos). Y de los 10 clientes que en la realidad no pagaron, el modelo solo llegó a detectar a 8 de ellos (se le escaparon 2).
  - **Precision** ("precisión"): de los que el modelo marcó como riesgosos, ¿cuántos realmente lo eran? → 8 de 12 = **67%**. Si la Precision es baja, el modelo está siendo "alarmista": marca a mucha gente como riesgosa sin serlo (eso tiene un costo — por ejemplo, rechazarle el préstamo a un buen cliente).
  - **Recall** ("exhaustividad" o "sensibilidad"): de los que realmente no iban a pagar, ¿a cuántos detectó el modelo? → 8 de 10 = **80%**. Si el Recall es bajo, el modelo está siendo "distraído": deja pasar casos riesgosos de verdad sin detectarlos (ese es el error más caro para el banco — prestarle plata a alguien que no va a pagar).
  - **F1** es un promedio especial entre Precision y Recall (técnicamente se llama "media armónica", pero para la intuición alcanza con pensarlo como un promedio) que tiene una propiedad importante: si **cualquiera** de las dos (Precision o Recall) es mala, el F1 también sale malo — no alcanza con que una de las dos sea excelente para "tapar" a la otra. En este ejemplo, con Precision 67% y Recall 80%, el F1 da aproximadamente **73%**.
  - Comparado con el modelo "vago" de antes (el que siempre dice "va a pagar", sin marcar a nadie como riesgoso): ese modelo tiene Recall = 0% (no detecta ni un solo caso riesgoso real) — y ahí el F1 se derrumba a 0%, aunque su Accuracy fuera 90%. Ese es justamente el contraste que hace útil a F1: expone a los modelos que "hacen trampa" con Accuracy sin detectar nada de lo que realmente importa.
  - Nota sobre el nombre: a diferencia de AUC-ROC (que sí es una sigla con significado, ver abajo), "F1" no es la abreviatura de ninguna frase — es simplemente el nombre técnico de esta fórmula puntual (también se la llama "F1-score" o "F-measure"). No hace falta buscarle un significado oculto al nombre, solo recordar que combina Precision y Recall.

- Una tercera métrica que se mencionó en la Clase 08 es **AUC-ROC** — acá sí conviene desglosar la sigla completa: **AUC** es *Area Under the Curve* (Área Bajo la Curva) de la **ROC**, que es *Receiver Operating Characteristic* (algo así como "Característica Operativa del Receptor" — un nombre que viene de la ingeniería de radares de mediados del siglo XX y que hoy no aporta ninguna intuición; no hace falta memorizar por qué se llama así, solo entender qué mide).

  Para entenderlo hay que retomar algo mencionado arriba: el modelo no dice "sí" o "no" directamente, calcula una **probabilidad** (por ejemplo, "este cliente tiene 75% de probabilidad de no pagar") y recién después esa probabilidad se convierte en una decisión final usando un **umbral** — por ejemplo, "lo marco como riesgoso si su probabilidad supera 50%". Pero ese umbral (el 50%) es una elección arbitraria: se podría usar 30% (el banco se vuelve más desconfiado, marca a más gente como riesgosa) o 70% (el banco se vuelve más permisivo, marca a menos gente).

  La **curva ROC** se construye probando **todos los umbrales posibles**, del 0% al 100%, y graficando en cada uno dos números uno contra el otro: cuántos clientes riesgosos de verdad logra detectar el modelo (el Recall de antes) contra cuántos clientes buenos termina marcando por error (la contracara de la Precision). El **AUC** es, literalmente, el área que queda debajo de esa curva — un único número que resume qué tan bien el modelo separa a los dos grupos (los que pagan de los que no), sin depender de qué umbral puntual se termine usando.

  Los dos valores de referencia para interpretarlo: **AUC = 1** sería un modelo perfecto — existe un umbral donde separa completamente a un grupo del otro, sin ningún error. **AUC = 0,5** es lo mismo que decidir tirando una moneda al aire — el modelo no tiene ninguna capacidad real de distinguir un cliente riesgoso de uno confiable, por más ajustes de umbral que se prueben. En la práctica, un AUC de 0,8-0,9 ya se considera bastante bueno para la mayoría de los problemas reales.

**Cómo se mide si un modelo de regresión es bueno** — con otro ejemplo: un modelo que predice precios de casas.
- **MAE** (Error Absoluto Medio): agarra la diferencia entre lo que predijo el modelo y el precio real de cada casa, y promedia esas diferencias (sin importar si se equivocó "de más" o "de menos"). Si el MAE da $10.000, quiere decir que, en promedio, el modelo se equivoca por $10.000 en cada predicción — un número fácil de interpretar porque está en la misma unidad (dólares) que lo que se está prediciendo.
- **RMSE**: muy parecido al MAE, pero antes de promediar los errores los eleva al cuadrado (y al final saca la raíz cuadrada del resultado). El efecto práctico: un error grande pesa mucho más que varios errores chicos — un modelo que casi siempre acierta bien pero se equivoca feo en un par de casas raras va a tener un RMSE bastante peor que su MAE, mientras que un modelo con errores parejos y moderados va a tener MAE y RMSE parecidos entre sí.
- **R²**: en vez de dar un error en dólares, da un número entre 0 y 1 (a veces se explica como porcentaje) que responde "¿qué tan bien el modelo explica por qué el precio de cada casa es el que es?". Un R² de 1 sería un modelo perfecto (acierta el precio exacto siempre); un R² de 0 significa que el modelo no es mejor que simplemente decir siempre "el precio promedio de todas las casas", sin mirar ninguna variable en particular.

### Los modelos que se vieron en la Clase 08 *(Filmina 03)*

Estos cinco modelos son las herramientas concretas con las que se resuelven los problemas de clasificación y regresión. Repasarlos uno por uno, con una idea intuitiva de cómo funciona cada uno:

- **Regresión Lineal**: el modelo más simple de todos — busca la "mejor línea recta" (o, con más de una variable de entrada, el mejor plano) que pase lo más cerca posible de todos los puntos de entrenamiento. Ejemplo: predecir el precio de una casa a partir de sus metros cuadrados — a más metros cuadrados, más precio, y la Regresión Lineal encuentra la relación numérica exacta ("cada metro cuadrado extra suma, en promedio, tantos dólares"). Es un modelo de **regresión** (predice un número), muy fácil de interpretar, pero limitado cuando la relación entre las variables no es una línea recta.
- **Árbol de Decisión**: funciona como un juego de "20 preguntas" — va haciendo preguntas de sí/no sobre los datos ("¿el ingreso es mayor a $50.000?", "¿tiene más de 30 años?"), y según las respuestas va bajando por ramas del árbol hasta llegar a una predicción final en una "hoja". Se puede usar tanto para clasificación ("¿el cliente paga el préstamo o no?") como para regresión ("¿cuánto va a gastar este cliente?"). Su gran ventaja es que es muy fácil de visualizar y explicar — literalmente se puede dibujar el árbol de preguntas y mostrárselo a alguien sin conocimientos técnicos.
- **Random Forest**: en vez de confiar en un único Árbol de Decisión (que puede memorizar demasiado los datos de entrenamiento y funcionar mal con datos nuevos), Random Forest entrena **muchos** árboles distintos — cada uno viendo una porción distinta, al azar, de los datos y de las variables — y después promedia (en regresión) o vota por mayoría (en clasificación) las predicciones de todos ellos. La idea es la misma que "preguntarle a un grupo de expertos en vez de a uno solo": el resultado grupal suele ser más confiable que el de un único árbol, porque los errores individuales de cada árbol tienden a cancelarse entre sí.
- **Regresión Logística**: a pesar del nombre (que confunde a todo el mundo la primera vez), **no es un modelo de regresión sino de clasificación**. Se usa para predecir la probabilidad de que algo pertenezca a una categoría — por ejemplo, la probabilidad de que un cliente no pague un préstamo, entre 0% y 100% — y después esa probabilidad se convierte en una predicción final ("riesgoso" si la probabilidad supera 50%, por ejemplo). El nombre viene de que matemáticamente usa una función llamada "logística" para convertir un cálculo interno en un número entre 0 y 1.
- **KNN (K-Nearest Neighbors, "K vecinos más cercanos")**: la idea más intuitiva de las cinco — para predecir la categoría (o el valor) de un caso nuevo, mira cuáles son los `K` casos **ya conocidos** más parecidos a él (los "vecinos más cercanos", midiendo distancia entre sus variables), y les copia la respuesta mayoritaria. Ejemplo: para adivinar si a alguien le va a gustar una película, KNN mira a los `K` usuarios con gustos más parecidos a los suyos, y se fija qué opinaron ellos de esa película. No necesita "entrenarse" en el sentido tradicional — simplemente guarda todos los datos y compara en el momento de predecir.

### Buenas prácticas: evitar el Data Leakage *(Filmina 04)*

Uno de los errores más peligrosos (porque no siempre se nota) en Machine Learning es el ***Data Leakage*** ("fuga de datos"): que información del conjunto de **test** (los datos que se supone el modelo nunca vio, usados solo para evaluar qué tan bien predice) se "filtre" de alguna forma hacia el proceso de entrenamiento. Cuando eso pasa, el modelo parece funcionar excelente durante la evaluación, pero en la vida real (con datos genuinamente nuevos) rinde mucho peor — porque en el fondo "hizo trampa" viendo pistas que no debería haber visto.

Un ejemplo concreto de cómo ocurre sin querer: si se calcula el promedio y el desvío estándar de una columna usando **todo** el dataset (entrenamiento + test juntos) para escalar los datos, y **después** se separa en train/test, el modelo ya "vio" información estadística de los datos de test (su promedio, su dispersión) antes de ser evaluado con ellos. Es una fuga sutil, fácil de cometer sin darse cuenta, y por eso la Clase 08 insistió en dos herramientas concretas para evitarla:

- **`StandardScaler`**: el nombre está compuesto de dos palabras en inglés — *"standard"* (estándar) y *"scaler"* (algo que escala, que cambia de tamaño/escala). Literalmente es "el escalador que lleva todo a una escala estándar". Y eso es exactamente lo que hace: reescala las variables numéricas para que todas queden en una escala comparable (en general, restando el promedio y dividiendo por el desvío estándar, de forma que la variable termine con promedio 0 y desvío 1 — esa combinación de promedio 0 y desvío 1 es, por convención estadística, "la escala estándar"). Es necesario porque muchos modelos (KNN es el caso más claro, ya que mide distancias) se ven distorsionados si una variable está en una escala mucho más grande que otra — por ejemplo, "ingresos" en miles de dólares vs. "edad" en años: sin escalar, la variable "ingresos" dominaría por completo cualquier cálculo de distancia o similitud, aunque "edad" fuera igual de importante para el problema.
- **`Pipeline`**: en inglés, *"pipeline"* es literalmente un **caño** o **tubería** — el mismo término que se usa para un oleoducto. La imagen mental es la de un líquido que entra por un extremo y va pasando por una serie de tramos conectados hasta salir transformado por el otro extremo; en informática se usa esa misma palabra para nombrar cualquier secuencia de pasos conectados, donde la salida de un paso es la entrada del siguiente. En scikit-learn, un `Pipeline` encadena todos los pasos (escalado, y después el modelo) en un único objeto — los datos "entran" por el escalador y "salen" ya transformados y clasificados/predichos, sin pasos sueltos en el medio. La ventaja concreta: cuando se usa `Pipeline` correctamente (ajustando el escalador **solo** con los datos de entrenamiento, nunca con los de test), es mucho más difícil cometer el error de fuga de datos por accidente — el `Pipeline` fuerza a que cada paso se aplique en el orden correcto, sin mezclar información de test dentro del entrenamiento.

**`train_test_split` con `stratify`**: el nombre de la función es literal en inglés — *"train"* (entrenar) + *"test"* (probar/evaluar) + *"split"* (dividir, partir en dos) — es, sin vueltas, "dividir en entrenamiento y prueba". Antes de entrenar cualquier modelo, se separa el dataset en dos partes usando esta función — una porción (típicamente 70-80%) para **entrenar** el modelo, y el resto para **evaluarlo** con datos que no vio durante el entrenamiento (simulando qué tan bien funcionaría con casos reales nuevos). El parámetro `stratify` viene de la palabra **estrato** (una capa o subgrupo dentro de una población) — en estadística, "muestreo estratificado" significa dividir a la población en subgrupos (estratos) y asegurarse de tomar una porción proporcional de **cada uno**, en vez de tomar una muestra completamente al azar que podría (por mala suerte) dejar algún subgrupo sub-representado. Acá los "estratos" son las categorías de `y`: si solo el 5% de los clientes del dataset no pagaron su préstamo, un split al azar (sin `stratify`) podría dejar casi ningún caso de impago en el conjunto de test, haciendo que la evaluación no sea representativa. `stratify=y` le asegura al split que mantenga la misma proporción de cada categoría (5% no paga / 95% paga) tanto en entrenamiento como en test.

### Validación: por qué un solo split no alcanza *(Filmina 04)*

Confiar en un único `train_test_split` tiene un problema: el resultado de la evaluación depende, en parte, de **qué** casos cayeron por azar en el conjunto de test — con otro split distinto (otros casos al azar), la métrica final podría salir un poco distinta, mejor o peor, sin que el modelo en sí haya cambiado. Para tener una medida más confiable y menos dependiente de la suerte del split, se usa la **validación cruzada** (*cross-validation*):

- **`StratifiedKFold`**: el nombre junta tres piezas — *"Stratified"* (estratificado, la misma idea de "muestra proporcional por subgrupo" que `stratify`), *"K"* (la cantidad de partes en las que se divide, un número que se elige — 5 y 10 son los valores más comunes) y *"Fold"* (en inglés, "pliegue" o "doblez" — como doblar una hoja de papel varias veces; cada doblez es una de las particiones del dataset, un "fold"). Entero, el nombre dice "dividir en K pliegues, de forma estratificada". En vez de partir el dataset en un solo par entrenamiento/test, lo divide en `K` partes iguales (folds) — por ejemplo, 5 partes. El proceso entrena y evalúa el modelo `K` veces distintas: en cada vuelta, usa una parte distinta como test y las `K-1` restantes como entrenamiento. Al final, se tienen `K` mediciones de la métrica elegida, no una sola. Que sea "Stratified" garantiza que cada uno de esos `K` folds mantenga la misma proporción de categorías que el dataset completo.
- **`cross_val_score`**: el nombre es la forma abreviada (típica en programación, para no escribir nombres kilométricos) de *"cross validation score"* — *"cross"* (cruzado/cruzada, en el sentido de que los folds se van intercambiando el rol de test), *"validation"* (validación, el proceso de comprobar qué tan bien funciona el modelo) y *"score"* (puntaje, el resultado numérico de esa validación). Es la función de scikit-learn que automatiza todo el proceso de `StratifiedKFold` — entrena y evalúa el modelo las `K` veces, y devuelve las `K` métricas resultantes, listas para promediar. En vez de reportar un único número ("el modelo tuvo 85% de Accuracy"), la buena práctica es reportar el promedio **y** la dispersión de esas `K` mediciones ("85% ± 3%") — un desvío chico entre folds indica que el modelo es estable y confiable; un desvío grande es una señal de alerta de que el resultado depende mucho de qué datos le tocaron, y que probablemente no generalice bien a casos nuevos.

### Lo que cambia hoy

El aprendizaje no supervisado parte de datos **sin `y`** — sin una respuesta correcta conocida de antemano. El objetivo deja de ser predecir y pasa a ser **descubrir estructura**, por dos caminos distintos (cada uno se desarrolla en profundidad más adelante en esta guía, esto es solo la idea de arranque):

- **Clustering** (agrupamiento): armar grupos de observaciones parecidas entre sí, sin que nadie le diga de antemano cuáles son esos grupos ni cuántos hay — por ejemplo, agrupar clientes con comportamientos de compra similares, dejando que el propio algoritmo descubra los perfiles, en vez de definirlos a mano.
- **Reducción de dimensionalidad**: cuando un dataset tiene muchísimas columnas (variables), resumir esa información en unas pocas "columnas nuevas" que capturan lo esencial, para poder analizarla o graficarla sin perder demasiado en el camino.

Una aplicación que combina ambas ideas y aparece una y otra vez en esta clase es la **detección de anomalías**: usar clustering (o la distancia a los grupos "normales") para encontrar los puntos que no se parecen a nada — el ejemplo típico es una transacción bancaria fraudulenta, que no encaja en ningún patrón de compra habitual.

Estos son los frentes que recorre el resto de esta clase, cada uno con su propio módulo.

---

## Módulo 1 — ¿Qué es el Aprendizaje No Supervisado?

### Apertura del módulo *(Filmina 05)*

Esta filmina es la divisoria que abre el Módulo 1 — el título en pantalla ("¿Qué es el Aprendizaje No Supervisado? — Definición, tipos de problemas, ejemplos industriales y flujo típico de trabajo") funciona como el "índice hablado" de los próximos 15-20 minutos de clase. Antes de avanzar a la Filmina 06, es el momento de instalar oralmente, en una frase cada una y sin apoyarte todavía en la próxima diapositiva, las dos definiciones generales que van a servir de ancla durante el resto de la clase:

- **Aprendizaje supervisado** (lo que se cerró en la Clase 08): a partir de datos históricos donde **cada** ejemplo trae una respuesta ya conocida (una etiqueta), el modelo aprende una función que relaciona las variables de entrada con esa respuesta, con el objetivo de predecir la respuesta de casos nuevos donde todavía no se conoce. Es aprender "con la solución del libro al lado".
- **Aprendizaje no supervisado** (lo que arranca ahora): a partir de datos donde **ningún** ejemplo trae una respuesta conocida, el modelo busca regularidades, agrupamientos o estructuras internas — no para predecir un valor puntual, sino para describir cómo están organizados los datos por sí mismos. Es aprender "sin la solución del libro", encontrando el patrón a fuerza de mirar los datos.

Una forma de presentar el contraste en clase, con un ejemplo cotidiano: un supervisado es como aprender a distinguir perros de gatos porque alguien te mostró miles de fotos ya etiquetadas "perro"/"gato"; un no supervisado es como que te den una pila de miles de fotos de animales sin ningún cartel, y tengas que agruparlas vos mismo por similitud, sin que nadie te haya dicho de antemano cuántos grupos hay ni cómo se llaman. El resultado del segundo ejercicio puede coincidir con "perros" y "gatos" — pero el algoritmo llegó ahí solo por semejanza visual, no porque alguien le haya enseñado esas categorías.

Conviene remarcar en voz alta, antes de pasar a la Filmina 06, los cuatro bloques que anuncia esta diapositiva y que se van a recorrer en orden: (1) una definición formal de qué es el aprendizaje no supervisado, (2) los tipos de problemas que lo componen, (3) ejemplos concretos de la industria, y (4) el flujo de trabajo típico que se va a repetir, con variaciones, en cada módulo siguiente de la clase.

### Definición y diferencias con el aprendizaje supervisado *(Filmina 06)*

El aprendizaje no supervisado es un conjunto de técnicas de Machine Learning que buscan identificar estructuras, patrones o relaciones en datos que **no cuentan con etiquetas o respuestas conocidas**. A diferencia del aprendizaje supervisado (Módulo 0), donde el modelo aprende a partir de ejemplos con etiquetas, acá el objetivo es descubrir información oculta sin guía explícita.

**Para desarrollar antes de mostrar la tabla comparativa**: en la Clase 08 el flujo siempre fue el mismo — separar `X` (variables) de `y` (la respuesta a predecir), entrenar un modelo que aprenda esa relación, y medir qué tan bien predice sobre datos nuevos. Ese flujo depende por completo de que `y` exista y esté bien etiquetada — conseguir ese etiquetado en la vida real casi siempre implica un costo (alguien tuvo que revisar cada transacción y marcarla "fraude"/"no fraude", cada imagen y marcarla "gato"/"no gato"). El aprendizaje no supervisado nace, en parte, como respuesta a ese costo: la enorme mayoría de los datos que genera cualquier empresa **no tienen etiqueta**, y etiquetarlos a mano no siempre es viable en tiempo o presupuesto. Estas técnicas permiten extraer valor de esos datos "tal como vienen", sin la etapa previa de etiquetado.

Otra forma de plantear la diferencia, útil para la clase: en el aprendizaje supervisado el científico de datos sabe de antemano **qué pregunta** está respondiendo el modelo ("¿es spam?", "¿cuánto va a costar?"). En el no supervisado, muchas veces ni siquiera se sabe con precisión qué se va a encontrar — el algoritmo puede revelar una segmentación de clientes que nadie había considerado, o una relación entre productos que el equipo de marketing no había notado. Por eso al aprendizaje no supervisado también se lo asocia con el **análisis exploratorio**: se usa tanto para resolver un problema puntual como para "conocer" un dataset nuevo antes de decidir qué hacer con él.

| Característica | Aprendizaje Supervisado | Aprendizaje No Supervisado |
|---|---|---|
| Datos de entrada | Con etiquetas o respuestas | Sin etiquetas |
| Objetivo | Predecir o clasificar | Encontrar patrones o estructuras |
| Ejemplos de problemas | Clasificación, regresión | Clustering, reducción de dimensionalidad, detección de anomalías |

**Un matiz que vale la pena mencionar en clase** (aunque se profundiza en cursos más avanzados): la frontera entre ambos mundos no siempre es absoluta. Existen enfoques intermedios — el aprendizaje **semi-supervisado** (una pequeña porción de datos etiquetados, mucha data sin etiquetar) y el aprendizaje **autosupervisado** (el propio dataset genera sus etiquetas, por ejemplo tapando parte de una imagen y pidiéndole al modelo que la reconstruya). No forman parte del temario de hoy, pero saber que existen ayuda a entender que "supervisado vs. no supervisado" es más un espectro que una dicotomía cerrada.

### Tres grandes tipos de problemas *(Filmina 07)*

1. **Clustering (agrupamiento)**: agrupa datos similares en clusters. Ejemplo: segmentar clientes según comportamiento de compra.
2. **Reducción de dimensionalidad**: simplifica datos complejos con muchas variables a representaciones más manejables. Ejemplo: usar PCA para visualizar datos en 2D o 3D.
3. **Detección de anomalías**: encuentra la "aguja en el pajar" — puntos que no se parecen a ningún grupo normal. Ejemplo: una transacción bancaria que no encaja con el comportamiento habitual del usuario.

Estas tres categorías se exploran en detalle en los Módulos 2 a 5 de esta clase, cada una con sus algoritmos y métricas propias.

**Material para desarrollar cada punto en clase, antes de pasar a la Filmina 08:**

- **Clustering** responde a la pregunta *"¿quién se parece a quién?"*. No hay un número de grupos predefinido de antemano (salvo que el algoritmo lo pida como parámetro, como K-Means) — el propio proceso de agrupar es el resultado que se busca. Vale la pena anticipar acá que en esta clase se van a ver **dos** algoritmos distintos de clustering (K-Means y DBSCAN), y que ninguno es "el mejor" en términos absolutos: cada uno asume cosas distintas sobre la forma de los grupos, y esa es la razón por la que hace falta conocer más de uno.
  - Segmentar clientes de un e-commerce por comportamiento de compra (frecuencia, monto, categorías) para armar campañas distintas por perfil.
  - Agrupar canciones de una plataforma de streaming por "sonido" (tempo, energía, instrumentación) para armar playlists automáticas sin que nadie las arme a mano.
  - Agrupar pacientes de un hospital por perfil de síntomas, para descubrir subtipos de una enfermedad que la clasificación clínica tradicional no distinguía.
  - Agrupar barrios de una ciudad por patrones de tráfico y movilidad, para decidir dónde priorizar inversión en transporte público.
- **Reducción de dimensionalidad** responde a *"¿puedo decir lo mismo con menos variables?"*. Es fácil de subestimar si nunca se trabajó con un dataset de verdad ancho — pero es común encontrar tablas con cientos de columnas (encuestas, datos genómicos, sensores IoT), donde ni siquiera es posible graficar todas las relaciones a la vez. PCA, que se ve en el Módulo 4, es la técnica de referencia acá — pero el concepto general ("comprimir información sin perder lo esencial") es más amplio que un solo algoritmo.
  - Comprimir una encuesta de 50 preguntas de satisfacción a 3 o 4 "factores" de fondo (ej. "satisfacción con el producto", "satisfacción con la atención"), en vez de mirar las 50 por separado.
  - En un estudio genómico, reducir miles de genes medidos a un puñado de componentes que expliquen la mayor parte de la variabilidad entre pacientes.
  - Simplificar decenas de indicadores financieros de una empresa a 2 o 3 ejes, para poder graficarla y compararla visualmente contra sus competidores.
  - Comprimir las variables de sensores de una máquina industrial (temperatura, vibración, presión, decenas de mediciones) a pocos indicadores que resuman su "estado de salud" general.
- **Detección de anomalías** responde a *"¿qué no encaja acá?"*. No es un algoritmo nuevo con su propio módulo — es una **forma de usar** el clustering (sobre todo DBSCAN, en el Módulo 3): en vez de preguntarse a qué grupo pertenece un punto, se busca a los puntos que no pertenecen bien a ninguno.
  - Detección de fraude bancario: el algoritmo aprende el "comportamiento normal" de cada tarjeta, y marca como sospechosa cualquier transacción que se aleje demasiado de ese patrón.
  - Mantenimiento predictivo industrial: la mayoría de los sensores de un motor muestran lecturas en una zona "normal" (alta densidad); cuando el motor empieza a fallar, sus datos se desplazan a zonas de baja densidad.
  - Ciberseguridad: modelar el tráfico de red "saludable" y marcar como anomalía cualquier patrón de tráfico que se desvíe (un ataque DDoS, una infiltración).
  - Control de calidad industrial: piezas que salen de la línea de producción con medidas que no se parecen a las del lote habitual.

Un ejercicio útil para la clase: para cada uno de los tres tipos, pedirle al grupo un ejemplo propio (no el que ya está en la filmina) de un problema de su día a día que encajaría en esa categoría — ayuda a consolidar la diferencia antes de entrar en el detalle técnico de cada algoritmo.

### Ejemplos de aplicación en la industria *(Filmina 08)*

- **Retail y E-commerce**: segmentación de clientes para campañas personalizadas, detección de fraude en devoluciones.
- **Tecnología y Big Data**: detección de anomalías en redes, agrupamiento de documentos o imágenes.
- **Analítica de negocios**: reducción de variables para simplificar reportes y visualizaciones.

Estos ejemplos muestran cómo el aprendizaje no supervisado ayuda a extraer valor de datos sin necesidad de etiquetas previas, facilitando la toma de decisiones basada en patrones reales — en retail, por ejemplo, saber cómo se agrupan los clientes puede mejorar significativamente las estrategias de marketing y ventas.

**Para ampliar cada rubro con más detalle antes de la filmina:**

- **Retail y E-commerce**: además de la segmentación de clientes, el no supervisado se usa para detectar **fraude de devoluciones** (agrupando patrones de compra-devolución atípicos) y para el **diseño de layout de tiendas físicas** — qué productos ubicar cerca de cuáles, a partir de patrones de compra reales, no de la intuición del gerente.
- **Tecnología y Big Data**: en ciberseguridad, la detección de anomalías en redes es en esencia un problema de clustering "al revés" — en vez de buscar el grupo al que pertenece un punto, se busca a los puntos que **no** encajan bien en ningún grupo (muy cerca del concepto de "ruido" que va a aparecer con DBSCAN en el Módulo 3). En NLP, agrupar documentos por similitud de contenido es la base de los sistemas de recomendación de artículos o noticias.
- **Analítica de negocios**: cuando un dashboard tiene 40 métricas y nadie sabe cuáles mirar primero, reducir dimensionalidad ayuda a identificar qué puñado de "meta-indicadores" resume la mayor parte de la variabilidad del negocio — un uso de PCA orientado a la comunicación con gerencia, no solo al preprocesamiento técnico.

Un cuarto sector que vale la pena mencionar aunque no esté explícito en la filmina: **salud**, donde el clustering se usa para descubrir subtipos de una enfermedad (pacientes que responden de forma distinta a un mismo tratamiento) sin que existiera antes una clasificación clínica formal para esos subgrupos.

### Flujo típico de trabajo *(Filmina 09)*

1. **Recolección y preparación de datos**: limpieza, selección y escalado de variables.
2. **Selección del método adecuado**: según el problema y el tipo de datos.
3. **Aplicación del algoritmo**: ejecución y ajuste de parámetros.
4. **Evaluación y validación**: métricas específicas para medir la calidad de agrupamientos o representaciones.
5. **Interpretación y uso de resultados**: integración en procesos de negocio o análisis posteriores.

Este flujo es la base para las prácticas y análisis de toda la clase — cambia el algoritmo módulo a módulo, no la lógica del proceso.

**Desarrollo de cada paso, para presentar antes de la filmina:**

1. **Recolección y preparación**: acá es donde más se apoya esta clase en la Clase 03/04 (Pandas) — sin datos limpios y bien tipados, ningún algoritmo de esta clase da resultados confiables. El **escalado** merece mención aparte: casi todas las técnicas de hoy (K-Means, Jerárquico, DBSCAN, PCA) miden distancias o varianza, y una variable en una escala mucho mayor que las demás (ingresos en miles vs. edad en años) puede dominar el resultado por completo si no se estandariza antes.
2. **Selección del método**: no existe "el" algoritmo de aprendizaje no supervisado — la elección depende de si se conoce de antemano cuántos grupos se esperan, si los datos tienen ruido, si las relaciones son lineales o no. Este paso es, en buena medida, el contenido de los Módulos 2, 3 y 4 de hoy.
3. **Aplicación del algoritmo**: a diferencia del supervisado, acá casi siempre hay al menos un **hiperparámetro crítico** que hay que decidir antes de correr el modelo (el `k` de K-Means, el `eps` de DBSCAN, el número de componentes de PCA) — y a diferencia también del supervisado, muchas veces no hay una única respuesta "correcta" para ese parámetro.
4. **Evaluación y validación**: sin `y`, no se puede usar Accuracy ni R². Por eso el Módulo 2 introduce el **coeficiente silhouette** y el **método del codo** — las métricas propias de este mundo, que evalúan qué tan bien separados y compactos quedaron los grupos, en vez de comparar contra una respuesta conocida.
5. **Interpretación y uso**: el paso que más distingue a esta rama del Machine Learning. Un modelo supervisado "sabe" si acertó (comparando contra `y`); un modelo no supervisado nunca sabe si el agrupamiento que encontró "tiene sentido" para el negocio — esa interpretación siempre requiere a una persona que conozca el dominio, mirando los grupos resultantes y poniéndoles nombre y sentido.

---

## Módulo 2 — K-Means y la Elección de k

**Contexto**: ¿cómo agrupar datos sin etiquetas? K-Means es el algoritmo más usado de clustering — divide un conjunto de datos en grupos naturales basándose en similitud.

### Apertura del módulo *(Filmina 10)*

La divisoria de este módulo trae el subtítulo "El algoritmo de clustering más usado, y cómo elegir bien su parámetro clave" — y es, en términos de duración, el módulo más largo de la clase (7 filminas), lo cual tiene sentido: K-Means es probablemente el algoritmo de aprendizaje no supervisado más usado en la industria, por su simplicidad conceptual y su bajo costo computacional.

**Para presentar antes del contenido técnico**: conviene retomar acá, en voz alta, la definición general de clustering del Módulo 1 ("agrupar datos similares en clusters") y anticipar que K-Means la resuelve con una idea muy visual: imaginar que cada cluster tiene un "centro de gravedad" (el centroide), y que cada punto del dataset "cae" naturalmente hacia el centro más cercano. Es una buena metáfora para instalar antes de entrar en el detalle algorítmico de la Filmina 11, porque todo el resto del módulo (los 4 pasos, los problemas de convergencia, la elección de k) gira alrededor de esa única idea: minimizar qué tan lejos está, en promedio, cada punto de su centro asignado.

### Qué es y cómo funciona *(Filmina 11)*

K-Means es un **algoritmo de partición**: divide un conjunto de datos en `k` grupos (clusters) según la similitud de sus características. El objetivo es minimizar la suma de las distancias entre cada punto y el **centroide** (promedio) de su cluster asignado. Se apoya en las métricas de distancia (Euclidiana, Manhattan, Coseno) que ya se usaron en clases anteriores para definir "similitud".

**Para ampliar antes de mostrar la filmina**: el nombre completo del algoritmo, "K-Means" (K-Medias), ya describe su mecánica — la "K" es la cantidad de grupos a formar, y "Means" (medias) es literalmente cómo se calcula cada centroide: el promedio de todos los puntos que pertenecen a ese cluster en un momento dado. Formalmente, el algoritmo minimiza una función llamada **inercia** o **WCSS** (que se retoma en la Filmina 14): la suma, sobre todos los puntos, de la distancia al cuadrado entre cada punto y el centroide de su cluster. Elevar al cuadrado la distancia (en vez de usarla directa) tiene una razón matemática concreta: penaliza mucho más fuerte a los puntos lejanos que a los cercanos, lo que empuja al algoritmo a formar grupos compactos en vez de tolerar unos pocos puntos muy alejados de su centro.

Sobre las métricas de distancia: K-Means usa por defecto la distancia **Euclidiana** (la "línea recta" entre dos puntos, el teorema de Pitágoras aplicado a más de dos dimensiones) — es la que mejor encaja con la definición de centroide como promedio aritmético. Usar Manhattan (la suma de diferencias absolutas, como moverse en cuadras de una ciudad) o Coseno (el ángulo entre dos vectores, típico en texto) requeriría, estrictamente, variantes del algoritmo (K-Medoids es la alternativa más conocida cuando se necesita otra métrica de distancia).

### Los 4 pasos del algoritmo *(Filmina 12)*

1. **Inicialización**: se eligen `k` centroides iniciales — al azar o con **k-means++** para mejorar la convergencia.
2. **Asignación**: cada punto se asigna al cluster cuyo centroide esté más cerca (distancia Euclidiana, típicamente).
3. **Actualización**: se recalculan los centroides como el promedio de los puntos asignados a cada cluster.
4. **Repetición**: se repiten Asignación y Actualización hasta que las asignaciones no cambien o se alcance un número máximo de iteraciones.

**Desarrollo paso a paso, para acompañar la animación de la filmina en vivo:**

Este algoritmo también se conoce como **"Lloyd's algorithm"** en la literatura técnica, y es un buen ejemplo de un procedimiento **iterativo**: no calcula la respuesta de una vez, sino que la va refinando en rondas sucesivas, cada una un poco mejor que la anterior. Vale la pena remarcar en clase que los pasos 2 y 3 son, en esencia, un ciclo de "adivinar y corregir": el paso 2 (Asignación) responde "con los centroides que tengo ahora, ¿cuál es la mejor partición posible?"; el paso 3 (Actualización) responde "con esta partición, ¿cuáles son los mejores centroides posibles?". Cada ronda del ciclo garantiza matemáticamente que el WCSS total **nunca aumenta** — por eso el algoritmo siempre termina convergiendo (ver Filmina 13), aunque no siempre al mejor resultado posible.

Sobre la Inicialización: la opción "al azar" simplemente elige `k` puntos cualquiera del dataset como primeros centroides — es simple pero puede arrancar en una posición muy mala. **k-means++** (el default en la implementación de scikit-learn) es más inteligente: elige el primer centroide al azar, y cada centroide siguiente lo elige con una probabilidad proporcional a qué tan lejos está de los centroides ya elegidos — favoreciendo que los `k` puntos de arranque queden bien repartidos por el espacio de datos, en vez de agrupados por casualidad en una sola zona.

### Convergencia, inicialización y problemas comunes *(Filmina 13)*

- K-Means **siempre converge**, pero a un **mínimo local**, no necesariamente al óptimo global.
- La inicialización de los centroides afecta la calidad y velocidad de convergencia; **k-means++** ayuda a elegir centroides iniciales más representativos, reduciendo la probabilidad de resultados pobres.
- **Outliers**: pueden distorsionar los centroides y afectar la agrupación.
- **Formas no esféricas**: K-Means asume clusters convexos y de tamaño similar; no funciona bien con formas arbitrarias.

**Para desarrollar cada punto con más profundidad:**

- **Mínimo local vs. global**: como el resultado final depende de dónde arrancaron los centroides, correr K-Means dos veces con inicializaciones distintas puede dar dos particiones **distintas**, ambas "válidas" en el sentido de que el algoritmo convergió correctamente en las dos, pero una puede ser mejor que la otra. La solución práctica que usa scikit-learn (y que aparece en el ejemplo de código de la Filmina 15, con el parámetro `n_init=10`) es correr el algoritmo completo varias veces con distintas inicializaciones al azar, y quedarse con el resultado que dio el WCSS más bajo de todos los intentos.
- **Sensibilidad a outliers**: como el centroide es un **promedio**, un solo punto muy alejado del resto puede "arrastrar" el centroide entero hacia él, distorsionando la posición de todo el cluster — el mismo fenómeno por el que la media aritmética es sensible a valores extremos (visto en clases anteriores de estadística descriptiva). Es una de las razones por las que suele convenir revisar y tratar outliers **antes** de correr K-Means, no después.
  - *Ejemplo concreto*: segmentando clientes por gasto mensual, un solo cliente corporativo que gasta 100 veces más que el resto puede correr el centroide de "clientes premium" tan lejos que termine agrupando mal a los clientes premium "reales" — conviene revisar outliers (Módulo 1) antes de clusterizar, no después.
- **Formas no esféricas**: como K-Means asigna cada punto según distancia al centroide más cercano, la "frontera" natural entre dos clusters siempre termina siendo una línea recta (o un plano, en más dimensiones) — geométricamente, solo puede separar bien grupos que tengan forma redondeada y tamaño parecido. Con clusters alargados, en forma de luna, o de tamaños muy distintos entre sí, K-Means directamente separa mal — y ese es exactamente el problema que resuelve DBSCAN, que se ve en el Módulo 4.
  - *Ejemplo concreto*: agrupar comercios por ubicación geográfica a lo largo de una costa o de un río da un cluster alargado y curvo — K-Means tiende a "cortarlo" en pedazos artificiales con fronteras rectas, en vez de respetar la forma real alargada de la zona.

### Elegir k: método del codo (Elbow Method) *(Filmina 14)*

Para cada valor de `k` se calcula el **WCSS** (*Within-Cluster Sum of Squares*): la suma de las distancias al cuadrado entre cada punto y el centroide de su cluster. Un WCSS más bajo indica clusters más compactos.

Se grafica WCSS en función de `k` — la curva baja a medida que `k` crece, porque agrupar en más clusters siempre reduce la distancia interna. El objetivo es identificar el punto donde la tasa de disminución se frena notablemente, formando un **"codo"**: a partir de ahí, agregar más clusters no mejora significativamente la calidad de la agrupación. Balancea complejidad del modelo (muchos clusters) contra calidad de la agrupación (pocos clusters, cada uno con sentido) — evitando tanto el subajuste como el sobreajuste.

**Para ampliar antes de mostrar el gráfico**: vale la pena mencionar el caso extremo para que la lógica quede clara — si `k` fuera igual a la cantidad total de puntos del dataset, cada punto sería su propio cluster, y el WCSS daría exactamente `0` (cada punto coincide con su propio centroide). Ese extremo es matemáticamente "perfecto" pero completamente inútil para el negocio: no agrupa nada. El método del codo es, en el fondo, una forma visual de encontrar el compromiso entre ese extremo inútil (`k` = cantidad de puntos, WCSS = 0) y el otro extremo igual de inútil (`k` = 1, todo en un solo grupo, WCSS máximo). Conviene aclarar también que la ubicación del "codo" no siempre es tan clara como en el ejemplo de esta clase — en datasets reales, la curva a veces baja de forma más gradual, sin un quiebre visualmente obvio, y ahí es donde el coeficiente silhouette (Filmina 15) aporta una segunda opinión más cuantitativa.

### Elegir k: coeficiente silhouette *(Filmina 15)*

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

### Aplicación práctica y relevancia en la industria *(Filmina 16)*

- **Retail**: segmentar clientes por frecuencia de compra, monto gastado y preferencia de categorías, para diseñar campañas de marketing personalizadas.
- **Finanzas**: identificar grupos de clientes con perfiles de riesgo similares, mejorando la gestión de cartera y la detección de fraudes.
- **Imágenes**: segmentación de regiones en análisis de imágenes, y sistemas de recomendación personalizados.

La correcta elección de `k` evita tanto la **sobresegmentación** (demasiados clusters que complican la interpretación) como la **subsegmentación** (pocos clusters que ocultan diferencias importantes).

**Para cerrar el módulo con ejemplos más desarrollados:**

- **Retail**: la técnica de segmentación de clientes más citada en la industria es el análisis **RFM** (*Recency, Frequency, Monetary* — hace cuánto compró por última vez, con qué frecuencia compra, cuánto gasta en total), que se calcula con Pandas (`groupby`/`agg`, contenido de la Clase 04) y después se pasa como input a K-Means para formar los segmentos finales — un ejemplo concreto de cómo se conectan las herramientas de las últimas clases entre sí.
- **Finanzas**: además de detección de fraude, K-Means se usa en gestión de portafolios para agrupar activos financieros con comportamientos de precio similares (una forma de diversificación basada en datos, en vez de en la clasificación tradicional por sector o industria).
- **Imágenes**: un uso muy visual para mostrar en clase es la **cuantización de color** — aplicar K-Means sobre los píxeles de una imagen (donde cada píxel es un punto en un espacio de 3 dimensiones: rojo, verde, azul) reduce la imagen a solo `k` colores distintos, el color de cada centroide. Es una forma concreta y visual de "ver" cómo trabaja el algoritmo, sin necesitar imaginar puntos abstractos en un gráfico.

**Más sectores, para tener variedad al elegir el ejemplo según el grupo:**
- **Salud**: agrupar pacientes por perfil de riesgo (edad, comorbilidades, hábitos) para priorizar seguimiento médico en los grupos de mayor riesgo, sin depender de una única regla clínica fija.
- **Educación**: agrupar estudiantes por patrón de desempeño (tiempo dedicado, ejercicios resueltos, tipo de errores) para detectar perfiles que necesitan un refuerzo distinto, en vez de un mismo plan para todos.
- **Logística**: agrupar puntos de entrega por ubicación geográfica para diseñar zonas de reparto eficientes — el mismo problema que resuelve, con matices, cualquier app de delivery.
- **Recursos Humanos**: agrupar empleados por perfil de desempeño y compromiso (encuestas de clima, antigüedad, ausentismo) para detectar patrones de rotación antes de que se conviertan en renuncias.

**Sobre la sobresegmentación y subsegmentación**: en términos de negocio, la sobresegmentación tiene un costo operativo real — si marketing tiene que diseñar 15 campañas distintas para 15 microsegmentos de clientes, el costo de gestionar esa complejidad puede superar el beneficio de la personalización. La subsegmentación, en cambio, tiene un costo de oportunidad: agrupar en pocos clusters muy amplios puede esconder un segmento pequeño pero muy rentable dentro de un grupo más grande y menos interesante. No existe una regla matemática que resuelva esta tensión — el método del codo y el silhouette dan candidatos razonables de `k`, pero la decisión final casi siempre involucra also una restricción práctica del negocio (cuántos segmentos puede gestionar realmente el equipo de marketing, por ejemplo).

---

## Módulo 3 — DBSCAN: Clustering Basado en Densidad

**Contexto**: una alternativa a K-Means, para cuando no querés (o no podés) definir `k` de antemano, o cuando tus datos tienen ruido y formas irregulares.

### Apertura del módulo, después del Break *(Filmina 18)*

Esta divisoria llega justo después del corte de 10 minutos (Filmina 17) — conviene arrancar retomando brevemente dónde había quedado la clase antes del break: K-Means resuelve bien el clustering cuando los grupos son razonablemente esféricos, de tamaño parecido, y se conoce (o se puede estimar) el número `k` de antemano. Este módulo presenta una **alternativa** que relaja esas mismas condiciones.

**Para presentar antes de entrar al contenido**: es útil anticipar la pregunta que motiva al algoritmo: *"¿qué hago cuando no sé cuántos grupos hay, o cuando mis grupos no tienen forma de círculo?"*. **DBSCAN** responde con una idea distinta a K-Means: en vez de definir clusters por cercanía a un centro, los define por **densidad** — dónde hay muchos puntos juntos versus dónde hay pocos. Instalar esta distinción de entrada ayuda a que el resto del módulo se entienda como "una solución a un problema que K-Means no resuelve bien" y no como un algoritmo suelto.

### DBSCAN: clustering basado en densidad *(Filminas 19–20)*

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

Los dos parámetros son las dos preguntas que hay que responder para definir "denso": `eps` responde *"¿qué tan cerca hay que estar para contar como vecino?"*, y `min_samples` responde *"¿cuántos vecinos hacen falta para considerar la zona densa?"*. Ajustar estos dos números cambia radicalmente el resultado: un `eps` muy chico deja casi todo como ruido (porque casi nada tiene suficientes vecinos tan cerca); un `eps` muy grande termina fusionando clusters que deberían quedar separados (porque "casi todo" pasa a ser vecino de "casi todo"). Por eso esta misma filmina trae la técnica del **k-distance plot** (que se ve en el ejemplo de código): una forma sistemática de estimar un buen valor de `eps` a partir de los propios datos, en vez de adivinarlo a prueba y error.

Sobre los tres tipos de punto: la distinción entre **core** y **border** es sutil pero importante — un border point sí forma parte de un cluster (queda "adentro" de la región densa por estar cerca de un core point), pero no tiene la densidad suficiente **por sí mismo** como para ser considerado el corazón de esa densidad. Es la diferencia entre "vivir en un barrio poblado" (border) y "ser, vos mismo, uno de los puntos que hace que el barrio esté poblado" (core). Solo el **noise point** queda completamente afuera de cualquier cluster — y a diferencia de K-Means, donde **todo** punto es forzado a pertenecer a algún cluster (incluso un outlier extremo), en DBSCAN el ruido es un resultado legítimo y esperado, no un error.

**¿Para qué usarías DBSCAN en la práctica, y en qué casos conviene más que los otros dos?**
- **Detección de fraude**: transacciones o comportamientos que no encajan en ningún patrón habitual son exactamente lo que DBSCAN marca como ruido — a diferencia de K-Means, que forzaría esa transacción rara a pertenecer al cluster más cercano aunque no se parezca en nada.
- **Análisis geoespacial**: identificar "zonas calientes" de actividad (pedidos de una app de delivery, denuncias en un mapa de una ciudad, brotes de una enfermedad) sin saber de antemano cuántas zonas hay ni su forma — las zonas reales casi nunca son círculos perfectos.
- **Astronomía**: agrupar estrellas o galaxias por densidad espacial para identificar cúmulos reales, dejando afuera como "ruido" a los objetos aislados que no pertenecen a ningún cúmulo.
- **Redes sociales**: detectar comunidades de usuarios muy conectados entre sí, identificando al mismo tiempo a los usuarios aislados (bots, cuentas inactivas) que no encajan en ninguna comunidad real.

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

### Comparación: K-Means vs. DBSCAN *(Filmina 21)*

| Característica | K-Means | DBSCAN |
|---|---|---|
| **Forma de clusters** | Convexa, esférica | Arbitraria |
| **Número de clusters** | Requiere definir `k` | No requiere definirlo |
| **Manejo de ruido** | No explícito (fuerza a cada punto dentro de un cluster) | Sí, lo detecta y lo marca como `-1` |
| **Parámetros clave** | Número de clusters (`k`) | `eps`, `min_samples` |

La elección del método depende del tipo de datos, la forma esperada de los clusters y la presencia de ruido — si esperás clusters esféricos y sabés (o podés estimar) cuántos hay, K-Means; si sospechás que hay ruido real (outliers, posibles fraudes) y formas irregulares, DBSCAN.

**Guía práctica para cerrar el módulo, útil como resumen para dictar de memoria:**

- Si el dataset es **grande** (cientos de miles de puntos o más): K-Means, el más rápido de los dos.
- Si los datos tienen **ruido real** (sensores con lecturas erróneas, usuarios anómalos, fraude) que no debería forzarse a ningún cluster: DBSCAN, el único de los dos pensado explícitamente para separar señal de ruido.
- Si los clusters esperados tienen **formas irregulares** (no convexas, tamaños muy distintos): DBSCAN; K-Means tiende a fallar en ese escenario.

Un caso de uso muy citado en clase para DBSCAN es el análisis geoespacial: agrupar coordenadas GPS de usuarios o eventos para encontrar "zonas calientes" de actividad (por ejemplo, dónde se concentran los pedidos de una app de delivery en determinado horario) — un escenario donde el número de zonas no se conoce de antemano, y donde puntos aislados (un pedido en una zona rural sin actividad alrededor) deberían quedar como ruido, no forzados dentro de la zona caliente más cercana.

---

## Módulo 4 — PCA: Reducción de Dimensionalidad

**Contexto**: ¿cómo simplificar un dataset con decenas o cientos de variables sin perder lo esencial? El Análisis de Componentes Principales (PCA) es la técnica fundamental para reducir dimensionalidad, facilitando la visualización y el análisis.

### Apertura del módulo *(Filmina 22)*

Esta divisoria anuncia el módulo de PCA: "simplificar datos complejos sin perder lo esencial: el arte de resumir". A diferencia de la versión anterior de esta guía (apoyada en el PDF viejo), el docx actual no pide desarrollar la matemática de covarianza ni eigenvectores/eigenvalores — se queda en la intuición geométrica, igual que los módulos de clustering.

**Para presentar antes del contenido técnico**: conviene arrancar retomando la Filmina 07 (Módulo 1), donde la reducción de dimensionalidad se definió como "simplificar datos complejos con muchas variables a representaciones más manejables". PCA es la técnica de referencia para resolver ese problema, y su lógica se puede resumir en una sola idea, sin fórmulas: encontrar las direcciones **nuevas** (no necesariamente las variables originales) a lo largo de las cuales los datos varían más — porque ahí es donde vive la mayor parte de la información. Es una buena analogía para instalar acá: PCA es como tomar una escultura en tres dimensiones y proyectar su sombra en una pared — si se elige bien el ángulo, esa sombra dice casi todo lo que hace falta saber de la escultura, pero de forma mucho más simple.

**Qué es cada Componente Principal, sin álgebra lineal**: la Primera Componente Principal (PC1) es la dirección donde los datos varían más; la Segunda Componente (PC2) es la segunda dirección con más variación, y es perpendicular a la primera. Por ejemplo, la PC1 podría explicar el 70% de la variación total de un dataset, la PC2 el 20%, y juntas el 90% — dos números nuevos que resumen casi toda la información de las variables originales.

### Varianza explicada y selección de componentes *(Filmina 23)*

Cada Componente Principal captura una porción de la "información total" (varianza) del dataset original. Esto ayuda a decidir cuántos componentes conservar:

- Conservar los primeros componentes que expliquen un porcentaje significativo (entre 70% y 95%) de la varianza acumulada.
- Un gráfico de codo (misma lógica que en K-Means) ayuda a ver dónde agregar más componentes deja de aportar varianza relevante.
- En algunos casos conviene priorizar **menos** componentes para simplificar el modelo, aunque se pierda algo de varianza — es una decisión de compromiso, no una regla fija.

**Para desarrollar antes de la filmina:**

Vale la pena remarcar el paralelismo explícito con el método del codo de K-Means (Módulo 2, Filmina 14): en los dos casos se grafica una curva (WCSS en un caso, varianza explicada acumulada en el otro) en función de un número entero que hay que elegir (`k` clusters, o cantidad de componentes), y en los dos casos se busca el punto donde agregar "una unidad más" deja de aportar una mejora proporcional. Es el mismo patrón de decisión — "¿cuánta complejidad adicional se justifica por la mejora que trae?" — aplicado a dos problemas distintos.

Esto permite, por ejemplo, pasar de 20 variables a solo 3, perdiendo muy poca información pero ganando muchísima claridad y velocidad. Si el objetivo final es solo **visualizar** los datos, casi siempre se usan exactamente 2 o 3 componentes, sin importar qué porcentaje de varianza expliquen — porque el límite ahí no es estadístico, es que un gráfico no puede tener más de 3 ejes.

### Limitaciones de PCA *(Filmina 24)*

- **Linealidad**: PCA solo captura relaciones **lineales** entre variables; con estructuras no lineales complejas, puede no ser suficiente.
- **Escalado**: es sensible a la escala de las variables — por eso es común normalizar o estandarizar los datos antes de aplicarlo (igual que en clustering).
- **Interpretabilidad**: las componentes principales son combinaciones lineales de las variables originales, lo que puede dificultar su interpretación directa frente a un público no técnico.

**Para ampliar cada limitación con más ejemplos:**

- **Linealidad**: el ejemplo clásico para ilustrar esta limitación en clase es un dataset con forma de espiral o de "S" en el espacio — PCA, al buscar solo direcciones **rectas** de máxima varianza, no puede "desenroscar" esa estructura y termina proyectando puntos que estaban lejos en la espiral original muy cerca entre sí en el resultado. Para esos casos existen alternativas no lineales (t-SNE, UMAP, autoencoders) que quedan fuera del temario de hoy, pero vale la pena que quien pregunte sepa que existen.
- **Escalado**: si no se estandariza antes, una variable con valores en millones (como `market_value_eur` del dataset de la Clase 04) tendría una varianza numéricamente gigantesca comparada con una variable en unidades chicas (como `age`) — y como PCA busca **maximizar varianza**, terminaría armando la primera componente casi exclusivamente a partir de esa única variable de escala grande, ignorando de hecho a todas las demás. Es la misma razón por la que el escalado es obligatorio en K-Means y DBSCAN, aplicada acá a un problema distinto (varianza en vez de distancia).
- **Interpretabilidad**: cuando la primera componente principal resulta ser, por ejemplo, `0.6 × ingresos + 0.5 × gasto_mensual - 0.3 × edad + ...`, explicarle a un directorio "qué es" esa componente en términos de negocio no es trivial — a diferencia de una variable original como "edad", que se entiende sin esfuerzo. Por eso, en contextos donde la explicabilidad ante un público no técnico es prioritaria, a veces se prefiere sacrificar algo de la reducción de dimensionalidad y quedarse con un subconjunto de variables originales, más fáciles de comunicar aunque menos eficientes matemáticamente.

### Aplicación práctica y relevancia en la industria *(Filmina 25)*

- **Visualización**: reducir dimensiones a 2 o 3 para graficar y detectar patrones o segmentos de clientes a simple vista.
- **Preprocesamiento**: simplificar datos antes de aplicar clustering o clasificación, mejorando el rendimiento y reduciendo ruido.

Por ejemplo, un analista puede usar PCA para transformar variables de comportamiento de compra en componentes principales que resumen tendencias clave, facilitando la segmentación de clientes — y entender la varianza explicada permite justificar cuántos componentes usar, balanceando precisión y simplicidad.

**Para cerrar el módulo con más contexto de uso:**

- **Visualización**: un flujo de trabajo muy habitual en la práctica es aplicar PCA para reducir un dataset de muchas variables a 2 componentes, graficar esos 2 componentes en un scatter plot, y **después** colorear cada punto según el cluster que le asignó K-Means (Módulo 2) — combinando las dos técnicas de la clase para poder "ver" en un gráfico 2D una segmentación que en realidad vive en un espacio de muchas más dimensiones, imposible de graficar directamente.
- **Preprocesamiento**: además de mejorar rendimiento (como se ve en el Módulo 5, con la demo de PCA + KNN), reducir dimensionalidad antes de clustering también ayuda a esquivar la llamada **"maldición de la dimensionalidad"** — un fenómeno donde, en espacios de muchísimas dimensiones, la noción misma de "distancia" empieza a perder sentido (todos los puntos terminan pareciendo casi igual de lejos unos de otros), lo que degrada la calidad de algoritmos como K-Means o DBSCAN que dependen exactamente de medir distancias.
- Un tercer uso, no mencionado explícitamente en la filmina pero común en la industria: la **compresión de datos** — guardar solo las primeras componentes principales de un dataset (en vez de todas las variables originales) para ahorrar espacio de almacenamiento, aceptando una pérdida controlada de información a cambio.

**Más sectores donde PCA es la técnica de referencia:**
- **Reconocimiento facial y de imágenes**: cada píxel de una foto es una variable — una imagen de 100×100 píxeles ya tiene 10.000 variables. PCA (en su variante clásica "Eigenfaces") comprime eso a un puñado de componentes que capturan los rasgos que más varían entre caras distintas.
- **Genómica**: estudios con miles de genes medidos por paciente; PCA reduce esa dimensión gigante a un puñado de componentes que después se usan para agrupar pacientes o buscar asociaciones con una enfermedad.
- **Finanzas cuantitativas**: reducir decenas de acciones o bonos correlacionados entre sí a un puñado de "factores de riesgo" comunes (el mercado en general, el sector, la tasa de interés) — la base de muchos modelos de gestión de portafolios.
- **Encuestas y estudios de mercado**: comprimir decenas de preguntas de una encuesta de satisfacción a 2 o 3 "ejes" interpretables (por ejemplo, "satisfacción con el precio" y "satisfacción con el servicio"), más fáciles de presentar a un directorio que 50 respuestas sueltas.

---

## Módulo 5 — Panorama de Métodos (Síntesis)

**Contexto**: cierre conceptual de la clase — comparar las técnicas vistas, entender sus límites, y ver PCA mejorando el rendimiento de un modelo real, no solo en teoría.

### Apertura del módulo de cierre *(Filmina 26)*

La última divisoria técnica de la clase funciona como el cierre conceptual de las casi dos horas de clase. A esta altura ya se recorrieron tres algoritmos concretos (K-Means, DBSCAN, PCA); este módulo no agrega un cuarto algoritmo, sino que da un paso atrás para mirarlos **a todos juntos**.

**Para presentar antes del contenido**: es un buen momento para pedirle al grupo, antes de mostrar ninguna tabla, que intente recordar de memoria los tres algoritmos vistos y a qué familia pertenece cada uno (clustering: K-Means, DBSCAN; reducción de dimensionalidad: PCA) — es un buen chequeo rápido de qué quedó instalado de la clase antes de pasar al repaso formal de las próximas filminas. También es el momento de anticipar que el módulo cierra con algo distinto a las clases anteriores: una demostración con números reales de que la elección de técnica (PCA en este caso) no es solo una cuestión teórica, sino que **cambia el resultado de un modelo posterior** de forma medible.

### Decisiones de diseño y parámetros clave *(Filmina 27)*

| Técnica | Parámetros clave | Consideración principal |
|---|---|---|
| **K-Means** | Número de clusters `k` | Elegir `k` adecuado; sensible a valores atípicos |
| **DBSCAN** | `eps`, `min_samples` | Detecta ruido; adecuado para formas arbitrarias |
| **PCA** | Número de componentes a conservar | Balance entre reducción y pérdida de información |

**Para desarrollar esta tabla en clase, columna por columna:**

Vale la pena remarcar un patrón que atraviesa las tres filas: **todas** las técnicas de hoy tienen al menos un hiperparámetro que hay que decidir a mano antes de correr el algoritmo, y en **ninguno** de los tres casos existe una fórmula única que lo calcule automáticamente — solo heurísticas (el codo, el silhouette, el k-distance plot, el umbral de varianza explicada) que ayudan a acercarse a un buen valor. Es una diferencia de fondo respecto al aprendizaje supervisado de la Clase 08, donde muchos hiperparámetros se pueden ajustar de forma más sistemática con `GridSearchCV` comparando contra una métrica objetiva como Accuracy — acá, al no existir una `y` contra la cual medir "qué tan bien salió", la elección de parámetros conserva siempre un componente de criterio humano.

También vale la pena conectar la columna "Consideración principal" con lo ya visto: la sensibilidad de K-Means a valores atípicos (Filmina 13), la capacidad de DBSCAN de manejar formas arbitrarias (Filminas 19-20), y el balance de PCA entre reducción y pérdida de información (Filmina 23) — esta tabla es, en esencia, un resumen de una idea clave por módulo, y sirve como buena guía de repaso rápido antes de un examen o de aplicar estas técnicas en un proyecto real.

### Limitaciones y supuestos básicos *(Filmina 28)*

- El **clustering** asume que la similitud/diferencia entre puntos es significativa y que los datos pueden agruparse con claridad.
- **PCA** asume relaciones lineales y que la varianza es una medida adecuada de "información".
- Ningún algoritmo "sabe" si `k` (o los grupos encontrados) tiene sentido real — un K-Means forzado a 3 grupos en ruido aleatorio los va a encontrar igual, aunque no signifiquen nada.

**Para reflexionar en clase**: ¿qué pasaría si aplicás K-Means a datos con clusters de formas muy irregulares? ¿O PCA a datos con relaciones fuertemente no lineales? (Spoiler: en ambos casos, conviene DBSCAN o técnicas no lineales en vez de forzar el método "de siempre".)

**Para ampliar cada supuesto antes de la filmina:**

El hilo conductor de esta filmina es que **ninguna técnica de hoy funciona "a ciegas"** — cada una parte de un supuesto sobre cómo son los datos, y cuando ese supuesto no se cumple, el resultado puede ser engañoso sin que el algoritmo avise del error. El clustering, por ejemplo, siempre va a devolver **algún** agrupamiento, incluso si se le pasan datos generados completamente al azar sin ninguna estructura real — el algoritmo no tiene forma de "darse cuenta" de que no había nada que agrupar, y es responsabilidad de quien lo usa evaluar (con silhouette, por ejemplo) si el resultado tiene sentido real o es ruido estadístico disfrazado de grupos.

Sobre PCA: además de asumir linealidad, asume que **más varianza significa más información relevante** — un supuesto razonable en la mayoría de los casos, pero que puede fallar si, por ejemplo, una variable tiene mucha varianza justamente por errores de medición (ruido de sensor) y no por señal real; en ese escenario, PCA podría terminar priorizando una dirección que en realidad es puro ruido.

### Aplicaciones prácticas por escenario *(Filmina 29)*

- **Clustering**: segmentación de clientes, detección de fraude agrupando comportamientos atípicos, análisis de patrones en sensores industriales.
- **PCA**: visualización de datos complejos, reducción de ruido antes de un modelo supervisado, compresión de datos para almacenamiento eficiente.
- **Detección de anomalías**: fraude bancario, fallos de motores industriales — el algoritmo aprende el "comportamiento normal" y marca lo que no encaja.

En la práctica, la elección depende del contexto de negocio: en un e-commerce con datos ruidosos y clusters de forma compleja, DBSCAN suele ganarle a K-Means.

**Para cerrar con un caso integrador, combinando varias técnicas de la clase:**

Un flujo de trabajo realista en una empresa de e-commerce podría combinar **dos técnicas en una sola cadena de análisis**: primero, PCA para reducir docenas de variables de comportamiento de cada cliente (frecuencia de compra, categorías preferidas, monto gastado, dispositivo usado, horario de navegación...) a un puñado de componentes principales que resuman lo esencial; segundo, K-Means o DBSCAN sobre esas componentes reducidas para segmentar a los clientes en grupos con comportamientos similares (más rápido y con mejores resultados que clusterizar sobre las variables originales sin reducir, por la maldición de la dimensionalidad mencionada en el Módulo 4). Es un buen ejemplo para cerrar la clase mostrando que estas técnicas no compiten entre sí — se combinan.

### Demostración: PCA mejorando un modelo real *(Filmina 30)*

El PDF cierra con un ejemplo didáctico controlado que demuestra, con números, que PCA puede **mejorar** el rendimiento de un modelo — no solo "comprimir" datos:

- Dataset de **cáncer de mama** (scikit-learn, 30 *features* reales).
- Se agregan **300 columnas de ruido** (features irrelevantes) a propósito, simulando un escenario de alta dimensionalidad con mucha señal desperdiciada.
- Clasificador **KNN**, elegido por ser sensible al *curse of dimensionality* (empeora notablemente con muchas features irrelevantes).
- Se compara accuracy **sin PCA** vs. **con PCA** (reducción fuerte: 330 → 30 componentes).

**Para presentar el diseño del experimento antes de correr el código:**

Vale la pena explicar por qué el experimento está armado exactamente así, porque el diseño es parte de lo que hace convincente la demostración. El dataset de cáncer de mama (`load_breast_cancer`) ya viene con 30 variables reales y significativas (medidas de núcleos celulares). Agregarle 300 columnas de **ruido gaussiano puro** — números aleatorios sin ninguna relación con si el tumor es maligno o benigno — simula, de forma controlada y medible, algo que pasa todo el tiempo en datasets reales: una tabla con muchas columnas donde solo una fracción de ellas realmente importa para el problema, y el resto es "ruido" (variables mal elegidas, redundantes, o simplemente irrelevantes para la pregunta puntual que se está resolviendo).

La elección de **KNN** como clasificador no es casual: KNN clasifica un punto nuevo mirando literalmente qué tan cerca está de sus vecinos ya clasificados — y esa noción de "cerca" se calcula con distancia sobre **todas** las columnas por igual, ruido incluido. Con 300 columnas de ruido contra solo 30 de señal real, la distancia entre dos puntos queda dominada casi por completo por coincidencias aleatorias en las columnas de ruido, y el vecino "más cercano" deja de ser realmente el más parecido en términos clínicos. Es la manifestación concreta de la maldición de la dimensionalidad mencionada en el Módulo 4. Un modelo como Random Forest, en cambio, sería mucho menos sensible a este mismo experimento — porque puede aprender a ignorar variables irrelevantes; la elección de KNN está pensada a propósito para que el efecto de PCA se note con claridad.

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

> **Con esto cierra la parte técnica de la clase.** El panorama completo: tres técnicas (K-Means, DBSCAN, PCA), cada una con su caso de uso, sus parámetros y sus límites — y una prueba concreta de que elegir bien la técnica de preprocesamiento (PCA) puede ser la diferencia entre un modelo mediocre y uno bueno, incluso antes de tocar el algoritmo de predicción en sí. Lo que sigue (Módulos 6 y 7) ya no es sobre algoritmos nuevos, sino sobre cómo traducir estos resultados en decisiones de negocio responsables.

---

## Módulo 6 — Customer Profiling

**Contexto**: a un gerente de marketing no le interesa el valor de la Inercia o del Epsilon por sí solos — le interesa "¿quiénes son estas personas y qué hacemos con ellas?". Este módulo es el puente entre el resultado técnico de K-Means/DBSCAN y una decisión de negocio real.

### Perfil vs. Comportamiento *(Filmina 31)*

Para diferenciar clústeres con sentido de negocio, conviene separar dos familias de variables:

- **Variables de perfil** (quién es): edad, ciudad de residencia — datos socio-demográficos, relativamente estáticos.
- **Variables de comportamiento** (qué hace): frecuencia de compra, categorías de productos visitadas — acciones que cambian con el tiempo.

La segmentación efectiva casi siempre combina ambas familias — saber "quién es" sin saber "qué hace" (o viceversa) deja la mitad de la foto incompleta.

**Qué caracteriza a un buen segmento**: alta **cohesión** interna (los puntos del grupo se parecen entre sí) y alta **separación** respecto a los demás grupos — el mismo principio de calidad que ya apareció con el coeficiente silhouette (Módulo 2), ahora aplicado a la lectura de negocio, no solo al número.

### De clúster a decisión: un ejemplo completo *(Filmina 32)*

Un K-Means identifica un clúster con **alto gasto histórico** pero **sin compras en los últimos 6 meses**. El algoritmo no sabe qué significa eso — esa interpretación es 100% trabajo humano.

- **Interpretación de negocio**: "Clientes en Riesgo" — tuvieron valor real en el pasado, y el patrón sugiere que se están por ir.
- **Acción**: diseñar una campaña de reactivación con descuentos especiales dirigida específicamente a ese grupo.
- **Lo que NO hay que hacer**: ignorar el grupo asumiendo que "ya se fueron" (perder una oportunidad de negocio detectada), ni eliminar esos datos pensando que son un error (K-Means no garantiza que el comportamiento sea permanente — es una "foto" del estado actual).

**Por qué la traducción importa tanto como el algoritmo**: un centroide es un promedio matemático; decir "el clúster 2 tiene gasto promedio de $500.000 mientras los demás promedian $50.000" es un dato. Decir "el clúster 2 es nuestro segmento Premium, y necesita un trato distinto" es la traducción a negocio que un algoritmo nunca va a hacer solo.

---

## Módulo 7 — Ética, Sesgos y Cierre

**Contexto**: el cierre de la clase, y el más importante en términos de responsabilidad profesional. Sin `y`, no hay una "verdad" contra la cual comparar — por eso toda la responsabilidad de interpretar bien recae en la persona, no en el algoritmo.

### Interpretación responsable: riesgos y sesgos *(Filmina 33)*

- **No hay Ground Truth**: el algoritmo encontrará patrones porque esa es su función — no valida si son reales, útiles o si esconden sesgos peligrosos. Que un K-Means encuentre 3 grupos no prueba que "existan" 3 tipos reales de clientes: si se le pide 10, va a dar 10.
- **Proyectar prejuicios propios**: al no haber etiquetas, es muy fácil interpretar un clúster con el propio sesgo en vez de con el dato real detrás.
- **El riesgo legal y ético, no solo técnico**: si un clúster separa personas por un patrón que refleja una desigualdad social (por ejemplo, una zona geográfica correlacionada con nivel socioeconómico) y ese resultado se usa ciegamente para decidir a quién otorgar un crédito, hay un problema serio — el modelo no es "racista" ni "injusto" por sí mismo, simplemente es un espejo de los datos con los que se construyó, pero usarlo sin ese criterio tiene consecuencias reales.

**Para desarrollar en clase, antes de la Pre-entrega:**

El aprendizaje no supervisado da el "qué" (los grupos, los componentes, las anomalías) — el criterio humano pone el "por qué" y el "para qué". Un buen ejercicio de cierre es preguntarle al grupo: de todo lo visto hoy (K-Means, DBSCAN, PCA, Customer Profiling), ¿en qué paso puntual se cuela más fácilmente un sesgo sin que nadie lo note? La respuesta esperada apunta casi siempre al mismo lugar: el momento de ponerle **nombre** a un clúster — ahí es donde la interpretación humana reemplaza al dato, y donde conviene pedir una segunda opinión antes de tomar una decisión que afecte personas reales.

---

## Pre-entrega: Aprendizaje No Supervisado

✅ **Entregable evaluado del módulo.**

**Escenario**: una aplicación de streaming de música (similar a Spotify) con 500.000 usuarios, sin etiquetas — no se sabe de antemano quién es "premium" ni qué "estilo de oyente" tiene cada uno. Los datos disponibles incluyen géneros más escuchados, horas de escucha al día, número de listas de reproducción creadas, edad y ubicación.

**Lo que hay que entregar, en un documento de análisis (PDF)**:

1. **Estrategia de clustering**: ¿K-Means o DBSCAN para segmentar a estos usuarios? Justificar comparando cómo cada uno maneja el ruido y las formas de los grupos.
2. **Reducción de dimensionalidad**: si hay 100 variables por usuario, ¿cómo se usaría PCA antes de clusterizar, y qué beneficio trae en términos de visualización y costo computacional?
3. **Interpretación de negocio**: una vez que el algoritmo devuelve 5 grupos de usuarios, ¿qué pasos seguirías para "ponerles nombre" y asegurar que esos grupos son útiles para el equipo de Marketing?
4. **Ética y sesgos**: mencionar un posible sesgo que podría ocurrir al agrupar usuarios sin supervisión humana, y cómo se intentaría mitigarlo.

**No se requiere código ejecutable** — sí una propuesta técnica y analítica bien fundamentada, con la terminología correcta del módulo (cohesión, separación, ruido, varianza explicada, Ground Truth).

**Criterios de evaluación**: claridad técnica en la distinción de algoritmos; razonamiento lógico sobre el uso de PCA; enfoque orientado a resultados de negocio; uso correcto de la terminología del módulo.

**Nota sobre el Podcast**: `Clase 09_fixed.docx` incluye, al cierre del módulo, un Podcast transcripto (diálogo entre dos presentadores repasando todo el recorrido: K-Means, DBSCAN, PCA, Customer Profiling y ética). Siguiendo el mismo criterio que en Clase 07, ese contenido de audio **no** se convierte en filmina — queda como material de repaso sugerido para los alumnos antes de encarar la Pre-entrega, sin sección propia en `Clase09.html`.

---

## Anexo — Apunte del Notebook Práctico (`Clase_9.ipynb`)

Esta sección documenta un notebook **aparte**, ya armado y con código funcionando (`Clase_9.ipynb`, en la raíz de la carpeta), que resuelve las cinco técnicas de la clase con un **dataset real de fútbol** en vez de datos sintéticos — 48 selecciones de un torneo, con estadísticas de Ataque, Distribución, Defensa, Portería, Movimiento y Físico. Es un apunte de referencia por si decidís dar la clase directamente desde ese notebook en vez de (o además de) las filminas.

✅ **Los dos problemas que tenía el notebook ya están corregidos**: el nombre del archivo Excel (`'Data-Set-Fifa.xlsx'`, con guiones) y el error de sintaxis en DBSCAN (`DBSCAN(eps=0.5, min_samples=3)`, antes tenía `+=3`, que no es Python válido). El código de abajo ya refleja ambas correcciones.

### Sobre el dataset: `Data-Set-Fifa.xlsx`

Es una planilla de estadísticas de un torneo de fútbol (48 selecciones), organizada en **6 hojas**, una por familia de métricas — cada hoja tiene una fila por equipo:

| Hoja | Qué mide | Algunas columnas |
|---|---|---|
| **Ataque** | Producción ofensiva | `Goles`, `Asistencias`, `Remates`, `Efectividad en los remates %`, `Posesión del balón %` |
| **Distribución** | Circulación de pelota | `Pase`, `Precisión en los pases %`, `Centro`, `Cambios de orientación intentados` |
| **Defensa** | Solidez defensiva | `Goles recibidos`, `Pérdidas de balón provocadas`, `Presiones ofensivas/defensivas` |
| **Portería** | En rigor, disciplina (ver nota) | `Faltas recibidas/cometidas`, `Tarjetas Amarillas/Rojas`, `Fueras de juego` |
| **Movimiento** | Desmarques y recepciones | `Desmarques para recibir`, `Recepciones bajo presión` |
| **Físico** | Rendimiento físico | `Velocidad Media (Km/h)`, `Esprints`, `Distancia recorrida (m)` |

**Un detalle real para comentar en clase**: la hoja se llama "Portería" pero sus columnas son de **disciplina** (faltas, tarjetas), no de arqueros — un desajuste entre el nombre de la hoja y lo que realmente contiene. Es un buen ejemplo real de por qué nunca hay que confiar en el nombre de una hoja o columna sin abrir los datos y confirmar qué hay adentro (la misma idea que `.info()` y `.head()` en Pandas, Clase 03).

Cada fila es un **equipo del torneo** (no un jugador ni un partido) — a diferencia del dataset de la Clase 04 (FIFA World Cup, jugador-partido), acá el nivel de análisis es "selección completa", lo que lo hace ideal para comparar estilos de juego entre países.

**¿Para qué se puede usar este dataset, más allá de lo que ya hace el notebook?** El notebook actual solo usa un puñado de columnas a la vez (2 para K-Means, 2 para el dendrograma/DBSCAN, 4 para PCA, 4 para Apriori) — pero con 6 hojas completas hay mucho más para explorar:

**Ejemplos de no supervisado (lo que se ve en esta clase) que todavía no están en el notebook:**
- **Clustering con todas las variables a la vez** (no de a 2): correr K-Means o Jerárquico sobre las ~40 columnas numéricas combinadas (previa reducción con PCA, para evitar la maldición de la dimensionalidad del Módulo 5) — daría un "estilo de juego integral" en vez de un perfil parcial por bloque.
- **PCA sobre "Distribución"**: reducir `Pase`, `Centro`, `Rupturas de líneas`, `Cambios de orientación` a 2 ejes que resuman el estilo de construcción de juego de cada selección (¿juego directo o de posesión?).
- **Reglas de asociación sobre "Defensa"**: qué comportamientos defensivos (`Presiones altas`, `Pérdidas provocadas`, `Recuperación rápida`) tienden a darse juntos — el mismo análisis que se hizo con Ataque, aplicado a la otra mitad de la cancha.
- **DBSCAN sobre el dataset completo**: después de un PCA a 2-3 componentes sobre todas las hojas combinadas, buscar equipos "atípicos" en un sentido más amplio que solo lo defensivo (bloque 4 del notebook).

**Ejemplos de supervisado (lo que se vio en la Clase 08) que se podrían construir con este mismo archivo:**
- **Clasificación**: predecir si un equipo llega a cuartos de final o más, usando como `X` sus métricas de Ataque/Defensa/Físico y como `y` una etiqueta "avanzó / no avanzó" (habría que conseguir ese dato de resultados, que no está en este Excel).
- **Regresión**: predecir la cantidad de goles que un equipo va a convertir en el torneo (`y` numérico) a partir de sus métricas de creación de juego (`Remates`, `Asistencias`, `Posesión`) como `X` — un caso de uso análogo al de "precio de una casa" de la Clase 08, pero en fútbol.
- **Árbol de Decisión o Random Forest**: combinando variables de las 6 hojas para predecir la posición final en la tabla, y de paso ver con `feature_importances_` qué familia de métricas (ataque, defensa, físico) pesa más en el resultado.

La diferencia clave entre estos dos grupos de ejemplos: los de no supervisado se pueden hacer **hoy mismo**, con el archivo tal cual está — los de supervisado necesitarían agregarle una columna con el resultado real de cada equipo en el torneo (`y`), que hoy no está en el dataset.

### Preparación de los datos (celda de inicio)

**🧭 Por qué absolutamente todo proyecto de Machine Learning arranca así**: no importa si el modelo final es supervisado o no supervisado, ni si es un árbol de decisión o K-Means — **ningún algoritmo puede compensar datos mal cargados**. Si una columna numérica quedó como texto, si un mismo equipo aparece con dos nombres distintos por un typo, o si faltan valores sin que nadie lo note, el algoritmo no "se da cuenta" del error: simplemente calcula sobre datos incorrectos y devuelve un resultado que **parece** válido pero no lo es. Por eso la limpieza siempre es el primer paso del flujo de trabajo (Módulo 1, Filmina 09: *Recolección y preparación → Selección del método → Aplicación → Evaluación → Interpretación*) — es la base de la que dependen los otros cuatro.

En términos generales, cualquier proceso de limpieza (no solo este notebook) sigue la misma secuencia lógica, la misma que ya se practicó con Pandas en las Clases 03 y 04:
1. **Detectar el problema**: ¿hay nulos? ¿tipos de dato incorrectos? ¿nombres duplicados con distinta escritura? ¿encoding roto?
2. **Decidir una estrategia**: ¿se corrige, se elimina, se imputa? (acá: corregir nombres, imputar con la media los huecos de cruce entre hojas)
3. **Aplicar la corrección** de forma sistemática — nunca a mano, fila por fila, porque no escala y no es reproducible.
4. **Verificar el resultado** con un chequeo concreto — acá, que el conteo final dé exactamente 48 equipos, ni uno más ni uno menos.

Este ejemplo puntual es un caso más desprolijo que el promedio (celdas combinadas, encoding roto, columnas basura) precisamente porque así viene un archivo Excel armado a mano por una persona, sin pensar en que después lo iba a leer un programa — el escenario más realista posible, mucho más parecido a lo que se encuentra en un trabajo real que un dataset ya limpio bajado de Kaggle.

🎯 **Para qué usamos este código**: no es un análisis en sí — es el paso obligatorio de "ingesta y saneamiento" (Módulo 1 de esta guía, aplicado ahora a un archivo real y desprolijo) que hay que correr **una sola vez, al principio**, para que las 5 técnicas de los bloques siguientes tengan un solo DataFrame limpio (`df_final`) del cual partir. Lo que queremos ver al final es la confirmación `"¡Exactamente 48!"` — si ese número no cierra, algo en el cruce de las 6 hojas salió mal y no tiene sentido seguir a los bloques de abajo.

El Excel viene con una particularidad: cada equipo ocupa **dos filas** (una con los datos numéricos, la fila siguiente con el nombre real del equipo) — rastro de celdas combinadas en el archivo original.

```python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)   # silencia warnings de la infraestructura de Jupyter, no de este código

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

archivo_excel = 'Data-Set-Fifa.xlsx'
xls = pd.ExcelFile(archivo_excel)

def limpiar_nombre_equipo(nombre):
    if pd.isna(nombre): return nombre
    s = str(nombre).strip()
    s = s.replace('Espa帽a', 'España').replace('EspaÃ±a', 'España')
    # ...más reemplazos de encoding roto, uno por cada país afectado
    return s

def procesar_hoja_con_glosario(xls_file, nombre_hoja):
    df_raw = pd.read_excel(xls_file, nombre_hoja)
    indices_datos = df_raw[df_raw['Puesto'].notna()].index

    registros = []
    for idx in indices_datos:
        datos_fila = df_raw.iloc[idx].copy()
        nombre_real = df_raw.iloc[idx + 1]['Equipo']
        datos_fila['Equipo'] = limpiar_nombre_equipo(nombre_real)
        registros.append(datos_fila)

    df_limpio = pd.DataFrame(registros).reset_index(drop=True)
    cols_validas = [c for c in df_limpio.columns if 'Unnamed' not in str(c) and 'glosario' not in str(c).lower() and c != 'Puesto']
    return df_limpio[cols_validas]

# 1. Lista maestra basada estrictamente en la primera hoja (Ataque)
df_maestro = procesar_hoja_con_glosario(xls, 'Ataque')
lista_48_equipos = df_maestro['Equipo'].dropna().unique()

# 2. DataFrame final arranca con la estructura maestra
df_final = df_maestro.copy()

# 3. Cruzamos el resto de las hojas de forma relacional permisiva
for hoja in xls.sheet_names[1:]:
    df_hoja_limpia = procesar_hoja_con_glosario(xls, hoja)
    df_final = pd.merge(df_final, df_hoja_limpia, on='Equipo', how='outer')

# 4. Índice provisorio
df_final = df_final.dropna(subset=['Equipo']).set_index('Equipo')

# 5. Recorte a los 48 equipos oficiales + imputación de huecos
df_final = df_final.reindex(lista_48_equipos)
df_final = df_final.fillna(df_final.mean(numeric_only=True))

# 6. Corregimos el nombre mal escrito que trae el Excel original
df_final = df_final.rename(columns={'Posecion del balon %': 'Posesión del balón %'})

scaler = StandardScaler()
```

**Línea por línea, qué hace y por qué:**
- `warnings.filterwarnings('ignore', category=DeprecationWarning)` → puesto **al principio de todo**, antes que cualquier otra cosa. Suprime, para el resto de la ejecución del notebook, una tanda larga de `DeprecationWarning` que tira `jupyter_client` (la infraestructura de mensajería de Jupyter, no código de este notebook) sobre `datetime.utcnow()` — inofensivos, pero ensucian mucho la salida si no se filtran desde el arranque. Antes esta línea estaba recién en el Bloque 2, así que no alcanzaba a cubrir la celda de inicio ni el resto de bloques anteriores a ese.
- `pd.ExcelFile(archivo_excel)` → abre el Excel una sola vez y permite leer sus 6 hojas (Ataque, Distribución, Defensa, Portería, Movimiento, Físico) sin reabrir el archivo en cada lectura — más eficiente que `pd.read_excel()` suelto por cada hoja.
- `limpiar_nombre_equipo` → el Excel original tiene nombres de país con **encoding roto** (`Espa帽a` en vez de `España`) — típico de un archivo guardado con una codificación de caracteres distinta a la que se usa para leerlo. La función hace un `.replace()` manual por cada caso conocido, uno por uno, porque no hay una forma automática de "adivinar" qué encoding se usó originalmente una vez que el texto ya se rompió.
- `df_raw[df_raw['Puesto'].notna()].index` → el truco central de todo el bloque: en el Excel, la fila con los **datos numéricos** de un equipo tiene algo en la columna `Puesto`, pero el **nombre del equipo** está vacío ahí y aparece recién en la fila siguiente (por las celdas combinadas). Esta línea encuentra los índices de las filas "con datos", para después ir a buscar el nombre a la fila de al lado.
- `df_raw.iloc[idx + 1]['Equipo']` → acá está la clave: agarra el nombre del equipo de la fila **siguiente** (`idx + 1`) a la de los datos — es la corrección concreta del problema de celdas combinadas.
- `cols_validas = [...]` → descarta tres tipos de columnas basura que trae el Excel original: las que Pandas nombró automáticamente `Unnamed: N` (columnas vacías sin encabezado), la columna `glosario` (texto explicativo pegado en la misma hoja, no es un dato) y `Puesto` (ya cumplió su función de "marcador de fila con datos", no aporta nada al análisis).
- `lista_48_equipos = df_maestro['Equipo'].dropna().unique()` → la hoja "Ataque" se toma como la **hoja de referencia**: los 48 equipos que aparecen ahí son "la verdad" sobre cuáles son los 48 equipos del torneo, para usar como base al cruzar el resto de las hojas.
- El `for hoja in xls.sheet_names[1:]` con `merge(..., how="outer")` → cruza cada una de las otras 5 hojas contra el DataFrame acumulado, usando `Equipo` como clave. `how="outer"` es "permisivo": conserva equipos aunque no crucen perfectamente en alguna hoja (por ejemplo, si un nombre quedó escrito distinto en una hoja puntual), en vez de perderlos silenciosamente con un `how="inner"`.
- `df_final.reindex(lista_48_equipos)` → fuerza al DataFrame final a tener **exactamente** esos 48 equipos, ni uno más ni uno menos, ordenados según la lista maestra — corrige cualquier duplicado o "sobrante" que se haya colado en los merges.
- `df_final.fillna(df_final.mean(numeric_only=True))` → si algún equipo quedó con un hueco puntual en alguna columna (por un cruce imperfecto entre hojas), lo rellena con el promedio de esa columna — la misma técnica de imputación por media que se vio en el Módulo 1 de esta clase, aplicada acá para no perder ningún equipo por un problema menor de cruce.
- `df_final.rename(columns={'Posecion del balon %': 'Posesión del balón %'})` → el Excel original trae ese nombre de columna mal escrito (sin la "s" de "Posesión" y sin el acento de "balón") — se corrige acá, **una sola vez**, para que el resto del notebook (Bloques 2 y 3) ya trabaje con el nombre correcto en vez de arrastrar el error en cada referencia.

### Bloque 1 — Intro al Aprendizaje No Supervisado (sin código, solo teoría)

Mismo concepto que el Módulo 1 de esta guía, con la analogía puntual del notebook: *"No sabemos quién ganó el torneo, ni qué táctica es la correcta; queremos que los datos nos digan de forma natural cómo se agrupan o se comportan los equipos de fútbol por sí solos."*

### Bloque 2 — Reglas de Asociación (Apriori con `mlxtend`)

**🧭 Por qué este es el segundo paso, y no el primero**: con los datos ya limpios (bloque anterior), este es el primer bloque que corresponde a la fase "Selección del método" y "Aplicación del algoritmo" del flujo general (Módulo 1, Filmina 09). Un patrón que se repite en **cualquier** proyecto de reglas de asociación, no solo en este: los datos casi nunca vienen ya en formato de "transacciones" — hay que **transformarlos** primero (acá, convertir 4 métricas numéricas continuas en categorías Alto/Bajo), porque Apriori no entiende números continuos, entiende presencia/ausencia de un ítem. Ese paso de "traducir tus datos al formato que pide el algoritmo" es previo a cualquier algoritmo de esta clase, y cambia según la técnica: acá son categorías binarias, en K-Means van a ser variables numéricas escaladas, en PCA también.

🎯 **Qué queremos ver y para qué sirve**: la pregunta de negocio es *"¿qué métricas ofensivas suelen destacarse juntas en un mismo equipo?"* — este código la responde de forma automática, cruzando `Goles`, `Asistencias`, `Remates` y `Posesión del balón %` sin tener que compararlas manualmente de a pares. Importante: acá **no** aparecen "estilos" distintos que se asocian entre sí (como si un estilo A implicara un estilo B) — lo que el resultado real muestra es que estas 4 métricas ofensivas tienden a aparecer **todas juntas, como un solo paquete**, en los mismos equipos. Lo que buscamos al final no es la tabla completa de reglas (pueden salir decenas), sino **las 2-3 reglas con mayor lift**: esas son las que valen la pena comentar en clase, porque muestran una asociación real y no una coincidencia estadística (Módulo 2 de esta guía).

```python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Transacciones booleanas (True/False), para evitar el Warning
features_rules = ['Goles', 'Asistencias', 'Remates', 'Posesión del balón %']
df_binario = df_final[features_rules].apply(lambda x: x > x.median()).astype(bool)

# 2. Apriori
frequent_itemsets = apriori(df_binario, min_support=0.3, use_colnames=True)

# 3. Reglas de asociación
reglas = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Top 3 ordenado por Lift
print(reglas[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False).head(3))
```

**Línea por línea:**
- `warnings.filterwarnings('ignore', category=DeprecationWarning)` → silencia un aviso conocido de la librería `mlxtend` sobre un cambio de tipo de dato pendiente en una versión futura — no afecta el resultado, solo evita que se imprima una advertencia irrelevante en medio de la clase.
- `df_final[features_rules].apply(lambda x: x > x.median())` → convierte cada una de las 4 columnas numéricas en una columna de `True`/`False`, comparando cada valor contra la **mediana de esa misma columna**. Esto es necesario porque Apriori (el algoritmo del Módulo 2 de esta guía) trabaja con **transacciones de ítems presentes/ausentes**, no con números continuos — "Alto" (por encima de la mediana) es el equivalente acá a "el ítem está en la transacción".
- `.astype(bool)` → fuerza el tipo de dato a booleano explícito; algunas versiones de `mlxtend` piden este tipo puntual para evitar el warning que se silenció arriba.
- `apriori(df_binario, min_support=0.3, use_colnames=True)` → encuentra todos los conjuntos de columnas "Altas" que aparecen juntas en al menos el 30% de los equipos (`min_support=0.3`); `use_colnames=True` hace que el resultado muestre los nombres reales de las columnas en vez de números de índice.
- `association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)` → a partir de esos conjuntos frecuentes, arma las reglas `A → B` y descarta las que tengan menos de 70% de confidence — el umbral de "qué tan seguido se cumple B, dado que se cumplió A" definido en el Módulo 2.
- `.sort_values(by='lift', ascending=False).head(3)` → de todas las reglas que pasaron el filtro de confidence, se queda con las 3 de mayor lift — la métrica que, como se explicó en el Módulo 2, distingue una asociación real de una coincidencia estadística.
- **Resultado real del notebook**: las 3 reglas con mayor lift combinan siempre `{Remates, Asistencias}` con `{Goles, Posesión del balón %}` — todas con lift entre 2,49 y 2,65, y support 0,3125 (15 de los 48 equipos cumplen la regla completa). Conclusión del notebook: *"en el fútbol moderno el éxito ofensivo es un ecosistema interconectado"* — no se puede aislar la posesión del gol, ni los remates de las asistencias.

### Bloque 3 — K-Means (Posesión vs. Efectividad en los remates)

**🧭 Los pasos generales de cualquier clustering con K-Means, no solo este**: (1) elegir qué variables numéricas describen mejor el fenómeno que se quiere agrupar — acá dos, pero podrían ser veinte; (2) escalarlas siempre, sin excepción; (3) probar varios valores de `k` y elegir uno con un criterio objetivo (el codo, y si hace falta el silhouette del Módulo 3 de la teoría); (4) entrenar el modelo final con ese `k`; (5) el paso que ningún algoritmo hace por vos: **interpretar** cada cluster y ponerle un nombre que tenga sentido para quien va a usar el resultado — acá "Los Contundentes", "Bloque Bajo", "Posesión Inofensiva". Ese último paso es el que separa un ejercicio técnico de un análisis útil para un cuerpo técnico real.

🎯 **Qué queremos ver y para qué sirve**: primero, el **gráfico del codo** — para decidir, con criterio y no a ojo, cuántos perfiles tácticos distintos tiene sentido buscar (acá da `k=3`). Después, con el modelo ya entrenado, lo que realmente importa mostrar en clase es el **perfil promedio de cada cluster** y la lista de equipos que cayó en cada uno — es la forma de convertir "3 grupos numéricos" en "3 estilos de juego con nombre y sentido futbolístico", que es en definitiva lo que un cuerpo técnico o analista se llevaría de este análisis.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Selección y escalado
X_kmeans = df_final[['Posesión del balón %', 'Efectividad en los remates %']].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_kmeans)

# 2. Método del codo
inercias = []
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inercias.append(kmeans.inertia_)

plt.plot(range(1, 8), inercias, marker='o')
plt.show()

# Modelo final con 3 clusters
kmeans_opt = KMeans(n_clusters=3, random_state=42, n_init=10)
X_kmeans['Cluster'] = kmeans_opt.fit_predict(X_scaled)

print(X_kmeans.groupby('Cluster').mean())

# Equipos por cluster
equipos_por_cluster = X_kmeans.groupby('Cluster').apply(lambda df: list(df.index), include_groups=False)
for num_cluster, lista_paises in equipos_por_cluster.items():
    print(f"CLUSTER {num_cluster}: ({len(lista_paises)} equipos)")
    print(", ".join(lista_paises))
```

**Línea por línea:**
- `df_final[[...]].dropna()` → selecciona solo las 2 columnas que interesan para este análisis puntual (`Posesión del balón %` y `Efectividad en los remates %`) y descarta cualquier equipo con hueco en esas dos — K-Means no puede calcular distancias con valores faltantes.
- `StandardScaler().fit_transform(X_kmeans)` → escala las dos columnas a media 0 y desvío 1 — imprescindible porque K-Means usa distancia Euclidiana (Módulo 3), y "posesión" y "efectividad" están en escalas distintas.
- El `for k in range(1, 8)` con `.inertia_` → calcula el WCSS (Módulo 3, Filmina 18) para cada valor de `k` de 1 a 7, guardando cada resultado en la lista `inercias` para después graficar el método del codo.
- `random_state=42` → fija la semilla aleatoria de la inicialización de centroides, para que el resultado sea **reproducible**: correr la celda dos veces da exactamente los mismos clusters, en vez de resultados ligeramente distintos cada vez.
- `n_init=10` → corre el algoritmo completo 10 veces con inicializaciones distintas (Módulo 3, k-means++) y se queda con la mejor — reduce el riesgo de quedar atrapado en un mínimo local malo.
- `kmeans_opt = KMeans(n_clusters=3, ...)` → el modelo final ya con el `k` decidido tras mirar el gráfico del codo.
- `X_kmeans['Cluster'] = kmeans_opt.fit_predict(X_scaled)` → ajusta el modelo **con los datos escalados** (`X_scaled`), pero guarda el resultado en el DataFrame **sin escalar** (`X_kmeans`) — para poder leer los promedios de cada cluster en las unidades originales (porcentajes reales), no en unidades de desvío estándar.
- `X_kmeans.groupby('Cluster').mean()` → el mismo patrón de `groupby` de las clases de Pandas: agrupa por el número de cluster asignado y promedia las columnas originales dentro de cada grupo — así se arma el "perfil promedio" de cada cluster.
- `.groupby('Cluster').apply(lambda df: list(df.index), include_groups=False)` → para cada cluster, arma la lista de nombres de equipo (que viven en el índice del DataFrame, por el `set_index('Equipo')` de la celda de inicio); `include_groups=False` evita un warning de versiones nuevas de Pandas al usar `apply` sobre un `groupby`.
- **Resultado real** (ya resumido más arriba en esta guía): 3 clusters — "Los Contundentes" (efectividad 19,44%, posesión media), "Bloque Bajo" (posesión y efectividad bajas), "Posesión Inofensiva" (posesión alta, efectividad la más baja del torneo).

### Bloque 4 — Clustering Jerárquico y DBSCAN (Goles recibidos vs. Pérdidas de balón provocadas)

**🧭 Por qué este bloque usa dos algoritmos y no solo K-Means**: en cualquier proyecto real, K-Means no siempre es la herramienta correcta — este bloque existe para mostrar en vivo **cuándo conviene cambiar de algoritmo**. La secuencia general (no específica de este notebook) es: si no sabés cuántos grupos hay, o si te interesa ver la estructura completa antes de decidir, recurrís a jerárquico; si sospechás que hay "ruido" real en los datos (casos que no deberían forzarse a ningún grupo), recurrís a DBSCAN. Ninguno de los dos pide `k` de antemano — esa es la diferencia de fondo con el bloque anterior, y el motivo por el que en la práctica conviene tener más de un algoritmo de clustering en la caja de herramientas, no solo el más popular.

**¿Qué es un dendrograma?** (repaso rápido, ya desarrollado en el Módulo 4 de esta guía) Es un diagrama en forma de árbol que muestra **todo el proceso de agrupamiento a la vez**, no un único resultado. Cada "hoja" del árbol (en la punta) es un equipo individual; a medida que subís, las hojas se van fusionando de a pares en ramas más grandes, hasta terminar todas juntas en una sola raíz. La **altura** a la que dos ramas se unen indica qué tan distintas son entre sí: cuanto más abajo se fusionan, más se parecen; cuanto más arriba, más diferentes son. No hace falta elegir un número de clusters de antemano (a diferencia de K-Means) — se elige **después**, mirando el árbol completo y decidiendo a qué altura "cortarlo" con una línea imaginaria: cuantas más ramas cruce esa línea, más clusters resultan.

🎯 **Qué queremos ver y para qué sirve**: acá se usan **dos algoritmos con objetivos distintos sobre las mismas variables defensivas**, a propósito, para que se note la diferencia en vivo. Del dendrograma queremos ver la **altura a la que se separan las ramas principales** (a qué distancia dejan de parecerse los grupos de equipos). De DBSCAN queremos ver algo totalmente distinto: no clusters, sino la **lista de equipos que quedaron como ruido** — los que tienen un comportamiento defensivo tan atípico que no encajan bien en ningún grupo denso.

```python
import scipy.cluster.hierarchy as sch
from sklearn.cluster import DBSCAN

# 1. Dendrograma
X_defensa = scaler.fit_transform(df_final[['Goles recibidos', 'Pérdidas de balon provocadas']].dropna())
dendrograma = sch.dendrogram(sch.linkage(X_defensa, method='ward'))
plt.show()

# 2. DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=3)   # corregido: el original tenía "+=3", un error de sintaxis
clusters_db = dbscan.fit_predict(X_defensa)
print(f"Equipos catalogados como Outliers/Ruido (-1): {np.sum(clusters_db == -1)}")

df_final['DBSCAN_Cluster'] = clusters_db
outliers = df_final[df_final['DBSCAN_Cluster'] == -1]
print(outliers[['Goles recibidos', 'Pérdidas de balon provocadas']])
```

**Línea por línea:**
- `scaler.fit_transform(df_final[[...]].dropna())` → reutiliza el mismo `scaler` creado en la celda de inicio (no crea uno nuevo); escala las 2 variables defensivas antes de medir cualquier distancia o similitud, la misma regla de siempre.
- `sch.linkage(X_defensa, method='ward')` → calcula la estructura completa del árbol de fusiones con el criterio Ward (Módulo 4: minimiza el incremento de varianza en cada fusión); el resultado es la matriz `Z` que describe todo el dendrograma.
- `sch.dendrogram(...)` → dibuja el árbol a partir de esa matriz.
- `DBSCAN(eps=0.5, min_samples=3)` → los dos parámetros del Módulo 4: `eps` es el radio de vecindad, `min_samples` la cantidad mínima de vecinos para considerar una zona "densa". Acá se usaron directamente sin pasar por el `k-distance plot` que se vio en la teoría — una simplificación válida para una demo rápida, aunque en un análisis más riguroso convendría estimar `eps` con esa técnica.
- `dbscan.fit_predict(X_defensa)` → ajusta el modelo y devuelve, para cada equipo, el número de cluster asignado o `-1` si quedó como ruido.
- `np.sum(clusters_db == -1)` → cuenta cuántos equipos quedaron marcados como ruido — el mismo truco de "sumar una máscara booleana" que se usó en Pandas para contar nulos, aplicado acá a un array de NumPy.
- `df_final[df_final['DBSCAN_Cluster'] == -1]` → filtro booleano estándar: se queda solo con las filas de los equipos marcados como outliers, para poder inspeccionar sus valores puntuales.
- **Resultado real**: el dendrograma muestra 3 macro-clusters al cortar a la altura ~5,5; DBSCAN marcó **11 equipos** como outliers — estadísticas defensivas en los extremos del torneo, no necesariamente "peores".

### Bloque 5 — PCA (bloque de variables físicas)

**🧭 Por qué PCA suele ser el último paso, no el primero**: a diferencia de los bloques 2 a 4 (que agrupan o buscan reglas), PCA no agrupa nada — **simplifica** para que otro paso (un gráfico, un clustering, un modelo supervisado) funcione mejor o sea posible de mostrar. Es, en general, una herramienta de **preprocesamiento**, no de análisis final: se aplica cuando el problema real (agrupar, predecir, visualizar) tiene demasiadas variables como para resolverse de forma directa. La secuencia general que se repite en cualquier uso de PCA: identificar el grupo de variables relacionadas que se quiere simplificar → escalarlas → decidir cuántas componentes conservar (mirando la varianza explicada) → aplicar → e **interpretar** qué representa cada componente en términos del problema original (acá, "intensidad de carrera" y "velocidad pura"), no solo mirar los números sueltos.

🎯 **Qué queremos ver y para qué sirve**: no podemos graficar 4 variables físicas a la vez en un plano — PCA las comprime a 2 sin perder casi nada (eso es lo primero que hay que mirar: el % de varianza acumulada, para justificar que la simplificación vale la pena). Con esas 2 componentes ya calculadas, lo que realmente queremos ver es el **mapa interactivo**: dónde cae cada uno de los 48 equipos, para detectar a simple vista quiénes corren mucho volumen, quiénes priorizan la velocidad puntual y quiénes rinden poco en lo físico — una lectura visual que sería imposible con las 4 variables originales por separado.

```python
from sklearn.decomposition import PCA
import plotly.express as px

# Seleccionar bloque Físico
cols_fisico = ['Velocidad Media (Km/h)', 'Esprint a gran velocidad', 'Esprints', 'Distancia recorrida (m)']
X_fisico = scaler.fit_transform(df_final[cols_fisico].dropna())

# PCA
pca = PCA(n_components=2)
componentes = pca.fit_transform(X_fisico)

print(f"Varianza explicada por componente: {pca.explained_variance_ratio_}")
print(f"Varianza explicada acumulada: {np.sum(pca.explained_variance_ratio_):.2%}")

df_pca_plot = pd.DataFrame({
    'PC1_Intensidad': componentes[:, 0],
    'PC2_Velocidad': componentes[:, 1],
    'Equipo': df_final.index
})

fig = px.scatter(df_pca_plot, x='PC1_Intensidad', y='PC2_Velocidad', hover_name='Equipo')
fig.show()
```

**Línea por línea:**
- `cols_fisico = [...]` → las 4 variables físicas que se van a comprimir: velocidad media, esprint a gran velocidad, cantidad de esprints, distancia recorrida.
- `scaler.fit_transform(...)` → escalado obligatorio antes de PCA (Módulo 5): sin esto, `Distancia recorrida (m)` (números grandes) dominaría por completo la varianza frente a `Velocidad Media (Km/h)` (números chicos).
- `PCA(n_components=2)` → pide quedarse con las 2 primeras componentes principales — la reducción de 4 variables a 2, elegida acá para poder graficar en un plano 2D.
- `pca.fit_transform(X_fisico)` → calcula las componentes principales (Módulo 5: los eigenvectores de la matriz de covarianza) y proyecta cada equipo sobre esas 2 nuevas direcciones; el resultado `componentes` es una matriz de 48 filas × 2 columnas.
- `pca.explained_variance_ratio_` → el atributo de scikit-learn que ya trae calculado qué porcentaje de la varianza total explica cada componente — no hace falta calcularlo a mano con eigenvalores, como sí se hizo en el ejemplo teórico del Módulo 5.
- `componentes[:, 0]` y `componentes[:, 1]` → las columnas 0 y 1 de la matriz de componentes — PC1 y PC2 para cada equipo, respectivamente.
- `'Equipo': df_final.index` → como el índice del DataFrame son los nombres de los equipos (desde la celda de inicio), esto arma la columna de nombres alineada fila a fila con sus componentes.
- `px.scatter(..., hover_name='Equipo')` → gráfico interactivo de Plotly; `hover_name` hace que, al pasar el mouse sobre un punto, se muestre el nombre del equipo en vez de solo las coordenadas numéricas.
- **Resultado real**: PC1 explica **80,60%** de la varianza (interpretado como "Intensidad de carrera") y PC2 explica **16,69%** ("Velocidad pura") — 97,29% acumulado entre las dos. Reducir de 4 variables a 2 casi no pierde información.

### Bloque 6 — Panorama de Métodos y Cierre

La regla rápida que resume el notebook, útil como diapositiva mental de cierre:

- **Reglas de Asociación**: patrones lógicos de coocurrencia (canastas de compra, sinergias de eventos).
- **K-Means**: grupos claros y circulares, cuando ya tenés una idea de cuántos querés.
- **Jerárquico**: cuando importa entender la taxonomía/árbol de relación entre los datos, no solo el grupo final.
- **DBSCAN**: datos con formas complejas, o necesidad de aislar ruido/anomalías con precisión.
- **PCA**: antes de modelar o graficar, para sacar la redundancia (correlación) y simplificar el problema.
