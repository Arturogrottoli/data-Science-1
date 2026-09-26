# Clase 07 — Fundamentos de Machine Learning: de la Teoría a Scikit-Learn

**Curso de Data Science I · Clase 07** — del mapa completo de la Inteligencia Artificial al primer contacto con `train_test_split`.

Esta guía es el **libreto completo para dictar la Clase 07**: reúne toda la teoría de `Clase 07.docx` (no solo lo que entra en una diapositiva) organizada en el mismo orden que las 50 filminas de `Clase07.html`. La lógica de cada bloque es siempre la misma: **primero el repaso de estadística** (lo que ya se sabe), **después una introducción apoyada en las filminas** (el mapa visual del tema), y **de ahí en adelante, filmina y teoría del docx intercaladas** — se proyecta la filmina correspondiente y, antes o después de mostrarla, se desarrolla en voz alta el texto de esta guía, que trae la profundidad completa que la filmina por sí sola no alcanza a mostrar.

> **Contexto de la clase anterior**: Clase 06 cerró con Estadística Descriptiva y Preprocesamiento (medidas de tendencia central/dispersión, normalización/estandarización, `StandardScaler`). Hoy no se repite esa teoría — se la repasa rápido al arrancar (Bloque 0 del notebook) y se la da por incorporada: entender un dato con estadística es el prerrequisito para que un algoritmo "aprenda" de ese mismo dato.

---

## Antes de Arrancar: ¿Qué es Machine Learning, y de qué va esta clase?

**¿Qué es el Machine Learning, en una frase?** Es la rama de la Inteligencia Artificial que, en vez de decirle a la computadora paso a paso qué hacer, le muestra muchos ejemplos y deja que ella misma encuentre el patrón. La computadora no "razona" como una persona — encuentra regularidades estadísticas en los datos que se le dan, y usa esas regularidades para opinar sobre datos nuevos que nunca vio. Ese es el salto que separa "programar" de "entrenar": no se escribe la regla, se muestra el ejemplo.

**El ejemplo que conviene tener siempre a mano hoy: un hospital que quiere detectar una enfermedad rara.**

Imaginá un hospital que atiende a 1.000 pacientes por semana, y de esos, solo 10 (el 1%) tienen una enfermedad grave pero poco frecuente. Se le pide a un equipo de Data Science armar un sistema que, a partir de los datos de cada paciente (síntomas, análisis, antecedentes), diga si tiene la enfermedad o no.

Acá aparece la primera trampa de toda la clase, y es la más importante de las 50 filminas: **un modelo que simplemente respondiera "sano" para TODOS los pacientes, sin mirar un solo dato real, acertaría el 99% de las veces** (990 de 1.000). Si solo se mirara ese número —la Accuracy, o "precisión global"— este modelo, que en realidad no aprendió absolutamente nada, parecería casi perfecto.

Pero ese 1% que falla es exactamente el que importa: son los 10 pacientes enfermos a los que el modelo les dijo que estaban sanos. Un sistema así no sirve para nada, por más espectacular que se vea el número — porque falló justo en el único propósito para el que fue creado.

**El mismo problema, con otro disfraz: el filtro de spam.** Un buzón de correo recibe muchísimos más mails legítimos que spam. Si un filtro decidiera "ningún mail es spam", acertaría la gran mayoría de las veces — pero dejaría pasar el 100% del spam real, que es justo lo que se le pidió que detecte. Es el mismo error de fondo que el del hospital: entrenar (o "resolver") de la forma más fácil posible, apostando a que el caso raro simplemente no importa lo suficiente como para arruinar el promedio.

**Por qué hace falta *entrenar* un modelo, y no programarlo a mano con reglas**: nadie puede escribir a mano una regla que capture todos los síntomas posibles de una enfermedad compleja, o todas las formas posibles de redactar un mail de spam — hay demasiadas combinaciones, y las reglas fijas fallan apenas aparece un caso que el programador no anticipó (esto se retoma en la Filmina 04, con el ejemplo médico exacto de "fiebre + tos"). En cambio, si se le muestran al modelo miles de historiales ya diagnosticados (o miles de mails ya marcados), el algoritmo puede aprender el patrón estadístico que distingue a unos de otros — sin que nadie le diga explícitamente cuál es la regla.

**Por qué esta trampa reaparece durante toda la clase**: que un modelo "perezoso" pueda parecer bueno con la métrica equivocada es una de las razones centrales por las que existe todo lo que se ve hoy — no alcanza con "entrenar un modelo", hace falta saber **medir** si realmente sirve (Tema 05, train/test), entender qué error es más caro para el negocio (Falso Positivo vs. Falso Negativo — un paciente sano al que se alarma innecesariamente no es lo mismo que un paciente enfermo al que se manda a su casa, Tema 03), y no confundir un número prometedor con una decisión correcta (Tema 06). El caso del hospital vuelve explícitamente en la Filmina 18 (Caso E) y en la Filmina 24 (Error 3, "un modelo con 99% de precisión es perfecto") — vale la pena, al llegar ahí, recordar que ya se había anticipado acá mismo, al principio de la clase.

**El mapa del día, en una frase**: primero se ve *qué es* el Machine Learning y cómo se relaciona con la IA y el Deep Learning (Tema 01); después, *cómo* aprende exactamente — con profesor, sin profesor, o por prueba y error (Tema 02); después se conecta todo con casos reales de impacto de negocio, en la Pre-entrega del módulo (Tema 03); recién ahí se entra al código, con la arquitectura de Scikit-Learn (Tema 04) y la forma correcta de evaluar sin hacer trampa (Tema 05); y se cierra viendo el ciclo de vida completo de un proyecto de ML en producción (Tema 06).

---

## Cómo está organizada esta guía

1. **Repaso de Estadística (Clase 06)** — el Bloque 0 del notebook, antes de tocar cualquier filmina.
2. **Introducción con las filminas** — recorrido rápido de portada y mapa general de los 6 temas del día.
3. **Filminas + teoría del docx, intercaladas** — el cuerpo principal de esta guía: cada uno de los 6 Temas, filmina por filmina, con el texto completo del docx desarrollado debajo de cada una.
4. **Guía del Notebook** y **Pre-entrega**, al final.

---

## Objetivos de la clase

1. Distinguir Inteligencia Artificial, Machine Learning y Deep Learning como una relación de inclusión, no como sinónimos.
2. Reconocer los tres paradigmas de aprendizaje (Supervisado, No Supervisado, Por Refuerzo) y saber diagnosticar cuál aplica a un problema real.
3. Conectar la teoría con aplicaciones reales de la industria (Netflix, Gmail, Uber, banca, salud) para entender el impacto de negocio del ML.
4. Entender la arquitectura interna de Scikit-Learn: Estimators, Transformers y Predictors, y el flujo `fit` → `transform`/`predict`.
5. Aplicar correctamente `train_test_split` y diagnosticar Overfitting, Underfitting y Data Leakage.
6. Completar la Pre-entrega "Aplicaciones Prácticas de ML — Del Algoritmo al Impacto Real".

---

## Repaso de la Clase 06 — Estadística y Preprocesamiento

Antes de entrar a la Clase 07, conviene tener fresco lo que se vio en Clase 06 — es el terreno sobre el que se apoya todo lo de hoy: **para que un algoritmo "aprenda" de un dataset, primero hace falta poder describirlo con estadística.** Esto es exactamente lo que hace el Bloque 0 del notebook (ver "Guía del Notebook" más abajo), sobre el dataset real de natalidad del DEIS. Los cuatro pilares que se repasan:

1. **Limpieza e Integración**: ningún dataset real llega listo para analizar. **Limpiar** significa decidir qué hacer con nulos (imputarlos con la media/mediana si son numéricos, con la moda o una etiqueta de negocio si son categóricos) y con duplicados (eliminarlos) — sin borrar a ciegas, porque un nulo puede tener una causa de negocio válida detrás. **Integrar** significa combinar o derivar columnas nuevas a partir de las existentes, para que la información cruda se vuelva accionable.

2. **Medidas de Tendencia Central y Dispersión**: la **tendencia central** responde "¿dónde está el centro de los datos?" — la **media** (sensible a valores extremos), la **mediana** (el valor que deja 50% de los datos a cada lado, no sensible a extremos) y la **moda** (el valor más frecuente). Cuando media y mediana difieren mucho, es señal de asimetría o de outliers. La **dispersión** responde "¿qué tan esparcidos están?" — el **desvío estándar** mide la variación promedio respecto a la media, y el **IQR** (rango intercuartílico, Q3 − Q1) mide el ancho del 50% central de los datos, siendo más robusto frente a outliers.

3. **Distribuciones y Correlación**: la **distribución** es la forma que toman los datos al graficarlos (histograma) — simétrica (campana/Normal) o sesgada a la izquierda/derecha. La **correlación** (coeficiente de Pearson, entre -1 y 1) mide qué tan asociadas linealmente están dos variables numéricas: cerca de 1 suben juntas, cerca de -1 una sube cuando la otra baja, cerca de 0 no hay relación lineal. Regla de oro que se repite todo el curso: **correlación no implica causalidad**.

4. **Transformación y Reducción de Dimensionalidad**: para que un algoritmo matemático procese los datos hace falta **transformarlos** — convertir texto a números y llevar las variables numéricas a una escala comparable (`StandardScaler`), porque los modelos basados en distancias son sensibles a la magnitud de cada columna. Cuando hay muchas columnas, **PCA** permite comprimirlas en unas pocas dimensiones que conservan la mayor parte de la variabilidad original.

**Por qué este repaso es más que un trámite**: el punto (4) —escalar antes de medir distancias, y ajustar el escalador solo con los datos de entrenamiento— es literalmente la misma regla de oro que se retoma formalmente hoy en el Tema 05 (Data Leakage). No es contenido nuevo disfrazado de repaso: es el mismo concepto, primero en estadística pura y después aplicado a Machine Learning.

---

## Introducción con las Filminas

## Filmina 01 — Portada

Apertura de la clase. El subtítulo ya anticipa el arco del día: arrancamos con el mapa conceptual completo de la IA (Tema 01) y terminamos con el primer código ejecutable de Scikit-Learn (Temas 04-05) — de la teoría más abstracta a la herramienta más concreta, en una sola clase. Recorrido rápido de los 6 temas antes de arrancar, para que la clase tenga un mapa mental de adónde va cada bloque: **(1)** el mapa de la IA, **(2)** los tres tipos de aprendizaje, **(3)** aplicaciones prácticas y la Pre-entrega, **(4)** Scikit-Learn por dentro, **(5)** train/test y sobreajuste, **(6)** un segundo repaso de aplicaciones, ahora con el ciclo de vida completo de un proyecto.

**La misma introducción, del lado del notebook**: `Clase07.ipynb` abre con su propio resumen del día — *"Dataset: `propiedades_sueca_ml.csv`, la continuación de la 'valija' de la clase pasada, ya limpia y lista para entrenar un primer modelo"* — y un recorrido en 4 bloques que comprime los 6 Temas de las filminas en una lógica de práctica: **(1)** el mapa de la IA/ML/DL y los tipos de aprendizaje, **(2)** Scikit-Learn por dentro, **(3)** entrenar y evaluar sin trampas, **(4)** consolidación y ciclo de vida completo. Vale la pena mostrar esta misma diapositiva de apertura junto con la introducción del notebook, para que quede clara la correspondencia: el notebook no es "otro tema", es la bajada práctica de las mismas 6 paradas, solo que agrupadas de a dos.

---

# Tema 01 — IA, Machine Learning y Deep Learning: el Mapa Completo (Filminas 02-09)

## Filmina 02 — División de Tema

Divisor de sección. La analogía de las matrioskas que trae la filmina es el ancla de todo el tema — conviene dibujarla en el pizarrón antes de seguir, porque los tres temas siguientes (reglas, ML, DL) son literalmente "abrir" cada muñeca rusa una por una.

**Teoría completa del docx para esta apertura**: imaginá que trabajás en el departamento de atención al cliente de una gran tienda online. Cada día llegan miles de correos electrónicos — algunos son felicitaciones, otros quejas por envíos retrasados, y muchos preguntas sobre devoluciones. Clasificarlos a mano tardaría horas. Se podría escribir un programa con reglas fijas: *"si el correo contiene la palabra 'retraso', enviarlo a Logística"*. Pero, ¿qué pasa si el cliente escribe *"mi paquete no ha llegado"*? La regla falla. Ahí es donde entra la Inteligencia Artificial. La IA no es una "caja negra" mágica, sino un ecosistema de tecnologías organizadas jerárquicamente que permiten resolver problemas complejos de formas que antes eran imposibles.

## Filmina 03 — El Ecosistema de la Inteligencia Artificial

**Teoría completa (1. El Ecosistema de la Inteligencia Artificial, del docx)**: a menudo se usan los términos Inteligencia Artificial (IA), Machine Learning (ML) y Deep Learning (DL) como si fueran intercambiables. Sin embargo, en el mundo profesional de la Ciencia de Datos es fundamental entender que existe una **relación de inclusión**. Imaginá una serie de muñecas rusas o matrioskas: la **IA** es la muñeca más grande (el campo general); el **Machine Learning** es la muñeca que está dentro de la IA; el **Deep Learning** es la muñeca más pequeña, ubicada dentro del Machine Learning. Esta jerarquía significa que **todo el Deep Learning es Machine Learning, y todo el Machine Learning es IA**, pero **no toda la IA es Machine Learning** (existen los sistemas de reglas, que se ven en la próxima filmina).

**Pregunta para tirar a la clase**: ¿alguien puede pensar en un ejemplo de su propio trabajo o vida diaria donde una regla fija "si esto, entonces aquello" dejó de funcionar apenas la situación se volvió un poco más compleja de lo previsto?

## Filmina 04 — IA Simbólica: Sistemas Basados en Reglas

**Teoría completa (2. Inteligencia Artificial: El concepto paraguas, del docx)**: la Inteligencia Artificial es la rama de las ciencias de la computación que busca crear sistemas capaces de realizar tareas que, si fueran hechas por humanos, requerirían inteligencia. Esto incluye razonamiento, aprendizaje, percepción y resolución de problemas. En los inicios de la IA, la mayoría de los sistemas funcionaban con reglas "Si-Entonces" (If-Then) definidas por humanos — la llamada **IA Simbólica**. Ejemplo: un sistema de diagnóstico médico donde un experto humano escribe *"Si el paciente tiene fiebre &gt; 38°C Y tiene tos, entonces sugiere test de gripe"*. **Limitación**: el programador debe prever cada escenario posible. Si el mundo cambia o el problema es muy complejo (como reconocer una cara en una foto), es imposible escribir suficientes reglas manuales.

**Matiz que no está en la filmina**: los sistemas basados en reglas no son "IA vieja e inútil" — siguen siendo la opción correcta cuando el proceso es 100% predecible (un termostato, un menú telefónico "Presione 1 para ventas"). El problema no es la técnica en sí, es usarla para un problema que no es predecible.

## Filmina 05 — Machine Learning: el Aprendizaje a través de Datos

**Teoría completa (3. Machine Learning: El aprendizaje a través de datos, del docx)**: el Machine Learning (Aprendizaje Automático) es un subconjunto de la IA que rompe con el paradigma de las reglas manuales. En lugar de decirle a la computadora qué reglas seguir, le damos **datos** (ejemplos) y un **algoritmo** para que ella misma descubra los patrones. En módulos anteriores se aprendió a usar condicionales en Python (`if`, `else`). Si se quisiera detectar correos de spam con Python básico, habría que listar miles de palabras prohibidas. Con Machine Learning, se le entregan al modelo 10.000 correos marcados como "Spam" y 10.000 como "No Spam". El algoritmo analiza la frecuencia de palabras, la hora de envío y la estructura para crear su propia "regla matemática" interna.

**El rol crítico de los datos y el Feature Engineering**: como ya se vio con Pandas y Estadística, la calidad del dato es lo más importante. En el ML clásico se practica el **Feature Engineering** (Ingeniería de Características): el proceso de seleccionar y transformar las variables (columnas) que se le entregan al modelo. Ejemplo: si se quiere predecir el precio de una casa, se decide que las columnas "m2", "barrio" y "número de habitaciones" son las importantes. El modelo no "sabe" qué es una casa, solo procesa los números que se eligieron.

## Filmina 06 — Deep Learning: la Potencia de las Redes Neuronales

**Teoría completa (4. Deep Learning: La potencia de las Redes Neuronales, del docx)**: el Deep Learning (Aprendizaje Profundo) es una evolución del Machine Learning que utiliza estructuras llamadas **Redes Neuronales Artificiales**. Recibe el nombre de "profundo" porque estas redes tienen muchas capas de procesamiento (decenas o cientos). La gran diferencia con el ML clásico radica en el tipo de datos que maneja y cómo procesa las características:

- **Datos no estructurados**: mientras que el ML clásico brilla con tablas de Excel (datos estructurados), el Deep Learning es el rey de las imágenes, el sonido y el texto libre (datos no estructurados).
- **Extracción automática de características**: a diferencia del ML, donde nosotros elegimos las variables, el Deep Learning puede aprender por sí solo qué partes de una imagen son importantes (bordes, texturas, formas) para identificar que lo que hay en la foto es un gato.

## Filmina 07 — Tabla Comparativa: ML Tradicional vs. Deep Learning

**Teoría completa (tabla del docx, desarrollada fila por fila)**:

| Característica | Machine Learning Tradicional | Deep Learning |
|---|---|---|
| Volumen de datos | Funciona bien con conjuntos pequeños/medianos | Requiere cantidades masivas de datos |
| Hardware | Puede correr en una laptop estándar | Suele requerir GPUs (procesadores gráficos potentes) |
| Intervención humana | Mucha (necesita Feature Engineering manual) | Baja (aprende características automáticamente) |
| Tiempo de entrenamiento | De segundos a horas | De días a semanas |
| Ejemplos | Predicción de ventas, scoring bancario | Reconocimiento facial, traducción automática, ChatGPT |

**Cómo usar esta tabla en clase**: en vez de leerla fila por fila, conviene pedirle al grupo que la lea al revés — dado un escenario ("tengo 500 filas de ventas en un Excel" vs. "tengo 2 millones de fotos de rayos X"), que decidan qué fila les da la pista de qué técnica conviene.

## Filmina 08 — Aplicaciones en la Industria Actual

**Teoría completa (5. Aplicaciones en la industria actual, del docx)**: para entender cuándo usar cada enfoque, tres casos reales:

- **IA Basada en Reglas**: los sistemas de control de temperatura de una oficina o los menús automáticos de un soporte telefónico ("Presione 1 para ventas"). Son eficientes cuando el proceso es 100% predecible.
- **Machine Learning (Scikit-Learn)**: un banco que quiere predecir si un cliente pagará un préstamo basándose en su historial crediticio, edad e ingresos. Aquí los datos están en tablas y el modelo puede explicar por qué tomó una decisión (interpretabilidad).
- **Deep Learning (Redes Neuronales)**: un sistema de conducción autónoma de Tesla que debe identificar peatones, semáforos y otros autos en milisegundos a partir de cámaras de video.

## Filmina 09 — Errores Comunes y Mejores Prácticas

**Teoría completa (6. Errores comunes y mejores prácticas, del docx)**: es fácil dejarse llevar por el entusiasmo de las nuevas tecnologías, pero un buen Data Scientist debe evitar estos errores:

- **"El Deep Learning siempre es mejor": Falso.** Para datos tabulares (hojas de cálculo), el ML clásico suele ser más rápido, barato y preciso que una red neuronal compleja. No usar un cañón para matar un mosquito.
- **Confundir el modelo con los datos**: un modelo de ML es el "estudiante" y el dataset es el "libro de texto". Si el libro es malo (datos sucios, sesgados o incompletos), el estudiante aprenderá mal por muy inteligente que sea.
- **Pensar que la IA "entiende"**: los modelos no tienen conciencia ni "entienden" conceptos. Son funciones matemáticas muy sofisticadas que encuentran correlaciones estadísticas. Si un modelo de lenguaje dice "Hola", no es porque sea educado, sino porque estadísticamente "Hola" es la respuesta más probable tras un saludo.

**Nota de cierre del tema**: 📌 *Entregable de este módulo: Pre-entrega — Aplicaciones Prácticas de ML (Del Algoritmo al Impacto Real)*, evaluable, suma al proyecto final. Su consigna llega en el Tema 03.

---

# Tema 02 — Tipos de Aprendizaje: Supervisado, No Supervisado y por Refuerzo (Filminas 10-18)

## Filmina 10 — División de Tema

**Teoría completa de apertura (del docx)**: imaginá que querés enseñarle a un niño a identificar diferentes tipos de frutas. Hay varias estrategias posibles: podrías mostrarle una manzana y decirle repetidamente "esto es una manzana"; podrías darle una cesta llena de frutas mezcladas y pedirle que las agrupe por su parecido sin decirle qué son; o podrías dejarlo en un huerto y darle un premio cada vez que recoja una fruta madura y deliciosa. En el mundo del Machine Learning, estas tres estrategias representan los tres grandes paradigmas o "modos" en los que una máquina puede aprender de los datos. Entender estos tres tipos de aprendizaje es fundamental porque determina todas las decisiones futuras de un científico de datos: desde qué algoritmo elegir hasta cómo medir si el modelo realmente funciona.

## Filmina 11 — El Concepto de "la Señal de Aprendizaje"

**Teoría completa (1. El concepto de "La Señal de Aprendizaje", del docx)**: antes de profundizar, hace falta entender un concepto clave: la **etiqueta (label)**. En Data Science se suele trabajar con tablas. Imaginá una tabla de datos de departamentos en alquiler: las **Features (Características o Entradas)** son las columnas como metros cuadrados, cantidad de habitaciones, barrio, tiene balcón — la información que se usa para alimentar al modelo. El **Label (Etiqueta o Salida)** es el resultado que se quiere predecir, por ejemplo el precio del alquiler. La presencia o ausencia de esta "etiqueta" es lo que define, en gran medida, ante qué tipo de aprendizaje se está.

## Filmina 12 — Aprendizaje Supervisado: "el Estudiante con Profesor"

**Teoría completa (2. Aprendizaje Supervisado, del docx)**: el Aprendizaje Supervisado es el paradigma más común en la industria. Se llama así porque el modelo cuenta con un "profesor" (el dataset etiquetado) que le proporciona ejemplos de la vida real junto con su respuesta correcta. Al modelo se le entregan miles de ejemplos con sus respectivas soluciones; el algoritmo intenta encontrar la relación matemática entre las features y la etiqueta. Una vez que "aprende" esa relación, se le entregan datos nuevos (sin etiqueta) para que prediga el resultado.

Las dos grandes tareas:
- **Clasificación**: predice una categoría o clase discreta (Sí/No, A/B/C). Ejemplo real: **Detección de Spam en Gmail** — el "profesor" le dio a Google millones de correos marcados manualmente como "Spam" o "No Spam". El modelo aprendió que palabras como "Gratis", "Gane dinero ya" o remitentes extraños suelen ser Spam.
- **Regresión**: predice un valor numérico continuo. Ejemplo real: **Precio de una vivienda** — el modelo analiza datos históricos de casas vendidas (m², ubicación, año) y sus precios finales, y estima el precio de una casa nueva.

**¿Por qué importa?** Porque la mayoría de las preguntas de negocio son supervisadas: "¿este cliente se va a dar de baja?", "¿cuánto va a vender mi tienda el próximo mes?", "¿es esta transacción un fraude?".

## Filmina 13 — Aprendizaje No Supervisado: "Buscando Estructura en el Caos"

**Teoría completa (3. Aprendizaje No Supervisado, del docx)**: ¿qué pasa si no hay etiquetas? Imaginá tener los datos de compras de 10 millones de clientes, pero sin saber quiénes son ahorradores, quiénes compran por impulso o quiénes prefieren productos de lujo — no hay una "respuesta correcta" previa. Acá entra el Aprendizaje No Supervisado: el modelo no intenta predecir nada, en su lugar explora los datos para encontrar **patrones ocultos o estructuras intrínsecas**.

Las tareas principales:
- **Clustering (Agrupamiento)**: agrupa los datos en "clusters" donde los elementos de un mismo grupo se parecen mucho entre sí y son muy distintos a los de otros grupos. Ejemplo real: **Segmentación de clientes en una app de música** — Spotify agrupa usuarios no por edad, sino por comportamiento: "usuarios que escuchan podcasts de noche", "usuarios que solo escuchan hits del momento". Esto permite campañas de marketing ultra-específicas sin que nadie haya etiquetado previamente a los usuarios.
- **Reducción de Dimensionalidad**: a veces hay demasiada información (cientos de columnas) y el modelo busca simplificar los datos quedándose solo con lo más importante, sin perder la esencia.

**Un error común**: muchos estudiantes creen que el aprendizaje no supervisado no tiene un objetivo. ¡Error! El objetivo es **descubrir**, no predecir. Es como organizar una colección de miles de fotos familiares por colores predominantes sin saber quién aparece en ellas; al final, hay una estructura que antes no se veía.

## Filmina 14 — Aprendizaje por Refuerzo: "Aprender por Ensayo y Error"

**Teoría completa (4. Aprendizaje por Refuerzo, del docx)**: este es el paradigma más distinto de los tres, y es la base de los avances más espectaculares en IA reciente, como los coches autónomos o los sistemas que vencen a campeones mundiales de ajedrez. A diferencia del supervisado (donde hay respuestas) o el no supervisado (donde hay patrones), acá un **Agente** (el algoritmo) interactúa con un **Entorno**.

**¿Cómo funciona?** El agente toma una **Acción**. Dependiendo de si esa acción lo acerca o lo aleja de su objetivo, recibe una **Recompensa** (+) o una **Penalización** (−). El objetivo del agente es maximizar la recompensa acumulada a largo plazo. Es exactamente como se aprende a jugar a un videojuego: no se nace sabiendo que tocar la lava mata; se prueba, se pierden puntos (penalización), y el cerebro aprende a no hacerlo de nuevo.

**Ejemplos emblemáticos**:
- **AlphaGo de Google DeepMind**: aprendió a jugar al Go (un juego de estrategia milenario) jugando millones de partidas contra sí mismo. No tenía un archivo CSV con las "mejores jugadas"; aprendió qué movimientos llevaban a la victoria mediante el refuerzo constante.
- **Robótica Industrial**: un brazo robótico en una fábrica puede aprender la trayectoria más eficiente para mover una pieza mediante pequeñas recompensas cada vez que el movimiento es fluido y preciso.

## Filmina 15 — Cuadro Comparativo: ¿Cuál Elegir?

**Teoría completa (5. Cuadro Comparativo, del docx)**: para un científico de datos principiante, saber distinguir cuál usar es el primer paso de cualquier proyecto.

| Característica | Supervisado | No Supervisado | Por Refuerzo |
|---|---|---|---|
| Datos iniciales | Etiquetados (Entrada + Salida) | No etiquetados (Solo Entrada) | Sin datos previos; aprende interactuando |
| Objetivo | Predecir resultados / Clasificar | Encontrar patrones / Agrupar | Tomar decisiones secuenciales |
| Feedback | Directo (Error vs. Respuesta real) | No tiene feedback explícito | Recompensa o Penalización |
| Analogía | Estudiar con el solucionario | Ordenar un ropero desordenado | Aprender a montar en bicicleta |

## Filmina 16 — Errores y Confusiones Comunes

**Teoría completa (6. Errores y Confusiones Comunes, del docx)**:

- **"¿Puedo usar clustering para clasificar correos?"**: No. El clustering agrupará correos parecidos, pero no sabrá cuál es "Spam". Para eso hacen falta etiquetas previas (Supervisado).
- **"El aprendizaje por refuerzo es solo prueba y error"**: No es azar. El algoritmo usa estructuras matemáticas (como Redes Neuronales) para decidir qué camino probar basándose en experiencias pasadas, para ser cada vez más inteligente.
- **"Si tengo muchos datos el modelo será perfecto"**: si las etiquetas en el aprendizaje supervisado están mal (por ejemplo, se marcaron correos buenos como spam por error), el modelo aprenderá a equivocarse. La calidad del dato manda sobre la cantidad.

## Filmina 17 — Síntesis y Conexiones

**Teoría completa (7. Síntesis y Conexiones, del docx)**: el Machine Learning no es una "caja negra" única, sino un conjunto de herramientas adaptables. Se usa **Aprendizaje Supervisado** si se tiene la respuesta histórica y se quiere predecir el futuro. Se usa **Aprendizaje No Supervisado** si se quiere explorar los datos y entender cómo se agrupan. Se usa **Aprendizaje por Refuerzo** si se necesita que un sistema aprenda a tomar decisiones complejas en un entorno dinámico. En las próximas unidades se implementan estos conceptos usando Scikit-Learn, la librería estándar de Python para ML, que utiliza una estructura lógica muy clara para manejar estos tipos de aprendizaje.

## Filmina 18 — Práctica (no entregable): Diagnóstico de 5 Casos

**Instrucciones completas (del docx)**: analizar 5 casos de uso de la industria y, para cada uno, indicar el **Tipo de Aprendizaje** (Supervisado —Clasificación o Regresión—, No Supervisado, o Por Refuerzo), la **Justificación** (¿existen etiquetas? ¿hay respuesta correcta? ¿se busca estructura? ¿hay recompensas?) y una **Métrica sugerida** para medir el éxito.

- **Caso A**: un banco quiere saber si un cliente que solicita un préstamo lo devolverá o no, basándose en el historial de pagos previos de miles de clientes antiguos.
- **Caso B**: una cadena de supermercados tiene datos de 50.000 clientes (compras, horarios, edad) y quiere encontrar grupos de "estilos de vida" para orientar sus folletos de ofertas.
- **Caso C**: una empresa de logística quiere entrenar a un vehículo autónomo para que aprenda a estacionarse solo en un depósito, dándole puntos positivos cuando queda derecho y restando puntos si choca.
- **Caso D**: una inmobiliaria quiere crear una herramienta que estime el valor de mercado de los departamentos basándose en metros cuadrados, ubicación y antigüedad.
- **Caso E**: un hospital tiene miles de imágenes de rayos X marcadas por médicos especialistas como "Normal" o "Infección". Quieren un sistema que ayude a los médicos a priorizar urgencias.

Cierra con una reflexión final: cuál de los tres paradigmas parece más complejo de implementar, y por qué.

**Errores comunes a evitar (del docx)**: confundir Clustering (No Supervisado) con Clasificación (Supervisado) — si el problema menciona datos ya "marcados" o "históricos con resultado", es Supervisado. Olvidar que en el Aprendizaje por Refuerzo no hay un dataset estático inicial, sino un proceso de interacción constante.

---

# Tema 03 — Aplicaciones Prácticas de ML: Del Algoritmo al Impacto Real (Filminas 19-27)

## Filmina 19 — División de Tema

✅ **Entregable evaluado del Módulo** — ver el detalle completo en la sección "Pre-entrega" al final de esta guía. El resto de las prácticas de la clase son guiadas y no evaluables; esta es la que se corrige y suma al proyecto final.

**Teoría completa de apertura (del docx)**: imaginá ser el dueño de una tienda de comercio electrónico que crece rápidamente. Al principio se podía saludar a cada cliente y recomendarle productos personalmente, pero con 100.000 clientes diarios es físicamente imposible que una persona (o incluso un equipo grande) analice el comportamiento de cada usuario para ofrecerle lo que busca en el momento justo. Ahí entra el Machine Learning: no como un concepto de ciencia ficción, sino como una herramienta práctica que automatiza la toma de decisiones a escala.

## Filmina 20 — El Cambio de Paradigma: de Reglas a Patrones

**Teoría completa (1. El Cambio de Paradigma: De Reglas a Patrones, del docx)**: para entender las aplicaciones prácticas, primero hace falta entender qué problema vino a solucionar el ML. **El Enfoque Tradicional (Basado en Reglas)**: antes del auge del ML, para que una computadora detectara correos de spam había que escribir cientos de reglas manuales ("si el correo contiene la palabra 'GRATIS' en mayúsculas, marcar como spam"; "si el remitente no está en la lista de contactos y pide dinero, marcar como spam"). **El problema**: los estafadores son creativos. Empezarían a escribir "G.R.A.T.I.S" o usar sinónimos; el programador tendría que actualizar las reglas constantemente hasta que el sistema se vuelve tan complejo que se rompe. **El Enfoque de Machine Learning**: en lugar de programar reglas, se le dan a la computadora miles de ejemplos de correos spam y legítimos, y el sistema aprende a identificar los patrones por sí solo. Si los estafadores cambian su táctica, simplemente se alimenta al modelo con los nuevos ejemplos y este se adapta. **Concepto clave**: el Machine Learning es la herramienta ideal cuando las reglas son demasiado numerosas, cambian con el tiempo, o son imposibles de explicar con palabras (como describir cómo se reconoce la cara de un amigo).

## Filmina 21 — El Ciclo de Vida de una Aplicación de ML

**Teoría completa (2. El Ciclo de Vida de una Aplicación de ML, del docx)**: en la práctica, implementar Machine Learning no es solo "entrenar un modelo". Es un proceso sistémico de cuatro grandes etapas:

- **Datos**: la materia prima. Sin datos históricos de calidad (ejemplos de lo que pasó en el pasado), no hay aprendizaje.
- **Entrenamiento**: el proceso donde el algoritmo analiza los datos para encontrar correlaciones. Acá se crea el "Modelo".
- **Evaluación**: antes de lanzar el modelo al mundo, se lo prueba con datos que nunca ha visto, para asegurarse de que realmente aprendió y no solo memorizó.
- **Uso Real (Inferencia)**: el modelo se integra en una aplicación (como una app de banco) para tomar decisiones sobre datos nuevos en tiempo real.

## Filmina 22 — Aplicaciones Reales: ¿Quién lo Usa y para Qué?

**Teoría completa (3. Aplicaciones Reales, del docx)**: para que el ML deje de ser una "caja negra", cuatro ejemplos de empresas conocidas — cada una resuelve un problema de negocio específico mediante la detección de patrones.

- **A. Sistemas de Recomendación — el caso Netflix**: Netflix no muestra películas al azar. Su sistema de ML analiza el historial de visualización, qué géneros se prefieren, a qué hora se conecta el usuario y qué personas con gustos similares han visto. ¿Qué predice? La probabilidad de que el usuario vea al menos el 70% de un título. Valor práctico: mantiene a los usuarios suscritos al reducir la fatiga de decisión.
- **B. Clasificación de Seguridad — Gmail y el Spam**: Google utiliza modelos que analizan el texto, los metadatos y la reputación del remitente para filtrar correos no deseados. ¿Qué predice? Una puntuación del 0 al 1, donde 1 es "definitivamente spam". Valor práctico: ahorra tiempo y protege de estafas (phishing) de forma automática.
- **C. Logística y Movilidad — Uber**: Uber utiliza ML para predecir el futuro cercano en la ciudad. ¿Qué predice? El tiempo estimado de llegada (ETA), la demanda de viajes en una zona específica (para activar precios dinámicos) y la ruta más eficiente. Valor práctico: optimiza el uso de los vehículos y mejora la experiencia del usuario.
- **D. Salud — Diagnóstico por Imagen**: en medicina se entrenan modelos de Deep Learning con miles de radiografías o resonancias marcadas por expertos. ¿Qué predice? La presencia de anomalías, como un tumor o una fractura, a veces con mayor precisión o velocidad que el ojo humano cansado. Valor práctico: sirve como una "segunda opinión" para los doctores, permitiendo detecciones tempranas.

## Filmina 23 — Más Casos de la Industria

**Teoría completa (tabla del docx)**:

| Caso de Uso | Tecnología/Empresa | Función Principal |
|---|---|---|
| Detección de Fraude | BBVA / PayPal | Identifica transacciones inusuales en milisegundos para bloquear robos |
| Predicción de Demanda | Zara / Amazon | Estima cuántas tallas "M" se venderán en una tienda para evitar falta de inventario |
| Mantenimiento Predictivo | General Electric | Predice cuándo fallará una turbina de avión antes de que ocurra, basándose en sensores de vibración |

## Filmina 24 — Errores Comunes y Falsas Expectativas

**Teoría completa (4. Errores Comunes y Falsas Expectativas, del docx)**: cuando un estudiante o una empresa comienza con ML, es fácil caer en trampas conceptuales.

- **Error 1: "El Machine Learning es Magia"**. Realidad: el ML es estadística aplicada a gran escala. No "entiende" conceptos filosóficos. Si se entrena un modelo para predecir ventas usando solo datos de temperatura, el modelo encontrará una relación, aunque no tenga sentido lógico. El ML detecta **correlaciones**, no necesariamente **causalidad**.
- **Error 2: "Más datos siempre es mejor"**. Realidad: los datos malos producen modelos malos (*Garbage In, Garbage Out*). Si los datos están sesgados, el modelo será sesgado. Ejemplo: si un algoritmo de contratación se entrena con datos históricos de una empresa que nunca contrató mujeres, el modelo aprenderá que "ser hombre" es un patrón de éxito — un error grave y discriminatorio.
- **Error 3: "Un modelo con 99% de precisión es perfecto"**. Realidad: depende del contexto. En la detección de una enfermedad rara que afecta a 1 de cada 100 personas, si el modelo siempre dice "estás sano", ¡tendrá un 99% de precisión! Pero habrá fallado en detectar al único enfermo, que era su propósito principal. En la práctica, hay que elegir la métrica que realmente importe para el problema.

## Filmina 25 — Terminología Clave para Profesionales

**Teoría completa (5. Terminología Clave, del docx)**: para hablar el lenguaje del sector, hay que dominar estos términos en su contexto práctico:

- **Features (Características)**: las variables que se le dan al modelo para que aprenda. En el caso de una casa: metros cuadrados, barrio, número de habitaciones.
- **Label (Etiqueta)**: lo que se quiere predecir. En el caso de la casa, el "precio".
- **Inferencia**: el acto de usar el modelo ya entrenado para obtener una respuesta. "Hacer una inferencia" es pedirle al modelo que prediga algo sobre un dato nuevo.
- **Sobreajuste (Overfitting)**: ocurre cuando el modelo memoriza los datos de entrenamiento tan bien que no sabe qué hacer cuando ve algo un poco diferente. Es como un estudiante que memoriza las respuestas del examen pero no entiende la materia: si cambia la pregunta, reprueba.

## Filmina 26 — Síntesis y Conexión

**Teoría completa (6. Síntesis y Conexión, del docx)**: el Machine Learning no es un ente aislado, sino un componente dentro de un sistema más grande. Su valor no reside en la complejidad del algoritmo, sino en la **decisión que ayuda a tomar**: ¿es este correo spam? ¿qué precio debe tener este producto? ¿está este paciente en riesgo? En las próximas unidades se pasa de la teoría a la herramienta: Scikit-Learn, la librería estándar de la industria, con una estructura lógica (estimadores y transformadores) que permite convertir datos en predicciones reales. **Pregunta para reflexionar**: mirá las aplicaciones de tu teléfono ahora mismo — ¿en cuáles creés que hay un modelo de Machine Learning trabajando en segundo plano? Probablemente, en casi todas.

## Filmina 27 — Práctica (no entregable): Perfilado de una Solución de ML

**Instrucciones completas (del docx)**: elegir un proceso del trabajo actual, estudio o vida cotidiana que hoy se haga manualmente o mediante reglas simples (categorizar facturas, decidir qué publicar en redes sociales, predecir cuándo ir al supermercado). Definir los componentes: **La Tarea** (¿qué decisión o predicción se quiere automatizar?), **las Features** (al menos 5 datos de entrada que el modelo necesitaría — fecha, monto, palabra clave, hora del día), **la Label** (¿cuál es la "respuesta correcta" que el modelo debe aprender a predecir?). Anticipar desafíos: ¿de dónde saldrían los datos históricos? ¿qué sesgo (bias) podría tener el modelo si los datos no son representativos? ¿cuál sería el "costo del error"?

---

# Tema 04 — Scikit-Learn por Dentro: Estimators, Transformers y Predictors (Filminas 28-33)

## Filmina 28 — División de Tema

**Teoría completa de apertura (del docx)**: entrar en el mundo de Scikit-Learn (o `sklearn`) es como entrar en una fábrica perfectamente organizada. No importa si se quiere predecir el precio de una casa, clasificar correos como spam o agrupar clientes: **todos los objetos se comportan de la misma manera**. Esta uniformidad es la mayor fortaleza de la librería, y se basa en tres conceptos fundamentales: **Estimators**, **Transformers** y **Predictors**.

## Filmina 29 — Estimators: la Base de Todo (el Alumno)

**Teoría completa (1. Estimators, del docx)**: un **Estimador** es cualquier objeto que aprende de los datos. El método estrella es `.fit()`. Imaginá que el estimador es un alumno: cuando se ejecuta `model.fit(X, y)`, el alumno abre su libro (los datos) y empieza a estudiar. **Parámetros**: son las "instrucciones" que se le dan al alumno antes de estudiar (ej. "estudia rápido" o "fíjate mucho en los detalles"). Se definen al crear el objeto: `Model(opcion=True)`. **Atributos aprendidos**: es lo que el alumno anotó en su cuaderno tras estudiar. En Scikit-Learn, estos atributos siempre terminan en un guion bajo, como `model.coef_` o `scaler.mean_`.

## Filmina 30 — Transformers: Transformando la Realidad (el Filtro)

**Teoría completa (2. Transformers, del docx)**: un **Transformer** es un tipo de estimador que, tras "estudiar" los datos, puede modificarlos. Para ello usa el método `.transform()`. ¿Para qué sirve? Para normalizar datos, rellenar valores faltantes (imputación) o convertir categorías en números. `fit_transform()`: una forma rápida de aprender la receta y aplicarla en el mismo paso.

## Filmina 31 — Predictors: Tomando Decisiones (el Juez)

**Teoría completa (3. Predictors, del docx)**: un **Predictor** es un estimador capaz de hacer pronósticos sobre datos nuevos mediante el método `.predict()`. Recibe datos (`X`) y devuelve una predicción (`y_pred`). **Importante**: para que un predictor funcione bien, los datos nuevos deben tener exactamente la misma forma y escala que los datos con los que el modelo "estudió".

## Filmina 32 — El Flujo de Trabajo Estándar

**Teoría completa (El Flujo de Trabajo Estándar, del docx)**:

1. **Instanciar**: se crea el objeto (ej. `scaler = StandardScaler()`).
2. **Ajustar (Fit)**: el objeto aprende de los datos de entrenamiento.
3. **Transformar o Predecir**: se aplica lo aprendido.

**Recordar**: `fit` es aprender la receta, `transform` es cocinar con ella. No hace falta volver a aprender la receta cada vez que se cocina un plato nuevo.

## Filmina 33 — Un Error Común: ¡Cuidado con el Fit!

**Teoría completa (Un error común: ¡Cuidado con el Fit!, del docx)**: un error muy frecuente de principiante es hacer `.fit()` sobre los datos de prueba o sobre datos nuevos. **¡No hay que hacerlo!** Solo se "estudia" (fit) con el conjunto de entrenamiento. Para los datos nuevos, solo se aplica lo aprendido con `.transform()` o `.predict()`.

**Por qué esto conecta directo con el Tema 05**: este mismo error es, en el fondo, una forma de Data Leakage — dejar que el conjunto de test "contamine" el proceso de aprendizaje. El Tema 05 lo retoma con nombre propio y en más profundidad.

---

# Tema 05 — Entrenar y Evaluar sin Trampas: Train/Test y Sobreajuste (Filminas 34-40)

## Filmina 34 — División de Tema

**Teoría completa de apertura (del docx)**: imaginá estar estudiando para un examen final de matemáticas. El profesor entrega una guía con 50 ejercicios resueltos para practicar. Se pasa toda la semana estudiando esos 50 ejercicios hasta saberlos de memoria. Si el profesor, el día del examen, pone exactamente los mismos 50 ejercicios, ¿realmente se aprendió matemática o simplemente se tiene una excelente memoria? Probablemente lo segundo. Ahora, si el profesor pone ejercicios diferentes pero basados en los mismos conceptos, y se logran resolver, entonces sí se aprendió. En Machine Learning sucede lo mismo: el objetivo no es que el modelo "memorice" los datos que se le dan, sino que "aprenda" los patrones generales para poder predecir resultados en situaciones que nunca ha visto.

## Filmina 35 — Train/Test Split: el "Examen" del Modelo

**Teoría completa (1. El Flujo de Entrenamiento y Evaluación, del docx)**: cuando se entrena un modelo, se quiere que sea capaz de **generalizar**: la capacidad de un modelo de ML de realizar predicciones precisas sobre datos nuevos, que no formaron parte del proceso de entrenamiento. Si se evalúa al modelo con los mismos datos que se usaron para enseñarle, se cae en una trampa: el modelo parecerá perfecto (porque ya conoce las respuestas), pero fallará estrepitosamente cuando se lo lleve al mundo real. **La solución: Train/Test Split** — dividir el dataset original en dos partes disjuntas:

- **Conjunto de Entrenamiento (Train Set)**: el material de estudio. El modelo utiliza estos datos para buscar patrones y ajustar sus parámetros internos. Normalmente representa entre el 70% y el 80% de los datos.
- **Conjunto de Prueba (Test Set)**: el examen final. Son datos que el modelo **nunca ve** durante el entrenamiento. Solo se usan al final para medir qué tan bien aprendió a generalizar.

**¿Por qué es esto fundamental?** Porque en la industria, un modelo que no generaliza no tiene valor. Por ejemplo, en un sistema de detección de fraudes bancarios, no sirve un modelo que reconozca los fraudes del año pasado si no es capaz de detectar un fraude nuevo con un patrón ligeramente distinto hoy.

## Filmina 36 — Los Tres Estados de un Modelo

**Teoría completa (2. Los tres estados de un modelo, del docx)**: al comparar el rendimiento del modelo en el conjunto de entrenamiento versus el de prueba, se puede diagnosticar qué tan bien está funcionando.

- **A. Underfitting (Subajuste): "El estudiante distraído"**. Ocurre cuando el modelo es demasiado simple para capturar la estructura subyacente de los datos. Señal: el error es alto tanto en entrenamiento como en prueba. Analogía: es como intentar explicar la economía global usando solo el precio del pan — faltan variables y complejidad.
- **B. Overfitting (Sobreajuste): "El estudiante que memoriza"**. Es el problema más común y peligroso. El modelo es tan complejo que empieza a aprender el **ruido** y los detalles aleatorios de los datos de entrenamiento, creyendo que son reglas generales. Señal: el error es casi cero en entrenamiento, pero muy alto en prueba. Analogía: es el estudiante que se memoriza que la respuesta a la pregunta 3 es "C", pero no sabe por qué — si en el examen la pregunta 3 cambia de orden, falla.
- **C. El "Sweet Spot" (Punto óptimo)**. Es el equilibrio perfecto: el modelo es lo suficientemente complejo para entender los patrones, pero lo suficientemente robusto para ignorar el ruido aleatorio. Acá, el error en entrenamiento y prueba es bajo y similar.

## Filmina 37 — Implementación con Scikit-Learn

**Teoría completa + código (3. Implementación con Scikit-Learn, del docx)**: en Python, la biblioteca Scikit-Learn facilita enormemente esta tarea con la función `train_test_split`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Parámetros clave**: `test_size=0.2` indica que se quiere reservar el 20% de los datos para la prueba y usar el 80% para entrenar. `random_state=42` es un punto vital para la **reproducibilidad**: al fijar una "semilla" aleatoria, se asegura que cada vez que se corra el código, la división de los datos sea la misma. Si no se hace, los resultados cambiarán cada vez que se ejecute el notebook, haciendo difícil comparar experimentos.

## Filmina 38 — Peligros: Fuga de Información y Series Temporales

**Teoría completa (4. Peligros y Buenas Prácticas, del docx)**:

- **Fuga de Información (Data Leakage)**: uno de los errores más graves para un principiante es permitir que información del conjunto de prueba "se filtre" al entrenamiento. Por ejemplo: si se calcula el promedio de una columna usando **todo** el dataset antes de dividirlo, el conjunto de entrenamiento ya conoce información (el promedio) que incluye datos del futuro (el test). **Regla de oro**: primero se divide en train/test, y cualquier cálculo estadístico (promedios, escalado, limpieza) se calcula **solo** sobre el conjunto de entrenamiento y luego se aplica al de prueba.
- **Series Temporales: la excepción a la regla aleatoria**: si se trabaja con datos que dependen del tiempo (precios de acciones, clima, ventas mensuales), **nunca** se debe usar una división aleatoria. Si se mezclan datos de 2023 en el entrenamiento y de 2021 en la prueba, se estaría usando el futuro para predecir el pasado. En estos casos, se usa una división cronológica: se entrena con los años 1 al 4 y se prueba con el año 5.

## Filmina 39 — Resumen de Conceptos Clave

**Teoría completa (5. Resumen de conceptos clave, del docx)**:

- **Entrenar no es memorizar**: se buscan patrones generales, no ruido.
- **Partición 80/20**: es el estándar inicial para tener suficientes datos para aprender y suficientes para evaluar con rigor.
- **Detectar Overfitting**: si un modelo tiene un "100% de precisión" en entrenamiento pero falla en prueba, hay que sospechar inmediatamente de sobreajuste.
- **Reproducibilidad**: usar siempre `random_state` para que los experimentos sean consistentes.
- **Honestidad intelectual**: el conjunto de prueba es sagrado. No hay que ajustarlo ni "espiar" sus resultados hasta que el modelo esté listo para la evaluación final.

Dominar la separación de datos es lo que separa a un programador que usa herramientas de un verdadero Científico de Datos. Sin una evaluación honesta, cualquier modelo es solo una ilusión tecnológica.

## Filmina 40 — Práctica (no entregable): Evaluación Honesta

**Instrucciones completas (del docx)**: utilizar el dataset con el que se trabajó en las unidades anteriores de Scikit-Learn. El objetivo es transformar el script anterior de entrenamiento simple en un flujo de evaluación profesional. **Preparación del Entorno**: cargar el dataset limpio con Pandas, definir `X` (características) e `y` (target). **La Gran División**: usar `train_test_split` con `test_size=0.2` y `random_state` fijo (ej. 42) para reproducibilidad. **Entrenamiento**: crear una instancia de un modelo (regresor o clasificador simple según el dataset) y entrenarlo **únicamente** con `X_train`/`y_train`. **Evaluación Dual**: calcular el `.score()` del modelo sobre train y sobre test. **Análisis Crítico**: comparar ambos resultados, describir si hay Overfitting (puntaje alto en train, bajo en test) o Underfitting (puntaje bajo en ambos), y proponer una acción correctiva (simplificar el modelo, conseguir más datos).

**Errores comunes a evitar (del docx)**: fuga de datos (no entrenar el modelo con el dataset completo antes de evaluarlo); olvidar el `random_state` (si no se fija, cada vez que se envía el trabajo las métricas podrían variar, dificultando la revisión técnica).

---

# Tema 06 — Aplicaciones Prácticas de ML: del Modelo al Mundo Real (Filminas 41-49)

## Filmina 41 — División de Tema

**Teoría completa de apertura (del docx)**: llegados a la última unidad del módulo, ya se desarmó el motor del Machine Learning: la diferencia entre IA y Deep Learning, los tipos de aprendizaje, la arquitectura de Scikit-Learn, y la importancia crítica de evaluar los modelos sin hacer "trampa". Pero, ¿para qué sirve todo esto en la vida real? Un modelo de Machine Learning no es un fin en sí mismo; es una herramienta para resolver problemas que el software tradicional (basado en reglas fijas) simplemente no puede manejar.

## Filmina 42 — El Porqué del Machine Learning Aplicado

**Teoría completa (1. El Porqué del Machine Learning Aplicado, del docx)**: imaginá trabajar en el equipo de seguridad de un banco, con la tarea de escribir un programa para detectar correos que intentan robar contraseñas (phishing). Con programación tradicional, harían falta miles de reglas manuales: *"SI el correo contiene la palabra 'urgente' Y tiene un enlace sospechoso → MARCAR COMO PHISHING"*; *"SI el remitente es desconocido Y pide datos bancarios → MARCAR COMO PHISHING"*. **El problema**: los atacantes son creativos — mañana cambiarán "urgente" por "prioritario" o usarán una imagen en lugar de texto, y las reglas quedarían obsoletas en horas. Ahí es donde el Machine Learning aplicado brilla: en lugar de programar reglas, se alimenta al sistema con miles de ejemplos de correos reales (benignos y maliciosos), y el modelo aprende a identificar las señales sutiles —el "ruido" y los "patrones"— que un humano o una lista de reglas estáticas pasarían por alto. **ML como un "Filtro Inteligente"**: se pueden ver las aplicaciones de ML como filtros que, dada una entrada compleja (una imagen, un historial de compras, una señal de sensor), producen una salida útil (una categoría, un precio estimado, una alerta).

## Filmina 43 — Casos de Uso: ¿Quién Está Usando ML Hoy?

**Teoría completa (2. Casos de Uso, del docx)**: para entender el impacto del ML, ejemplos concretos de industrias transformadas por esta tecnología.

- **A. Sistemas de Recomendación — el "Efecto Netflix"**: Netflix, Spotify y Amazon son los reyes de este dominio, con un enfoque llamado **Filtro Colaborativo**. El problema: millones de productos y poco tiempo del usuario. La solución ML: el modelo analiza los patrones (qué se vio, qué se saltó, a qué hora se conecta) y los compara con millones de otros usuarios similares. Resultado: no solo se recomiendan "películas de acción", sino "películas de acción que le gustaron a personas que tienen gustos idénticos".
- **B. Detección de Fraude Bancario**: empresas como Mastercard o Visa procesan miles de transacciones por segundo. El problema: es imposible que un humano revise cada compra en tiempo real. La solución ML: modelos de **detección de anomalías**. El sistema conoce el "comportamiento normal" del usuario (dónde suele comprar, montos típicos); si aparece una compra de un reloj de lujo en otro continente, el modelo asigna una "puntuación de riesgo" alta y bloquea la transacción en milisegundos.
- **C. Logística y Movilidad — Uber y la Estimación de Tiempos**: cuando se pide un Uber y dice "llega en 4 minutos", hay un modelo de **Regresión** trabajando. El problema: el tráfico, el clima y los accidentes cambian constantemente. La solución ML: el modelo toma características (hora del día, datos históricos de la ruta, condiciones climáticas) para predecir un valor continuo: el tiempo de llegada (ETA).
- **D. Salud — Diagnóstico por Imagen**: en medicina, el ML ayuda a salvar vidas mediante visión por computadora. El problema: un radiólogo puede estar fatigado después de revisar 100 radiografías. La solución ML: modelos de **Clasificación de Imágenes** entrenados con millones de escaneos pueden resaltar áreas sospechosas de tumores con una precisión que iguala o supera a los expertos, funcionando como un "segundo par de ojos" incansable.

## Filmina 44 — Salud: Diagnóstico por Imagen

**Ampliación de la Filmina 43, punto D**: este caso merece su propio momento porque conecta con la advertencia de la Filmina 09 (pensar que la IA "entiende") — el modelo no "ve" un tumor como lo vería un médico. Detecta patrones estadísticos en píxeles que correlacionan con casos ya diagnosticados por expertos humanos. No reemplaza al radiólogo: funciona como apoyo para detecciones tempranas, en un contexto donde el costo de un error es altísimo.

## Filmina 45 — El Ciclo de Vida de un Proyecto de ML: el Pipeline

**Teoría completa (3. El Ciclo de Vida de un Proyecto de ML: El Pipeline, del docx)**: entrenar el modelo (lo que se hizo con Scikit-Learn en el Tema 04-05) es solo una pequeña parte del trabajo real. En el mundo profesional se sigue un flujo de trabajo o **pipeline**:

1. **Definición del Problema**: ¿qué se quiere predecir? ¿Es una clasificación (Sí/No) o una regresión (Número)?
2. **Recolección y Limpieza de Datos**: el paso más largo. Como se dice en la industria: *"Garbage in, garbage out"* (si entran datos basura, sale un modelo basura).
3. **Ingeniería de Características (Feature Engineering)**: elegir qué datos son relevantes. Para predecir el precio de una casa, el número de baños es clave; el color de la puerta, probablemente no.
4. **Entrenamiento y Evaluación**: acá se usa Scikit-Learn para ajustar el modelo y probarlo con datos que nunca ha visto (test set).
5. **Despliegue (Inferencia)**: se pone el modelo a trabajar en una aplicación real.
6. **Monitoreo**: los modelos pueden "deteriorarse" si el mundo cambia (por ejemplo, un modelo de predicción de ventas antes y después de una pandemia).

## Filmina 46 — Trampas y Errores Comunes: lo que Nadie te Dice

**Teoría completa (4. Trampas y Errores Comunes, del docx)**: incluso con los mejores datos, es fácil cometer errores conceptuales que arruinan una aplicación práctica.

- **Error 1: Confundir Correlación con Causalidad**. Un modelo de ML encuentra correlaciones. Si un modelo nota que "las personas que compran protector solar también compran helados", podría sugerir que el protector solar **causa** hambre de helado. Realidad: hay una variable oculta (el sol/verano). Lección: el modelo no entiende el "porqué", solo el "qué" — las decisiones de negocio deben ser validadas por humanos.
- **Error 2: El Sesgo en los Datos (Bias)**. Si se entrena un modelo de selección de personal usando solo currículums de personas contratadas en los últimos 20 años en una empresa que históricamente favoreció a hombres, el modelo aprenderá que "ser hombre" es una característica de éxito. Realidad: el modelo no es racista ni sexista por sí mismo; simplemente es un espejo de los datos que se le dieron.
- **Error 3: Sobreajuste (Overfitting)**. Un modelo que memoriza los datos de entrenamiento pero falla en la vida real es inútil — es como un estudiante que se memoriza las respuestas del examen pero no entiende la materia: si cambia un número en el examen, reprueba.

## Filmina 47 — Glosario para el Mundo Profesional

**Teoría completa (5. Glosario para el Mundo Profesional, del docx)**:

- **Modelo**: el "cerebro" que ya aprendió y está listo para decidir.
- **Inferencia**: el acto de usar el modelo para predecir algo nuevo (ej. cuando se sube una foto y Facebook sugiere etiquetas).
- **Features (Características)**: las columnas de entrada de los datos.
- **Labels (Etiquetas)**: la respuesta correcta que el modelo intenta aprender en aprendizaje supervisado.
- **Métricas**: los termómetros para saber si el modelo es bueno (Precisión, Recall, Error Cuadrático Medio).

## Filmina 48 — Síntesis y Cierre del Módulo

**Teoría completa (6. Síntesis y Cierre del Módulo, del docx)**: se completó un viaje desde la teoría de la IA hasta la mecánica del entrenamiento de modelos. El Machine Learning no es magia; es **estadística aplicada a gran escala**. La clave para ser un buen científico de datos no es conocer todos los algoritmos del mundo, sino saber: qué problema amerita usar ML (y cuál no); cómo preparar los datos para que el modelo aprenda patrones reales; cómo evaluar el éxito no solo con números, sino con impacto en el mundo real. El Machine Learning es una herramienta poderosa — hay que usarla para construir sistemas que no solo sean precisos, sino también éticos y transparentes.

## Filmina 49 — Práctica (no entregable): Diseño de una Solución de ML

**Instrucciones completas (del docx)**: **Identificar un problema** — pensar en el trabajo actual, un hobby o una empresa admirada, y qué proceso manual o repetitivo podría beneficiarse de una predicción o clasificación automática. **Definir la tarea**: ¿Clasificación (Categorías) o Regresión (Valores numéricos)? ¿Aprendizaje Supervisado o No Supervisado? **Proponer las Características (Features)**: enumerar al menos 5 datos (columnas) que el modelo necesitaría para aprender a tomar esa decisión. **Definir el éxito**: ¿cómo se sabría que el modelo funciona? Elegir una métrica (precisión, tiempo ahorrado, reducción de errores). **Considerar la ética**: ¿qué sesgos potenciales podrían existir en los datos que se recolectarían?

**Qué mirar al corregir (no está en el docx)**: esta práctica final retoma exactamente la misma estructura que la Pre-entrega del Tema 03 (Tarea, Features, Label, métrica, sesgo) — es una buena señal si el alumno ya la resuelve más rápido y con más soltura que la primera vez, porque significa que el vocabulario y el razonamiento quedaron incorporados.

## Filmina 50 (última) — ¿Dudas? ¿Consultas?

Cierre de la clase — espacio abierto antes de que el grupo se ponga a trabajar en la Pre-entrega.

---

## Guía del Notebook

**Estado actual**: el notebook completo ya está construido en [`Clase07.ipynb`](Clase07.ipynb) — Bloque 0 (repaso de Clase 06) + 4 Bloques prácticos que cubren los 6 Temas de esta guía, con horarios sugeridos de clase (0:00 a 1:55) y un solucionario para el docente al final. Dos datasets conviven en el notebook a propósito: `tasa-natalidad-deis-2000-2024.csv` para el repaso de estadística (Bloque 0, el mismo dataset ya conocido de Clase 06), y `propiedades_sueca_ml.csv` (precios de propiedades, ya limpio) como el dataset nuevo para entrenar el primer modelo real de la clase. El notebook viejo en `material/Viejo/Clase_7_Fundamentos_de_Ciencia_de_Datos_1_.ipynb` (Pipelines + K-Means) queda obsoleto — no coincide con `Clase07.html` ni con `Clase 07.docx`.

### Bloque 0 — Repaso de la Clase 06 (Estadística y Preprocesamiento)

**Por qué arranca acá**: los seis temas de hoy dan por sentado que ya se sabe leer un dato con estadística (ver la sección "Repaso de la Clase 06" más arriba) — antes de que un algoritmo "aprenda" de un dataset, hace falta poder describirlo. Este bloque lo aplica en código real sobre el dataset de natalidad del DEIS (25 años, 25 provincias).

**Ejemplo 1 — Tendencia Central y Dispersión:**
```python
serie_nacional = df_raw['natalidad_argentina']

media = serie_nacional.mean()
mediana = serie_nacional.median()
std = serie_nacional.std()
q1 = serie_nacional.quantile(0.25)
q3 = serie_nacional.quantile(0.75)
iqr = q3 - q1
```
**Línea por línea:** `.mean()`, `.median()` y `.std()` calculan los tres estadísticos básicos sobre la serie completa de 25 años. `.quantile(0.25)` y `.quantile(0.75)` devuelven los valores que dejan el 25% y el 75% de los datos por debajo (Q1 y Q3); `iqr = q3 - q1` es el ancho de esa caja central. La deducción real que da el notebook: la mediana (17.9) es más alta que la media (16.35) — no por outliers, sino porque la natalidad viene en caída sostenida (hay más años "altos" al principio de la serie que "bajos" al final, y eso desplaza el promedio hacia abajo más de lo que desplaza al valor central).

**Ejemplo 2 — Distribuciones y Correlación:**
```python
sns.histplot(serie_nacional, kde=True, bins=10, color='teal')
...
matriz_corr = provincias_comparar.corr()
sns.heatmap(matriz_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, center=0)
```
**Línea por línea:** `sns.histplot(..., kde=True)` dibuja el histograma de la serie nacional con una curva suavizada (KDE) superpuesta, para ver la forma real de la distribución. `.corr()` calcula la matriz de correlación de Pearson entre Buenos Aires, Córdoba y Santa Fe; `sns.heatmap(...)` la pinta como cuadrícula de colores, con `vmin=-1, vmax=1` para que la escala de color sea siempre comparable. La deducción: la correlación entre provincias es altísima (>0.95) porque comparten la misma tendencia demográfica nacional — remarcando que eso es correlación, no causalidad.

**Ejemplo 3 — Transformación:**
```python
scaler_demo = StandardScaler()
columnas_escaladas = scaler_demo.fit_transform(columnas_ejemplo)
```
**Línea por línea:** `StandardScaler()` instancia el transformador; `.fit_transform(...)` aprende la media y el desvío de cada columna y aplica la estandarización en el mismo paso — dejando cada columna con media ≈ 0 y desvío ≈ 1. Es el mismo objeto (`StandardScaler`) que reaparece en los Bloques 2 y 3, ahora aplicado a un modelo de verdad.

### Bloque 1 — El Mapa de la IA, ML y DL (Temas 01-02, 0:00-0:30)

**El "rompehielo" (código real del notebook):**
```python
df.drop(columns=["precio_eur", "precio_por_m2", "id_propiedad"]).sample(5, random_state=1)
```
**Línea por línea:** `.drop(columns=[...])` saca del DataFrame las columnas que serían la "respuesta" (`precio_eur`, `precio_por_m2`) y el identificador (que no es una feature real); `.sample(5, random_state=1)` muestra 5 filas al azar, pero fijas gracias a la semilla. **Pregunta para el grupo, tal como la trae el notebook**: "si tuviera que escribir un programa con reglas fijas (`SI superficie > 100 Y barrio == Centro, ENTONCES precio > 200.000`) para estimar el precio de estas propiedades, ¿cuántas reglas necesitaría? ¿Alcanzaría alguna vez?" — la misma pregunta que abre el Tema 01, pero vivida en código antes de nombrarla.

Sigue con el mapa IA → ML → DL (matrioskas) y los tipos de aprendizaje, todo referido al mismo dataset de propiedades — sin código nuevo, son celdas de texto que retoman lo ya visto en las Filminas 03-15.

**El mini-quiz de tipos de aprendizaje (para resolver oral en el momento)**: el notebook cierra el Bloque 1 con 3 casos para que el grupo diagnostique en vivo, antes de seguir:

- Agrupar clientes de un supermercado por hábitos de compra, sin categorías previas.
- Predecir si un mail es spam, usando miles de mails ya marcados.
- Un termostato inteligente que aprende a ahorrar energía probando distintas temperaturas.

**Ojo**: estos son exactamente los 3 casos que después reaparecen como "Tarea 1" en el Solucionario del Bloque 4 (ver más abajo) — el notebook los plantea acá sin responder, y da la respuesta recién al cierre. Si se resuelven ya en el Bloque 1, la Tarea 1 del plenario final queda como repaso en vez de ejercicio nuevo — vale la pena decidir en qué momento conviene resolverlos según el ritmo del grupo.

### Bloque 2 — Scikit-Learn por Dentro (Tema 04, 0:30-1:00)

**División train/test y Transformer:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit_transform en train
X_test_scaled = scaler.transform(X_test)         # SOLO transform en test
```
**Línea por línea:** igual que en el Bloque 0, pero ahora con la regla de oro explícita en el comentario del propio notebook: `fit_transform` en train (aprende y aplica), `transform` solamente en test (aplica lo ya aprendido, sin volver a "estudiar"). Es la puesta en práctica literal de la Filmina 33 ("Cuidado con el Fit").

**Estimator + Predictor:**
```python
modelo = LinearRegression()
modelo.fit(X_train, y_train)          # fit: el modelo "estudia" la relación entre features y precio
predicciones = modelo.predict(X_test) # predict: usamos lo aprendido sobre datos nuevos
```
**Nota del propio notebook**: para esta Regresión Lineal no se escalan los features — no hace falta, y así los coeficientes quedan directamente interpretables en euros (con KNN o una regresión regularizada sí haría falta escalar).

**Atributos aprendidos:**
```python
coeficientes = pd.Series(modelo.coef_, index=features).round(1)
print(coeficientes)
print("\nIntercepto (precio base):", round(modelo.intercept_))
```
**Línea por línea:** `modelo.coef_` es el atributo aprendido (termina en `_`, como marca la Filmina 29) — un coeficiente por feature; `pd.Series(..., index=features)` le pone nombre a cada número para poder leerlo. La lectura de negocio que propone el notebook: por cada m² adicional el precio sube esa cantidad de euros, manteniendo todo lo demás constante; la antigüedad debería restar valor (conviene revisar el signo en vivo); `score_amenities` suma directo.

### Bloque 3 — Entrenar y Evaluar sin Trampas (Tema 05, 1:10-1:35)

**Baseline (Regresión Lineal) — R² y MAE en train vs. test:**
```python
print(f"R² train: {r2_score(y_train, pred_train):.3f}   R² test: {r2_score(y_test, pred_test):.3f}")
```
Train y test dan valores parecidos — señal de que el modelo generalizó, no memorizó. Sirve de punto de comparación para lo que sigue.

**Overfitting en acción — el contraste central del bloque:**
```python
arbol_libre = DecisionTreeRegressor(random_state=42)  # sin max_depth: crece sin límite
arbol_libre.fit(X_train, y_train)
# R² train: prácticamente perfecto | R² test: mucho más bajo

arbol_limitado = DecisionTreeRegressor(max_depth=4, random_state=42)
arbol_limitado.fit(X_train, y_train)
# R² train y test: mucho más parecidos entre sí
```
**Línea por línea:** `DecisionTreeRegressor(random_state=42)` sin `max_depth` puede crecer sin límite hasta memorizar cada fila del train — es el "estudiante que se memoriza las respuestas". `max_depth=4` limita cuántas veces se puede dividir el árbol, forzándolo a quedarse con los patrones generales en vez de los detalles particulares de cada fila. El propio notebook arma después una tabla comparando los 3 modelos (Regresión Lineal, árbol libre, árbol limitado) con una columna `gap` (`R2_train - R2_test`): cuanto más grande el gap, más sobreajuste.

**La trampa del Data Leakage:**
```python
# ERROR A PROPÓSITO: precio_por_m2 se calculó A PARTIR de precio_eur
features_con_leakage = features + ["precio_por_m2"]
...
print("R² test CON data leakage:", round(r2_score(y_test_l, modelo_leak.predict(X_test_l)), 4))
```
**Por qué es una trampa y no un logro**: `precio_por_m2` se calculó dividiendo `precio_eur` por `superficie_m2` — o sea que contiene casi la respuesta escondida adentro. El R² se dispara de forma sospechosa, y esa sospecha es justamente la señal de alarma a entrenar: en la vida real, ese dato ni siquiera existiría todavía al momento de predecir el precio de una propiedad nueva.

### Bloque 4 — Consolidación Guiada (Temas 03 y 06, 1:35-1:55)

Cierre con el ciclo de vida completo de un proyecto de ML (Definición → Datos → Entrenamiento → Evaluación → Despliegue/Inferencia → Monitoreo), y 4 tareas para resolver en plenario:

1. Tipos de aprendizaje sobre 3 mini-casos nuevos.
2. Interpretar `coeficientes.sort_values(ascending=False)` en términos de negocio.
3. Diagnosticar Overfitting/Underfitting/Sweet Spot a partir de una tabla de 3 modelos con R² dados (sin volver a entrenar nada — puro diagnóstico de números).
4. Proponer el próximo paso ante un modelo sobreajustado (consigna abierta).

Cierra con un **Solucionario** (uso docente) con las respuestas esperadas de las 4 tareas, para tener a mano mientras se conduce el plenario en vivo:

- **Tarea 1** (los 3 casos del mini-quiz del Bloque 1): agrupar clientes sin categorías previas → **No Supervisado** (clustering); predecir spam con mails ya marcados → **Supervisado** (clasificación); termostato que prueba y aprende de la reacción → **Por Refuerzo**.
- **Tarea 2** (coeficientes): la lectura esperada es que `superficie_m2` y `score_amenities` suman valor, `antiguedad_anios` resta. `ambientes` puede dar un coeficiente chico o levemente negativo controlando por superficie — el notebook aclara que es un caso de **multicolinealidad** (una propiedad más grande ya "trae" más ambientes, así que la variable superficie ya captura buena parte de esa información).
- **Tarea 3** (diagnóstico): Modelo A (R² train 0.95, test 0.93, gap 0.02) → **Sweet Spot**, generaliza bien. Modelo B (R² train 0.55, test 0.52, gap 0.03) → **Underfitting** — el gap es chico, pero el error es alto en ambos, señal de que el modelo es demasiado simple. Modelo C (R² train 0.99, test 0.61, gap 0.38) → **Overfitting**, memorizó el train.
- **Tarea 4** (consigna abierta, próximo paso ante overfitting): no hay una única respuesta correcta — opciones válidas que puede proponer el grupo: reducir la complejidad del modelo (bajar `max_depth`, menos features), conseguir más datos de entrenamiento, aplicar regularización (Ridge/Lasso), usar validación cruzada para elegir mejor los hiperparámetros, o eliminar features irrelevantes o correlacionadas entre sí.

**Nota sobre la Pre-entrega y el Podcast**: el notebook cierra mencionando que en la Pre-entrega de esta semana no se programa un modelo, sino que se "piensa como Data Scientist" (exactamente la Filmina 27) — y recomienda escuchar el Podcast del módulo como repaso antes de encararla. Ese podcast es contenido de audio de `Clase 07.docx` que se decidió **no** convertir en filminas (a diferencia de los 6 Temas, que sí están 1 a 1 en `Clase07.html`) — por eso no tiene una sección propia en esta guía.

---

## Pre-entrega: "Aplicaciones Prácticas de ML"

✅ **Entregable evaluado del Módulo**, anunciado en la Filmina 19 (división del Tema 03). A diferencia de otras clases del curso, `Clase 07.docx` no incluye una sección separada con "Qué tenés que presentar / Criterios de Aceptación / Formato de entrega" para esta Pre-entrega — el contenido más cercano a una consigna es la práctica del Tema 03 (Filmina 27, "Perfilado de una Solución de Machine Learning"), marcada en el propio texto como el ejercicio que **precede** al entregable evaluado.

**Lo que sí está definido explícitamente en el docx**: el módulo completo (Temas 01 a 03) cierra con la nota — *"Entregable de este módulo: Pre-entrega — Aplicaciones Prácticas de ML (Del Algoritmo al Impacto Real), evaluable, suma al proyecto final"* — pero sin una rúbrica propia dentro de este documento.

**Nota para quien dicte la clase**: si existe una consigna formal de esta Pre-entrega en otro documento (una rúbrica separada, un enunciado en el campus), conviene traerla a esta guía para completar la sección — tal como están las fuentes disponibles hoy (docx + html), esto es todo lo que se puede documentar sin inventar criterios que no están en el material original.

---

## Síntesis y Conexión Final

La clase entera se puede resumir en una progresión: primero entendemos el mapa completo de la Inteligencia Artificial y dónde vive el Machine Learning dentro de él (Tema 01); después aprendemos a diagnosticar qué tipo de aprendizaje aplica a un problema según si hay o no una etiqueta (Tema 02); conectamos esa teoría con aplicaciones reales de alto impacto de negocio, en la Pre-entrega del módulo (Tema 03); entramos por primera vez al código con la arquitectura interna de Scikit-Learn (Tema 04); aprendemos a evaluar sin hacer trampa, con train/test y el diagnóstico de sobreajuste (Tema 05); y cerramos viendo cómo todo esto se integra en el ciclo de vida completo de un proyecto de ML real, de punta a punta (Tema 06).

En la próxima unidad se retoman estos mismos conceptos aplicados a modelos concretos — Regresión Lineal, Árboles de Decisión y Regresión Logística — construyendo directamente sobre el flujo `train_test_split` + Estimators/Predictors que hoy se vio por primera vez.
