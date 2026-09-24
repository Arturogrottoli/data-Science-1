# Clase 07 — Fundamentos de Machine Learning: de la Teoría a Scikit-Learn

**Curso de Data Science I · Clase 07** — del mapa completo de la Inteligencia Artificial al primer contacto con `train_test_split`.

Esta guía es el **libreto completo para dictar la Clase 07**: reúne toda la teoría de `Clase 07.docx` (no solo lo que entra en una diapositiva) organizada en el mismo orden que las 51 filminas de `Clase07.html`. La lógica de cada bloque es siempre la misma: **primero el repaso de estadística** (lo que ya se sabe), **después una introducción apoyada en las filminas** (el mapa visual del tema), y **de ahí en adelante, filmina y teoría del docx intercaladas** — se proyecta la filmina correspondiente y, antes o después de mostrarla, se desarrolla en voz alta el texto de esta guía, que trae la profundidad completa que la filmina por sí sola no alcanza a mostrar.

> **Contexto de la clase anterior**: Clase 06 cerró con Estadística Descriptiva y Preprocesamiento (medidas de tendencia central/dispersión, normalización/estandarización, `StandardScaler`). Hoy no se repite esa teoría — se la repasa rápido al arrancar (Bloque 0 del notebook) y se la da por incorporada: entender un dato con estadística es el prerrequisito para que un algoritmo "aprenda" de ese mismo dato.

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

## Introducción con las Filminas

## Filmina 01 — Portada

Apertura de la clase. El subtítulo ya anticipa el arco del día: arrancamos con el mapa conceptual completo de la IA (Tema 01) y terminamos con el primer código ejecutable de Scikit-Learn (Temas 04-05) — de la teoría más abstracta a la herramienta más concreta, en una sola clase. Recorrido rápido de los 6 temas antes de arrancar, para que la clase tenga un mapa mental de adónde va cada bloque: **(1)** el mapa de la IA, **(2)** los tres tipos de aprendizaje, **(3)** aplicaciones prácticas y la Pre-entrega, **(4)** Scikit-Learn por dentro, **(5)** train/test y sobreajuste, **(6)** un segundo repaso de aplicaciones, ahora con el ciclo de vida completo de un proyecto.

---

# Tema 01 — IA, Machine Learning y Deep Learning: el Mapa Completo (Filminas 02-10)

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

## Filmina 10 — Práctica (no entregable): Mapa de Círculos Concéntricos

**Instrucciones completas (del docx)**: elegir uno de tres escenarios (detectar transacciones bancarias fraudulentas, identificar razas de perros por foto, o un recomendador de películas de streaming). Para el escenario elegido, mostrar en un diagrama cómo se abordaría el problema con los tres enfoques: **IA basada en reglas** (¿qué reglas manuales `if/else` se intentarían escribir?), **Machine Learning** (¿qué datos tabulares harían falta y cuáles serían las features principales?), **Deep Learning** (¿qué tipo de datos masivos alimentaría a la red neuronal y qué ventaja tendría sobre el ML clásico?). El diagrama debe usar círculos concéntricos o una jerarquía de bloques mostrando que el DL está dentro del ML y este dentro de la IA, con flechas de flujo de datos y etiquetas de ventajas/desventajas ("Requiere GPU", "Interpretable", "Fácil de programar"). Cierra con una conclusión de 2 frases sobre qué enfoque se elegiría y por qué.

**Errores comunes a evitar (del docx)**: colocar IA, ML y DL como cajas separadas e independientes (recordar que son subconjuntos, no alternativas); sugerir que el ML clásico puede procesar imágenes de alta resolución tan eficientemente como el Deep Learning sin un preprocesamiento manual extremo.

**Nota de cierre del tema**: 📌 *Entregable de este módulo: Pre-entrega — Aplicaciones Prácticas de ML (Del Algoritmo al Impacto Real)*, evaluable, suma al proyecto final. Su consigna llega en el Tema 03 — esta práctica es preparación conceptual, no se corrige ni se entrega.

---

# Tema 02 — Tipos de Aprendizaje: Supervisado, No Supervisado y por Refuerzo (Filminas 11-19)

## Filmina 11 — División de Tema

**Teoría completa de apertura (del docx)**: imaginá que querés enseñarle a un niño a identificar diferentes tipos de frutas. Hay varias estrategias posibles: podrías mostrarle una manzana y decirle repetidamente "esto es una manzana"; podrías darle una cesta llena de frutas mezcladas y pedirle que las agrupe por su parecido sin decirle qué son; o podrías dejarlo en un huerto y darle un premio cada vez que recoja una fruta madura y deliciosa. En el mundo del Machine Learning, estas tres estrategias representan los tres grandes paradigmas o "modos" en los que una máquina puede aprender de los datos. Entender estos tres tipos de aprendizaje es fundamental porque determina todas las decisiones futuras de un científico de datos: desde qué algoritmo elegir hasta cómo medir si el modelo realmente funciona.

## Filmina 12 — El Concepto de "la Señal de Aprendizaje"

**Teoría completa (1. El concepto de "La Señal de Aprendizaje", del docx)**: antes de profundizar, hace falta entender un concepto clave: la **etiqueta (label)**. En Data Science se suele trabajar con tablas. Imaginá una tabla de datos de departamentos en alquiler: las **Features (Características o Entradas)** son las columnas como metros cuadrados, cantidad de habitaciones, barrio, tiene balcón — la información que se usa para alimentar al modelo. El **Label (Etiqueta o Salida)** es el resultado que se quiere predecir, por ejemplo el precio del alquiler. La presencia o ausencia de esta "etiqueta" es lo que define, en gran medida, ante qué tipo de aprendizaje se está.

## Filmina 13 — Aprendizaje Supervisado: "el Estudiante con Profesor"

**Teoría completa (2. Aprendizaje Supervisado, del docx)**: el Aprendizaje Supervisado es el paradigma más común en la industria. Se llama así porque el modelo cuenta con un "profesor" (el dataset etiquetado) que le proporciona ejemplos de la vida real junto con su respuesta correcta. Al modelo se le entregan miles de ejemplos con sus respectivas soluciones; el algoritmo intenta encontrar la relación matemática entre las features y la etiqueta. Una vez que "aprende" esa relación, se le entregan datos nuevos (sin etiqueta) para que prediga el resultado.

Las dos grandes tareas:
- **Clasificación**: predice una categoría o clase discreta (Sí/No, A/B/C). Ejemplo real: **Detección de Spam en Gmail** — el "profesor" le dio a Google millones de correos marcados manualmente como "Spam" o "No Spam". El modelo aprendió que palabras como "Gratis", "Gane dinero ya" o remitentes extraños suelen ser Spam.
- **Regresión**: predice un valor numérico continuo. Ejemplo real: **Precio de una vivienda** — el modelo analiza datos históricos de casas vendidas (m², ubicación, año) y sus precios finales, y estima el precio de una casa nueva.

**¿Por qué importa?** Porque la mayoría de las preguntas de negocio son supervisadas: "¿este cliente se va a dar de baja?", "¿cuánto va a vender mi tienda el próximo mes?", "¿es esta transacción un fraude?".

## Filmina 14 — Aprendizaje No Supervisado: "Buscando Estructura en el Caos"

**Teoría completa (3. Aprendizaje No Supervisado, del docx)**: ¿qué pasa si no hay etiquetas? Imaginá tener los datos de compras de 10 millones de clientes, pero sin saber quiénes son ahorradores, quiénes compran por impulso o quiénes prefieren productos de lujo — no hay una "respuesta correcta" previa. Acá entra el Aprendizaje No Supervisado: el modelo no intenta predecir nada, en su lugar explora los datos para encontrar **patrones ocultos o estructuras intrínsecas**.

Las tareas principales:
- **Clustering (Agrupamiento)**: agrupa los datos en "clusters" donde los elementos de un mismo grupo se parecen mucho entre sí y son muy distintos a los de otros grupos. Ejemplo real: **Segmentación de clientes en una app de música** — Spotify agrupa usuarios no por edad, sino por comportamiento: "usuarios que escuchan podcasts de noche", "usuarios que solo escuchan hits del momento". Esto permite campañas de marketing ultra-específicas sin que nadie haya etiquetado previamente a los usuarios.
- **Reducción de Dimensionalidad**: a veces hay demasiada información (cientos de columnas) y el modelo busca simplificar los datos quedándose solo con lo más importante, sin perder la esencia.

**Un error común**: muchos estudiantes creen que el aprendizaje no supervisado no tiene un objetivo. ¡Error! El objetivo es **descubrir**, no predecir. Es como organizar una colección de miles de fotos familiares por colores predominantes sin saber quién aparece en ellas; al final, hay una estructura que antes no se veía.

## Filmina 15 — Aprendizaje por Refuerzo: "Aprender por Ensayo y Error"

**Teoría completa (4. Aprendizaje por Refuerzo, del docx)**: este es el paradigma más distinto de los tres, y es la base de los avances más espectaculares en IA reciente, como los coches autónomos o los sistemas que vencen a campeones mundiales de ajedrez. A diferencia del supervisado (donde hay respuestas) o el no supervisado (donde hay patrones), acá un **Agente** (el algoritmo) interactúa con un **Entorno**.

**¿Cómo funciona?** El agente toma una **Acción**. Dependiendo de si esa acción lo acerca o lo aleja de su objetivo, recibe una **Recompensa** (+) o una **Penalización** (−). El objetivo del agente es maximizar la recompensa acumulada a largo plazo. Es exactamente como se aprende a jugar a un videojuego: no se nace sabiendo que tocar la lava mata; se prueba, se pierden puntos (penalización), y el cerebro aprende a no hacerlo de nuevo.

**Ejemplos emblemáticos**:
- **AlphaGo de Google DeepMind**: aprendió a jugar al Go (un juego de estrategia milenario) jugando millones de partidas contra sí mismo. No tenía un archivo CSV con las "mejores jugadas"; aprendió qué movimientos llevaban a la victoria mediante el refuerzo constante.
- **Robótica Industrial**: un brazo robótico en una fábrica puede aprender la trayectoria más eficiente para mover una pieza mediante pequeñas recompensas cada vez que el movimiento es fluido y preciso.

## Filmina 16 — Cuadro Comparativo: ¿Cuál Elegir?

**Teoría completa (5. Cuadro Comparativo, del docx)**: para un científico de datos principiante, saber distinguir cuál usar es el primer paso de cualquier proyecto.

| Característica | Supervisado | No Supervisado | Por Refuerzo |
|---|---|---|---|
| Datos iniciales | Etiquetados (Entrada + Salida) | No etiquetados (Solo Entrada) | Sin datos previos; aprende interactuando |
| Objetivo | Predecir resultados / Clasificar | Encontrar patrones / Agrupar | Tomar decisiones secuenciales |
| Feedback | Directo (Error vs. Respuesta real) | No tiene feedback explícito | Recompensa o Penalización |
| Analogía | Estudiar con el solucionario | Ordenar un ropero desordenado | Aprender a montar en bicicleta |

## Filmina 17 — Errores y Confusiones Comunes

**Teoría completa (6. Errores y Confusiones Comunes, del docx)**:

- **"¿Puedo usar clustering para clasificar correos?"**: No. El clustering agrupará correos parecidos, pero no sabrá cuál es "Spam". Para eso hacen falta etiquetas previas (Supervisado).
- **"El aprendizaje por refuerzo es solo prueba y error"**: No es azar. El algoritmo usa estructuras matemáticas (como Redes Neuronales) para decidir qué camino probar basándose en experiencias pasadas, para ser cada vez más inteligente.
- **"Si tengo muchos datos el modelo será perfecto"**: si las etiquetas en el aprendizaje supervisado están mal (por ejemplo, se marcaron correos buenos como spam por error), el modelo aprenderá a equivocarse. La calidad del dato manda sobre la cantidad.

## Filmina 18 — Síntesis y Conexiones

**Teoría completa (7. Síntesis y Conexiones, del docx)**: el Machine Learning no es una "caja negra" única, sino un conjunto de herramientas adaptables. Se usa **Aprendizaje Supervisado** si se tiene la respuesta histórica y se quiere predecir el futuro. Se usa **Aprendizaje No Supervisado** si se quiere explorar los datos y entender cómo se agrupan. Se usa **Aprendizaje por Refuerzo** si se necesita que un sistema aprenda a tomar decisiones complejas en un entorno dinámico. En las próximas unidades se implementan estos conceptos usando Scikit-Learn, la librería estándar de Python para ML, que utiliza una estructura lógica muy clara para manejar estos tipos de aprendizaje.

## Filmina 19 — Práctica (no entregable): Diagnóstico de 5 Casos

**Instrucciones completas (del docx)**: analizar 5 casos de uso de la industria y, para cada uno, indicar el **Tipo de Aprendizaje** (Supervisado —Clasificación o Regresión—, No Supervisado, o Por Refuerzo), la **Justificación** (¿existen etiquetas? ¿hay respuesta correcta? ¿se busca estructura? ¿hay recompensas?) y una **Métrica sugerida** para medir el éxito.

- **Caso A**: un banco quiere saber si un cliente que solicita un préstamo lo devolverá o no, basándose en el historial de pagos previos de miles de clientes antiguos.
- **Caso B**: una cadena de supermercados tiene datos de 50.000 clientes (compras, horarios, edad) y quiere encontrar grupos de "estilos de vida" para orientar sus folletos de ofertas.
- **Caso C**: una empresa de logística quiere entrenar a un vehículo autónomo para que aprenda a estacionarse solo en un depósito, dándole puntos positivos cuando queda derecho y restando puntos si choca.
- **Caso D**: una inmobiliaria quiere crear una herramienta que estime el valor de mercado de los departamentos basándose en metros cuadrados, ubicación y antigüedad.
- **Caso E**: un hospital tiene miles de imágenes de rayos X marcadas por médicos especialistas como "Normal" o "Infección". Quieren un sistema que ayude a los médicos a priorizar urgencias.

Cierra con una reflexión final: cuál de los tres paradigmas parece más complejo de implementar, y por qué.

**Errores comunes a evitar (del docx)**: confundir Clustering (No Supervisado) con Clasificación (Supervisado) — si el problema menciona datos ya "marcados" o "históricos con resultado", es Supervisado. Olvidar que en el Aprendizaje por Refuerzo no hay un dataset estático inicial, sino un proceso de interacción constante.

---

# Tema 03 — Aplicaciones Prácticas de ML: Del Algoritmo al Impacto Real (Filminas 20-28)

## Filmina 20 — División de Tema

✅ **Entregable evaluado del Módulo** — ver el detalle completo en la sección "Pre-entrega" al final de esta guía. El resto de las prácticas de la clase son guiadas y no evaluables; esta es la que se corrige y suma al proyecto final.

**Teoría completa de apertura (del docx)**: imaginá ser el dueño de una tienda de comercio electrónico que crece rápidamente. Al principio se podía saludar a cada cliente y recomendarle productos personalmente, pero con 100.000 clientes diarios es físicamente imposible que una persona (o incluso un equipo grande) analice el comportamiento de cada usuario para ofrecerle lo que busca en el momento justo. Ahí entra el Machine Learning: no como un concepto de ciencia ficción, sino como una herramienta práctica que automatiza la toma de decisiones a escala.

## Filmina 21 — El Cambio de Paradigma: de Reglas a Patrones

**Teoría completa (1. El Cambio de Paradigma: De Reglas a Patrones, del docx)**: para entender las aplicaciones prácticas, primero hace falta entender qué problema vino a solucionar el ML. **El Enfoque Tradicional (Basado en Reglas)**: antes del auge del ML, para que una computadora detectara correos de spam había que escribir cientos de reglas manuales ("si el correo contiene la palabra 'GRATIS' en mayúsculas, marcar como spam"; "si el remitente no está en la lista de contactos y pide dinero, marcar como spam"). **El problema**: los estafadores son creativos. Empezarían a escribir "G.R.A.T.I.S" o usar sinónimos; el programador tendría que actualizar las reglas constantemente hasta que el sistema se vuelve tan complejo que se rompe. **El Enfoque de Machine Learning**: en lugar de programar reglas, se le dan a la computadora miles de ejemplos de correos spam y legítimos, y el sistema aprende a identificar los patrones por sí solo. Si los estafadores cambian su táctica, simplemente se alimenta al modelo con los nuevos ejemplos y este se adapta. **Concepto clave**: el Machine Learning es la herramienta ideal cuando las reglas son demasiado numerosas, cambian con el tiempo, o son imposibles de explicar con palabras (como describir cómo se reconoce la cara de un amigo).

## Filmina 22 — El Ciclo de Vida de una Aplicación de ML

**Teoría completa (2. El Ciclo de Vida de una Aplicación de ML, del docx)**: en la práctica, implementar Machine Learning no es solo "entrenar un modelo". Es un proceso sistémico de cuatro grandes etapas:

- **Datos**: la materia prima. Sin datos históricos de calidad (ejemplos de lo que pasó en el pasado), no hay aprendizaje.
- **Entrenamiento**: el proceso donde el algoritmo analiza los datos para encontrar correlaciones. Acá se crea el "Modelo".
- **Evaluación**: antes de lanzar el modelo al mundo, se lo prueba con datos que nunca ha visto, para asegurarse de que realmente aprendió y no solo memorizó.
- **Uso Real (Inferencia)**: el modelo se integra en una aplicación (como una app de banco) para tomar decisiones sobre datos nuevos en tiempo real.

## Filmina 23 — Aplicaciones Reales: ¿Quién lo Usa y para Qué?

**Teoría completa (3. Aplicaciones Reales, del docx)**: para que el ML deje de ser una "caja negra", cuatro ejemplos de empresas conocidas — cada una resuelve un problema de negocio específico mediante la detección de patrones.

- **A. Sistemas de Recomendación — el caso Netflix**: Netflix no muestra películas al azar. Su sistema de ML analiza el historial de visualización, qué géneros se prefieren, a qué hora se conecta el usuario y qué personas con gustos similares han visto. ¿Qué predice? La probabilidad de que el usuario vea al menos el 70% de un título. Valor práctico: mantiene a los usuarios suscritos al reducir la fatiga de decisión.
- **B. Clasificación de Seguridad — Gmail y el Spam**: Google utiliza modelos que analizan el texto, los metadatos y la reputación del remitente para filtrar correos no deseados. ¿Qué predice? Una puntuación del 0 al 1, donde 1 es "definitivamente spam". Valor práctico: ahorra tiempo y protege de estafas (phishing) de forma automática.
- **C. Logística y Movilidad — Uber**: Uber utiliza ML para predecir el futuro cercano en la ciudad. ¿Qué predice? El tiempo estimado de llegada (ETA), la demanda de viajes en una zona específica (para activar precios dinámicos) y la ruta más eficiente. Valor práctico: optimiza el uso de los vehículos y mejora la experiencia del usuario.
- **D. Salud — Diagnóstico por Imagen**: en medicina se entrenan modelos de Deep Learning con miles de radiografías o resonancias marcadas por expertos. ¿Qué predice? La presencia de anomalías, como un tumor o una fractura, a veces con mayor precisión o velocidad que el ojo humano cansado. Valor práctico: sirve como una "segunda opinión" para los doctores, permitiendo detecciones tempranas.

## Filmina 24 — Más Casos de la Industria

**Teoría completa (tabla del docx)**:

| Caso de Uso | Tecnología/Empresa | Función Principal |
|---|---|---|
| Detección de Fraude | BBVA / PayPal | Identifica transacciones inusuales en milisegundos para bloquear robos |
| Predicción de Demanda | Zara / Amazon | Estima cuántas tallas "M" se venderán en una tienda para evitar falta de inventario |
| Mantenimiento Predictivo | General Electric | Predice cuándo fallará una turbina de avión antes de que ocurra, basándose en sensores de vibración |

## Filmina 25 — Errores Comunes y Falsas Expectativas

**Teoría completa (4. Errores Comunes y Falsas Expectativas, del docx)**: cuando un estudiante o una empresa comienza con ML, es fácil caer en trampas conceptuales.

- **Error 1: "El Machine Learning es Magia"**. Realidad: el ML es estadística aplicada a gran escala. No "entiende" conceptos filosóficos. Si se entrena un modelo para predecir ventas usando solo datos de temperatura, el modelo encontrará una relación, aunque no tenga sentido lógico. El ML detecta **correlaciones**, no necesariamente **causalidad**.
- **Error 2: "Más datos siempre es mejor"**. Realidad: los datos malos producen modelos malos (*Garbage In, Garbage Out*). Si los datos están sesgados, el modelo será sesgado. Ejemplo: si un algoritmo de contratación se entrena con datos históricos de una empresa que nunca contrató mujeres, el modelo aprenderá que "ser hombre" es un patrón de éxito — un error grave y discriminatorio.
- **Error 3: "Un modelo con 99% de precisión es perfecto"**. Realidad: depende del contexto. En la detección de una enfermedad rara que afecta a 1 de cada 100 personas, si el modelo siempre dice "estás sano", ¡tendrá un 99% de precisión! Pero habrá fallado en detectar al único enfermo, que era su propósito principal. En la práctica, hay que elegir la métrica que realmente importe para el problema.

## Filmina 26 — Terminología Clave para Profesionales

**Teoría completa (5. Terminología Clave, del docx)**: para hablar el lenguaje del sector, hay que dominar estos términos en su contexto práctico:

- **Features (Características)**: las variables que se le dan al modelo para que aprenda. En el caso de una casa: metros cuadrados, barrio, número de habitaciones.
- **Label (Etiqueta)**: lo que se quiere predecir. En el caso de la casa, el "precio".
- **Inferencia**: el acto de usar el modelo ya entrenado para obtener una respuesta. "Hacer una inferencia" es pedirle al modelo que prediga algo sobre un dato nuevo.
- **Sobreajuste (Overfitting)**: ocurre cuando el modelo memoriza los datos de entrenamiento tan bien que no sabe qué hacer cuando ve algo un poco diferente. Es como un estudiante que memoriza las respuestas del examen pero no entiende la materia: si cambia la pregunta, reprueba.

## Filmina 27 — Síntesis y Conexión

**Teoría completa (6. Síntesis y Conexión, del docx)**: el Machine Learning no es un ente aislado, sino un componente dentro de un sistema más grande. Su valor no reside en la complejidad del algoritmo, sino en la **decisión que ayuda a tomar**: ¿es este correo spam? ¿qué precio debe tener este producto? ¿está este paciente en riesgo? En las próximas unidades se pasa de la teoría a la herramienta: Scikit-Learn, la librería estándar de la industria, con una estructura lógica (estimadores y transformadores) que permite convertir datos en predicciones reales. **Pregunta para reflexionar**: mirá las aplicaciones de tu teléfono ahora mismo — ¿en cuáles creés que hay un modelo de Machine Learning trabajando en segundo plano? Probablemente, en casi todas.

## Filmina 28 — Práctica (no entregable): Perfilado de una Solución de ML

**Instrucciones completas (del docx)**: elegir un proceso del trabajo actual, estudio o vida cotidiana que hoy se haga manualmente o mediante reglas simples (categorizar facturas, decidir qué publicar en redes sociales, predecir cuándo ir al supermercado). Definir los componentes: **La Tarea** (¿qué decisión o predicción se quiere automatizar?), **las Features** (al menos 5 datos de entrada que el modelo necesitaría — fecha, monto, palabra clave, hora del día), **la Label** (¿cuál es la "respuesta correcta" que el modelo debe aprender a predecir?). Anticipar desafíos: ¿de dónde saldrían los datos históricos? ¿qué sesgo (bias) podría tener el modelo si los datos no son representativos? ¿cuál sería el "costo del error"?

---

# Tema 04 — Scikit-Learn por Dentro: Estimators, Transformers y Predictors (Filminas 29-34)

## Filmina 29 — División de Tema

**Teoría completa de apertura (del docx)**: entrar en el mundo de Scikit-Learn (o `sklearn`) es como entrar en una fábrica perfectamente organizada. No importa si se quiere predecir el precio de una casa, clasificar correos como spam o agrupar clientes: **todos los objetos se comportan de la misma manera**. Esta uniformidad es la mayor fortaleza de la librería, y se basa en tres conceptos fundamentales: **Estimators**, **Transformers** y **Predictors**.

## Filmina 30 — Estimators: la Base de Todo (el Alumno)

**Teoría completa (1. Estimators, del docx)**: un **Estimador** es cualquier objeto que aprende de los datos. El método estrella es `.fit()`. Imaginá que el estimador es un alumno: cuando se ejecuta `model.fit(X, y)`, el alumno abre su libro (los datos) y empieza a estudiar. **Parámetros**: son las "instrucciones" que se le dan al alumno antes de estudiar (ej. "estudia rápido" o "fíjate mucho en los detalles"). Se definen al crear el objeto: `Model(opcion=True)`. **Atributos aprendidos**: es lo que el alumno anotó en su cuaderno tras estudiar. En Scikit-Learn, estos atributos siempre terminan en un guion bajo, como `model.coef_` o `scaler.mean_`.

## Filmina 31 — Transformers: Transformando la Realidad (el Filtro)

**Teoría completa (2. Transformers, del docx)**: un **Transformer** es un tipo de estimador que, tras "estudiar" los datos, puede modificarlos. Para ello usa el método `.transform()`. ¿Para qué sirve? Para normalizar datos, rellenar valores faltantes (imputación) o convertir categorías en números. `fit_transform()`: una forma rápida de aprender la receta y aplicarla en el mismo paso.

## Filmina 32 — Predictors: Tomando Decisiones (el Juez)

**Teoría completa (3. Predictors, del docx)**: un **Predictor** es un estimador capaz de hacer pronósticos sobre datos nuevos mediante el método `.predict()`. Recibe datos (`X`) y devuelve una predicción (`y_pred`). **Importante**: para que un predictor funcione bien, los datos nuevos deben tener exactamente la misma forma y escala que los datos con los que el modelo "estudió".

## Filmina 33 — El Flujo de Trabajo Estándar

**Teoría completa (El Flujo de Trabajo Estándar, del docx)**:

1. **Instanciar**: se crea el objeto (ej. `scaler = StandardScaler()`).
2. **Ajustar (Fit)**: el objeto aprende de los datos de entrenamiento.
3. **Transformar o Predecir**: se aplica lo aprendido.

**Recordar**: `fit` es aprender la receta, `transform` es cocinar con ella. No hace falta volver a aprender la receta cada vez que se cocina un plato nuevo.

## Filmina 34 — Un Error Común: ¡Cuidado con el Fit!

**Teoría completa (Un error común: ¡Cuidado con el Fit!, del docx)**: un error muy frecuente de principiante es hacer `.fit()` sobre los datos de prueba o sobre datos nuevos. **¡No hay que hacerlo!** Solo se "estudia" (fit) con el conjunto de entrenamiento. Para los datos nuevos, solo se aplica lo aprendido con `.transform()` o `.predict()`.

**Por qué esto conecta directo con el Tema 05**: este mismo error es, en el fondo, una forma de Data Leakage — dejar que el conjunto de test "contamine" el proceso de aprendizaje. El Tema 05 lo retoma con nombre propio y en más profundidad.

---

# Tema 05 — Entrenar y Evaluar sin Trampas: Train/Test y Sobreajuste (Filminas 35-41)

## Filmina 35 — División de Tema

**Teoría completa de apertura (del docx)**: imaginá estar estudiando para un examen final de matemáticas. El profesor entrega una guía con 50 ejercicios resueltos para practicar. Se pasa toda la semana estudiando esos 50 ejercicios hasta saberlos de memoria. Si el profesor, el día del examen, pone exactamente los mismos 50 ejercicios, ¿realmente se aprendió matemática o simplemente se tiene una excelente memoria? Probablemente lo segundo. Ahora, si el profesor pone ejercicios diferentes pero basados en los mismos conceptos, y se logran resolver, entonces sí se aprendió. En Machine Learning sucede lo mismo: el objetivo no es que el modelo "memorice" los datos que se le dan, sino que "aprenda" los patrones generales para poder predecir resultados en situaciones que nunca ha visto.

## Filmina 36 — Train/Test Split: el "Examen" del Modelo

**Teoría completa (1. El Flujo de Entrenamiento y Evaluación, del docx)**: cuando se entrena un modelo, se quiere que sea capaz de **generalizar**: la capacidad de un modelo de ML de realizar predicciones precisas sobre datos nuevos, que no formaron parte del proceso de entrenamiento. Si se evalúa al modelo con los mismos datos que se usaron para enseñarle, se cae en una trampa: el modelo parecerá perfecto (porque ya conoce las respuestas), pero fallará estrepitosamente cuando se lo lleve al mundo real. **La solución: Train/Test Split** — dividir el dataset original en dos partes disjuntas:

- **Conjunto de Entrenamiento (Train Set)**: el material de estudio. El modelo utiliza estos datos para buscar patrones y ajustar sus parámetros internos. Normalmente representa entre el 70% y el 80% de los datos.
- **Conjunto de Prueba (Test Set)**: el examen final. Son datos que el modelo **nunca ve** durante el entrenamiento. Solo se usan al final para medir qué tan bien aprendió a generalizar.

**¿Por qué es esto fundamental?** Porque en la industria, un modelo que no generaliza no tiene valor. Por ejemplo, en un sistema de detección de fraudes bancarios, no sirve un modelo que reconozca los fraudes del año pasado si no es capaz de detectar un fraude nuevo con un patrón ligeramente distinto hoy.

## Filmina 37 — Los Tres Estados de un Modelo

**Teoría completa (2. Los tres estados de un modelo, del docx)**: al comparar el rendimiento del modelo en el conjunto de entrenamiento versus el de prueba, se puede diagnosticar qué tan bien está funcionando.

- **A. Underfitting (Subajuste): "El estudiante distraído"**. Ocurre cuando el modelo es demasiado simple para capturar la estructura subyacente de los datos. Señal: el error es alto tanto en entrenamiento como en prueba. Analogía: es como intentar explicar la economía global usando solo el precio del pan — faltan variables y complejidad.
- **B. Overfitting (Sobreajuste): "El estudiante que memoriza"**. Es el problema más común y peligroso. El modelo es tan complejo que empieza a aprender el **ruido** y los detalles aleatorios de los datos de entrenamiento, creyendo que son reglas generales. Señal: el error es casi cero en entrenamiento, pero muy alto en prueba. Analogía: es el estudiante que se memoriza que la respuesta a la pregunta 3 es "C", pero no sabe por qué — si en el examen la pregunta 3 cambia de orden, falla.
- **C. El "Sweet Spot" (Punto óptimo)**. Es el equilibrio perfecto: el modelo es lo suficientemente complejo para entender los patrones, pero lo suficientemente robusto para ignorar el ruido aleatorio. Acá, el error en entrenamiento y prueba es bajo y similar.

## Filmina 38 — Implementación con Scikit-Learn

**Teoría completa + código (3. Implementación con Scikit-Learn, del docx)**: en Python, la biblioteca Scikit-Learn facilita enormemente esta tarea con la función `train_test_split`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Parámetros clave**: `test_size=0.2` indica que se quiere reservar el 20% de los datos para la prueba y usar el 80% para entrenar. `random_state=42` es un punto vital para la **reproducibilidad**: al fijar una "semilla" aleatoria, se asegura que cada vez que se corra el código, la división de los datos sea la misma. Si no se hace, los resultados cambiarán cada vez que se ejecute el notebook, haciendo difícil comparar experimentos.

## Filmina 39 — Peligros: Fuga de Información y Series Temporales

**Teoría completa (4. Peligros y Buenas Prácticas, del docx)**:

- **Fuga de Información (Data Leakage)**: uno de los errores más graves para un principiante es permitir que información del conjunto de prueba "se filtre" al entrenamiento. Por ejemplo: si se calcula el promedio de una columna usando **todo** el dataset antes de dividirlo, el conjunto de entrenamiento ya conoce información (el promedio) que incluye datos del futuro (el test). **Regla de oro**: primero se divide en train/test, y cualquier cálculo estadístico (promedios, escalado, limpieza) se calcula **solo** sobre el conjunto de entrenamiento y luego se aplica al de prueba.
- **Series Temporales: la excepción a la regla aleatoria**: si se trabaja con datos que dependen del tiempo (precios de acciones, clima, ventas mensuales), **nunca** se debe usar una división aleatoria. Si se mezclan datos de 2023 en el entrenamiento y de 2021 en la prueba, se estaría usando el futuro para predecir el pasado. En estos casos, se usa una división cronológica: se entrena con los años 1 al 4 y se prueba con el año 5.

## Filmina 40 — Resumen de Conceptos Clave

**Teoría completa (5. Resumen de conceptos clave, del docx)**:

- **Entrenar no es memorizar**: se buscan patrones generales, no ruido.
- **Partición 80/20**: es el estándar inicial para tener suficientes datos para aprender y suficientes para evaluar con rigor.
- **Detectar Overfitting**: si un modelo tiene un "100% de precisión" en entrenamiento pero falla en prueba, hay que sospechar inmediatamente de sobreajuste.
- **Reproducibilidad**: usar siempre `random_state` para que los experimentos sean consistentes.
- **Honestidad intelectual**: el conjunto de prueba es sagrado. No hay que ajustarlo ni "espiar" sus resultados hasta que el modelo esté listo para la evaluación final.

Dominar la separación de datos es lo que separa a un programador que usa herramientas de un verdadero Científico de Datos. Sin una evaluación honesta, cualquier modelo es solo una ilusión tecnológica.

## Filmina 41 — Práctica (no entregable): Evaluación Honesta

**Instrucciones completas (del docx)**: utilizar el dataset con el que se trabajó en las unidades anteriores de Scikit-Learn. El objetivo es transformar el script anterior de entrenamiento simple en un flujo de evaluación profesional. **Preparación del Entorno**: cargar el dataset limpio con Pandas, definir `X` (características) e `y` (target). **La Gran División**: usar `train_test_split` con `test_size=0.2` y `random_state` fijo (ej. 42) para reproducibilidad. **Entrenamiento**: crear una instancia de un modelo (regresor o clasificador simple según el dataset) y entrenarlo **únicamente** con `X_train`/`y_train`. **Evaluación Dual**: calcular el `.score()` del modelo sobre train y sobre test. **Análisis Crítico**: comparar ambos resultados, describir si hay Overfitting (puntaje alto en train, bajo en test) o Underfitting (puntaje bajo en ambos), y proponer una acción correctiva (simplificar el modelo, conseguir más datos).

**Errores comunes a evitar (del docx)**: fuga de datos (no entrenar el modelo con el dataset completo antes de evaluarlo); olvidar el `random_state` (si no se fija, cada vez que se envía el trabajo las métricas podrían variar, dificultando la revisión técnica).

---

# Tema 06 — Aplicaciones Prácticas de ML: del Modelo al Mundo Real (Filminas 42-50)

## Filmina 42 — División de Tema

**Teoría completa de apertura (del docx)**: llegados a la última unidad del módulo, ya se desarmó el motor del Machine Learning: la diferencia entre IA y Deep Learning, los tipos de aprendizaje, la arquitectura de Scikit-Learn, y la importancia crítica de evaluar los modelos sin hacer "trampa". Pero, ¿para qué sirve todo esto en la vida real? Un modelo de Machine Learning no es un fin en sí mismo; es una herramienta para resolver problemas que el software tradicional (basado en reglas fijas) simplemente no puede manejar.

## Filmina 43 — El Porqué del Machine Learning Aplicado

**Teoría completa (1. El Porqué del Machine Learning Aplicado, del docx)**: imaginá trabajar en el equipo de seguridad de un banco, con la tarea de escribir un programa para detectar correos que intentan robar contraseñas (phishing). Con programación tradicional, harían falta miles de reglas manuales: *"SI el correo contiene la palabra 'urgente' Y tiene un enlace sospechoso → MARCAR COMO PHISHING"*; *"SI el remitente es desconocido Y pide datos bancarios → MARCAR COMO PHISHING"*. **El problema**: los atacantes son creativos — mañana cambiarán "urgente" por "prioritario" o usarán una imagen en lugar de texto, y las reglas quedarían obsoletas en horas. Ahí es donde el Machine Learning aplicado brilla: en lugar de programar reglas, se alimenta al sistema con miles de ejemplos de correos reales (benignos y maliciosos), y el modelo aprende a identificar las señales sutiles —el "ruido" y los "patrones"— que un humano o una lista de reglas estáticas pasarían por alto. **ML como un "Filtro Inteligente"**: se pueden ver las aplicaciones de ML como filtros que, dada una entrada compleja (una imagen, un historial de compras, una señal de sensor), producen una salida útil (una categoría, un precio estimado, una alerta).

## Filmina 44 — Casos de Uso: ¿Quién Está Usando ML Hoy?

**Teoría completa (2. Casos de Uso, del docx)**: para entender el impacto del ML, ejemplos concretos de industrias transformadas por esta tecnología.

- **A. Sistemas de Recomendación — el "Efecto Netflix"**: Netflix, Spotify y Amazon son los reyes de este dominio, con un enfoque llamado **Filtro Colaborativo**. El problema: millones de productos y poco tiempo del usuario. La solución ML: el modelo analiza los patrones (qué se vio, qué se saltó, a qué hora se conecta) y los compara con millones de otros usuarios similares. Resultado: no solo se recomiendan "películas de acción", sino "películas de acción que le gustaron a personas que tienen gustos idénticos".
- **B. Detección de Fraude Bancario**: empresas como Mastercard o Visa procesan miles de transacciones por segundo. El problema: es imposible que un humano revise cada compra en tiempo real. La solución ML: modelos de **detección de anomalías**. El sistema conoce el "comportamiento normal" del usuario (dónde suele comprar, montos típicos); si aparece una compra de un reloj de lujo en otro continente, el modelo asigna una "puntuación de riesgo" alta y bloquea la transacción en milisegundos.
- **C. Logística y Movilidad — Uber y la Estimación de Tiempos**: cuando se pide un Uber y dice "llega en 4 minutos", hay un modelo de **Regresión** trabajando. El problema: el tráfico, el clima y los accidentes cambian constantemente. La solución ML: el modelo toma características (hora del día, datos históricos de la ruta, condiciones climáticas) para predecir un valor continuo: el tiempo de llegada (ETA).
- **D. Salud — Diagnóstico por Imagen**: en medicina, el ML ayuda a salvar vidas mediante visión por computadora. El problema: un radiólogo puede estar fatigado después de revisar 100 radiografías. La solución ML: modelos de **Clasificación de Imágenes** entrenados con millones de escaneos pueden resaltar áreas sospechosas de tumores con una precisión que iguala o supera a los expertos, funcionando como un "segundo par de ojos" incansable.

## Filmina 45 — Salud: Diagnóstico por Imagen

**Ampliación de la Filmina 44, punto D**: este caso merece su propio momento porque conecta con la advertencia de la Filmina 09 (pensar que la IA "entiende") — el modelo no "ve" un tumor como lo vería un médico. Detecta patrones estadísticos en píxeles que correlacionan con casos ya diagnosticados por expertos humanos. No reemplaza al radiólogo: funciona como apoyo para detecciones tempranas, en un contexto donde el costo de un error es altísimo.

## Filmina 46 — El Ciclo de Vida de un Proyecto de ML: el Pipeline

**Teoría completa (3. El Ciclo de Vida de un Proyecto de ML: El Pipeline, del docx)**: entrenar el modelo (lo que se hizo con Scikit-Learn en el Tema 04-05) es solo una pequeña parte del trabajo real. En el mundo profesional se sigue un flujo de trabajo o **pipeline**:

1. **Definición del Problema**: ¿qué se quiere predecir? ¿Es una clasificación (Sí/No) o una regresión (Número)?
2. **Recolección y Limpieza de Datos**: el paso más largo. Como se dice en la industria: *"Garbage in, garbage out"* (si entran datos basura, sale un modelo basura).
3. **Ingeniería de Características (Feature Engineering)**: elegir qué datos son relevantes. Para predecir el precio de una casa, el número de baños es clave; el color de la puerta, probablemente no.
4. **Entrenamiento y Evaluación**: acá se usa Scikit-Learn para ajustar el modelo y probarlo con datos que nunca ha visto (test set).
5. **Despliegue (Inferencia)**: se pone el modelo a trabajar en una aplicación real.
6. **Monitoreo**: los modelos pueden "deteriorarse" si el mundo cambia (por ejemplo, un modelo de predicción de ventas antes y después de una pandemia).

## Filmina 47 — Trampas y Errores Comunes: lo que Nadie te Dice

**Teoría completa (4. Trampas y Errores Comunes, del docx)**: incluso con los mejores datos, es fácil cometer errores conceptuales que arruinan una aplicación práctica.

- **Error 1: Confundir Correlación con Causalidad**. Un modelo de ML encuentra correlaciones. Si un modelo nota que "las personas que compran protector solar también compran helados", podría sugerir que el protector solar **causa** hambre de helado. Realidad: hay una variable oculta (el sol/verano). Lección: el modelo no entiende el "porqué", solo el "qué" — las decisiones de negocio deben ser validadas por humanos.
- **Error 2: El Sesgo en los Datos (Bias)**. Si se entrena un modelo de selección de personal usando solo currículums de personas contratadas en los últimos 20 años en una empresa que históricamente favoreció a hombres, el modelo aprenderá que "ser hombre" es una característica de éxito. Realidad: el modelo no es racista ni sexista por sí mismo; simplemente es un espejo de los datos que se le dieron.
- **Error 3: Sobreajuste (Overfitting)**. Un modelo que memoriza los datos de entrenamiento pero falla en la vida real es inútil — es como un estudiante que se memoriza las respuestas del examen pero no entiende la materia: si cambia un número en el examen, reprueba.

## Filmina 48 — Glosario para el Mundo Profesional

**Teoría completa (5. Glosario para el Mundo Profesional, del docx)**:

- **Modelo**: el "cerebro" que ya aprendió y está listo para decidir.
- **Inferencia**: el acto de usar el modelo para predecir algo nuevo (ej. cuando se sube una foto y Facebook sugiere etiquetas).
- **Features (Características)**: las columnas de entrada de los datos.
- **Labels (Etiquetas)**: la respuesta correcta que el modelo intenta aprender en aprendizaje supervisado.
- **Métricas**: los termómetros para saber si el modelo es bueno (Precisión, Recall, Error Cuadrático Medio).

## Filmina 49 — Síntesis y Cierre del Módulo

**Teoría completa (6. Síntesis y Cierre del Módulo, del docx)**: se completó un viaje desde la teoría de la IA hasta la mecánica del entrenamiento de modelos. El Machine Learning no es magia; es **estadística aplicada a gran escala**. La clave para ser un buen científico de datos no es conocer todos los algoritmos del mundo, sino saber: qué problema amerita usar ML (y cuál no); cómo preparar los datos para que el modelo aprenda patrones reales; cómo evaluar el éxito no solo con números, sino con impacto en el mundo real. El Machine Learning es una herramienta poderosa — hay que usarla para construir sistemas que no solo sean precisos, sino también éticos y transparentes.

## Filmina 50 — Práctica (no entregable): Diseño de una Solución de ML

**Instrucciones completas (del docx)**: **Identificar un problema** — pensar en el trabajo actual, un hobby o una empresa admirada, y qué proceso manual o repetitivo podría beneficiarse de una predicción o clasificación automática. **Definir la tarea**: ¿Clasificación (Categorías) o Regresión (Valores numéricos)? ¿Aprendizaje Supervisado o No Supervisado? **Proponer las Características (Features)**: enumerar al menos 5 datos (columnas) que el modelo necesitaría para aprender a tomar esa decisión. **Definir el éxito**: ¿cómo se sabría que el modelo funciona? Elegir una métrica (precisión, tiempo ahorrado, reducción de errores). **Considerar la ética**: ¿qué sesgos potenciales podrían existir en los datos que se recolectarían?

**Qué mirar al corregir (no está en el docx)**: esta práctica final retoma exactamente la misma estructura que la Pre-entrega del Tema 03 (Tarea, Features, Label, métrica, sesgo) — es una buena señal si el alumno ya la resuelve más rápido y con más soltura que la primera vez, porque significa que el vocabulario y el razonamiento quedaron incorporados.

## Filmina 51 (última) — ¿Dudas? ¿Consultas?

Cierre de la clase — espacio abierto antes de que el grupo se ponga a trabajar en la Pre-entrega.

---

## Guía del Notebook

**Estado actual**: el notebook completo ya está construido en [`Clase07.ipynb`](Clase07.ipynb) — Bloque 0 (repaso de Clase 06) + 4 Bloques prácticos que cubren los 6 Temas de esta guía, con horarios sugeridos de clase (0:00 a 1:55) y un solucionario para el docente al final. Dos datasets conviven en el notebook a propósito: `tasa-natalidad-deis-2000-2024.csv` para el repaso de estadística (Bloque 0, el mismo dataset ya conocido de Clase 06), y `propiedades_sueca_ml.csv` (precios de propiedades, ya limpio) como el dataset nuevo para entrenar el primer modelo real de la clase. El notebook viejo en `material/Clase_7_Fundamentos_de_Ciencia_de_Datos_1_.ipynb` (Pipelines + K-Means) queda obsoleto — no coincide con `Clase07.html` ni con `Clase 07.docx`.

### Bloque 0 — Repaso de la Clase 06 (Estadística y Preprocesamiento)

**Por qué arranca acá**: los seis temas de hoy dan por sentado que ya se sabe leer un dato con estadística — antes de que un algoritmo "aprenda" de un dataset, hace falta poder describirlo. Este bloque repasa, sobre el dataset real de natalidad del DEIS, los cuatro pilares de Clase 06:

1. **Limpieza e Integración**: `isnull().sum()`, `duplicated()`, y una variable derivada de negocio con `pd.cut()`.
2. **Tendencia Central y Dispersión**: `.mean()`, `.median()`, `.std()`, `.quantile()` e IQR — con una deducción real (la mediana de natalidad nacional es más alta que la media, por la caída sostenida a lo largo de los años, no por outliers).
3. **Distribuciones y Correlación**: `sns.histplot(kde=True)`, `.skew()`, `.corr()` + `sns.heatmap()` — comparando la natalidad de Buenos Aires, Córdoba y Santa Fe.
4. **Transformación y Reducción**: `StandardScaler` y `PCA`, cerrando con la regla de oro que se retoma formalmente en el Bloque 3 (Data Leakage): ajustar el escalador solo con train.

### Bloque 1 — El Mapa de la IA, ML y DL (Temas 01-02, 0:00-0:30)

Arranca con un "rompehielo": mostrar 5 filas del dataset de propiedades **sin** la columna `precio_eur`, para que el grupo note que sin etiqueta no hay forma de "adivinar" qué se está prediciendo — la misma idea de la Filmina 12 (la Señal de Aprendizaje), pero mostrada antes de nombrarla. Sigue con el mapa IA → ML → DL (matrioskas) y los tipos de aprendizaje, todo referido al mismo dataset de propiedades.

### Bloque 2 — Scikit-Learn por Dentro (Tema 04, 0:30-1:00)

Código real: `StandardScaler` como **Transformer** (`fit_transform` en train) y `LinearRegression` como **Estimator/Predictor**, entrenando sobre `superficie_m2`, `ambientes`, `antiguedad_anios` y `score_amenities` para predecir `precio_eur`. Muestra los atributos aprendidos (`modelo.coef_`, `modelo.intercept_`) e interpreta cada coeficiente en términos de negocio ("por cada m² adicional, el precio sube...").

### Bloque 3 — Entrenar y Evaluar sin Trampas (Tema 05, 1:10-1:35)

El bloque más denso, con tres demostraciones en código real:

- **R² y MAE** en train vs. test para la Regresión Lineal — como generaliza bien, sirve de punto de comparación ("baseline").
- **Overfitting en acción**: un `DecisionTreeRegressor` sin `max_depth` (memoriza el train, R² casi perfecto, pero falla en test) contra el mismo árbol con `max_depth=4` — el contraste numérico exacto entre "el estudiante que memoriza" y el "sweet spot", con una tabla comparativa de los 3 modelos.
- **La trampa del Data Leakage**: usar `precio_por_m2` (calculada a partir del propio `precio_eur`) como feature — el R² se dispara de forma sospechosa, y esa sospecha es justamente la señal de alarma que hay que aprender a reconocer.

### Bloque 4 — Consolidación Guiada (Temas 03 y 06, 1:35-1:55)

Cierre con el ciclo de vida completo de un proyecto de ML, y tres tareas guiadas para resolver en plenario: interpretar coeficientes en términos de negocio, diagnosticar overfitting/underfitting/sweet-spot a partir de números dados (3 casos), y proponer el próximo paso ante un modelo sobreajustado. Cierra con un **Solucionario** (uso docente) con las respuestas esperadas de las 4 tareas, para tener a mano mientras se conduce el plenario.

**Nota sobre la Pre-entrega**: el notebook menciona "en la pre-entrega de esta semana..." al cierre — es el mismo entregable "Aplicaciones Prácticas de ML" del Tema 03 de esta guía, no un ejercicio nuevo.

---

## Pre-entrega: "Aplicaciones Prácticas de ML"

✅ **Entregable evaluado del Módulo**, anunciado en la Filmina 20 (división del Tema 03). A diferencia de otras clases del curso, `Clase 07.docx` no incluye una sección separada con "Qué tenés que presentar / Criterios de Aceptación / Formato de entrega" para esta Pre-entrega — el contenido más cercano a una consigna es la práctica del Tema 03 (Filmina 28, "Perfilado de una Solución de Machine Learning"), marcada en el propio texto como el ejercicio que **precede** al entregable evaluado.

**Lo que sí está definido explícitamente en el docx**: el módulo completo (Temas 01 a 03) cierra con la nota — *"Entregable de este módulo: Pre-entrega — Aplicaciones Prácticas de ML (Del Algoritmo al Impacto Real), evaluable, suma al proyecto final"* — pero sin una rúbrica propia dentro de este documento.

**Nota para quien dicte la clase**: si existe una consigna formal de esta Pre-entrega en otro documento (una rúbrica separada, un enunciado en el campus), conviene traerla a esta guía para completar la sección — tal como están las fuentes disponibles hoy (docx + html), esto es todo lo que se puede documentar sin inventar criterios que no están en el material original.

---

## Síntesis y Conexión Final

La clase entera se puede resumir en una progresión: primero entendemos el mapa completo de la Inteligencia Artificial y dónde vive el Machine Learning dentro de él (Tema 01); después aprendemos a diagnosticar qué tipo de aprendizaje aplica a un problema según si hay o no una etiqueta (Tema 02); conectamos esa teoría con aplicaciones reales de alto impacto de negocio, en la Pre-entrega del módulo (Tema 03); entramos por primera vez al código con la arquitectura interna de Scikit-Learn (Tema 04); aprendemos a evaluar sin hacer trampa, con train/test y el diagnóstico de sobreajuste (Tema 05); y cerramos viendo cómo todo esto se integra en el ciclo de vida completo de un proyecto de ML real, de punta a punta (Tema 06).

En la próxima unidad se retoman estos mismos conceptos aplicados a modelos concretos — Regresión Lineal, Árboles de Decisión y Regresión Logística — construyendo directamente sobre el flujo `train_test_split` + Estimators/Predictors que hoy se vio por primera vez.
