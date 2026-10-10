# Clase 09: Aprendizaje No Supervisado — Guía Completa para el Docente

Esta guía es el **libreto de apoyo para dictar la Clase 09**. Reúne, en un solo lugar y con más profundidad de la que entra en una diapositiva, toda la teoría que aparece en:

- **`Clase 09_fixed.docx`** — el material teórico oficial de la unidad (reemplaza a `Clase 09_teoria.pdf`, que queda obsoleto).
- **`Clase09.html`** — las diapositivas que se proyectan en clase (36 filminas).

> **Estado de esta guía**: actualizada para seguir el nuevo `Clase 09_fixed.docx`. Se sacaron Reglas de Asociación (Apriori/FP-Growth) y la base matemática de PCA (covarianza/eigenvectores) porque el docx nuevo ya no los trae; se agregó Customer Profiling, Ética y Sesgos, y la Pre-entrega, que sí trae. El notebook práctico de la clase es `Clase09_aprendizaje no supervisado.ipynb` (no `Clase_9.ipynb`, que queda obsoleto) — documentado en el Anexo, al final de esta guía. Ese notebook sí conserva Clustering Jerárquico y t-SNE como contenido práctico adicional, aunque ya no tengan su propio módulo teórico en las filminas (ver nota al principio del Anexo).

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

**Ejemplos de Machine Learning que el grupo ya usa todos los días, sin saberlo:**
- **Filtro de spam de Gmail**: nadie escribió a mano la lista de todas las frases sospechosas — el modelo aprendió mirando millones de mails que los usuarios marcaron como spam.
- **Recomendaciones de Netflix / YouTube / Spotify**: "porque viste X, te puede gustar Y" no sale de una regla escrita por una persona, sino de patrones aprendidos sobre lo que miró o escuchó gente con gustos parecidos.
- **Desbloqueo facial del celular**: el teléfono aprendió a reconocer tu cara a partir de varias fotos tomadas en la configuración inicial, y la reconoce aunque cambie la luz, tengas anteojos o no te hayas afeitado.
- **Tiempo estimado de llegada en Google Maps / Uber**: el modelo aprende de millones de viajes anteriores cuánto se tarda en cada tramo según la hora, el día y el tráfico.
- **Autocorrector y texto predictivo del teclado**: aprende qué palabra suele venir después de otra mirando enormes cantidades de texto escrito.

Dentro de Machine Learning hay distintas formas de "dejar que la máquina aprenda sola", según qué tipo de datos y qué tipo de ayuda se le da durante ese aprendizaje. La que ya se vio en la Clase 08 —y la que se repasa a continuación— es el **aprendizaje supervisado**; la que arranca hoy es el **aprendizaje no supervisado**, con una diferencia central que se explica más abajo.

### ¿Qué es el aprendizaje supervisado? *(Filmina 02)*

La idea de fondo es muy parecida a cómo aprende una persona con ejemplos resueltos: si querés aprender a distinguir mails de spam, lo más fácil es que alguien te muestre miles de mails **ya marcados** como "spam" o "no spam", y con el tiempo empezás a notar patrones (ciertas palabras, remitentes raros, exceso de mayúsculas) que te ayudan a clasificar un mail nuevo que nunca viste. Eso es exactamente lo que hace un modelo de aprendizaje supervisado: se le muestran muchos ejemplos donde la respuesta correcta **ya se conoce**, y el modelo va ajustando sus parámetros internos hasta encontrar una regla (una función matemática) que relacione los datos de entrada con esa respuesta. Una vez entrenado, se usa esa regla para predecir la respuesta de casos **nuevos**, donde no se conoce de antemano.

Otros ejemplos del mismo mecanismo ("aprender de casos ya resueltos"):
- **Diagnóstico por imágenes**: miles de radiografías de tórax ya revisadas por médicos y marcadas como "neumonía" / "sana" → el modelo aprende a marcar radiografías nuevas.
- **Tasación de propiedades**: miles de casas ya vendidas, con su precio final conocido → el modelo estima el precio de una casa que recién sale a la venta.
- **Scoring crediticio**: el historial de clientes a los que el banco ya les prestó y se sabe si pagaron o no → el modelo estima el riesgo de un cliente nuevo que pide un préstamo.
- **Reconocimiento de dígitos escritos a mano**: miles de imágenes de números escritos por personas distintas, cada una con el dígito correcto anotado → el modelo lee códigos postales o cheques.

En la notación que se usa en la jerga de Machine Learning: a las variables de entrada (edad, ingresos, antigüedad laboral, cantidad de habitaciones de una casa...) se las llama `X`; a la respuesta que se quiere predecir (spam o no, precio de la casa) se la llama `y`. Entrenar un modelo supervisado es, ni más ni menos, buscar una función `f` tal que `f(X)` se parezca lo más posible a `y`, usando los ejemplos históricos donde ambas cosas ya se conocen.

| Problema | `X` (lo que se sabe de entrada) | `y` (lo que se quiere predecir) |
|---|---|---|
| Spam | Remitente, asunto, palabras del mail, cantidad de links | `spam` / `no spam` |
| Precio de una casa | Metros cuadrados, barrio, habitaciones, antigüedad | Precio en dólares |
| Préstamo bancario | Edad, ingresos, antigüedad laboral, deudas previas | `paga` / `no paga` |
| Demanda de un supermercado | Día de la semana, feriado sí/no, clima, promociones activas | Unidades vendidas de un producto |
| Abandono de clientes (*churn*) | Meses como cliente, reclamos, uso mensual del servicio | `se va` / `se queda` |

Existen dos grandes familias, según qué tipo de dato es `y`:

| | Clasificación | Regresión |
|---|:---:|:---:|
| **`y` es...** | Una categoría | Un número |
| **Ejemplo** | ¿El cliente paga el préstamo? | ¿Precio de la vivienda? |
| **Métricas** | Accuracy, F1, AUC-ROC | MAE, RMSE, R² |

- **Clasificación**: la respuesta que se quiere predecir es una **etiqueta**, elegida entre un grupo cerrado de opciones — "paga" o "no paga", "spam" o "no spam". No hay término medio: la predicción es una de esas categorías, no un número.
- **Regresión**: la respuesta que se quiere predecir es un **número** que puede tomar cualquier valor — el precio de una casa (podría ser $150.234 o $150.987, cualquier cifra), la temperatura de mañana. Acá sí hay término medio: el modelo puede acertar "más o menos", no es todo o nada.

**Más ejemplos de cada familia, para que el grupo practique distinguirlas:**

| Clasificación (`y` es una categoría) | Regresión (`y` es un número) |
|---|---|
| ¿Este mail es spam? (sí / no) | ¿Cuánto va a costar este departamento? |
| ¿Qué dígito (0-9) está escrito en esta imagen? | ¿Cuántos grados va a hacer mañana? |
| ¿Este tumor es maligno o benigno? | ¿Cuántas unidades de yerba se van a vender la semana que viene? |
| ¿Este cliente se va a dar de baja este mes? | ¿Cuántos minutos va a tardar este pedido de delivery? |
| ¿Este comentario es positivo, negativo o neutro? | ¿Cuánto va a gastar este cliente en los próximos 12 meses? |

Un truco para la clase: preguntar "¿la respuesta se puede promediar?". El promedio de dos precios ($100 y $200 → $150) tiene sentido — es regresión. El promedio de "spam" y "no spam" no tiene sentido — es clasificación.

**Cómo se mide si un modelo de clasificación es bueno** — con un ejemplo concreto: un banco evalúa el modelo sobre 100 clientes a los que ya les prestó dinero en el pasado, así que ya se sabe qué pasó realmente con cada uno (90 pagaron a tiempo, 10 no pagaron / entraron en mora).
- **Accuracy** es lo más simple de entender: de esos 100 casos, ¿en cuántos acertó el modelo (predijo "no paga" cuando efectivamente no pagó, o "paga" cuando efectivamente pagó)? Si acertó en 92, el Accuracy es 92%.
- El problema de quedarse solo con Accuracy: si el modelo fuera tan vago que dijera **siempre** "va a pagar", sin analizar nada, igual acertaría en los 90 clientes que sí pagaron y solo fallaría en los 10 que no — un Accuracy del 90%, que suena bien pero es un modelo completamente inútil para el banco (nunca detecta a un cliente riesgoso, que es justo el caso que importa detectar antes de prestarle plata).
  - La misma trampa aparece en cualquier problema donde un caso es mucho más raro que el otro:
    - **Fraude con tarjeta**: si el 99,8% de las transacciones son legítimas, un modelo que dice "nunca es fraude" tiene 99,8% de Accuracy y no detecta ni un solo fraude.
    - **Enfermedad poco frecuente**: si 1 de cada 1.000 pacientes la tiene, un modelo que dice "todos sanos" tiene 99,9% de Accuracy y no le sirve a ningún médico.
    - **Fallas de una máquina**: si una turbina falla 2 días al año, un modelo que dice "hoy no falla" acierta 363 de 365 días (99,5%) — y se pierde justo los 2 días que importaban.
- Por eso existe **F1**, que en realidad combina dos métricas más chicas y específicas. Sigamos con el ejemplo: supongamos que el modelo marca a **12 clientes** como "riesgosos" (predijo que no van a pagar). De esos 12, después se descubre que **8 realmente no pagaron** y **4 sí pagaron** (el modelo se equivocó con ellos). Y de los 10 clientes que en la realidad no pagaron, el modelo solo llegó a detectar a 8 de ellos (se le escaparon 2).
  - **Precision** ("precisión"): de los que el modelo marcó como riesgosos, ¿cuántos realmente lo eran? → 8 de 12 = **67%**. Si la Precision es baja, el modelo está siendo "alarmista": marca a mucha gente como riesgosa sin serlo (eso tiene un costo — por ejemplo, rechazarle el préstamo a un buen cliente).
  - **Recall** ("exhaustividad" o "sensibilidad"): de los que realmente no iban a pagar, ¿a cuántos detectó el modelo? → 8 de 10 = **80%**. Si el Recall es bajo, el modelo está siendo "distraído": deja pasar casos riesgosos de verdad sin detectarlos (ese es el error más caro para el banco — prestarle plata a alguien que no va a pagar).
  - **F1** es un promedio especial entre Precision y Recall (técnicamente se llama "media armónica", pero para la intuición alcanza con pensarlo como un promedio) que tiene una propiedad importante: si **cualquiera** de las dos (Precision o Recall) es mala, el F1 también sale malo — no alcanza con que una de las dos sea excelente para "tapar" a la otra. En este ejemplo, con Precision 67% y Recall 80%, el F1 da aproximadamente **73%**.
  - Comparado con el modelo "vago" de antes (el que siempre dice "va a pagar", sin marcar a nadie como riesgoso): ese modelo tiene Recall = 0% (no detecta ni un solo caso riesgoso real) — y ahí el F1 se derrumba a 0%, aunque su Accuracy fuera 90%. Ese es justamente el contraste que hace útil a F1: expone a los modelos que "hacen trampa" con Accuracy sin detectar nada de lo que realmente importa.
  - Nota sobre el nombre: a diferencia de AUC-ROC (que sí es una sigla con significado, ver abajo), "F1" no es la abreviatura de ninguna frase — es simplemente el nombre técnico de esta fórmula puntual (también se la llama "F1-score" o "F-measure"). No hace falta buscarle un significado oculto al nombre, solo recordar que combina Precision y Recall.
  - **¿Cuándo importa más cada una?** Depende de qué error sale más caro:

    | Situación | Métrica que más importa | Por qué |
    |---|---|---|
    | Filtro de spam | **Precision** | Mandar a spam un mail importante (una oferta de trabajo) es peor que dejar pasar un spam más. |
    | Detección de cáncer en un screening | **Recall** | Dejar pasar a un paciente enfermo es mucho más grave que pedirle un estudio extra a uno sano. |
    | Detección de fraude con tarjeta | **Recall** (con un piso de Precision) | Se quiere atrapar la mayoría de los fraudes, pero si se bloquean demasiadas tarjetas legítimas los clientes se enojan. |
    | Recomendación de un video en YouTube | **Precision** | Mostrar algo que no le interesa al usuario lo aburre; no mostrarle *todos* los videos que le gustarían no es grave. |
    | Búsqueda de sospechosos en un aeropuerto | **Recall** | Es preferible revisar de más que dejar pasar a alguien peligroso. |

- Una tercera métrica que se mencionó en la Clase 08 es **AUC-ROC** — acá sí conviene desglosar la sigla completa: **AUC** es *Area Under the Curve* (Área Bajo la Curva) de la **ROC**, que es *Receiver Operating Characteristic* (algo así como "Característica Operativa del Receptor" — un nombre que viene de la ingeniería de radares de mediados del siglo XX y que hoy no aporta ninguna intuición; no hace falta memorizar por qué se llama así, solo entender qué mide).

  Para entenderlo hay que retomar algo mencionado arriba: el modelo no dice "sí" o "no" directamente, calcula una **probabilidad** (por ejemplo, "este cliente tiene 75% de probabilidad de no pagar") y recién después esa probabilidad se convierte en una decisión final usando un **umbral** — por ejemplo, "lo marco como riesgoso si su probabilidad supera 50%". Pero ese umbral (el 50%) es una elección arbitraria: se podría usar 30% (el banco se vuelve más desconfiado, marca a más gente como riesgosa) o 70% (el banco se vuelve más permisivo, marca a menos gente).

  La **curva ROC** se construye probando **todos los umbrales posibles**, del 0% al 100%, y graficando en cada uno dos números uno contra el otro: cuántos clientes riesgosos de verdad logra detectar el modelo (el Recall de antes) contra cuántos clientes buenos termina marcando por error (la contracara de la Precision). El **AUC** es, literalmente, el área que queda debajo de esa curva — un único número que resume qué tan bien el modelo separa a los dos grupos (los que pagan de los que no), sin depender de qué umbral puntual se termine usando.

  Los dos valores de referencia para interpretarlo: **AUC = 1** sería un modelo perfecto — existe un umbral donde separa completamente a un grupo del otro, sin ningún error. **AUC = 0,5** es lo mismo que decidir tirando una moneda al aire — el modelo no tiene ninguna capacidad real de distinguir un cliente riesgoso de uno confiable, por más ajustes de umbral que se prueben. En la práctica, un AUC de 0,8-0,9 ya se considera bastante bueno para la mayoría de los problemas reales.

  Tres lecturas de ejemplo, para practicar en clase:
  - **AUC = 0,95** en un modelo de detección de fraude → si se toma al azar una transacción fraudulenta y una legítima, en el 95% de los casos el modelo le asigna más probabilidad de fraude a la fraudulenta. Excelente.
  - **AUC = 0,75** en un modelo que predice si un cliente va a abandonar una suscripción → separa bastante mejor que el azar, pero se le escapan muchos casos; útil para priorizar a quién llamar, no para tomar decisiones automáticas.
  - **AUC = 0,52** en un modelo que intenta predecir si una acción va a subir o bajar mañana → prácticamente una moneda al aire; el modelo no encontró ningún patrón real.

**Cómo se mide si un modelo de regresión es bueno** — con otro ejemplo: un modelo que predice precios de casas.
- **MAE** (Error Absoluto Medio): agarra la diferencia entre lo que predijo el modelo y el precio real de cada casa, y promedia esas diferencias (sin importar si se equivocó "de más" o "de menos"). Si el MAE da $10.000, quiere decir que, en promedio, el modelo se equivoca por $10.000 en cada predicción — un número fácil de interpretar porque está en la misma unidad (dólares) que lo que se está prediciendo.
- **RMSE**: muy parecido al MAE, pero antes de promediar los errores los eleva al cuadrado (y al final saca la raíz cuadrada del resultado). El efecto práctico: un error grande pesa mucho más que varios errores chicos — un modelo que casi siempre acierta bien pero se equivoca feo en un par de casas raras va a tener un RMSE bastante peor que su MAE, mientras que un modelo con errores parejos y moderados va a tener MAE y RMSE parecidos entre sí.
- **R²**: en vez de dar un error en dólares, da un número entre 0 y 1 (a veces se explica como porcentaje) que responde "¿qué tan bien el modelo explica por qué el precio de cada casa es el que es?". Un R² de 1 sería un modelo perfecto (acierta el precio exacto siempre); un R² de 0 significa que el modelo no es mejor que simplemente decir siempre "el precio promedio de todas las casas", sin mirar ninguna variable en particular.

**Las mismas métricas, en otros problemas** (para que no queden atadas solo al ejemplo de casas):
- **MAE**:
  - App de delivery que predice el tiempo de entrega: MAE = 6 minutos → "en promedio, el horario que le mostramos al cliente le erra por 6 minutos".
  - Pronóstico del clima: MAE = 1,8 °C → "en promedio, la temperatura pronosticada difiere 1,8 grados de la real".
  - Supermercado que predice ventas diarias de leche: MAE = 40 unidades → "en promedio, el pedido al proveedor queda corto o largo por 40 cartones".
- **RMSE vs. MAE**, con dos modelos que predicen el tiempo de entrega de 4 pedidos:
  - Modelo A, errores de 5, 5, 5 y 5 minutos → MAE = 5, RMSE = 5 (errores parejos, los dos números coinciden).
  - Modelo B, errores de 0, 0, 0 y 20 minutos → MAE = 5, RMSE = 10 (mismo MAE, pero el RMSE se dispara por el único error grande).
  - Modelo C, errores de 1, 2, 3 y 14 minutos → MAE = 5, RMSE ≈ 7,2 (en el medio).
  - Moraleja: si un cliente esperando 20 minutos de más es un desastre para el negocio, conviene mirar RMSE; si solo importa el error promedio, alcanza con MAE.
- **R²**:
  - R² = 0,92 en un modelo de precio de autos usados → el modelo explica el 92% de las diferencias de precio entre autos (año, kilometraje, marca hacen casi todo el trabajo).
  - R² = 0,45 en un modelo que predice el gasto mensual de un cliente → explica menos de la mitad; hay mucho comportamiento que las variables disponibles no capturan.
  - R² = 0,05 en un modelo que predice el resultado de un partido de fútbol por la diferencia de goles → prácticamente igual a decir "siempre el promedio".

### Los modelos que se vieron en la Clase 08 *(Filmina 03)*

Estos cinco modelos son las herramientas concretas con las que se resuelven los problemas de clasificación y regresión. Repasarlos uno por uno, con una idea intuitiva de cómo funciona cada uno:

- **Regresión Lineal**: el modelo más simple de todos — busca la "mejor línea recta" (o, con más de una variable de entrada, el mejor plano) que pase lo más cerca posible de todos los puntos de entrenamiento. Ejemplos: predecir el precio de una casa a partir de sus metros cuadrados — a más metros cuadrados, más precio, y la Regresión Lineal encuentra la relación numérica exacta ("cada metro cuadrado extra suma, en promedio, tantos dólares"); predecir las ventas de una heladería según la temperatura del día ("cada grado extra suma tantos helados"); estimar el consumo de nafta de un auto según su peso y cilindrada. Es un modelo de **regresión** (predice un número), muy fácil de interpretar, pero limitado cuando la relación entre las variables no es una línea recta.
- **Árbol de Decisión**: funciona como un juego de "20 preguntas" — va haciendo preguntas de sí/no sobre los datos ("¿el ingreso es mayor a $50.000?", "¿tiene más de 30 años?"), y según las respuestas va bajando por ramas del árbol hasta llegar a una predicción final en una "hoja". Se puede usar tanto para clasificación ("¿el cliente paga el préstamo o no?", "¿este paciente que llega a la guardia es urgente o puede esperar?") como para regresión ("¿cuánto va a gastar este cliente?", "¿cuántos días va a durar la internación de este paciente?"). Su gran ventaja es que es muy fácil de visualizar y explicar — literalmente se puede dibujar el árbol de preguntas y mostrárselo a alguien sin conocimientos técnicos.
- **Random Forest**: en vez de confiar en un único Árbol de Decisión (que puede memorizar demasiado los datos de entrenamiento y funcionar mal con datos nuevos), Random Forest entrena **muchos** árboles distintos — cada uno viendo una porción distinta, al azar, de los datos y de las variables — y después promedia (en regresión) o vota por mayoría (en clasificación) las predicciones de todos ellos. La idea es la misma que "preguntarle a un grupo de expertos en vez de a uno solo": el resultado grupal suele ser más confiable que el de un único árbol, porque los errores individuales de cada árbol tienden a cancelarse entre sí. Ejemplos de uso típicos: detección de fraude en transacciones, predicción de abandono de clientes en una telefónica, y estimación del rinde de un cultivo a partir de datos de suelo y clima.
- **Regresión Logística**: a pesar del nombre (que confunde a todo el mundo la primera vez), **no es un modelo de regresión sino de clasificación**. Se usa para predecir la probabilidad de que algo pertenezca a una categoría — por ejemplo, la probabilidad de que un cliente no pague un préstamo, entre 0% y 100% — y después esa probabilidad se convierte en una predicción final ("riesgoso" si la probabilidad supera 50%, por ejemplo). El nombre viene de que matemáticamente usa una función llamada "logística" para convertir un cálculo interno en un número entre 0 y 1. Otros ejemplos: la probabilidad de que un usuario haga clic en un anuncio, la probabilidad de que un paciente tenga diabetes según sus análisis, la probabilidad de que un alumno abandone el curso según su asistencia y entregas.
- **KNN (K-Nearest Neighbors, "K vecinos más cercanos")**: la idea más intuitiva de las cinco — para predecir la categoría (o el valor) de un caso nuevo, mira cuáles son los `K` casos **ya conocidos** más parecidos a él (los "vecinos más cercanos", midiendo distancia entre sus variables), y les copia la respuesta mayoritaria. Ejemplo: para adivinar si a alguien le va a gustar una película, KNN mira a los `K` usuarios con gustos más parecidos a los suyos, y se fija qué opinaron ellos de esa película. Otros dos ejemplos: estimar el precio de un departamento mirando el precio de los 5 departamentos más parecidos (mismo barrio, metros, ambientes) que se vendieron hace poco; o clasificar un vino como "bueno"/"regular" comparándolo con los vinos de composición química más parecida ya catados por expertos. No necesita "entrenarse" en el sentido tradicional — simplemente guarda todos los datos y compara en el momento de predecir.

### Buenas prácticas: evitar el Data Leakage *(Filmina 04)*

Uno de los errores más peligrosos (porque no siempre se nota) en Machine Learning es el ***Data Leakage*** ("fuga de datos"): que información del conjunto de **test** (los datos que se supone el modelo nunca vio, usados solo para evaluar qué tan bien predice) se "filtre" de alguna forma hacia el proceso de entrenamiento. Cuando eso pasa, el modelo parece funcionar excelente durante la evaluación, pero en la vida real (con datos genuinamente nuevos) rinde mucho peor — porque en el fondo "hizo trampa" viendo pistas que no debería haber visto.

Un ejemplo concreto de cómo ocurre sin querer: si se calcula el promedio y el desvío estándar de una columna usando **todo** el dataset (entrenamiento + test juntos) para escalar los datos, y **después** se separa en train/test, el modelo ya "vio" información estadística de los datos de test (su promedio, su dispersión) antes de ser evaluado con ellos. Es una fuga sutil, fácil de cometer sin darse cuenta.

Otros casos típicos de fuga de datos, para mostrar que no es solo un problema del escalado:
- **Imputar con todo el dataset**: rellenar los valores faltantes de "ingresos" con el promedio calculado sobre train + test juntos — mismo problema que el escalado, el promedio "ya vio" los datos de test.
- **Una variable que se conoce recién después del resultado**: para predecir si un cliente se va a dar de baja, usar la columna `fecha_de_baja` o `motivo_de_baja` — en el entrenamiento el modelo parece perfecto, pero en la vida real esa columna está vacía en el momento de predecir.
- **Filas duplicadas entre train y test**: si el mismo paciente aparece dos veces (dos consultas), y una cae en train y la otra en test, el modelo "reconoce" al paciente en vez de aprender el patrón.
- **Mezclar el tiempo en series temporales**: para predecir las ventas de diciembre, entrenar con datos de enero del año siguiente — el modelo usa el futuro para predecir el pasado.

Por eso la Clase 08 insistió en dos herramientas concretas para evitarla:

- **`StandardScaler`**: el nombre está compuesto de dos palabras en inglés — *"standard"* (estándar) y *"scaler"* (algo que escala, que cambia de tamaño/escala). Literalmente es "el escalador que lleva todo a una escala estándar". Y eso es exactamente lo que hace: reescala las variables numéricas para que todas queden en una escala comparable (en general, restando el promedio y dividiendo por el desvío estándar, de forma que la variable termine con promedio 0 y desvío 1 — esa combinación de promedio 0 y desvío 1 es, por convención estadística, "la escala estándar"). Es necesario porque muchos modelos (KNN es el caso más claro, ya que mide distancias) se ven distorsionados si una variable está en una escala mucho más grande que otra — por ejemplo, "ingresos" en miles de dólares vs. "edad" en años: sin escalar, la variable "ingresos" dominaría por completo cualquier cálculo de distancia o similitud, aunque "edad" fuera igual de importante para el problema. Otros pares típicos donde pasa lo mismo: "superficie en m²" (decenas o cientos) contra "cantidad de baños" (1 a 3) en un dataset de casas; "precio en pesos" (miles o millones) contra "calificación del producto" (1 a 5 estrellas) en un e-commerce; "pasos diarios" (miles) contra "horas de sueño" (5 a 9) en datos de un reloj inteligente.
- **`Pipeline`**: en inglés, *"pipeline"* es literalmente un **caño** o **tubería** — el mismo término que se usa para un oleoducto. La imagen mental es la de un líquido que entra por un extremo y va pasando por una serie de tramos conectados hasta salir transformado por el otro extremo; en informática se usa esa misma palabra para nombrar cualquier secuencia de pasos conectados, donde la salida de un paso es la entrada del siguiente. En scikit-learn, un `Pipeline` encadena todos los pasos (escalado, y después el modelo) en un único objeto — los datos "entran" por el escalador y "salen" ya transformados y clasificados/predichos, sin pasos sueltos en el medio. La ventaja concreta: cuando se usa `Pipeline` correctamente (ajustando el escalador **solo** con los datos de entrenamiento, nunca con los de test), es mucho más difícil cometer el error de fuga de datos por accidente — el `Pipeline` fuerza a que cada paso se aplique en el orden correcto, sin mezclar información de test dentro del entrenamiento.

**`train_test_split` con `stratify`**: el nombre de la función es literal en inglés — *"train"* (entrenar) + *"test"* (probar/evaluar) + *"split"* (dividir, partir en dos) — es, sin vueltas, "dividir en entrenamiento y prueba". Antes de entrenar cualquier modelo, se separa el dataset en dos partes usando esta función — una porción (típicamente 70-80%) para **entrenar** el modelo, y el resto para **evaluarlo** con datos que no vio durante el entrenamiento (simulando qué tan bien funcionaría con casos reales nuevos). El parámetro `stratify` viene de la palabra **estrato** (una capa o subgrupo dentro de una población) — en estadística, "muestreo estratificado" significa dividir a la población en subgrupos (estratos) y asegurarse de tomar una porción proporcional de **cada uno**, en vez de tomar una muestra completamente al azar que podría (por mala suerte) dejar algún subgrupo sub-representado. Acá los "estratos" son las categorías de `y`: si solo el 5% de los clientes del dataset no pagaron su préstamo, un split al azar (sin `stratify`) podría dejar casi ningún caso de impago en el conjunto de test, haciendo que la evaluación no sea representativa. `stratify=y` le asegura al split que mantenga la misma proporción de cada categoría (5% no paga / 95% paga) tanto en entrenamiento como en test. Otros casos donde `stratify` es casi obligatorio: detección de fraude (0,2% de transacciones fraudulentas — sin estratificar, el test podría quedar con 0 fraudes y no habría nada que evaluar); diagnóstico de una enfermedad rara (2% de positivos); clasificación de especies con un grupo minoritario (en un dataset de 1.000 aves donde una especie tiene solo 30 ejemplares).

### Validación: por qué un solo split no alcanza *(Filmina 04)*

Confiar en un único `train_test_split` tiene un problema: el resultado de la evaluación depende, en parte, de **qué** casos cayeron por azar en el conjunto de test — con otro split distinto (otros casos al azar), la métrica final podría salir un poco distinta, mejor o peor, sin que el modelo en sí haya cambiado. Para tener una medida más confiable y menos dependiente de la suerte del split, se usa la **validación cruzada** (*cross-validation*):

- **`StratifiedKFold`**: el nombre junta tres piezas — *"Stratified"* (estratificado, la misma idea de "muestra proporcional por subgrupo" que `stratify`), *"K"* (la cantidad de partes en las que se divide, un número que se elige — 5 y 10 son los valores más comunes) y *"Fold"* (en inglés, "pliegue" o "doblez" — como doblar una hoja de papel varias veces; cada doblez es una de las particiones del dataset, un "fold"). Entero, el nombre dice "dividir en K pliegues, de forma estratificada". En vez de partir el dataset en un solo par entrenamiento/test, lo divide en `K` partes iguales (folds) — por ejemplo, 5 partes. El proceso entrena y evalúa el modelo `K` veces distintas: en cada vuelta, usa una parte distinta como test y las `K-1` restantes como entrenamiento. Al final, se tienen `K` mediciones de la métrica elegida, no una sola. Que sea "Stratified" garantiza que cada uno de esos `K` folds mantenga la misma proporción de categorías que el dataset completo.
- **`cross_val_score`**: el nombre es la forma abreviada (típica en programación, para no escribir nombres kilométricos) de *"cross validation score"* — *"cross"* (cruzado/cruzada, en el sentido de que los folds se van intercambiando el rol de test), *"validation"* (validación, el proceso de comprobar qué tan bien funciona el modelo) y *"score"* (puntaje, el resultado numérico de esa validación). Es la función de scikit-learn que automatiza todo el proceso de `StratifiedKFold` — entrena y evalúa el modelo las `K` veces, y devuelve las `K` métricas resultantes, listas para promediar. En vez de reportar un único número ("el modelo tuvo 85% de Accuracy"), la buena práctica es reportar el promedio **y** la dispersión de esas `K` mediciones ("85% ± 3%") — un desvío chico entre folds indica que el modelo es estable y confiable; un desvío grande es una señal de alerta de que el resultado depende mucho de qué datos le tocaron, y que probablemente no generalice bien a casos nuevos.

  Tres resultados de ejemplo con 5 folds, para leer en clase:
  - Folds `[0.86, 0.84, 0.85, 0.87, 0.83]` → **85% ± 1,4%**: modelo estable, el resultado es confiable.
  - Folds `[0.95, 0.70, 0.88, 0.92, 0.80]` → **85% ± 9%**: mismo promedio, pero muy inestable — en algún fold le fue mal de verdad; conviene investigar por qué antes de confiar.
  - Folds `[0.99, 0.98, 0.99, 0.99, 0.98]` → **99% ± 0,5%**: estable pero "demasiado bueno para ser cierto" — en un problema real difícil, es una señal típica de Data Leakage que hay que revisar.

### Lo que cambia hoy

El aprendizaje no supervisado parte de datos **sin `y`** — sin una respuesta correcta conocida de antemano. El objetivo deja de ser predecir y pasa a ser **descubrir estructura**, por dos caminos distintos (cada uno se desarrolla en profundidad más adelante en esta guía, esto es solo la idea de arranque):

- **Clustering** (agrupamiento): armar grupos de observaciones parecidas entre sí, sin que nadie le diga de antemano cuáles son esos grupos ni cuántos hay — por ejemplo, agrupar clientes con comportamientos de compra similares, dejando que el propio algoritmo descubra los perfiles, en vez de definirlos a mano; agrupar noticias del día por tema sin tener una lista previa de temas; o agrupar alumnos por su forma de estudiar en una plataforma online.
- **Reducción de dimensionalidad**: cuando un dataset tiene muchísimas columnas (variables), resumir esa información en unas pocas "columnas nuevas" que capturan lo esencial, para poder analizarla o graficarla sin perder demasiado en el camino — por ejemplo, resumir las 30 materias de un plan de estudios en 2 ejes ("rendimiento general" y "perfil ciencias vs. humanidades"), resumir 50 preguntas de una encuesta en 3 factores, o resumir 100 indicadores económicos de cada país en 2 números para poder graficarlos en un mapa.

Una aplicación que combina ambas ideas y aparece una y otra vez en esta clase es la **detección de anomalías**: usar clustering (o la distancia a los grupos "normales") para encontrar los puntos que no se parecen a nada — el ejemplo típico es una transacción bancaria fraudulenta, que no encaja en ningún patrón de compra habitual; otros son un inicio de sesión desde un país donde el usuario nunca estuvo, o un sensor de temperatura de una heladera industrial que de golpe marca valores fuera de lo habitual.

Estos son los frentes que recorre el resto de esta clase, cada uno con su propio módulo.

### 👉 En Python — `Clase09_Bloque0_Repaso_Supervisado.ipynb`

Este repaso tiene, además de las filminas, su propio notebook corto — separado por ahora del notebook principal de la clase (`Clase09_aprendizaje no supervisado.ipynb`), para poder mostrarlo como un bloque de arranque independiente.

**Qué hace en general**: entrena un clasificador simple sobre el dataset de **Iris** (150 flores, 4 medidas, 3 especies ya conocidas), para que el grupo vea en código un ejemplo completo de Aprendizaje Supervisado antes de pasar a lo que no tiene etiqueta.

```python
iris = load_iris()
X = iris.data   # features: 4 medidas de cada flor
y = iris.target # label: la especie real (0, 1 o 2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
```python
modelo = LogisticRegression(max_iter=200)
modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)
accuracy = accuracy_score(y_test, predicciones)
```

**Línea por línea**: `load_iris()` trae el dataset ya cargado desde scikit-learn, sin necesidad de ningún archivo externo — `iris.data` son las 4 features (largo/ancho de sépalo y pétalo) y `iris.target` es la especie real de cada flor, ya codificada como 0/1/2. `train_test_split(..., stratify=y)` separa 80%/20% manteniendo la misma proporción de las 3 especies en ambos conjuntos — el mismo criterio de la Clase 08, para que el test sea representativo y no quede, por mala suerte, con muy pocos ejemplos de alguna especie. `LogisticRegression(max_iter=200)` instancia un clasificador (a pesar del nombre, es un modelo de **Clasificación**, no de Regresión); `.fit(X_train, y_train)` lo entrena mostrándole las flores de train **con** su especie real; `.predict(X_test)` genera una predicción para cada flor de test, que el modelo nunca vio; `accuracy_score(...)` compara esas predicciones contra las especies reales del test y devuelve el porcentaje de aciertos.

**Por qué Iris y no otro dataset**: no es una elección arbitraria — es el mismo dataset que después se reutiliza, **sin la columna de especie**, en el Módulo 4 (PCA) y en el notebook principal para t-SNE. La idea pedagógica es mostrar el mismo conjunto de flores dos veces en la misma clase: primero **con** la respuesta correcta a mano (acá, Supervisado — el modelo aprende a distinguir las 3 especies y acierta en la gran mayoría del test), y más adelante **sin** ella (No Supervisado — un algoritmo de reducción de dimensionalidad o de clustering tiene que encontrar esa misma estructura de 3 grupos por su cuenta, sin que nadie le diga cuántas especies hay ni cuáles son). Ver el mismo dataset resuelto de las dos formas, una al lado de la otra, es mucho más contundente que explicar la diferencia solo con palabras.

**Qué significa el número de accuracy que tira la celda**: Iris es un dataset fácil para un clasificador (las 3 especies están bastante bien separadas por sus medidas), así que es normal y esperable que el accuracy dé muy alto (por encima del 90%) — no es un logro excepcional del modelo, es una propiedad conocida de este dataset en particular. Vale la pena aclararlo en clase para que nadie se lleve la idea de que un 90%+ de accuracy es lo normal en cualquier problema real.

**Pendiente de decidir**: si este Bloque 0 se deja como notebook separado (como está ahora) o se pega al principio de `Clase09_aprendizaje no supervisado.ipynb` para que quede todo en un solo archivo — todavía no se definió.

---

## Módulo 1 — ¿Qué es el Aprendizaje No Supervisado?

### Apertura del módulo *(Filmina 05)*

Esta filmina es la divisoria que abre el Módulo 1 — el título en pantalla ("¿Qué es el Aprendizaje No Supervisado? — Definición, tipos de problemas, ejemplos industriales y flujo típico de trabajo") funciona como el "índice hablado" de los próximos 15-20 minutos de clase. Antes de avanzar a la Filmina 06, es el momento de instalar oralmente, en una frase cada una y sin apoyarte todavía en la próxima diapositiva, las dos definiciones generales que van a servir de ancla durante el resto de la clase:

- **Aprendizaje supervisado** (lo que se cerró en la Clase 08): a partir de datos históricos donde **cada** ejemplo trae una respuesta ya conocida (una etiqueta), el modelo aprende una función que relaciona las variables de entrada con esa respuesta, con el objetivo de predecir la respuesta de casos nuevos donde todavía no se conoce. Es aprender "con la solución del libro al lado".
- **Aprendizaje no supervisado** (lo que arranca ahora): a partir de datos donde **ningún** ejemplo trae una respuesta conocida, el modelo busca regularidades, agrupamientos o estructuras internas — no para predecir un valor puntual, sino para describir cómo están organizados los datos por sí mismos. Es aprender "sin la solución del libro", encontrando el patrón a fuerza de mirar los datos.

Una forma de presentar el contraste en clase, con un ejemplo cotidiano: un supervisado es como aprender a distinguir perros de gatos porque alguien te mostró miles de fotos ya etiquetadas "perro"/"gato"; un no supervisado es como que te den una pila de miles de fotos de animales sin ningún cartel, y tengas que agruparlas vos mismo por similitud, sin que nadie te haya dicho de antemano cuántos grupos hay ni cómo se llaman. El resultado del segundo ejercicio puede coincidir con "perros" y "gatos" — pero el algoritmo llegó ahí solo por semejanza visual, no porque alguien le haya enseñado esas categorías.

Más analogías del mismo contraste, por si la primera no termina de "caer" en el grupo:
- **Botones mezclados en una caja**: supervisado sería tener un cajón ya ordenado con carteles ("rojos", "grandes", "de 4 agujeros") y aprender a guardar cada botón nuevo en el cajón correcto; no supervisado es vaciar la caja sobre la mesa y armar montoncitos por parecido, decidiendo vos mismo el criterio sobre la marcha.
- **Llegar a una fiesta donde no conocés a nadie**: sin que nadie te lo explique, a los pocos minutos detectás "grupitos" (los compañeros de trabajo del anfitrión, la familia, los amigos del club) solo mirando quién habla con quién — eso es clustering.
- **Una biblioteca sin catálogo**: si te dan 5.000 libros sin clasificar, podés agruparlos por tema mirando su contenido, aunque nadie te haya dado la lista de secciones ("Historia", "Novela", "Cocina") de antemano.

Conviene remarcar en voz alta, antes de pasar a la Filmina 06, los cuatro bloques que anuncia esta diapositiva y que se van a recorrer en orden: (1) una definición formal de qué es el aprendizaje no supervisado, (2) los tipos de problemas que lo componen, (3) ejemplos concretos de la industria, y (4) el flujo de trabajo típico que se va a repetir, con variaciones, en cada módulo siguiente de la clase.

### Definición y diferencias con el aprendizaje supervisado *(Filmina 06)*

El aprendizaje no supervisado es un conjunto de técnicas de Machine Learning que buscan identificar estructuras, patrones o relaciones en datos que **no cuentan con etiquetas o respuestas conocidas**. A diferencia del aprendizaje supervisado (Módulo 0), donde el modelo aprende a partir de ejemplos con etiquetas, acá el objetivo es descubrir información oculta sin guía explícita.

**Para desarrollar antes de mostrar la tabla comparativa**: en la Clase 08 el flujo siempre fue el mismo — separar `X` (variables) de `y` (la respuesta a predecir), entrenar un modelo que aprenda esa relación, y medir qué tan bien predice sobre datos nuevos. Ese flujo depende por completo de que `y` exista y esté bien etiquetada — conseguir ese etiquetado en la vida real casi siempre implica un costo (alguien tuvo que revisar cada transacción y marcarla "fraude"/"no fraude", cada imagen y marcarla "gato"/"no gato", un radiólogo tuvo que mirar cada tomografía y anotar si había un tumor, un operador de call center tuvo que escuchar cada llamada y clasificar el motivo del reclamo). El aprendizaje no supervisado nace, en parte, como respuesta a ese costo: la enorme mayoría de los datos que genera cualquier empresa **no tienen etiqueta**, y etiquetarlos a mano no siempre es viable en tiempo o presupuesto. Estas técnicas permiten extraer valor de esos datos "tal como vienen", sin la etapa previa de etiquetado.

Otra forma de plantear la diferencia, útil para la clase: en el aprendizaje supervisado el científico de datos sabe de antemano **qué pregunta** está respondiendo el modelo ("¿es spam?", "¿cuánto va a costar?"). En el no supervisado, muchas veces ni siquiera se sabe con precisión qué se va a encontrar — el algoritmo puede revelar una segmentación de clientes que nadie había considerado, una relación entre productos que el equipo de marketing no había notado, o un grupo de sucursales que, sin estar cerca geográficamente, tienen exactamente el mismo patrón de ventas a lo largo de la semana. Por eso al aprendizaje no supervisado también se lo asocia con el **análisis exploratorio**: se usa tanto para resolver un problema puntual como para "conocer" un dataset nuevo antes de decidir qué hacer con él.

| Característica | Aprendizaje Supervisado | Aprendizaje No Supervisado |
|---|---|---|
| Datos de entrada | Con etiquetas o respuestas | Sin etiquetas |
| Objetivo | Predecir o clasificar | Encontrar patrones o estructuras |
| Ejemplos de problemas | Clasificación, regresión | Clustering, reducción de dimensionalidad, detección de anomalías |

**Un matiz que vale la pena mencionar en clase** (aunque se profundiza en cursos más avanzados): la frontera entre ambos mundos no siempre es absoluta. Existen enfoques intermedios — el aprendizaje **semi-supervisado** (una pequeña porción de datos etiquetados, mucha data sin etiquetar) y el aprendizaje **autosupervisado** (el propio dataset genera sus etiquetas, por ejemplo tapando parte de una imagen y pidiéndole al modelo que la reconstruya). No forman parte del temario de hoy, pero saber que existen ayuda a entender que "supervisado vs. no supervisado" es más un espectro que una dicotomía cerrada.

- Ejemplos de **semi-supervisado**: Google Fotos te pide que le pongas nombre a 3 o 4 fotos de una persona y después reconoce sola a esa persona en otras miles de fotos; un hospital con 200 radiografías diagnosticadas y 50.000 sin diagnosticar; un banco con unas pocas transacciones confirmadas como fraude y millones sin revisar.
- Ejemplos de **autosupervisado**: el texto predictivo que aprende a adivinar la palabra siguiente usando el propio texto como "respuesta"; un modelo que aprende a colorear fotos en blanco y negro usando fotos a color (se les saca el color y se le pide reconstruirlo); los grandes modelos de lenguaje, entrenados tapando palabras de un texto y pidiéndole al modelo que las adivine.

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
- **Analítica de negocios**: cuando un dashboard tiene 40 métricas y nadie sabe cuáles mirar primero, reducir dimensionalidad ayuda a identificar qué puñado de "meta-indicadores" resume la mayor parte de la variabilidad del negocio — un uso de PCA orientado a la comunicación con gerencia, no solo al preprocesamiento técnico. Otros dos usos en el mismo rubro: agrupar sucursales de una cadena por su patrón de ventas (en vez de por región geográfica) para definir metas comparables entre sucursales "parecidas", y detectar meses o días atípicos en las ventas que merecen una explicación antes de presentar un reporte.

Más sectores que vale la pena mencionar aunque no estén explícitos en la filmina:
- **Salud**: el clustering se usa para descubrir subtipos de una enfermedad (pacientes que responden de forma distinta a un mismo tratamiento) sin que existiera antes una clasificación clínica formal para esos subgrupos; también para agrupar hospitales por tipo de casos que atienden, y para detectar recetas o facturaciones anómalas a una obra social.
- **Telecomunicaciones**: agrupar antenas por patrón de tráfico horario para planificar mantenimiento, segmentar clientes por uso de datos/llamadas para diseñar planes, y detectar líneas con comportamiento de llamadas anómalo (fraude de SIM).
- **Agro**: agrupar lotes de un campo por características de suelo e imágenes satelitales para aplicar fertilizante de forma diferenciada, detectar zonas del cultivo con un comportamiento anómalo (posible plaga), y resumir decenas de variables climáticas en pocos índices.

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

**Los mismos 5 pasos, aplicados a tres casos distintos** (útil para mostrar que el flujo no cambia aunque cambie el problema):

| Paso | Supermercado: segmentar clientes | Banco: detectar transacciones raras | App de música: resumir gustos |
|---|---|---|---|
| 1. Preparación | Calcular por cliente frecuencia, ticket promedio y categorías compradas; escalar | Calcular por transacción monto, hora, país, comercio; escalar | Calcular por usuario % de escucha de cada uno de 80 géneros; escalar |
| 2. Método | K-Means (se buscan segmentos para campañas) | DBSCAN (interesa el ruido, no los grupos) | PCA (hay demasiadas columnas) |
| 3. Aplicación | Probar `k` entre 2 y 10 | Ajustar `eps` y `min_samples` | Elegir cuántas componentes conservar |
| 4. Evaluación | Codo + silhouette | Cantidad de puntos marcados como ruido, revisión manual de una muestra | Varianza explicada acumulada |
| 5. Interpretación | "Familias de compra grande semanal", "Compradores de último momento"... | Un analista de fraude revisa las transacciones marcadas | "Eje 1 = mainstream vs. nicho", "Eje 2 = música tranquila vs. enérgica" |

---

## Módulo 2 — K-Means y la Elección de k

**Contexto**: ¿cómo agrupar datos sin etiquetas? K-Means es el algoritmo más usado de clustering — divide un conjunto de datos en grupos naturales basándose en similitud.

### Apertura del módulo *(Filmina 10)*

La divisoria de este módulo trae el subtítulo "El algoritmo de clustering más usado, y cómo elegir bien su parámetro clave" — y es, en términos de duración, el módulo más largo de la clase (7 filminas), lo cual tiene sentido: K-Means es probablemente el algoritmo de aprendizaje no supervisado más usado en la industria, por su simplicidad conceptual y su bajo costo computacional.

**Para presentar antes del contenido técnico**: conviene retomar acá, en voz alta, la definición general de clustering del Módulo 1 ("agrupar datos similares en clusters") y anticipar que K-Means la resuelve con una idea muy visual: imaginar que cada cluster tiene un "centro de gravedad" (el centroide), y que cada punto del dataset "cae" naturalmente hacia el centro más cercano. Es una buena metáfora para instalar antes de entrar en el detalle algorítmico de la Filmina 11, porque todo el resto del módulo (los 4 pasos, los problemas de convergencia, la elección de k) gira alrededor de esa única idea: minimizar qué tan lejos está, en promedio, cada punto de su centro asignado.

Tres imágenes cotidianas de "cada punto va al centro más cercano", para elegir la que mejor funcione con el grupo:
- **Escuelas de un barrio**: cada chico se anota en la escuela que le queda más cerca de su casa; las escuelas son los centroides y los "radios escolares" son los clusters.
- **Antenas de celular**: tu teléfono se conecta automáticamente a la antena más cercana; el mapa de qué zona atiende cada antena es literalmente una partición tipo K-Means.
- **Sucursales de una cadena de pizzerías**: cada pedido lo despacha la sucursal más cercana; si la empresa pudiera mover sus sucursales al "centro" de sus clientes, estaría haciendo el paso de Actualización del algoritmo.

### Qué es y cómo funciona *(Filmina 11)*

K-Means es un **algoritmo de partición**: divide un conjunto de datos en `k` grupos (clusters) según la similitud de sus características. El objetivo es minimizar la suma de las distancias entre cada punto y el **centroide** (promedio) de su cluster asignado. Se apoya en las métricas de distancia (Euclidiana, Manhattan, Coseno) que ya se usaron en clases anteriores para definir "similitud".

**Para ampliar antes de mostrar la filmina**: el nombre completo del algoritmo, "K-Means" (K-Medias), ya describe su mecánica — la "K" es la cantidad de grupos a formar, y "Means" (medias) es literalmente cómo se calcula cada centroide: el promedio de todos los puntos que pertenecen a ese cluster en un momento dado. Formalmente, el algoritmo minimiza una función llamada **inercia** o **WCSS** (que se retoma en la Filmina 14): la suma, sobre todos los puntos, de la distancia al cuadrado entre cada punto y el centroide de su cluster. Elevar al cuadrado la distancia (en vez de usarla directa) tiene una razón matemática concreta: penaliza mucho más fuerte a los puntos lejanos que a los cercanos, lo que empuja al algoritmo a formar grupos compactos en vez de tolerar unos pocos puntos muy alejados de su centro.

Sobre las métricas de distancia: K-Means usa por defecto la distancia **Euclidiana** (la "línea recta" entre dos puntos, el teorema de Pitágoras aplicado a más de dos dimensiones) — es la que mejor encaja con la definición de centroide como promedio aritmético. Usar Manhattan (la suma de diferencias absolutas, como moverse en cuadras de una ciudad) o Coseno (el ángulo entre dos vectores, típico en texto) requeriría, estrictamente, variantes del algoritmo (K-Medoids es la alternativa más conocida cuando se necesita otra métrica de distancia).

Las tres distancias con números concretos, entre el punto A = (0, 0) y el punto B = (3, 4):
- **Euclidiana**: √(3² + 4²) = √25 = **5** → la distancia "a vuelo de pájaro", como mide un dron.
- **Manhattan**: |3| + |4| = **7** → la distancia caminando por cuadras: 3 cuadras para un lado y 4 para el otro, sin poder cruzar en diagonal.
- **Coseno**: mira el ángulo, no el largo. Los vectores (1, 1) y (10, 10) tienen distancia coseno **0** (apuntan exactamente para el mismo lado) aunque la Euclidiana entre ellos sea grande.

Y un ejemplo de dónde conviene cada una:
- **Euclidiana**: clientes descriptos por edad e ingreso (ya escalados) — las variables son continuas y "la línea recta" tiene sentido.
- **Manhattan**: repartos en una ciudad con calles en cuadrícula, o datos con muchas variables donde se quiere que un solo valor extremo no pese tanto (no se eleva al cuadrado).
- **Coseno**: textos — un mail de 100 palabras y uno de 1.000 que hablan del mismo tema tienen "perfiles" de palabras que apuntan en la misma dirección, aunque uno tenga diez veces más palabras.

### Los 4 pasos del algoritmo *(Filmina 12)*

1. **Inicialización**: se eligen `k` centroides iniciales — al azar o con **k-means++** para mejorar la convergencia.
2. **Asignación**: cada punto se asigna al cluster cuyo centroide esté más cerca (distancia Euclidiana, típicamente).
3. **Actualización**: se recalculan los centroides como el promedio de los puntos asignados a cada cluster.
4. **Repetición**: se repiten Asignación y Actualización hasta que las asignaciones no cambien o se alcance un número máximo de iteraciones.

**Desarrollo paso a paso, para acompañar la animación de la filmina en vivo:**

Este algoritmo también se conoce como **"Lloyd's algorithm"** en la literatura técnica, y es un buen ejemplo de un procedimiento **iterativo**: no calcula la respuesta de una vez, sino que la va refinando en rondas sucesivas, cada una un poco mejor que la anterior. Vale la pena remarcar en clase que los pasos 2 y 3 son, en esencia, un ciclo de "adivinar y corregir": el paso 2 (Asignación) responde "con los centroides que tengo ahora, ¿cuál es la mejor partición posible?"; el paso 3 (Actualización) responde "con esta partición, ¿cuáles son los mejores centroides posibles?". Cada ronda del ciclo garantiza matemáticamente que el WCSS total **nunca aumenta** — por eso el algoritmo siempre termina convergiendo (ver Filmina 13), aunque no siempre al mejor resultado posible.

Sobre la Inicialización: la opción "al azar" simplemente elige `k` puntos cualquiera del dataset como primeros centroides — es simple pero puede arrancar en una posición muy mala. **k-means++** (el default en la implementación de scikit-learn) es más inteligente: elige el primer centroide al azar, y cada centroide siguiente lo elige con una probabilidad proporcional a qué tan lejos está de los centroides ya elegidos — favoreciendo que los `k` puntos de arranque queden bien repartidos por el espacio de datos, en vez de agrupados por casualidad en una sola zona.

**Un ejemplo a mano, en una sola dimensión, para hacer en el pizarrón** — 6 clientes según su gasto mensual (en miles): `1, 2, 3, 10, 11, 12`, con `k = 2`:
1. Inicialización (mala a propósito): centroides en `1` y `2`.
2. Asignación: el `1` va al centroide 1; todos los demás (`2, 3, 10, 11, 12`) están más cerca del `2`.
3. Actualización: centroide 1 = promedio de `{1}` = `1`; centroide 2 = promedio de `{2, 3, 10, 11, 12}` = `7,6`.
4. Nueva asignación: `1, 2, 3` quedan más cerca del `1`; `10, 11, 12` más cerca del `7,6`. Nueva actualización: centroides `2` y `11`.
5. Otra vuelta: nadie cambia de grupo → el algoritmo terminó. Dos clusters: "gasto bajo" `{1, 2, 3}` y "gasto alto" `{10, 11, 12}`.

Para variar en clase, el mismo ejercicio funciona con otros datos de una dimensión: edades de los asistentes a un evento (`18, 20, 22, 60, 63, 65`), o tiempos de entrega de pedidos en minutos (`15, 18, 20, 55, 58, 62`).

### Convergencia, inicialización y problemas comunes *(Filmina 13)*

- K-Means **siempre converge**, pero a un **mínimo local**, no necesariamente al óptimo global.
- La inicialización de los centroides afecta la calidad y velocidad de convergencia; **k-means++** ayuda a elegir centroides iniciales más representativos, reduciendo la probabilidad de resultados pobres.
- **Outliers**: pueden distorsionar los centroides y afectar la agrupación.
- **Formas no esféricas**: K-Means asume clusters convexos y de tamaño similar; no funciona bien con formas arbitrarias.

**Para desarrollar cada punto con más profundidad:**

- **Mínimo local vs. global**: como el resultado final depende de dónde arrancaron los centroides, correr K-Means dos veces con inicializaciones distintas puede dar dos particiones **distintas**, ambas "válidas" en el sentido de que el algoritmo convergió correctamente en las dos, pero una puede ser mejor que la otra. La solución práctica que usa scikit-learn (y que aparece en el ejemplo de código de la Filmina 15, con el parámetro `n_init=10`) es correr el algoritmo completo varias veces con distintas inicializaciones al azar, y quedarse con el resultado que dio el WCSS más bajo de todos los intentos.
- **Sensibilidad a outliers**: como el centroide es un **promedio**, un solo punto muy alejado del resto puede "arrastrar" el centroide entero hacia él, distorsionando la posición de todo el cluster — el mismo fenómeno por el que la media aritmética es sensible a valores extremos (visto en clases anteriores de estadística descriptiva). Es una de las razones por las que suele convenir revisar y tratar outliers **antes** de correr K-Means, no después.
  - *Ejemplo concreto*: segmentando clientes por gasto mensual, un solo cliente corporativo que gasta 100 veces más que el resto puede correr el centroide de "clientes premium" tan lejos que termine agrupando mal a los clientes premium "reales" — conviene revisar outliers antes de clusterizar, no después.
  - *Ejemplo inmobiliario*: agrupando propiedades por precio y superficie, una sola mansión de 2.000 m² arrastra el centroide del grupo "casas grandes" y hace que casas de 200 m² terminen agrupadas con departamentos chicos.
  - *Ejemplo de sensores*: un sensor que por una falla registra `9999 °C` una sola vez puede llevarse un centroide entero a una zona donde no hay ningún dato real.
  - *Ejemplo de usuarios web*: un bot que visita 50.000 páginas por día, mezclado con usuarios humanos que visitan 20, deforma el centroide de "usuarios muy activos" — y de paso es justo el tipo de caso que DBSCAN dejaría como ruido.
- **Formas no esféricas**: como K-Means asigna cada punto según distancia al centroide más cercano, la "frontera" natural entre dos clusters siempre termina siendo una línea recta (o un plano, en más dimensiones) — geométricamente, solo puede separar bien grupos que tengan forma redondeada y tamaño parecido. Con clusters alargados, en forma de luna, o de tamaños muy distintos entre sí, K-Means directamente separa mal — y ese es exactamente el problema que resuelve DBSCAN, que se ve en el Módulo 3.
  - *Ejemplo concreto*: agrupar comercios por ubicación geográfica a lo largo de una costa o de un río da un cluster alargado y curvo — K-Means tiende a "cortarlo" en pedazos artificiales con fronteras rectas, en vez de respetar la forma real alargada de la zona.
  - *Ejemplo de anillos*: puntos dispuestos en dos círculos concéntricos (por ejemplo, locales en el centro de una ciudad y locales sobre una avenida de circunvalación) — K-Means los corta "como una pizza" en porciones, en vez de separar el anillo interior del exterior.
  - *Ejemplo de tamaños muy distintos*: 50 clientes corporativos frente a 10.000 clientes particulares — K-Means tiende a partir el grupo grande en varios pedazos y a "comerse" el grupo chico, porque busca clusters de tamaño parecido.

### Elegir k: método del codo (Elbow Method) *(Filmina 14)*

Para cada valor de `k` se calcula el **WCSS** (*Within-Cluster Sum of Squares*): la suma de las distancias al cuadrado entre cada punto y el centroide de su cluster. Un WCSS más bajo indica clusters más compactos.

Se grafica WCSS en función de `k` — la curva baja a medida que `k` crece, porque agrupar en más clusters siempre reduce la distancia interna. El objetivo es identificar el punto donde la tasa de disminución se frena notablemente, formando un **"codo"**: a partir de ahí, agregar más clusters no mejora significativamente la calidad de la agrupación. Balancea complejidad del modelo (muchos clusters) contra calidad de la agrupación (pocos clusters, cada uno con sentido) — evitando tanto el subajuste como el sobreajuste.

**Para ampliar antes de mostrar el gráfico**: vale la pena mencionar el caso extremo para que la lógica quede clara — si `k` fuera igual a la cantidad total de puntos del dataset, cada punto sería su propio cluster, y el WCSS daría exactamente `0` (cada punto coincide con su propio centroide). Ese extremo es matemáticamente "perfecto" pero completamente inútil para el negocio: no agrupa nada. El método del codo es, en el fondo, una forma visual de encontrar el compromiso entre ese extremo inútil (`k` = cantidad de puntos, WCSS = 0) y el otro extremo igual de inútil (`k` = 1, todo en un solo grupo, WCSS máximo). Conviene aclarar también que la ubicación del "codo" no siempre es tan clara como en el ejemplo de esta clase — en datasets reales, la curva a veces baja de forma más gradual, sin un quiebre visualmente obvio, y ahí es donde el coeficiente silhouette (Filmina 15) aporta una segunda opinión más cuantitativa.

Tres curvas de WCSS de ejemplo (k = 1 a 6), para practicar la lectura del codo:
- `1000, 400, 120, 100, 90, 82` → **codo clarísimo en k = 3**: de 2 a 3 la caída es enorme (280), de 3 a 4 casi nada (20).
- `1000, 750, 560, 420, 320, 250` → **sin codo visible**: la curva baja de forma pareja; probablemente los datos no tienen grupos bien marcados, y conviene mirar el silhouette o cuestionar si tiene sentido clusterizar.
- `1000, 300, 250, 120, 110, 105` → **dos codos posibles (k = 2 y k = 4)**: puede haber una estructura en dos niveles (2 grandes grupos, cada uno con 2 subgrupos); la decisión depende de qué nivel de detalle necesita el negocio.

### Elegir k: coeficiente silhouette *(Filmina 15)*

Para cada punto, compara su **cohesión** (distancia promedio a los demás puntos de su propio cluster) contra su **separación** (distancia promedio al cluster más cercano al que no pertenece). El resultado es un valor entre **-1 y 1**:

- Cerca de **1**: el punto está muy bien asignado a su cluster.
- Cerca de **-1**: el punto probablemente está mal asignado, y encajaría mejor en otro cluster.

Se calcula el promedio del coeficiente para todos los puntos, para cada `k` candidato, y se elige el `k` que **maximiza** ese promedio — el que da los clusters más definidos y separados. También sirve para detectar outliers: puntos con coeficiente cercano a -1 son candidatos a estar mal asignados.

**Para desarrollar el mecanismo con más detalle antes del ejemplo de código:**

El coeficiente silhouette de un punto se calcula, formalmente, como `(b - a) / max(a, b)`, donde `a` es la distancia promedio del punto a los demás puntos de **su propio** cluster (la cohesión — cuanto más chica, mejor) y `b` es la distancia promedio a los puntos del cluster **vecino más cercano** al que no pertenece (la separación — cuanto más grande, mejor). Un valor cercano a **0** (no solo los extremos -1 y 1) también es informativo: significa que el punto está prácticamente sobre el límite entre dos clusters, ni claramente adentro de uno ni del otro — una zona ambigua que suele señalar que, en esa región del espacio, tal vez `k` no está bien elegido.

Tres puntos de ejemplo, con la fórmula aplicada:
- `a = 1`, `b = 4` → (4 − 1) / 4 = **0,75**: el punto está mucho más cerca de su grupo que del vecino. Bien asignado.
- `a = 2`, `b = 2` → (2 − 2) / 2 = **0**: el punto está a la misma distancia de los dos grupos, justo en la frontera.
- `a = 3`, `b = 1` → (1 − 3) / 3 = **−0,67**: el punto está más cerca del grupo vecino que del suyo. Casi seguro está mal asignado.

Y tres lecturas del silhouette **promedio** de un modelo completo, como referencia orientativa:
- Mayor a ~0,7 → estructura fuerte, grupos bien separados (como el ejemplo de `make_blobs` de abajo, que da 0,78).
- Entre ~0,5 y 0,7 → estructura razonable, típica de datos reales "buenos" (el ejercicio de Mall Customers del Anexo cae acá).
- Menor a ~0,25 → estructura débil o artificial: los grupos se solapan mucho, o directamente no hay grupos reales.

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

**Sobre la sobresegmentación y subsegmentación**: en términos de negocio, la sobresegmentación tiene un costo operativo real — si marketing tiene que diseñar 15 campañas distintas para 15 microsegmentos de clientes, el costo de gestionar esa complejidad puede superar el beneficio de la personalización. La subsegmentación, en cambio, tiene un costo de oportunidad: agrupar en pocos clusters muy amplios puede esconder un segmento pequeño pero muy rentable dentro de un grupo más grande y menos interesante. No existe una regla matemática que resuelva esta tensión — el método del codo y el silhouette dan candidatos razonables de `k`, pero la decisión final casi siempre involucra también una restricción práctica del negocio (cuántos segmentos puede gestionar realmente el equipo de marketing, por ejemplo).

| | Sobresegmentación (`k` demasiado grande) | Subsegmentación (`k` demasiado chico) |
|---|---|---|
| **Supermercado** | 20 segmentos de clientes → 20 folletos distintos que nadie tiene tiempo de diseñar | 2 segmentos ("compra mucho" / "compra poco") → se pierde el grupo de clientes veganos que respondería muy bien a una promo específica |
| **Banco** | 15 perfiles de riesgo → los analistas no logran definir una política distinta para cada uno | 2 perfiles → los emprendedores jóvenes con ingresos variables quedan mezclados con los clientes de alto riesgo y se les niega crédito |
| **Escuela / plataforma educativa** | 12 grupos de alumnos → imposible armar 12 planes de refuerzo | 2 grupos ("aprueba" / "no aprueba") → no se distingue al alumno que no entiende el tema del que entiende pero no entrega |

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

Los tres tipos de punto, con tres analogías distintas:

| Analogía | Core point | Border point | Noise point |
|---|---|---|---|
| **Una ciudad** | Vecino del centro, rodeado de edificios por todos lados | Casa en el último barrio antes del campo: tiene vecinos de un lado, pero no alrededor | Una chacra aislada a 30 km de cualquier pueblo |
| **Una fiesta** | Persona en el medio de la ronda de baile | Persona parada en el borde de la ronda, mirando | Persona sola en la barra, sin hablar con ningún grupo |
| **Transacciones de una tarjeta** | Compras en el súper del barrio, de montos habituales, todas las semanas | Una compra en un comercio nuevo, pero de monto y horario parecidos a los habituales | Una compra de USD 3.000 a las 4 de la mañana en otro país |

Y tres ejemplos del efecto de los parámetros, usando pedidos de delivery en un mapa:
- `eps = 50 metros`, `min_samples = 5` → casi todo queda como ruido; solo aparecen como cluster dos o tres esquinas con muchísimos pedidos.
- `eps = 500 metros`, `min_samples = 5` → aparecen los barrios con mucha actividad como clusters separados, y los pedidos sueltos quedan como ruido. Probablemente el punto justo.
- `eps = 5 km`, `min_samples = 5` → toda la ciudad se fusiona en un único cluster gigante; el resultado no sirve para nada.

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

**Mini-ejercicio para la clase: ¿K-Means o DBSCAN?** (respuestas sugeridas entre paréntesis)
1. Una cadena de gimnasios quiere dividir a sus 20.000 socios en 4 grupos para diseñar 4 tipos de plan. (**K-Means**: el número de grupos ya está definido por el negocio, y todo socio tiene que caer en algún plan.)
2. Una empresa de logística quiere encontrar zonas de la ciudad donde se concentran los robos a camiones, sin saber cuántas hay. (**DBSCAN**: número de zonas desconocido, formas irregulares, y los robos aislados deberían quedar como ruido.)
3. Un banco quiere marcar transacciones sospechosas entre millones de compras normales. (**DBSCAN**: lo que interesa es justamente el ruido.)
4. Una tienda online con 5 millones de clientes quiere segmentarlos todas las noches de forma automática. (**K-Means**: con ese volumen, la velocidad pesa mucho.)
5. Un sismólogo quiere agrupar epicentros de terremotos que siguen la forma de una falla geológica alargada. (**DBSCAN**: los grupos tienen forma alargada, no esférica.)

---

## Módulo 4 — PCA: Reducción de Dimensionalidad

**Contexto**: ¿cómo simplificar un dataset con decenas o cientos de variables sin perder lo esencial? El Análisis de Componentes Principales (PCA) es la técnica fundamental para reducir dimensionalidad, facilitando la visualización y el análisis.

### Apertura del módulo *(Filmina 22)*

Esta divisoria anuncia el módulo de PCA: "simplificar datos complejos sin perder lo esencial: el arte de resumir". A diferencia de la versión anterior de esta guía (apoyada en el PDF viejo), el docx actual no pide desarrollar la matemática de covarianza ni eigenvectores/eigenvalores — se queda en la intuición geométrica, igual que los módulos de clustering.

**Para presentar antes del contenido técnico**: conviene arrancar retomando la Filmina 07 (Módulo 1), donde la reducción de dimensionalidad se definió como "simplificar datos complejos con muchas variables a representaciones más manejables". PCA es la técnica de referencia para resolver ese problema, y su lógica se puede resumir en una sola idea, sin fórmulas: encontrar las direcciones **nuevas** (no necesariamente las variables originales) a lo largo de las cuales los datos varían más — porque ahí es donde vive la mayor parte de la información. Es una buena analogía para instalar acá: PCA es como tomar una escultura en tres dimensiones y proyectar su sombra en una pared — si se elige bien el ángulo, esa sombra dice casi todo lo que hace falta saber de la escultura, pero de forma mucho más simple.

Otras tres analogías de la misma idea:
- **Sacarle una foto a un auto**: un auto es un objeto en 3D, pero una buena foto de costado (2D) alcanza para reconocer el modelo; una foto desde arriba, en cambio, pierde mucha información. PCA busca automáticamente "el mejor ángulo para la foto".
- **El promedio general de un alumno**: en vez de mirar las 12 notas de cada materia, el promedio resume todo en un solo número que ya dice mucho (quién rinde bien y quién no). Es, en esencia, una "primera componente principal" hecha a mano.
- **El índice de inflación**: el INDEC no publica el precio de cada uno de los cientos de productos de la canasta como titular — los resume en un solo número que captura hacia dónde se mueven todos juntos.

**Qué es cada Componente Principal, sin álgebra lineal**: la Primera Componente Principal (PC1) es la dirección donde los datos varían más; la Segunda Componente (PC2) es la segunda dirección con más variación, y es perpendicular a la primera. Por ejemplo, la PC1 podría explicar el 70% de la variación total de un dataset, la PC2 el 20%, y juntas el 90% — dos números nuevos que resumen casi toda la información de las variables originales.

Tres ejemplos de cómo podría verse esto en datasets reales:
- **Notas de 10 materias de un colegio**: PC1 ≈ 55% ("rendimiento general": sube en todas las materias a la vez), PC2 ≈ 15% ("perfil ciencias vs. letras": sube en Matemática/Física y baja en Lengua/Historia).
- **Medidas corporales (altura, peso, largo de brazos, talle de calzado, contorno de cintura)**: PC1 ≈ 80% ("tamaño general" de la persona — todas las medidas crecen juntas), PC2 ≈ 10% ("contextura": más ancha o más delgada para una misma altura).
- **Datos de 200 países (PBI per cápita, esperanza de vida, alfabetización, mortalidad infantil...)**: PC1 ≈ 65% ("nivel de desarrollo"), PC2 ≈ 12% (por ejemplo, "tamaño de la economía vs. calidad de vida").

### Varianza explicada y selección de componentes *(Filmina 23)*

Cada Componente Principal captura una porción de la "información total" (varianza) del dataset original. Esto ayuda a decidir cuántos componentes conservar:

- Conservar los primeros componentes que expliquen un porcentaje significativo (entre 70% y 95%) de la varianza acumulada.
- Un gráfico de codo (misma lógica que en K-Means) ayuda a ver dónde agregar más componentes deja de aportar varianza relevante.
- En algunos casos conviene priorizar **menos** componentes para simplificar el modelo, aunque se pierda algo de varianza — es una decisión de compromiso, no una regla fija.

**Para desarrollar antes de la filmina:**

Vale la pena remarcar el paralelismo explícito con el método del codo de K-Means (Módulo 2, Filmina 14): en los dos casos se grafica una curva (WCSS en un caso, varianza explicada acumulada en el otro) en función de un número entero que hay que elegir (`k` clusters, o cantidad de componentes), y en los dos casos se busca el punto donde agregar "una unidad más" deja de aportar una mejora proporcional. Es el mismo patrón de decisión — "¿cuánta complejidad adicional se justifica por la mejora que trae?" — aplicado a dos problemas distintos.

Esto permite, por ejemplo, pasar de 20 variables a solo 3, perdiendo muy poca información pero ganando muchísima claridad y velocidad.

Tres ejemplos de cómo leer la varianza acumulada para decidir:
- Acumulada `[0.62, 0.85, 0.93, 0.96, 0.98, ...]` → con **3 componentes** ya se pasa el 90%; es una elección natural.
- Acumulada `[0.20, 0.35, 0.47, 0.57, 0.65, 0.72, ...]` → la varianza está muy repartida; hacen falta muchas componentes para llegar al 90%, lo que sugiere que PCA no va a comprimir tanto en este dataset (las variables están poco correlacionadas entre sí).
- Acumulada `[0.97, 0.98, 0.99, ...]` → **una sola componente** explica casi todo; probablemente casi todas las variables miden "lo mismo" (por ejemplo, el mismo precio expresado en pesos, dólares y euros).

Si el objetivo final es solo **visualizar** los datos, casi siempre se usan exactamente 2 o 3 componentes, sin importar qué porcentaje de varianza expliquen — porque el límite ahí no es estadístico, es que un gráfico no puede tener más de 3 ejes.

### Limitaciones de PCA *(Filmina 24)*

- **Linealidad**: PCA solo captura relaciones **lineales** entre variables; con estructuras no lineales complejas, puede no ser suficiente.
- **Escalado**: es sensible a la escala de las variables — por eso es común normalizar o estandarizar los datos antes de aplicarlo (igual que en clustering).
- **Interpretabilidad**: las componentes principales son combinaciones lineales de las variables originales, lo que puede dificultar su interpretación directa frente a un público no técnico.

**Para ampliar cada limitación con más ejemplos:**

- **Linealidad**: el ejemplo clásico para ilustrar esta limitación en clase es un dataset con forma de espiral o de "S" en el espacio — PCA, al buscar solo direcciones **rectas** de máxima varianza, no puede "desenroscar" esa estructura y termina proyectando puntos que estaban lejos en la espiral original muy cerca entre sí en el resultado. Otros dos casos donde pasa lo mismo: dos círculos concéntricos (cualquier proyección recta los superpone, no hay "ángulo de sombra" que separe el anillo de adentro del de afuera); y la relación entre temperatura y consumo eléctrico de una ciudad, que tiene forma de "U" (se consume mucho con mucho frío por la calefacción y con mucho calor por el aire acondicionado) — PCA, que busca tendencias lineales, puede llegar a concluir que temperatura y consumo "no tienen relación". Para esos casos existen alternativas no lineales (t-SNE, UMAP, autoencoders) que quedan fuera del temario de hoy, pero vale la pena que quien pregunte sepa que existen.
- **Escalado**: si no se estandariza antes, una variable con valores en millones (como `market_value_eur` del dataset de la Clase 04) tendría una varianza numéricamente gigantesca comparada con una variable en unidades chicas (como `age`) — y como PCA busca **maximizar varianza**, terminaría armando la primera componente casi exclusivamente a partir de esa única variable de escala grande, ignorando de hecho a todas las demás. Es la misma razón por la que el escalado es obligatorio en K-Means y DBSCAN, aplicada acá a un problema distinto (varianza en vez de distancia). Otros dos ejemplos: en un dataset de casas, "precio en pesos" (millones) taparía por completo a "cantidad de ambientes" (1 a 5); en datos de un smartwatch, "pasos diarios" (miles) taparía a "horas de sueño" (5 a 9), y la PC1 sería en la práctica solo "cuánto caminó la persona". Un tercer caso, más sutil: la misma variable medida en distintas unidades cambia el resultado — la altura en milímetros pesa 1.000 veces más que en metros, aunque la información sea idéntica.
- **Interpretabilidad**: cuando la primera componente principal resulta ser, por ejemplo, `0.6 × ingresos + 0.5 × gasto_mensual - 0.3 × edad + ...`, explicarle a un directorio "qué es" esa componente en términos de negocio no es trivial — a diferencia de una variable original como "edad", que se entiende sin esfuerzo. Por eso, en contextos donde la explicabilidad ante un público no técnico es prioritaria, a veces se prefiere sacrificar algo de la reducción de dimensionalidad y quedarse con un subconjunto de variables originales, más fáciles de comunicar aunque menos eficientes matemáticamente.
  - Tres niveles de dificultad para interpretar una componente:
    - **Fácil**: en notas escolares, PC1 = `0.32 × Matemática + 0.31 × Lengua + 0.30 × Historia + ...` (todos los pesos parecidos y positivos) → se lee sin problema como "rendimiento general".
    - **Intermedia**: PC2 = `0.5 × Matemática + 0.4 × Física − 0.4 × Lengua − 0.5 × Historia` → con un poco de esfuerzo se lee como "perfil de ciencias vs. humanidades".
    - **Difícil**: PC3 = `0.4 × edad − 0.3 × cantidad_de_reclamos + 0.35 × uso_app_nocturno − 0.2 × antigüedad + ...` → mezcla variables sin relación evidente entre sí, y no hay un nombre de negocio honesto para ponerle. Ahí la interpretabilidad se pierde.

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

Tres situaciones donde el supuesto de "los datos se pueden agrupar con claridad" falla sin que el algoritmo avise:
- **Ingresos de una población**: suelen formar un continuo (de muy bajo a muy alto, sin saltos). Un K-Means con `k = 3` va a devolver igual "bajo / medio / alto", pero los cortes son arbitrarios — no hay tres grupos reales, hay una sola curva cortada en tres.
- **Edades de los clientes de un supermercado**: hay clientes de todas las edades repartidos de forma bastante pareja; pedirle 4 clusters solo por edad da 4 franjas etarias que no dicen nada nuevo.
- **Variables mal elegidas**: agrupar clientes por "número de DNI" y "código postal" produce clusters perfectamente válidos en lo matemático, pero que no significan nada para el negocio — la similitud entre puntos no es significativa.

Sobre PCA: además de asumir linealidad, asume que **más varianza significa más información relevante** — un supuesto razonable en la mayoría de los casos, pero que puede fallar si, por ejemplo, una variable tiene mucha varianza justamente por errores de medición (ruido de sensor) y no por señal real; en ese escenario, PCA podría terminar priorizando una dirección que en realidad es puro ruido. Otros dos ejemplos del mismo problema: en una encuesta, una pregunta mal redactada que cada persona entiende distinto genera respuestas muy dispersas (mucha varianza) que no reflejan ninguna opinión real; y en datos de ventas, una variable como "descuento aplicado" puede variar muchísimo por promociones puntuales sin decir nada del comportamiento de fondo del cliente, mientras que una variable de poca varianza (por ejemplo, "compró alguna vez un producto premium": casi todos dicen que no) puede ser justo la más valiosa para el negocio.

### Aplicaciones prácticas por escenario *(Filmina 29)*

- **Clustering**: segmentación de clientes, detección de fraude agrupando comportamientos atípicos, análisis de patrones en sensores industriales.
- **PCA**: visualización de datos complejos, reducción de ruido antes de un modelo supervisado, compresión de datos para almacenamiento eficiente.
- **Detección de anomalías**: fraude bancario, fallos de motores industriales — el algoritmo aprende el "comportamiento normal" y marca lo que no encaja.

En la práctica, la elección depende del contexto de negocio: en un e-commerce con datos ruidosos y clusters de forma compleja, DBSCAN suele ganarle a K-Means.

**Para cerrar con un caso integrador, combinando varias técnicas de la clase:**

Un flujo de trabajo realista en una empresa de e-commerce podría combinar **dos técnicas en una sola cadena de análisis**: primero, PCA para reducir docenas de variables de comportamiento de cada cliente (frecuencia de compra, categorías preferidas, monto gastado, dispositivo usado, horario de navegación...) a un puñado de componentes principales que resuman lo esencial; segundo, K-Means o DBSCAN sobre esas componentes reducidas para segmentar a los clientes en grupos con comportamientos similares (más rápido y con mejores resultados que clusterizar sobre las variables originales sin reducir, por la maldición de la dimensionalidad mencionada en el Módulo 4). Es un buen ejemplo para cerrar la clase mostrando que estas técnicas no compiten entre sí — se combinan.

Dos cadenas más, en otros rubros:
- **Banco — PCA + DBSCAN para fraude**: cada transacción tiene 40 variables (monto, hora, comercio, distancia al domicilio, tiempo desde la última compra...). PCA las reduce a 8 componentes; DBSCAN sobre esas 8 encuentra las zonas densas de "comportamiento normal", y lo que queda como ruido (`label = -1`) pasa a la cola de revisión del equipo de fraude.
- **Hospital — PCA + K-Means para pacientes crónicos**: cada paciente tiene 60 resultados de laboratorio. PCA los resume en 5 componentes; K-Means con `k = 4` arma grupos de pacientes con perfiles parecidos; un equipo médico revisa cada grupo y decide si alguno merece un protocolo de seguimiento distinto.
- **App de música — PCA para visualizar + K-Means para segmentar** (el mismo escenario de la Pre-entrega): 100 variables de escucha por usuario → PCA a 10 componentes → K-Means → PCA a 2 componentes solo para dibujar el gráfico de los grupos y mostrárselo a Marketing.

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

| Rubro | Variables de perfil (quién es) | Variables de comportamiento (qué hace) |
|---|---|---|
| **E-commerce** | Edad, género, provincia, dispositivo principal | Frecuencia de compra, ticket promedio, categorías visitadas, carritos abandonados |
| **Banco** | Edad, ocupación, nivel de ingresos declarado, antigüedad como cliente | Uso de tarjeta, transferencias por mes, uso de la app vs. sucursal, productos contratados |
| **Streaming de música** | Edad, país, tipo de plan (gratis/pago) | Horas de escucha por día, géneros, horario de escucha, playlists creadas, canciones salteadas |
| **Gimnasio** | Edad, barrio, objetivo declarado al inscribirse | Días de asistencia por semana, horario, clases grupales vs. sala de musculación |

Tres ejemplos de por qué hace falta combinar las dos familias:
- Dos mujeres de 35 años de Córdoba (mismo perfil) pueden ser una que compra todas las semanas y otra que compró una sola vez hace dos años — necesitan campañas completamente distintas.
- Un chico de 19 y un señor de 65 (perfiles opuestos) pueden escuchar exactamente los mismos géneros y en los mismos horarios — para recomendar música, se parecen más entre sí que a gente de su misma edad.
- Un cliente que compra mucho en la categoría "bebés" (comportamiento) se interpreta distinto si es un abuelo de 70 años (regalos) o una persona de 30 (padre o madre reciente) — el perfil le da contexto al comportamiento.

**Qué caracteriza a un buen segmento**: alta **cohesión** interna (los puntos del grupo se parecen entre sí) y alta **separación** respecto a los demás grupos — el mismo principio de calidad que ya apareció con el coeficiente silhouette (Módulo 2), ahora aplicado a la lectura de negocio, no solo al número.

### De clúster a decisión: un ejemplo completo *(Filmina 32)*

Un K-Means identifica un clúster con **alto gasto histórico** pero **sin compras en los últimos 6 meses**. El algoritmo no sabe qué significa eso — esa interpretación es 100% trabajo humano.

- **Interpretación de negocio**: "Clientes en Riesgo" — tuvieron valor real en el pasado, y el patrón sugiere que se están por ir.
- **Acción**: diseñar una campaña de reactivación con descuentos especiales dirigida específicamente a ese grupo.
- **Lo que NO hay que hacer**: ignorar el grupo asumiendo que "ya se fueron" (perder una oportunidad de negocio detectada), ni eliminar esos datos pensando que son un error (K-Means no garantiza que el comportamiento sea permanente — es una "foto" del estado actual).

**Tres ejemplos más del mismo recorrido "clúster → interpretación → acción"**, para que el grupo practique:

| Lo que muestra el clúster | Interpretación de negocio | Acción | Lo que NO hay que hacer |
|---|---|---|---|
| Compran solo cuando hay descuento, ticket bajo, muchas visitas a la sección "ofertas" | "Cazadores de ofertas" | Avisarles primero de las liquidaciones; no gastar en publicidad de productos a precio lleno con ellos | Asumir que son "malos clientes" — pueden ser muy fieles mientras haya promociones |
| Pocas compras pero de ticket altísimo, casi siempre en la categoría electrónica | "Compradores de alto valor ocasional" | Programa de garantía extendida y atención preferencial; recomendaciones de accesorios | Bombardearlos con mails semanales — compran poco por naturaleza, no por falta de estímulo |
| Usuarios de una app de streaming que escuchan solo de noche, siempre playlists de música tranquila | "Oyentes para dormir/relajarse" | Playlists automáticas de relajación, recordatorio nocturno | Interpretarlo como "usuarios poco activos" y ofrecerles un plan más barato — escuchan todos los días |

**Por qué la traducción importa tanto como el algoritmo**: un centroide es un promedio matemático; decir "el clúster 2 tiene gasto promedio de $500.000 mientras los demás promedian $50.000" es un dato. Decir "el clúster 2 es nuestro segmento Premium, y necesita un trato distinto" es la traducción a negocio que un algoritmo nunca va a hacer solo. Lo mismo vale para otros números: "el clúster 4 tiene 0,3 compras por mes y 18 meses de antigüedad promedio" es un dato; "el clúster 4 son clientes fieles pero de baja frecuencia: no los perdamos con cambios de precio bruscos" es una decisión. Y "el clúster 1 visita la app 9 veces por día pero nunca compra" es un dato; "el clúster 1 son curiosos que comparan precios: probemos mostrarles un cupón de primera compra" es una acción.

---

## Módulo 7 — Ética, Sesgos y Cierre

**Contexto**: el cierre de la clase, y el más importante en términos de responsabilidad profesional. Sin `y`, no hay una "verdad" contra la cual comparar — por eso toda la responsabilidad de interpretar bien recae en la persona, no en el algoritmo.

### Interpretación responsable: riesgos y sesgos *(Filmina 33)*

- **No hay Ground Truth**: el algoritmo encontrará patrones porque esa es su función — no valida si son reales, útiles o si esconden sesgos peligrosos. Que un K-Means encuentre 3 grupos no prueba que "existan" 3 tipos reales de clientes: si se le pide 10, va a dar 10. Lo mismo con DBSCAN: que marque 500 transacciones como ruido no prueba que sean fraude (pueden ser compras legítimas de un viaje); y con PCA: que la PC1 explique el 60% de la varianza no prueba que esa dirección sea la más importante para el negocio.
- **Proyectar prejuicios propios**: al no haber etiquetas, es muy fácil interpretar un clúster con el propio sesgo en vez de con el dato real detrás. Ejemplos:
  - Un clúster con mayoría de mujeres que compra en la categoría "hogar" se bautiza "amas de casa" — cuando el dato real solo dice "compran productos de hogar"; muchas pueden trabajar fuera de casa, y hay varones en el mismo grupo.
  - Un clúster de clientes mayores de 60 con poco uso de la app se bautiza "no saben usar tecnología" — cuando tal vez prefieren la sucursal por la atención personalizada.
  - Un clúster de usuarios jóvenes con muchos pagos atrasados se bautiza "irresponsables" — cuando el patrón puede explicarse por ingresos variables (trabajos temporales), algo que pide otro tipo de producto, no un castigo.
- **El riesgo legal y ético, no solo técnico**: si un clúster separa personas por un patrón que refleja una desigualdad social (por ejemplo, una zona geográfica correlacionada con nivel socioeconómico) y ese resultado se usa ciegamente para decidir a quién otorgar un crédito, hay un problema serio — el modelo no es "racista" ni "injusto" por sí mismo, simplemente es un espejo de los datos con los que se construyó, pero usarlo sin ese criterio tiene consecuencias reales. Otros ejemplos del mismo riesgo:
  - **Seguros**: segmentar asegurados y cobrarle más a un clúster que, en la práctica, coincide casi exactamente con un barrio de bajos ingresos.
  - **Selección de personal**: agrupar CVs por similitud con "los empleados exitosos actuales" — si históricamente la empresa contrató casi solo varones para un puesto, el clúster "perfil ideal" reproduce ese desequilibrio.
  - **Precios dinámicos**: mostrar precios más altos al clúster de usuarios que entra desde celulares caros o desde ciertas zonas, sin que nadie haya decidido explícitamente "cobrarle más a esa gente".

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

## Anexo — Apunte del Notebook Práctico (`Clase09_aprendizaje no supervisado.ipynb`)

**Qué notebook es este, y por qué no es `Clase_9.ipynb`**: la carpeta tiene dos notebooks con el mismo propósito. `Clase_9.ipynb` usa un dataset de fútbol (48 selecciones del Mundial) y todavía tiene la estructura vieja (incluye Reglas de Asociación, que ya no está en el docx). `Clase09_aprendizaje no supervisado.ipynb` es el que se usa de ahora en más: no tiene Reglas de Asociación, cubre exactamente K-Means, Jerárquico, DBSCAN, PCA (+ t-SNE de yapa) y cierra con un ejercicio guiado completo de segmentación de clientes — más alineado con el docx nuevo y, en general, más prolijo.

**Nota de limpieza ya aplicada**: el notebook tal como se armó originalmente tenía 48 celdas con algunas duplicadas y una celda fuera de lugar (la introducción de "Reducción de Dimensionalidad" aparecía en medio del bloque de código de DBSCAN). Ya se corrigió: se fusionaron las dos introducciones de DBSCAN en una sola, se reordenó la celda de Reducción de Dimensionalidad a su lugar correcto, se sacó una celda de t-SNE duplicada y más corta (quedó la versión más completa), y se sacó una recarga redundante del dataset Iris. El notebook quedó en 44 celdas, con un único hilo narrativo de principio a fin.

**⚠️ Desalineación pendiente de decidir, entre la teoría (filminas/README) y este notebook**: el docx nuevo ya no incluye Clustering Jerárquico ni t-SNE como temas propios — por eso no tienen Módulo ni Filmina en esta guía. Pero este notebook **sí** los sigue teniendo, con código y gráficos completos. Eso significa que, tal como está hoy, en algún momento de la clase vas a correr código de un algoritmo (Jerárquico) que el grupo nunca vio explicado en ninguna filmina — puede generar la pregunta lógica de "¿y esto cuándo lo vimos?". Tres salidas posibles, a decidir: **(a)** sacar esas celdas del notebook para que quede 100% alineado con el docx; **(b)** dejarlas pero presentarlas en vivo como contenido "bonus" fuera de programa, aclarándolo explícitamente antes de correrlas; **(c)** agregar de nuevo un mini-resumen teórico de Jerárquico/t-SNE en las filminas, aunque el docx no lo pida. Por ahora el notebook se dejó intacto (opción b, de hecho, sin haberlo aclarado todavía en ningún lado) — falta definir esto.

**Dataset usado en los ejemplos de K-Means/Jerárquico/DBSCAN**: no es un CSV real, sino datos **sintéticos** generados con NumPy — 240 "ciudades" ficticias con dos variables (temperatura promedio anual y humedad relativa media), armadas a propósito en 3 grupos bien diferenciados (tropicales, templadas, áridas) para que el resultado del clustering se pueda comparar contra la "verdad" que se usó para generarlos. Es una elección pedagógica deliberada: al ser datos inventados con grupos conocidos de antemano, se puede confirmar que el algoritmo "encontró lo que tenía que encontrar", algo que no se puede hacer tan fácil con datos reales (donde, precisamente, no se sabe de antemano cuántos grupos hay).

### Configuración Inicial

**Qué hace en general**: importa todas las librerías que se usan en el resto del notebook, de una sola vez.

```python
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import dendrogram, linkage
```

**Línea por línea**: Pandas/NumPy para datos; de `sklearn.cluster` los 3 algoritmos de clustering de la clase (`KMeans`, `AgglomerativeClustering` para el Jerárquico, `DBSCAN`); `StandardScaler` para escalar (obligatorio en todo lo que mide distancias) y `OneHotEncoder` por si hiciera falta codificar alguna variable categórica (no se termina usando en este notebook, queda importado por si se necesita); `PCA` y `TSNE` para reducción de dimensionalidad; `silhouette_score` para la métrica de validación de clusters; Matplotlib/Seaborn para gráficos; `dendrogram`/`linkage` de SciPy, específicas para dibujar el árbol del clustering jerárquico (`scikit-learn` no trae una función de dendrograma propia).

---

## Módulo 2 en código — K-Means

### Ejemplo ilustrativo: datos climáticos sintéticos

**Qué hace en general**: genera 240 "ciudades" ficticias repartidas en 3 grupos climáticos conocidos de antemano (tropical, templado, árido), y las grafica **sin mostrar a qué grupo pertenece cada una** — el mismo punto de partida que tendría K-Means en la vida real, donde no hay colores ni etiquetas previas.

```python
np.random.seed(42)
n = 80  # ciudades por grupo

tropicales = np.random.multivariate_normal([29, 78], [[4, 3], [3, 12]], n)
templadas  = np.random.multivariate_normal([14, 55], [[6, -2], [-2, 10]], n)
aridas     = np.random.multivariate_normal([26, 18], [[5, 1], [1, 8]], n)

X_clima = np.vstack([tropicales, templadas, aridas])
X_clima[:, 1] = np.clip(X_clima[:, 1], 0, 100)
```

**Línea por línea**: `np.random.multivariate_normal([media_temp, media_hum], matriz_covarianza, n)` genera 80 puntos al azar alrededor de un centro dado (ej. 29°C/78% para el grupo tropical), con una dispersión que define la matriz de covarianza — es la forma de "simular" un grupo real sin tener que salir a medir 80 ciudades de verdad. `np.vstack([...])` apila los 3 grupos de 80 en una única tabla de 240 filas. `np.clip(X_clima[:, 1], 0, 100)` recorta la columna de humedad para que ningún valor quede fuera del rango físicamente posible (0% a 100%), porque una distribución normal puede generar, por puro azar, algún valor absurdo como -3% o 105%.

**Por qué este dataset en particular, y no uno real desde el arranque**: al conocer de antemano los 3 grupos "verdaderos" (porque los generó el propio código), se puede comparar el resultado de K-Means contra esa verdad y confirmar que el algoritmo hizo bien su trabajo — algo que con un dataset real nunca se puede chequear con la misma certeza, porque ahí es K-Means quien *define* qué son los grupos.

### Escalado y aplicación de K-Means

**Qué hace en general**: estandariza las 2 variables (para que la escala de la temperatura, en decenas, no le gane de entrada a la escala de la humedad, en unidades porcentuales) y entrena K-Means con `k=3`.

```python
scaler_clima = StandardScaler()
X_clima_scaled = scaler_clima.fit_transform(X_clima)

kmeans_clima = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_clima.fit(X_clima_scaled)
y_kmeans_clima = kmeans_clima.predict(X_clima_scaled)

centers_clima = scaler_clima.inverse_transform(kmeans_clima.cluster_centers_)
```

**Línea por línea**: `StandardScaler().fit_transform(X_clima)` estandariza las 2 columnas (media 0, desvío 1) — el mismo criterio de la Filmina 14/Módulo 2 del repaso teórico. `KMeans(n_clusters=3, random_state=42, n_init=10)` instancia el modelo con `k=3` fijo (porque ya se sabe, en este ejemplo armado, que hay 3 grupos); `n_init=10` corre el algoritmo completo 10 veces con distintas inicializaciones al azar y se queda con la mejor (para esquivar el problema del "mínimo local" visto en la Filmina 13). `.fit(...)` entrena, `.predict(...)` asigna cada ciudad a un cluster (0, 1 o 2). `scaler_clima.inverse_transform(kmeans_clima.cluster_centers_)` es un paso sutil pero importante: los centroides que aprendió el modelo están en la escala **escalada** (media 0, desvío 1) — `inverse_transform` los devuelve a la escala original (°C y %), para que los números impresos tengan sentido real ("Cluster 0: Temperatura = 28.9°C") en vez de números abstractos como "-0.03".

### Gráfico de clusters y Método del Codo

**Qué hace en general**: dos piezas separadas — primero un gráfico de dispersión coloreado por cluster (con los centroides marcados con una X), después el cálculo del WCSS para `k` de 1 a 10, para ilustrar el método del codo con datos reales.

```python
for k in k_range:
    km_elbow = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_elbow.fit(X_clima_scaled)
    inertia.append(km_elbow.inertia_)
```

**Línea por línea**: el loop entrena un K-Means **distinto** para cada valor de `k` entre 1 y 10, y guarda el `.inertia_` (el WCSS) de cada uno en la lista `inertia` — exactamente el procedimiento descripto en la Filmina 14 (Módulo 2), ahora con números reales en vez de solo la explicación teórica. El resultado esperado: la curva cae fuerte hasta `k=3` y después se aplana — el "codo" coincide con los 3 climas reales que se usaron para generar los datos.

### Validación con Coeficiente Silhouette

**Qué hace en general**: repite el mismo barrido de `k` (esta vez de 2 a 10, porque Silhouette no se puede calcular con un solo cluster) y compara, en un gráfico de 2 paneles, el Codo contra el Silhouette — para ver si las dos métricas coinciden en el mismo `k` recomendado.

```python
for k in k_sil_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km.fit_predict(X_clima_scaled)
    silhouette_avgs.append(silhouette_score(X_clima_scaled, labels_k))
```

**Línea por línea**: `fit_predict(...)` entrena y asigna clusters en un solo paso; `silhouette_score(X, labels)` calcula el Silhouette promedio de **todo** el dataset para ese `k` puntual (no punto por punto, el promedio general). `np.argmax(silhouette_avgs)` identifica el `k` con el Silhouette más alto. El mensaje final del notebook ("cuando Codo y Silhouette coinciden en el mismo k → mayor confianza en la elección") es la bajada práctica de la Filmina 15: ninguna de las dos métricas es "la verdad absoluta" por sí sola, pero si ambas apuntan al mismo número, es una señal mucho más confiable que cualquiera de las dos por separado.

---

## Módulo 2 en código (continuación) — Clustering Jerárquico

### Dendrograma sobre una muestra

**Qué hace en general**: toma una submuestra de 60 ciudades (de las 240 totales) del mismo dataset climático, y construye un dendrograma con linkage `ward` — menos puntos que con K-Means, a propósito, para que el árbol sea legible.

```python
idx_sample = np.random.choice(len(X_clima), 60, replace=False)
X_demo_small = X_clima[idx_sample]

linked_clima = linkage(X_demo_small, method='ward')
```

**Línea por línea**: `np.random.choice(..., replace=False)` elige 60 índices al azar sin repetir, para tomar una submuestra representativa. `linkage(X_demo_small, method='ward')` calcula, paso a paso, qué par de clusters fusionar en cada nivel, usando el criterio Ward (minimizar el incremento de varianza interna al fusionar) — el resultado `linked_clima` es la estructura completa que describe todo el árbol de fusiones, lista para graficar con `dendrogram(...)`. El `color_threshold=25` del gráfico pinta de distinto color las ramas que quedan por debajo de esa altura, ayudando a ver a simple vista dónde "cortar" el árbol en 3 grupos.

### AgglomerativeClustering y comparación con K-Means

**Qué hace en general**: aplica el clustering jerárquico sobre **todas** las 240 ciudades (ya no solo la submuestra de 60), pidiéndole directamente 3 clusters, y compara cuantitativamente ese resultado contra el de K-Means.

```python
agg_clima = AgglomerativeClustering(n_clusters=3, linkage='ward')
y_agg_clima = agg_clima.fit_predict(X_clima)

ari = adjusted_rand_score(y_kmeans_clima, y_agg_clima)
```

**Línea por línea**: `AgglomerativeClustering(n_clusters=3, linkage='ward')` — a diferencia del dendrograma (que no necesita saber `k` de antemano), acá sí se le pide un número fijo de clusters, porque `.fit_predict()` necesita devolver una asignación concreta, no un árbol completo. `adjusted_rand_score(y_kmeans_clima, y_agg_clima)` es una métrica que compara dos particiones distintas del mismo dataset y dice qué tan parecidas son, **sin importar qué número de cluster le puso cada algoritmo a cada grupo** (K-Means podría llamar "Cluster 0" a lo que el Jerárquico llama "Cluster 2", y el ARI lo detecta igual). Un ARI cercano a 1.0 significa que, aunque usan lógicas internas distintas (centroides vs. fusiones), los dos algoritmos llegaron prácticamente al mismo resultado — un buen chequeo de que el agrupamiento encontrado es robusto, no un capricho de un solo algoritmo.

**Nota importante no mencionada en el notebook**: el Jerárquico corre acá sobre `X_clima` **sin escalar** (a diferencia de K-Means, que sí usó `X_clima_scaled`) — una inconsistencia menor del notebook original. En este dataset puntual el resultado no cambia demasiado porque ambas variables ya están en rangos parecidos, pero en un dataset real con escalas muy distintas, esto sí podría cambiar el resultado del Jerárquico — vale la pena mencionarlo en clase como ejemplo de un error común (Filmina 13/Módulo 2 del repaso teórico) que se puede colar incluso en un notebook ya armado.

---

## Módulo 3 en código — DBSCAN

### Generar datos con forma de "lunas"

**Qué hace en general**: genera un dataset con 2 grupos en forma de medialuna entrelazada — a propósito, porque son formas que K-Means (que asume grupos esféricos) no puede separar bien, y DBSCAN sí.

```python
X_moons, y_moons_true = make_moons(n_samples=250, noise=0.1, random_state=42)
X_moons_scaled = StandardScaler().fit_transform(X_moons)
```

**Línea por línea**: `make_moons(n_samples=250, noise=0.1)` es una función de scikit-learn pensada específicamente para generar este tipo de forma no convexa; `noise=0.1` agrega algo de dispersión aleatoria a cada punto, para que no sean dos líneas perfectas sino algo más parecido a datos reales.

### Aplicar DBSCAN y graficar

**Qué hace en general**: corre DBSCAN sobre las lunas, y separa visualmente los puntos que quedaron en algún cluster de los que quedaron marcados como ruido.

```python
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_dbscan = dbscan.fit_predict(X_moons_scaled)
```

**Línea por línea**: `DBSCAN(eps=0.3, min_samples=5)` fija los dos parámetros clave de la Filmina 19/Módulo 3 "a ojo" en este ejemplo (el notebook no corre acá un k-distance plot para estimarlo, a diferencia de lo que sugiere la teoría — queda como posible mejora). `.fit_predict(...)` devuelve, para cada punto, el número de cluster al que pertenece, o `-1` si quedó como ruido. En el gráfico, `noise_mask = y_dbscan == -1` separa los puntos de ruido (dibujados con una X roja) del resto (coloreados por cluster) — el resultado esperado es que las dos lunas queden separadas en 2 clusters bien definidos, algo que K-Means con `k=2` no lograría (tendería a cortar cada luna por la mitad).

---

## Módulo 4 en código — PCA y t-SNE

### Cargar y escalar Iris

**Qué hace en general**: carga el dataset de **Iris** (el mismo que se usó, con etiqueta, en el Bloque 0 de repaso de Supervisado) y lo escala, como paso previo obligatorio antes de aplicar PCA.

```python
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

X_iris_scaled = StandardScaler().fit_transform(X_iris)
```

**Línea por línea**: `load_iris()` trae las 4 features y la especie real (`y_iris`) de cada una de las 150 flores. Acá es importante notar algo: `y_iris` se carga igual, pero **no se usa para entrenar nada** en esta sección — queda disponible únicamente para, más adelante, colorear el gráfico y poder *verificar* si el agrupamiento que encuentra PCA (sin mirar la especie) coincide con la especie real. Es exactamente la continuidad pedagógica con el Bloque 0: ahí se usó `y_iris` para entrenar un clasificador; acá se la guarda aparte, como "respuesta correcta" para comparar después, pero el algoritmo en sí (PCA) nunca la ve.

### Aplicar PCA

**Qué hace en general**: reduce las 4 features originales a 2 Componentes Principales, y mide cuánta varianza conservan esas 2 componentes.

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_iris_scaled)

print(f"Varianza explicada por componente: {pca.explained_variance_ratio_}")
print(f"Varianza explicada acumulada: {np.sum(pca.explained_variance_ratio_)}")
```

**Línea por línea**: `PCA(n_components=2)` fija de antemano que se quieren solo 2 componentes (porque el objetivo acá es graficar en 2D, no explicar un umbral de varianza — la razón que la Filmina 23/Módulo 4 menciona como el caso típico donde "el límite no es estadístico, es que un gráfico no tiene más de 3 ejes"). `pca.fit_transform(X_iris_scaled)` aprende las direcciones de máxima varianza y proyecta los datos sobre ellas en el mismo paso. `pca.explained_variance_ratio_` es un array con el porcentaje de varianza que capturó cada componente — el resultado real da algo como `[0.73, 0.23]`, o sea que con solo 2 de las 4 variables originales ya se conserva más del 95% de la información total.

### Graficar PCA en 2D, coloreado por especie real

**Qué hace en general**: grafica las 150 flores en el plano de las 2 Componentes Principales, pero coloreando cada punto según su especie **real** (`y_iris`) — el primer momento del notebook donde se puede *ver* si la estructura que encontró PCA (sin mirar la especie) coincide con las 3 especies reales.

**Por qué esto es más una demostración que un ejercicio no supervisado "puro"**: en un escenario 100% no supervisado no se tendría `y_iris` para colorear — acá se la usa a propósito, solo para **validar** visualmente el resultado de PCA, no para entrenarlo. Vale la pena aclarar esta distinción en clase: el momento en que aparecen los 3 colores separados en el gráfico es la confirmación visual de que, incluso sin que nadie le dijera la especie, la reducción de dimensionalidad conservó la estructura que separa a las 3 especies — es el mismo "cierre de círculo" con el Bloque 0 de Supervisado mencionado más arriba.

### Aplicar t-SNE

**Qué hace en general**: reduce las mismas 4 features de Iris a 2 dimensiones, pero con una técnica **no lineal** (t-SNE) en vez de PCA, reutilizando `X_iris_scaled` ya calculado.

```python
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_iris_scaled)
```

**Línea por línea**: a diferencia de PCA, acá no hay un `.transform()` separado — t-SNE no aprende una transformación reusable, cada corrida calcula el mapa 2D desde cero para ese dataset puntual (por eso no tiene sentido "aplicar" un t-SNE ya entrenado a datos nuevos, a diferencia de PCA). No se fija `random_state`, así que correr esta celda dos veces puede dar mapas visualmente distintos (aunque la estructura de grupos que revele debería ser parecida) — vale la pena mencionarlo en vivo si alguien nota que el gráfico cambió al re-ejecutar.

---

## Ejercicio Práctico — Segmentación de Clientes (Mall Customers)

**Contexto del ejercicio**: a diferencia de los ejemplos anteriores (con datos sintéticos o el clásico Iris), acá se usa un dataset **real y público**: 200 clientes de un centro comercial, con edad, ingreso anual y un "Spending Score" (un puntaje de 1 a 100 que el propio shopping ya calculó para medir cuánto gasta cada cliente). Es el ejercicio más cercano, de todo el notebook, al Módulo 6 de Customer Profiling — termina pidiéndole explícitamente al alumno que le **ponga nombre** a cada cluster encontrado, la misma traducción de dato a negocio que se desarrolla en la teoría.

### Paso 1 — Cargar los datos

```python
url = 'https://raw.githubusercontent.com/erkansirin78/datasets/master/Mall_Customers.csv'
df = pd.read_csv(url)
```

**Línea por línea**: `pd.read_csv(url)` carga el CSV directamente desde una URL pública de GitHub — no hace falta tener el archivo descargado localmente. Columnas: `CustomerID` (no es una feature real, solo identificador), `Gender`, `Age`, `AnnualIncome`, `SpendingScore`.

### Pasos 2 y 3 — Exploración y distribución

**Qué hace en general**: antes de clusterizar nada, mira el tamaño del dataset, sus estadísticas descriptivas (`df.describe()`) y la distribución de cada variable numérica con 3 histogramas — el mismo primer paso de "entender los datos antes de tocarlos" que ya se vio en el Repaso de la Clase 08 y en la Clase 06/07 del curso.

### Paso 4 — Selección de features y escalado

**Qué hace en general**: de las 4 variables numéricas disponibles (`Age`, `AnnualIncome`, `SpendingScore`), elige deliberadamente solo 2 (`AnnualIncome` y `SpendingScore`) para poder graficar el resultado en un plano 2D simple, y las escala.

```python
X = df[['AnnualIncome', 'SpendingScore']].values
X_scaled = StandardScaler().fit_transform(X)
```

**Por qué solo 2 de las 4 variables**: el propio notebook lo aclara — simplicidad para visualizar, y el Spending Score ya es en sí mismo un resumen del comportamiento de compra. Vale la pena usarlo como gancho para mencionar en clase que, en un caso real con más variables, acá es exactamente donde entraría PCA (Módulo 4) antes de clusterizar, para no tener que elegir "a mano" solo 2 de muchas variables disponibles.

Tres variantes para proponer como ejercicio extra, cambiando solo la línea de `X = df[[...]]`:
- **`['Age', 'SpendingScore']`** → los grupos se leen por etapa de vida ("jóvenes que gastan mucho", "adultos mayores moderados"...).
- **`['Age', 'AnnualIncome']`** → los grupos dicen poco del comportamiento de compra; es un buen ejemplo de segmentar solo por variables de **perfil** (Módulo 6) y ver que la lectura de negocio queda pobre.
- **`['Age', 'AnnualIncome', 'SpendingScore']`** → ya no se puede graficar directo en 2D: es el momento natural para aplicar PCA a 2 componentes solo para dibujar el resultado.

### Pasos 5 y 6 — Método del codo y elección de K

**Qué hace en general**: corre el mismo barrido de `k` de 1 a 10 ya visto con los datos climáticos, pero ahora sobre los clientes reales — y, a diferencia del ejemplo anterior (donde `k=3` ya estaba decidido de antemano), acá el notebook le pide explícitamente al alumno que **mire el gráfico y decida** su propio valor de `K` antes de seguir.

```python
K = 5  # <-- CAMBIÁ ESTE VALOR si elegís otro K
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(X_scaled)
labels = kmeans.labels_
```

**Para el docente**: con este dataset en particular, el codo del gráfico suele verse bastante claro en `k=5` — y da pie a mostrar en vivo qué pasa si alguien elige un `k` distinto (por ejemplo `k=3` o `k=8`) y cómo cambian los clusters resultantes, conectando directo con la Filmina 14/15 (Módulo 2) sobre que no hay un único "k correcto", solo un rango razonable.

Qué esperar, a grandes rasgos, si en vivo se cambia `K`:
- **`K = 3`**: los clientes de ingreso alto quedan separados en "gastan mucho" y "gastan poco", pero los de ingreso bajo y medio se mezclan en un único grupo grande — se pierde la diferencia entre "Entusiastas" y "Bajo potencial" (subsegmentación).
- **`K = 5`**: aparecen los 5 perfiles clásicos (ver Paso 9) — cada grupo tiene una lectura de negocio clara.
- **`K = 8`**: algunos de los 5 grupos se parten en dos (por ejemplo, "VIP jóvenes" vs. "VIP un poco mayores" según la posición dentro del gráfico) — matemáticamente válido, pero difícil de justificar como segmentos distintos para Marketing (sobresegmentación).

### Pasos 7 y 8 — Visualizar y analizar cada cluster

**Qué hace en general**: grafica los 200 clientes coloreados por cluster (con los centroides marcados), y después arma una tabla con el promedio de edad/ingreso/gasto de cada cluster — la materia prima para poder interpretarlos.

```python
cluster_stats = df.groupby('Cluster').agg({
    'Cluster': 'count', 'Age': 'mean',
    'AnnualIncome': 'mean', 'SpendingScore': 'mean'
})
```

**Línea por línea**: `df.groupby('Cluster').agg({...})` agrupa las 200 filas por el cluster que les asignó K-Means (no agrupa por algo que el analista decidió a mano, sino por el resultado del algoritmo), y calcula el promedio de cada variable dentro de cada grupo — es, literalmente, el "centroide" expresado en términos de negocio en vez de en coordenadas abstractas.

### Paso 9 — Clasificar los clusters (la parte más importante)

**Qué hace en general**: le pide al alumno completar un diccionario poniéndole un **nombre de negocio** a cada cluster, basándose en las estadísticas del paso anterior — no hay una respuesta ya resuelta en el código, es deliberadamente una celda para completar en vivo.

```python
clasificacion = {
    0: ('PONÉ TU NOMBRE AQUÍ', 'Escribí por qué elegiste este nombre...'),
    ...
}
```

**Por qué esta celda es la bisagra pedagógica de todo el ejercicio**: es la puesta en práctica exacta del Módulo 6 (Customer Profiling) — el algoritmo entregó 5 grupos con sus promedios de ingreso/gasto/edad, pero **nunca** dice "este es el segmento Premium" o "estos son los Conservadores" — ese paso de ponerle nombre, decidir qué hacer con cada grupo, es 100% criterio humano. Vale la pena, en vivo, pedirle al grupo que proponga nombres para los 5 clusters mirando la tabla de estadísticas del paso anterior, antes de mostrar cualquier respuesta sugerida.

**Respuesta orientativa para el docente** (con `K=5`, los números van a variar levemente según la semilla y el hardware, pero la lógica general se mantiene): un cluster de **ingreso alto + gasto alto** → "Clientes VIP/Premium" (el segmento más rentable, foco de retención); uno de **ingreso alto + gasto bajo** → "Clientes Conservadores de Alto Poder Adquisitivo" (tienen plata pero no la gastan ahí — una oportunidad de marketing específica); uno de **ingreso bajo + gasto alto** → "Clientes Entusiastas" (gastan por encima de lo que su ingreso sugeriría — cuidado con ofrecerles crédito sin análisis adicional); uno de **ingreso bajo + gasto bajo** → "Clientes de Bajo Potencial"; y uno de **ingreso medio + gasto medio** → "Clientes Estándar" (el grueso de la base, sin un patrón extremo en ningún sentido).

### Paso 10 — Reflexión final

Cierra el notebook sin código nuevo, con una síntesis de los 5 pasos recorridos y una pregunta abierta para el grupo ("¿cómo cambiarían las clasificaciones si eligieras otro K?") — buen gancho para conectar con la Pre-entrega del Módulo 7, que le pide al alumno hacer exactamente este mismo ejercicio de interpretación, pero sobre un dataset distinto (usuarios de una app de streaming) y de forma completamente escrita, sin código.

---

## Pendiente de documentar

El notebook de repaso de Supervisado (`Clase09_Bloque0_Repaso_Supervisado.ipynb`) ya está documentado más arriba, en la sección "👉 En Python" del Módulo 0. La demo de "PCA mejorando un modelo real" (dataset de cáncer de mama + 300 columnas de ruido + KNN) que describe la Filmina 30 (Módulo 5) **todavía no tiene celda propia** en ninguno de los dos notebooks — por ahora solo está documentada en el README como texto, igual que viene del docx. Si se decide correrla en vivo, habría que agregarla como notebook nuevo o como bloque adicional.
