# Clase 08 — Aprendizaje Supervisado: Fundamentos, Modelos y Evaluación

**Curso de Data Science I · Clase 08** — de enseñar con ejemplos etiquetados a comparar modelos con F1-Score y curvas ROC.

Esta guía sigue el **orden exacto de las 48 filminas** de `Clase08.html`, organizadas en 6 módulos. No es un resumen de lo que ya dice cada filmina — la idea es que la filmina sea el disparador visual y el texto de acá sea el material **adicional** para decir en voz alta: ejemplos numéricos, matices técnicos que no entran en una diapositiva, preguntas para tirarle al grupo y conexiones con otras clases. Al final hay una guía del notebook real de la clase (`Clase_8_Fundamentos_Data_Science_1.ipynb`) y el detalle de la Pre-entrega evaluada.

> **Contexto de la clase anterior**: Clase 07 ya tocó Supervisado vs. No Supervisado (de forma conceptual, con K-Means) y ya introdujo Pipelines y K-Fold. Hoy formalizamos y profundizamos esos tres conceptos — no arrancan de cero.

---

## Objetivos de la clase

1. Distinguir Clasificación de Regresión, y Aprendizaje Supervisado de No Supervisado.
2. Interpretar coeficientes de una Regresión Lineal con sentido de negocio ("manteniendo todo lo demás constante").
3. Entender cómo un Árbol de Decisión decide sus divisiones, y por qué Random Forest lo hace más estable.
4. Aplicar Regresión Logística y KNN a un problema de clasificación, con el escalado correcto.
5. Reconocer y evitar Overfitting, Data Leakage, Concept Drift y el problema de las clases desbalanceadas.
6. Completar la Pre-entrega "Evaluación y Comparación del Pipeline": comparar 2 modelos de clasificación con Matriz de Confusión, F1-Score y Curva ROC.

---

## Filmina 01 — Portada

Apertura de la clase. El subtítulo ya anticipa el arco del día: arrancamos enseñando con ejemplos (Módulo 01) y terminamos comparando modelos con las mismas métricas que va a pedir la Pre-entrega (Módulo 06).

---

# Módulo 01 — Fundamentos del Aprendizaje Supervisado (Filminas 02-10)

## Filmina 02 — División de Módulo

Divisor de sección. El pilar sobre el que se construye la mayoría de las soluciones de IA que usamos hoy — de los filtros de spam a los sistemas de diagnóstico médico.

## Filmina 03 — ¿Qué es el Aprendizaje Supervisado?

**Ejemplo numérico para el pizarrón (no está en la filmina)**: una inmobiliaria quiere predecir el precio de venta. Una fila de datos sería X = [120 m², 3 habitaciones, "Palermo"], y = \$250.000.000. El dataset de entrenamiento tiene miles de filas así — la función f que el modelo busca es la que, dado un X nuevo (una casa que se acaba de publicar), estima un y creíble.

**Pregunta para tirar a la clase**: ¿el tasador de arte de la filmina "sabe" explícitamente qué pincelada indica falsificación? — No, y ese es el punto: el modelo encuentra el patrón solo, a partir de los ejemplos. Nadie le programa reglas de pinceladas a mano.

## Filmina 04 — ¿Por Qué se Llama "Supervisado"?

**El nombre técnico del "ajuste de parámetros" (no está en la filmina)**: ese ciclo de predecir → comparar con la verdad → ajustar es, en la mayoría de los algoritmos modernos (Regresión Logística, Redes Neuronales), literalmente **descenso de gradiente** (*gradient descent*) — el algoritmo calcula en qué dirección debe mover cada parámetro para reducir el error, y da un pequeño paso en esa dirección, miles de veces. Vale la pena nombrarlo así de una vez, porque va a reaparecer en unidades futuras con ese nombre técnico.

## Filmina 05 — Clasificación vs. Regresión: ¿Qué Predecimos?

**Un caso ambiguo, que no está en la filmina, y que genera buena discusión**: la cantidad de estrellas de una reseña (1 a 5). Es técnicamente un número discreto y ordenado — ¿Clasificación Multiclase o Regresión? En la práctica se usa de las dos formas: como Clasificación si solo importa acertar la categoría exacta, o como Regresión si importa que un error de "predije 4, era 5" sea menos grave que "predije 1, era 5". No hay una respuesta única — depende de qué error le cuesta más caro al negocio.

## Filmina 06 — El Pipeline (Flujo de Trabajo) en 4 Pasos

**Un tercer set que no está en la filmina, y que se usa en proyectos serios**: además de train y test, muchos flujos reservan un **Validation Set** — se usa para probar distintas configuraciones del modelo (hiperparámetros) durante el desarrollo, dejando el test set completamente "virgen" hasta la evaluación final. Si ajustás el modelo mirando repetidamente el test set, terminás filtrando información de todas formas, aunque nunca hayas entrenado directamente con él.

## Filmina 07 — Diferencias Cruciales: Supervisado vs. No Supervisado

**Un tercer paradigma, intermedio, que no está en la filmina**: el **Aprendizaje Semi-Supervisado** — cuando tenés una pequeña porción de datos etiquetados y una masa grande sin etiquetar (etiquetar todo sería carísimo). Google Photos, por ejemplo, aprende a reconocer una cara con muy pocas fotos que vos etiquetaste ("esta es mi hermana") y después la reconoce en miles de fotos no etiquetadas. Es el punto intermedio entre los dos extremos de la tabla de la filmina.

## Filmina 08 — Errores Comunes y "Trampas" Conceptuales

**El resultado numérico concreto de tratar un ID como número (la filmina lo menciona sin resolverlo)**: promediar los códigos postales 1401 y 1425 da 1413 — un código que puede pertenecer a un barrio geográficamente ajeno a ambos, o no existir. La operación es matemáticamente válida pero semánticamente absurda, porque el código postal es una etiqueta, no una cantidad.

## Filmina 09 — Casos de Uso Reales: el Impacto en la Economía Actual

**Un cuarto caso, de Marketing, que no está en la filmina**: predecir qué usuarios van a abrir un email antes de enviarlo (Clasificación binaria: abre / no abre), usando como features el historial de aperturas pasadas, el horario de envío y el asunto. Es el mismo mecanismo de Netflix (Filmina 09) aplicado a una industria completamente distinta — vale la pena mostrar que el patrón "predecir una acción del usuario" se repite en cualquier negocio con historial de comportamiento.

## Filmina 10 — Práctica (no entregable): Diseñando tu Primer Escenario

**Qué mirar al corregir (no está en la filmina)**: el error más común no es elegir mal entre Clasificación y Regresión — es proponer 5 características (X) que en realidad son la misma información repetida (ej. "ingresos mensuales" y "categoría de ingresos" como dos features separadas cuando una deriva directamente de la otra). Vale la pena preguntar explícitamente si cada X aporta información *nueva*.

---

# Módulo 02 — Regresión Lineal (Filminas 11-17)

## Filmina 11 — División de Módulo

El gancho: "si aumentamos el presupuesto de publicidad en \$1.000, ¿cuánto crecen las ventas?" — la pregunta que la Regresión Lineal está diseñada para responder con un número concreto.

## Filmina 12 — ¿Qué es la Regresión Lineal?

**Ejemplo numérico mínimo para el pizarrón (no está en la filmina)**: publicidad = [1, 2, 3] (miles de \$), ventas = [105, 190, 310] (unidades). Ninguna línea recta pasa exactamente por los tres puntos — la Regresión Lineal encuentra la que minimiza la distancia total a los tres, no la que "acierta" en alguno de ellos.

## Filmina 13 — El Motor Bajo el Capó: Mínimos Cuadrados

**Por qué se eleva al cuadrado, y no se usa el valor absoluto (no está en la filmina)**: usar la suma de errores absolutos también evitaría que los positivos cancelen a los negativos, pero esa función no es diferenciable en cero — matemáticamente más incómoda de minimizar. Elevar al cuadrado da una función suave con un único mínimo, resoluble con álgebra directa (por eso "Mínimos Cuadrados", no "Mínimos Absolutos"). Es una decisión matemática, no arbitraria.

## Filmina 14 — Anatomía de la Ecuación: Intercepto y Pendiente

**Un ejemplo con un intercepto que sí tiene sentido de negocio (la filmina usa uno que no lo tiene)**: en un modelo de tiempo de entrega = intercepto + pendiente × distancia_km, el intercepto representa el tiempo fijo de preparación del pedido antes de salir a repartir (por ejemplo, 8 minutos) — acá el intercepto sí es interpretable, a diferencia del ejemplo de altura-vs-peso de la filmina. Sirve para mostrar que "a veces no tiene sentido físico" no es una regla universal, es caso por caso.

## Filmina 15 — De la Matemática al Negocio: Interpretación de Coeficientes

**Una trampa que no está en la filmina, y que rompe la interpretación "manteniendo todo lo demás constante"**: si dos features están muy correlacionadas entre sí (ej. metros cuadrados y cantidad de ambientes, que casi siempre suben juntos), los coeficientes individuales se vuelven inestables y difíciles de interpretar — es la **multicolinealidad**. El modelo puede seguir prediciendo bien en conjunto, pero el coeficiente de cada variable por separado deja de ser confiable.

## Filmina 16 — Errores Comunes y Trampas Conceptuales

**Un ejemplo numérico de extrapolación (la filmina lo describe sin cifras)**: un modelo entrenado con casas de 50 a 200 m² que da un coeficiente de \$1.500/m² no garantiza que una mansión de 800 m² valga exactamente el intercepto más 800×1.500 — el modelo nunca vio una relación de precio en ese rango, y las relaciones lineales rara vez se sostienen indefinidamente (el m² 700 no suele valer lo mismo que el m² 70).

## Filmina 17 — Práctica (no entregable): Predicción de Ingresos de Tiendas

**Qué mirar al corregir (no está en la filmina)**: pedirle al alumno que compare el intercepto del modelo simple contra el del modelo múltiple — deberían ser distintos, porque al agregar "Número de Empleados" el intercepto ahora representa el ingreso esperado con 0 empleados *y* 0 m², un escenario aún más hipotético. Es una buena forma de verificar que entendió que el intercepto cambia de significado según qué otras variables lo acompañan.

---

# Módulo 03 — Árboles de Decisión y Random Forest (Filminas 18-26)

## Filmina 18 — División de Módulo

El gancho: elegir un teléfono nuevo mediante preguntas secuenciales ("¿presupuesto > \$500? ¿iOS o Android?") es exactamente la lógica de un Árbol de Decisión.

## Filmina 19 — Árboles de Decisión: la Lógica de las Reglas

**Cálculo numérico de Gini para el ejemplo bancario (no está en la filmina)**: en el nodo raíz, si de 10 solicitantes 6 pagan y 4 no pagan, el Índice Gini es 1 − (0.6² + 0.4²) = 1 − (0.36 + 0.16) = **0.48**. Después de dividir por "¿Ingresos > \$3.000?", si el grupo de la derecha queda con 5 pagan / 0 no pagan, su Gini es 1 − (1² + 0²) = **0** — un nodo perfectamente puro. Esa reducción de 0.48 a 0 (ponderada por ambos grupos) es la "ganancia" que el algoritmo persigue en cada split.

## Filmina 20 — Impureza y Splits (Divisiones)

**Cómo se ve en código (no está en la filmina)**: el parámetro `criterion` de `DecisionTreeClassifier` en scikit-learn acepta literalmente `'gini'` o `'entropy'` — son dos formas de medir lo mismo (desorden en un nodo) con fórmulas ligeramente distintas; en la práctica dan árboles casi idénticos la mayoría de las veces.

## Filmina 21 — El Peligro de la Perfección: Overfitting

**Cómo se detecta en la práctica, no solo en teoría (no está en la filmina)**: se comparan el score del modelo en train contra el score en test. Un árbol con 100% de acierto en train y 70% en test está sobreajustado — la brecha entre ambos números es la señal de alarma, más que cualquiera de los dos números por separado.

## Filmina 22 — De un Árbol a un Bosque: Random Forest

**Una validación "gratis" que no está en la filmina**: como cada árbol se entrena con una muestra Bootstrap (con reemplazo), en promedio cada árbol deja afuera cerca de un tercio de los datos de entrenamiento (las filas "Out-Of-Bag", OOB). Random Forest puede usar esas filas no vistas por cada árbol como una validación interna gratuita, sin gastar datos de un test set separado (`oob_score=True` en scikit-learn).

## Filmina 23 — La Aleatoriedad de Variables: el Toque "Random"

**La distinción técnica que no está en la filmina**: Bagging (Bootstrap de filas) por sí solo ya reduce el overfitting, pero sigue produciendo árboles parecidos si hay una variable muy dominante. La selección aleatoria de columnas en cada split es específicamente lo que separa a **Random Forest** de un simple "Bagging de árboles" — es el ingrediente extra que le da el nombre.

## Filmina 24 — Comparativa: ¿Cuándo Usar Cada Uno?

**Una tercera familia que no está en la tabla de la filmina**: *Gradient Boosting* (XGBoost, LightGBM) — a diferencia de Random Forest, que entrena árboles en paralelo e independientes, Boosting entrena árboles en secuencia, donde cada uno corrige los errores del anterior. Suele dar más precisión que Random Forest, a costa de ser más lento de entrenar y más sensible a overfitting si no se regula bien. Vale la pena nombrarlo como el "siguiente paso" natural después de este módulo.

## Filmina 25 — Casos de Aplicación en la Industria

**Un cuarto caso, de Agricultura de Precisión, que no está en la filmina**: sensores de suelo y clima (humedad, temperatura, nutrientes) alimentan un Random Forest que predice el rendimiento esperado del cultivo por hectárea — permite decidir dónde regar o fertilizar más antes de la cosecha, en vez de tratar todo el campo por igual.

## Filmina 26 — Práctica (no entregable): Árbol vs. Random Forest

**Qué mirar al corregir (no está en la filmina)**: si el alumno reporta que el árbol podado (`max_depth=3`) tiene *mejor* accuracy en test que el Random Forest, no es necesariamente un error — con datasets chicos y relaciones simples, un árbol bien podado puede superar a un bosque. Vale la pena pedir que interprete el resultado en vez de asumir que "más complejo siempre gana".

---

# Módulo 04 — Clasificación: Regresión Logística y KNN (Filminas 28-32)

*(Filmina 27 es el Break — 10 minutos.)*

## Filmina 28 — División de Módulo

El gancho: "¿Spam o no? ¿El cliente se irá o se quedará?" — el mundo de las respuestas categóricas, después de haber trabajado solo con números continuos en el Módulo 02.

## Filmina 29 — Regresión Logística: la Función Sigmoide

**El matiz que no está en la filmina, y que suele generar confusión**: a diferencia de la Regresión Lineal, los coeficientes de la Regresión Logística no se interpretan directamente en las unidades de y — se interpretan en escala de **log-odds** (logaritmo de la razón de probabilidades). Un coeficiente positivo de 0.7 no significa "sube 0.7 la probabilidad"; significa que multiplica las chances (*odds*) por e⁰·⁷ ≈ 2. Para la clase alcanza con la intuición (positivo = sube la probabilidad, negativo = la baja), pero vale la pena mencionar que la lectura exacta del número no es tan directa como en la lineal.

## Filmina 30 — K-Nearest Neighbors (KNN)

**Cómo elegir K, que no está en la filmina**: es un trade-off de sesgo-varianza clásico. K muy chico (K=1) hace que el modelo sea muy sensible al ruido de cada punto individual (alto riesgo de overfitting); K muy grande empieza a "promediar" con vecinos poco relevantes y pierde matices (underfitting). En la práctica se prueban varios valores de K con Cross-Validation y se elige el que mejor generaliza — el mismo K-Fold que ya vimos en Clase 07.

## Filmina 31 — El Rol Crítico del Escalado

**Conexión directa con el notebook de hoy (no está en la filmina)**: esto no es solo teoría — en el Bloque 6 del notebook, KNN se entrena sobre `X_train_scaled`/`X_test_scaled` (los mismos datos ya estandarizados en el Bloque 2), reutilizando el mismo `StandardScaler` que ya se había ajustado para la Regresión Lineal. No hace falta escalar dos veces.

## Filmina 32 — ¿Cómo Evaluar un Clasificador?

**La trampa que se retoma con más detalle en el Módulo 06 (no está en la filmina)**: la Accuracy sola puede mentir en datasets desbalanceados — un modelo que siempre predice "no fraude" en un dataset con 1% de fraude tiene 99% de Accuracy y es completamente inútil. Vale la pena adelantar acá que la Matriz de Confusión existe justamente para no depender de un solo número que puede engañar.

---

# Módulo 05 — La Ciencia Detrás del Aprendizaje Supervisado (Filminas 33-40)

## Filmina 33 — División de Módulo

Segundo pase, más profundo, sobre los mismos fundamentos del Módulo 01 — por qué el aprendizaje supervisado es el estándar de oro de la industria, y los errores que separan a un principiante de un experto.

## Filmina 34 — El Núcleo del Aprendizaje Supervisado: Datos Etiquetados

**Un matiz que no está en la filmina**: el "error" de la fórmula f(X) = y + error no es solo ruido de medición — a veces es información genuinamente ausente. Dos hogares con exactamente el mismo X (mismos m², misma ubicación) pueden tener y distintos por razones que el dataset no capta (urgencia del vendedor, estado real de la cocina). Ningún modelo, por más sofisticado, puede reducir ese error a cero — es el límite teórico de lo predecible con esas features.

## Filmina 35 — Clasificación vs. Regresión: el Reto del Umbral

**La conexión más importante de todo el módulo, y que no está en la filmina**: un problema de Regresión se puede convertir en uno de Clasificación cortando la variable continua en categorías — es exactamente lo que hace el Bloque 6 del notebook de hoy: toma el `IPCF` (continuo, Regresión) y lo corta en "bajo la línea de pobreza" sí/no (categórico, Clasificación) con un umbral. La tabla de la filmina las presenta como dos mundos separados, pero en la práctica son dos formas distintas de mirar la misma variable.

## Filmina 36 — El Costo Oculto: el Cuello de Botella del Etiquetado

**Una técnica que no está en la filmina, pensada exactamente para este problema**: el *Active Learning* — en vez de etiquetar datos al azar, un modelo entrenado con pocas etiquetas identifica cuáles son los ejemplos donde está *más inseguro* (probabilidad cerca de 0.5) y le pide a un humano que etiquete específicamente esos. Maximiza la mejora del modelo por cada etiqueta pagada, en vez de gastar presupuesto en ejemplos "obvios" que el modelo ya clasifica bien.

## Filmina 37 — Generalización: el Verdadero Objetivo

**Cómo se ve el ruido en un dataset tabular, no solo en la analogía de la seta (no está en la filmina)**: si el `ID_Transaccion` quedara sin querer dentro de las features de un modelo, el árbol podría "memorizar" que la transacción #4521 fue fraude — un patrón que no existe en la realidad y que no sirve para ninguna transacción futura. Es la misma idea de la piedra gris, pero con una columna real que a veces se cuela por descuido.

## Filmina 38 — Overfitting y Underfitting

**La herramienta de diagnóstico que une ambos conceptos (no está en la filmina)**: la *curva de validación* (`validation_curve` en scikit-learn) grafica el score de train y el de test a medida que aumenta la complejidad del modelo (por ejemplo, `max_depth` de 1 a 20). Se ve un patrón en forma de U invertida para el test: sube, llega a un punto óptimo, y después baja mientras el train sigue subiendo — ese punto óptimo es el que se busca, ni underfitting ni overfitting.

## Filmina 39 — Casos de Uso Reales: el Impacto en la Industria

**Un quinto caso, de Agricultura, que conecta con la Filmina 25 (no está en la filmina)**: los mismos sensores que alimentan un Random Forest para predecir rendimiento de cultivo también pueden entrenar un clasificador de "enfermedad de la planta detectada / no detectada" a partir de fotos — el mismo dataset de sensores sirve tanto para un problema de Regresión como para uno de Clasificación, según qué variable se use como target.

## Filmina 40 — Errores Comunes: Desbalanceo, Data Leakage y Concept Drift

**Una técnica concreta contra el desbalanceo que no está en la filmina**: además de mirar Recall o F1 en vez de Accuracy, existe **SMOTE** (*Synthetic Minority Oversampling Technique*) — genera ejemplos sintéticos de la clase minoritaria (no los copia, los interpola entre vecinos reales) para balancear el dataset de entrenamiento antes de entrenar el modelo.

---

# Módulo 06 — Pre-entrega: Evaluación y Comparación del Pipeline (Filminas 41-47)

## Filmina 41 — División de Módulo

El anuncio: consolidar todo el módulo en un artefacto concreto que se integra directamente al proyecto final de la carrera.

## Filmina 42 — El Desafío de las Clases Desbalanceadas

**Conexión directa con el notebook (no está en la filmina)**: en el Bloque 6 de hoy, el target "bajo la línea de pobreza" (construido con un umbral del 50% de la mediana del IPCF) casi con certeza va a salir desbalanceado — es un buen momento para correr esa celda en vivo y mirar el porcentaje real que imprime, en vez de solo hablar del caso abstracto del 99.9% de la filmina.

## Filmina 43 — F1-Score y Curva ROC/AUC

**Cómo leer la curva ROC de un vistazo (no está en la filmina)**: cuanto más se "empuje" la curva hacia la esquina superior izquierda, mejor el modelo — significa que a un umbral dado, captura muchos verdaderos positivos (eje Y alto) sin acumular casi falsos positivos (eje X bajo). Una curva pegada a la diagonal (la línea de "azar") es un modelo que no aprendió nada útil.

## Filmina 44 — Curva Precision-Recall

**Por qué la ROC puede "mentir" con clases muy desbalanceadas (no está en la filmina)**: la ROC incluye los Verdaderos Negativos en su cálculo, y cuando la clase negativa es enorme (99% de "no fraude"), esos aciertos triviales inflan la curva y la hacen ver mejor de lo que realmente es. La curva Precision-Recall ignora los Verdaderos Negativos por diseño — por eso es más honesta cuando la clase de interés es chica.

## Filmina 45 — Errores Comunes a Evitar

**Un cuarto error que no está en la lista de la filmina**: reportar el AUC de train en vez del de test — un modelo con AUC = 0.99 en train y 0.65 en test no es "excelente", está sobreajustado. La regla de oro del Bloque 3/4 (evaluar siempre sobre datos que el modelo nunca vio) aplica exactamente igual acá.

## Filmina 46 — Entregable: Qué Tenés que Presentar

**El puente directo con el notebook (no está en la filmina)**: el Bloque 6 del notebook de hoy (`Clase_8_Fundamentos_Data_Science_1.ipynb`) es, celda por celda, una plantilla ejecutable de exactamente esta consigna — Regresión Logística vs. KNN, Matriz de Confusión y F1-Score de cada uno, y una Curva ROC comparando ambos en el mismo gráfico. El alumno puede correrlo, entenderlo, y adaptar la misma estructura a su propio dataset del proyecto.

## Filmina 47 — Criterios de Aceptación y Formato

**Qué mirar al corregir, más allá de la checklist (no está en la filmina)**: la parte más difícil de calificar no es si están los 5 elementos técnicos (matriz, F1, ROC, etc.) — es si la "Conclusión de Negocio" realmente distingue cuál error es más caro para ese caso específico (Falso Positivo vs. Falso Negativo), y no repite una frase genérica de manual. Esa es la diferencia entre un informe técnico y un informe que un gerente podría usar para decidir.

## Filmina 48 (última) — ¿Dudas? ¿Consultas?

Cierre de la clase — espacio abierto antes de que el grupo se ponga a trabajar en la Pre-entrega.

---

## Guía del Notebook (`Clase_8_Fundamentos_Data_Science_1.ipynb`)

El notebook aplica el contenido de la clase sobre datos reales de la **EPH** (Encuesta Permanente de Hogares, INDEC): dos archivos relacionados (`usu_hogar_T425.txt` y `usu_individual_T425.txt`) que se unen para construir un dataset de hogares con `IPCF` (Ingreso Per Cápita Familiar) como variable objetivo. Es distinto del `Unidad_8.ipynb` que está en `Material/` (ese usa Iris, Breast Cancer y datos de acciones D/EXC/NEE/SO/DUK) — el notebook de referencia para esta guía es el de la raíz de la carpeta, el que efectivamente se corre en clase.

**Mapa rápido:**

| Bloque | Contenido | Módulo de la filmina que aplica |
|---|---|---|
| **Bloque 0** | Repaso de Clase 07: Supervisado vs. No Supervisado, con un scatter sintético de juguete. | Repaso, no es contenido nuevo |
| **Bloque 1** | Carga y join de la EPH (`usu_hogar` + `usu_individual`), sanitización del `IPCF` (coma decimal). | Módulo 01 |
| **Bloque 2** | Limpieza de códigos de no-respuesta, imputación por mediana, estandarización, split train/test. | Prerrequisito de Clase 06, aplicado sobre datos reales |
| **Bloque 3** | Regresión Lineal sobre `IPCF` + diagnóstico visual de residuos (real vs. predicho, distribución de errores). | Módulo 02 |
| **Bloque 4** | Random Forest Regressor + métricas de regresión (MAE, RMSE, R²) + Feature Importance. | Módulo 03 |
| **Bloque 5** | Pipeline (imputación + escalado + modelo) y Cross-Validation de 5 folds. | Módulo 06 (versión de regresión) |
| **Bloque 6** | Target categórico "bajo la línea de pobreza", Regresión Logística vs. KNN, Matriz de Confusión, F1-Score, Curva ROC. | Módulo 04 + Módulo 06 (Pre-entrega) |

### Bloque 0 — Repaso de la Clase Anterior (Clase 07)

**Por qué arranca acá el notebook**: los Bloques 1 a 5 dan por sentado que ya se entiende la diferencia entre tener y no tener etiquetas — este bloque lo refresca con un ejemplo visual de 30 segundos antes de tocar los datos reales de la EPH.

**Qué decir mientras corre**: se genera un dataset sintético de 3 grupos (`make_blobs`) y se grafica dos veces — una vez en gris (sin etiquetas, como lo vería K-Means en Clase 07) y otra coloreado por grupo (con etiquetas, como lo ve un algoritmo supervisado). Es el mismo dataset en ambos paneles — lo único que cambia es si el color (la etiqueta) está disponible o no. Vale la pena señalar que la EPH que viene después sí trae la etiqueta (`IPCF`) desde el origen — no hay que inventarla ni agruparla.

👉 **En el notebook:**
```python
X_demo, y_demo = make_blobs(n_samples=150, centers=3, random_state=42)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(X_demo[:, 0], X_demo[:, 1], color='gray')          # sin etiquetas
ax[1].scatter(X_demo[:, 0], X_demo[:, 1], c=y_demo, cmap='viridis')  # con etiquetas
plt.show()
```

**Línea por línea:**
- `make_blobs(n_samples=150, centers=3, random_state=42)` → genera 150 puntos sintéticos agrupados en 3 grupos "de fábrica"; `random_state=42` fija la semilla para que el resultado sea siempre el mismo.
- `ax[0].scatter(..., color='gray')` → grafica los puntos todos del mismo color — así los "ve" un algoritmo no supervisado, sin ninguna pista de a qué grupo pertenece cada uno.
- `ax[1].scatter(..., c=y_demo, cmap='viridis')` → el mismo dataset, coloreado según `y_demo` (la etiqueta que `make_blobs` generó internamente) — así lo ve un algoritmo supervisado.
- El único cambio entre ambos paneles es si `y` está disponible o no — ese es el contraste completo que el bloque quiere mostrar.

### Bloque 1 y Bloque 2 — Carga de la EPH y Preprocesamiento

**Qué decir sobre la sanitización del `IPCF`**: es un caso real de "suciedad" de una fuente oficial — la EPH exporta el ingreso con coma decimal (`'633333,33'`) en vez de punto, y Python no puede convertir eso a número directamente. El código lo resuelve con `.str.replace(',', '.')` antes de `pd.to_numeric(..., errors='coerce')` — el mismo patrón de "detectar y corregir antes de calcular" del Módulo 03/04 de Clase 06.

**Qué decir sobre el código 99 en `IV2`**: es un ejemplo real de valor centinela (*sentinel value*) — 99 no significa "99 ambientes", significa "sin dato". Si no se reemplaza por `NaN` antes de imputar, el promedio de ambientes del dataset queda completamente distorsionado por hogares que en realidad no respondieron la pregunta.

👉 **En el notebook (Bloque 1 — carga y join):**
```python
df_hogar      = pd.read_csv("usu_hogar_T425.txt", sep=";")
df_individual = pd.read_csv("usu_individual_T425.txt", sep=";")

df_jefes = df_individual[df_individual['CH03'] == 1][['CODUSU', 'NRO_HOGAR', 'NIVEL_ED']]
df = pd.merge(df_hogar, df_jefes, on=['CODUSU', 'NRO_HOGAR'], how='inner')

features = ['IV1', 'IV2', 'IX_TOT', 'NIVEL_ED']
target   = 'IPCF'

if df[target].dtype == 'object':
    df[target] = df[target].astype(str).str.replace(',', '.')
df[target] = pd.to_numeric(df[target], errors='coerce')

df_model = df[features + [target]].dropna(subset=[target]).copy()
```

**Línea por línea:**
- `pd.read_csv(..., sep=";")` → los archivos de la EPH usan punto y coma como separador, no coma; hay que indicarlo o Pandas leería todo como una sola columna de texto.
- `df_individual['CH03'] == 1` → `CH03` es el código de relación de parentesco; `1` identifica específicamente al Jefe/a de Hogar dentro de cada grupo familiar.
- `pd.merge(..., on=['CODUSU', 'NRO_HOGAR'], how='inner')` → une hogares con su jefe usando **dos** columnas combinadas como clave (un hogar se identifica por esas dos juntas); `inner` descarta cualquier hogar sin jefe identificado en ambas tablas.
- `df[target].dtype == 'object'` → chequea si la columna llegó como texto — síntoma de que trae comas u otros caracteres no numéricos.
- `.str.replace(',', '.')` → reemplaza la coma decimal por punto en cada celda, antes de convertir a número.
- `pd.to_numeric(..., errors='coerce')` → convierte a número; lo que no sea convertible se vuelve `NaN` en vez de frenar el programa con un error.
- `dropna(subset=[target])` → descarta solo las filas sin `IPCF` válido; los nulos de las demás columnas se resuelven recién en el Bloque 2.

👉 **En el notebook (Bloque 2 — imputación, escalado y split):**
```python
df_model['IV2'] = df_model['IV2'].replace({99: np.nan})

X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)   # aprende + transforma
X_test_imputed  = imputer.transform(X_test)          # solo transforma

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled  = scaler.transform(X_test_imputed)
```

**Línea por línea:**
- `df_model['IV2'].replace({99: np.nan})` → reemplaza el código centinela `99` por `NaN`, para que el imputador lo trate como faltante real y no como "99 ambientes".
- `train_test_split(X, y, test_size=0.2, random_state=42)` → separa el 20% de las filas para test **antes** de tocar nada más; `random_state=42` hace que el split sea siempre el mismo.
- `SimpleImputer(strategy='median')` → crea el objeto que va a rellenar nulos con la mediana de cada columna.
- `imputer.fit_transform(X_train)` → **aprende** la mediana de cada columna mirando solo train, y aplica el reemplazo en el mismo paso.
- `imputer.transform(X_test)` → aplica esas mismas medianas (ya calculadas con train) sobre test, sin recalcularlas — la regla de oro contra el Data Leakage.
- `StandardScaler()` + `fit_transform` / `transform` → la misma lógica que el imputador, pero para estandarizar (media 0, desvío 1): se ajusta con train y se aplica igual a test.

### Bloque 3 y Bloque 4 — Regresión Lineal y Random Forest

**Qué decir sobre el resultado (R² ≈ 0.15)**: con solo 4 variables (tipo de vivienda, ambientes, cantidad de personas, nivel educativo del jefe de hogar) explicando apenas el 15% de la varianza del ingreso, es un resultado realista, no un fracaso — el ingreso de un hogar depende de decenas de factores que no están en este dataset reducido (rubro de actividad, informalidad laboral, edad, región). Es un buen momento para conectar con la Filmina 37 (Módulo 05): el modelo no está "mal", está limitado por la información disponible en X.

**Qué decir sobre el gráfico de residuos**: dado que el `IPCF` tiene una distribución muy sesgada a la derecha (muchos hogares con ingreso bajo, pocos con ingreso muy alto), es esperable ver residuos asimétricos en vez de una campana perfecta — es la misma advertencia de la Filmina 04 de Clase 06 (sesgo) apareciendo en un modelo real.

👉 **En el notebook (Bloque 3 — Regresión Lineal):**
```python
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

for fname, coef in zip(features, lr_model.coef_):
    print(f"{fname}: {coef:+,.0f} pesos")

y_pred_lr = lr_model.predict(X_test_scaled)
residuals = y_test - y_pred_lr
```

**Línea por línea:**
- `LinearRegression()` → instancia el modelo; todavía no calculó nada.
- `lr_model.fit(X_train_scaled, y_train)` → ajusta los coeficientes β por Mínimos Cuadrados usando solo los datos de entrenamiento ya escalados.
- `zip(features, lr_model.coef_)` → empareja cada nombre de columna con su coeficiente aprendido, para poder imprimir "qué feature pesa cuánto" en pesos.
- `lr_model.predict(X_test_scaled)` → usa los coeficientes ya aprendidos para predecir el `IPCF` de las filas de test, que el modelo nunca vio.
- `residuals = y_test - y_pred_lr` → la diferencia entre el valor real y el predicho, fila por fila; es lo que alimenta el histograma de residuos.

👉 **En el notebook (Bloque 4 — Random Forest y métricas):**
```python
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

mae  = mean_absolute_error(y_test, y_pred_rf)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2   = r2_score(y_test, y_pred_rf)

for fname, imp in sorted(zip(features, rf_model.feature_importances_), key=lambda x: -x[1]):
    print(f"{fname}: {imp:.4f}")
```

**Línea por línea:**
- `RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)` → 100 árboles, cada uno limitado a profundidad 5 para no sobreajustar; `random_state` fija la aleatoriedad del bootstrap y de la selección de columnas por split.
- `rf_model.fit(X_train_scaled, y_train)` → entrena los 100 árboles sobre train.
- `mean_absolute_error` / `mean_squared_error` / `r2_score` → tres formas distintas de resumir qué tan lejos quedó `y_pred_rf` de `y_test`; `rmse` se obtiene aplicando `np.sqrt` sobre el MSE.
- `rf_model.feature_importances_` → un array con la contribución relativa de cada feature a las divisiones del bosque; `sorted(..., key=lambda x: -x[1])` lo ordena de mayor a menor importancia para imprimirlo legible.

### Bloque 5 — Pipeline y Cross-Validation

**Qué decir**: la desviación estándar del R² entre los 5 folds (~0.016 en la corrida de referencia) es baja — señal de que el modelo es estable, aunque no sea muy potente. Vale la pena distinguir explícitamente "estable" de "bueno": un modelo puede dar siempre el mismo R² bajo en todos los folds (estable pero débil) o un R² alto que varía mucho entre folds (potente pero inestable) — son dos preguntas distintas.

👉 **En el notebook:**
```python
pipeline_produccion = Pipeline([
    ('imputador_central', SimpleImputer(strategy='median')),
    ('escalador_estandar', StandardScaler()),
    ('modelo_bosque', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42))
])

scores_cv = cross_val_score(pipeline_produccion, X_train, y_train, cv=5, scoring='r2')

pipeline_produccion.fit(X_train, y_train)
y_pred_final = pipeline_produccion.predict(X_test)
```

**Línea por línea:**
- `Pipeline([...])` → encadena tres pasos con nombre (imputar, escalar, modelar) en un solo objeto; cada paso alimenta al siguiente, en orden.
- `cross_val_score(pipeline_produccion, X_train, y_train, cv=5, scoring='r2')` → parte `X_train` en 5 folds y entrena el pipeline **completo** 5 veces (una por fold), devolviendo el R² de cada corrida; como el `fit` del imputador/escalador ocurre **dentro** de cada fold, ningún fold de test se filtra al ajuste.
- `pipeline_produccion.fit(X_train, y_train)` → una vez validada la estabilidad, se reentrena una última vez con **todo** train (no solo 4/5 folds) para aprovechar el máximo de datos disponibles.
- `pipeline_produccion.predict(X_test)` → predice sobre test pasando los datos crudos, sin escalar a mano — el Pipeline aplica imputación y escalado automáticamente antes de llegar al modelo.

### Bloque 6 — Clasificación: Regresión Logística vs. KNN

**Por qué este bloque es el más importante del notebook para la Pre-entrega**: hasta acá todo fue Regresión. Este bloque convierte el mismo `IPCF` en una variable categórica ("¿bajo la línea de pobreza?", con un umbral del 50% de la mediana de train) y reproduce, celda por celda, exactamente lo que pide la consigna de la Pre-entrega: dos modelos de clasificación comparados, Matriz de Confusión, F1-Score y Curva ROC con AUC en la leyenda.

**Qué decir sobre la línea de pobreza**: es un umbral relativo (50% de la mediana), no una cifra oficial del INDEC — se explica así a propósito, para que quede claro que es una simplificación pedagógica y no una afirmación estadística oficial sobre pobreza en Argentina. Si el grupo pregunta por la línea de pobreza real, vale la pena aclarar la diferencia entre esta construcción didáctica y la metodología oficial de INDEC (que usa una Canasta Básica Total, no un porcentaje de la mediana).

**Qué decir sobre por qué el umbral se calcula solo con `y_train`**: es la misma regla de oro del imputador y el escalador (Bloque 2) — calcular el umbral con todo el dataset filtraría información del test hacia el proceso de entrenamiento, aunque el "objeto" que se ajusta acá no sea un `Scaler` sino un número simple.

👉 **En el notebook:**
```python
linea_pobreza = y_train.median() * 0.5
y_train_clf = (y_train < linea_pobreza).astype(int)
y_test_clf  = (y_test < linea_pobreza).astype(int)

log_model = LogisticRegression(random_state=42)
log_model.fit(X_train_scaled, y_train_clf)
y_proba_log = log_model.predict_proba(X_test_scaled)[:, 1]

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train_clf)
y_proba_knn = knn_model.predict_proba(X_test_scaled)[:, 1]

for nombre, y_proba in [('Regresión Logística', y_proba_log), ('KNN', y_proba_knn)]:
    fpr, tpr, _ = roc_curve(y_test_clf, y_proba)
    auc = roc_auc_score(y_test_clf, y_proba)
```

**Línea por línea:**
- `y_train.median() * 0.5` → la línea de pobreza es el 50% de la mediana del `IPCF`, calculada **solo** con `y_train` — el mismo criterio del imputador/escalador del Bloque 2, para no filtrar información de test.
- `(y_train < linea_pobreza).astype(int)` → convierte el target continuo en binario: `1` si el ingreso está bajo esa línea, `0` si no; `.astype(int)` pasa el booleano a `0`/`1` numérico.
- `LogisticRegression().fit(...)` / `KNeighborsClassifier(n_neighbors=5).fit(...)` → entrenan los dos modelos a comparar sobre los mismos datos ya escalados (`X_train_scaled`, reutilizado del Bloque 2).
- `.predict_proba(X_test_scaled)[:, 1]` → en vez de la clase predicha, pide la **probabilidad** de pertenecer a la clase `1` (bajo la línea de pobreza) — la columna `[:, 1]` es la que hace falta para trazar la curva ROC.
- `roc_curve(y_test_clf, y_proba)` → calcula, para muchos umbrales de decisión distintos, la tasa de falsos positivos (`fpr`) y de verdaderos positivos (`tpr`) — los ejes X e Y de la curva.
- `roc_auc_score(y_test_clf, y_proba)` → resume toda la curva en un solo número entre 0 y 1: el área bajo la curva (AUC).

---

## Pre-entrega: "Evaluación y Comparación del Pipeline"

✅ **Entregable evaluado del Módulo.** Es la que se corrige y suma al proyecto final — todas las demás prácticas de la clase son guiadas y no evaluables.

### Objetivo

Integrar el trabajo de las unidades anteriores para comparar formalmente al menos dos modelos de clasificación y documentar cuál es el más apto para resolver el problema planteado.

### Qué tiene que presentar el estudiante

1. **Selección de Datos**: un dataset ya limpio y preprocesado (escalado y codificación) de trabajos anteriores o sugerido por el tutor.
2. **Entrenamiento Comparativo**: un modelo "base" simple (ej. Regresión Logística) y uno "complejo" o de ensamble (ej. Random Forest, o un Árbol optimizado con GridSearch).
3. **Generación de Evidencia**: Matriz de Confusión de ambos modelos sobre el test set, métricas de Precision/Recall/F1, y una Curva ROC comparando ambos modelos en la misma imagen.
4. **Redacción del Informe**: portada, tabla comparativa de métricas, análisis visual de las Curvas ROC (explicando el AUC de cada una), y una conclusión de negocio que indique qué modelo recomendarías y por qué (justificando según qué error —Falso Positivo o Falso Negativo— es más crítico para el caso de estudio).

### Criterios de Aceptación

- Al menos dos modelos distintos comparados.
- El informe incluye explícitamente el F1-Score y la Matriz de Confusión.
- Existe un gráfico de Curva ROC correctamente etiquetado (ejes y leyenda).
- La conclusión técnica demuestra comprensión de la diferencia entre precisión y cobertura (recall).

**Formato de entrega**: un único archivo PDF o Word. Nombre sugerido: `Apellido_Nombre_PreEntrega_M9.pdf`.

---

## Síntesis y Conexión Final

La clase entera se puede resumir en una progresión: primero entendemos qué significa "aprender de ejemplos etiquetados" (Módulo 01); después construimos el modelo más simple posible que traduce esa idea en números interpretables (Regresión Lineal, Módulo 02); subimos en complejidad con modelos basados en reglas y ensambles (Árboles y Random Forest, Módulo 03); cruzamos al mundo de las categorías con Regresión Logística y KNN (Módulo 04); profundizamos en los riesgos reales que separan un modelo de juguete de uno de producción —desbalanceo, fuga de información, deriva de concepto— (Módulo 05); y cerramos demostrando, con métricas correctas, cuál de dos modelos conviene realmente usar (Módulo 06, la Pre-entrega).

En la próxima unidad se retoma la optimización de estos mismos modelos —ajuste de hiperparámetros con GridSearch/RandomizedSearch— construyendo directamente sobre el Pipeline que hoy se armó en el Bloque 5 del notebook.
