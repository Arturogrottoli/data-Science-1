# Clase 01 — Nivelación Técnica, Lógica de Programación y Transformación Digital

**Curso de Data Science I · Clase 01** — Bloque 1: Nivelación Técnica & Lógica | Bloque 2: Transformación Digital

Esta guía sigue el **orden exacto de las 26 filminas** de `Semana 1.html`. Cada sección está etiquetada con la filmina a la que corresponde, para poder ir mostrando la diapositiva y leyendo/ampliando en paralelo. El contenido de cada sección suele ser **más profundo que lo que dice literalmente la filmina** — la idea es que el material de la filmina sea el disparador visual, y el texto de acá sea lo que se dice en voz alta para darle cuerpo.

---

## Objetivos de la clase

1. Comprender el rol de los datos en la organización desde la perspectiva de la Transformación Digital y la Industria 4.0.
2. Identificar los componentes importantes de una estrategia de Data Science.
3. Detectar oportunidades de uso de datos para la transformación digital y la estrategia de negocios.
4. Clasificar las principales herramientas para un Científico de Datos y sus características.

---

## Filmina 01 — Portada

Apertura de la clase. El subtítulo ya anticipa la estructura del día: **Bloque 1** es nivelación técnica pura (lógica, Python, herramientas) y **Bloque 2** es la mirada de negocio (Industria 4.0, transformación digital, datasets). Vale la pena decirlo así de entrada para que el grupo entienda por qué la clase "cambia de tema" a mitad de camino — no son dos clases distintas, son las dos mitades de la misma pregunta: *¿cómo se genera valor con datos?* Bloque 1 responde "con qué herramientas" y Bloque 2 responde "para qué".

## Filmina 02 — Presentación del Equipo Docente

Presentación de **Guillermo Mallo** y **Arturo Grottoli**, profesores del curso. Es el momento de contar brevemente el enfoque pedagógico: la cursada combina teoría corta con mucha práctica en Google Colab, y el hilo conductor de todas las clases va a ser ir armando, de a poco, un proyecto de Data Science completo sobre un dataset propio — el mismo que se elige hoy en la Pre-Entrega 1 (Filmina 24).

## Filmina 03 — Presentación de Estudiantes: el Icebreaker

Antes de entrar en contenido, la clase abre con una ronda de presentación entre los propios estudiantes. No es solo un trámite social — sirve para calibrar el resto de la cursada:

- **¿De dónde son?** Ayuda a organizar horarios de consulta y a tener en cuenta husos horarios si hay gente cursando desde otro país.
- **¿Tienen conocimientos de programación?** Aunque sea de otro lenguaje (Java, JavaScript, VBA de Excel), da una idea de qué tan rápido se puede avanzar en el Bloque 1 sin perder a nadie.
- **¿Tienen conocimientos de Data Analytics?** Excel avanzado, tablas dinámicas, algún dashboard armado antes — es experiencia previa real aunque no haya sido en Python.
- **¿Hicieron algún curso anterior a este?** De Coderhouse o de otra plataforma — permite saber si el grupo ya viene con hábitos de trabajo en notebooks o si es la primera vez que abren un Jupyter.

**Por qué importa como docente:** un curso con la mitad del grupo viniendo de Excel avanzado se puede dar distinto a uno donde nadie programó nunca — no cambia el contenido, pero sí el ritmo y los ejemplos que conviene usar para enganchar a cada perfil.

---

# Bloque 1 — Nivelación Técnica y Lógica de Programación

## Filmina 04 — División: Arranca el Bloque 1

Divisor de sección. El mensaje para el grupo: antes de hablar de negocio o de Industria 4.0 hace falta un piso técnico común — vocabulario de roles, herramientas de trabajo y lógica de programación básica. Sin esto, cualquier ejemplo de negocio de la segunda mitad de la clase suena abstracto.

## Filmina 05 — ¿Qué es Data Science?

**Qué decir (ampliando lo que dice la filmina):**

- **Ciencia interdisciplinar:** Data Science no es "programar". Es la intersección de tres campos: **estadística** (para medir incertidumbre y validar si un patrón es real o casualidad), **programación** (para procesar volúmenes de datos que a mano serían imposibles) y **conocimiento de negocio/dominio** (para saber qué pregunta vale la pena hacerle a los datos). Sacar cualquiera de las tres patas y el proyecto falla: sin estadística se confunde ruido con señal, sin programación no se puede escalar, sin negocio se resuelve un problema que a nadie le importa.
- **Optimización empírica:** en vez de asumir una relación causa-efecto por intuición ("seguro que el precio es lo que más influye en la venta"), el objetivo es dejar que el patrón emerja de los datos. Es un cambio de enfoque: de "creo que..." a "los datos muestran que...".
- **Enfoque pragmático:** el valor de un proyecto no se mide en qué tan sofisticado es el modelo, sino en el valor accionable que genera. Una regresión lineal simple que reduce un costo real vale más que una red neuronal compleja que termina en un PDF que nadie lee. Esta idea va a volver muy fuerte en la Filmina 22 (Kahoot), donde justamente se desarma el mito de que "más complejo es mejor".

*Analogía útil:* un médico no diagnostica por corazonada — mide síntomas (datos), los compara contra casos previos (patrones históricos) y recién ahí decide (acción). Un científico de datos hace exactamente lo mismo, pero con datasets en vez de pacientes.

## Filmina 06 — Cómo Trabajan los Equipos de Datos

**Qué decir (ampliando cada tile):**

| Rol | Qué hace | Pregunta que responde |
|---|---|---|
| **Data Engineer** | Diseña y mantiene los *pipelines* que mueven datos desde su origen (sensores, apps, bases de datos) hasta un lugar limpio y accesible — el **Data Lake**. Se asegura de que los datos que llegan a los otros dos roles ya estén disponibles, actualizados y sin errores estructurales. | "¿Cómo hacemos que el dato llegue limpio y a tiempo?" |
| **Data Scientist** | Formula hipótesis estadísticas y aplica Machine Learning / IA para **predecir** comportamientos futuros (churn, fraude, demanda). Es el rol que este curso entrena principalmente. | "¿Qué va a pasar, y por qué?" |
| **Data Analyst** | Traduce la información ya procesada en **dashboards e informes** que un gerente puede leer en 30 segundos, para decisiones inmediatas (no predictivas). | "¿Qué está pasando ahora mismo?" |

En una empresa real estos tres roles trabajan encadenados: el Data Engineer entrega los datos limpios, el Data Analyst los reporta para el día a día, y el Data Scientist los usa para anticipar el futuro. En equipos chicos (o en este curso) una sola persona termina haciendo un poco de los tres — es útil saber en qué "sombrero" está uno en cada momento del proyecto.

**Herramientas típicas de cada rol** (ampliando el ecosistema que se va a usar en el curso):

| Herramienta | Rol que más la usa | Uso principal |
|---|---|---|
| **Power BI / Tableau** | Data Analyst | Dashboards interactivos para gerencia. No requiere programar. |
| **Pandas** | Data Scientist / Data Engineer | Manipulación de datos estructurados — el corazón de este curso. |
| **NumPy** | Data Scientist / Data Engineer | Cálculo numérico eficiente, base matemática de Pandas y de los modelos. |
| **Scikit-learn** | Data Scientist | Algoritmos de clasificación, regresión, clustering y validación de modelos. |

## Filmina 07 — Ecosistema de Análisis Reproducible: Conda, Jupyter y Colab

**Qué decir (ampliando cada tile):**

- **Anaconda / Conda:** un gestor de **ambientes virtuales**. El problema que resuelve: si un proyecto necesita `pandas 1.5` y otro necesita `pandas 2.0` instalados en la misma computadora, sin ambientes virtuales entran en conflicto. Con `conda create -n proyecto python=3.11` se crea una "burbuja" aislada con sus propias versiones de librerías, sin pisar las de otros proyectos.
- **Jupyter Notebook:** el entorno interactivo **local** donde se combinan celdas de código (que se ejecutan y muestran resultado al instante) con celdas de Markdown (texto, títulos, explicaciones). Es ideal para el trabajo exploratorio típico de Data Science, donde se va probando de a pasos chicos en vez de escribir un script entero de una sola vez.
- **Google Colab:** la alternativa **en la nube** al Jupyter local. Mismo concepto de celdas de código + Markdown, pero corriendo en un servidor de Google — no hay que instalar Python ni librerías, se accede desde cualquier navegador, y es fácil de compartir con un link (clave para trabajo colaborativo y para que los profesores revisen entregas). La contracara: los archivos subidos (como un CSV) **no persisten** entre sesiones a menos que se monte Google Drive, así que hay que volver a cargarlos cada vez que se abre el notebook de nuevo.

**Cuándo usar cada uno:** Colab para practicar y para el curso (cero fricción de instalación); Conda + Jupyter local para proyectos profesionales donde se necesita control total del ambiente, trabajar sin conexión a internet, o mover cargas de trabajo muy pesadas.

## Filmina 08 — Pensamiento Computacional

**Qué decir (ampliando lo que dice la filmina):**

El pensamiento computacional es la forma de "trocear" un problema grande para que se pueda resolver con código, mucho antes de escribir la primera línea:

- **Descomposición:** partir un problema grande en subproblemas chicos y manejables. Ejemplo: "predecir si un cliente se va a dar de baja" se descompone en "¿qué datos tengo del cliente?", "¿cómo mido si ya se dio de baja antes?", "¿qué variables podrían anticiparlo?".
- **Reconocimiento de patrones:** identificar similitudes entre problemas o casos — si ya resolví "predecir baja de cliente", la lógica se parece mucho a "predecir fraude en una transacción" (ambos son clasificación binaria).
- **Abstracción:** separar lo relevante del ruido. No todos los datos disponibles importan — un ID de cliente autoincremental no predice nada, pero su antigüedad sí.
- **Algoritmia:** diseñar la secuencia de pasos, en orden, para llegar de los datos crudos a la respuesta. Es literalmente lo que se hace después en Python: cargar datos → limpiar → explorar → modelar → evaluar.

Esta forma de pensar es independiente del lenguaje de programación — es la habilidad que después se traduce a código Python en las filminas siguientes, y es la misma habilidad que se usa para armar un pipeline de Pandas varias clases más adelante.

## Filmina 09 — Introducción a Python

**Qué decir (ampliando lo que dice la filmina):**

- **Tipado dinámico:** en Python no hace falta declarar el tipo de una variable antes de crearla — el intérprete lo infiere en el momento en que se le asigna un valor. Esto contrasta con lenguajes de tipado estático (Java, C#), donde hay que escribir `int edad = 25` explícitamente. En Python simplemente `edad = 25` alcanza, y Python decide solo que es un `int`.
- **Ecosistema gigante:** la razón principal por la que Python domina Data Science no es el lenguaje en sí, sino sus librerías: **NumPy** (cálculo numérico), **Pandas** (tablas de datos) y **Scikit-Learn** (Machine Learning) — las tres van a ser el stack central de todo el curso.

**El ejemplo de la filmina, línea por línea:**

```python
temperatura_actual = 24.5
nombre_sensor = "Sensor Termocupla A"

if temperatura_actual > 25.0:
    print("Alerta: temperatura elevada")
else:
    print("Estado óptimo del sistema")
```

`temperatura_actual` queda tipada como `float` (número con decimales) y `nombre_sensor` como `str` (texto), sin que se haya declarado nada explícitamente. El bloque `if/else` es la primera estructura de control: evalúa una condición booleana (`True`/`False`) y ejecuta un bloque u otro según el resultado — acá, como 24.5 no es mayor a 25.0, se imprime "Estado óptimo".

**Vocabulario base de tipos** (referencia rápida para todo el curso):

| Categoría | Tipos | Característica clave |
|-----------|-------|----------------------|
| **Tipos Simples** | `int`, `float`, `bool`, `str` | Inmutables — el valor no puede modificarse in-place. |
| **Tipos Estructurados** | `list`, `tuple`, `dict`, `set` | `list` y `dict` son mutables. `tuple` es inmutable. `set` no permite duplicados. |
| **Estructuras de Control** | `if/elif/else`, `for`, `while` | `if` para condiciones, `for` para iteraciones definidas, `while` para condiciones dinámicas. |
| **Funciones** | `def nombre(params):` | Bloques reutilizables. Evitan redundancia y mejoran mantenibilidad. |

## Filmina 10 — Colecciones de Datos I: Listas

**Qué decir (ampliando lo que dice la filmina):**

- **Indexación desde cero:** en Python (como en la mayoría de los lenguajes), el primer elemento de una lista está en la posición `[0]`, no en `[1]`. Es una de las confusiones más comunes al empezar — "el segundo elemento" se accede con `[1]`, no con `[2]`.
- **Estructura mutable:** una lista se puede modificar después de creada — agregar, quitar o cambiar elementos sin tener que crear una lista nueva.

**El ejemplo de la filmina:**

```python
lecturas = [22.4, 25.1, 23.8, 26.0]
lecturas.append(24.7)

print(lecturas[0])   # 22.4 -> primer elemento
print(lecturas[-1])  # 24.7 -> último elemento (índice negativo)
```

`.append(24.7)` agrega el valor al final de la lista, sin necesidad de crear una lista nueva — la lista original se modifica "in-place". Los índices negativos (`[-1]`) son un atajo muy usado para acceder al último elemento sin tener que saber cuántos hay en total. Este patrón — lista + `append()` dentro de un `for` — va a reaparecer todo el curso como la forma más básica de ir acumulando resultados.

## Filmina 11 — Colecciones II: Diccionarios (#FindTheBug)

**Qué decir (ampliando lo que dice la filmina):**

- **Acceso veloz:** a diferencia de una lista (donde para encontrar un valor a veces hay que recorrerla entera), un diccionario accede a un valor directamente por su **clave**, sin importar cuántos elementos tenga — es la estructura ideal para representar un registro con campos con nombre, como una fila de una tabla.

**El desafío #FindTheBug:** la filmina muestra un diccionario con un error de sintaxis intencional — falta una **coma** entre dos pares clave-valor (después de `"id_control": 501` y antes de la siguiente clave). En Python, cada par `clave: valor` de un diccionario literal tiene que estar separado por coma; si falta, el intérprete tira un `SyntaxError` y el programa ni siquiera llega a ejecutarse.

```python
# Con el bug (falta la coma):
registro = {
    "id_control": 501
    "sensor": "Termocupla A"   # <- SyntaxError acá
}

# Corregido:
registro = {
    "id_control": 501,
    "sensor": "Termocupla A"
}
```

**Por qué vale la pena el ejercicio:** es la primera exposición del grupo a leer un mensaje de error de Python y encontrar la causa exacta — una habilidad que se va a usar constantemente en el resto del curso, mucho más que memorizar sintaxis de memoria.

## Filmina 12 — Bloque 1: Recapitulativa Final

**Qué decir (cerrando el bloque, tile por tile):**

- **Conceptos Clave:** la diferencia entre los tres perfiles de datos (Filmina 06) y la idea de que la lógica de descomposición (Filmina 08) va **antes** de escribir código, no después — pensar el problema primero evita reescribir todo a mitad de camino.
- **Herramientas:** uso práctico de Conda para aislar paquetes en proyectos locales, y Google Colab para compartir cuadernos reproducibles de forma ágil — el entorno que se va a usar en el resto del curso.
- **Colecciones:** capacidad de estructurar datos secuenciales con listas (Filmina 10) y mapear registros con nombre de campo usando diccionarios (Filmina 11) — las dos estructuras que, más adelante en el curso, se combinan para formar las columnas de un DataFrame de Pandas.

## Filmina 13 — Break del Coder

Corte de 10 minutos. Buen momento para adelantar que después del break el enfoque cambia: se deja la lógica de programación pura y se pasa a la mirada de negocio — Industria 4.0 y Transformación Digital.

---

# Bloque 2 — Transformación Digital e Industria 4.0

## Filmina 14 — División: Arranca el Bloque 2

Divisor de sección. El mensaje: ya con las herramientas técnicas del Bloque 1 instaladas, ahora la pregunta es *para qué* se usan en un contexto real de empresa — cómo una organización pasa de tener datos sueltos a tomar decisiones con ellos.

## Filmina 15 — La Empresa en la Industria 4.0

**Qué decir (ampliando lo que dice la filmina):**

- **Sistemas Ciberfísicos (IoT):** máquinas que se comunican entre sí y reportan datos constantemente. Es la base física de todo lo que sigue — sin sensores generando datos, no hay nada que analizar.
- **De reactivo a predictivo:** el cambio de paradigma central de la Industria 4.0 es dejar de arreglar la máquina rota y pasar a predecir matemáticamente cuándo va a fallar (ver Caso 1 más abajo).

### Historia de las Revoluciones Industriales

Cada revolución industrial redefinió las relaciones de trabajo, la organización empresarial y el rol de la tecnología en la producción.

| Era | Tecnología clave | Qué cambió |
|---|---|---|
| **1.0** — Fines s. XVIII-XIX | Máquina de vapor · Carbón · Ferrocarril | El vapor movió maquinaria pesada sin depender de fuerza humana o animal. Las fábricas centralizaron la producción por primera vez. |
| **2.0** — Fines s. XIX-XX | Electricidad · Motor de combustión · Línea de ensamblaje (Ford) | Ford popularizó la línea de ensamblaje: cada operario hace una tarea repetida, acelerando la producción masiva. |
| **3.0** — 2ª mitad s. XX | Electrónica · Computadoras · PLC | Los Controladores Lógicos Programables automatizaron procesos sin intervención humana constante. Inicio de la gestión basada en datos, pero aislada por sistema. |
| **4.0** — Siglo XXI (hoy) | IoT · IA · Big Data · Cloud · CPS | Los sistemas ya no solo automatizan: **aprenden, se comunican entre sí y toman decisiones en tiempo real**. |

> La diferencia clave entre la 3.0 y la 4.0 es que antes se automatizaban tareas siguiendo instrucciones fijas; ahora los sistemas **aprenden y se adaptan solos** a partir de los datos que reciben.

### Los Nueve Elementos Fundamentales de la Industria 4.0

La Industria 4.0 integra múltiples tecnologías que no actúan en forma aislada, sino como un **ecosistema interconectado**:

| # | Tecnología | Descripción |
|---|-----------|-------------|
| 1 | **Big Data** | Manejo y análisis de grandes volúmenes de datos generados por sensores y máquinas. |
| 2 | **Robótica Avanzada** | Robots colaborativos (*cobots*) que trabajan junto a operarios humanos. |
| 3 | **Simulación** | Gemelos digitales para probar y optimizar procesos antes de implementarlos. |
| 4 | **Realidad Aumentada (AR)** | Superposición de información digital al entorno real — capacitación, mantenimiento. |
| 5 | **Internet de las Cosas (IoT)** | Conexión de dispositivos y sensores a internet para compartir datos continuamente. |
| 6 | **Cloud Computing** | Servidores remotos para almacenar y procesar datos con acceso global. |
| 7 | **Ciberseguridad** | A mayor conectividad, mayor superficie de ataque y necesidad de protección. |
| 8 | **Manufactura Aditiva** | Impresión 3D para prototipos rápidos y menos desperdicio de material. |
| 9 | **Sistemas Ciberfísicos (CPS)** | Integración total de sistemas computacionales con procesos físicos (ver diagrama en Filmina 21). |

### Actividad en clase — Discusión: Revoluciones Industriales

**Consigna:** para cada etapa, pensar: ¿qué dejó de hacerse a mano? ¿qué habilitó la tecnología que antes era imposible?

- **1ra (Vapor):** se reemplazó la fuerza humana y animal. Nacen las fábricas como concepto.
- **2da (Electricidad):** Ford demostró que dividir el trabajo en tareas repetidas bajaba costos masivamente. Nace el consumo en masa.
- **3ra (Informática):** las máquinas empezaron a recibir instrucciones programadas (PLC), aunque en sistemas aislados entre sí.
- **4ta (Digital):** los sistemas ya no solo ejecutan instrucciones: aprenden, se comunican entre sí y actúan solos.

> Cada revolución no eliminó a la anterior: la incorporó. La 4.0 no existe sin la base eléctrica de la 2.0 ni la automatización de la 3.0.

## Filmina 16 — ¿Qué es la Transformación Digital?

**Qué decir (ampliando lo que dice la filmina):**

- **Digitalización vs. Transformación:** digitalizar es pasar un papel a PDF — el proceso sigue siendo el mismo, solo cambia el soporte. La **Transformación Digital** es distinta: implica rediseñar la estrategia entera del negocio usando capacidades digitales nativas, no solo digitalizar lo que ya existía.
- **El núcleo operativo:** colocar el dato crudo en el centro de la toma de decisiones corporativas. Esto desmantela los silos de información aislados (un área que no comparte datos con otra) y conecta a toda la compañía en tiempo real.

### Los cuatro pilares de la Industria 4.0

- **Interoperabilidad:** sistemas y máquinas que se comunican entre sí sin intervención humana.
- **Ecosistemas conectados:** proveedores, fábricas, distribuidores y clientes integrados en una red de datos.
- **Flexibilidad:** capacidad de reconfigurar la producción rápidamente ante cambios de demanda.
- **Uso intensivo de datos:** cada decisión se basa en métricas reales, no en intuición.

### El Ambiente 4.0: IoT, IoS, IoD e IoP

El ecosistema de la Industria 4.0 no es solo tecnología: es la interacción entre dispositivos, servicios, datos y personas.

- **IoT — Internet de las Cosas:** dispositivos físicos (sensores, actuadores, máquinas) que generan datos del entorno físico.
- **IoS — Internet de los Servicios:** plataformas digitales que procesan esos datos y ofrecen funcionalidades como mantenimiento predictivo o gestión de calidad.
- **IoD — Internet de los Datos:** infraestructura que garantiza disponibilidad, integridad e interoperabilidad de los datos entre sistemas.
- **IoP — Internet de las Personas:** las personas interactuando con los sistemas, aportando conocimiento experto y tomando decisiones estratégicas.

**Tres desafíos técnicos de este ecosistema:** 1) **Latencia** (tiempo entre que el dato se genera y se procesa), 2) **Integridad de los paquetes** (que los datos lleguen sin corrupción), 3) **Ciberseguridad** (proteger la red ante accesos no autorizados).

## Filmina 17 — El Ciclo de Vida del Dato

**Qué decir (ampliando cada tile):**

1. **Captura:** ingesta de datos proveniente de sensores, compras web, aplicaciones de clientes o APIs externas.
2. **Procesamiento:** limpieza de duplicados, corrección de registros inválidos y estandarización del formato.
3. **Modelado:** búsqueda de correlaciones, estimaciones estadísticas y entrenamiento de modelos predictivos.
4. **Acción:** toma de decisiones automatizadas, optimización de recursos y visualización corporativa de métricas.

Esto es, en el fondo, la misma idea que el pipeline técnico que se va a usar en Python durante todo el curso:

```
Ingesta → Limpieza → Transformación → Reproducibilidad
```

**Tecnologías típicas de este pipeline:**

- **Bases de Datos Relacionales (SQL):** MySQL, PostgreSQL, SQL Server — datos estructurados con esquema fijo.
- **Bases de Datos No Relacionales (NoSQL):** MongoDB, Cassandra, Redis — datos flexibles, semiestructurados, gran escalabilidad horizontal.
- **Lenguajes de Programación:** Python (versátil, ML, análisis) y R (estadística avanzada, visualización científica).
- **Cloud Computing:** infraestructura remota bajo modelos IaaS/PaaS/SaaS — AWS, Microsoft Azure, Google Cloud Platform.

### Caso real — Mantenimiento Predictivo con IoT

Una planta embotelladora perdía el **15% de su producción** por paradas inesperadas en la cinta transportadora.

**Solución aplicada:** se instalaron sensores de vibración y temperatura en los motores (Captura), los datos se enviaron a la nube en tiempo real (Procesamiento), y un modelo de IA detectó patrones que preceden a una falla, anticipándola **48 horas antes** (Modelado → Acción).

> *¿Qué es más barato: cambiar un rodamiento de $50 hoy, o detener la planta 5 horas mañana?* El costo de la intervención planificada siempre es menor al costo de la falla no anticipada.

## Filmina 18 — Ética y Privacidad del Dato

**Qué decir (ampliando lo que dice la filmina):**

- **Sesgo algorítmico:** un modelo aprende exactamente los patrones que hay en sus datos de entrenamiento — incluidos los prejuicios humanos presentes en la historia de esos datos. Ejemplo clásico: un algoritmo de selección de personal entrenado con contrataciones pasadas puede aprender a discriminar por género o edad simplemente porque así se contrataba antes, aunque nadie se lo haya pedido explícitamente. Evitar el sesgo es un desafío activo, no algo que se resuelve solo por usar un algoritmo "objetivo" — los datos nunca son neutrales.
- **Regulaciones globales:** normativas como el **GDPR** (Europa) o la **LGPD** (Brasil) regulan cómo se puede recolectar, almacenar y usar información personal, y exigen consentimiento explícito. En Argentina, la referencia local es la **Ley de Protección de Datos Personales (25.326)**. El incumplimiento no es solo un problema ético — trae multas severas y pérdida de confianza del usuario.

**Punto clave para remarcar:** un algoritmo extraordinario es peligroso si se alimenta de datos capturados sin consentimiento legal y ético — la potencia técnica de un modelo no lo exime de sus obligaciones legales ni éticas.

## Filmina 19 — Toma de Decisiones Guiada por Datos

**Qué decir (ampliando lo que dice la filmina):**

- **Decisiones por intuición ("Gut Feeling"):** históricamente los líderes empresariales decidían basándose en su instinto o experiencia previa no estructurada. Este enfoque está sujeto a sesgos cognitivos personales y falta de precisión empírica en entornos de alto riesgo.
- **Decisiones Data-Driven:** se desprenden directamente de lo que revelan los datos estructurados, mitigando el error subjetivo. Al basarse en hechos empíricos y métricas medibles, se optimiza el uso de recursos y se maximiza el retorno.

A diferencia del petróleo o el capital, los datos **no se agotan al usarse**: cada análisis genera nuevo conocimiento que retroalimenta el sistema. Su impacto se ve en tres frentes: **operativo** (detectar cuellos de botella, reducir desperdicios en tiempo real), **logístico** (gestión de inventarios, anticipar demanda, optimizar rutas) y de **respuesta ante crisis** (decisiones fundamentadas para minimizar el impacto de fallas).

### KPIs Relevantes en Industria 4.0

Sin datos no hay KPIs; sin KPIs no hay mejora continua.

| KPI | Descripción | Aplicación |
|----|-------------|-----------|
| **Tasa de defectos** | % de productos defectuosos sobre el total producido. | Control de calidad |
| **Tiempo de ciclo** | Duración promedio para completar un proceso de principio a fin. | Optimización de producción |
| **Nivel de inventario** | Cantidad de stock disponible en un momento dado. | Gestión logística |
| **Rotación de inventario** | Frecuencia con que se renueva el stock en un período. | Eficiencia en retail |

## Filmina 20 — Ciclo de un Proyecto Data Science

**Qué decir (ampliando cada bullet):**

1. **Entender el negocio:** definir qué problema real se quiere resolver y mapear las métricas de éxito del proyecto — sin esto, el resto del ciclo no tiene rumbo.
2. **Preparar la información:** el 80% de un proyecto analítico consiste en limpiar y transformar los datos en bruto (esto se retoma con la respuesta del Kahoot en la Filmina 22).
3. **Iterar y validar:** ajustar los modelos numéricos hasta garantizar precisión empírica antes del despliegue productivo.

**Vista desde el lado del negocio**, este mismo ciclo suele describirse en cuatro fases más formales — saltearse alguna genera proyectos que fallan al escalar:

| Fase | Objetivos Clave | Entregables | Criterios de Éxito |
|------|----------------|-------------|-------------------|
| **1. Ideación** | Definir el problema, el alcance y las metas. | Propuesta y análisis de viabilidad. | Alineación estratégica y compromiso del equipo. |
| **2. Prototipo** | Validar el concepto a baja escala con el mínimo esfuerzo. | MVP (Producto Mínimo Viable). | Validación técnica y feedback de usuarios. |
| **3. Piloto** | Evaluar la solución en un entorno real y controlado. | Informe de piloto y análisis de KPIs. | Cumplimiento de KPIs definidos en la ideación. |
| **4. Escalado** | Despliegue masivo, optimización y capacitación. | Plan de escalado y documentación. | Adopción general y ROI positivo. |

> **Fase Piloto** ≠ producción. Es probar a escala limitada para medir KPIs concretos antes de comprometer recursos masivos.

**El cierre de todo ciclo — el Plan Ejecutivo Data-Driven:** la etapa final es traducir los hallazgos técnicos en decisiones estratégicas de negocio. El objetivo es comunicar valor, no tecnología: redactar el plan evitando tecnicismos innecesarios para los líderes del negocio, priorizar acciones con una **matriz de Impacto vs. Esfuerzo** (atacar primero lo de alto impacto y bajo esfuerzo), y presentar resultados con visualizaciones claras que cuenten una historia (**data storytelling**).

## Filmina 21 — El ADN: Los Datasets

**Qué decir (ampliando lo que dice la filmina):**

- **Formato Tidy:** cada variable representa una columna, cada observación es una fila y cada tipo de unidad observacional constituye una tabla distinta. Es el estándar que se va a usar en Pandas en las próximas clases.
- **Consistencia:** es vital resolver campos vacíos, registros fuera de rango y errores de formato para no alterar los cálculos de los modelos estadísticos. Un modelo predictivo es tan bueno como la calidad del dataset con el que fue entrenado.

### Sistemas Ciberfísicos: de dónde salen estos datos

Los **Sistemas Ciberfísicos (CPS)** son la columna vertebral de la fábrica inteligente y la fuente original de muchos datasets industriales — combinan componentes físicos con computación embebida, operando en ciclo cerrado:

```
[Entorno Físico] → Sensores → Procesamiento/IA → Actuadores → [Entorno Físico]
                                      ↑_____________________________|
```

- **Sensores:** adquieren datos del entorno físico (temperatura, presión, velocidad, posición).
- **Unidad de Procesamiento y Control:** analiza los datos en tiempo real con algoritmos de control o modelos predictivos.
- **Actuadores:** ejecutan correcciones mecánicas automáticas (válvulas, motores, brazos robóticos).

### Caso real — Control de Calidad Inteligente

| Campo | Detalle |
|-------|---------|
| Proceso ineficiente | Inspección manual de piezas en línea de ensamblaje |
| Tecnología aplicada | Visión artificial + IA |
| KPI a mejorar | Tasa de defectos |
| Resultado esperado | Reducción de devoluciones en un 20% |

## Filmina 22 — Evaluación de Conceptos (Kahoot)

Momento de preparar al grupo para el Kahoot gamificado. Dos enunciados para discutir en voz alta antes de jugar — **ambos son falsos**, y vale la pena remarcar por qué:

- **Enunciado A:** *"Si automatizamos un proceso de reporte ineficiente, hemos logrado una verdadera Transformación Digital."* — **Falso.** Automatizar un proceso malo solo lo hace un proceso malo más rápido (ver la distinción Digitalización vs. Transformación de la Filmina 16). La transformación real implica rediseñar el proceso, no solo acelerarlo.
- **Enunciado B:** *"El 80% del tiempo de un científico de datos se dedica a refinar parámetros del modelo de Machine Learning."* — **Falso.** En la práctica, el 80% del tiempo se va en **preparar la información** (limpieza, transformación) — ya mencionado en la Filmina 20 — no en el modelado en sí.

### Preguntas de Repaso (Bloque 1 y Bloque 2)

Útiles como repaso rápido antes o después del Kahoot:

**¿Por qué los datos son un recurso estratégico?**
Porque permiten tomar decisiones operativas basadas en el desempeño real de los procesos, anticipar problemas y optimizar recursos de forma continua. A diferencia de otros recursos, no se agotan al usarse.

**¿Qué caracteriza a la Industria 4.0?**
Permite la producción flexible y personalización en masa mediante ecosistemas interconectados de máquinas, datos y personas. Los sistemas no solo se automatizan, sino que aprenden y se adaptan.

**¿Qué caracteriza a Python como lenguaje?**
Es interpretado (permite prueba rápida de código), de tipado dinámico y orientado a objetos. Al asignar una lista a otra variable (`lista_b = lista_a`), ambas referencian el mismo objeto en memoria (mutabilidad) — usar `.copy()` para copiar de forma independiente.

**¿Qué es una fase Piloto?**
Probar la solución en un entorno real a escala limitada para medir KPIs concretos antes del despliegue masivo. Es distinta al prototipo porque se usa con datos y condiciones reales.

**¿Cuál es el rol de los sensores en un CPS?**
Capturar datos del entorno físico (temperatura, presión, movimiento) para su posterior análisis y acción automática por parte de los actuadores.

## Filmina 23 — Consolidación de Conceptos

| Módulo | Tema Principal | Herramienta o Técnica Clave | Meta Pedagógica |
|---|---|---|---|
| **Módulo 1** | Nivelación técnica de código | Conda, Jupyter Notebook & Python Base | Descomponer problemas y estructurar flujos secuenciales básicos mediante listas y diccionarios. |
| **Módulo 2** | Transformación Digital | Ciclo de vida del dato & Datasets | Conceptualizar la gobernanza, ética y diseño técnico de datasets limpios para resolver problemas. |
| **Evaluación** | Prueba integradora final | Caso práctico y Kahoot gamificado | Aplicación holística de soluciones analíticas en el contexto de la Industria 4.0 moderna. |

### Actividad grupal de cierre — Plan Ejecutivo Data-Driven

**Consigna:** elegir un proceso ineficiente (de una empresa real o inventada) y armar un plan de mejora con datos.

1. **Identificar** el proceso con problema: ¿dónde se pierden tiempo, dinero o calidad?
2. **Seleccionar** 1 o 2 tecnologías de los 9 pilares de la Industria 4.0 (Filmina 15) que podrían resolverlo.
3. **Definir el KPI** que va a medir el éxito: tiene que ser un número concreto, no una sensación.

*Tip: usar la matriz Impacto vs. Esfuerzo (Filmina 20) para priorizar. Atacar primero lo de alto impacto y bajo esfuerzo.*

---

# Cierre de la Clase

## Filmina 24 — Pre-Entrega 1: Selección del Dataset

Es la última filmina de contenido antes de cerrar la clase — el **primer hito evaluable** de todo el curso. Todo lo que se construya en las próximas clases (limpieza, visualización, modelos) se hace **sobre este mismo dataset**, así que una mala elección acá se paga durante meses.

### Objetivo de la entrega

Seleccionar el conjunto de datos con el que se va a trabajar durante **todo el curso** para el proyecto final. No alcanza con encontrar un archivo — hay que justificar la elección, identificar el problema de negocio a resolver, y analizar la estructura inicial de los datos.

### Qué hay que entregar: las 4 secciones

**1. Contexto de Negocio y Motivación**
- ¿Cuál es el tema elegido? (Finanzas, E-commerce, Deportes, Salud, etc.)
- ¿Por qué es importante este problema? Explicar brevemente el contexto y qué valor aportaría resolverlo.
- *Tip:* elegir un tema que genuinamente interese — se va a trabajar con este dataset varias semanas, y el interés personal sostiene la motivación cuando el análisis se pone difícil.

**2. Definición del Problema de Data Science**
- Plantear una pregunta o hipótesis clara que se quiera resolver con los datos.
- Definir si el problema es de **Clasificación** (la respuesta es una categoría: sí/no, tipo A/B/C) o de **Regresión** (la respuesta es un número continuo: precio, cantidad, monto).
- *Tip:* una buena pregunta es específica y accionable — "¿qué factores influyen en las ventas?" es vago; "¿puedo predecir si un cliente va a cancelar su suscripción el próximo mes?" ya define target, unidad de análisis y tipo de problema.

**3. Ficha Técnica del Dataset**
- **Origen de los datos:** enlace directo y funcional a la fuente.
- **Volumen inicial:** cantidad aproximada de filas (observaciones) y columnas (variables).
- **Nota mínima:** se sugiere un dataset de **al menos 5.000 a 10.000 filas** y **un mínimo de 12 variables** — para garantizar que haya "tela para cortar" durante toda la cursada (limpieza, EDA, visualización, modelos supervisados y no supervisados).

**4. Diccionario de Datos Preliminar**
- Nombrar y describir brevemente **al menos 5 o 6 variables clave** del dataset, indicando qué representa cada una.
- Esto obliga a abrir el archivo y realmente mirar las columnas, no solo confiar en la descripción de la página de origen — muchas veces el nombre de una columna no alcanza para entender qué mide.

### Dónde buscar el dataset

| Fuente | Qué ofrece | A tener en cuenta |
|---|---|---|
| **[Kaggle](https://www.kaggle.com/datasets)** | El catálogo más grande y variado — miles de datasets subidos por la comunidad, la mayoría con descripción, diccionario de datos y hasta notebooks de ejemplo ya resueltos. | Justamente por tener notebooks de ejemplo, conviene evitar los datasets "de tutorial" clásicos (ver más abajo) — la idea es un problema propio, no repetir un ejercicio ya hecho mil veces. |
| **[datos.gob.ar](https://datos.gob.ar)** | El portal de datos públicos abiertos del Estado argentino — salud, economía, transporte, educación, seguridad. Son datos reales de organismos oficiales. | Suelen venir menos "curados" que los de Kaggle (columnas con nombres largos, formatos que hay que ajustar) — es una buena forma de practicar limpieza real desde el día uno. |
| **UCI Machine Learning Repository** | El repositorio académico clásico, con datasets muy documentados — la fuente original de muchos datasets famosos. | Varios de sus datasets más conocidos (Titanic, Iris) son justamente los que hay que evitar por ser demasiado simples. |
| **Google Dataset Search** | Un buscador que indexa datasets de miles de fuentes distintas (no es una fuente en sí, es un motor de búsqueda). | Útil cuando ya se tiene un tema en mente y se quiere encontrar datos específicos de ese sector, en vez de navegar un catálogo general. |

### Ejemplo de cómo debería verse la entrega

> **1. Contexto de Negocio:** Elegí un dataset de un e-commerce de ropa. El objetivo es entender el comportamiento de los clientes para optimizar las campañas de marketing y evitar que dejen de comprar.
>
> **2. Problema de Data Science:** El objetivo es predecir si un cliente activo va a abandonar la plataforma el próximo mes (Churn). Es un problema de Aprendizaje Supervisado: Clasificación Binaria (Abandona / No Abandona).
>
> **3. Ficha Técnica:** Fuente: [Enlace a Kaggle - E-commerce Customer Dataset]. Tamaño: 5.500 filas y 12 columnas.
>
> **4. Diccionario de Datos (muestra):**
> - `Customer_ID`: Identificador único del cliente (Numérico/Categórico).
> - `Tenure`: Meses que el cliente lleva usando la plataforma (Numérico).
> - `Complain`: Si el cliente hizo un reclamo en el último mes (Binario: 0 = No, 1 = Sí).
> - `Churn`: Si el cliente abandonó la app (Variable Target / Etiqueta).

### Pasos sugeridos para resolverla

1. **Exploración y búsqueda:** navegar Kaggle, datos.gob.ar, UCI ML Repository o Google Dataset Search. Buscar un tema que interese profesionalmente (finanzas, salud, marketing, deportes, lo que sea).
2. **Definición del problema:** redactar el contexto de negocio y decidir si el problema es de Clasificación o de Regresión.
3. **Validación de factibilidad:** confirmar que el dataset cumple el volumen mínimo (5.000-10.000 filas, ≥12 columnas) — y **evitar datasets excesivamente simples como Titanic o Iris**, que no permiten un análisis profundo durante todo el curso.
4. **Documentación:** un único documento (PDF o DOCX) con las 4 secciones organizadas.
5. **Diccionario de datos:** describir al menos 5 variables críticas — nombre, tipo de dato, y qué representan en el mundo real.

### Checklist de entrega

- [ ] Documento único en PDF o DOCX.
- [ ] Las 4 secciones: Contexto de negocio, Problema de Data Science (Clasificación o Regresión), Ficha técnica y Diccionario de datos.
- [ ] Enlace público y funcional a la fuente (Kaggle / datos.gob.ar / UCI / Google Dataset Search).
- [ ] Dataset con al menos 5.000 filas y 12 columnas (evitar Titanic / Iris).
- [ ] Diccionario con al menos 5 variables (nombre, tipo de dato y significado).

**Qué tenés que entregar:** un documento único en formato PDF o DOCX que contenga la justificación del proyecto, el enlace a la fuente y la ficha técnica del dataset seleccionado.

## Filmina 25 — ¿Dudas? ¿Consultas?

Espacio abierto de preguntas antes del cierre formal. Buen momento para repasar en voz alta, sin filmina de por medio, los dos puntos donde más suelen surgir dudas: la diferencia entre Clasificación y Regresión (Pre-Entrega, Filmina 24) y el volumen mínimo exigido para el dataset.

## Filmina 26 — Cierre / Q&A

**Tres ideas para llevarse de esta clase:**

**1. La evolución es constante.**
La Industria 4.0 no es un destino, es un proceso en curso. Las organizaciones que no adoptan una mentalidad de mejora basada en datos quedan fuera de competencia.

**2. La conectividad es la base.**
Sin IoT no hay datos. Sin datos no hay modelos. Sin modelos no hay automatización inteligente. Todo empieza por conectar el mundo físico al digital.

**3. No alcanza con la tecnología: hace falta cultura data-driven.**
Una organización data-driven es aquella donde las decisiones —en todos los niveles— se basan en datos reales, no en intuición o jerarquía. Implica medir todo lo que se pueda, iterar rápido y democratizar el acceso a la información. Si los líderes siguen decidiendo por instinto y los equipos no confían en los datos, la transformación digital no ocurre aunque tengan todo el stack tecnológico.

> *"Los datos son el nuevo petróleo, pero la analítica es el motor que los procesa."*
