# Clase 07: Pipelines Reproducibles y Casos de Uso de ML en la Industria — Guía Completa para el Docente

Esta guía es el **libreto de apoyo para dictar la Clase 07**. Reúne, en un solo lugar y con más profundidad de la que entra en una diapositiva, toda la teoría de:

- **`Clase 07.pdf`** — el material teórico oficial de la unidad.
- **`Semana 7.html`** — las diapositivas que se proyectan en clase (44 filminas, mismo orden que este documento).
- **`Clase_7_Fundamentos_de_Ciencia_de_Datos_1_.ipynb`** — el notebook con el ejercicio práctico en vivo: un pipeline reproducible + segmentación con K-Means sobre el dataset real de natalidad del DEIS.

A diferencia del PDF/HTML (que explican los 4 casos de éxito con nombre propio — San Cristóbal, Medplaya, Amazon, Mazda), el notebook **no los replica en código**: es un quinto ejercicio, autocontenido y con datos reales, que aplica exactamente la misma lógica de fondo (pipeline → modelo no supervisado → validación → traducción a una recomendación de negocio) que el caso **Mazda**. Este documento explica ambas cosas por separado y después las conecta.

---

## Índice

0. [Mapa rápido de la clase](#0-mapa-rápido-de-la-clase)
0.1. [Introducción General de la Clase](#introducción-general-de-la-clase)
1. [Módulo 1 — Principios de Diseño de Pipelines Reproducibles](#módulo-1--principios-de-diseño-de-pipelines-reproducibles)
2. [Módulo 2 — Casos de Estudio: Segmentación y Recomendaciones](#módulo-2--casos-de-estudio-segmentación-y-recomendaciones)
3. [Módulo 3 — Casos de Éxito: ML en Acción](#módulo-3--casos-de-éxito-ml-en-acción)
4. [Módulo 4 — Supervisado vs. No Supervisado](#módulo-4--supervisado-vs-no-supervisado)
5. [Break del Coder](#break-del-coder)
6. [Módulo 5 — Métricas y Estrategias de Validación](#módulo-5--métricas-y-estrategias-de-validación)
7. [El ejercicio práctico del notebook, explicado en profundidad](#7-el-ejercicio-práctico-del-notebook-explicado-en-profundidad)
8. [Preguntas frecuentes y errores típicos a anticipar](#preguntas-frecuentes-y-errores-típicos-a-anticipar)
9. [Material de la clase](#material-de-la-clase)

---

## 0. Mapa rápido de la clase

| # | Módulo | Slides | Idea central |
|---|--------|--------|---------------|
| 0 | **Repaso Clase 6** (Bloque 0 del notebook) | — (no está en el PDF/HTML) | Los 4 pilares de estadística/preprocesamiento, repasados sobre el dataset de hoy antes de empezar |
| 1 | Principios de Diseño de Pipelines Reproducibles | 03–08 | Un pipeline que cualquiera pueda replicar y auditar |
| 2 | Casos de Estudio: Segmentación y Recomendaciones | 09–13 | Dos aplicaciones clave de ML en la industria (teoría general) |
| 3 | Casos de Éxito: ML en Acción | 14–29 | 4 casos reales con nombre propio: San Cristóbal, Medplaya, Amazon, Mazda |
| 4 | Supervisado vs. No Supervisado | 30–34 | Cuándo usar cada enfoque |
| — | **Break del Coder** | 35 | Corte de ~10 minutos |
| 5 | Métricas y Estrategias de Validación | 36–42 | Cómo saber si un modelo es realmente bueno |
| — | **Ejercicio práctico (notebook)** | — | Bloque 0 + Pipeline + K-Means sobre natalidad real — aplica los módulos 1, 4 y 5 |

> **Nota sobre el notebook**: arranca con un **Bloque 0 (Repaso de Clase 6)** que no pertenece a esta unidad — es intencional y está bien dejarlo, sirve de entrada en calor. El resto (Bloques 1 a 4) corre en paralelo a los Módulos 1, 4 y 5 de esta guía, pero **no cubre el Módulo 2 ni el Módulo 3** (segmentación/recomendación teórica ni los 4 casos de éxito con nombre propio) — el notebook ya tiene, en su Bloque 1 y Bloque 2, celdas de teoría agregadas que tienden puentes explícitos hacia esos casos para que la clase quede conectada aunque no se repliquen en código.

---

## Introducción General de la Clase

### El disparador teórico del PDF

El material oficial abre la unidad con esta pregunta: *¿Qué hace que un pipeline de ciencia de datos sea realmente reproducible?* Y la plantea con un escenario muy concreto: desarrollaste un modelo predictivo que mejora significativamente la toma de decisiones en tu empresa, pero cuando un colega intenta replicar tu trabajo, los resultados no coinciden. Ese desajuste es, en la práctica, el problema número uno que esta clase busca resolver: **la reproducibilidad es lo que determina si un proyecto de ciencia de datos es confiable y escalable en la industria, o si se queda como un experimento aislado que nadie más puede usar.**

Conviene abrir la clase con esa pregunta tal cual, en voz alta, antes de mostrar ninguna diapositiva — es la misma pregunta que dispara el Módulo 1, y funciona bien como gancho porque casi todos los alumnos ya vivieron una versión de ese problema ("en mi máquina andaba").

### El diagrama que abre el PDF

La primera página del material resume el recorrido completo de la clase con un diagrama de 7 etapas encadenadas — vale la pena dibujarlo en el pizarrón o proyectarlo al empezar:

```
Ingesta de datos → Procesamiento / EDA → Feature Engineering → Modelado → Evaluación → Artefactos Reproducibles → Entrega Mínima
```

Este diagrama es el "mapa madre" de toda la unidad: el **Módulo 1** lo explica en detalle (qué hace reproducible a cada etapa), los **Módulos 2 y 3** muestran ese mismo pipeline aplicado en 4 empresas reales, el **Módulo 4** profundiza en la etapa de "Modelado" (qué tipo de algoritmo elegir según el problema) y el **Módulo 5** profundiza en la etapa de "Evaluación" (cómo saber si el modelo realmente funciona). Todo lo que viene en la clase es, en el fondo, un zoom progresivo sobre distintas partes de este mismo diagrama — es útil volver a él verbalmente entre módulo y módulo ("ahora estamos en la etapa de Evaluación de este mismo pipeline").

### Qué se lleva el alumno al final de la clase

Según el propio material, al cerrar esta unidad el alumno debería poder:

1. Diseñar y explicar la estructura de un pipeline end-to-end reproducible.
2. Comparar segmentación de clientes vs. sistemas de recomendación y elegir cuál aplica a un problema de negocio dado.
3. Identificar, frente a un caso real, si conviene un enfoque supervisado o no supervisado.
4. Elegir la métrica y la estrategia de validación correctas según el tipo de problema (clasificación, regresión o clustering) y las restricciones del negocio.

### La conexión con el repaso del notebook

El notebook no abre directamente con este diagrama: primero hace un ejemplito de 10 minutos que repasa 4 conceptos de la clase pasada (detallado en la [sección 7, Bloque 0](#bloque-0--un-ejemplito-para-repasar-4-conceptos-de-la-semana-6)), porque esos cuatro pilares (limpieza, estadística descriptiva, distribuciones/correlación, transformación/reducción) son insumo directo de las etapas 2 y 3 del pipeline de hoy (Procesamiento/EDA y Feature Engineering). Es una forma de que el repaso no quede "suelto": se siente como el cimiento sobre el que se construye el pipeline reproducible del Bloque 1.

---

## Módulo 1 — Principios de Diseño de Pipelines Reproducibles

**Pregunta disparadora para abrir la clase**: imaginá que desarrollaste un modelo predictivo que mejora significativamente la toma de decisiones en tu empresa. Un colega intenta replicar tu trabajo... y los resultados no coinciden. ¿Qué salió mal? La reproducibilidad es lo que separa un experimento de laboratorio de un proyecto de ciencia de datos confiable y escalable en la industria.

**Analogía útil para el pizarrón**: un pipeline reproducible es como una **receta de cocina bien escrita**, no como "cocinar de memoria". Si la receta especifica ingredientes exactos (los datos versionados), pasos numerados (el código modular) y el punto de cocción (las métricas de evaluación), cualquier persona puede reproducir el mismo plato. Si en cambio el chef improvisa "un poco de esto, un poco de aquello" (celdas de Jupyter sueltas, ejecutadas en cualquier orden, con decisiones que solo están en la cabeza de quien las tomó), el resultado depende de quién cocine — y eso es exactamente lo que un pipeline reproducible busca eliminar.

### 1.1 ¿Qué es un pipeline end-to-end en ciencia de datos?

Un **pipeline end-to-end** es un flujo completo que transforma datos crudos en un modelo funcional y evaluado, listo para su uso o despliegue. Incluye seis etapas:

1. **Ingestión de datos**: recolección y carga desde diversas fuentes.
2. **Análisis exploratorio de datos (EDA)**: comprensión inicial, detección de patrones y limpieza.
3. **Feature engineering**: creación y selección de variables relevantes.
4. **Modelado**: entrenamiento de algoritmos para aprender patrones.
5. **Evaluación**: medición del desempeño con métricas adecuadas.
6. **Entrega mínima**: preparación para compartir o desplegar el modelo (notebooks reproducibles, servicios simples).

**Para remarcar en clase**: este pipeline debe estar diseñado para que **cualquier persona** pueda seguirlo y obtener resultados consistentes — esa es, literalmente, la definición de reproducibilidad.

### 1.2 Componentes clave para la reproducibilidad

| Componente | Qué implica |
|---|---|
| **Gestión de artefactos** | Guardar versiones de datasets, modelos entrenados y resultados intermedios. |
| **Control de versiones** | Usar Git para rastrear cambios en código y documentación. |
| **Entornos reproducibles** | Definir dependencias y versiones de librerías para evitar discrepancias entre máquinas. |
| **Documentación clara** | Explicar cada paso y decisión tomada. |

### 1.3 Prácticas para despliegue mínimo y compartición

El **despliegue mínimo** busca entregar una versión funcional del pipeline que permita a otros reproducir y validar resultados sin complejidades innecesarias:

- Notebooks bien estructurados (Jupyter, Google Colab) que integren código, visualizaciones y explicaciones.
- Repositorios organizados con código, datos y documentación.
- Artefactos guardados con nombres y formatos estándar.
- Demos simples con herramientas como **Streamlit** o **Flask** para mostrar resultados interactivos.

### 1.4 Por qué importa en la industria

En un proyecto de **detección de fraude**, un pipeline reproducible permite que el equipo de ingeniería valide el modelo antes de integrarlo en producción, asegurando que los resultados sean consistentes y confiables. La gestión de artefactos y el control de versiones evitan pérdidas de trabajo y facilitan la colaboración entre equipos multidisciplinarios; el despliegue mínimo permite entregar prototipos funcionales que stakeholders pueden evaluar sin infraestructuras complejas.

**Gancho hacia el ejercicio práctico**: la función `pipeline_preprocesamiento()` del notebook (Bloque 1) implementa exactamente las etapas 1-3 de este módulo (ingesta, limpieza, transformación) sobre datos reales — es el módulo teórico convertido en código ejecutable.

---

## Módulo 2 — Casos de Estudio: Segmentación y Recomendaciones

**Pregunta disparadora**: trabajás en una empresa de comercio electrónico que quiere aumentar sus ventas personalizando la experiencia de sus clientes. ¿Cómo identificar grupos de clientes con comportamientos similares? ¿Cómo recomendar productos que realmente interesen a cada usuario? Estas preguntas son el corazón de dos aplicaciones clave de ML en la industria.

### 2.1 Segmentación de Clientes

Dividir una población de clientes en grupos homogéneos según características o comportamientos similares, para personalizar estrategias de marketing, mejorar la retención y optimizar recursos.

- **Métodos**: clustering no supervisado (K-means, DBSCAN, clustering jerárquico) o segmentación basada en reglas definidas por expertos.
- **Requisitos de datos**: variables relevantes y limpias (demográficas, transaccionales, comportamiento web), con volumen suficiente para detectar patrones significativos.
- **Métricas de éxito**: Silhouette Score (cohesión y separación de clusters) e impacto en KPIs comerciales (ventas, retención, conversión).

### 2.2 Sistemas de Recomendación

Sugerir productos o contenidos personalizados para cada usuario, aumentando satisfacción y ventas.

- **Tipos**: filtrado colaborativo (similitud entre usuarios o ítems), filtrado basado en contenido (características del producto + preferencias del usuario), modelos híbridos (combinan ambos).
- **Requisitos de datos**: historial de interacciones usuario-producto, información contextual (tiempo, ubicación).
- **Métricas de éxito**: precisión y recall, tasa de clics (CTR) y conversión, diversidad y novedad (para evitar recomendaciones repetitivas).

### 2.3 Comparación y Trade-offs

| Aspecto | Segmentación | Recomendaciones |
|---|---|---|
| Tipo de ML | No supervisado | Supervisado / híbrido |
| Objetivo | Agrupar clientes | Personalizar experiencia |
| Datos requeridos | Variables descriptivas | Interacciones usuario-producto |
| Métricas clave | Cohesión de grupos, impacto negocio | Precisión, CTR, diversidad |
| Complejidad | Moderada | Alta (requiere modelado avanzado) |

### 2.4 Aplicación práctica combinada

En la práctica, una empresa puede usar **segmentación** para identificar grupos de clientes con alta propensión a comprar un nuevo producto, y luego aplicar **sistemas de recomendación** para personalizar ofertas dentro de cada segmento. Ejemplo: un retailer online segmenta a sus clientes por frecuencia de compra y categorías preferidas, y luego usa un recomendador híbrido para sugerir productos nuevos o complementarios — maximizando el impacto comercial y el uso de datos y recursos.

**Concepto clave que atraviesa toda la unidad — "Analytics to Action"**: ni la segmentación ni la recomendación generan valor por sí solas. Un cluster de clientes o un score de similitud entre productos es solo el punto medio del pipeline (etapa "Evaluación" del diagrama de la Introducción); el valor de negocio aparece recién cuando ese resultado técnico se traduce en una **decisión o acción concreta**: una campaña dirigida, una oferta personalizada, una alerta. Este es el hilo que conecta este módulo con los 4 casos del Módulo 3 y con el cierre del ejercicio práctico del notebook (Bloque 4).

---

## Módulo 3 — Casos de Éxito: ML en Acción

Cuatro casos reales que muestran cómo se ve todo lo anterior aplicado en la industria. Conviene presentarlos como historias, no como listas de bullets — cada uno tiene un problema de negocio, una técnica y un cierre con impacto medible.

### 3.1 San Cristóbal — Detección de Fraudes

**El problema**: ¿cómo protegerse eficazmente contra fraudes que amenazan la operación y la confianza con los clientes? La detección de fraude es un problema clásico de **clasificación supervisada**: identificar transacciones o eventos fraudulentos entre un gran volumen de datos legítimos.

**El pipeline en 3 etapas**:
1. **Detección inicial**: modelos supervisados entrenados con datos históricos clasifican eventos como fraudulentos o no.
2. **Investigación**: análisis detallado, incluyendo revisión manual y análisis de imágenes de siniestros mediante Deep Learning.
3. **Resolución**: toma de decisiones basada en resultados y métricas para mitigar el fraude.

**El desafío del desbalance**: los fraudes son eventos raros, por lo que los datos están altamente desbalanceados. Se abordan con:
- **Oversampling**: aumentar artificialmente los ejemplos minoritarios (fraudes) — la técnica más usada es **SMOTE**.
- **Undersampling**: reducir los ejemplos de la clase mayoritaria.

**Métricas críticas**: **Precisión** (proporción de predicciones positivas correctas) y **Recall** (proporción de fraudes reales detectados) — en este dominio, **un alto recall es vital** para no dejar pasar fraudes, aunque se sacrifiquen algunos falsos positivos. El **AUC** (área bajo la curva ROC) mide la capacidad de distinguir clases en distintos umbrales, y se usa **cross-validation** para asegurar la robustez del modelo.

**El plus de Deep Learning**: San Cristóbal complementa la detección tabular con **redes neuronales convolucionales (CNN)** que analizan imágenes de siniestros, extrayendo características automáticas y detectando patrones visuales indicativos de fraude. **Matplotlib** se usa para visualizar resultados y comunicar hallazgos a stakeholders.

**Código de referencia de la práctica** (dataset: `csv/creditcardfraud`):

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier

smote = SMOTE(random_state=RANDOM_STATE)
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)

pipeline_smote_rf = ImbPipeline(steps=[
    ('smote', smote),
    ('rf', rf)
])
pipeline_smote_rf.fit(X_train, y_train)
```

La práctica completa pide: carga y EDA del desbalance → oversampling/undersampling → entrenar Random Forest o Logistic Regression → calcular precisión/recall/F1/AUC, matriz de confusión y curva ROC → interpretar y justificar la elección de métricas y técnicas.

### 3.2 Medplaya — Analítica Predictiva en Hotelería

**El problema**: las cancelaciones de reservas afectan la ocupación y los ingresos de una cadena hotelera. ¿Cómo anticiparlas para optimizar la gestión de habitaciones y maximizar el revenue?

**Modelos de clasificación supervisada** para predecir si una reserva será cancelada:
- **Árboles de decisión**: interpretables, segmentan el espacio de características.
- **Random Forest**: ensamble de árboles, mejora precisión y reduce sobreajuste.
- **Regresión logística**: modelo probabilístico para clasificación binaria.
- **Boosting (XGBoost)**: potencia modelos débiles para mejorar rendimiento.

**Selección de features**:
- **Comportamiento histórico**: frecuencia de cancelaciones previas, tiempo entre reserva y llegada.
- **Señales contextuales**: temporada, eventos locales, tipo de habitación, canal de reserva.
- **Feature engineering**: transformaciones numéricas, one-hot encoding, variables derivadas (ej. tasa de cancelación por cliente).

**Desequilibrio**: las cancelaciones son menos frecuentes que las confirmaciones — se aborda con re-muestreo o algoritmos que ponderan clases.

**Métricas de evaluación**:

| Métrica | Descripción | Importancia en cancelaciones |
|---|---|---|
| Precisión | Proporción de predicciones correctas | Evita falsas alarmas de cancelación |
| Recall | Proporción de cancelaciones detectadas | Crucial para anticipar cancelaciones reales |
| F1-Score | Balance entre precisión y recall | Útil en desequilibrio de clases |
| AUC-ROC | Capacidad de distinguir entre clases | Evalúa rendimiento global del modelo |

**Pregunta para lanzar a la clase (viene textual del PDF)**: *¿por qué podría ser más importante maximizar el recall que la precisión en este caso?* (Respuesta esperada: una cancelación no detectada cuesta más que una falsa alarma — se pierde la oportunidad de re-vender esa habitación.)

**El cierre de negocio — Overbooking Controlado**: con predicciones confiables, se acepta más reservas que la capacidad real para compensar cancelaciones esperadas, optimizando ocupación y revenue. Requiere un balance cuidadoso para evitar sobreventa y mala experiencia al cliente. Medplaya aplicó este pipeline completo (limpieza → features → Random Forest + regresión logística → evaluación con F1/AUC-ROC → política de overbooking) y logró aumentar la ocupación promedio y mejorar el revenue.

### 3.3 Amazon — Sistemas de Recomendación

**El problema**: ¿cómo logra Amazon ofrecer recomendaciones precisas entre millones de productos y usuarios?

**Tres enfoques**: filtrado colaborativo, basado en contenido, y sistemas híbridos (la combinación de ambos suele ofrecer mejores resultados, equilibrando precisión y diversidad).

**Matrix Factorization y Embeddings**: la técnica central del filtrado colaborativo. Representa las interacciones usuario-ítem en espacios latentes:
- **SVD** (Singular Value Decomposition): descompone la matriz de interacciones en factores latentes que capturan características implícitas.
- **ALS** (Alternating Least Squares): optimiza iterativamente la factorización, eficiente para grandes conjuntos de datos.

Estas técnicas generan **embeddings** que permiten predecir la afinidad entre usuarios e ítems no observados. *Ejemplo del PDF*: en Amazon, la matriz usuario-producto es enorme y dispersa; ALS permite factorizarla para descubrir patrones latentes y recomendar productos relevantes.

**Impacto en el negocio**: se mide con **experimentación A/B** (comparar grupos con y sin recomendaciones) y métricas de negocio (ROI, tasa de conversión, valor promedio de pedido).

**Consideraciones de despliegue en tiempo real**: minimizar latencia, actualizar modelos periódicamente, integrar pipelines de scoring batch y streaming.

**Práctica: recomendador de películas con NLP + similitud de coseno** (el código que trae el PDF, pensado para correr en Colab):

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF sobre el texto combinado de géneros + keywords de cada película
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

# Similitud de coseno entre todas las películas
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        return ["La película no existe en la base de datos."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # top 10, excluyendo la misma película
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()
```

**Para explicar el mecanismo en el pizarrón**: TF-IDF convierte el texto de cada película (géneros + keywords) en un vector numérico que pesa cada palabra según su importancia; la similitud de coseno mide el ángulo entre dos vectores — cuanto más chico el ángulo, más parecidas son las películas en contenido. No usa historial de usuarios (no es filtrado colaborativo), es **basado en contenido**.

### 3.4 Mazda — Segmentación de Clientes con Clustering

**El problema**: ¿cómo puede una empresa como Mazda entender mejor a sus clientes para ofrecer productos y servicios personalizados? A partir de un conjunto de **más de 30 variables**, se prepara los datos, se seleccionan características relevantes y se aplican algoritmos de clustering para identificar segmentos significativos.

**Conceptos clave**:
- **Clustering**: agrupamiento de datos sin etiquetas previas, buscando patrones latentes.
- **K-Means**: particiona los datos en *k* clusters minimizando la varianza intra-cluster.
- **Gaussian Mixture Models (GMM)**: modelo probabilístico que asume una mezcla de distribuciones gaussianas, permitiendo clusters con formas elípticas (no solo esféricas como K-Means).

*Cita del PDF*: según Aggarwal (2015), el clustering es efectivo para segmentación cuando existen patrones latentes en atributos de clientes, facilitando la personalización y optimización de campañas.

**Feature engineering y preparación**: limpieza (valores faltantes, errores), selección de variables relevantes, y **escalado** (fundamental — herramientas como `pandas` y `numpy` facilitan estas tareas).

**Pipeline de datos para clustering (6 pasos)**:
1. Ingestión y limpieza de datos
2. Selección y transformación de features
3. Escalado y normalización
4. Aplicación del algoritmo de clustering
5. Evaluación y validación de clusters
6. Interpretación y aplicación de resultados

**Evaluación de clusters**: **Inercia** (suma de distancias cuadradas dentro de clusters) y **Silhouette Score** (medida de separación y cohesión) — ayudan a decidir el número óptimo de segmentos y la robustez del modelo.

**Traducir lo técnico a negocio**:

| Métrica Técnica | Indicador de Negocio |
|---|---|
| Segmentos definidos y estables | Estrategias de marketing dirigidas y efectivas |
| Características distintivas | Personalización de ofertas y comunicación |

> **Este es el caso que el notebook de hoy reproduce en vivo**, con datos de natalidad en vez de clientes de Mazda — mismo pipeline de 6 pasos, mismo algoritmo (K-Means), mismas métricas de validación (Inercia + Silhouette). Ver la [sección 7](#7-el-ejercicio-práctico-del-notebook-explicado-en-profundidad) para el desarrollo completo.

---

## Módulo 4 — Supervisado vs. No Supervisado

**Pregunta disparadora**: tenés un enorme conjunto de datos de clientes, pero no sabés qué patrones o grupos existen dentro de ellos. ¿Cómo segmentarlos para campañas personalizadas? ¿O cómo predecir si un cliente comprará, en base a su comportamiento previo? Estas preguntas ilustran los dos grandes enfoques de la ciencia de datos.

### 4.1 Aprendizaje Supervisado

Los modelos aprenden a partir de **datos etiquetados** (pares entrada-salida conocidos), buscando generalizar para predecir la salida correcta en nuevas entradas.

- **Clasificación**: predice una categoría (ej. ¿es spam?).
- **Regresión**: predice un valor numérico continuo (ej. precio de una vivienda).

| Algoritmo | Descripción breve | Ejemplo de uso en industria |
|---|---|---|
| Regresión lineal | Modela la relación lineal entre variables | Predicción de ventas según inversión publicitaria |
| Random Forest | Conjunto de árboles de decisión | Detección de fraude en transacciones bancarias (San Cristóbal) |

**Supuestos y consideraciones**: requiere datos etiquetados (costoso/difícil de conseguir); el modelo aprende patrones explícitos entre entrada y salida; es fundamental elegir métricas adecuadas (precisión, recall, RMSE, etc.).

**Concepto para reforzar (no viene textual en el PDF, pero es la base de todo lo anterior)**: el objetivo de un modelo supervisado no es memorizar los datos de entrenamiento, sino **generalizar** — funcionar bien con datos nuevos que nunca vio. Cuando un modelo aprende "de memoria" el ruido específico de sus datos de entrenamiento y pierde capacidad de generalizar, se llama **overfitting** (sobreajuste); es la razón por la que en el Módulo 5 se insiste tanto en evaluar siempre con datos separados del entrenamiento (hold-out, K-Fold), y no con las mismas filas que el modelo ya vio.

### 4.2 Aprendizaje No Supervisado

Trabaja con **datos sin etiquetas**, buscando estructuras o patrones ocultos.

- **Clustering**: agrupa datos similares en segmentos (ej. segmentación de clientes — casos Mazda y el ejercicio de hoy).
- **Reducción de dimensionalidad**: simplifica datos complejos conservando la mayor información posible (ej. PCA).

| Algoritmo | Descripción breve | Ejemplo de uso en industria |
|---|---|---|
| K-Means | Agrupa datos en *k* clusters basados en distancia | Segmentación de usuarios en plataformas digitales |
| PCA | Transforma variables correlacionadas en componentes independientes | Visualización y reducción de variables en análisis financiero |

**Supuestos y consideraciones**: no requiere etiquetas, ideal para exploración; los resultados pueden ser menos interpretables que en supervisado; hay que validar la calidad de los clusters o componentes obtenidos.

### 4.3 Comparación práctica

| Aspecto | Aprendizaje Supervisado | Aprendizaje No Supervisado |
|---|---|---|
| Datos | Etiquetados (entrada-salida) | Sin etiquetas |
| Objetivo | Predecir o clasificar | Encontrar estructura o patrones |
| Ejemplos de aplicación | Detección de fraude, clasificación de imágenes | Segmentación de clientes, reducción de variables |

> La elección entre supervisado y no supervisado depende del problema, la disponibilidad de datos y el objetivo final.

### 4.4 Aplicaciones combinadas en la industria

En la práctica, los científicos de datos suelen combinar ambos enfoques:

- **Segmentación de clientes**: clustering para identificar grupos, luego modelos supervisados para predecir la respuesta a campañas.
- **Detección de fraude**: Random Forest para clasificar transacciones, apoyado por análisis no supervisado para descubrir patrones nuevos.
- **Optimización logística**: reducción de dimensionalidad para simplificar variables y mejorar la eficiencia de modelos predictivos.

---

## Break del Coder

Corte de ~10 minutos, después del Módulo 4 y antes de arrancar el Módulo 5 (Métricas y Validación) — cierra la parte de "qué algoritmo elegir" y abre la parte de "cómo saber si funcionó".

---

## Módulo 5 — Métricas y Estrategias de Validación

**Pregunta disparadora**: trabajás en una empresa de logística que quiere optimizar rutas de entrega usando modelos predictivos. ¿Cómo estar seguro de que el modelo realmente mejora la eficiencia y no solo funciona bien con los datos que ya tenés? La respuesta está en evaluar correctamente el modelo.

### 5.1 Métricas de Clasificación

- **Accuracy**: proporción de predicciones correctas sobre el total. Útil cuando las clases están balanceadas.
- **Precision**: proporción de verdaderos positivos sobre todos los positivos predichos. Importante cuando el costo de falsos positivos es alto.
- **Recall (Sensibilidad)**: proporción de verdaderos positivos sobre todos los positivos reales. Clave cuando es crítico detectar todos los casos positivos.
- **F1-Score**: media armónica entre precision y recall.
- **AUC-ROC**: área bajo la curva ROC, mide la capacidad de distinguir clases en diferentes umbrales.

*Ejemplo del PDF*: en detección de fraude, un alto recall es vital para no dejar pasar fraudes, aunque se sacrifiquen algunos falsos positivos (conecta directo con San Cristóbal, Módulo 3).

**La base de todas estas métricas (útil tenerla a mano, aunque el PDF no la despliega explícitamente)**: todas salen de comparar la predicción del modelo contra la realidad en una **matriz de confusión** de 2x2:

| | Predicho Positivo | Predicho Negativo |
|---|---|---|
| **Real Positivo** | Verdadero Positivo (VP) | Falso Negativo (FN) |
| **Real Negativo** | Falso Positivo (FP) | Verdadero Negativo (VN) |

De ahí salen las fórmulas: `Precision = VP / (VP + FP)` (de todo lo que dije que era positivo, ¿cuánto acerté?) y `Recall = VP / (VP + FN)` (de todo lo que era positivo en la realidad, ¿cuánto detecté?). Tenerlas escritas así ayuda mucho cuando un alumno pregunta "¿pero por qué no es lo mismo precision que recall?" — la diferencia está en el denominador: uno mira desde las predicciones, el otro desde la realidad.

### 5.2 Métricas de Regresión

- **RMSE** (Root Mean Squared Error): raíz del promedio de los errores al cuadrado, penaliza errores grandes.
- **MAE** (Mean Absolute Error): promedio de errores absolutos, más robusto a outliers.
- **R²** (Coeficiente de determinación): proporción de varianza explicada por el modelo.

*Ejemplo del PDF*: para predecir demanda de productos, RMSE ayuda a entender el error típico en unidades vendidas.

### 5.3 Métricas de Clustering

- **Silhouette Score**: mide qué tan bien separado está cada cluster.
- **Davies-Bouldin Index**: evalúa la separación y compacidad de clusters.
- **Inercia**: suma de distancias cuadradas dentro de clusters, usada en k-means.

*Ejemplo del PDF*: en segmentación de clientes, un buen silhouette indica grupos bien definidos para campañas personalizadas (conecta con Mazda y con el ejercicio del notebook).

### 5.4 Estrategias de validación

- **Hold-out**: dividir el dataset en entrenamiento y prueba. Rápido pero puede ser inestable si los datos son pocos.
- **K-Fold**: dividir el dataset en *k* partes; entrenar *k* veces, cada vez con un fold distinto como prueba y el resto para entrenamiento; promediar las métricas. Ventaja: reduce la varianza en la estimación del desempeño.
- **Time-Split**: para datos secuenciales o series temporales — se respeta el orden temporal, entrenando con datos anteriores y probando con datos posteriores. *Ejemplo del PDF*: en predicción de demanda diaria, no se debe usar datos futuros para entrenar.

### 5.5 Trade-offs en la selección de métricas y validación

- **Complejidad vs. interpretabilidad**: métricas simples como accuracy son fáciles de entender, pero pueden ser engañosas en datasets desbalanceados.
- **Tiempo de cómputo**: k-fold es más preciso pero consume más recursos.
- **Naturaleza del problema**: en problemas críticos (fraude, salud), priorizar recall o precisión según el impacto.
- **Datos disponibles**: en series temporales, usar time-split para evitar fugas de información.

> **Reflexión del PDF**: no existe una métrica o estrategia universal; la elección debe alinearse con el contexto y los objetivos del negocio.

### 5.6 Código de práctica (el que trae el PDF completo)

El PDF incluye un script de práctica de 5 partes que vale la pena mostrar o correr en vivo si hay tiempo:

1. **Clasificación**: `make_classification` + `LogisticRegression` + accuracy/precision/recall/F1/AUC-ROC.
2. **Regresión**: `make_regression` + `LinearRegression` + MAE/RMSE/R².
3. **Clustering**: `make_blobs` + `KMeans` + Silhouette/Davies-Bouldin/Inercia + scatter plot.
4. **K-Fold**: `KFold(n_splits=5, shuffle=True)` sobre el dataset de clasificación, promediando accuracy por fold.
5. **Time-Split**: `TimeSeriesSplit(n_splits=5)` sobre una serie temporal simulada (`np.sin(...)` + ruido), midiendo MSE por fold.

```python
# Fragmento representativo (K-Fold)
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_i, test_i in kf.split(Xc):
    model = LogisticRegression()
    model.fit(Xc[train_i], yc[train_i])
    pred = model.predict(Xc[test_i])
    scores.append(accuracy_score(yc[test_i], pred))
print("Promedio K-fold:", np.mean(scores))
```

**Para remarcar en clase**: aunque el ejercicio del día (natalidad + K-Means) solo necesita las métricas de clustering (5.3), es importante que los alumnos vean el panorama completo de las 5 partes — es exactamente el código que van a necesitar apenas trabajen con un problema supervisado.

---

## 7. El ejercicio práctico del notebook, explicado en profundidad

El notebook `Clase_7_Fundamentos_de_Ciencia_de_Datos_1_.ipynb` construye, sobre el dataset real **`tasa-natalidad-deis-2000-2024.csv`** (Ministerio de Salud, vía datos.salud.gob.ar), un ejercicio completo de segmentación de provincias argentinas según su evolución de natalidad 2000–2024. Es, en esencia, **el caso Mazda hecho en vivo con datos públicos** en lugar de datos de clientes.

### Bloque 0 — Un Ejemplito para Repasar 4 Conceptos de la Semana 6

No es contenido nuevo de Clase 07: es un **ejemplito rápido** —aplicado sobre el mismo dataset de natalidad que se usa en el resto de la clase— para repasar de forma ágil 4 conceptos de estadística y preprocesamiento vistos la clase pasada. La idea no es volver a dar la teoría completa (eso ya se vio), sino refrescarla en 10 minutos con código concreto, porque el pipeline del Bloque 1 usa los cuatro conceptos sin excepción. A continuación, cada uno bien desarrollado.

#### 1) Limpieza e Integración

```python
df_raw = pd.read_csv('tasa-natalidad-deis-2000-2024.csv')
nulos = df_raw.isnull().sum().sum()
duplicados = df_raw.duplicated().sum()
```

**Qué es "limpiar" un dataset, en profundidad**: ningún dataset real llega listo para analizar. Limpiar es la etapa donde se toman **decisiones** sobre dos problemas típicos:

- **Valores nulos (`NaN`)**: pueden aparecer por errores de carga, por integraciones entre fuentes distintas, o porque el campo simplemente no aplica a ese registro (lo cual no es un error, es información). Las estrategias más comunes son: **eliminar** la fila/columna (solo si son pocos casos y no introducen sesgo), **imputar** con la media o la mediana si la variable es numérica (la mediana es preferible si hay outliers, porque no se deja arrastrar por ellos), o con la **moda** o una **etiqueta de negocio explícita** (ej. `"No aplica"`) si la variable es categórica.
- **Filas duplicadas**: registros exactamente repetidos, que se eliminan con `.drop_duplicates()` porque inflarían artificialmente cualquier estadística o conteo.

En este dataset del DEIS puntual, la salida del código da **0 nulos y 0 duplicados** — llega limpio. Es un buen punto para remarcar en clase que "limpiar" **no es un paso que se corre siempre igual**: acá el resultado es "no hace falta hacer nada", y eso también es una conclusión válida del diagnóstico, no una excepción a la regla.

**Qué es "integrar", en profundidad**: integrar es combinar información de distintas fuentes o **derivar columnas nuevas** a partir de las existentes, para que un dato crudo se convierta en información accionable para quien toma decisiones. En el notebook, esto se ve creando la variable `categoria_natalidad`:

```python
natalidad_2024['categoria_natalidad'] = pd.cut(
    natalidad_2024['natalidad_2024'],
    bins=[0, 8.4, 9.7, np.inf],
    labels=['Baja', 'Media', 'Alta']
)
```

`pd.cut()` toma una variable numérica continua (la tasa de natalidad 2024 de cada provincia) y la convierte en una variable **categórica ordinal** según los cortes (`bins`) definidos: entre 0 y 8.4 es "Baja", entre 8.4 y 9.7 es "Media", y por encima de 9.7 es "Alta". El resultado es mucho más fácil de comunicar a alguien que no analiza datos todos los días que la cifra decimal cruda — esa es, en esencia, la utilidad de integrar/derivar variables.

#### 2) Medidas de Tendencia Central y Dispersión

```python
serie_nacional = df_raw['natalidad_argentina']
media, mediana, std = serie_nacional.mean(), serie_nacional.median(), serie_nacional.std()
iqr = serie_nacional.quantile(0.75) - serie_nacional.quantile(0.25)
```

**Tendencia central, en profundidad** — responde "¿dónde está el centro de los datos?":

- **Media**: la suma de todos los valores dividida la cantidad de valores. Es sensible a valores extremos: un solo dato muy alto o muy bajo puede "arrastrarla" en esa dirección.
- **Mediana**: el valor que queda justo en el medio cuando se ordenan todos los datos (deja 50% a cada lado). No le importa cuán extremos sean los valores de los bordes, solo su posición — por eso es **robusta** frente a outliers.
- **Moda**: el valor que más se repite. Es la única de las tres que también sirve para variables categóricas.

**Regla práctica para dar en clase**: si media y mediana son parecidas, la distribución es razonablemente simétrica. Si difieren mucho, hay dos explicaciones posibles: **outliers** (unos pocos valores extremos que "tiran" de la media) o **asimetría/tendencia** en los datos (más valores concentrados de un lado que del otro).

**Dispersión, en profundidad** — responde "¿qué tan esparcidos están los datos respecto al centro?":

- **Desvío estándar**: en promedio, cuánto se aleja cada dato de la media. Un desvío bajo indica datos concentrados; uno alto, datos muy variados.
- **IQR (rango intercuartílico)** = Q3 − Q1: el ancho del 50% central de los datos (entre el percentil 25 y el percentil 75). Es más robusto que el desvío estándar porque ignora directamente los valores extremos de los bordes.

**El caso concreto para trabajar en el pizarrón**: acá la mediana (17.9) resulta *más alta* que la media (16.35). A primera vista uno esperaría lo contrario si hubiera outliers altos empujando la media hacia arriba, pero la causa real es otra: la natalidad viene en **caída sostenida** durante los 25 años de la serie, así que hay más años "altos" (al principio de la serie) que años "bajos" (al final), y eso corre el centro (mediana) por encima del promedio simple. Es el ejemplo perfecto para instalar la idea de que "media distinta de mediana" no siempre delata outliers: a veces delata una **tendencia** en el tiempo. El IQR de ~3.1 puntos confirma que, aun con esa caída, la dispersión año a año es moderada (no hay saltos bruscos).

#### 3) Distribuciones y Correlación

```python
sns.histplot(serie_nacional, kde=True, bins=10)  # forma de la serie nacional
provincias_comparar = df_raw[['natalidad_buenos_aires', 'natalidad_cordoba', 'natalidad_santa_fe']]
sns.heatmap(provincias_comparar.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
```

**Distribución, en profundidad**: es la "silueta" que toman los datos cuando se grafican en un histograma — cuántas observaciones caen en cada rango de valores. Puede ser:

- **Simétrica** (tipo campana o Normal): los valores se reparten parejo a ambos lados del centro.
- **Sesgada a la derecha**: una "cola" larga de valores altos poco frecuentes (típico en ingresos, precios).
- **Sesgada a la izquierda**: una "cola" larga de valores bajos poco frecuentes.

El coeficiente de **asimetría (skew)** cuantifica esto: cerca de 0 es simétrica, positivo es sesgo a la derecha, negativo es sesgo a la izquierda.

**Correlación, en profundidad**: el coeficiente de correlación de Pearson mide qué tan asociadas están **linealmente** dos variables numéricas, en una escala de -1 a 1:

- **Cerca de 1**: cuando una sube, la otra también sube (relación positiva fuerte).
- **Cerca de -1**: cuando una sube, la otra baja (relación negativa fuerte).
- **Cerca de 0**: no hay relación lineal detectable entre ambas.

**El caso concreto para trabajar en el pizarrón**: la correlación entre Buenos Aires y Córdoba da **por encima de 0.95** — altísima. Acá está la mejor oportunidad de la clase para instalar con fuerza el principio de que **correlación no implica causalidad**: no es que una provincia le "contagie" la baja natalidad a la otra. Lo que ocurre es que **ambas comparten la misma tendencia demográfica nacional** — hay una tercera variable de fondo (el fenómeno país, que afecta a todas las provincias por igual) explicando el movimiento conjunto de las dos. Córdoba no baja su natalidad *porque* baja Buenos Aires; ambas bajan por la misma causa compartida. Es el mismo tipo de trampa que el clásico ejemplo de "las ventas de helado y los ahogamientos están correlacionados" (ambas suben en verano por el calor, no porque una cause la otra).

#### 4) Transformación y Reducción de Dimensionalidad

```python
scaler_demo = StandardScaler()
columnas_escaladas = scaler_demo.fit_transform(df_raw[['natalidad_buenos_aires', 'natalidad_cordoba']])

pca_demo = PCA(n_components=2)
pca_demo.fit_transform(StandardScaler().fit_transform(df_raw.drop(columns='indice_tiempo').T))
```

**Transformación, en profundidad**: para que un algoritmo matemático pueda procesar los datos, muchas veces hace falta prepararlos primero:

- **Codificación de texto a número**: `LabelEncoder` (asigna un número entero a cada categoría, útil cuando hay un orden implícito) o **one-hot encoding** (crea una columna binaria por categoría, preferible cuando no hay orden, para no inventarle una jerarquía artificial a los datos).
- **Escalado con `StandardScaler`**: transforma cada columna para que tenga media ≈0 y desvío ≈1 (lo que se conoce como *z-score*). Es indispensable para algoritmos que miden **distancias** entre puntos (como K-Means, que se usa más adelante en la clase): si una columna tiene valores entre 0 y 100.000 y otra entre 0 y 1, la primera va a dominar completamente el cálculo de distancia solo por su magnitud numérica, sin que eso refleje ninguna importancia real de esa variable.

**Reducción de dimensionalidad con PCA, en profundidad**: cuando hay muchas columnas (en este caso, 25 años = 25 features por provincia), **PCA** (Análisis de Componentes Principales) las combina matemáticamente en un número mucho menor de "componentes principales" que conservan la mayor parte posible de la variabilidad (información) original. En el ejemplo del notebook, comprimir 25 años en solo 2 componentes principales conserva **la mayor parte de la varianza original** (el notebook imprime el porcentaje exacto al correrlo — suele rondar el 90% o más), lo que permite, por ejemplo, graficar en un plano 2D algo que originalmente tenía 25 dimensiones, sin perder la estructura esencial de los datos.

**La conclusión que cierra el bloque, y que es la bisagra directa hacia el Bloque 1**: **escalar es obligatorio antes de PCA o K-Means**, porque ambos algoritmos miden distancias, y sin escalar, una columna con números más grandes "pesaría" más en el resultado solo por su magnitud, no porque sea más relevante para el problema.

### Bloque 1 — Pipeline de Ingesta y Transformación (20 min)

**El problema real**: el DEIS publica un nuevo renglón de datos cada año. Procesar "a mano" con celdas sueltas rompe con cada actualización. La solución es envolver la lógica en una función reutilizable:

```python
def pipeline_preprocesamiento(path_archivo):
    """Pipeline reproducible para limpiar y transformar el dataset de natalidad."""
    df = pd.read_csv(path_archivo)
    df['indice_tiempo'] = pd.to_datetime(df['indice_tiempo']).dt.year
    df.set_index('indice_tiempo', inplace=True)

    # Transposición crucial: filas = provincias (instancias), columnas = años (features)
    df_provincias = df.T

    # Imputación de nulos por media de la provincia
    df_provincias = df_provincias.fillna(df_provincias.mean())

    # Escalado para algoritmos basados en distancia
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_provincias)

    return df_provincias, X_scaled
```

**El detalle no obvio para explicar bien en el pizarrón**: el CSV original tiene los **años en las filas** y las **provincias en las columnas** — el formato natural para leer una serie de tiempo. Pero para que scikit-learn segmente **provincias** (no años), necesitamos que cada fila sea una provincia y cada columna sea una característica (un año) — de ahí la **transposición (`.T`)**. Es un paso conceptual, no solo técnico: cambia qué es "una instancia" para el algoritmo.

Esto implementa en código las etapas 1-3 del Módulo 1 (ingestión, limpieza, feature engineering vía transposición + escalado). Conceptualmente, el pipeline reproducible completo también incluiría gestión de artefactos, control de versiones, entornos fijados y despliegue mínimo — el notebook lo menciona explícitamente en su celda de teoría, aunque no lo implemente hoy.

### Bloque 2 — Supervisado vs. No Supervisado + K-Means (20 min)

**El razonamiento de negocio**: el Ministerio de Salud no tiene etiquetas de "provincia con natalidad decreciente" — nadie las definió de antemano. Por eso es un problema **no supervisado**: se busca que el algoritmo encuentre esos perfiles por sí solo.

```python
kmeans_prueba = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_prueba = kmeans_prueba.fit_predict(X)
```

**K-Means en una frase para el pizarrón**: el algoritmo ubica *k* centros geométricos (centroides) y asigna cada provincia al centroide más cercano por distancia euclidiana, iterando hasta que las asignaciones se estabilizan. `random_state=42` fija la semilla aleatoria para que el resultado sea reproducible entre corridas — otro gancho directo al Módulo 1.

Este bloque es, en la práctica, una implementación completa del **caso Mazda** (Módulo 3.4): mismo algoritmo, mismo tipo de problema (segmentación sin etiquetas), aplicado a un dominio distinto.

### Bloque 3 — Métricas y Estrategias de Validación (20 min)

**El dilema a plantear en clase**: elegimos K=3 "porque sí" en el bloque anterior. ¿Cómo justificarlo matemáticamente? Acá no sirven accuracy/precision (no hay etiquetas verdaderas) — se necesitan métricas específicas de clustering:

```python
inercias, siluetas = [], []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inercias.append(km.inertia_)
    siluetas.append(silhouette_score(X, labels))
```

- **Método del codo (inercia)**: a más clusters, la inercia siempre baja — se busca el punto donde agregar un cluster más deja de aportar una mejora significativa (el "codo" del gráfico).
- **Silhouette Score**: para cada K probado, mide qué tan bien separados y cohesionados quedan los grupos (rango -1 a 1, más alto es mejor).

Corresponde directamente al Módulo 5.3 (métricas de clustering) de esta guía. La celda de teoría agregada en el notebook también tiende el puente hacia las métricas de clasificación/regresión y las estrategias hold-out/K-Fold/Time-Split (Módulo 5 completo), aclarando que hoy no hacen falta porque el problema es no supervisado, pero van a ser necesarias en cuanto el proyecto pase a predecir un valor.

### Bloque 4 — Casos de Estudio y "Recomendaciones" (15 min)

**El cierre "Analytics to Action"**: un cluster por sí solo no genera valor de negocio — hay que interpretarlo y traducirlo en una acción.

```python
def sistema_recomendacion_politica(cluster_id):
    if cluster_id == 0:
        return "Alerta Demográfica: Reorientar presupuesto a salud de adultos mayores."
    elif cluster_id == 1:
        return "Prioridad Alta: Planificar construcción de nuevos jardines y escuelas primarias."
    else:
        return "Estable: Mantener subsidios existentes y monitorear tasas de control prenatal."

df_provincias['Accion_Recomendada'] = df_provincias['cluster_final'].apply(sistema_recomendacion_politica)
```

**El paralelo a remarcar con los 4 casos del Módulo 3**: esta función es el mismo patrón de cierre que Mazda (cluster → estrategia de marketing), Medplaya (predicción → overbooking), San Cristóbal (predicción → investigación) y Amazon (similitud → recomendación al usuario). En los cuatro casos —y en este ejercicio— **el modelo nunca es el final del pipeline**: el valor aparece cuando el resultado técnico se traduce en una decisión accionable.

**Para leer el resultado con la clase**: la tabla `perfil_clusters` (promedio de natalidad en 2000, 2012 y 2024 por cluster) permite nombrar cada grupo con criterio propio antes de mostrar las recomendaciones — es un buen momento para pedirle a los alumnos que interpreten los tres clusters *antes* de revelar las etiquetas que puso la función.

---

## Preguntas frecuentes y errores típicos a anticipar

- **"¿Por qué transponemos el DataFrame en el pipeline?"** → porque necesitamos que las provincias sean las filas (instancias) y los años las columnas (features) para que K-Means las segmente correctamente. Ver Bloque 1.
- **"¿Por qué hay que escalar antes de K-Means?"** → porque el algoritmo mide distancias euclidianas; sin escalar, una columna con valores más grandes domina el resultado solo por su magnitud, no por su relevancia real (Módulo 3.4 / Bloque 0-1).
- **"¿Por qué no usamos accuracy para evaluar los clusters?"** → porque no hay etiquetas verdaderas contra las cuales comparar; accuracy es una métrica de clasificación supervisada. Se usan Inercia y Silhouette Score en su lugar (Módulo 5.3 / Bloque 3).
- **"¿Cuándo usar oversampling y cuándo undersampling?"** → depende de cuántos datos hay disponibles: con pocos datos de la clase minoritaria conviene oversampling (SMOTE); con abundancia de datos de la clase mayoritaria, undersampling puede ser más eficiente sin perder información relevante (Módulo 3.1, San Cristóbal).
- **"¿Por qué el recall importa más que la precisión en fraude/cancelaciones?"** → porque el costo de no detectar un caso positivo real (un fraude que pasa, una cancelación no anticipada) suele ser mayor que el costo de una falsa alarma (Módulo 3.1 y 3.2).
- **"¿Por qué no se puede usar K-Fold normal en series temporales?"** → porque mezclaría datos del futuro en el entrenamiento (data leakage); hace falta Time-Split, que respeta el orden cronológico (Módulo 5.4).

---

## Material de la clase

| Archivo | Qué es |
|---|---|
| `Clase 07.pdf` | Material teórico oficial de la unidad (fuente original de esta guía). |
| `Semana 7.html` | Diapositivas para proyectar en clase (44 filminas). Navegación con flechas del teclado o los botones inferiores. |
| `Clase_7_Fundamentos_de_Ciencia_de_Datos_1_.ipynb` | Notebook con el ejercicio práctico completo (repaso + pipeline + K-Means + validación + recomendaciones) sobre datos reales de natalidad. |
| `tasa-natalidad-deis-2000-2024.csv` | Dataset real usado en el notebook (Ministerio de Salud, DEIS). |
| `material/` | Carpeta con recursos adicionales: teoría de aprendizaje supervisado/no supervisado, paso a paso, PPTs, notebooks de referencia adicionales. |

**Cómo usar esta guía durante la clase**: los Módulos 1 a 5 siguen el mismo orden que las diapositivas; la sección 7 sigue el orden del notebook. Podés alternar entre proyectar la filmina/notebook correspondiente y volver acá si necesitás un dato de contexto, una analogía o una pregunta frecuente para anticipar.
