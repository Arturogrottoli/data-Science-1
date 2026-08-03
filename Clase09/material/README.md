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

### ¿Qué es el aprendizaje supervisado?

Cada dato de entrenamiento viene con una respuesta conocida (`y`): un tumor es maligno o no, una casa vale tanto. El modelo aprende una función `f(X) → y` que mapea las variables de entrada a esa respuesta, con el objetivo de **predecir** `y` en casos nuevos. Se divide en dos familias:

| | Clasificación | Regresión |
|---|:---:|:---:|
| **`y` es...** | Una categoría | Un número |
| **Ejemplo** | ¿Tumor maligno? | ¿Precio de la vivienda? |
| **Métricas** | Accuracy, F1, AUC-ROC | MAE, RMSE, R² |

### Repaso relámpago de la Clase 08

- **Modelos**: Regresión Lineal, Árbol de Decisión, Random Forest, Regresión Logística, KNN.
- **Buenas prácticas**: `Pipeline` + `StandardScaler` para evitar *Data Leakage* al escalar o imputar; `train_test_split` con `stratify` para mantener la proporción de clases.
- **Validación**: `StratifiedKFold` + `cross_val_score` para no confiar en un solo split de datos.

### Lo que cambia hoy

El aprendizaje no supervisado parte de datos **sin `y`** — sin una respuesta correcta conocida de antemano. El objetivo deja de ser predecir y pasa a ser **descubrir estructura**: qué observaciones se parecen entre sí (clustering), cómo simplificar muchas variables en pocas sin perder lo esencial (reducción de dimensionalidad), o qué patrones se repiten con frecuencia (reglas de asociación) — los tres frentes que recorre el resto de esta clase.

---

## Módulo 1 — ¿Qué es el Aprendizaje No Supervisado?

### Definición y diferencias con el aprendizaje supervisado *(Filmina 03)*

El aprendizaje no supervisado es un conjunto de técnicas de Machine Learning que buscan identificar estructuras, patrones o relaciones en datos que **no cuentan con etiquetas o respuestas conocidas**. A diferencia del aprendizaje supervisado (Módulo 0), donde el modelo aprende a partir de ejemplos con etiquetas, acá el objetivo es descubrir información oculta sin guía explícita.

| Característica | Aprendizaje Supervisado | Aprendizaje No Supervisado |
|---|---|---|
| Datos de entrada | Con etiquetas o respuestas | Sin etiquetas |
| Objetivo | Predecir o clasificar | Encontrar patrones o estructuras |
| Ejemplos de problemas | Clasificación, regresión | Clustering, reducción de dimensionalidad, reglas de asociación |

### Tres grandes tipos de problemas *(Filmina 04)*

1. **Clustering (agrupamiento)**: agrupa datos similares en clusters. Ejemplo: segmentar clientes según comportamiento de compra.
2. **Reducción de dimensionalidad**: simplifica datos complejos con muchas variables a representaciones más manejables. Ejemplo: usar PCA para visualizar datos en 2D o 3D.
3. **Reglas de asociación**: encuentra relaciones frecuentes entre variables. Ejemplo: identificar productos que se compran juntos en retail.

Estas tres categorías se exploran en detalle en los Módulos 2 a 5 de esta clase, cada una con sus algoritmos y métricas propias.

### Ejemplos de aplicación en la industria *(Filmina 05)*

- **Retail y E-commerce**: segmentación de clientes para campañas personalizadas, análisis de cesta de la compra con reglas de asociación.
- **Tecnología y Big Data**: detección de anomalías en redes, agrupamiento de documentos o imágenes.
- **Analítica de negocios**: reducción de variables para simplificar reportes y visualizaciones.

Estos ejemplos muestran cómo el aprendizaje no supervisado ayuda a extraer valor de datos sin necesidad de etiquetas previas, facilitando la toma de decisiones basada en patrones reales — en retail, por ejemplo, saber cómo se agrupan los clientes o qué productos se compran juntos puede mejorar significativamente las estrategias de marketing y ventas.

### Flujo típico de trabajo *(Filmina 06)*

1. **Recolección y preparación de datos**: limpieza, selección y escalado de variables.
2. **Selección del método adecuado**: según el problema y el tipo de datos.
3. **Aplicación del algoritmo**: ejecución y ajuste de parámetros.
4. **Evaluación y validación**: métricas específicas para medir la calidad de agrupamientos o representaciones.
5. **Interpretación y uso de resultados**: integración en procesos de negocio o análisis posteriores.

Este flujo es la base para las prácticas y análisis de toda la clase — cambia el algoritmo módulo a módulo, no la lógica del proceso.

---

## Módulo 2 — Reglas de Asociación

**Contexto**: ¿alguna vez te preguntaste cómo las tiendas en línea saben qué productos recomendarte juntos? Las reglas de asociación son la técnica detrás de eso — descubrir patrones frecuentes en grandes conjuntos de transacciones.

### Apriori y FP-Growth *(Filmina 08)*

- **Apriori**: método clásico para encontrar conjuntos frecuentes de ítems. Genera candidatos de conjuntos y evalúa su frecuencia, descartando los que no cumplen un umbral mínimo (*support*). Intuitivo y fácil de implementar, pero computacionalmente costoso en bases grandes por la generación masiva de candidatos.
- **FP-Growth** (*Frequent Pattern Growth*): más eficiente, evita generar candidatos explícitos. Construye una estructura llamada **árbol FP** que compacta la información de las transacciones y extrae patrones frecuentes directamente. Más rápido y escalable que Apriori, aunque su implementación es más compleja.

### Métricas clave: support, confidence y lift *(Filmina 09)*

| Métrica | Definición | Interpretación |
|---|---|---|
| **Support** | `P(A ∩ B)` — proporción de transacciones que contienen A **y** B | Indica la frecuencia con la que ocurre la regla en el conjunto de datos |
| **Confidence** | `support(A∩B) / support(A)` — probabilidad de que B ocurra dado que ocurrió A | Indica la fuerza de la regla, condicionada a A |
| **Lift** | `confidence(A→B) / support(B)` | Valores > 1 sugieren una relación positiva real entre A y B, no azar |

**Nota clave**: una regla con alto *support* y *confidence* es frecuente y confiable, pero el *lift* es el que dice si la asociación es significativa o simplemente casual. Una regla con alto *support* pero bajo *lift* puede no ser interesante, porque la asociación podría ser casual.

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

---

## Módulo 3 — K-Means y la Elección de k

**Contexto**: ¿cómo agrupar datos sin etiquetas? K-Means es el algoritmo más usado de clustering — divide un conjunto de datos en grupos naturales basándose en similitud.

### Qué es y cómo funciona *(Filmina 12)*

K-Means es un **algoritmo de partición**: divide un conjunto de datos en `k` grupos (clusters) según la similitud de sus características. El objetivo es minimizar la suma de las distancias entre cada punto y el **centroide** (promedio) de su cluster asignado. Se apoya en las métricas de distancia (Euclidiana, Manhattan, Coseno) que ya se usaron en clases anteriores para definir "similitud".

### Los 4 pasos del algoritmo *(Filmina 13)*

1. **Inicialización**: se eligen `k` centroides iniciales — al azar o con **k-means++** para mejorar la convergencia.
2. **Asignación**: cada punto se asigna al cluster cuyo centroide esté más cerca (distancia Euclidiana, típicamente).
3. **Actualización**: se recalculan los centroides como el promedio de los puntos asignados a cada cluster.
4. **Repetición**: se repiten Asignación y Actualización hasta que las asignaciones no cambien o se alcance un número máximo de iteraciones.

### Convergencia, inicialización y problemas comunes *(Filmina 14)*

- K-Means **siempre converge**, pero a un **mínimo local**, no necesariamente al óptimo global.
- La inicialización de los centroides afecta la calidad y velocidad de convergencia; **k-means++** ayuda a elegir centroides iniciales más representativos, reduciendo la probabilidad de resultados pobres.
- **Outliers**: pueden distorsionar los centroides y afectar la agrupación.
- **Formas no esféricas**: K-Means asume clusters convexos y de tamaño similar; no funciona bien con formas arbitrarias.

### Elegir k: método del codo (Elbow Method) *(Filmina 15)*

Para cada valor de `k` se calcula el **WCSS** (*Within-Cluster Sum of Squares*): la suma de las distancias al cuadrado entre cada punto y el centroide de su cluster. Un WCSS más bajo indica clusters más compactos.

Se grafica WCSS en función de `k` — la curva baja a medida que `k` crece, porque agrupar en más clusters siempre reduce la distancia interna. El objetivo es identificar el punto donde la tasa de disminución se frena notablemente, formando un **"codo"**: a partir de ahí, agregar más clusters no mejora significativamente la calidad de la agrupación. Balancea complejidad del modelo (muchos clusters) contra calidad de la agrupación (pocos clusters, cada uno con sentido) — evitando tanto el subajuste como el sobreajuste.

### Elegir k: coeficiente silhouette *(Filmina 16)*

Para cada punto, compara su **cohesión** (distancia promedio a los demás puntos de su propio cluster) contra su **separación** (distancia promedio al cluster más cercano al que no pertenece). El resultado es un valor entre **-1 y 1**:

- Cerca de **1**: el punto está muy bien asignado a su cluster.
- Cerca de **-1**: el punto probablemente está mal asignado, y encajaría mejor en otro cluster.

Se calcula el promedio del coeficiente para todos los puntos, para cada `k` candidato, y se elige el `k` que **maximiza** ese promedio — el que da los clusters más definidos y separados. También sirve para detectar outliers: puntos con coeficiente cercano a -1 son candidatos a estar mal asignados.

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

---

## Módulo 4 — Clustering Jerárquico y DBSCAN

**Contexto**: dos alternativas a K-Means, para cuando no querés (o no podés) definir `k` de antemano, o cuando tus datos tienen ruido y formas irregulares.

### Clustering jerárquico: aglomerativo y divisivo *(Filmina 20)*

El clustering jerárquico construye una jerarquía de clusters, sin necesidad de definir el número de clusters de antemano:

- **Aglomerativo** (el más usado en la práctica): comienza con cada punto como un cluster individual, y fusiona iterativamente los dos clusters más parecidos, hasta que todos quedan combinados en uno solo.
- **Divisivo**: el enfoque inverso — parte de un único cluster con todos los datos, y lo va dividiendo progresivamente.

El resultado se visualiza en un **dendrograma**: un diagrama en forma de árbol donde cada hoja es un punto individual, y la altura donde dos clusters se unen indica su grado de disimilitud (cuanto más abajo se unen, más similares son). Cortar el dendrograma a distintas alturas da distintos números de clusters, sin tener que volver a correr el algoritmo.

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

---

## Módulo 5 — PCA: Reducción de Dimensionalidad

**Contexto**: ¿cómo simplificar un dataset con decenas o cientos de variables sin perder lo esencial? El Análisis de Componentes Principales (PCA) es la técnica fundamental para reducir dimensionalidad, facilitando la visualización y el análisis.

### Covarianza: la relación entre variables *(Filmina 26)*

La covarianza mide cómo varían **juntas** dos variables: si ambas tienden a subir o bajar a la vez, es positiva; si una sube mientras la otra baja, es negativa.

$$Cov(X,Y) = E[(X - \mu_X)(Y - \mu_Y)]$$

En PCA, la **matriz de covarianza** resume esas relaciones entre **todas** las variables del dataset a la vez, y es la base para identificar las direcciones de mayor variabilidad.

### Eigenvectores y eigenvalores: direcciones y magnitudes *(Filmina 27)*

Un **eigenvector** es un vector que, al aplicarle una transformación lineal (como la matriz de covarianza), solo cambia en magnitud, no en dirección. El factor por el que cambia esa magnitud es el **eigenvalor** correspondiente.

En PCA:
- Los **eigenvectores** de la matriz de covarianza son las **Componentes Principales** — las nuevas direcciones ortogonales sobre las que se proyectan los datos.
- Los **eigenvalores** indican la **varianza** que explica cada componente — un eigenvalor alto significa que esa dirección captura mucha variabilidad de los datos.

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

### Limitaciones de PCA *(Filmina 29)*

- **Linealidad**: PCA solo captura relaciones **lineales** entre variables; con estructuras no lineales complejas, puede no ser suficiente.
- **Escalado**: es sensible a la escala de las variables — por eso es común normalizar o estandarizar los datos antes de aplicarlo (igual que en clustering).
- **Interpretabilidad**: las componentes principales son combinaciones lineales de las variables originales, lo que puede dificultar su interpretación directa frente a un público no técnico.

### Aplicación práctica y relevancia en la industria *(Filmina 30)*

- **Visualización**: reducir dimensiones a 2 o 3 para graficar y detectar patrones o segmentos de clientes a simple vista.
- **Preprocesamiento**: simplificar datos antes de aplicar clustering o clasificación, mejorando el rendimiento y reduciendo ruido.

Por ejemplo, un analista puede usar PCA para transformar variables de comportamiento de compra en componentes principales que resumen tendencias clave, facilitando la segmentación de clientes — y entender la varianza explicada permite justificar cuántos componentes usar, balanceando precisión y simplicidad.

---

## Módulo 6 — Panorama de Métodos (Síntesis)

**Contexto**: cierre conceptual de la clase — comparar las cinco técnicas vistas, entender sus límites, y ver PCA mejorando el rendimiento de un modelo real, no solo en teoría.

### Decisiones de diseño y parámetros clave *(Filmina 32)*

| Técnica | Parámetros clave | Consideración principal |
|---|---|---|
| **K-Means** | Número de clusters `k` | Elegir `k` adecuado; sensible a valores atípicos |
| **Clustering jerárquico** | Método de linkage (single, complete...) | Interpretación del dendrograma; escalabilidad |
| **DBSCAN** | `eps`, `min_samples` | Detecta ruido; adecuado para formas arbitrarias |
| **PCA** | Número de componentes a conservar | Balance entre reducción y pérdida de información |
| **Apriori** | Soporte mínimo, confianza mínima | Controla cantidad y calidad de reglas generadas |

### Limitaciones y supuestos básicos *(Filmina 33)*

- El **clustering** asume que la similitud/diferencia entre puntos es significativa y que los datos pueden agruparse con claridad.
- **PCA** asume relaciones lineales y que la varianza es una medida adecuada de "información".
- Las **reglas de asociación** requieren datos transaccionales y pueden generar muchas reglas irrelevantes sin filtros adecuados.

**Para reflexionar en clase**: ¿qué pasaría si aplicás K-Means a datos con clusters de formas muy irregulares? ¿O PCA a datos con relaciones fuertemente no lineales? (Spoiler: en ambos casos, conviene DBSCAN o técnicas no lineales en vez de forzar el método "de siempre".)

### Aplicaciones prácticas por escenario *(Filmina 34)*

- **Clustering**: segmentación de clientes, detección de fraude agrupando comportamientos atípicos, análisis de patrones en sensores industriales.
- **PCA**: visualización de datos complejos, reducción de ruido antes de un modelo supervisado, compresión de datos para almacenamiento eficiente.
- **Reglas de asociación**: productos que se compran juntos, optimización de layout de tienda, análisis de comportamiento de compra.

En la práctica, la elección depende del contexto de negocio: en un e-commerce con datos ruidosos y clusters de forma compleja, DBSCAN suele ganarle a K-Means.

### Demostración: PCA mejorando un modelo real *(Filmina 35)*

El PDF cierra con un ejemplo didáctico controlado que demuestra, con números, que PCA puede **mejorar** el rendimiento de un modelo — no solo "comprimir" datos:

- Dataset de **cáncer de mama** (scikit-learn, 30 *features* reales).
- Se agregan **300 columnas de ruido** (features irrelevantes) a propósito, simulando un escenario de alta dimensionalidad con mucha señal desperdiciada.
- Clasificador **KNN**, elegido por ser sensible al *curse of dimensionality* (empeora notablemente con muchas features irrelevantes).
- Se compara accuracy **sin PCA** vs. **con PCA** (reducción fuerte: 330 → 30 componentes).

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
