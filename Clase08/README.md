[Material Diapositivas](https://docs.google.com/presentation/d/14g5eAuGgTasx_D4dJ7w7txt2qOZEY7EqvrTVTMddh74/edit#slide=id.p44)

# 📚 Clase 8: Aprendizaje No Supervisado

---

## 📋 **8.1 Introducción al Aprendizaje No Supervisado**

### **🎯 ¿Qué es el Aprendizaje No Supervisado?**

El **Aprendizaje No Supervisado** es una rama del Machine Learning que encuentra patrones ocultos en datos **sin etiquetas** (target variables). A diferencia del aprendizaje supervisado, no tenemos la "respuesta correcta" para entrenar el modelo.

#### **🔍 Características Principales:**

1. **Sin etiquetas**: Los datos no tienen variables objetivo conocidas
2. **Descubrimiento de patrones**: El objetivo es encontrar estructuras ocultas
3. **Exploratorio**: Se usa para entender mejor los datos
4. **Flexible**: No hay restricciones sobre qué patrones buscar

#### **📊 Tipos de Aprendizaje No Supervisado:**

| Tipo | Descripción | Ejemplos |
|------|-------------|----------|
| **Clustering** | Agrupa datos similares | Segmentación de clientes, análisis de genes |
| **Reducción de Dimensionalidad** | Reduce variables manteniendo información | PCA, t-SNE, UMAP |
| **Reglas de Asociación** | Encuentra relaciones entre elementos | Market basket analysis |
| **Detección de Anomalías** | Identifica patrones inusuales | Detección de fraudes |

#### **🎯 Casos de Uso Reales:**

```python
# Ejemplo conceptual: ¿Por qué usar aprendizaje no supervisado?
import pandas as pd
import numpy as np

# Simulamos datos de clientes sin etiquetas
np.random.seed(42)
n_clientes = 1000

datos_clientes = {
    'edad': np.random.normal(35, 10, n_clientes),
    'ingresos_anuales': np.random.lognormal(10, 0.5, n_clientes),
    'gasto_mensual': np.random.gamma(2, 500, n_clientes),
    'frecuencia_compras': np.random.poisson(8, n_clientes),
    'satisfaccion': np.random.choice([1,2,3,4,5], n_clientes, p=[0.1,0.2,0.3,0.3,0.1])
}

df_clientes = pd.DataFrame(datos_clientes)

print("=== EJEMPLO DE DATOS SIN ETIQUETAS ===")
print("No sabemos qué tipos de clientes tenemos...")
print("¿Cómo los agrupamos para estrategias de marketing?")
print(f"\nDatos de muestra:")
print(df_clientes.head())
print(f"\nEstadísticas descriptivas:")
print(df_clientes.describe())
```

#### **⚖️ Ventajas vs Desventajas:**

**✅ Ventajas:**
- No necesita datos etiquetados (más barato)
- Descubre patrones inesperados
- Útil para exploración de datos
- Reduce dimensionalidad

**❌ Desventajas:**
- Más difícil de evaluar
- Resultados subjetivos
- Requiere interpretación humana
- Menos predictivo que supervisado

---

## 🧠 **8.2 Algoritmos de Clustering**

### **¿Qué es el Clustering?**

El **clustering** es una técnica de aprendizaje no supervisado que agrupa objetos similares dentro de un mismo grupo (o clúster), separándolos de objetos diferentes que pertenecen a otros grupos. Es ampliamente utilizado para segmentación de clientes, análisis de imágenes, detección de patrones y más.

#### **🎯 Objetivos del Clustering:**
- **Agrupar**: Encontrar grupos naturales en los datos
- **Descubrir**: Identificar patrones ocultos
- **Simplificar**: Reducir complejidad de grandes datasets
- **Segmentar**: Crear categorías para estrategias de negocio

#### **📊 Ejemplo Práctico: Segmentación de Clientes**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Crear dataset realista de clientes
np.random.seed(42)
n_clientes = 500

# Simular diferentes tipos de clientes
# Tipo 1: Jóvenes con ingresos bajos, gastos moderados
jovenes = {
    'edad': np.random.normal(25, 3, 150),
    'ingresos': np.random.normal(30000, 5000, 150),
    'gasto_mensual': np.random.normal(800, 200, 150),
    'frecuencia': np.random.poisson(6, 150)
}

# Tipo 2: Adultos medios con ingresos altos, gastos altos
adultos_ricos = {
    'edad': np.random.normal(45, 5, 200),
    'ingresos': np.random.normal(80000, 10000, 200),
    'gasto_mensual': np.random.normal(2500, 500, 200),
    'frecuencia': np.random.poisson(12, 200)
}

# Tipo 3: Adultos mayores con ingresos medios, gastos bajos
mayores = {
    'edad': np.random.normal(65, 5, 150),
    'ingresos': np.random.normal(50000, 8000, 150),
    'gasto_mensual': np.random.normal(600, 150, 150),
    'frecuencia': np.random.poisson(3, 150)
}

# Combinar todos los datos
data = {
    'edad': np.concatenate([jovenes['edad'], adultos_ricos['edad'], mayores['edad']]),
    'ingresos': np.concatenate([jovenes['ingresos'], adultos_ricos['ingresos'], mayores['ingresos']]),
    'gasto_mensual': np.concatenate([jovenes['gasto_mensual'], adultos_ricos['gasto_mensual'], mayores['gasto_mensual']]),
    'frecuencia': np.concatenate([jovenes['frecuencia'], adultos_ricos['frecuencia'], mayores['frecuencia']])
}

df = pd.DataFrame(data)

# Agregar ruido para hacer más realista
df['edad'] += np.random.normal(0, 2, len(df))
df['ingresos'] += np.random.normal(0, 2000, len(df))
df['gasto_mesual'] += np.random.normal(0, 100, len(df))

print("=== DATASET DE CLIENTES ===")
print(f"Forma del dataset: {df.shape}")
print(f"\nEstadísticas descriptivas:")
print(df.describe())

# Visualización inicial
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(df['edad'], df['gasto_mensual'], alpha=0.6)
plt.xlabel('Edad')
plt.ylabel('Gasto Mensual')
plt.title('Edad vs Gasto Mensual')

plt.subplot(1, 3, 2)
plt.scatter(df['ingresos'], df['gasto_mensual'], alpha=0.6)
plt.xlabel('Ingresos')
plt.ylabel('Gasto Mensual')
plt.title('Ingresos vs Gasto Mensual')

plt.subplot(1, 3, 3)
plt.scatter(df['frecuencia'], df['gasto_mensual'], alpha=0.6)
plt.xlabel('Frecuencia de Compras')
plt.ylabel('Gasto Mensual')
plt.title('Frecuencia vs Gasto Mensual')

plt.tight_layout()
plt.show()
```

---

## 🧩 **8.2.1 Principales enfoques de Clustering y sus características**

### 1. 🔹 **Clustering por Particiones (K-Means)**

Divide los datos en *k* grupos predefinidos maximizando la similitud dentro de los clústeres y minimizando la similitud entre ellos.

#### **🎯 K-Means en Acción:**

```python
# Preparar datos para clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['edad', 'ingresos', 'gasto_mensual', 'frecuencia']])

# Método del Codo para encontrar k óptimo
def metodo_codo(X, max_k=10):
    """Encuentra el número óptimo de clusters usando el método del codo"""
    inertias = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    return k_range, inertias

# Aplicar método del codo
k_range, inertias = metodo_codo(X_scaled)

# Visualizar método del codo
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo')
plt.grid(True)

# Método de Silhouette
def metodo_silhouette(X, max_k=10):
    """Encuentra el número óptimo de clusters usando silhouette"""
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    return k_range, silhouette_scores

k_range_sil, silhouette_scores = metodo_silhouette(X_scaled)

plt.subplot(1, 2, 2)
plt.plot(k_range_sil, silhouette_scores, 'ro-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Método de Silhouette')
plt.grid(True)

plt.tight_layout()
plt.show()

# Aplicar K-Means con k=3 (basado en nuestro conocimiento del dataset)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Agregar clusters al dataframe
df['cluster'] = cluster_labels

# Visualizar resultados
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
scatter = plt.scatter(df['edad'], df['gasto_mensual'], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Edad')
plt.ylabel('Gasto Mensual')
plt.title('K-Means: Edad vs Gasto')
plt.colorbar(scatter)

plt.subplot(1, 3, 2)
scatter = plt.scatter(df['ingresos'], df['gasto_mensual'], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Ingresos')
plt.ylabel('Gasto Mensual')
plt.title('K-Means: Ingresos vs Gasto')
plt.colorbar(scatter)

plt.subplot(1, 3, 3)
scatter = plt.scatter(df['frecuencia'], df['gasto_mensual'], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Frecuencia')
plt.ylabel('Gasto Mensual')
plt.title('K-Means: Frecuencia vs Gasto')
plt.colorbar(scatter)

plt.tight_layout()
plt.show()

# Análisis de clusters
print("\n=== ANÁLISIS DE CLUSTERS K-MEANS ===")
for i in range(3):
    cluster_data = df[df['cluster'] == i]
    print(f"\nCluster {i}:")
    print(f"  Tamaño: {len(cluster_data)} clientes ({len(cluster_data)/len(df)*100:.1f}%)")
    print(f"  Edad promedio: {cluster_data['edad'].mean():.1f}")
    print(f"  Ingresos promedio: ${cluster_data['ingresos'].mean():,.0f}")
    print(f"  Gasto promedio: ${cluster_data['gasto_mensual'].mean():,.0f}")
    print(f"  Frecuencia promedio: {cluster_data['frecuencia'].mean():.1f}")

# Calcular silhouette score
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"\nSilhouette Score: {silhouette_avg:.3f}")
```

**🎯 Interpretación de Resultados:**
- **Cluster 0**: Jóvenes con ingresos bajos
- **Cluster 1**: Adultos mayores con gastos conservadores  
- **Cluster 2**: Adultos medios con altos ingresos y gastos

👉 *Ventajas:* Simple, rápido, escalable
👉 *Desventajas:* Requiere definir k, sensible a outliers, asume clusters esféricos

---

### 2. 🔹 **Clustering Jerárquico**

Crea una estructura tipo árbol (dendrograma) que refleja cómo se agrupan los datos paso a paso.

#### **🌳 Clustering Jerárquico en Acción:**

```python
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Aplicar clustering jerárquico
hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# Crear dendrograma
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title('Dendrograma - Clustering Jerárquico')
plt.xlabel('Índice de Muestra')
plt.ylabel('Distancia')

# Visualizar clusters jerárquicos
plt.subplot(1, 3, 2)
scatter = plt.scatter(df['edad'], df['gasto_mensual'], c=hierarchical_labels, cmap='viridis', alpha=0.6)
plt.xlabel('Edad')
plt.ylabel('Gasto Mensual')
plt.title('Clustering Jerárquico')
plt.colorbar(scatter)

plt.subplot(1, 3, 3)
scatter = plt.scatter(df['ingresos'], df['gasto_mensual'], c=hierarchical_labels, cmap='viridis', alpha=0.6)
plt.xlabel('Ingresos')
plt.ylabel('Gasto Mensual')
plt.title('Clustering Jerárquico')
plt.colorbar(scatter)

plt.tight_layout()
plt.show()

print("\n=== COMPARACIÓN K-MEANS vs JERÁRQUICO ===")
print(f"K-Means Silhouette: {silhouette_score(X_scaled, cluster_labels):.3f}")
print(f"Jerárquico Silhouette: {silhouette_score(X_scaled, hierarchical_labels):.3f}")
```

**🎯 Tipos de Clustering Jerárquico:**
- **Aglomerativo**: Funde clústeres de abajo hacia arriba
- **Divisivo**: Divide de arriba hacia abajo

👉 *Ventajas:* No necesita predefinir k, interpretable
👉 *Desventajas:* Costoso computacionalmente, sensible a outliers

---

### 3. 🔹 **Clustering por Densidad (DBSCAN)**

Agrupa puntos que están densamente conectados entre sí, detectando automáticamente outliers.

#### **🎯 DBSCAN en Acción:**

```python
from sklearn.neighbors import NearestNeighbors

# Función para encontrar eps óptimo usando k-distance graph
def encontrar_eps_optimo(X, k=4):
    """Encuentra el eps óptimo usando el gráfico de k-distancia"""
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, k-1], axis=0)
    return distances

# Encontrar eps óptimo
distances = encontrar_eps_optimo(X_scaled)
eps_optimo = distances[int(len(distances) * 0.1)]  # Usar percentil 10

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(distances)
plt.axhline(y=eps_optimo, color='r', linestyle='--', label=f'eps óptimo: {eps_optimo:.2f}')
plt.xlabel('Puntos ordenados por distancia')
plt.ylabel('Distancia al k-ésimo vecino')
plt.title('Método de k-distancia para encontrar eps')
plt.legend()
plt.grid(True)

# Aplicar DBSCAN
dbscan = DBSCAN(eps=eps_optimo, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Visualizar resultados DBSCAN
plt.subplot(1, 2, 2)
scatter = plt.scatter(df['edad'], df['gasto_mensual'], c=dbscan_labels, cmap='viridis', alpha=0.6)
plt.xlabel('Edad')
plt.ylabel('Gasto Mensual')
plt.title(f'DBSCAN (eps={eps_optimo:.2f})')
plt.colorbar(scatter)

plt.tight_layout()
plt.show()

# Análisis de resultados DBSCAN
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"\n=== RESULTADOS DBSCAN ===")
print(f"Número de clusters encontrados: {n_clusters}")
print(f"Número de puntos de ruido: {n_noise}")
print(f"Porcentaje de ruido: {n_noise/len(dbscan_labels)*100:.1f}%")

# Comparar todos los métodos
plt.figure(figsize=(15, 5))

algoritmos = [
    ('K-Means', cluster_labels),
    ('Jerárquico', hierarchical_labels),
    ('DBSCAN', dbscan_labels)
]

for i, (nombre, labels) in enumerate(algoritmos):
    plt.subplot(1, 3, i+1)
    scatter = plt.scatter(df['edad'], df['gasto_mensual'], c=labels, cmap='viridis', alpha=0.6)
    plt.xlabel('Edad')
    plt.ylabel('Gasto Mensual')
    plt.title(f'{nombre}')
    plt.colorbar(scatter)

plt.tight_layout()
plt.show()

# Comparar métricas
print(f"\n=== COMPARACIÓN DE ALGORITMOS ===")
for nombre, labels in algoritmos:
    if len(set(labels)) > 1:  # Solo si hay más de un cluster
        score = silhouette_score(X_scaled, labels)
        print(f"{nombre}: Silhouette Score = {score:.3f}")
    else:
        print(f"{nombre}: No se puede calcular silhouette (solo un cluster)")
```

**🎯 Características de DBSCAN:**
- **eps**: Distancia máxima entre puntos para ser considerados vecinos
- **min_samples**: Número mínimo de puntos para formar un cluster
- **Outliers**: Puntos marcados como -1 (ruido)

👉 *Ventajas:* Detecta clusters de forma arbitraria, maneja ruido, no necesita k
👉 *Desventajas:* Sensible a parámetros, difícil con clusters de densidad variable

---

### 4. 🔹 **Clustering Basado en Grid**
Divide el espacio en celdas y agrupa según densidad local en cada celda.
- **Wavecluster**: Transformación de onda
- **STING**: Grillas estadísticas jerárquicas
- **CLIQUE**: Alta dimensión
👉 *Ventajas:* Eficiente en altas dimensiones
👉 *Desventajas:* Sensible al tamaño de grilla

---

### 5. 🔹 **Clustering Basado en Modelos**
Asume que los datos se generan a partir de un modelo estadístico.
- **GMM**: Distribuciones normales múltiples
- **COBWEB**: Árbol jerárquico categórico
- **SOMs**: Red neuronal auto-organizativa
👉 *Ventajas:* Modelos probabilísticos flexibles
👉 *Desventajas:* Requiere supuestos sobre distribución

---

## 📊 **8.3 Reducción de Dimensionalidad**

### **🎯 ¿Por qué reducir dimensiones?**
1. **Visualización**: Solo podemos ver 2-3 dimensiones
2. **Curse of Dimensionality**: Más dimensiones = más datos necesarios
3. **Ruido**: Dimensiones irrelevantes confunden algoritmos
4. **Computación**: Menos dimensiones = más rápido

### **🔧 PCA (Principal Component Analysis):**

```python
from sklearn.decomposition import PCA

# Aplicar PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualizar resultados
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('PCA - Reducción a 2D')
plt.colorbar()
plt.show()

print(f"Varianza explicada: {sum(pca.explained_variance_ratio_):.1%}")
```

---

## 🛒 **8.4 Reglas de Asociación**

### **🎯 Market Basket Analysis:**

```python
from mlxtend.frequent_patterns import apriori, association_rules

# Ejemplo de transacciones
transacciones = [
    ['Leche', 'Pan', 'Huevos'],
    ['Pan', 'Mantequilla'],
    ['Leche', 'Mantequilla', 'Huevos'],
    ['Pan', 'Leche']
]

# Convertir a formato binario
te = TransactionEncoder()
te_ary = te.fit(transacciones).transform(transacciones)
df_transacciones = pd.DataFrame(te_ary, columns=te.columns_)

# Encontrar itemsets frecuentes
frequent_itemsets = apriori(df_transacciones, min_support=0.5, use_colnames=True)

# Generar reglas
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print("Reglas de asociación:")
print(rules[['antecedents', 'consequents', 'support', 'confidence']])
```

**🎯 Métricas:**
- **Soporte**: Frecuencia de la regla
- **Confianza**: Probabilidad condicional
- **Lift**: Mejora sobre aleatorio

---

## 📈 **8.5 Evaluación y Comparación**

### **🎯 Métricas de Evaluación:**

```python
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Métricas internas (no necesitan etiquetas verdaderas)
silhouette = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Score: {silhouette:.3f}")

# Métricas externas (necesitan etiquetas verdaderas)
ari = adjusted_rand_score(true_labels, cluster_labels)
print(f"Adjusted Rand Index: {ari:.3f}")

# Comparar algoritmos
algoritmos = [('K-Means', cluster_labels), ('DBSCAN', dbscan_labels)]
for nombre, labels in algoritmos:
    score = silhouette_score(X_scaled, labels)
    print(f"{nombre}: {score:.3f}")
```

**🎯 Interpretación:**
- **Silhouette**: [-1, 1] - 1 es mejor
- **ARI**: [-1, 1] - 1 es mejor
- **Davies-Bouldin**: [0, ∞) - Menor es mejor

---

## 🎯 **8.6 Actividad Práctica**

### **📋 Ejercicio: Segmentación de Clientes E-commerce**

```python
# Dataset realista de e-commerce
np.random.seed(42)
n_clientes = 500

# Crear perfiles de clientes
perfiles = {
    'joven_tecnologico': {
        'edad': np.random.normal(25, 3, 125),
        'ingresos': np.random.normal(35000, 5000, 125),
        'gasto_electronica': np.random.gamma(2, 200, 125),
        'frecuencia_online': np.random.poisson(15, 125)
    },
    'adulto_familia': {
        'edad': np.random.normal(40, 5, 200),
        'ingresos': np.random.normal(60000, 10000, 200),
        'gasto_electronica': np.random.gamma(1.5, 150, 200),
        'frecuencia_online': np.random.poisson(8, 200)
    },
    'mayor_conservador': {
        'edad': np.random.normal(65, 5, 175),
        'ingresos': np.random.normal(45000, 8000, 175),
        'gasto_electronica': np.random.gamma(1, 50, 175),
        'frecuencia_online': np.random.poisson(3, 175)
    }
}

# Combinar datos
data_ecommerce = {
    'edad': np.concatenate([p['edad'] for p in perfiles.values()]),
    'ingresos': np.concatenate([p['ingresos'] for p in perfiles.values()]),
    'gasto_electronica': np.concatenate([p['gasto_electronica'] for p in perfiles.values()]),
    'frecuencia_online': np.concatenate([p['frecuencia_online'] for p in perfiles.values()])
}

df_ecommerce = pd.DataFrame(data_ecommerce)

# Aplicar clustering
X_ecommerce = StandardScaler().fit_transform(df_ecommerce)
kmeans_ecommerce = KMeans(n_clusters=3, random_state=42)
segmentos = kmeans_ecommerce.fit_predict(X_ecommerce)

# Análisis de segmentos
df_ecommerce['segmento'] = segmentos
for i in range(3):
    segmento = df_ecommerce[df_ecommerce['segmento'] == i]
    print(f"\nSegmento {i} ({len(segmento)} clientes):")
    print(f"  Edad promedio: {segmento['edad'].mean():.1f} años")
    print(f"  Ingresos promedio: ${segmento['ingresos'].mean():,.0f}")
    print(f"  Gasto electrónica: ${segmento['gasto_electronica'].mean():.0f}")

# Visualización
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
scatter = plt.scatter(df_ecommerce['edad'], df_ecommerce['ingresos'], c=segmentos, cmap='viridis', alpha=0.6)
plt.xlabel('Edad')
plt.ylabel('Ingresos')
plt.title('Segmentación de Clientes')
plt.colorbar(scatter)

plt.subplot(1, 2, 2)
scatter = plt.scatter(df_ecommerce['gasto_electronica'], df_ecommerce['frecuencia_online'], c=segmentos, cmap='viridis', alpha=0.6)
plt.xlabel('Gasto Electrónica')
plt.ylabel('Frecuencia Online')
plt.title('Segmentación de Clientes')
plt.colorbar(scatter)
plt.show()
```

---

## 📚 **8.7 Recursos Complementarios**

### **📖 Lecturas:**
- "Pattern Recognition and Machine Learning" - Christopher Bishop
- "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman

### **🛠️ Librerías:**
```python
import sklearn.cluster          # K-Means, DBSCAN
import sklearn.decomposition   # PCA
import mlxtend                 # Reglas de asociación
import umap                    # UMAP
```

### **🎯 Casos de Uso:**
| Industria | Aplicación | Algoritmo |
|-----------|------------|-----------|
| Retail | Segmentación clientes | K-Means |
| Banca | Detección fraudes | DBSCAN |
| Salud | Análisis genes | Hierarchical |
| Marketing | Market Basket | Apriori |

---

## 📝 **8.8 Glosario**

| Término | Definición |
|---------|------------|
| **Clustering** | Agrupación de objetos similares |
| **Centroide** | Punto promedio de un cluster |
| **Silhouette** | Métrica de calidad del clustering |
| **Outlier** | Punto que no pertenece a ningún cluster |
| **PCA** | Reducción de dimensionalidad lineal |
| **Soporte** | Frecuencia de un itemset |
| **Confianza** | Probabilidad condicional |

---

## 🤖 **8.9 Agrupación y Segmentación con IA**

### **🚀 Tendencias Futuras:**
1. **Clustering con Deep Learning**: Autoencoders para reducción de dimensionalidad
2. **Clustering Adaptativo**: Actualización en tiempo real
3. **Interpretabilidad**: SHAP para explicar clusters
4. **Clustering Multimodal**: Combinar texto, imágenes y datos tabulares

### **🎯 Aplicaciones Avanzadas:**
- **Segmentación Dinámica**: Clusters que evolucionan en el tiempo
- **Clustering Federado**: Privacidad en clustering distribuido
- **Clustering Ético**: Evitar sesgos en segmentación

---

## ✅ **Resumen de la Clase 8**

### **🎯 Lo que aprendiste:**
1. **Fundamentos**: Aprendizaje no supervisado
2. **Algoritmos**: K-Means, DBSCAN, Jerárquico
3. **Reducción de Dimensionalidad**: PCA
4. **Reglas de Asociación**: Market Basket Analysis
5. **Evaluación**: Métricas de calidad
6. **Aplicaciones**: Segmentación de clientes

### **💡 Consejos Clave:**
- **Empezar simple**: K-Means es un buen punto de partida
- **Visualizar siempre**: Los gráficos revelan patrones
- **Validar resultados**: Usar múltiples métricas
- **Interpretar en contexto**: Los clusters deben tener sentido de negocio

¡Ahora puedes descubrir patrones ocultos en tus datos! 🎉

