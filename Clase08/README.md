[Material Diapositivas](https://docs.google.com/presentation/d/14g5eAuGgTasx_D4dJ7w7txt2qOZEY7EqvrTVTMddh74/edit#slide=id.p44)

# 📚 Clase 8: Aprendizaje No Supervisado

---

## 🔄 **Repaso Clase 7: Aprendizaje Supervisado**

### **📚 ¿Qué vimos en la Clase 7?**

En la **Clase 7** exploramos el **Aprendizaje Supervisado**, donde tenemos datos con etiquetas conocidas (target variables) para entrenar nuestros modelos.

#### **🎯 Conceptos Clave de la Clase 7:**

1. **Clasificación**: Predecir categorías (ej: spam/no spam)
   - **Árboles de Decisión**: Reglas if-else interpretables
   - **Random Forest**: Múltiples árboles para mayor precisión
   - **Métricas**: Accuracy, Precision, Recall, F1-Score

2. **Regresión**: Predecir valores numéricos (ej: precio de casa)
   - **Regresión Lineal**: Relación lineal entre variables
   - **Regresión Múltiple**: Múltiples variables predictoras
   - **Métricas**: MSE, RMSE, R²

#### **🔗 Conexión con Clase 8:**

```python
# REPASO: En Clase 7 teníamos datos ETIQUETADOS
import pandas as pd
import numpy as np

# Ejemplo de Clase 7: Clasificación de clientes
datos_clase7 = {
    'edad': [25, 35, 45, 55, 65],
    'ingresos': [30000, 50000, 70000, 60000, 40000],
    'gasto_mensual': [800, 1500, 2500, 1800, 900],
    'cliente_valioso': ['No', 'Sí', 'Sí', 'Sí', 'No']  # ← ETIQUETA CONOCIDA
}

df_clase7 = pd.DataFrame(datos_clase7)
print("=== CLASE 7: DATOS CON ETIQUETAS ===")
print("Sabemos qué clientes son valiosos...")
print(df_clase7)
print("\nObjetivo: Predecir si un NUEVO cliente será valioso")

# En Clase 8: ¿Qué pasa si NO tenemos etiquetas?
print("\n=== CLASE 8: DATOS SIN ETIQUETAS ===")
datos_clase8 = df_clase7.drop('cliente_valioso', axis=1)
print("No sabemos qué tipos de clientes tenemos...")
print("Objetivo: DESCUBRIR grupos naturales en los datos")
print(datos_clase8)
```

#### **🔄 Transición: De Supervisado a No Supervisado**

| Aspecto | Clase 7 (Supervisado) | Clase 8 (No Supervisado) |
|---------|----------------------|-------------------------|
| **Datos** | Con etiquetas conocidas | Sin etiquetas |
| **Objetivo** | Predecir/Clasificar | Descubrir patrones |
| **Evaluación** | Métricas claras (accuracy, etc.) | Métricas internas (silhouette) |
| **Interpretación** | Modelo predictivo | Exploración de datos |

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

### **🎯 ¿Qué es el Clustering?**

El **clustering** es una técnica clave en el aprendizaje no supervisado que se utiliza para agrupar un conjunto de datos no etiquetados en grupos o clústeres de datos similares. A través del clustering, los datos que comparten características similares se agrupan en el mismo clúster, mientras que los datos que son diferentes se separan en clústeres distintos.

#### **🔍 ¿Cómo Funciona?**

El proceso de clustering implica el uso de algoritmos que identifican similitudes y diferencias en los datos para formar estos grupos. Estos algoritmos no requieren información previa sobre las categorías o etiquetas de los datos, lo que les permite operar en conjuntos de datos no etiquetados.

#### **🎯 Objetivos del Clustering:**
- **Agrupar**: Encontrar grupos naturales en los datos
- **Descubrir**: Identificar patrones ocultos
- **Simplificar**: Reducir complejidad de grandes datasets
- **Segmentar**: Crear categorías para estrategias de negocio

#### **📊 Propósito en el Análisis de Datos No Etiquetados**

El objetivo principal del clustering es descubrir la estructura subyacente de un conjunto de datos no etiquetados. Al identificar y agrupar observaciones similares, el clustering permite a los analistas entender mejor las relaciones en los datos y extraer insights valiosos.

**Ejemplos de Aplicación:**
- **Marketing**: Segmentar clientes en grupos con comportamientos similares
- **Biología**: Descubrir nuevas especies al agrupar organismos con características similares
- **Medicina**: Clasificar tipos de células o tejidos
- **Finanzas**: Detectar patrones de fraude o riesgo crediticio

---

## 🔧 **8.2.1 Algoritmos Populares de Clustering**

El clustering es una técnica de aprendizaje no supervisado que agrupa datos similares en clústeres. Entre los algoritmos más utilizados en clustering se encuentran **K-means**, **DBSCAN** y el **clustering jerárquico**, cada uno con características y aplicaciones particulares.

### **1. 🔹 K-Means**

**K-means** es uno de los algoritmos de clustering más populares debido a su simplicidad y eficacia. Su objetivo es dividir un conjunto de datos en K clústeres predefinidos, donde cada dato pertenece al clúster con el centroide más cercano.

#### **🎯 Proceso del Algoritmo K-Means:**

1. **Inicialización**: Se seleccionan aleatoriamente K centroides (puntos de referencia) en el espacio de los datos
2. **Asignación**: Cada punto de datos se asigna al clúster cuyo centroide esté más cercano, minimizando la distancia euclidiana
3. **Actualización**: Se recalculan los centroides de los clústeres basándose en los datos asignados
4. **Iteración**: Los pasos de asignación y actualización se repiten hasta que los centroides ya no cambian significativamente

#### **✅ Ventajas:**
- Simple y rápido
- Escalable a grandes datasets
- Funciona bien con clusters esféricos

#### **❌ Desventajas:**
- Requiere especificar el número K de clusters
- Sensible a outliers
- Asume clusters de forma esférica
- Sensible a la inicialización aleatoria

### **2. 🔹 DBSCAN (Density-Based Spatial Clustering)**

**DBSCAN** es un algoritmo basado en la densidad que agrupa puntos que están densamente conectados, separándolos de los puntos menos densos, considerados como ruido.

#### **🎯 Conceptos Clave de DBSCAN:**

- **Vecindad ε (epsilon)**: Un parámetro que define un radio alrededor de un punto
- **MinPts**: Número mínimo de puntos dentro de la vecindad para que un punto sea considerado un punto central
- **Clústeres**: Se forman conectando puntos densamente conectados
- **Ruido**: Los puntos aislados se consideran ruido

#### **✅ Ventajas:**
- No requiere especificar el número de clusters
- Detecta clusters de forma arbitraria
- Maneja outliers automáticamente
- Robusto contra ruido

#### **❌ Desventajas:**
- Sensible a los parámetros ε y MinPts
- Difícil de ajustar para datos con densidades variables
- No funciona bien con clusters de densidad muy diferente

### **3. 🔹 Clustering Jerárquico**

El **clustering jerárquico** es un enfoque que construye una jerarquía de clústeres. Existen dos métodos principales:

#### **🌳 Tipos de Clustering Jerárquico:**

1. **Aglomerativo (bottom-up)**: 
   - Comienza tratando cada punto de datos como un clúster individual
   - Fusiona los clústeres más cercanos hasta que todos los puntos formen un solo clúster

2. **Divisivo (top-down)**:
   - Comienza con un solo clúster que contiene todos los puntos de datos
   - Lo divide en clústeres más pequeños

#### **📊 Dendrograma:**
El resultado del clustering jerárquico se visualiza comúnmente con un **dendrograma**, un árbol que muestra la estructura de fusión o división.

#### **✅ Ventajas:**
- No necesita predefinir el número de clusters
- Proporciona una estructura completa de clusters
- Interpretable visualmente

#### **❌ Desventajas:**
- Computacionalmente costoso para grandes datasets
- Sensible a outliers
- Difícil de escalar

---

## 🎓 **8.2.2 Ejemplos para Clase**

### **📚 Ejemplo 1: Clustering Simple con Datos de Estudiantes**

Vamos a empezar con un ejemplo simple y didáctico que los alumnos pueden entender fácilmente.

```python
# EJEMPLO 1: Clustering de Estudiantes por Rendimiento
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Crear datos de estudiantes (simulados)
np.random.seed(42)
n_estudiantes = 100

# Generar datos de diferentes tipos de estudiantes
# Tipo 1: Estudiantes aplicados (altas notas, muchas horas de estudio)
aplicados = {
    'horas_estudio': np.random.normal(8, 1, 30),
    'nota_promedio': np.random.normal(8.5, 0.5, 30),
    'asistencia': np.random.normal(95, 3, 30)
}

# Tipo 2: Estudiantes promedio (notas y estudio moderados)
promedio = {
    'horas_estudio': np.random.normal(5, 1, 40),
    'nota_promedio': np.random.normal(6.5, 0.8, 40),
    'asistencia': np.random.normal(80, 5, 40)
}

# Tipo 3: Estudiantes con dificultades (bajas notas, pocas horas)
dificultades = {
    'horas_estudio': np.random.normal(2, 0.5, 30),
    'nota_promedio': np.random.normal(4, 0.7, 30),
    'asistencia': np.random.normal(60, 10, 30)
}

# Combinar todos los datos
datos_estudiantes = {
    'horas_estudio': np.concatenate([aplicados['horas_estudio'], promedio['horas_estudio'], dificultades['horas_estudio']]),
    'nota_promedio': np.concatenate([aplicados['nota_promedio'], promedio['nota_promedio'], dificultades['nota_promedio']]),
    'asistencia': np.concatenate([aplicados['asistencia'], promedio['asistencia'], dificultades['asistencia']])
}

df_estudiantes = pd.DataFrame(datos_estudiantes)

print("=== EJEMPLO 1: DATOS DE ESTUDIANTES ===")
print("Objetivo: Descubrir tipos de estudiantes sin etiquetas previas")
print(f"\nDataset shape: {df_estudiantes.shape}")
print(f"\nPrimeras 5 filas:")
print(df_estudiantes.head())
print(f"\nEstadísticas descriptivas:")
print(df_estudiantes.describe())

# Visualización inicial
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(df_estudiantes['horas_estudio'], df_estudiantes['nota_promedio'], alpha=0.6)
plt.xlabel('Horas de Estudio')
plt.ylabel('Nota Promedio')
plt.title('Datos Originales: Horas vs Notas')

plt.subplot(1, 3, 2)
plt.scatter(df_estudiantes['horas_estudio'], df_estudiantes['asistencia'], alpha=0.6)
plt.xlabel('Horas de Estudio')
plt.ylabel('Asistencia (%)')
plt.title('Datos Originales: Horas vs Asistencia')

plt.subplot(1, 3, 3)
plt.scatter(df_estudiantes['nota_promedio'], df_estudiantes['asistencia'], alpha=0.6)
plt.xlabel('Nota Promedio')
plt.ylabel('Asistencia (%)')
plt.title('Datos Originales: Notas vs Asistencia')

plt.tight_layout()
plt.show()

# PASO 1: Preparar datos para clustering
print("\n=== PASO 1: PREPARACIÓN DE DATOS ===")
print("¿Por qué normalizar? Las variables tienen escalas diferentes:")
print(f"Horas de estudio: {df_estudiantes['horas_estudio'].min():.1f} - {df_estudiantes['horas_estudio'].max():.1f}")
print(f"Notas: {df_estudiantes['nota_promedio'].min():.1f} - {df_estudiantes['nota_promedio'].max():.1f}")
print(f"Asistencia: {df_estudiantes['asistencia'].min():.1f} - {df_estudiantes['asistencia'].max():.1f}")

# Normalizar datos
scaler = StandardScaler()
X_estudiantes = scaler.fit_transform(df_estudiantes)
print(f"\nDatos normalizados shape: {X_estudiantes.shape}")

# PASO 2: Método del Codo para encontrar K óptimo
print("\n=== PASO 2: ENCONTRAR NÚMERO ÓPTIMO DE CLUSTERS ===")

def metodo_codo(X, max_k=8):
    """Método del codo para encontrar k óptimo"""
    inertias = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    return k_range, inertias

k_range, inertias = metodo_codo(X_estudiantes)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo - Estudiantes')
plt.grid(True)
plt.axvline(x=3, color='red', linestyle='--', label='K=3 (óptimo)')
plt.legend()
plt.show()

print("Interpretación: El 'codo' está en K=3, donde la inercia deja de disminuir rápidamente")

# PASO 3: Aplicar K-Means con K=3
print("\n=== PASO 3: APLICAR K-MEANS ===")
kmeans_estudiantes = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_estudiantes = kmeans_estudiantes.fit_predict(X_estudiantes)

# Agregar clusters al dataframe
df_estudiantes['cluster'] = clusters_estudiantes

# PASO 4: Visualizar resultados
print("\n=== PASO 4: VISUALIZAR RESULTADOS ===")
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
scatter = plt.scatter(df_estudiantes['horas_estudio'], df_estudiantes['nota_promedio'], 
                     c=df_estudiantes['cluster'], cmap='viridis', alpha=0.7)
plt.xlabel('Horas de Estudio')
plt.ylabel('Nota Promedio')
plt.title('K-Means: Horas vs Notas')
plt.colorbar(scatter, label='Cluster')

plt.subplot(1, 3, 2)
scatter = plt.scatter(df_estudiantes['horas_estudio'], df_estudiantes['asistencia'], 
                     c=df_estudiantes['cluster'], cmap='viridis', alpha=0.7)
plt.xlabel('Horas de Estudio')
plt.ylabel('Asistencia (%)')
plt.title('K-Means: Horas vs Asistencia')
plt.colorbar(scatter, label='Cluster')

plt.subplot(1, 3, 3)
scatter = plt.scatter(df_estudiantes['nota_promedio'], df_estudiantes['asistencia'], 
                     c=df_estudiantes['cluster'], cmap='viridis', alpha=0.7)
plt.xlabel('Nota Promedio')
plt.ylabel('Asistencia (%)')
plt.title('K-Means: Notas vs Asistencia')
plt.colorbar(scatter, label='Cluster')

plt.tight_layout()
plt.show()

# PASO 5: Interpretar clusters
print("\n=== PASO 5: INTERPRETAR CLUSTERS ===")
for i in range(3):
    cluster_data = df_estudiantes[df_estudiantes['cluster'] == i]
    print(f"\n🔹 Cluster {i} ({len(cluster_data)} estudiantes):")
    print(f"   Horas de estudio promedio: {cluster_data['horas_estudio'].mean():.1f}")
    print(f"   Nota promedio: {cluster_data['nota_promedio'].mean():.1f}")
    print(f"   Asistencia promedio: {cluster_data['asistencia'].mean():.1f}%")
    
    # Interpretación
    if cluster_data['horas_estudio'].mean() > 6:
        print("   📚 Interpretación: ESTUDIANTES APLICADOS")
    elif cluster_data['horas_estudio'].mean() > 3:
        print("   📖 Interpretación: ESTUDIANTES PROMEDIO")
    else:
        print("   📝 Interpretación: ESTUDIANTES CON DIFICULTADES")

print("\n✅ CONCLUSIÓN: El algoritmo descubrió automáticamente 3 tipos de estudiantes")
print("   sin necesidad de etiquetas previas!")
```

### **📚 Ejemplo 2: Clustering de Productos de E-commerce**

Un ejemplo más complejo y realista para mostrar aplicaciones empresariales.

```python
# EJEMPLO 2: Segmentación de Productos de E-commerce
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Crear datos de productos de e-commerce
np.random.seed(42)

# Generar diferentes tipos de productos
# Tipo 1: Productos premium (alto precio, baja rotación, alta satisfacción)
premium = {
    'precio': np.random.normal(500, 100, 50),
    'ventas_mensuales': np.random.poisson(20, 50),
    'rating_promedio': np.random.normal(4.8, 0.2, 50),
    'stock_dias': np.random.normal(45, 10, 50)
}

# Tipo 2: Productos populares (precio medio, alta rotación, buen rating)
populares = {
    'precio': np.random.normal(150, 30, 100),
    'ventas_mensuales': np.random.poisson(200, 100),
    'rating_promedio': np.random.normal(4.2, 0.3, 100),
    'stock_dias': np.random.normal(15, 5, 100)
}

# Tipo 3: Productos básicos (bajo precio, rotación media, rating variable)
basicos = {
    'precio': np.random.normal(50, 15, 80),
    'ventas_mensuales': np.random.poisson(80, 80),
    'rating_promedio': np.random.normal(3.5, 0.5, 80),
    'stock_dias': np.random.normal(30, 8, 80)
}

# Combinar datos
datos_productos = {
    'precio': np.concatenate([premium['precio'], populares['precio'], basicos['precio']]),
    'ventas_mensuales': np.concatenate([premium['ventas_mensuales'], populares['ventas_mensuales'], basicos['ventas_mensuales']]),
    'rating_promedio': np.concatenate([premium['rating_promedio'], populares['rating_promedio'], basicos['rating_promedio']]),
    'stock_dias': np.concatenate([premium['stock_dias'], populares['stock_dias'], basicos['stock_dias']])
}

df_productos = pd.DataFrame(datos_productos)

print("=== EJEMPLO 2: PRODUCTOS DE E-COMMERCE ===")
print("Objetivo: Segmentar productos para estrategias de marketing y pricing")
print(f"\nDataset shape: {df_productos.shape}")
print(f"\nEstadísticas descriptivas:")
print(df_productos.describe())

# Visualización inicial
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(df_productos['precio'], df_productos['ventas_mensuales'], alpha=0.6)
plt.xlabel('Precio ($)')
plt.ylabel('Ventas Mensuales')
plt.title('Precio vs Ventas')

plt.subplot(1, 3, 2)
plt.scatter(df_productos['precio'], df_productos['rating_promedio'], alpha=0.6)
plt.xlabel('Precio ($)')
plt.ylabel('Rating Promedio')
plt.title('Precio vs Rating')

plt.subplot(1, 3, 3)
plt.scatter(df_productos['ventas_mensuales'], df_productos['rating_promedio'], alpha=0.6)
plt.xlabel('Ventas Mensuales')
plt.ylabel('Rating Promedio')
plt.title('Ventas vs Rating')

plt.tight_layout()
plt.show()

# Normalizar datos
scaler_productos = StandardScaler()
X_productos = scaler_productos.fit_transform(df_productos)

# Aplicar K-Means
kmeans_productos = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_productos = kmeans_productos.fit_predict(X_productos)

# Agregar clusters
df_productos['cluster'] = clusters_productos

# Visualizar resultados
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
scatter = plt.scatter(df_productos['precio'], df_productos['ventas_mensuales'], 
                     c=df_productos['cluster'], cmap='viridis', alpha=0.7)
plt.xlabel('Precio ($)')
plt.ylabel('Ventas Mensuales')
plt.title('K-Means: Precio vs Ventas')
plt.colorbar(scatter, label='Cluster')

plt.subplot(1, 3, 2)
scatter = plt.scatter(df_productos['precio'], df_productos['rating_promedio'], 
                     c=df_productos['cluster'], cmap='viridis', alpha=0.7)
plt.xlabel('Precio ($)')
plt.ylabel('Rating Promedio')
plt.title('K-Means: Precio vs Rating')
plt.colorbar(scatter, label='Cluster')

plt.subplot(1, 3, 3)
scatter = plt.scatter(df_productos['ventas_mensuales'], df_productos['rating_promedio'], 
                     c=df_productos['cluster'], cmap='viridis', alpha=0.7)
plt.xlabel('Ventas Mensuales')
plt.ylabel('Rating Promedio')
plt.title('K-Means: Ventas vs Rating')
plt.colorbar(scatter, label='Cluster')

plt.tight_layout()
plt.show()

# Análisis de clusters
print("\n=== ANÁLISIS DE SEGMENTOS DE PRODUCTOS ===")
for i in range(3):
    cluster_data = df_productos[df_productos['cluster'] == i]
    print(f"\n🔹 Segmento {i} ({len(cluster_data)} productos):")
    print(f"   Precio promedio: ${cluster_data['precio'].mean():.0f}")
    print(f"   Ventas promedio: {cluster_data['ventas_mensuales'].mean():.0f} unidades/mes")
    print(f"   Rating promedio: {cluster_data['rating_promedio'].mean():.1f}/5")
    print(f"   Stock promedio: {cluster_data['stock_dias'].mean():.0f} días")
    
    # Estrategia de negocio
    if cluster_data['precio'].mean() > 300:
        print("   💎 Estrategia: PRODUCTOS PREMIUM - Marketing exclusivo, alta calidad")
    elif cluster_data['ventas_mensuales'].mean() > 150:
        print("   🚀 Estrategia: PRODUCTOS POPULARES - Promociones, stock alto")
    else:
        print("   📦 Estrategia: PRODUCTOS BÁSICOS - Precios competitivos, rotación media")

# Calcular métricas de calidad
silhouette_avg = silhouette_score(X_productos, clusters_productos)
print(f"\n📊 Métrica de Calidad:")
print(f"   Silhouette Score: {silhouette_avg:.3f} (0.5+ es bueno)")

print("\n✅ APLICACIÓN PRÁCTICA:")
print("   - Segmento 0: Estrategia de pricing premium")
print("   - Segmento 1: Campañas de marketing masivo")
print("   - Segmento 2: Optimización de inventario")
```

---

## 📋 **8.2.4 Resumen de Algoritmos de Clustering**

### **🎯 Comparación de Algoritmos Populares**

| Algoritmo | Mejor Para | Ventajas | Desventajas | Cuándo Usar |
|-----------|------------|----------|-------------|-------------|
| **K-Means** | Clusters esféricos bien separados | Simple, rápido, escalable | Requiere K, sensible a outliers | Datos con clusters claros y esféricos |
| **DBSCAN** | Clusters de forma arbitraria | No requiere K, maneja outliers | Sensible a parámetros | Datos con ruido, formas irregulares |
| **Jerárquico** | Exploración de estructura | No requiere K, interpretable | Costoso computacionalmente | Datasets pequeños, exploración |

### **🔍 Clustering Basado en Densidad (DBSCAN)**

Los métodos de clustering basados en densidad identifican grupos en un conjunto de datos considerando la densidad de puntos en el espacio de datos. A diferencia de otros métodos como K-means, que dependen de la distancia entre puntos y requieren definir el número de clústeres, estos métodos se enfocan en encontrar regiones densamente pobladas separadas por áreas de baja densidad.

#### **🎯 ¿Cómo Funcionan?**

El principio central es que los clústeres se forman en áreas contiguas de alta densidad, con las regiones de baja densidad actuando como separadores. Un algoritmo destacado es **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise), que funciona así:

1. **Vecindad ε (epsilon)**: Define un radio alrededor de un punto. Si hay suficientes puntos dentro de este radio (MinPts), el área es densa.
2. **Puntos Centrales y Frontera**: Un punto central tiene al menos MinPts en su vecindad ε; un punto frontera tiene menos pero está cerca de un punto central.
3. **Expansión de Clústeres**: Un clúster crece incluyendo puntos en la vecindad ε hasta que no se puedan agregar más.
4. **Ruido**: Los puntos que no pertenecen a una vecindad densa se consideran ruido o outliers.

#### **🎯 Propósito y Aplicaciones**

DBSCAN es útil para detectar clústeres de formas arbitrarias y manejar outliers sin necesidad de especificar el número de clústeres. Es aplicado en:
- **Análisis geoespacial**: Agrupar ubicaciones por densidad
- **Segmentación de clientes**: En marketing, donde los datos no tienen formas esféricas definidas
- **Detección de anomalías**: Identificar patrones inusuales

### **🌳 Clustering Jerárquico y Dendrogramas**

Un **dendrograma** es una representación visual que muestra cómo se agrupan y dividen los clústeres en un proceso de clustering jerárquico. Cada bifurcación en el diagrama representa un punto de unión entre los datos, ayudando a visualizar las relaciones entre diferentes clústeres y la estructura jerárquica de los mismos.

#### **🎯 Interpretación del Dendrograma:**
- **Altura de las líneas**: Indica la distancia entre clusters
- **Bifurcaciones**: Muestran cómo se fusionan los clusters
- **Corte horizontal**: Determina el número final de clusters

### **✅ Resumen Final**

En resumen, el clustering es una herramienta poderosa en el análisis de datos no etiquetados, proporcionando un método para organizar y explorar grandes volúmenes de datos al identificar patrones y estructuras ocultas, sin necesidad de etiquetas o categorías predefinidas.

**🎯 Consejos para Elegir el Algoritmo Correcto:**
1. **K-Means**: Ideal para clusters bien definidos, rápido y sencillo, pero sensible a los outliers y requiere que se especifique K.
2. **DBSCAN**: Eficaz para clusters de formas arbitrarias y robusto contra outliers, no requiere especificar el número de clusters, pero depende de la correcta elección de parámetros.
3. **Clustering Jerárquico**: Ofrece una estructura completa de clusters y no necesita predefinir el número de clusters, aunque puede ser costoso en términos de tiempo de cálculo para grandes datasets.

Estos algoritmos son fundamentales en el análisis de datos no etiquetados, proporcionando diferentes enfoques para descubrir patrones ocultos y estructurar la información de manera significativa.

---

## 📊 **8.2.3 Ejemplo Práctico Avanzado: Segmentación de Clientes**

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

### **🎯 ¿Qué es la Reducción de Dimensionalidad?**

La **reducción de dimensionalidad** es una técnica clave en el análisis de datos complejos que busca simplificar los conjuntos de datos sin perder información relevante. En la práctica, los conjuntos de datos suelen tener un gran número de variables o características, lo que puede hacer que el análisis sea complicado y computacionalmente costoso. La reducción de dimensionalidad permite disminuir el número de variables en un dataset, facilitando así la visualización, el almacenamiento y el procesamiento de los datos.

#### **🎯 Objetivos de la Reducción de Dimensionalidad:**

1. **Simplificación del Modelo**: Reducir la complejidad del modelo al disminuir el número de variables sin sacrificar la precisión en las predicciones o análisis.

2. **Mejora de la Interpretabilidad**: Facilitar la comprensión y visualización de los datos al representarlos en un espacio de menor dimensión.

3. **Mitigación de la Maldición de la Dimensionalidad**: Enfrentar los problemas que surgen cuando los datos tienen demasiadas dimensiones, lo que puede llevar a un sobreajuste del modelo y a una interpretación errónea de los resultados.

#### **✅ Beneficios en el Análisis de Datos:**

- **Mejora del Rendimiento Computacional**: Menos dimensiones implican menos datos que procesar, lo que se traduce en algoritmos más rápidos y eficientes.

- **Reducción de Ruido**: Eliminar variables irrelevantes o redundantes puede mejorar la calidad del análisis al enfocarse solo en las características más importantes.

- **Facilitación de la Visualización**: Los datos en menor dimensión pueden representarse más fácilmente en gráficos, ayudando a detectar patrones y relaciones que no serían evidentes en un espacio de mayor dimensión.

#### **🔍 ¿Por qué reducir dimensiones?**

1. **Visualización**: Solo podemos ver 2-3 dimensiones
2. **Curse of Dimensionality**: Más dimensiones = más datos necesarios
3. **Ruido**: Dimensiones irrelevantes confunden algoritmos
4. **Computación**: Menos dimensiones = más rápido

---

## 🔧 **8.3.1 Técnicas de Reducción: PCA (Principal Component Analysis)**

El **Análisis de Componentes Principales (PCA)** es una de las técnicas más utilizadas en la reducción de dimensionalidad en conjuntos de datos. Su objetivo principal es transformar un conjunto de variables posiblemente correlacionadas en un conjunto de valores de variables no correlacionadas, llamadas componentes principales.

### **🎯 ¿Qué es PCA?**

PCA es un método estadístico que convierte un conjunto de observaciones de variables posiblemente correlacionadas en un conjunto de valores de variables no correlacionadas denominadas **componentes principales**. Este proceso se lleva a cabo de tal manera que:

- El **primer componente principal** tiene la mayor varianza posible (explica la mayor parte de la variabilidad en los datos)
- Cada **componente sucesivo** tiene la mayor varianza posible bajo la restricción de ser ortogonal a los componentes anteriores

### **🔍 Aplicación de PCA en la Reducción de Dimensiones:**

1. **Identificación de la Varianza**: PCA identifica las direcciones (componentes principales) en las que la varianza en los datos es máxima. Esto permite capturar la estructura esencial de los datos con menos dimensiones.

2. **Transformación de Datos**: Los datos originales se proyectan sobre los componentes principales seleccionados, reduciendo así el número de dimensiones mientras se retiene la mayor parte de la información original.

3. **Eliminación de Ruido**: Al centrarse en los componentes principales que capturan la mayor parte de la variabilidad, PCA ayuda a eliminar el ruido y las redundancias de los datos.

### **✅ Beneficios de Usar PCA:**

- **Simplificación**: PCA reduce la cantidad de datos a procesar, facilitando el análisis y mejorando el rendimiento computacional.
- **Visualización**: La reducción de dimensiones mediante PCA permite una mejor visualización de los datos, especialmente cuando se reduce a 2 o 3 dimensiones.
- **Optimización de Modelos**: Al eliminar la multicolinealidad y concentrar la varianza en un menor número de componentes, los modelos pueden ser más precisos y menos propensos al sobreajuste.

---

## 🎓 **8.3.2 Ejemplo Práctico: PCA con Datos de Clientes**

Vamos a ver PCA en acción con un ejemplo completo y bien explicado:

```python
# EJEMPLO: PCA con Datos de Clientes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Crear dataset con múltiples variables
np.random.seed(42)
n_clientes = 200

# Generar datos correlacionados (simulando comportamiento real)
# Variables relacionadas con el comportamiento del cliente
datos_clientes = {
    'edad': np.random.normal(35, 12, n_clientes),
    'ingresos_anuales': np.random.normal(50000, 15000, n_clientes),
    'gasto_mensual': np.random.normal(2000, 600, n_clientes),
    'frecuencia_compras': np.random.poisson(8, n_clientes),
    'satisfaccion': np.random.normal(4.2, 0.8, n_clientes),
    'tiempo_cliente': np.random.normal(24, 8, n_clientes),  # meses
    'productos_comprados': np.random.poisson(15, n_clientes),
    'descuentos_usados': np.random.poisson(3, n_clientes)
}

# Agregar correlaciones realistas
datos_clientes['gasto_mensual'] += datos_clientes['ingresos_anuales'] * 0.02  # Correlación positiva
datos_clientes['frecuencia_compras'] += datos_clientes['gasto_mensual'] * 0.01
datos_clientes['satisfaccion'] += datos_clientes['tiempo_cliente'] * 0.01

df_clientes = pd.DataFrame(datos_clientes)

print("=== EJEMPLO: PCA CON DATOS DE CLIENTES ===")
print("Objetivo: Reducir 8 variables a 2-3 componentes principales")
print(f"\nDataset shape: {df_clientes.shape}")
print(f"\nVariables originales:")
print(df_clientes.columns.tolist())
print(f"\nEstadísticas descriptivas:")
print(df_clientes.describe())

# PASO 1: Análisis de correlaciones
print("\n=== PASO 1: ANÁLISIS DE CORRELACIONES ===")
correlation_matrix = df_clientes.corr()
print("Matriz de correlaciones:")
print(correlation_matrix.round(3))

# Visualizar correlaciones
plt.figure(figsize=(10, 8))
import seaborn as sns
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f')
plt.title('Matriz de Correlaciones - Variables Originales')
plt.tight_layout()
plt.show()

print("\nObservación: Hay correlaciones fuertes entre variables")
print("Esto indica redundancia que PCA puede eliminar")

# PASO 2: Normalizar datos
print("\n=== PASO 2: NORMALIZACIÓN DE DATOS ===")
print("¿Por qué normalizar? Las variables tienen escalas muy diferentes:")
for col in df_clientes.columns:
    print(f"{col}: {df_clientes[col].min():.1f} - {df_clientes[col].max():.1f}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clientes)
print(f"\nDatos normalizados shape: {X_scaled.shape}")

# PASO 3: Aplicar PCA
print("\n=== PASO 3: APLICAR PCA ===")

# Primero, analizar cuántos componentes necesitamos
pca_full = PCA()
pca_full.fit(X_scaled)

# Calcular varianza explicada acumulada
explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Visualizar varianza explicada
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.xlabel('Componente Principal')
plt.ylabel('Varianza Explicada')
plt.title('Varianza Explicada por Componente')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
plt.axhline(y=0.95, color='red', linestyle='--', label='95% de varianza')
plt.axhline(y=0.85, color='orange', linestyle='--', label='85% de varianza')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada Acumulada')
plt.title('Varianza Explicada Acumulada')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Mostrar información detallada
print("Varianza explicada por cada componente:")
for i, var in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {var:.1%}")

print(f"\nVarianza explicada acumulada:")
for i, cum_var in enumerate(cumulative_variance):
    print(f"PC1-PC{i+1}: {cum_var:.1%}")

# Decidir número de componentes (usar 2 para visualización)
n_components = 2
print(f"\nDecisión: Usar {n_components} componentes principales")
print(f"Esto explica el {cumulative_variance[n_components-1]:.1%} de la varianza")

# Aplicar PCA con 2 componentes
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

print(f"\nDatos transformados shape: {X_pca.shape}")
print("Reducción: 8 variables → 2 componentes principales")

# PASO 4: Interpretar componentes principales
print("\n=== PASO 4: INTERPRETAR COMPONENTES PRINCIPALES ===")

# Crear dataframe con los componentes
components_df = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(n_components)],
    index=df_clientes.columns
)

print("Cargas de los componentes principales:")
print(components_df.round(3))

# Visualizar cargas
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.barh(components_df.index, components_df['PC1'])
plt.xlabel('Carga en PC1')
plt.title('Cargas del Primer Componente Principal')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.barh(components_df.index, components_df['PC2'])
plt.xlabel('Carga en PC2')
plt.title('Cargas del Segundo Componente Principal')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Interpretar componentes
print("\n🔍 INTERPRETACIÓN DE COMPONENTES:")
print("PC1 (Primer Componente):")
pc1_high = components_df['PC1'].abs().nlargest(3)
for var, load in pc1_high.items():
    direction = "positiva" if components_df.loc[var, 'PC1'] > 0 else "negativa"
    print(f"  - {var}: carga {direction} fuerte ({load:.3f})")

print("\nPC2 (Segundo Componente):")
pc2_high = components_df['PC2'].abs().nlargest(3)
for var, load in pc2_high.items():
    direction = "positiva" if components_df.loc[var, 'PC2'] > 0 else "negativa"
    print(f"  - {var}: carga {direction} fuerte ({load:.3f})")

# PASO 5: Visualizar datos en 2D
print("\n=== PASO 5: VISUALIZACIÓN EN 2D ===")

# Aplicar clustering en el espacio reducido para colorear puntos
kmeans_pca = KMeans(n_clusters=3, random_state=42)
clusters_pca = kmeans_pca.fit_predict(X_pca)

plt.figure(figsize=(15, 5))

# Visualización original (primeras 2 variables)
plt.subplot(1, 3, 1)
plt.scatter(df_clientes['edad'], df_clientes['ingresos_anuales'], 
           c=clusters_pca, cmap='viridis', alpha=0.6)
plt.xlabel('Edad')
plt.ylabel('Ingresos Anuales')
plt.title('Datos Originales (2 variables)')

# Visualización PCA
plt.subplot(1, 3, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_pca, 
                     cmap='viridis', alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('PCA: Reducción a 2D')
plt.colorbar(scatter, label='Cluster')

# Comparar con datos originales (otras 2 variables)
plt.subplot(1, 3, 3)
plt.scatter(df_clientes['gasto_mensual'], df_clientes['frecuencia_compras'], 
           c=clusters_pca, cmap='viridis', alpha=0.6)
plt.xlabel('Gasto Mensual')
plt.ylabel('Frecuencia Compras')
plt.title('Datos Originales (otras 2 variables)')

plt.tight_layout()
plt.show()

# PASO 6: Análisis de clusters en espacio PCA
print("\n=== PASO 6: ANÁLISIS DE CLUSTERS EN ESPACIO PCA ===")

# Crear dataframe con datos PCA
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
df_pca['cluster'] = clusters_pca

for i in range(3):
    cluster_data = df_pca[df_pca['cluster'] == i]
    print(f"\n🔹 Cluster {i} ({len(cluster_data)} clientes):")
    print(f"   PC1 promedio: {cluster_data['PC1'].mean():.2f}")
    print(f"   PC2 promedio: {cluster_data['PC2'].mean():.2f}")

print("\n✅ BENEFICIOS OBTENIDOS CON PCA:")
print("   - Reducción de 8 variables a 2 componentes")
print("   - Mantenimiento del 85%+ de la información")
print("   - Eliminación de redundancias")
print("   - Mejor visualización de patrones")
print("   - Clustering más eficiente")

# PASO 7: Comparar rendimiento
print("\n=== PASO 7: COMPARAR RENDIMIENTO ===")

# Clustering en datos originales
kmeans_original = KMeans(n_clusters=3, random_state=42)
clusters_original = kmeans_original.fit_predict(X_scaled)

# Comparar inercias
print(f"Inercia con datos originales: {kmeans_original.inertia_:.2f}")
print(f"Inercia con datos PCA: {kmeans_pca.inertia_:.2f}")
print(f"Mejora en eficiencia: {((kmeans_original.inertia_ - kmeans_pca.inertia_) / kmeans_original.inertia_ * 100):.1f}%")

print("\n🎯 CONCLUSIÓN:")
print("PCA nos permitió reducir significativamente la dimensionalidad")
print("manteniendo la información esencial y mejorando la interpretabilidad")
```

---

## 🚀 **8.3.3 Aplicaciones Prácticas de Reducción de Dimensionalidad**

La reducción de dimensionalidad es una técnica ampliamente utilizada en la ciencia de datos, no solo para simplificar conjuntos de datos complejos, sino también para mejorar la precisión y eficiencia de los modelos de predicción y facilitar la visualización de datos.

### **🎯 Mejora de Modelos de Predicción**

#### **1. Mitigación de la Maldición de la Dimensionalidad:**
A medida que el número de características en un conjunto de datos aumenta, los modelos de predicción pueden volverse más propensos al sobreajuste, ya que se incrementa la complejidad del modelo. La reducción de dimensionalidad ayuda a simplificar el modelo al eliminar variables irrelevantes o redundantes, mejorando así su capacidad de generalización.

#### **2. Reducción del Ruido:**
Los datos de alta dimensionalidad a menudo contienen ruido, es decir, variables que no contribuyen significativamente al modelo o incluso pueden confundirlo. Técnicas como PCA permiten identificar y conservar solo las componentes principales que explican la mayor parte de la varianza, reduciendo así el ruido y mejorando la precisión del modelo.

#### **3. Optimización Computacional:**
Menos dimensiones significan menos datos que procesar. Esto reduce el tiempo y los recursos computacionales necesarios para entrenar modelos de aprendizaje automático, permitiendo trabajar con conjuntos de datos más grandes o con algoritmos más complejos sin comprometer el rendimiento.

### **📊 Visualización de Datos Complejos**

#### **1. Simplificación de la Representación Gráfica:**
Visualizar datos en alta dimensionalidad es un desafío. La reducción de dimensionalidad permite proyectar datos multidimensionales en 2 o 3 dimensiones, haciendo posible la representación gráfica y la identificación visual de patrones, tendencias y agrupamientos.

#### **2. Descubrimiento de Patrones Ocultos:**
Al reducir las dimensiones, se puede visualizar la estructura subyacente de los datos que de otro modo permanecería oculta en un espacio de alta dimensionalidad. Por ejemplo, PCA puede ayudar a identificar grupos o clústeres de datos que no son evidentes en las dimensiones originales.

#### **3. Facilitación de la Interpretación de Datos:**
La representación de datos en un espacio de menor dimensión facilita la interpretación y el análisis, permitiendo a los analistas de datos y a los responsables de la toma de decisiones comprender mejor las relaciones entre las variables y las tendencias generales en los datos.

### **🎯 Casos de Uso Comunes:**

| Aplicación | Técnica | Beneficio |
|------------|---------|-----------|
| **Análisis de Imágenes** | PCA, t-SNE | Reducir píxeles a características principales |
| **Análisis de Texto** | LSA, LDA | Reducir dimensiones de vectores de palabras |
| **Genómica** | PCA, UMAP | Analizar miles de genes simultáneamente |
| **Finanzas** | PCA, Factor Analysis | Identificar factores de riesgo principales |
| **Marketing** | PCA, MDS | Segmentar clientes en espacios reducidos |

---

## 📋 **8.3.4 Resumen de Reducción de Dimensionalidad**

### **✅ En Resumen:**

La reducción de dimensionalidad es una herramienta esencial en la ciencia de datos, particularmente cuando se trabaja con conjuntos de datos grandes y complejos, donde el objetivo es mantener la mayor cantidad de información posible mientras se simplifican los modelos y se mejora la interpretación.

**🎯 Técnicas Principales:**
- **PCA**: Para datos lineales, preserva varianza
- **t-SNE**: Para visualización, preserva estructura local
- **UMAP**: Balance entre estructura local y global
- **Factor Analysis**: Para datos con estructura factorial

**🔍 Cuándo Usar:**
- Datasets con muchas variables (>10)
- Variables altamente correlacionadas
- Necesidad de visualización
- Problemas de rendimiento computacional
- Eliminación de ruido

**⚠️ Consideraciones:**
- Pérdida de interpretabilidad
- Posible pérdida de información
- Sensibilidad a escalas
- Supuestos sobre la estructura de datos

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

