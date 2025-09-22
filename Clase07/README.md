## 📚 Repaso Clase Anterior (Clase 6)

### Teoría Fundamental del EDA y Preprocesamiento

En la **Clase 6** establecimos las bases fundamentales para el análisis de datos que ahora aplicaremos en **Machine Learning**:

#### 🔍 **Análisis Exploratorio de Datos (EDA)**
- **Filosofía**: Acercarse a los datos sin prejuicios para descubrir patrones inesperados
- **Objetivo**: Entender la estructura de los datos antes de aplicar modelos predictivos
- **Herramientas**: Estadística descriptiva + visualización de datos

#### 📊 **Estadística Descriptiva**
- **Medidas de tendencia central**: Media, mediana, moda
- **Medidas de dispersión**: Varianza, desviación estándar, IQR
- **Distribuciones**: Normal, uniforme, y su visualización con histogramas
- **Correlación**: Relación entre variables (importante: correlación ≠ causalidad)

#### 🧹 **Preprocesamiento de Datos**
- **Limpieza**: Manejo de valores faltantes y outliers
- **Transformación**: Normalización, codificación de variables categóricas
- **Integración**: Combinar datos de múltiples fuentes
- **Reducción**: PCA para simplificar la dimensionalidad

---

### 💡 Ejemplo 1: Análisis Estadístico Descriptivo

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Datos de ventas mensuales de una empresa
np.random.seed(42)
ventas = np.random.normal(50000, 12000, 24)  # 2 años de datos
ventas = np.append(ventas, [150000, -5000])  # outliers

df_ventas = pd.DataFrame({"Ventas_Mensuales": ventas})

# Medidas descriptivas
print("=== ESTADÍSTICAS DESCRIPTIVAS ===")
print(f"Media: {df_ventas['Ventas_Mensuales'].mean():.2f}")
print(f"Mediana: {df_ventas['Ventas_Mensuales'].median():.2f}")
print(f"Desviación estándar: {df_ventas['Ventas_Mensuales'].std():.2f}")

# Visualización
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.histplot(df_ventas["Ventas_Mensuales"], kde=True, bins=15)
plt.title("Distribución de Ventas Mensuales")

plt.subplot(1,2,2)
sns.boxplot(x=df_ventas["Ventas_Mensuales"])
plt.title("Boxplot - Detección de Outliers")
plt.show()
```

**🎯 Objetivo**: Identificar patrones en los datos de ventas y detectar valores atípicos que podrían afectar modelos predictivos.

---

### 💡 Ejemplo 2: Preprocesamiento con PCA

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Dataset con múltiples variables correlacionadas
np.random.seed(42)
X1 = np.random.normal(100, 20, 100)
X2 = X1 * 0.8 + np.random.normal(0, 5, 100)  # correlacionada con X1
X3 = np.random.normal(50, 10, 100)           # independiente

df_features = pd.DataFrame({
    "Ingresos": X1,
    "Gastos": X2, 
    "Ahorros": X3
})

print("=== CORRELACIONES ORIGINALES ===")
print(df_features.corr())

# Estandarización (importante para PCA)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)

# Aplicar PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

print(f"\n=== VARIANZA EXPLICADA ===")
print(f"PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"PC2: {pca.explained_variance_ratio_[1]:.2%}")

# Visualización
df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
plt.figure(figsize=(6,6))
sns.scatterplot(x="PC1", y="PC2", data=df_pca, s=60)
plt.title("Datos transformados con PCA")
plt.show()
```

**🎯 Objetivo**: Reducir la dimensionalidad eliminando redundancia entre variables correlacionadas, preparando datos más limpios para algoritmos de ML.

---

### 🔗 **Conexión con Machine Learning**

Los conceptos de la **Clase 6** son **fundamentales** para el éxito en ML:

- **EDA** → Nos ayuda a entender qué variables son relevantes para predecir
- **Estadística descriptiva** → Identifica distribuciones y relaciones que los algoritmos pueden aprovechar
- **Preprocesamiento** → Asegura que los datos estén limpios y listos para entrenar modelos
- **PCA** → Reduce complejidad y mejora el rendimiento de los algoritmos

---

### **¿Qué es Machine Learning?**  
**Machine Learning (ML)** o *Aprendizaje Automático* es una rama de la **Inteligencia Artificial (IA)** que se centra en desarrollar algoritmos y modelos capaces de aprender patrones a partir de datos, **sin ser programados explícitamente**. En lugar de seguir reglas predefinidas, los sistemas de ML **mejoran su desempeño con la experiencia** (datos).  

Su objetivo principal es **generalizar** a partir de ejemplos para realizar predicciones o tomar decisiones en situaciones nuevas.

---

### **Particularidades y Características Principales**  

1. **Aprendizaje basado en datos**  
   - ML requiere **datos históricos o de entrenamiento** para identificar patrones y relaciones.  
   - A diferencia de la programación tradicional (donde las reglas son fijas), en ML **el modelo "aprende" de los datos**.  

2. **Capacidad predictiva y adaptativa**  
   - Los modelos de ML pueden **predecir resultados futuros** (ej.: ventas, fallos en equipos) o **clasificar información** (ej.: spam/no spam).  
   - Algunos sistemas se adaptan a cambios en los datos (ej.: recomendaciones en tiempo real en Netflix o Amazon).  

3. **Tipos principales de aprendizaje**  
   - **Supervisado**: Usa datos etiquetados (ej.: predecir precios de casas basado en ejemplos pasados).  
   - **No supervisado**: Encuentra patrones en datos sin etiquetas (ej.: agrupación de clientes por comportamiento).  
   - **Por refuerzo**: Aprende mediante prueba/error y recompensas (ej.: robots que aprenden a caminar).  

4. **Automatización y escalabilidad**  
   - ML permite automatizar tareas complejas (ej.: diagnóstico médico, detección de fraudes).  
   - Escala bien con grandes volúmenes de datos (*Big Data*).  

5. **Énfasis en la evaluación**  
   - Los modelos se validan con métricas como **precisión, recall o error cuadrático medio** para garantizar su fiabilidad.  

6. **Dependencia de la calidad de los datos**  
   - El rendimiento del ML está ligado a la **calidad, cantidad y representatividad** de los datos.  
   - Problemas como *sesgos* o *datos incompletos* afectan los resultados.  

7. **Uso de algoritmos diversos**  
   - Desde métodos clásicos (*regresión lineal, árboles de decisión*) hasta técnicas avanzadas (*redes neuronales, deep learning*).  

---

### **Diferencia clave con Data Science**  
Mientras **Data Science** abarca todo el ciclo de análisis de datos (limpieza, visualización, estadística, etc.), **ML es una herramienta dentro de DS** enfocada específicamente en **automatizar el aprendizaje** para predicción o toma de decisiones.  

**Ejemplo práctico**:  
- Un modelo de ML podría predecir el *churn* (abandono) de clientes en una empresa, mientras que un data scientist también analizaría *por qué* ocurre y cómo comunicarlo.  

En resumen, **Machine Learning es la tecnología que permite a las máquinas "aprender" de los datos para resolver problemas complejos de manera autónoma o semiautónoma**. 

---


### _Si quisiéramos resolver un problema donde tenemos información georeferenciada de clientes ¿cómo podríamos utilizar el ML para incrementar las ventas de un producto?_

Para incrementar las ventas de un producto utilizando **Machine Learning (ML)** con datos georreferenciados de clientes, puedes aplicar diversas estrategias basadas en análisis espacial y modelos predictivos. Aquí te detallo un enfoque estructurado:

---

### **1. Análisis Exploratorio de Datos (EDA) Geoespacial**  
- **Visualización de datos**:  
  - Usar mapas de calor (*heatmaps*) para identificar zonas con alta concentración de clientes o ventas.  
  - Segmentar por zonas geográficas (barrios, ciudades, códigos postales).  
- **Detección de patrones**:  
  - Correlacionar ubicación con variables como ingresos, edad, clima o proximidad a puntos de interés (ej.: centros comerciales).  

**Herramientas**: Python (`geopandas`, `folium`), Power BI (integrado con Mapas).  

---

### **2. Segmentación de Clientes por Ubicación**  
- **Clustering no supervisado** (ej.: *K-means* o *DBSCAN*) para agrupar clientes con características similares:  
  - Crear clusters basados en:  
    - Ubicación (coordenadas).  
    - Comportamiento de compra + datos demográficos locales.  
  - Ejemplo: Identificar "zonas de alto potencial" con clientes similares a los que ya compran el producto.  

---

### **3. Modelos Predictivos para Ventas**  
- **Aprendizaje supervisado** (ej.: *Random Forest, XGBoost*) para predecir:  
  - **Propensión de compra**: Qué clientes (o zonas) tienen mayor probabilidad de comprar el producto.  
  - **Demanda geográfica**: Dónde habrá mayor demanda en función de variables temporales (ej.: festividades locales).  
- **Variables de entrada**:  
  - Datos geográficos (latitud, longitud, distancia a tiendas).  
  - Datos socioeconómicos de la zona (nivel de ingresos, densidad poblacional).  
  - Historial de ventas en la región.  

---

### **4. Recomendaciones Geo-Personalizadas**  
- **Sistemas de recomendación** con filtrado colaborativo o basado en contenido:  
  - Sugerir productos populares en la zona (ej.: "En tu barrio, otros clientes compran X").  
  - Adaptar promociones según el perfil geográfico (ej.: descuentos en zonas con menor penetración).  

---

### **5. Optimización Logística y Ubicación de Puntos de Venta**  
- **Modelos de ubicación óptima**:  
  - Usar *algoritmos de optimización* (ej.: *p-median*) para decidir dónde abrir nuevas tiendas o colocar anuncios.  
  - Ejemplo: "Las zonas con radio de 5 km sin cobertura tienen un 20% de clientes potenciales sin atender".  

---

### **6. Campañas de Marketing Dirigido**  
- **Geo-targeting publicitario**:  
  - Entrenar modelos para identificar zonas donde campañas específicas (ej.: SMS, redes sociales) tendrán mayor ROI.  
  - Ejemplo: Anuncios en Facebook Ads para un radio de 10 km alrededor de tiendas con stock alto.  

---

### **7. Ejemplo Práctico**  
**Problema**: Una cadena de cafeterías quiere aumentar ventas en Ciudad de México.  
**Solución con ML**:  
1. Agrupa clientes por colonia usando *DBSCAN*.  
2. Entrena un modelo para predecir ventas según:  
   - Proximidad a estaciones de metro.  
   - Nivel socioeconómico (datos del INEGI).  
3. Descubre que las colonias *Roma* y *Condesa* tienen alta demanda los fines de semana.  
4. Lanza promociones "2x1" los sábados en esas zonas via WhatsApp.  

---

### **Beneficios**  
- **Reducción de costos**: Enfoque en zonas de alto impacto.  
- **Personalización**: Ofertas relevantes por ubicación.  
- **Escalabilidad**: Aplicable a múltiples regiones o productos.  

# ML

*  ## [Aprendizaje Supervizado](clase_7/aprendizaje-supervisado.md)
* ## [Aprendizaje no Supervisado](clase_7/aprendizaje-no-supervisado.md)

* ## [Paso a paso para un ML funcional](clase_7/paso-a-pas.md)


---

## 🚀 Implementación Práctica de Machine Learning

### 📋 **7.6 Implementación Práctica**

La implementación práctica de Machine Learning es el proceso que transforma la teoría en soluciones reales. Esta fase es crucial porque determina el éxito o fracaso de un proyecto de ML en el mundo real.

---

### 🧹 **1. Preparación de Datos**

La preparación de datos es el **paso más crítico** en cualquier proyecto de ML. Se estima que el 80% del tiempo en un proyecto de ML se dedica a la preparación y limpieza de datos.

#### **🎯 Filosofía de la Preparación de Datos**
- **Principio**: "Garbage in, garbage out" - Si los datos de entrada son de mala calidad, el modelo será inútil
- **Objetivo**: Transformar datos brutos en un formato limpio y estructurado
- **Enfoque**: Iterativo y sistemático

#### **📊 Ejemplo Práctico: Dataset de Ventas de Tienda**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Cargar datos (simulando un dataset real con problemas típicos)
np.random.seed(42)
n_samples = 1000

# Crear datos con problemas reales
data = {
    'edad': np.random.normal(35, 10, n_samples),
    'ingresos': np.random.lognormal(10, 0.5, n_samples),
    'genero': np.random.choice(['M', 'F', 'masculino', 'femenino', None], n_samples, p=[0.4, 0.3, 0.1, 0.1, 0.1]),
    'ciudad': np.random.choice(['Buenos Aires', 'Córdoba', 'Mendoza', 'BA', 'Cordoba', None], n_samples),
    'ventas_mes': np.random.gamma(2, 1000, n_samples),
    'satisfaccion': np.random.choice([1, 2, 3, 4, 5, None], n_samples, p=[0.1, 0.15, 0.2, 0.3, 0.2, 0.05])
}

df = pd.DataFrame(data)

# Agregar algunos outliers
df.loc[df.index[:50], 'ventas_mes'] *= 10
df.loc[df.index[50:60], 'edad'] = 200  # Valores imposibles

print("=== ESTADO INICIAL DE LOS DATOS ===")
print(f"Forma del dataset: {df.shape}")
print(f"\nValores faltantes por columna:")
print(df.isnull().sum())
print(f"\nTipos de datos:")
print(df.dtypes)
```

#### **🔧 1.1 Limpieza de Datos**

```python
# 1. Manejo de Valores Faltantes
print("\n=== LIMPIEZA DE DATOS ===")

# Estrategia para valores faltantes
def limpiar_valores_faltantes(df):
    df_limpio = df.copy()
    
    # Para variables numéricas: usar la mediana
    numeric_columns = ['edad', 'ingresos', 'ventas_mes']
    for col in numeric_columns:
        if df_limpio[col].isnull().sum() > 0:
            median_value = df_limpio[col].median()
            df_limpio[col].fillna(median_value, inplace=True)
            print(f"Imputado {col} con mediana: {median_value:.2f}")
    
    # Para variables categóricas: usar la moda
    categorical_columns = ['genero', 'ciudad', 'satisfaccion']
    for col in categorical_columns:
        if df_limpio[col].isnull().sum() > 0:
            mode_value = df_limpio[col].mode()[0]
            df_limpio[col].fillna(mode_value, inplace=True)
            print(f"Imputado {col} con moda: {mode_value}")
    
    return df_limpio

df_limpio = limpiar_valores_faltantes(df)

# 2. Corrección de Inconsistencias
def estandarizar_categorias(df):
    df_estandarizado = df.copy()
    
    # Estandarizar género
    df_estandarizado['genero'] = df_estandarizado['genero'].map({
        'M': 'Masculino', 'masculino': 'Masculino',
        'F': 'Femenino', 'femenino': 'Femenino'
    })
    
    # Estandarizar ciudades
    df_estandarizado['ciudad'] = df_estandarizado['ciudad'].map({
        'BA': 'Buenos Aires', 'Cordoba': 'Córdoba'
    })
    
    return df_estandarizado

df_estandarizado = estandarizar_categorias(df_limpio)

# 3. Manejo de Outliers
def detectar_y_tratar_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Outliers detectados en {column}: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
    
    # Estrategia: Capar los valores extremos
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

# Tratar outliers en ventas_mes y edad
df_final = detectar_y_tratar_outliers(df_estandarizado, 'ventas_mes')
df_final = detectar_y_tratar_outliers(df_final, 'edad')

print(f"\nDataset final: {df_final.shape}")
print(f"Valores faltantes restantes: {df_final.isnull().sum().sum()}")
```

#### **🔄 1.2 Transformación de Datos**

```python
# Transformación de variables
def transformar_datos(df):
    df_transformado = df.copy()
    
    # 1. Normalización/Estandarización
    scaler = StandardScaler()
    numeric_columns = ['edad', 'ingresos', 'ventas_mes']
    df_transformado[numeric_columns] = scaler.fit_transform(df_transformado[numeric_columns])
    
    # 2. Codificación de variables categóricas
    # One-hot encoding para ciudad
    ciudad_encoded = pd.get_dummies(df_transformado['ciudad'], prefix='ciudad')
    df_transformado = pd.concat([df_transformado, ciudad_encoded], axis=1)
    
    # Label encoding para género y satisfacción
    le_genero = LabelEncoder()
    df_transformado['genero_encoded'] = le_genero.fit_transform(df_transformado['genero'])
    
    # 3. Transformación logarítmica (si es necesario)
    # Para variables con sesgo positivo
    if df['ingresos'].skew() > 1:
        df_transformado['ingresos_log'] = np.log1p(df['ingresos'])
    
    return df_transformado

df_transformado = transformar_datos(df_final)

print("\n=== DATOS TRANSFORMADOS ===")
print(f"Forma final: {df_transformado.shape}")
print(f"Columnas: {list(df_transformado.columns)}")
```

---

### ✅ **2. Validación del Modelo**

La validación es esencial para asegurar que nuestro modelo funcionará en datos reales.

#### **📊 2.1 División del Conjunto de Datos**

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Preparar datos para modelado
# Asumimos que queremos predecir 'ventas_mes' basado en otras variables
X = df_transformado.drop(['ventas_mes', 'genero', 'ciudad'], axis=1)
y = df['ventas_mes']  # Usar valores originales, no normalizados

# División 70-20-10: Train-Validation-Test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.22, random_state=42)

print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Conjunto de validación: {X_val.shape[0]} muestras")
print(f"Conjunto de prueba: {X_test.shape[0]} muestras")
```

#### **🔄 2.2 Validación Cruzada**

```python
from sklearn.model_selection import cross_val_score, KFold

def evaluar_modelo_con_cv(modelo, X, y, cv=5):
    """Función para evaluar modelo con validación cruzada"""
    
    # Configurar validación cruzada
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Métricas a evaluar
    mse_scores = cross_val_score(modelo, X, y, cv=kf, scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(modelo, X, y, cv=kf, scoring='r2')
    
    return {
        'MSE_mean': -mse_scores.mean(),
        'MSE_std': mse_scores.std(),
        'R2_mean': r2_scores.mean(),
        'R2_std': r2_scores.std()
    }

# Entrenar modelo
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Evaluación con validación cruzada
cv_results = evaluar_modelo_con_cv(modelo, X_train, y_train)

print("\n=== RESULTADOS DE VALIDACIÓN CRUZADA ===")
print(f"MSE promedio: {cv_results['MSE_mean']:.2f} ± {cv_results['MSE_std']:.2f}")
print(f"R² promedio: {cv_results['R2_mean']:.3f} ± {cv_results['R2_std']:.3f}")
```

#### **📈 2.3 Métricas de Evaluación**

```python
def evaluar_modelo_completo(modelo, X_test, y_test):
    """Evaluación completa del modelo"""
    
    # Predicciones
    y_pred = modelo.predict(X_test)
    
    # Métricas de regresión
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Error absoluto medio
    mae = np.mean(np.abs(y_test - y_pred))
    
    # Error porcentual absoluto medio
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    resultados = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }
    
    return resultados, y_pred

# Evaluar en conjunto de prueba
resultados, predicciones = evaluar_modelo_completo(modelo, X_test, y_test)

print("\n=== MÉTRICAS EN CONJUNTO DE PRUEBA ===")
for metrica, valor in resultados.items():
    print(f"{metrica}: {valor:.3f}")

# Visualización de resultados
plt.figure(figsize=(15, 5))

# 1. Predicciones vs Valores Reales
plt.subplot(1, 3, 1)
plt.scatter(y_test, predicciones, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title(f'Predicciones vs Reales\nR² = {resultados["R²"]:.3f}')

# 2. Residuos
plt.subplot(1, 3, 2)
residuos = y_test - predicciones
plt.scatter(predicciones, residuos, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('Análisis de Residuos')

# 3. Distribución de errores
plt.subplot(1, 3, 3)
plt.hist(residuos, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Distribución de Errores')

plt.tight_layout()
plt.show()
```

#### **⚙️ 2.4 Ajuste de Hiperparámetros**

```python
from sklearn.model_selection import GridSearchCV

def ajustar_hiperparametros(modelo, X_train, y_train, param_grid, cv=3):
    """Ajuste de hiperparámetros con GridSearch"""
    
    grid_search = GridSearchCV(
        modelo, 
        param_grid, 
        cv=cv, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search

# Definir grid de parámetros para Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("\n=== AJUSTE DE HIPERPARÁMETROS ===")
print("Buscando mejores parámetros...")

# Ajustar hiperparámetros
grid_search = ajustar_hiperparametros(
    RandomForestRegressor(random_state=42), 
    X_train, y_train, 
    param_grid
)

print(f"\nMejores parámetros: {grid_search.best_params_}")
print(f"Mejor score (negativo MSE): {grid_search.best_score_:.2f}")

# Evaluar modelo optimizado
modelo_optimizado = grid_search.best_estimator_
resultados_optimizado, predicciones_optimizado = evaluar_modelo_completo(modelo_optimizado, X_test, y_test)

print("\n=== COMPARACIÓN DE MODELOS ===")
print("Modelo Original vs Optimizado:")
print(f"R² Original: {resultados['R²']:.3f}")
print(f"R² Optimizado: {resultados_optimizado['R²']:.3f}")
print(f"Mejora: {(resultados_optimizado['R²'] - resultados['R²'])*100:.1f}%")
```

---

### 🚀 **3. Despliegue del Modelo**

El despliegue es donde el modelo pasa del laboratorio al mundo real.

#### **📦 3.1 Preparación para Producción**

```python
import joblib
import json

def guardar_modelo_completo(modelo, scaler, columnas, metadatos, ruta_base='modelo_ml'):
    """Guardar modelo y metadatos para producción"""
    
    # 1. Guardar modelo
    joblib.dump(modelo, f'{ruta_base}_modelo.pkl')
    
    # 2. Guardar scaler
    joblib.dump(scaler, f'{ruta_base}_scaler.pkl')
    
    # 3. Guardar información de columnas
    with open(f'{ruta_base}_columnas.json', 'w') as f:
        json.dump(columnas, f)
    
    # 4. Guardar metadatos
    with open(f'{ruta_base}_metadatos.json', 'w') as f:
        json.dump(metadatos, f)
    
    print(f"Modelo guardado en: {ruta_base}_*")

# Preparar metadatos
metadatos = {
    'version': '1.0',
    'fecha_entrenamiento': '2024-01-15',
    'algoritmo': 'RandomForestRegressor',
    'metricas_entrenamiento': resultados_optimizado,
    'caracteristicas': list(X.columns),
    'descripcion': 'Modelo para predecir ventas mensuales basado en perfil del cliente'
}

# Guardar modelo
guardar_modelo_completo(
    modelo_optimizado, 
    scaler, 
    list(X.columns), 
    metadatos
)
```

#### **🔧 3.2 Función de Predicción**

```python
class PredictorVentas:
    """Clase para hacer predicciones en producción"""
    
    def __init__(self, ruta_modelo='modelo_ml'):
        self.modelo = joblib.load(f'{ruta_modelo}_modelo.pkl')
        self.scaler = joblib.load(f'{ruta_modelo}_scaler.pkl')
        
        with open(f'{ruta_modelo}_columnas.json', 'r') as f:
            self.columnas_esperadas = json.load(f)
        
        with open(f'{ruta_modelo}_metadatos.json', 'r') as f:
            self.metadatos = json.load(f)
    
    def preprocesar_datos(self, datos_cliente):
        """Preprocesar datos de un cliente individual"""
        
        # Convertir a DataFrame
        df_cliente = pd.DataFrame([datos_cliente])
        
        # Aplicar mismas transformaciones que en entrenamiento
        df_cliente['genero'] = df_cliente['genero'].map({
            'M': 'Masculino', 'masculino': 'Masculino',
            'F': 'Femenino', 'femenino': 'Femenino'
        })
        
        df_cliente['ciudad'] = df_cliente['ciudad'].map({
            'BA': 'Buenos Aires', 'Cordoba': 'Córdoba'
        })
        
        # One-hot encoding para ciudad
        ciudad_encoded = pd.get_dummies(df_cliente['ciudad'], prefix='ciudad')
        df_cliente = pd.concat([df_cliente, ciudad_encoded], axis=1)
        
        # Label encoding para género
        le_genero = LabelEncoder()
        df_cliente['genero_encoded'] = le_genero.fit_transform(df_cliente['genero'])
        
        # Normalizar variables numéricas
        numeric_columns = ['edad', 'ingresos']
        df_cliente[numeric_columns] = self.scaler.transform(df_cliente[numeric_columns])
        
        # Asegurar que todas las columnas esperadas estén presentes
        for col in self.columnas_esperadas:
            if col not in df_cliente.columns:
                df_cliente[col] = 0
        
        # Reordenar columnas
        df_cliente = df_cliente[self.columnas_esperadas]
        
        return df_cliente
    
    def predecir(self, datos_cliente):
        """Hacer predicción para un cliente"""
        
        # Preprocesar datos
        X_cliente = self.preprocesar_datos(datos_cliente)
        
        # Hacer predicción
        prediccion = self.modelo.predict(X_cliente)[0]
        
        # Calcular intervalo de confianza (aproximado)
        # Nota: Para un intervalo real necesitarías más información del modelo
        std_error = 0.1 * prediccion  # Aproximación simple
        intervalo = (prediccion - 1.96*std_error, prediccion + 1.96*std_error)
        
        return {
            'prediccion': prediccion,
            'intervalo_confianza_95': intervalo,
            'modelo_version': self.metadatos['version']
        }

# Ejemplo de uso en producción
print("\n=== EJEMPLO DE PREDICCIÓN EN PRODUCCIÓN ===")

# Simular datos de un nuevo cliente
nuevo_cliente = {
    'edad': 28,
    'ingresos': 45000,
    'genero': 'F',
    'ciudad': 'Buenos Aires',
    'satisfaccion': 4
}

# Crear predictor
predictor = PredictorVentas()

# Hacer predicción
resultado = predictor.predecir(nuevo_cliente)

print(f"Cliente: {nuevo_cliente}")
print(f"Predicción de ventas: ${resultado['prediccion']:.2f}")
print(f"Intervalo 95%: ${resultado['intervalo_confianza_95'][0]:.2f} - ${resultado['intervalo_confianza_95'][1]:.2f}")
```

#### **📊 3.3 Monitoreo del Modelo**

```python
def monitorear_rendimiento_modelo(y_real, y_pred, umbral_drift=0.1):
    """Función para monitorear el rendimiento del modelo en producción"""
    
    # Calcular métricas actuales
    r2_actual = r2_score(y_real, y_pred)
    rmse_actual = np.sqrt(mean_squared_error(y_real, y_pred))
    
    # Detectar drift (cambio en la distribución)
    # Comparar con métricas de referencia (del entrenamiento)
    r2_referencia = 0.85  # Valor de referencia del entrenamiento
    drift_detectado = abs(r2_actual - r2_referencia) > umbral_drift
    
    # Generar alerta si hay drift
    if drift_detectado:
        print(f"⚠️ ALERTA: Drift detectado en el modelo!")
        print(f"R² actual: {r2_actual:.3f}")
        print(f"R² referencia: {r2_referencia:.3f}")
        print(f"Diferencia: {abs(r2_actual - r2_referencia):.3f}")
    
    return {
        'r2_actual': r2_actual,
        'rmse_actual': rmse_actual,
        'drift_detectado': drift_detectado,
        'necesita_retrenamiento': drift_detectado
    }

# Ejemplo de monitoreo
print("\n=== MONITOREO DEL MODELO ===")
estado_modelo = monitorear_rendimiento_modelo(y_test, predicciones_optimizado)

print(f"Estado del modelo:")
for key, value in estado_modelo.items():
    print(f"  {key}: {value}")
```

---

### 🎯 **Resumen de la Implementación Práctica**

#### **✅ Checklist de Implementación Exitosa**

1. **Preparación de Datos** ✅
   - [ ] Limpieza de valores faltantes
   - [ ] Estandarización de categorías
   - [ ] Manejo de outliers
   - [ ] Normalización/estandarización
   - [ ] Codificación de variables categóricas

2. **Validación del Modelo** ✅
   - [ ] División adecuada de datos
   - [ ] Validación cruzada
   - [ ] Métricas de evaluación apropiadas
   - [ ] Ajuste de hiperparámetros
   - [ ] Análisis de residuos

3. **Despliegue** ✅
   - [ ] Serialización del modelo
   - [ ] Función de predicción robusta
   - [ ] Sistema de monitoreo
   - [ ] Documentación de metadatos

#### **🚨 Errores Comunes a Evitar**

1. **Data Leakage**: No usar información del futuro para predecir el pasado
2. **Overfitting**: Validar siempre en datos no vistos
3. **Falta de monitoreo**: Los modelos se degradan con el tiempo
4. **Ignorar el contexto de negocio**: Las métricas técnicas no siempre reflejan el valor real

#### **📈 Próximos Pasos**

1. **A/B Testing**: Comparar el modelo contra métodos actuales
2. **Feedback Loop**: Incorporar feedback de usuarios
3. **Retrenamiento Automático**: Sistema para actualizar el modelo periódicamente
4. **Escalabilidad**: Preparar para mayor volumen de datos

---

### Apartado especial para lo que es Feature Selection

### 🧹 1. Métodos de Filtro (*Filter Methods*)

Piensa en estos como un primer filtro rápido.

* **Cómo funcionan:** Evalúan cada característica de forma individual usando pruebas estadísticas que miden su relación con la variable objetivo (la etiqueta).
* **Velocidad:** Muy rápidos, ya que no requieren entrenar modelos.
* **Ejemplos:**

  * **Prueba de Chi-cuadrado** (para variables categóricas)
  * **Prueba ANOVA**
  * **Información mutua**
  * **Coeficiente de correlación**

📌 **Ventajas:** Rápidos y no dependen del modelo
📌 **Desventajas:** No consideran las interacciones entre características

---

### 🧪 2. Métodos Envolventes (*Wrapper Methods*)

Estos son como probar diferentes combinaciones de ropa para ver cuál te queda mejor.

* **Cómo funcionan:** Prueban múltiples subconjuntos de características entrenando modelos, y eligen el subconjunto que da el mejor rendimiento.
* **Velocidad:** Lentos, porque entrenan muchos modelos.
* **Ejemplos:**

  * **Eliminación recursiva de características (RFE)**
  * **Selección hacia adelante**
  * **Eliminación hacia atrás**

📌 **Ventajas:** Tienen en cuenta interacciones entre características
📌 **Desventajas:** Muy costosos en tiempo y recursos computacionales

---

### ⚙️ 3. Métodos Embebidos (*Embedded Methods*)

Estos seleccionan características **mientras entrenan** el modelo.

* **Cómo funcionan:** La selección ocurre como parte del proceso de entrenamiento.
* **Ejemplos:**

  * **Lasso (regularización L1)** – reduce coeficientes a cero
  * **Árboles de decisión / Bosques aleatorios** – calculan importancia de cada variable
  * **Elastic Net** – combina L1 y L2

📌 **Ventajas:** Más eficientes y consideran interacciones
📌 **Desventajas:** Dependientes del modelo utilizado

---

### Tabla Resumen:

| Tipo de método | ¿Usa modelo? | ¿Considera interacciones? | Velocidad | Ejemplos                      |
| -------------- | ------------ | ------------------------- | --------- | ----------------------------- |
| Filtro         | ❌ No         | ❌ No                      | 🚀 Rápido | Chi-cuadrado, correlación     |
| Envolvente     | ✅ Sí         | ✅ Sí                      | 🐢 Lento  | RFE, selección hacia adelante |
| Embebido       | ✅ Sí         | ✅ Sí                      | ⚡ Medio   | Lasso, árboles de decisión    |


## Recomendaciones para aprender:
¡Claro! Vamos a ver cada uno de estos métodos de selección de características supervisadas con **ejemplos, ventajas y desventajas**, relacionándolos con los tipos que ya vimos: filtro, envolvente o embebido.

---

### 📉 1. **Variance Threshold**

🔹 **Tipo:** Filtro
🔹 **Qué hace:** Elimina características con **baja varianza**, es decir, que no cambian mucho entre ejemplos (por ejemplo, una columna donde casi todos los valores son iguales).

#### ✅ Ventajas:

* Muy simple y rápido de aplicar
* No necesita etiquetas (también se puede usar en problemas no supervisados)

#### ❌ Desventajas:

* No considera la relación con la variable objetivo
* Puede eliminar características útiles si tienen poca varianza pero alta relevancia

#### 🧪 Ejemplo en Python:

```python
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_reduced = selector.fit_transform(X)
```

---

### ⭐ 2. **SelectKBest**

🔹 **Tipo:** Filtro
🔹 **Qué hace:** Selecciona las **k mejores características** según una métrica estadística que mide la relación con la variable objetivo (como ANOVA, chi-cuadrado, etc.).

#### ✅ Ventajas:

* Fácil de interpretar y aplicar
* Rápido y eficiente
* Permite usar distintas métricas

#### ❌ Desventajas:

* No considera interacciones entre variables
* Necesita que tú elijas el valor de *k* (el número de características a conservar)

#### 🧪 Ejemplo en Python:

```python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=10)
X_kbest = selector.fit_transform(X, y)
```

---

### 🔄 3. **RFE (Recursive Feature Elimination)**

🔹 **Tipo:** Envolvente
🔹 **Qué hace:** Usa un modelo (como una regresión o un SVM) para eliminar **recursivamente** las características menos importantes hasta quedarse con las más relevantes.

#### ✅ Ventajas:

* Considera interacciones entre variables
* Da muy buenos resultados si se elige bien el modelo base

#### ❌ Desventajas:

* Lento, especialmente con muchos datos o características
* Depende fuertemente del modelo usado

#### 🧪 Ejemplo en Python:

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(estimator=model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
```

---

### 🌳 4. **Boruta**

🔹 **Tipo:** Envolvente (basado en árboles, como Random Forest)
🔹 **Qué hace:** Crea versiones "aleatorias" de las características y las compara con las reales. Solo conserva las que son mejores que las versiones aleatorias (shadow features).

#### ✅ Ventajas:

* Muy robusto y preciso
* Tiene en cuenta interacciones y relaciones no lineales
* Funciona bien con datos complejos

#### ❌ Desventajas:

* Bastante lento (usa muchos modelos de Random Forest)
* No está en `scikit-learn` directamente (hay que usar una librería aparte como `boruta_py`)

#### 🧪 Ejemplo con `boruta_py`:

```python
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
boruta_selector = BorutaPy(estimator=model, n_estimators='auto', random_state=42)
boruta_selector.fit(X.values, y.values)
```

---

### 🔍 Resumen Comparativo

| Método             | Tipo       | Interacción | Velocidad    | Modelo requerido | Ventaja principal                    |
| ------------------ | ---------- | ----------- | ------------ | ---------------- | ------------------------------------ |
| Variance Threshold | Filtro     | ❌ No        | 🚀 Rápido    | ❌ No             | Muy simple y rápido                  |
| SelectKBest        | Filtro     | ❌ No        | 🚀 Rápido    | ❌ No             | Métricas estadísticas supervisadas   |
| RFE                | Envolvente | ✅ Sí        | 🐢 Lento     | ✅ Sí             | Preciso si el modelo es adecuado     |
| Boruta             | Envolvente | ✅ Sí        | 🐌 Muy lento | ✅ Sí             | Robusto frente a ruido y redundancia |

