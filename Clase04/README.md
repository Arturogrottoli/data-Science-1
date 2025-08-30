
# Clase: Manejo de Datos Nulos y Series Temporales en Pandas

## **Repaso de Clases Anteriores**

### 🐍 Ejemplo Python - Condicionales para Ciencia de Datos
```python
# Clasificación de datos según rangos
def clasificar_edad(edad):
    if edad < 18:
        return "Menor de edad"
    elif 18 <= edad <= 65:
        return "Adulto"
    else:
        return "Adulto mayor"

# Aplicar a una lista de edades
edades = [15, 25, 70, 30, 12]
clasificaciones = [clasificar_edad(edad) for edad in edades]
print(clasificaciones)  # ['Menor de edad', 'Adulto', 'Adulto mayor', 'Adulto', 'Menor de edad']
```

### 🔢 Ejemplo NumPy - Operaciones Vectorizadas
```python
import numpy as np

# Crear arrays y operaciones vectorizadas
temperaturas = np.array([22, 25, 18, 30, 15])
humedad = np.array([60, 70, 45, 80, 35])

# Normalización z-score
temperaturas_norm = (temperaturas - np.mean(temperaturas)) / np.std(temperaturas)
print(f"Temperaturas normalizadas: {temperaturas_norm}")

# Filtrado condicional
dias_calidos = temperaturas > 25
print(f"Días calurosos: {dias_calidos}")  # [False False False True False]
```

### 📊 Ejemplo Pandas - Análisis Básico
```python
import pandas as pd

# Crear DataFrame de ventas
ventas_data = {
    'producto': ['A', 'B', 'A', 'C', 'B'],
    'cantidad': [10, 5, 15, 8, 12],
    'precio': [100, 200, 100, 150, 200]
}
df_ventas = pd.DataFrame(ventas_data)

# Análisis por producto
resumen = df_ventas.groupby('producto').agg({
    'cantidad': 'sum',
    'precio': 'mean'
}).round(2)
print(resumen)
```

---

## **Creación de Series y DataFrames**

### 1.1 Creación de Series
```python
import pandas as pd

# Desde una lista
serie_lista = pd.Series([1, 2, 3, 4, 5])
print(serie_lista)

# Desde un diccionario
serie_dict = pd.Series({'A': 10, 'B': 20, 'C': 30})
print(serie_dict)

# Con índice personalizado
serie_custom = pd.Series([100, 200, 300], index=['enero', 'febrero', 'marzo'])
print(serie_custom)
```

### 1.2 Creación de DataFrames
```python
# Desde diccionario
data_dict = {
    'nombre': ['Ana', 'Juan', 'María'],
    'edad': [25, 30, 28],
    'ciudad': ['Madrid', 'Barcelona', 'Valencia']
}
df_dict = pd.DataFrame(data_dict)
print(df_dict)

# Desde lista de listas
data_lista = [
    ['Ana', 25, 'Madrid'],
    ['Juan', 30, 'Barcelona'],
    ['María', 28, 'Valencia']
]
df_lista = pd.DataFrame(data_lista, columns=['nombre', 'edad', 'ciudad'])
print(df_lista)
```

---

## **Operaciones Básicas en DataFrames**

### 2.1 Selección de Datos

**Selección de Columnas:**
```python
# Una columna
columna_simple = df['nombre_columna']

# Múltiples columnas
columnas_multiple = df[['columna1', 'columna2', 'columna3']]
```

**Selección de Filas:**
```python
# Por etiqueta (loc)
fila_etiqueta = df.loc['indice_etiqueta']

# Por posición (iloc)
fila_posicion = df.iloc[0]  # Primera fila
filas_rango = df.iloc[0:3]  # Filas 0, 1, 2
```

### 2.2 Filtrado de Datos

**Filtrado Simple:**
```python
# Condición única
filtro_simple = df[df['precio'] > 100]

# Múltiples condiciones
filtro_multiple = df[(df['precio'] > 100) & (df['categoria'] == 'Electrónicos')]
```

**Filtrado con Operadores Lógicos:**
```python
# AND (&)
filtro_and = df[(df['edad'] > 25) & (df['edad'] < 50)]

# OR (|)
filtro_or = df[(df['ciudad'] == 'Madrid') | (df['ciudad'] == 'Barcelona')]

# NOT (~)
filtro_not = df[~(df['categoria'] == 'Descartado')]
```

### 2.3 Agregación de Datos

**Agregación Simple:**
```python
# Estadísticas básicas
suma_total = df['cantidad'].sum()
promedio = df['precio'].mean()
conteo = df['producto'].count()
maximo = df['ventas'].max()
minimo = df['ventas'].min()
```

**Agregación Agrupada:**
```python
# Agrupar por una columna
ventas_por_categoria = df.groupby('categoria')['ventas'].sum()

# Agrupar por múltiples columnas
ventas_por_cat_y_region = df.groupby(['categoria', 'region'])['ventas'].sum()

# Múltiples funciones de agregación
resumen_completo = df.groupby('categoria').agg({
    'ventas': ['sum', 'mean', 'count'],
    'precio': ['mean', 'std'],
    'cantidad': 'sum'
})
```

### 2.4 Ejemplos Prácticos
```python
# Dataset de ejemplo
ventas_data = {
    'producto': ['Laptop', 'Mouse', 'Laptop', 'Teclado', 'Mouse'],
    'categoria': ['Electrónicos', 'Accesorios', 'Electrónicos', 'Accesorios', 'Accesorios'],
    'precio': [1200, 25, 1200, 80, 25],
    'cantidad': [5, 20, 3, 15, 30],
    'region': ['Norte', 'Sur', 'Norte', 'Este', 'Oeste']
}
df_ventas = pd.DataFrame(ventas_data)

# Ejemplo 1: Seleccionar productos electrónicos
electronicos = df_ventas[df_ventas['categoria'] == 'Electrónicos']

# Ejemplo 2: Calcular ventas totales por categoría
ventas_categoria = df_ventas.groupby('categoria')['cantidad'].sum()

# Ejemplo 3: Productos con precio mayor a 100
productos_caros = df_ventas[df_ventas['precio'] > 100]

# Ejemplo 4: Resumen estadístico por región
resumen_region = df_ventas.groupby('region').agg({
    'precio': 'mean',
    'cantidad': 'sum',
    'producto': 'count'
}).round(2)
```

---

## **Manejo Avanzado de Datos Ausentes**

### 3.1 Identificación de Datos Ausentes
```python
# Detectar valores ausentes
df.isnull()  # DataFrame booleano indicando valores nulos
df.isnull().sum()  # Conteo de valores nulos por columna

# Información detallada
print(f"Total de valores nulos: {df.isnull().sum().sum()}")
print(f"Porcentaje de valores nulos: {(df.isnull().sum().sum() / df.size) * 100:.2f}%")
```

### 3.2 Eliminación de Datos Ausentes
```python
# Eliminar filas con valores nulos
df_sin_nulos = df.dropna()  # Elimina filas con al menos un valor nulo

# Eliminar columnas con valores nulos
df_sin_columnas_nulas = df.dropna(axis=1)

# Eliminar solo si todos los valores son nulos
df_limpiado = df.dropna(how='all')  # Solo filas completamente vacías

# Eliminar si hay al menos 2 valores nulos
df_parcial = df.dropna(thresh=len(df.columns)-2)
```

### 3.3 Imputación Básica con Pandas
```python
# Rellenar con valor constante
df.fillna(0, inplace=True)

# Imputar con estadísticas por columna
df['edad'].fillna(df['edad'].mean(), inplace=True)      # Media
df['salario'].fillna(df['salario'].median(), inplace=True)  # Mediana
df['categoria'].fillna(df['categoria'].mode()[0], inplace=True)  # Moda

# Forward fill y backward fill
df['precio'].fillna(method='ffill', inplace=True)  # Llenar con valor anterior
df['stock'].fillna(method='bfill', inplace=True)   # Llenar con valor siguiente
```

### 3.4 Imputación Avanzada con Scikit-Learn
```python
from sklearn.impute import SimpleImputer, KNNImputer

# SimpleImputer con diferentes estrategias
imputer_media = SimpleImputer(strategy='mean')
imputer_mediana = SimpleImputer(strategy='median')
imputer_moda = SimpleImputer(strategy='most_frequent')
imputer_constante = SimpleImputer(strategy='constant', fill_value=0)

# Aplicar a columnas numéricas
columnas_numericas = df.select_dtypes(include=[np.number]).columns
df[columnas_numericas] = imputer_media.fit_transform(df[columnas_numericas])

# KNN Imputer (más sofisticado)
knn_imputer = KNNImputer(n_neighbors=5)
df_imputado_knn = pd.DataFrame(
    knn_imputer.fit_transform(df),
    columns=df.columns,
    index=df.index
)
```

### 3.5 Ejemplo Práctico Completo
```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Dataset con valores ausentes
data_ejemplo = {
    'id': [1, 2, 3, 4, 5],
    'nombre': ['Ana', 'Juan', None, 'María', 'Carlos'],
    'edad': [25, None, 30, 28, None],
    'salario': [50000, 60000, None, 55000, 70000],
    'departamento': ['IT', 'HR', 'IT', None, 'IT']
}
df_ejemplo = pd.DataFrame(data_ejemplo)

print("Dataset original:")
print(df_ejemplo)
print("\nValores nulos por columna:")
print(df_ejemplo.isnull().sum())

# Estrategia de imputación
# 1. Texto: moda
# 2. Números: media
# 3. Categorías: moda

# Imputar texto
df_ejemplo['nombre'].fillna(df_ejemplo['nombre'].mode()[0], inplace=True)

# Imputar números
df_ejemplo['edad'].fillna(df_ejemplo['edad'].mean(), inplace=True)
df_ejemplo['salario'].fillna(df_ejemplo['salario'].mean(), inplace=True)

# Imputar categorías
df_ejemplo['departamento'].fillna(df_ejemplo['departamento'].mode()[0], inplace=True)

print("\nDataset después de imputación:")
print(df_ejemplo)
```

---

## **Manipulación de Strings en Pandas**

### 4.1 Operaciones Básicas de Strings
```python
# Acceder a métodos de string
df['nombre'].str.upper()  # Convertir a mayúsculas
df['email'].str.lower()   # Convertir a minúsculas
df['texto'].str.strip()   # Eliminar espacios en blanco

# Longitud de strings
df['nombre'].str.len()    # Longitud de cada string

# Concatenación
df['nombre_completo'] = df['nombre'].str.cat(df['apellido'], sep=' ')
```

### 4.2 Búsqueda y Filtrado
```python
# Contiene un patrón
df[df['producto'].str.contains('laptop', case=False)]

# Comienza con
df[df['email'].str.startswith('admin')]

# Termina con
df[df['archivo'].str.endswith('.csv')]

# Coincidencia exacta
df[df['categoria'].str.match('^Electrónicos$')]
```

### 4.3 Extracción y Reemplazo
```python
# Extraer parte del string
df['codigo_pais'] = df['telefono'].str.extract(r'\+(\d+)')
df['dominio'] = df['email'].str.extract(r'@(.+)')

# Reemplazar patrones
df['telefono_limpio'] = df['telefono'].str.replace(r'[^\d]', '', regex=True)
df['precio_limpio'] = df['precio'].str.replace('$', '').str.replace(',', '')

# Split y join
df['palabras'] = df['descripcion'].str.split()
df['primera_palabra'] = df['descripcion'].str.split().str[0]
```

### 4.4 Validación y Limpieza
```python
# Validar formato de email
def es_email_valido(email):
    import re
    patron = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(patron, str(email)))

df['email_valido'] = df['email'].apply(es_email_valido)

# Limpiar nombres
df['nombre_limpio'] = (df['nombre']
                      .str.strip()
                      .str.title()
                      .str.replace(r'\s+', ' ', regex=True))
```

### 4.5 Ejemplo Práctico de Limpieza de Datos
```python
# Dataset con datos textuales sucios
datos_sucios = {
    'nombre': ['  juan pérez  ', 'MARÍA GARCÍA', 'carlos lopez'],
    'email': ['juan@email.com', 'maria@email.com', 'carlos@email.com'],
    'telefono': ['+34 123-456-789', '+34 987-654-321', '+34 555-123-456'],
    'precio': ['$1,234.56', '$2,345.67', '$3,456.78']
}
df_sucio = pd.DataFrame(datos_sucios)

print("Datos originales:")
print(df_sucio)

# Limpieza completa
df_limpio = df_sucio.copy()

# Limpiar nombres
df_limpio['nombre'] = (df_limpio['nombre']
                      .str.strip()
                      .str.title()
                      .str.replace(r'\s+', ' ', regex=True))

# Limpiar teléfonos
df_limpio['telefono_limpio'] = df_limpio['telefono'].str.replace(r'[^\d]', '', regex=True)

# Limpiar precios
df_limpio['precio_numerico'] = (df_limpio['precio']
                               .str.replace('$', '')
                               .str.replace(',', '')
                               .astype(float))

# Extraer dominio de email
df_limpio['dominio_email'] = df_limpio['email'].str.extract(r'@(.+)')

print("\nDatos limpios:")
print(df_limpio)
```

---

## **Objetivos de Aprendizaje**
1. Identificar y clasificar tipos de datos nulos
2. Aplicar técnicas básicas y avanzadas de imputación
3. Manipular series temporales con Pandas
4. Implementar estrategias para manejar valores faltantes en series de tiempo

### [DIAPOSITIVAS](https://docs.google.com/presentation/d/1BDjUNhpNr1TD8qRBV6XkHkUUbZhho8Ks5wzouYfjLRY/edit?slide=id.g2f3430c3b8e_0_258#slide=id.g2f3430c3b8e_0_258)


## **Bloque 1: Manejo de Datos Nulos**

### 1.1 Teoría Fundamental
**¿Qué son los datos nulos?**  
Valores ausentes representados como `NaN` (Not a Number) o `None` en Python. Surgen por:
- Errores en captura de datos
- Campos opcionales no completados
- Fallas en sensores/dispositivos [1][3]

**Impacto en el análisis:**
- Reducción de potencia estadística
- Sesgos en modelos predictivos
- Errores en cálculos matemáticos

### 1.2 Técnicas de Imputación

**Básicas:**
```python
# Eliminación
df.dropna(axis=0)  # Filas
df.dropna(axis=1)  # Columnas [1]

# Relleno con constante
df.fillna(0)

# Medidas de tendencia central
df.fillna(df.mean())  # Media
df.fillna(df.median())  # Mediana [4]
```

**Avanzadas (usando scikit-learn):**
```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=3)
df_imputado = imputer.fit_transform(df) [2][4]
```

**Interpolación temporal:**
```python
df['serie'].interpolate(method='time') [7]
```

### 1.3 Ejercicio Práctico
```python
# Dataset con valores faltantes
data = {'A': [1, 2, None, 4], 'B': [None, 2, 3, 4]}
df = pd.DataFrame(data)

# Tarea: Implementar 3 técnicas diferentes
# y comparar resultados
```

---

## **Bloque 2: Series Temporales en Pandas**

### 2.1 Fundamentos
**Características clave:**
- Índice datetime como eje principal
- Frecuencia consistente (diaria, mensual, etc.)
- Operaciones especializadas (resampling, ventanas móviles)

**Creación básica:**
```python
# Conversión a datetime
df['fecha'] = pd.to_datetime(df['fecha'], format='%d-%m-%Y')
df = df.set_index('fecha') [5][6]
```

### 2.2 Técnicas Avanzadas

**Resampling:**
```python
# Mensual a diario con forward fill
daily_df = df.resample('D').asfreq().ffill() [7]

# Promedio móvil 7 días
df.rolling(window='7D').mean()
```

**Manejo de faltantes temporales:**
```python
# Interpolación cuadrática
df.interpolate(method='quadratic')

# Llenado con último valor válido
df.ffill() [1][5]
```

### 2.3 Ejercicio Integrador
```python
# Dataset de ventas con huecos temporales
ventas = pd.Series([100, None, 150, None, 200], 
                  index=pd.date_range('2024-01-01', periods=5))

# Tarea: 
# 1. Completar valores faltantes con interpolación
# 2. Calcular promedio móvil de 2 días
# 3. Convertir a frecuencia horaria con forward fill
```

---

## **Actividades Sugeridas**
1. **Análisis comparativo:** Probar diferentes métodos de imputación en un dataset real y evaluar impacto en estadísticas descriptivas
2. **Competición de imputación:** En grupos, desarrollar estrategias para dataset con 30% valores faltantes y comparar resultados
3. **Proyecto de series temporales:** Analizar datos climáticos con patrones estacionales y valores faltantes

---

# Graficos




###  Tabla: Gráficos recomendados para análisis exploratorio

| Gráfico         | Uso principal                                                       | ¿Cuándo es excelente?                                                            | Librería en Python      |
| --------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ----------------------- |
| **Boxplot**     | Visualizar distribución, mediana, rangos y outliers                 | Para comparar la dispersión de una variable numérica entre categorías            | `seaborn`, `matplotlib` |
| **Scatterplot** | Mostrar la relación entre dos variables numéricas                   | Para detectar patrones, tendencias, relaciones lineales o no lineales y outliers | `seaborn`, `matplotlib` |
| **Lineplot**    | Mostrar la evolución de una variable numérica a lo largo del tiempo | Para observar tendencias temporales o secuenciales                               | `seaborn`, `matplotlib` |
| **Barplot**     | Comparar valores agregados (suma, media, etc.) entre categorías     | Ideal para comparar cantidades entre grupos o categorías                         | `seaborn`, `matplotlib` |
| **Histograma**  | Visualizar la distribución de frecuencia de una variable numérica   | Para ver cómo se distribuyen los valores, si hay sesgo, bimodalidad, etc.        | `matplotlib`, `seaborn` |

---

### ✅ Explicación rápida:

#### 1. **Boxplot**

* **Útil para:** ver la dispersión, detectar valores extremos.
* **Ideal para:** comparar por categoría (ej.: ventas por tipo de producto).

#### 2. **Scatterplot**

* **Útil para:** descubrir relaciones entre dos variables numéricas.
* **Ideal para:** detectar correlaciones, clústeres o anomalías.

#### 3. **Lineplot**

* **Útil para:** análisis de series de tiempo o progresión.
* **Ideal para:** ver tendencias de ventas, precios, tráfico, etc.

#### 4. **Barplot**

* **Útil para:** comparar cantidades resumidas.
* **Ideal para:** mostrar promedios o totales por grupo.

#### 5. **Histograma**

* **Útil para:** ver la distribución de valores.
* **Ideal para:** analizar simetría, sesgo o agrupaciones naturales.


## **Recursos Adicionales**
1. Documentación oficial de Pandas: [Manejo de datos faltantes](https://pandas.pydata.org/docs/user_guide/missing_data.html) [1]
2. Libro: "Python for Data Analysis" de Wes McKinney (Cap. 10)
3. Tutorial avanzado: [DataCamp: Missing Data Techniques](https://www.datacamp.com/tutorial/techniques-to-handle-missing-data-values) [4]
4. Dataset práctico: [Air Quality Time Series](https://archive.ics.uci.edu/ml/datasets/Air+Quality)



Citations:
- [1] https://docs.kanaries.net/es/topics/Pandas/pandas-where
- [2] https://fastercapital.com/es/tema/manejo-de-datos-faltantes-y-valores-at%C3%ADpicos.html
- [3] https://www.growupcr.com/post/valores-nulos-python
- [4] https://www.datacamp.com/es/tutorial/techniques-to-handle-missing-data-values
- [5] https://www.codetodevs.com/como-utilizar-pandas-para-procesamiento-series-temporales/
- [6] https://www.codigofacilito.com/articulos/fechas-python
- [7] https://labex.io/es/tutorials/pandas-pandas-dataframe-asfreq-method-68584
- [8] https://www.youtube.com/watch?v=XKyX1ag4tnE
- [9] https://www.datacamp.com/tutorial/pandas-resample-asfreq
- [10] https://es.linkedin.com/advice/0/how-can-you-handle-missing-data-pandas-effectively-lilzc?lang=es
- [11] https://es.linkedin.com/advice/0/what-best-practices-using-groupby-pandas-time-series-xgbjf?lang=es
- [12] https://www.youtube.com/watch?v=ujODwjG3lv4
- [13] https://www.analyticslane.com/2021/08/12/pandas-contar-los-valores-nulos-en-dataframe/
- [14] https://certidevs.com/tutorial-pandas-operaciones-con-fechas
- [15] https://es.linkedin.com/pulse/manejo-de-datos-nulos-en-python-luis-felipe-castro-calder%C3%B3n
- [16] https://www.codetodevs.com/como-trabajar-con-datos-vacios-en-pandas/
- [17] https://www.freecodecamp.org/espanol/news/limpieza-de-datos-en-pandas-explicado-con-ejemplos/
- [18] https://interactivechaos.com/es/manual/tutorial-de-pandas/el-metodo-fillna
- [19] https://interactivechaos.com/es/manual/tutorial-de-pandas/el-metodo-dropna
- [20] https://www.freecodecamp.org/espanol/news/introduccion-a-las-series-de-pandas-con-ejemplos-intuitivos/
- [21] https://www.youtube.com/watch?v=CH7FWj3xcpo
- [22] https://joserzapata.github.io/courses/python-ciencia-datos/pandas/
- [23] https://www.youtube.com/watch?v=u5vfTWKLHe8
