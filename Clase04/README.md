
# Clase: Manejo de Datos Nulos y Series Temporales en Pandas

## **Repaso de Clases Anteriores**

### 🐍 Ejemplo Python - Condicionales para Ciencia de Datos

**¿Por qué son importantes los condicionales en ciencia de datos?**
Los condicionales nos permiten crear lógica de decisión en nuestros análisis, clasificar datos automáticamente y aplicar diferentes tratamientos según las características de los datos.

```python
# Clasificación de datos según rangos
def clasificar_edad(edad):
    """
    Función que clasifica personas según su edad en categorías demográficas.
    
    ¿Por qué usar rangos específicos?
    - 0-17: Menor de edad (restricciones legales, comportamientos diferentes)
    - 18-65: Adulto (población económicamente activa)
    - 65+: Adulto mayor (necesidades especiales, patrones de consumo diferentes)
    """
    if edad < 18:
        return "Menor de edad"
    elif 18 <= edad <= 65:
        return "Adulto"
    else:
        return "Adulto mayor"

# Aplicar a una lista de edades usando list comprehension
# ¿Por qué list comprehension? Es más eficiente y legible que un bucle for tradicional
edades = [15, 25, 70, 30, 12]
clasificaciones = [clasificar_edad(edad) for edad in edades]
print(clasificaciones)  # ['Menor de edad', 'Adulto', 'Adulto mayor', 'Adulto', 'Menor de edad']

# Ejemplo práctico: Análisis de clientes por edad
clientes = {
    'nombres': ['Ana', 'Juan', 'María', 'Carlos', 'Lucía'],
    'edades': [15, 25, 70, 30, 12],
    'gastos': [50, 200, 150, 300, 30]
}

# Crear DataFrame y agregar clasificación
import pandas as pd
df_clientes = pd.DataFrame(clientes)
df_clientes['categoria_edad'] = df_clientes['edades'].apply(clasificar_edad)

# Análisis por categoría de edad
analisis_por_edad = df_clientes.groupby('categoria_edad').agg({
    'gastos': ['mean', 'count', 'sum']
}).round(2)

print("\nAnálisis de gastos por categoría de edad:")
print(analisis_por_edad)
```

### 🔢 Ejemplo NumPy - Operaciones Vectorizadas

**¿Por qué NumPy es fundamental en ciencia de datos?**
NumPy proporciona arrays multidimensionales y operaciones vectorizadas que son mucho más eficientes que los bucles tradicionales de Python. Esto es crucial cuando trabajamos con grandes volúmenes de datos.

```python
import numpy as np

# Crear arrays y operaciones vectorizadas
# Los arrays de NumPy permiten operaciones element-wise (elemento por elemento)
# ¿Por qué element-wise? Permite aplicar la misma operación a todos los elementos simultáneamente
temperaturas = np.array([22, 25, 18, 30, 15])
humedad = np.array([60, 70, 45, 80, 35])

print("Arrays originales:")
print(f"Temperaturas: {temperaturas}")
print(f"Humedad: {humedad}")

# Operaciones vectorizadas básicas
# ¿Por qué vectorizadas? Son más rápidas que los bucles y más legibles
temperaturas_fahrenheit = (temperaturas * 9/5) + 32
print(f"\nTemperaturas en Fahrenheit: {temperaturas_fahrenheit}")

# Normalización z-score: (x - media) / desviación_estándar
# ¿Por qué normalizar? Para que los datos tengan media=0 y desviación=1
# Esto es útil en machine learning para que todas las variables tengan la misma escala
temperaturas_norm = (temperaturas - np.mean(temperaturas)) / np.std(temperaturas)
print(f"\nNormalización Z-score:")
print(f"Temperaturas originales: {temperaturas}")
print(f"Media: {np.mean(temperaturas):.2f}")
print(f"Desviación estándar: {np.std(temperaturas):.2f}")
print(f"Temperaturas normalizadas: {temperaturas_norm}")

# Verificar que la normalización funcionó
print(f"\nVerificación de normalización:")
print(f"Media de datos normalizados: {np.mean(temperaturas_norm):.6f} (debería ser ~0)")
print(f"Desv. estándar de datos normalizados: {np.std(temperaturas_norm):.6f} (debería ser ~1)")

# Filtrado condicional: crea un array booleano
# ¿Por qué usar arrays booleanos? Para indexación eficiente y filtrado
dias_calidos = temperaturas > 25
print(f"\nFiltrado condicional:")
print(f"Días calurosos (booleanos): {dias_calidos}")  # [False False False True False]
print(f"Temperaturas de días calurosos: {temperaturas[dias_calidos]}")  # [30]

# Operaciones más complejas: índice de calor aproximado
# Fórmula: 0.5 * (T + 61.0 + ((T-68.0)*1.2) + (H*0.094))
# Donde T = temperatura en Fahrenheit, H = humedad relativa
indice_calor = 0.5 * (temperaturas_fahrenheit + 61.0 + 
                     ((temperaturas_fahrenheit - 68.0) * 1.2) + 
                     (humedad * 0.094))

print(f"\nÍndice de calor aproximado:")
for i, (temp, hum, indice) in enumerate(zip(temperaturas, humedad, indice_calor)):
    print(f"Día {i+1}: Temp={temp}°C, Hum={hum}%, Índice={indice:.1f}°F")

# Estadísticas descriptivas completas
print(f"\nEstadísticas descriptivas de temperaturas:")
print(f"Media: {np.mean(temperaturas):.2f}°C")
print(f"Mediana: {np.median(temperaturas):.2f}°C")
print(f"Desviación estándar: {np.std(temperaturas):.2f}°C")
print(f"Mínimo: {np.min(temperaturas)}°C")
print(f"Máximo: {np.max(temperaturas)}°C")
print(f"Rango: {np.max(temperaturas) - np.min(temperaturas)}°C")
```

### 📊 Ejemplo Pandas - Análisis Básico

**¿Por qué Pandas es la herramienta principal para análisis de datos?**
Pandas proporciona estructuras de datos flexibles (Series y DataFrames) que permiten manipular, analizar y visualizar datos de manera eficiente. Es especialmente útil para datos tabulares y series temporales.

```python
import pandas as pd
import numpy as np

# Crear DataFrame de ventas
# Los diccionarios son útiles para crear DataFrames porque las claves se convierten en nombres de columnas
ventas_data = {
    'producto': ['Laptop', 'Mouse', 'Laptop', 'Teclado', 'Mouse'],
    'cantidad': [10, 5, 15, 8, 12],
    'precio': [1000, 25, 1000, 80, 25],
    'fecha': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
    'vendedor': ['Juan', 'Ana', 'Juan', 'Carlos', 'Ana']
}
df_ventas = pd.DataFrame(ventas_data)

# Convertir fecha a datetime para análisis temporal
df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'])

print("DataFrame original:")
print(df_ventas)
print(f"\nInformación del DataFrame:")
print(f"Dimensiones: {df_ventas.shape}")
print(f"Tipos de datos:")
print(df_ventas.dtypes)

# Calcular columna derivada: ingresos totales por venta
# ¿Por qué calcular ingresos? Es una métrica clave para el negocio
df_ventas['ingresos'] = df_ventas['cantidad'] * df_ventas['precio']

print(f"\nDataFrame con ingresos calculados:")
print(df_ventas[['producto', 'cantidad', 'precio', 'ingresos']])

# Análisis por producto usando groupby
# ¿Por qué groupby? Permite agrupar datos por categorías y aplicar funciones de agregación
# ¿Por qué 'sum' en cantidad? Para obtener el total vendido de cada producto
# ¿Por qué 'mean' en precio? Para obtener el precio promedio de cada producto
resumen_productos = df_ventas.groupby('producto').agg({
    'cantidad': 'sum',        # Suma total de cantidades por producto
    'precio': 'mean',         # Precio promedio por producto
    'ingresos': 'sum',        # Ingresos totales por producto
    'fecha': 'count'          # Número de ventas por producto
}).round(2)

resumen_productos.columns = ['Total_Unidades', 'Precio_Promedio', 'Ingresos_Totales', 'Num_Ventas']
print(f"\nResumen por producto:")
print(resumen_productos)

# Análisis por vendedor
resumen_vendedores = df_ventas.groupby('vendedor').agg({
    'ingresos': ['sum', 'mean', 'count']
}).round(2)

resumen_vendedores.columns = ['Ingresos_Totales', 'Ingresos_Promedio', 'Num_Ventas']
print(f"\nResumen por vendedor:")
print(resumen_vendedores)

# Análisis temporal: ventas por día
ventas_por_dia = df_ventas.groupby('fecha').agg({
    'ingresos': 'sum',
    'cantidad': 'sum'
}).round(2)

print(f"\nVentas por día:")
print(ventas_por_dia)

# Estadísticas descriptivas completas
print(f"\nEstadísticas descriptivas de ingresos:")
print(df_ventas['ingresos'].describe())

# Identificar el producto más vendido y el más rentable
producto_mas_vendido = df_ventas.groupby('producto')['cantidad'].sum().idxmax()
producto_mas_rentable = df_ventas.groupby('producto')['ingresos'].sum().idxmax()

print(f"\nAnálisis de productos:")
print(f"Producto más vendido (por unidades): {producto_mas_vendido}")
print(f"Producto más rentable (por ingresos): {producto_mas_rentable}")

# Calcular margen de beneficio (asumiendo costo del 60% del precio)
df_ventas['costo'] = df_ventas['precio'] * 0.6
df_ventas['beneficio'] = df_ventas['ingresos'] - (df_ventas['cantidad'] * df_ventas['costo'])
df_ventas['margen_beneficio'] = (df_ventas['beneficio'] / df_ventas['ingresos']) * 100

print(f"\nAnálisis de rentabilidad:")
print(df_ventas[['producto', 'ingresos', 'beneficio', 'margen_beneficio']].round(2))
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

**¿Por qué es crucial identificar datos ausentes?**
Los datos ausentes pueden sesgar nuestros análisis, afectar la precisión de los modelos de machine learning y llevar a conclusiones incorrectas. Por eso, el primer paso es siempre entender la magnitud y el patrón de los valores faltantes.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Crear dataset de ejemplo con valores ausentes
np.random.seed(42)
data_ejemplo = {
    'id': range(1, 101),
    'edad': np.random.normal(35, 10, 100),
    'salario': np.random.normal(50000, 15000, 100),
    'experiencia': np.random.normal(8, 5, 100),
    'departamento': np.random.choice(['IT', 'HR', 'Marketing', 'Ventas'], 100),
    'satisfaccion': np.random.uniform(1, 10, 100)
}

df = pd.DataFrame(data_ejemplo)

# Introducir valores ausentes de manera controlada para simular escenarios reales
# 10% de valores ausentes en edad (aleatorios)
df.loc[np.random.choice(df.index, size=int(len(df)*0.1), replace=False), 'edad'] = np.nan

# 15% de valores ausentes en salario (aleatorios)
df.loc[np.random.choice(df.index, size=int(len(df)*0.15), replace=False), 'salario'] = np.nan

# 5% de valores ausentes en departamento (aleatorios)
df.loc[np.random.choice(df.index, size=int(len(df)*0.05), replace=False), 'departamento'] = np.nan

print("Dataset con valores ausentes:")
print(df.head())
print(f"\nDimensiones del dataset: {df.shape}")

# Detectar valores ausentes
# isnull() devuelve True donde hay valores nulos, False donde no los hay
print(f"\nMatriz de valores nulos (primeras 10 filas):")
print(df.isnull().head(10))

# sum() cuenta los True (valores nulos) por columna
# ¿Por qué sum()? Porque True=1, False=0, entonces sum() cuenta los nulos
valores_nulos_por_columna = df.isnull().sum()
print(f"\nValores nulos por columna:")
print(valores_nulos_por_columna)

# Información detallada para entender la magnitud del problema
# sum().sum() suma todos los valores nulos del DataFrame completo
total_nulos = df.isnull().sum().sum()
# df.size es el total de elementos en el DataFrame (filas × columnas)
porcentaje_nulos = (total_nulos / df.size) * 100

print(f"\nResumen de valores ausentes:")
print(f"Total de valores nulos: {total_nulos}")
print(f"Porcentaje de valores nulos: {porcentaje_nulos:.2f}%")
print(f"Total de elementos en el dataset: {df.size}")

# Análisis más detallado
print(f"\nAnálisis detallado por columna:")
for columna in df.columns:
    nulos = df[columna].isnull().sum()
    porcentaje = (nulos / len(df)) * 100
    print(f"{columna}: {nulos} valores nulos ({porcentaje:.1f}%)")

# Visualización de valores ausentes
plt.figure(figsize=(12, 8))

# Gráfico 1: Conteo de valores nulos por columna
plt.subplot(2, 2, 1)
valores_nulos_por_columna.plot(kind='bar', color='red', alpha=0.7)
plt.title('Valores Nulos por Columna')
plt.ylabel('Cantidad de Valores Nulos')
plt.xticks(rotation=45)

# Gráfico 2: Porcentaje de valores nulos por columna
plt.subplot(2, 2, 2)
porcentajes_nulos = (valores_nulos_por_columna / len(df)) * 100
porcentajes_nulos.plot(kind='bar', color='orange', alpha=0.7)
plt.title('Porcentaje de Valores Nulos por Columna')
plt.ylabel('Porcentaje (%)')
plt.xticks(rotation=45)

# Gráfico 3: Matriz de valores ausentes (heatmap)
plt.subplot(2, 2, 3)
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Matriz de Valores Ausentes')
plt.xlabel('Columnas')

# Gráfico 4: Distribución de valores nulos en el dataset
plt.subplot(2, 2, 4)
filas_con_nulos = df.isnull().sum(axis=1)
plt.hist(filas_con_nulos, bins=range(filas_con_nulos.max() + 2), alpha=0.7, color='green')
plt.title('Distribución de Valores Nulos por Fila')
plt.xlabel('Número de Valores Nulos por Fila')
plt.ylabel('Frecuencia')

plt.tight_layout()
plt.show()

# Análisis de patrones de valores ausentes
print(f"\nAnálisis de patrones:")
print(f"Filas sin valores nulos: {(df.isnull().sum(axis=1) == 0).sum()}")
print(f"Filas con al menos un valor nulo: {(df.isnull().sum(axis=1) > 0).sum()}")

# Identificar filas con múltiples valores nulos
filas_problematicas = df[df.isnull().sum(axis=1) >= 2]
print(f"\nFilas con 2 o más valores nulos: {len(filas_problematicas)}")
if len(filas_problematicas) > 0:
    print("Ejemplos de filas problemáticas:")
    print(filas_problematicas.head())
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

**¿Cuándo usar cada técnica de imputación?**
La elección del método de imputación depende del tipo de datos, el contexto del problema y el patrón de valores ausentes. Es crucial entender las implicaciones de cada método.

```python
# Continuamos con el dataset anterior
print("Dataset original con valores ausentes:")
print(df.head())
print(f"\nValores nulos antes de imputación:")
print(df.isnull().sum())

# Crear copias para comparar diferentes métodos
df_media = df.copy()
df_mediana = df.copy()
df_constante = df.copy()
df_forward = df.copy()

# 1. Rellenar con valor constante
# ¿Cuándo usar 0? Cuando los valores nulos representan "sin valor" o "no aplica"
# Ejemplo: gastos de marketing para clientes que no tienen campaña activa
df_constante.fillna(0, inplace=True)

# 2. Imputar con estadísticas por columna
# ¿Por qué media para edad? Es representativa cuando los datos están normalmente distribuidos
# Ventaja: Mantiene la media de la distribución original
# Desventaja: Puede no ser representativa si hay outliers
df_media['edad'].fillna(df_media['edad'].mean(), inplace=True)

# ¿Por qué mediana para salario? Es robusta a outliers (valores extremos)
# Ventaja: No se ve afectada por valores extremos
# Desventaja: Puede no reflejar la distribución real si hay muchos outliers
df_mediana['salario'].fillna(df_mediana['salario'].median(), inplace=True)

# ¿Por qué moda para categorías? Es el valor más frecuente, lógico para datos categóricos
# Ventaja: Mantiene la categoría más común
# Desventaja: Puede crear sesgo si una categoría es muy dominante
df_media['departamento'].fillna(df_media['departamento'].mode()[0], inplace=True)

# 3. Forward fill y backward fill para series temporales
# ¿Cuándo usar ffill? Cuando el valor anterior es una buena estimación (ej: precio de ayer)
# Ventaja: Mantiene la continuidad temporal
# Desventaja: Puede propagar errores
df_forward['edad'].fillna(method='ffill', inplace=True)

# ¿Cuándo usar bfill? Cuando el valor siguiente es más relevante
# Ventaja: Útil cuando los valores futuros son más informativos
# Desventaja: Puede no estar disponible para el último valor
df_forward['salario'].fillna(method='bfill', inplace=True)

# Comparar resultados de diferentes métodos
print(f"\nComparación de métodos de imputación:")

# Estadísticas de edad antes y después
print(f"\nEstadísticas de EDAD:")
print(f"Original (sin nulos): Media={df['edad'].mean():.2f}, Mediana={df['edad'].median():.2f}")
print(f"Con media: Media={df_media['edad'].mean():.2f}, Mediana={df_media['edad'].median():.2f}")
print(f"Con forward fill: Media={df_forward['edad'].mean():.2f}, Mediana={df_forward['edad'].median():.2f}")

# Estadísticas de salario antes y después
print(f"\nEstadísticas de SALARIO:")
print(f"Original (sin nulos): Media={df['salario'].mean():.2f}, Mediana={df['salario'].median():.2f}")
print(f"Con mediana: Media={df_mediana['salario'].mean():.2f}, Mediana={df_mediana['salario'].median():.2f}")
print(f"Con backward fill: Media={df_forward['salario'].mean():.2f}, Mediana={df_forward['salario'].median():.2f}")

# Visualizar el impacto de diferentes métodos
plt.figure(figsize=(15, 10))

# Gráfico 1: Distribución de edad
plt.subplot(2, 3, 1)
plt.hist(df['edad'].dropna(), bins=20, alpha=0.7, label='Original', color='blue')
plt.hist(df_media['edad'], bins=20, alpha=0.7, label='Con media', color='red')
plt.title('Distribución de Edad')
plt.legend()

# Gráfico 2: Distribución de salario
plt.subplot(2, 3, 2)
plt.hist(df['salario'].dropna(), bins=20, alpha=0.7, label='Original', color='blue')
plt.hist(df_mediana['salario'], bins=20, alpha=0.7, label='Con mediana', color='green')
plt.title('Distribución de Salario')
plt.legend()

# Gráfico 3: Boxplot comparativo de edad
plt.subplot(2, 3, 3)
datos_edad = [df['edad'].dropna(), df_media['edad'], df_forward['edad']]
plt.boxplot(datos_edad, labels=['Original', 'Media', 'Forward Fill'])
plt.title('Boxplot de Edad por Método')

# Gráfico 4: Boxplot comparativo de salario
plt.subplot(2, 3, 4)
datos_salario = [df['salario'].dropna(), df_mediana['salario'], df_forward['salario']]
plt.boxplot(datos_salario, labels=['Original', 'Mediana', 'Backward Fill'])
plt.title('Boxplot de Salario por Método')

# Gráfico 5: Distribución de departamentos
plt.subplot(2, 3, 5)
df['departamento'].value_counts().plot(kind='bar', alpha=0.7, label='Original', color='blue')
df_media['departamento'].value_counts().plot(kind='bar', alpha=0.7, label='Con moda', color='orange')
plt.title('Distribución de Departamentos')
plt.legend()
plt.xticks(rotation=45)

# Gráfico 6: Comparación de estadísticas
plt.subplot(2, 3, 6)
metodos = ['Original', 'Media', 'Mediana', 'Forward Fill']
medias_edad = [df['edad'].mean(), df_media['edad'].mean(), 
               df_mediana['edad'].mean(), df_forward['edad'].mean()]
plt.bar(metodos, medias_edad, color=['blue', 'red', 'green', 'purple'], alpha=0.7)
plt.title('Media de Edad por Método')
plt.ylabel('Edad Promedio')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Análisis de correlaciones antes y después de la imputación
print(f"\nAnálisis de correlaciones:")
print(f"Correlación original (edad vs salario): {df['edad'].corr(df['salario']):.3f}")
print(f"Correlación con imputación por media: {df_media['edad'].corr(df_media['salario']):.3f}")
print(f"Correlación con imputación por mediana: {df_mediana['edad'].corr(df_mediana['salario']):.3f}")

# Evaluación de la calidad de la imputación
def evaluar_imputacion(df_original, df_imputado, columna):
    """
    Evalúa la calidad de la imputación comparando estadísticas
    """
    original_sin_nulos = df_original[columna].dropna()
    imputado = df_imputado[columna]
    
    # Calcular diferencias en estadísticas
    diff_media = abs(original_sin_nulos.mean() - imputado.mean())
    diff_mediana = abs(original_sin_nulos.median() - imputado.median())
    diff_std = abs(original_sin_nulos.std() - imputado.std())
    
    print(f"\nEvaluación para {columna}:")
    print(f"Diferencia en media: {diff_media:.2f}")
    print(f"Diferencia en mediana: {diff_mediana:.2f}")
    print(f"Diferencia en desviación estándar: {diff_std:.2f}")

evaluar_imputacion(df, df_media, 'edad')
evaluar_imputacion(df, df_mediana, 'salario')
```

### 3.4 Imputación Avanzada con Scikit-Learn
```python
from sklearn.impute import SimpleImputer, KNNImputer

# SimpleImputer con diferentes estrategias
# ¿Por qué usar SimpleImputer? Es más robusto y permite aplicar la misma estrategia a múltiples columnas
imputer_media = SimpleImputer(strategy='mean')
imputer_mediana = SimpleImputer(strategy='median')
imputer_moda = SimpleImputer(strategy='most_frequent')
imputer_constante = SimpleImputer(strategy='constant', fill_value=0)

# Aplicar a columnas numéricas
# ¿Por qué select_dtypes? Para aplicar solo a columnas numéricas, no a texto
columnas_numericas = df.select_dtypes(include=[np.number]).columns
df[columnas_numericas] = imputer_media.fit_transform(df[columnas_numericas])

# KNN Imputer (más sofisticado)
# ¿Por qué KNN? Usa los datos más similares para imputar, preservando relaciones entre variables
# ¿Por qué n_neighbors=5? Balance entre precisión y velocidad (más vecinos = más preciso pero más lento)
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
# ¿Por qué .str? Permite aplicar métodos de string a cada elemento de la columna
df['nombre'].str.upper()  # Convertir a mayúsculas - útil para estandarizar
df['email'].str.lower()   # Convertir a minúsculas - emails no distinguen mayúsculas/minúsculas
df['texto'].str.strip()   # Eliminar espacios en blanco - limpia datos ingresados manualmente

# Longitud de strings
# ¿Para qué sirve? Detectar valores anómalos (nombres muy cortos/largos)
df['nombre'].str.len()    # Longitud de cada string

# Concatenación
# ¿Por qué sep=' '? Para separar nombre y apellido con un espacio
df['nombre_completo'] = df['nombre'].str.cat(df['apellido'], sep=' ')
```

### 4.2 Búsqueda y Filtrado
```python
# Contiene un patrón
# ¿Por qué case=False? Para hacer búsqueda insensible a mayúsculas/minúsculas
df[df['producto'].str.contains('laptop', case=False)]

# Comienza con
# ¿Para qué sirve? Filtrar emails administrativos, códigos que empiecen igual
df[df['email'].str.startswith('admin')]

# Termina con
# ¿Para qué sirve? Filtrar por extensiones de archivo, dominios de email
df[df['archivo'].str.endswith('.csv')]

# Coincidencia exacta con regex
# ¿Por qué ^ y $? ^ = inicio de string, $ = fin de string, para coincidencia exacta
df[df['categoria'].str.match('^Electrónicos$')]
```

### 4.3 Extracción y Reemplazo
```python
# Extraer parte del string usando regex
# ¿Qué hace r'\+(\d+)'? \+ = símbolo + literal, (\d+) = uno o más dígitos (capturados)
df['codigo_pais'] = df['telefono'].str.extract(r'\+(\d+)')

# ¿Qué hace r'@(.+)'? @ = símbolo @ literal, (.+) = cualquier carácter después del @
df['dominio'] = df['email'].str.extract(r'@(.+)')

# Reemplazar patrones
# ¿Qué hace r'[^\d]'? [^\d] = cualquier carácter que NO sea dígito
df['telefono_limpio'] = df['telefono'].str.replace(r'[^\d]', '', regex=True)

# ¿Por qué encadenar replace? Para eliminar múltiples caracteres en secuencia
df['precio_limpio'] = df['precio'].str.replace('$', '').str.replace(',', '')

# Split y join
# ¿Por qué split()? Para separar texto en palabras individuales
df['palabras'] = df['descripcion'].str.split()
# ¿Por qué .str[0]? Para obtener la primera palabra después del split
df['primera_palabra'] = df['descripcion'].str.split().str[0]
```

### 4.4 Validación y Limpieza
```python
# Validar formato de email
# ¿Por qué usar regex para validar? Para verificar que el formato sea correcto
def es_email_valido(email):
    import re
    # Patrón regex: usuario@dominio.extension
    # ^ = inicio, [a-zA-Z0-9._%+-]+ = caracteres permitidos en usuario
    # @ = símbolo @, [a-zA-Z0-9.-]+ = caracteres permitidos en dominio
    # \. = punto literal, [a-zA-Z]{2,} = extensión de 2+ letras, $ = fin
    patron = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(patron, str(email)))

df['email_valido'] = df['email'].apply(es_email_valido)

# Limpiar nombres - encadenamiento de métodos
# ¿Por qué encadenar? Para aplicar múltiples transformaciones en una línea
df['nombre_limpio'] = (df['nombre']
                      .str.strip()                    # Eliminar espacios
                      .str.title()                    # Capitalizar palabras
                      .str.replace(r'\s+', ' ', regex=True))  # Múltiples espacios → uno
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

---

## **4.4 Matplotlib - Visualización de Datos**

### ¿Qué es Matplotlib?

**Matplotlib** es una de las bibliotecas más populares en Python para la creación de gráficos y visualizaciones de datos. Fundada en 2003 por John D. Hunter, proporciona herramientas robustas para generar gráficos bidimensionales de alta calidad. Es fundamental para cualquiera que trabaje con datos en Python, siendo la base sobre la cual se han construido otras bibliotecas como Seaborn y Plotly.

### ¿Por qué Matplotlib?

Matplotlib se destaca por su **simplicidad y flexibilidad**. Aunque existen otras bibliotecas para la visualización de datos, Matplotlib sigue siendo la más utilizada gracias a su capacidad para generar gráficos altamente personalizables. Además, al ser de código abierto, es accesible y ampliamente utilizada por la comunidad científica y de desarrollo.

### Interfaces Principales de Matplotlib

Matplotlib ofrece dos interfaces principales:

1. **Interfaz orientada a objetos**: Es la más flexible y poderosa, permitiendo un control detallado sobre los gráficos. Trata a los gráficos como objetos reutilizables, lo que es ideal para crear gráficos complejos o manejar múltiples gráficos simultáneamente.

2. **Interfaz orientada a estados (pyplot)**: Similar a MATLAB, esta interfaz es más simple y directa, ideal para gráficos rápidos. Sin embargo, para gráficos más complejos, la interfaz orientada a objetos es preferida por su mayor control y flexibilidad.

### Configuración Inicial y Uso Básico

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuración de estilo para gráficos más atractivos
plt.style.use('seaborn-v0_8')  # Estilo moderno y atractivo
plt.rcParams['figure.figsize'] = (10, 6)  # Tamaño por defecto de las figuras
plt.rcParams['font.size'] = 12  # Tamaño de fuente por defecto

# Crear datos de ejemplo
np.random.seed(42)  # Para reproducibilidad
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)

# Gráfico básico usando la interfaz orientada a objetos
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, 'b-', linewidth=2, label='Datos con ruido')
ax.plot(x, np.sin(x), 'r--', linewidth=2, label='Función seno original')
ax.set_xlabel('Eje X', fontsize=14)
ax.set_ylabel('Eje Y', fontsize=14)
ax.set_title('Gráfico de Líneas con Matplotlib', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Tipos de Gráficos Comunes

#### 1. **Gráfico de Líneas (Line Plot)**

```python
# Datos de ventas mensuales
meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun']
ventas = [120, 150, 180, 200, 220, 250]
gastos = [100, 120, 140, 160, 180, 200]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(meses, ventas, 'o-', linewidth=2, markersize=8, label='Ventas', color='blue')
ax.plot(meses, gastos, 's-', linewidth=2, markersize=8, label='Gastos', color='red')
ax.set_xlabel('Mes', fontsize=14)
ax.set_ylabel('Monto ($)', fontsize=14)
ax.set_title('Evolución de Ventas vs Gastos', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

#### 2. **Gráfico de Barras (Bar Plot)**

```python
# Datos de productos más vendidos
productos = ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Auriculares']
ventas = [45, 120, 80, 30, 95]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(productos, ventas, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83'])
ax.set_xlabel('Producto', fontsize=14)
ax.set_ylabel('Unidades Vendidas', fontsize=14)
ax.set_title('Ventas por Producto', fontsize=16, fontweight='bold')

# Agregar valores sobre las barras
for bar, venta in zip(bars, ventas):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{venta}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
```

#### 3. **Gráfico de Dispersión (Scatter Plot)**

```python
# Datos de relación entre precio y ventas
np.random.seed(42)
precios = np.random.uniform(50, 500, 50)
ventas = 1000 - 1.5 * precios + np.random.normal(0, 50, 50)

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(precios, ventas, c=ventas, cmap='viridis', s=100, alpha=0.7)
ax.set_xlabel('Precio ($)', fontsize=14)
ax.set_ylabel('Ventas (unidades)', fontsize=14)
ax.set_title('Relación Precio vs Ventas', fontsize=16, fontweight='bold')

# Agregar línea de tendencia
z = np.polyfit(precios, ventas, 1)
p = np.poly1d(z)
ax.plot(precios, p(precios), "r--", alpha=0.8, linewidth=2, label='Tendencia')

# Agregar barra de color
cbar = plt.colorbar(scatter)
cbar.set_label('Nivel de Ventas', fontsize=12)

ax.legend(fontsize=12)
plt.tight_layout()
plt.show()
```

#### 4. **Histograma**

```python
# Datos de edades de clientes
np.random.seed(42)
edades = np.random.normal(35, 12, 1000)  # Media=35, Desv=12, 1000 clientes

fig, ax = plt.subplots(figsize=(10, 6))
n, bins, patches = ax.hist(edades, bins=30, alpha=0.7, color='skyblue', edgecolor='black')

# Agregar línea de densidad normal
from scipy.stats import norm
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, np.mean(edades), np.std(edades))
ax.plot(x, p * len(edades) * (bins[1] - bins[0]), 'r-', linewidth=2, label='Distribución Normal')

ax.set_xlabel('Edad', fontsize=14)
ax.set_ylabel('Frecuencia', fontsize=14)
ax.set_title('Distribución de Edades de Clientes', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

#### 5. **Boxplot**

```python
# Datos de salarios por departamento
np.random.seed(42)
it_salarios = np.random.normal(65000, 15000, 100)
hr_salarios = np.random.normal(55000, 12000, 100)
marketing_salarios = np.random.normal(60000, 18000, 100)

fig, ax = plt.subplots(figsize=(10, 6))
data = [it_salarios, hr_salarios, marketing_salarios]
labels = ['IT', 'Recursos Humanos', 'Marketing']

box_plot = ax.boxplot(data, labels=labels, patch_artist=True)

# Colorear las cajas
colors = ['lightblue', 'lightgreen', 'lightcoral']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)

ax.set_ylabel('Salario Anual ($)', fontsize=14)
ax.set_title('Distribución de Salarios por Departamento', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Subplots y Múltiples Gráficos

```python
# Crear múltiples gráficos en una sola figura
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Dashboard de Análisis de Datos', fontsize=16, fontweight='bold')

# Gráfico 1: Línea temporal
axes[0, 0].plot(meses, ventas, 'o-', color='blue')
axes[0, 0].set_title('Ventas Mensuales')
axes[0, 0].set_ylabel('Ventas ($)')
axes[0, 0].grid(True, alpha=0.3)

# Gráfico 2: Barras
axes[0, 1].bar(productos, ventas, color='green', alpha=0.7)
axes[0, 1].set_title('Ventas por Producto')
axes[0, 1].set_ylabel('Unidades')

# Gráfico 3: Dispersión
axes[1, 0].scatter(precios, ventas, alpha=0.6, color='red')
axes[1, 0].set_title('Precio vs Ventas')
axes[1, 0].set_xlabel('Precio ($)')
axes[1, 0].set_ylabel('Ventas')

# Gráfico 4: Histograma
axes[1, 1].hist(edades, bins=20, alpha=0.7, color='purple')
axes[1, 1].set_title('Distribución de Edades')
axes[1, 1].set_xlabel('Edad')
axes[1, 1].set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()
```

### Personalización Avanzada

```python
# Gráfico con personalización completa
fig, ax = plt.subplots(figsize=(12, 8))

# Crear datos
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Gráfico principal
line1 = ax.plot(x, y1, 'b-', linewidth=3, label='Seno', alpha=0.8)
line2 = ax.plot(x, y2, 'r-', linewidth=3, label='Coseno', alpha=0.8)

# Personalizar ejes
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel('Ángulo (radianes)', fontsize=14, fontweight='bold')
ax.set_ylabel('Valor', fontsize=14, fontweight='bold')
ax.set_title('Funciones Trigonométricas', fontsize=16, fontweight='bold', pad=20)

# Personalizar cuadrícula
ax.grid(True, linestyle='--', alpha=0.7, color='gray')
ax.set_axisbelow(True)  # Poner la cuadrícula detrás de los datos

# Personalizar leyenda
ax.legend(loc='upper right', fontsize=12, framealpha=0.9, shadow=True)

# Agregar anotaciones
ax.annotate('Máximo del Seno', xy=(np.pi/2, 1), xytext=(np.pi/2 + 0.5, 1.1),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            fontsize=12, color='blue')

ax.annotate('Máximo del Coseno', xy=(0, 1), xytext=(-0.5, 1.1),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, color='red')

# Personalizar el fondo
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.show()
```

### Guardar Gráficos

```python
# Crear un gráfico y guardarlo en diferentes formatos
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y1, 'b-', linewidth=2, label='Seno')
ax.plot(x, y2, 'r-', linewidth=2, label='Coseno')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Funciones Trigonométricas')
ax.legend()
ax.grid(True, alpha=0.3)

# Guardar en diferentes formatos
plt.savefig('grafico.png', dpi=300, bbox_inches='tight')  # PNG de alta calidad
plt.savefig('grafico.pdf', bbox_inches='tight')  # PDF vectorial
plt.savefig('grafico.svg', bbox_inches='tight')  # SVG escalable

plt.show()
```

### Mejores Prácticas

1. **Siempre usar la interfaz orientada a objetos** para gráficos complejos
2. **Configurar el estilo al inicio** del script
3. **Usar `plt.tight_layout()`** para evitar superposición
4. **Guardar gráficos con `bbox_inches='tight'`** para recortar espacios en blanco
5. **Usar colores consistentes** y accesibles
6. **Agregar títulos y etiquetas descriptivas**
7. **Incluir leyendas cuando sea necesario**
8. **Usar `plt.show()`** al final para mostrar el gráfico

### Integración con Pandas

```python
# Crear DataFrame de ejemplo
df = pd.DataFrame({
    'fecha': pd.date_range('2024-01-01', periods=100, freq='D'),
    'ventas': np.random.normal(100, 20, 100).cumsum(),
    'gastos': np.random.normal(80, 15, 100).cumsum()
})

# Gráfico usando Pandas con Matplotlib
fig, ax = plt.subplots(figsize=(12, 6))
df.plot(x='fecha', y=['ventas', 'gastos'], ax=ax, linewidth=2)
ax.set_title('Evolución de Ventas y Gastos', fontsize=16, fontweight='bold')
ax.set_xlabel('Fecha', fontsize=14)
ax.set_ylabel('Monto ($)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## **4.5 Creación de Gráficos Básicos - Análisis de Datos Secuenciales y Series de Tiempo**

### Introducción a Datos Secuenciales y Series de Tiempo

**¿Por qué son importantes las series de tiempo?**
Las series de tiempo y los datos secuenciales son fundamentales en el análisis de datos, ya que permiten visualizar y analizar cómo cambia una variable a lo largo del tiempo. Esto es crucial para:
- Identificar tendencias y patrones
- Detectar estacionalidad
- Predecir comportamientos futuros
- Tomar decisiones basadas en evolución temporal

### 1. Gráficos de Líneas - Herramienta Principal para Series de Tiempo

**¿Cuándo usar gráficos de líneas?**
Los gráficos de líneas son la herramienta más común para representar series de tiempo. Muestran la evolución de los datos en función del tiempo, permitiendo identificar tendencias, patrones estacionales y posibles anomalías.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configuración de estilo
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)

# Crear datos de series de tiempo
np.random.seed(42)
fechas = pd.date_range('2024-01-01', periods=365, freq='D')

# Simular datos realistas de ventas con tendencia y estacionalidad
tendencia = np.linspace(100, 150, 365)  # Tendencia creciente
estacionalidad = 20 * np.sin(2 * np.pi * np.arange(365) / 365)  # Patrón anual
ruido = np.random.normal(0, 10, 365)  # Ruido aleatorio
ventas = tendencia + estacionalidad + ruido

# Crear DataFrame
df_ventas = pd.DataFrame({
    'fecha': fechas,
    'ventas': ventas
})

# Gráfico de líneas básico
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(df_ventas['fecha'], df_ventas['ventas'], linewidth=2, color='blue', alpha=0.8)
ax.set_title('Evolución de Ventas Diarias - 2024', fontsize=16, fontweight='bold')
ax.set_xlabel('Fecha', fontsize=14)
ax.set_ylabel('Ventas ($)', fontsize=14)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Gráfico de líneas con múltiples series
# Agregar datos de gastos y beneficios
gastos = ventas * 0.6 + np.random.normal(0, 5, 365)  # 60% de las ventas + ruido
beneficios = ventas - gastos

df_completo = pd.DataFrame({
    'fecha': fechas,
    'ventas': ventas,
    'gastos': gastos,
    'beneficios': beneficios
})

fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(df_completo['fecha'], df_completo['ventas'], linewidth=2, label='Ventas', color='blue')
ax.plot(df_completo['fecha'], df_completo['gastos'], linewidth=2, label='Gastos', color='red')
ax.plot(df_completo['fecha'], df_completo['beneficios'], linewidth=2, label='Beneficios', color='green')

ax.set_title('Análisis Financiero Anual - 2024', fontsize=16, fontweight='bold')
ax.set_xlabel('Fecha', fontsize=14)
ax.set_ylabel('Monto ($)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Análisis de tendencias con líneas de tendencia
fig, ax = plt.subplots(figsize=(15, 8))

# Datos originales
ax.plot(df_ventas['fecha'], df_ventas['ventas'], linewidth=2, color='blue', alpha=0.7, label='Ventas Reales')

# Línea de tendencia (polinomio de grado 1)
z = np.polyfit(range(len(df_ventas)), df_ventas['ventas'], 1)
p = np.poly1d(z)
tendencia_lineal = p(range(len(df_ventas)))
ax.plot(df_ventas['fecha'], tendencia_lineal, 'r--', linewidth=3, label='Tendencia Lineal')

# Promedio móvil (suavizado)
ventana = 30  # Promedio móvil de 30 días
promedio_movil = df_ventas['ventas'].rolling(window=ventana).mean()
ax.plot(df_ventas['fecha'], promedio_movil, 'g-', linewidth=2, label=f'Promedio Móvil ({ventana} días)')

ax.set_title('Análisis de Tendencia de Ventas', fontsize=16, fontweight='bold')
ax.set_xlabel('Fecha', fontsize=14)
ax.set_ylabel('Ventas ($)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 2. Gráficos de Dispersión (Scatter Plot) - Análisis de Relaciones Temporales

**¿Cuándo usar gráficos de dispersión?**
Los gráficos de puntos son ideales para analizar la relación entre dos variables diferentes en el tiempo, especialmente cuando se busca identificar correlaciones temporales.

```python
# Crear datos de ejemplo: temperatura vs ventas de helados
np.random.seed(42)
temperaturas = np.random.uniform(15, 35, 100)  # Temperaturas entre 15-35°C
ventas_helados = 50 + 3 * temperaturas + np.random.normal(0, 10, 100)  # Relación positiva con ruido

# Gráfico de dispersión básico
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(temperaturas, ventas_helados, alpha=0.7, c=temperaturas, cmap='viridis', s=50)
ax.set_xlabel('Temperatura (°C)', fontsize=14)
ax.set_ylabel('Ventas de Helados ($)', fontsize=14)
ax.set_title('Relación entre Temperatura y Ventas de Helados', fontsize=16, fontweight='bold')

# Agregar línea de tendencia
z = np.polyfit(temperaturas, ventas_helados, 1)
p = np.poly1d(z)
ax.plot(temperaturas, p(temperaturas), "r--", alpha=0.8, linewidth=2, label='Tendencia')

# Agregar barra de color
cbar = plt.colorbar(scatter)
cbar.set_label('Temperatura (°C)', fontsize=12)

ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Gráfico de dispersión con series de tiempo
# Crear datos de ventas vs publicidad a lo largo del tiempo
fechas = pd.date_range('2024-01-01', periods=100, freq='D')
publicidad = np.random.uniform(1000, 5000, 100)  # Gasto en publicidad
ventas = 200 + 0.05 * publicidad + np.random.normal(0, 50, 100)  # Ventas relacionadas con publicidad

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: Dispersión
scatter = ax1.scatter(publicidad, ventas, c=range(len(fechas)), cmap='plasma', alpha=0.7, s=50)
ax1.set_xlabel('Gasto en Publicidad ($)', fontsize=14)
ax1.set_ylabel('Ventas ($)', fontsize=14)
ax1.set_title('Relación Publicidad vs Ventas', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Línea de tendencia
z = np.polyfit(publicidad, ventas, 1)
p = np.poly1d(z)
ax1.plot(publicidad, p(publicidad), "r--", alpha=0.8, linewidth=2)

# Gráfico 2: Series de tiempo
ax2.plot(fechas, publicidad, 'b-', linewidth=2, label='Gasto en Publicidad', alpha=0.7)
ax2_twin = ax2.twinx()
ax2_twin.plot(fechas, ventas, 'r-', linewidth=2, label='Ventas', alpha=0.7)

ax2.set_xlabel('Fecha', fontsize=14)
ax2.set_ylabel('Gasto en Publicidad ($)', fontsize=14, color='blue')
ax2_twin.set_ylabel('Ventas ($)', fontsize=14, color='red')
ax2.set_title('Evolución Temporal', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Leyendas
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()
```

### 3. Gráficos de Barras - Comparación de Valores Temporales

**¿Cuándo usar gráficos de barras?**
Los gráficos de barras son útiles para comparar valores en diferentes intervalos de tiempo, especialmente cuando se quiere resaltar diferencias entre períodos.

```python
# Datos de precipitaciones mensuales
meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
precipitaciones_2023 = [120, 95, 110, 85, 70, 45, 30, 25, 60, 90, 105, 130]
precipitaciones_2024 = [115, 100, 105, 90, 75, 50, 35, 30, 65, 95, 110, 125]

# Gráfico de barras comparativo
fig, ax = plt.subplots(figsize=(15, 8))

x = np.arange(len(meses))
width = 0.35

bars1 = ax.bar(x - width/2, precipitaciones_2023, width, label='2023', alpha=0.8, color='skyblue')
bars2 = ax.bar(x + width/2, precipitaciones_2024, width, label='2024', alpha=0.8, color='lightcoral')

ax.set_xlabel('Mes', fontsize=14)
ax.set_ylabel('Precipitaciones (mm)', fontsize=14)
ax.set_title('Precipitaciones Mensuales - Comparación 2023 vs 2024', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(meses)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# Agregar valores sobre las barras
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(bars1)
autolabel(bars2)

plt.tight_layout()
plt.show()

# Gráfico de barras apiladas para análisis de composición
# Datos de ventas por categoría de producto
categorias = ['Electrónicos', 'Ropa', 'Hogar', 'Deportes']
ventas_q1 = [45000, 32000, 28000, 15000]
ventas_q2 = [52000, 35000, 30000, 18000]
ventas_q3 = [48000, 38000, 32000, 20000]
ventas_q4 = [60000, 42000, 35000, 22000]

fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(categorias))
width = 0.2

ax.bar(x - width*1.5, ventas_q1, width, label='Q1', alpha=0.8, color='#2E86AB')
ax.bar(x - width*0.5, ventas_q2, width, label='Q2', alpha=0.8, color='#A23B72')
ax.bar(x + width*0.5, ventas_q3, width, label='Q3', alpha=0.8, color='#F18F01')
ax.bar(x + width*1.5, ventas_q4, width, label='Q4', alpha=0.8, color='#C73E1D')

ax.set_xlabel('Categoría de Producto', fontsize=14)
ax.set_ylabel('Ventas ($)', fontsize=14)
ax.set_title('Ventas por Categoría y Trimestre', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categorias)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

### 4. Histogramas - Distribución de Datos Temporales

**¿Cuándo usar histogramas?**
Los histogramas son ideales para visualizar la distribución de datos continuos a lo largo del tiempo, especialmente cuando se agrupan en intervalos.

```python
# Crear datos de ejemplo: distribución de temperaturas diarias
np.random.seed(42)
temperaturas_verano = np.random.normal(28, 5, 90)  # Verano: media 28°C
temperaturas_invierno = np.random.normal(12, 4, 90)  # Invierno: media 12°C
temperaturas_primavera = np.random.normal(20, 6, 90)  # Primavera: media 20°C
temperaturas_otono = np.random.normal(18, 5, 90)  # Otoño: media 18°C

# Histograma comparativo por estación
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Distribución de Temperaturas por Estación', fontsize=16, fontweight='bold')

ax1.hist(temperaturas_verano, bins=20, alpha=0.7, color='red', edgecolor='black')
ax1.set_title('Verano')
ax1.set_xlabel('Temperatura (°C)')
ax1.set_ylabel('Frecuencia')
ax1.grid(True, alpha=0.3)

ax2.hist(temperaturas_invierno, bins=20, alpha=0.7, color='blue', edgecolor='black')
ax2.set_title('Invierno')
ax2.set_xlabel('Temperatura (°C)')
ax2.set_ylabel('Frecuencia')
ax2.grid(True, alpha=0.3)

ax3.hist(temperaturas_primavera, bins=20, alpha=0.7, color='green', edgecolor='black')
ax3.set_title('Primavera')
ax3.set_xlabel('Temperatura (°C)')
ax3.set_ylabel('Frecuencia')
ax3.grid(True, alpha=0.3)

ax4.hist(temperaturas_otono, bins=20, alpha=0.7, color='orange', edgecolor='black')
ax4.set_title('Otoño')
ax4.set_xlabel('Temperatura (°C)')
ax4.set_ylabel('Frecuencia')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Histograma con curva de densidad
fig, ax = plt.subplots(figsize=(12, 6))

# Combinar todas las temperaturas
todas_temperaturas = np.concatenate([temperaturas_verano, temperaturas_invierno, 
                                   temperaturas_primavera, temperaturas_otono])

# Histograma
n, bins, patches = ax.hist(todas_temperaturas, bins=30, alpha=0.7, color='skyblue', 
                          edgecolor='black', density=True, label='Frecuencia')

# Curva de densidad normal
from scipy.stats import norm
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, np.mean(todas_temperaturas), np.std(todas_temperaturas))
ax.plot(x, p, 'r-', linewidth=2, label='Distribución Normal')

ax.set_xlabel('Temperatura (°C)', fontsize=14)
ax.set_ylabel('Densidad de Probabilidad', fontsize=14)
ax.set_title('Distribución de Temperaturas Anuales', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 5. Boxplots - Análisis de Distribución y Outliers Temporales

**¿Cuándo usar boxplots?**
Los boxplots son útiles para mostrar la distribución de una variable numérica y cómo esta se distribuye a través de diferentes categorías de tiempo, ayudando a identificar outliers y patrones estacionales.

```python
# Crear datos de ventas diarias por día de la semana
np.random.seed(42)
dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

# Simular ventas con patrones semanales
ventas_lunes = np.random.normal(800, 150, 52)  # 52 semanas
ventas_martes = np.random.normal(850, 140, 52)
ventas_miercoles = np.random.normal(900, 130, 52)
ventas_jueves = np.random.normal(950, 120, 52)
ventas_viernes = np.random.normal(1100, 100, 52)  # Viernes más alto
ventas_sabado = np.random.normal(1200, 80, 52)   # Sábado más alto
ventas_domingo = np.random.normal(600, 200, 52)  # Domingo más bajo

# Agregar algunos outliers
ventas_lunes[0] = 1500  # Outlier alto
ventas_domingo[10] = 50  # Outlier bajo

# Gráfico de boxplot
fig, ax = plt.subplots(figsize=(12, 8))
data = [ventas_lunes, ventas_martes, ventas_miercoles, ventas_jueves, 
        ventas_viernes, ventas_sabado, ventas_domingo]

box_plot = ax.boxplot(data, labels=dias_semana, patch_artist=True)

# Colorear las cajas
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 
          'lightpink', 'lightsteelblue', 'lightgray']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)

ax.set_ylabel('Ventas Diarias ($)', fontsize=14)
ax.set_title('Distribución de Ventas por Día de la Semana', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Boxplot con datos mensuales
# Crear datos de ventas mensuales con estacionalidad
meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

# Simular 3 años de datos con estacionalidad
ventas_mensuales = []
for mes in range(12):
    # Patrón estacional: más ventas en verano (jun-ago) y navidad (dic)
    if mes in [5, 6, 7]:  # Verano
        base_ventas = 1200
    elif mes == 11:  # Diciembre (navidad)
        base_ventas = 1500
    else:
        base_ventas = 800
    
    # Agregar variabilidad y tendencia
    ventas_mes = np.random.normal(base_ventas, base_ventas * 0.2, 3)  # 3 años
    ventas_mensuales.append(ventas_mes)

fig, ax = plt.subplots(figsize=(15, 8))
box_plot = ax.boxplot(ventas_mensuales, labels=meses_nombres, patch_artist=True)

# Colorear por estación
colores_estacion = ['lightblue']*2 + ['lightgreen']*3 + ['lightcoral']*3 + ['lightyellow']*3 + ['lightblue']
for patch, color in zip(box_plot['boxes'], colores_estacion):
    patch.set_facecolor(color)

ax.set_ylabel('Ventas Mensuales ($)', fontsize=14)
ax.set_title('Distribución de Ventas Mensuales (3 años)', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Agregar anotaciones para estaciones
ax.annotate('Invierno', xy=(1.5, 1600), xytext=(1.5, 1600), 
            fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
ax.annotate('Primavera', xy=(5, 1600), xytext=(5, 1600), 
            fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
ax.annotate('Verano', xy=(8, 1600), xytext=(8, 1600), 
            fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
ax.annotate('Otoño', xy=(11, 1600), xytext=(11, 1600), 
            fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

plt.tight_layout()
plt.show()
```

### Dashboard Completo de Análisis Temporal

```python
# Crear un dashboard completo con múltiples visualizaciones
fig = plt.figure(figsize=(20, 12))
fig.suptitle('Dashboard de Análisis de Series de Tiempo', fontsize=20, fontweight='bold')

# Gráfico 1: Línea temporal (2x2, posición 1)
ax1 = plt.subplot(2, 3, 1)
ax1.plot(df_ventas['fecha'], df_ventas['ventas'], linewidth=2, color='blue')
ax1.set_title('Evolución de Ventas')
ax1.set_ylabel('Ventas ($)')
ax1.grid(True, alpha=0.3)
plt.setp(ax1.get_xticklabels(), rotation=45)

# Gráfico 2: Dispersión (2x2, posición 2)
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(temperaturas, ventas_helados, alpha=0.7, c=temperaturas, cmap='viridis')
ax2.set_title('Temperatura vs Ventas')
ax2.set_xlabel('Temperatura (°C)')
ax2.set_ylabel('Ventas ($)')
ax2.grid(True, alpha=0.3)

# Gráfico 3: Barras (2x2, posición 3)
ax3 = plt.subplot(2, 3, 3)
x = np.arange(len(meses))
ax3.bar(x, precipitaciones_2024, alpha=0.8, color='lightblue')
ax3.set_title('Precipitaciones 2024')
ax3.set_ylabel('Precipitaciones (mm)')
ax3.set_xticks(x)
ax3.set_xticklabels(meses, rotation=45)
ax3.grid(True, alpha=0.3, axis='y')

# Gráfico 4: Histograma (2x2, posición 4)
ax4 = plt.subplot(2, 3, 4)
ax4.hist(todas_temperaturas, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax4.set_title('Distribución de Temperaturas')
ax4.set_xlabel('Temperatura (°C)')
ax4.set_ylabel('Frecuencia')
ax4.grid(True, alpha=0.3)

# Gráfico 5: Boxplot (2x2, posición 5)
ax5 = plt.subplot(2, 3, 5)
box_plot = ax5.boxplot(data, labels=dias_semana, patch_artist=True)
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 
          'lightpink', 'lightsteelblue', 'lightgray']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
ax5.set_title('Ventas por Día de la Semana')
ax5.set_ylabel('Ventas ($)')
ax5.grid(True, alpha=0.3, axis='y')
plt.setp(ax5.get_xticklabels(), rotation=45)

# Gráfico 6: Resumen estadístico (2x2, posición 6)
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
stats_text = f"""
RESUMEN ESTADÍSTICO

Ventas Anuales:
• Total: ${df_ventas['ventas'].sum():,.0f}
• Promedio: ${df_ventas['ventas'].mean():.0f}
• Máximo: ${df_ventas['ventas'].max():.0f}
• Mínimo: ${df_ventas['ventas'].min():.0f}

Temperaturas:
• Promedio: {np.mean(todas_temperaturas):.1f}°C
• Desv. Est.: {np.std(todas_temperaturas):.1f}°C

Correlación Temp-Ventas:
• Coeficiente: {np.corrcoef(temperaturas, ventas_helados)[0,1]:.3f}
"""
ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=12, 
         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

plt.tight_layout()
plt.show()
```

### Mejores Prácticas para Gráficos de Series de Tiempo

1. **Siempre usar el eje X para el tiempo** y mantener la escala temporal consistente
2. **Agregar líneas de tendencia** para identificar patrones a largo plazo
3. **Usar colores consistentes** para diferentes series de datos
4. **Incluir leyendas claras** cuando se muestren múltiples series
5. **Considerar la estacionalidad** y agregar marcadores para eventos importantes
6. **Usar gráficos complementarios** (líneas + dispersión) para análisis completo
7. **Agregar anotaciones** para puntos importantes o outliers
8. **Mantener la simplicidad** - no sobrecargar con demasiada información

### Conclusión

El análisis de datos secuenciales y series de tiempo es crucial para comprender la evolución de variables y tomar decisiones informadas. Los gráficos de líneas, dispersión, barras, histogramas y boxplots son herramientas esenciales, cada una con su aplicación específica:

- **Líneas**: Para tendencias y evolución temporal
- **Dispersión**: Para relaciones entre variables
- **Barras**: Para comparaciones entre períodos
- **Histogramas**: Para distribución de valores
- **Boxplots**: Para análisis de outliers y distribución por categorías

La combinación de estas visualizaciones permite un análisis completo y profundo de los datos temporales.

---

## **4.6 Gráficos de Dispersión, de Barras e Histogramas - Análisis Detallado**

### Introducción a los Gráficos de Dispersión

**¿Qué son los gráficos de dispersión?**
Un gráfico de dispersión es una herramienta fundamental en el análisis de datos que permite visualizar la relación entre dos variables numéricas. Cada punto en el gráfico representa una observación individual, donde la posición del punto está determinada por los valores de las dos variables que se están comparando.

### Utilidad del Gráfico de Dispersión

**1. Identificación de Relaciones:**
Los gráficos de dispersión son particularmente útiles para identificar si existe una relación entre las dos variables. Por ejemplo, se puede observar si a medida que una variable aumenta, la otra también lo hace (relación positiva), o si, por el contrario, una variable disminuye mientras la otra aumenta (relación negativa).

**2. Detección de Patrones:**
Estos gráficos permiten detectar patrones en los datos, como tendencias lineales, no lineales, cúmulos de datos o la presencia de outliers.

**3. Evaluación de la Fuerza de la Relación:**
La dispersión o concentración de los puntos en el gráfico también da una idea de la fuerza de la relación entre las variables. Una nube de puntos muy dispersa indica una relación débil o nula, mientras que una nube alineada sugiere una relación fuerte.

### Creación de Gráficos de Dispersión con Matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configuración de estilo
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Gráfico de Dispersión Básico
print("=== GRÁFICO DE DISPERSIÓN BÁSICO ===")

# Preparar los datos
np.random.seed(42)
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

# Crear el gráfico de dispersión básico
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, color='blue', s=100, alpha=0.7)
ax.set_title('Gráfico de Dispersión Básico', fontsize=16, fontweight='bold')
ax.set_xlabel('Variable X', fontsize=14)
ax.set_ylabel('Variable Y', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Gráfico de Dispersión con Personalización Avanzada
print("\n=== GRÁFICO DE DISPERSIÓN PERSONALIZADO ===")

# Crear datos más realistas
np.random.seed(42)
n_points = 100
x = np.random.normal(50, 15, n_points)
y = 2 * x + 10 + np.random.normal(0, 20, n_points)  # Relación lineal con ruido

fig, ax = plt.subplots(figsize=(12, 8))

# Gráfico de dispersión con personalización
scatter = ax.scatter(x, y, 
                    c=y,  # Color basado en valores de y
                    s=100,  # Tamaño de los puntos
                    alpha=0.6,  # Transparencia
                    cmap='viridis',  # Mapa de colores
                    edgecolors='black',  # Borde de los puntos
                    linewidth=0.5)

# Agregar línea de tendencia
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label='Línea de Tendencia')

# Agregar barra de color
cbar = plt.colorbar(scatter)
cbar.set_label('Valor de Y', fontsize=12)

# Personalizar el gráfico
ax.set_title('Gráfico de Dispersión con Línea de Tendencia', fontsize=16, fontweight='bold')
ax.set_xlabel('Variable X', fontsize=14)
ax.set_ylabel('Variable Y', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Agregar anotación con coeficiente de correlación
correlation = np.corrcoef(x, y)[0, 1]
ax.annotate(f'Correlación: {correlation:.3f}', 
            xy=(0.05, 0.95), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            fontsize=12)

plt.tight_layout()
plt.show()

# 3. Múltiples Gráficos de Dispersión - Diferentes Tipos de Relaciones
print("\n=== DIFERENTES TIPOS DE RELACIONES ===")

# Crear datos con diferentes tipos de relaciones
np.random.seed(42)

# Relación positiva fuerte
x1 = np.random.uniform(0, 10, 50)
y1 = 2 * x1 + np.random.normal(0, 0.5, 50)

# Relación negativa
x2 = np.random.uniform(0, 10, 50)
y2 = -1.5 * x2 + 15 + np.random.normal(0, 1, 50)

# Sin relación (ruido)
x3 = np.random.uniform(0, 10, 50)
y3 = np.random.normal(5, 2, 50)

# Relación no lineal (cuadrática)
x4 = np.random.uniform(-3, 3, 50)
y4 = x4**2 + np.random.normal(0, 0.5, 50)

# Crear subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Diferentes Tipos de Relaciones en Gráficos de Dispersión', fontsize=16, fontweight='bold')

# Gráfico 1: Relación positiva fuerte
ax1.scatter(x1, y1, alpha=0.7, color='blue', s=50)
z1 = np.polyfit(x1, y1, 1)
p1 = np.poly1d(z1)
ax1.plot(x1, p1(x1), "r--", alpha=0.8)
ax1.set_title('Relación Positiva Fuerte')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.grid(True, alpha=0.3)
corr1 = np.corrcoef(x1, y1)[0, 1]
ax1.text(0.05, 0.95, f'r = {corr1:.3f}', transform=ax1.transAxes, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

# Gráfico 2: Relación negativa
ax2.scatter(x2, y2, alpha=0.7, color='red', s=50)
z2 = np.polyfit(x2, y2, 1)
p2 = np.poly1d(z2)
ax2.plot(x2, p2(x2), "r--", alpha=0.8)
ax2.set_title('Relación Negativa')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.grid(True, alpha=0.3)
corr2 = np.corrcoef(x2, y2)[0, 1]
ax2.text(0.05, 0.95, f'r = {corr2:.3f}', transform=ax2.transAxes,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))

# Gráfico 3: Sin relación
ax3.scatter(x3, y3, alpha=0.7, color='green', s=50)
ax3.set_title('Sin Relación (Ruido)')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.grid(True, alpha=0.3)
corr3 = np.corrcoef(x3, y3)[0, 1]
ax3.text(0.05, 0.95, f'r = {corr3:.3f}', transform=ax3.transAxes,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

# Gráfico 4: Relación no lineal
ax4.scatter(x4, y4, alpha=0.7, color='purple', s=50)
# Ajuste polinomial de grado 2
z4 = np.polyfit(x4, y4, 2)
p4 = np.poly1d(z4)
x4_sorted = np.sort(x4)
ax4.plot(x4_sorted, p4(x4_sorted), "r--", alpha=0.8, label='Ajuste Cuadrático')
ax4.set_title('Relación No Lineal (Cuadrática)')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.legend()
ax4.grid(True, alpha=0.3)
corr4 = np.corrcoef(x4, y4)[0, 1]
ax4.text(0.05, 0.95, f'r = {corr4:.3f}', transform=ax4.transAxes,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="plum", alpha=0.7))

plt.tight_layout()
plt.show()

# 4. Gráfico de Dispersión con Categorías
print("\n=== GRÁFICO DE DISPERSIÓN CON CATEGORÍAS ===")

# Crear datos con categorías
np.random.seed(42)
categorias = ['A', 'B', 'C']
colores = ['red', 'blue', 'green']

# Generar datos para cada categoría
datos_categorizados = {}
for i, cat in enumerate(categorias):
    n_points = 30
    x_cat = np.random.normal(50 + i*20, 10, n_points)
    y_cat = 1.5 * x_cat + np.random.normal(0, 15, n_points)
    datos_categorizados[cat] = {'x': x_cat, 'y': y_cat}

fig, ax = plt.subplots(figsize=(12, 8))

# Crear gráfico de dispersión por categorías
for i, (cat, color) in enumerate(zip(categorias, colores)):
    x_cat = datos_categorizados[cat]['x']
    y_cat = datos_categorizados[cat]['y']
    ax.scatter(x_cat, y_cat, c=color, s=80, alpha=0.7, label=f'Categoría {cat}', edgecolors='black', linewidth=0.5)

ax.set_title('Gráfico de Dispersión por Categorías', fontsize=16, fontweight='bold')
ax.set_xlabel('Variable X', fontsize=14)
ax.set_ylabel('Variable Y', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5. Análisis de Outliers en Gráficos de Dispersión
print("\n=== ANÁLISIS DE OUTLIERS ===")

# Crear datos con outliers
np.random.seed(42)
x_normal = np.random.normal(50, 10, 95)
y_normal = 2 * x_normal + np.random.normal(0, 5, 95)

# Agregar outliers
x_outliers = np.array([20, 80, 90, 10, 85])
y_outliers = np.array([120, 30, 180, 5, 200])

x_completo = np.concatenate([x_normal, x_outliers])
y_completo = np.concatenate([y_normal, y_outliers])

fig, ax = plt.subplots(figsize=(12, 8))

# Gráfico de dispersión con outliers
ax.scatter(x_normal, y_normal, c='blue', s=60, alpha=0.7, label='Datos Normales')
ax.scatter(x_outliers, y_outliers, c='red', s=100, alpha=0.8, label='Outliers', edgecolors='black', linewidth=1)

# Línea de tendencia sin outliers
z_normal = np.polyfit(x_normal, y_normal, 1)
p_normal = np.poly1d(z_normal)
ax.plot(x_normal, p_normal(x_normal), "b--", alpha=0.8, linewidth=2, label='Tendencia (sin outliers)')

# Línea de tendencia con outliers
z_completo = np.polyfit(x_completo, y_completo, 1)
p_completo = np.poly1d(z_completo)
ax.plot(x_completo, p_completo(x_completo), "r--", alpha=0.8, linewidth=2, label='Tendencia (con outliers)')

ax.set_title('Impacto de Outliers en la Relación', fontsize=16, fontweight='bold')
ax.set_xlabel('Variable X', fontsize=14)
ax.set_ylabel('Variable Y', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Mostrar correlaciones
corr_normal = np.corrcoef(x_normal, y_normal)[0, 1]
corr_completo = np.corrcoef(x_completo, y_completo)[0, 1]

ax.text(0.05, 0.95, f'Sin outliers: r = {corr_normal:.3f}', transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
ax.text(0.05, 0.88, f'Con outliers: r = {corr_completo:.3f}', transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))

plt.tight_layout()
plt.show()
```

### Gráficos de Barras - Representación de Datos Categóricos

**¿Cuándo usar gráficos de barras?**
Los gráficos de barras son ideales para representar datos categóricos y comparar valores entre diferentes grupos o categorías. Son especialmente útiles para:
- Comparar frecuencias o cantidades entre categorías
- Mostrar rankings o clasificaciones
- Visualizar datos de encuestas o conteos
- Representar datos temporales discretos

```python
# 1. Gráfico de Barras Básico
print("=== GRÁFICO DE BARRAS BÁSICO ===")

# Datos de ejemplo: ventas por producto
productos = ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Auriculares', 'Tablet']
ventas = [45, 120, 80, 30, 95, 60]

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(productos, ventas, color='skyblue', alpha=0.8, edgecolor='black', linewidth=1)

# Personalizar el gráfico
ax.set_title('Ventas por Producto', fontsize=16, fontweight='bold')
ax.set_xlabel('Producto', fontsize=14)
ax.set_ylabel('Unidades Vendidas', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

# Agregar valores sobre las barras
for bar, venta in zip(bars, ventas):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{venta}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Gráfico de Barras Comparativo
print("\n=== GRÁFICO DE BARRAS COMPARATIVO ===")

# Datos de ventas por trimestre
trimestres = ['Q1', 'Q2', 'Q3', 'Q4']
ventas_2023 = [1200, 1350, 1100, 1500]
ventas_2024 = [1300, 1400, 1250, 1600]

fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(trimestres))
width = 0.35

bars1 = ax.bar(x - width/2, ventas_2023, width, label='2023', alpha=0.8, color='lightblue')
bars2 = ax.bar(x + width/2, ventas_2024, width, label='2024', alpha=0.8, color='lightcoral')

ax.set_xlabel('Trimestre', fontsize=14)
ax.set_ylabel('Ventas ($)', fontsize=14)
ax.set_title('Ventas por Trimestre - Comparación 2023 vs 2024', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(trimestres)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# Agregar valores sobre las barras
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'${height:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(bars1)
autolabel(bars2)

plt.tight_layout()
plt.show()

# 3. Gráfico de Barras Horizontales
print("\n=== GRÁFICO DE BARRAS HORIZONTALES ===")

# Datos de países por PIB
paises = ['Estados Unidos', 'China', 'Japón', 'Alemania', 'India', 'Reino Unido', 'Francia', 'Italia']
pib = [25462700, 17963170, 4231141, 4072191, 3385090, 3070667, 2782905, 2010430]

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(paises, pib, color='lightgreen', alpha=0.8, edgecolor='black', linewidth=1)

ax.set_title('PIB por País (Millones de USD)', fontsize=16, fontweight='bold')
ax.set_xlabel('PIB (Millones USD)', fontsize=14)
ax.set_ylabel('País', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')

# Agregar valores al final de las barras
for bar, valor in zip(bars, pib):
    width = bar.get_width()
    ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
            f'{valor:,.0f}', ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# 4. Gráfico de Barras Apiladas
print("\n=== GRÁFICO DE BARRAS APILADAS ===")

# Datos de ventas por categoría y región
categorias = ['Electrónicos', 'Ropa', 'Hogar', 'Deportes']
region_norte = [45000, 32000, 28000, 15000]
region_sur = [38000, 35000, 25000, 18000]
region_este = [42000, 30000, 30000, 12000]
region_oeste = [40000, 28000, 32000, 20000]

fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(categorias))
width = 0.2

ax.bar(x - width*1.5, region_norte, width, label='Norte', alpha=0.8, color='#2E86AB')
ax.bar(x - width*0.5, region_sur, width, label='Sur', alpha=0.8, color='#A23B72')
ax.bar(x + width*0.5, region_este, width, label='Este', alpha=0.8, color='#F18F01')
ax.bar(x + width*1.5, region_oeste, width, label='Oeste', alpha=0.8, color='#C73E1D')

ax.set_xlabel('Categoría de Producto', fontsize=14)
ax.set_ylabel('Ventas ($)', fontsize=14)
ax.set_title('Ventas por Categoría y Región', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categorias)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# 5. Gráfico de Barras con Error
print("\n=== GRÁFICO DE BARRAS CON ERROR ===")

# Datos con errores estándar
categorias = ['Grupo A', 'Grupo B', 'Grupo C', 'Grupo D']
medias = [75, 82, 68, 90]
errores = [5, 3, 7, 4]

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.bar(categorias, medias, yerr=errores, capsize=5, 
              color='lightblue', alpha=0.8, edgecolor='black', linewidth=1)

ax.set_title('Promedios por Grupo con Error Estándar', fontsize=16, fontweight='bold')
ax.set_xlabel('Grupo', fontsize=14)
ax.set_ylabel('Puntuación Promedio', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

# Agregar valores sobre las barras
for bar, media, error in zip(bars, medias, errores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + error + 1,
            f'{media}±{error}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
```

### Histogramas - Visualización de Distribuciones

**¿Cuándo usar histogramas?**
Los histogramas son ideales para visualizar la distribución de una variable numérica continua. Son especialmente útiles para:
- Entender la forma de la distribución de datos
- Identificar patrones como simetría, sesgo o bimodalidad
- Detectar outliers o valores atípicos
- Comparar distribuciones entre grupos

```python
# 1. Histograma Básico
print("=== HISTOGRAMA BÁSICO ===")

# Generar datos de ejemplo
np.random.seed(42)
datos = np.random.normal(100, 15, 1000)  # Distribución normal

fig, ax = plt.subplots(figsize=(12, 8))
n, bins, patches = ax.hist(datos, bins=30, alpha=0.7, color='skyblue', 
                          edgecolor='black', linewidth=1)

ax.set_title('Distribución de Datos - Histograma Básico', fontsize=16, fontweight='bold')
ax.set_xlabel('Valor', fontsize=14)
ax.set_ylabel('Frecuencia', fontsize=14)
ax.grid(True, alpha=0.3)

# Agregar estadísticas
media = np.mean(datos)
mediana = np.median(datos)
desv_std = np.std(datos)

ax.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Media: {media:.2f}')
ax.axvline(mediana, color='green', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.2f}')
ax.legend(fontsize=12)

plt.tight_layout()
plt.show()

# 2. Histograma con Curva de Densidad
print("\n=== HISTOGRAMA CON CURVA DE DENSIDAD ===")

fig, ax = plt.subplots(figsize=(12, 8))

# Histograma normalizado
n, bins, patches = ax.hist(datos, bins=30, alpha=0.7, color='lightblue', 
                          edgecolor='black', density=True, label='Histograma')

# Curva de densidad normal
from scipy.stats import norm
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, media, desv_std)
ax.plot(x, p, 'r-', linewidth=2, label='Distribución Normal Teórica')

ax.set_title('Histograma con Curva de Densidad', fontsize=16, fontweight='bold')
ax.set_xlabel('Valor', fontsize=14)
ax.set_ylabel('Densidad de Probabilidad', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 3. Comparación de Histogramas
print("\n=== COMPARACIÓN DE HISTOGRAMAS ===")

# Generar datos de diferentes distribuciones
np.random.seed(42)
normal_data = np.random.normal(100, 15, 500)
uniform_data = np.random.uniform(80, 120, 500)
skewed_data = np.random.exponential(20, 500) + 80

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Comparación de Diferentes Distribuciones', fontsize=16, fontweight='bold')

# Histograma 1: Distribución Normal
ax1.hist(normal_data, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
ax1.set_title('Distribución Normal')
ax1.set_xlabel('Valor')
ax1.set_ylabel('Frecuencia')
ax1.grid(True, alpha=0.3)
ax1.axvline(np.mean(normal_data), color='red', linestyle='--', linewidth=2)

# Histograma 2: Distribución Uniforme
ax2.hist(uniform_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
ax2.set_title('Distribución Uniforme')
ax2.set_xlabel('Valor')
ax2.set_ylabel('Frecuencia')
ax2.grid(True, alpha=0.3)
ax2.axvline(np.mean(uniform_data), color='red', linestyle='--', linewidth=2)

# Histograma 3: Distribución Sesgada
ax3.hist(skewed_data, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
ax3.set_title('Distribución Sesgada')
ax3.set_xlabel('Valor')
ax3.set_ylabel('Frecuencia')
ax3.grid(True, alpha=0.3)
ax3.axvline(np.mean(skewed_data), color='red', linestyle='--', linewidth=2)

plt.tight_layout()
plt.show()

# 4. Histograma con Múltiples Grupos
print("\n=== HISTOGRAMA CON MÚLTIPLES GRUPOS ===")

# Datos de edades por género
np.random.seed(42)
edades_hombres = np.random.normal(35, 8, 300)
edades_mujeres = np.random.normal(32, 7, 300)

fig, ax = plt.subplots(figsize=(12, 8))

# Histogramas superpuestos
ax.hist(edades_hombres, bins=25, alpha=0.7, color='lightblue', 
        edgecolor='black', label='Hombres', density=True)
ax.hist(edades_mujeres, bins=25, alpha=0.7, color='lightcoral', 
        edgecolor='black', label='Mujeres', density=True)

ax.set_title('Distribución de Edades por Género', fontsize=16, fontweight='bold')
ax.set_xlabel('Edad', fontsize=14)
ax.set_ylabel('Densidad de Probabilidad', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Agregar estadísticas
ax.axvline(np.mean(edades_hombres), color='blue', linestyle='--', 
           linewidth=2, label=f'Hombres - Media: {np.mean(edades_hombres):.1f}')
ax.axvline(np.mean(edades_mujeres), color='red', linestyle='--', 
           linewidth=2, label=f'Mujeres - Media: {np.mean(edades_mujeres):.1f}')

plt.tight_layout()
plt.show()

# 5. Histograma 2D (Scatter Plot con Densidad)
print("\n=== HISTOGRAMA 2D (SCATTER PLOT CON DENSIDAD) ===")

# Generar datos correlacionados
np.random.seed(42)
x_2d = np.random.normal(0, 1, 1000)
y_2d = 0.7 * x_2d + np.random.normal(0, 0.5, 1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico de dispersión
ax1.scatter(x_2d, y_2d, alpha=0.6, s=20)
ax1.set_title('Gráfico de Dispersión')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.grid(True, alpha=0.3)

# Histograma 2D
ax2.hist2d(x_2d, y_2d, bins=30, cmap='Blues', alpha=0.8)
ax2.set_title('Histograma 2D')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

# Agregar barra de color
cbar = plt.colorbar(ax2.collections[0], ax=ax2)
cbar.set_label('Frecuencia')

plt.tight_layout()
plt.show()
```

### Dashboard Integrado - Análisis Completo

```python
# Dashboard que combina todos los tipos de gráficos
print("=== DASHBOARD INTEGRADO ===")

# Crear datos de ejemplo para el dashboard
np.random.seed(42)
n_points = 200

# Datos para dispersión
x_scatter = np.random.normal(50, 15, n_points)
y_scatter = 1.5 * x_scatter + np.random.normal(0, 10, n_points)

# Datos para barras
categorias = ['A', 'B', 'C', 'D', 'E']
valores = [45, 78, 32, 91, 56]

# Datos para histograma
datos_hist = np.random.normal(100, 20, 500)

# Crear dashboard
fig = plt.figure(figsize=(20, 12))
fig.suptitle('Dashboard de Análisis de Datos - Gráficos de Dispersión, Barras e Histogramas', 
             fontsize=18, fontweight='bold')

# Gráfico 1: Dispersión
ax1 = plt.subplot(2, 3, 1)
scatter = ax1.scatter(x_scatter, y_scatter, c=y_scatter, cmap='viridis', alpha=0.7, s=50)
ax1.set_title('Gráfico de Dispersión')
ax1.set_xlabel('Variable X')
ax1.set_ylabel('Variable Y')
ax1.grid(True, alpha=0.3)

# Línea de tendencia
z = np.polyfit(x_scatter, y_scatter, 1)
p = np.poly1d(z)
ax1.plot(x_scatter, p(x_scatter), "r--", alpha=0.8, linewidth=2)

# Gráfico 2: Barras
ax2 = plt.subplot(2, 3, 2)
bars = ax2.bar(categorias, valores, color='lightblue', alpha=0.8, edgecolor='black')
ax2.set_title('Gráfico de Barras')
ax2.set_xlabel('Categoría')
ax2.set_ylabel('Valor')
ax2.grid(True, alpha=0.3, axis='y')

# Agregar valores sobre las barras
for bar, valor in zip(bars, valores):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{valor}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Gráfico 3: Histograma
ax3 = plt.subplot(2, 3, 3)
n, bins, patches = ax3.hist(datos_hist, bins=30, alpha=0.7, color='lightgreen', 
                           edgecolor='black', density=True)
ax3.set_title('Histograma')
ax3.set_xlabel('Valor')
ax3.set_ylabel('Densidad')
ax3.grid(True, alpha=0.3)

# Curva de densidad
from scipy.stats import norm
xmin, xmax = ax3.get_xlim()
x = np.linspace(xmin, xmax, 100)
p_hist = norm.pdf(x, np.mean(datos_hist), np.std(datos_hist))
ax3.plot(x, p_hist, 'r-', linewidth=2)

# Gráfico 4: Dispersión con categorías
ax4 = plt.subplot(2, 3, 4)
categorias_scatter = np.random.choice(['X', 'Y', 'Z'], n_points)
colores_cat = {'X': 'red', 'Y': 'blue', 'Z': 'green'}

for cat in ['X', 'Y', 'Z']:
    mask = np.array(categorias_scatter) == cat
    ax4.scatter(x_scatter[mask], y_scatter[mask], c=colores_cat[cat], 
                label=f'Categoría {cat}', alpha=0.7, s=50)

ax4.set_title('Dispersión por Categorías')
ax4.set_xlabel('Variable X')
ax4.set_ylabel('Variable Y')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Gráfico 5: Barras comparativas
ax5 = plt.subplot(2, 3, 5)
valores_1 = [45, 78, 32, 91, 56]
valores_2 = [52, 71, 38, 85, 62]

x = np.arange(len(categorias))
width = 0.35

ax5.bar(x - width/2, valores_1, width, label='Grupo 1', alpha=0.8, color='lightblue')
ax5.bar(x + width/2, valores_2, width, label='Grupo 2', alpha=0.8, color='lightcoral')

ax5.set_title('Barras Comparativas')
ax5.set_xlabel('Categoría')
ax5.set_ylabel('Valor')
ax5.set_xticks(x)
ax5.set_xticklabels(categorias)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Gráfico 6: Resumen estadístico
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Calcular estadísticas
corr_scatter = np.corrcoef(x_scatter, y_scatter)[0, 1]
stats_text = f"""
RESUMEN ESTADÍSTICO

Datos de Dispersión:
• Correlación: {corr_scatter:.3f}
• Puntos: {len(x_scatter)}

Datos de Barras:
• Total: {sum(valores)}
• Promedio: {np.mean(valores):.1f}
• Máximo: {max(valores)}

Datos de Histograma:
• Media: {np.mean(datos_hist):.1f}
• Mediana: {np.median(datos_hist):.1f}
• Desv. Est.: {np.std(datos_hist):.1f}
"""

ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
         facecolor="lightgray", alpha=0.8))

plt.tight_layout()
plt.show()
```

### Mejores Prácticas para Cada Tipo de Gráfico

#### **Gráficos de Dispersión:**
1. **Usar colores para representar una tercera variable** cuando sea relevante
2. **Agregar líneas de tendencia** para mostrar patrones
3. **Incluir coeficientes de correlación** para cuantificar la relación
4. **Usar transparencia (alpha)** cuando hay muchos puntos
5. **Identificar y marcar outliers** cuando sea importante

#### **Gráficos de Barras:**
1. **Ordenar las barras** por valor cuando sea apropiado
2. **Agregar valores numéricos** sobre las barras
3. **Usar colores consistentes** para categorías
4. **Considerar barras horizontales** para etiquetas largas
5. **Incluir barras de error** cuando sea relevante

#### **Histogramas:**
1. **Elegir el número correcto de bins** (no muy pocos, no muy muchos)
2. **Agregar curvas de densidad** para comparar con distribuciones teóricas
3. **Mostrar estadísticas clave** (media, mediana, desviación estándar)
4. **Usar densidad en lugar de frecuencia** para comparar grupos de diferentes tamaños
5. **Considerar histogramas 2D** para relaciones entre dos variables

### Conclusión

Los gráficos de dispersión, barras e histogramas son herramientas fundamentales en el análisis exploratorio de datos:

- **Gráficos de dispersión**: Ideales para identificar relaciones entre variables numéricas
- **Gráficos de barras**: Perfectos para comparar valores entre categorías
- **Histogramas**: Esenciales para entender la distribución de variables numéricas

La combinación de estos tres tipos de gráficos permite un análisis completo y profundo de los datos, desde relaciones entre variables hasta patrones de distribución y comparaciones categóricas.


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
