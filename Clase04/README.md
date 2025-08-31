
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
