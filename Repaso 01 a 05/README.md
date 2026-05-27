# Repaso Clases 01 a 05 — Guía de Clase

Archivo de referencia para seguir el notebook `repaso1a5.ipynb` durante la clase.

---

## Índice

1. [Python y NumPy — Fundamentos](#1-python-y-numpy--fundamentos)
2. [Pandas — Carga y Análisis de Datos](#2-pandas--carga-y-análisis-de-datos)
3. [Limpieza de Datos — Strings y NaN](#3-limpieza-de-datos--strings-y-nan)
4. [Visualizaciones — Matplotlib y Seaborn](#4-visualizaciones--matplotlib-y-seaborn)
5. [Guía de Gráficos](#5-guía-de-gráficos)

---

## 1. Python y NumPy — Fundamentos

### Tipos de datos más usados en Data Science

```python
ventas_semana = [4500, 3200, 5100, 4800, 6000]   # list: ordenada, mutable
producto = {'nombre': 'Laptop', 'precio': 1299.99} # dict: clave-valor
```

> **Lista** → base de arrays y Series. **Dict** → base de DataFrames y JSON.

### Funciones y list comprehensions

```python
def clasificar_venta(valor):
    if valor >= 5000:   return 'Alta'
    elif valor >= 3500: return 'Media'
    else:               return 'Baja'

categorias = [clasificar_venta(v) for v in ventas_semana]
```

> List comprehension = `for` + `if` en una línea. Más legible y ligeramente más rápido.

### Por qué NumPy y no listas de Python

| | Lista Python | NumPy array |
|--|--|--|
| Almacenamiento | Punteros a objetos | Bloque de memoria fija (tipo C) |
| Operación en 1M datos | ~1 seg | ~0.01 seg |
| Sintaxis | `[x*1.21 for x in lista]` | `array * 1.21` |

### NumPy — Operaciones esenciales

```python
import numpy as np

ventas = np.array([4500, 3200, 5100, 4800, 6000], dtype=float)

ventas * 1.21                        # + IVA a todos los elementos
ventas[ventas > 4500]                # filtrado booleano

ventas.mean()                        # media
np.median(ventas)                    # mediana
ventas.std()                         # desvío estándar
np.percentile(ventas, 75)            # percentil 75
```

### Arrays 2D (matrices)

```python
ventas_2d = np.array([
    [4500, 3200, 5100],   # Region Norte
    [2800, 4100, 3600],   # Region Sur
])

ventas_2d.shape           # (2, 3) → filas x columnas
ventas_2d.sum(axis=1)     # total por fila (por región)
ventas_2d.mean(axis=0)    # promedio por columna (por producto)
```

> **axis=0** → colapsa filas (resultado por columna). **axis=1** → colapsa columnas (resultado por fila).

---

## 2. Pandas — Carga y Análisis de Datos

### Por qué Pandas

| vs. | Ventaja Pandas |
|-----|----------------|
| Excel | Millones de filas, automatización, integración con ML |
| SQL | En memoria, sin servidor, fácil de combinar con gráficos |
| NumPy | Nombres de columna, índices, NaN nativo, operaciones de alto nivel |

### Los 3 comandos obligatorios al cargar un dataset

```python
import pandas as pd

df = pd.read_csv('ventas.csv', encoding='utf-8')

print(df.shape)       # (filas, columnas)
print(df.head())      # primeras 5 filas
print(df.dtypes)      # tipo de cada columna
```

### Exploración rápida

```python
df.describe()          # estadísticas de columnas numéricas
df['Region'].unique()  # valores únicos de una columna
```

> `count < total filas` en describe() → hay NaN. `mean` muy distinta de `median` → sesgo u outliers.

### Filtrado booleano

```python
# Condición simple
df[df['Ventas'] > 4000]

# Condición compuesta — cada parte entre paréntesis
norte_alta = df[(df['Region'] == 'Norte') & (df['Ventas'] > 4000)]
```

### GroupBy — Split, Apply, Combine

```python
resumen = df.groupby('Region').agg(
    ventas_total    = ('Ventas', 'sum'),
    ventas_promedio = ('Ventas', 'mean'),
    satisfaccion    = ('Satisfaccion_Cliente', 'mean'),
).round(2)
```

> Equivalente al `GROUP BY` de SQL y las tablas dinámicas de Excel.

### Pivot table y columnas calculadas

```python
# Tabla cruzada
tabla = pd.pivot_table(df, values='Ventas', index='Region',
                       columns='Producto', aggfunc='mean')

# Columna nueva a partir de columnas existentes
df['Margen']     = df['Ventas'] - df['Costo_Operativo']
df['Margen_Pct'] = (df['Margen'] / df['Ventas'] * 100).round(2)
```

---

## 3. Limpieza de Datos — Strings y NaN

> **GIGO — Garbage In, Garbage Out:** un modelo entrenado con datos sucios aprende los errores.

### Problemas más comunes

- Strings con espacios o mayúsculas inconsistentes (`'norte'`, `'NORTE'`, `'  norte '`)
- Valores faltantes (`NaN`)
- Filas duplicadas

### Limpiar strings con `.str`

```python
df['Region'] = df['Region'].str.strip().str.title()
# 'NORTE', '  norte ', 'Norte' → todos quedan como 'Norte'

df = df.drop_duplicates().reset_index(drop=True)
```

> `.str.strip()` elimina espacios. `.str.title()` capitaliza la primera letra. Sin esto, un `groupby` los trata como 3 grupos distintos.

### Tratamiento de NaN

| Estrategia | Método | Cuándo usarla |
|------------|--------|---------------|
| Eliminar | `dropna()` | Pocos NaN y aleatorios |
| Media | `fillna(mean)` | Distribución simétrica, sin outliers |
| Mediana | `fillna(median)` | Hay valores extremos |
| Forward fill | `ffill()` | Series de tiempo: último valor conocido |
| Backward fill | `bfill()` | Series de tiempo: próximo valor conocido |

```python
# Imputación robusta ante outliers
df['Ventas'] = df['Ventas'].fillna(df['Ventas'].median())

# Para series de tiempo
df['col'].ffill()   # rellena hacia adelante
df['col'].bfill()   # rellena hacia atrás
```

---

## 4. Visualizaciones — Matplotlib y Seaborn

> **Regla práctica:** explorá con Seaborn, presentá con Matplotlib.

| Librería | Rol |
|----------|-----|
| Matplotlib | Control total, verbosa, ideal para el gráfico final |
| Seaborn | Abstracción sobre Matplotlib, gráficos estadísticos con una línea |

### Anatomía de una figura Matplotlib

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 5))   # crea figura y ejes

ax.plot(x, y)                              # dibuja datos

ax.set_title('Título', fontsize=14, fontweight='bold')
ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
ax.legend(title='Categoría')
ax.grid(True, linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)        # limpia el marco
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
```

### Lineplot — tendencias en el tiempo

```python
ax.plot(x, y, color='steelblue', marker='o', linewidth=2, markersize=7, label='Norte')
```

### Barras agrupadas con etiquetas

```python
bars = ax.bar(x, valores, width=0.2, color='#4C72B0')
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
            f'{b.get_height():.1f}', ha='center', va='bottom', fontsize=8)
```

### Histograma + KDE (Seaborn)

```python
sns.histplot(data=df, x='Ventas', hue='Region', kde=True, bins=20, alpha=0.45, ax=ax)
```

> `kde=True` agrega la curva de densidad. `hue=` colorea por grupo automáticamente.

### Boxplot y Violinplot

```python
sns.boxplot(data=df, x='Producto', y='Ventas', hue='Region', palette='Set2', ax=ax)
sns.violinplot(data=df, x='Region', y='Ventas', inner='quartile', ax=ax)
```

> **Boxplot:** caja = Q1–Q3, línea = mediana, bigotes = 1.5×IQR, puntos = outliers.  
> **Violinplot:** forma completa de la distribución (KDE rotado) + cuartiles.

### Heatmap de correlaciones

```python
corr = df[['Ventas', 'Satisfaccion_Cliente', 'Costo_Operativo', 'Margen']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
```

> **Pearson:** +1 = suben juntas, −1 = una sube y la otra baja, 0 = sin relación lineal.

### Scatter + regresión

```python
sns.scatterplot(data=df, x='Costo_Operativo', y='Ventas', hue='Region', alpha=0.7)
sns.regplot(data=df, x='Satisfaccion_Cliente', y='Ventas', ci=95)
```

> `ci=95` dibuja la banda de confianza al 95%. Banda angosta = relación lineal sólida.

---

## 5. Guía de Gráficos

### Qué gráfico usar según el dato

| Situación | Gráfico | Función |
|-----------|---------|---------|
| Distribución de una variable numérica | Histograma + KDE | `sns.histplot(kde=True)` |
| Comparar distribución entre grupos | Boxplot / Violinplot | `sns.boxplot` / `sns.violinplot` |
| Evolución en el tiempo | Lineplot | `ax.plot` / `sns.lineplot` |
| Comparar categorías | Barras | `ax.bar` / `sns.barplot` |
| Relación entre dos variables numéricas | Scatter | `sns.scatterplot` / `sns.regplot` |
| Relación entre todas las variables | Matriz de correlación | `sns.heatmap(df.corr())` |
| Proporción de un total | Pie chart | `ax.pie` |
| Distribución por pares de variables | Pairplot | `sns.pairplot` |

### Colores en Matplotlib

```python
color='steelblue'           # por nombre
color='#4C72B0'             # por hex (igual que HTML/CSS)
color=(0.3, 0.5, 0.8)       # por RGB normalizado (0 a 1)
color='b'                   # por letra: b, r, g, k, w
```

### Paletas en Seaborn (`palette=`)

| Paleta | Tipo | Mejor para |
|--------|------|------------|
| `'Set1'`, `'Set2'`, `'Set3'` | Categórica | Variables sin orden (regiones, categorías) |
| `'tab10'`, `'tab20'` | Categórica | Muchos grupos distintos |
| `'Blues'`, `'Greens'` | Secuencial | Una variable de bajo a alto |
| `'coolwarm'`, `'RdBu'` | Divergente | Valores positivos y negativos (correlaciones) |
| `'viridis'`, `'magma'` | Secuencial perceptual | Recomendadas para impresión en grises |

### Estilos globales de Matplotlib

```python
plt.style.use('seaborn-v0_8')    # fondo gris claro, grilla blanca
plt.style.use('ggplot')          # inspirado en R ggplot2
plt.style.use('dark_background') # fondo negro, para presentaciones
```

### Pairplot — explorar todas las relaciones de una vez

```python
cols = ['Ventas', 'Satisfaccion_Cliente', 'Costo_Operativo', 'Margen']
sns.pairplot(df[cols + ['Region']], hue='Region', diag_kind='kde',
             plot_kws={'alpha': 0.5}, palette='Set2')
```

> El gráfico más útil para el primer EDA de un dataset nuevo.

### Guardar un gráfico

```python
fig.savefig('grafico.png', dpi=150, bbox_inches='tight')  # PNG para web/presentación
fig.savefig('grafico.pdf', bbox_inches='tight')           # PDF vectorial sin pérdida
```

> `savefig()` debe ir **antes** de `plt.show()` — show() vacía la figura.

---

## Flujo completo del repaso

```
1. pd.read_csv()                            → ingestar datos
2. df.shape + df.head() + df.dtypes         → conocer la estructura
3. df.describe() + df.isnull().sum()        → detectar problemas
4. str.strip() + fillna() + drop_dupes()    → limpiar
5. groupby() + pivot_table() + columnas     → transformar
6. histplot / boxplot / heatmap / pairplot  → visualizar e interpretar
```
