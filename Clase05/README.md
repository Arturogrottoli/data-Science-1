## Repaso Clase 3 y 4

# ======================================================
# Repaso Clase 3 y 4
# ======================================================

# ======================================================
# Clase 3 - Numpy
# ======================================================
# Numpy es una librería de Python para trabajar con arreglos
# y cálculos numéricos de forma rápida y eficiente.
# Es mucho más veloz que usar listas comunes de Python para operaciones matemáticas.

import numpy as np

# Crear un arreglo
a = np.array([1, 2, 3, 4, 5])

# Operaciones básicas
print("Suma total:", a.sum())
print("Promedio:", a.mean())
print("Array multiplicado por 2:", a * 2)


# ======================================================
# Clase 3 - Pandas
# ======================================================
# Pandas es una librería que facilita el manejo y análisis de datos
# en tablas (DataFrames). Permite leer datos desde archivos (CSV, Excel, SQL, etc.)
# y procesarlos de forma muy flexible.

import pandas as pd

# Crear un DataFrame
data = {
    "Producto": ["A", "B", "C"],
    "Precio": [100, 200, 150],
    "Cantidad": [5, 3, 8]
}

df = pd.DataFrame(data)

print(df)
print("Promedio de precios:", df["Precio"].mean())


# ======================================================
# Clase 4 - Visualizaciones con Matplotlib y Pandas
# ======================================================
# Matplotlib es una librería para hacer gráficos.
# Con pandas se integra fácilmente, ya que un DataFrame puede graficarse directamente.

import matplotlib.pyplot as plt

# Ejemplo 1: gráfico simple con pandas
# -------------------------------------
# Usamos un DataFrame con ventas mensuales
data = {"Mes": ["Enero", "Febrero", "Marzo", "Abril"],
        "Ventas": [250, 300, 400, 350]}
df = pd.DataFrame(data)

# Gráfico de línea
df.plot(x="Mes", y="Ventas", kind="line", marker="o", title="Ventas mensuales")
plt.show()


# Ejemplo 2: dos gráficos diferentes
# -------------------------------------

# 1. Gráfico de barras
df.plot(x="Mes", y="Ventas", kind="bar", color="skyblue", title="Ventas por Mes")
plt.show()

# 2. Gráfico circular (pie chart)
df.set_index("Mes")["Ventas"].plot(kind="pie", autopct="%1.1f%%", title="Distribución de ventas")
plt.ylabel("")  # ocultar eje y
plt.show()


# ======================================================
# Conceptos importantes a recordar:
# ======================================================
# - df.plot() permite graficar directamente desde pandas
# - kind="line", "bar", "pie", etc. define el tipo de gráfico
# - plt.show() muestra el gráfico en pantalla
# - Matplotlib permite personalizar colores, títulos, etiquetas, etc.

#5.1 CLASE NUEVA Introducción a la Visualización de Datos

# ------------------------------------------------------------
# Introducción a la Visualización de Datos
# ------------------------------------------------------------
# La visualización de datos es clave en ciencia de datos porque permite
# comunicar de manera clara los resultados de un análisis.
# Pero conlleva una gran responsabilidad:
# "Con un gran poder viene una gran responsabilidad".
# Esto significa que debemos ser cuidadosos y éticos, porque los gráficos
# influyen en la toma de decisiones.

# Un gráfico no se hace al azar: requiere preparación para que sea claro,
# preciso y fiel a los datos.

# ------------------------------------------------------------
# Elementos de un Buen Gráfico
# ------------------------------------------------------------
# 1. Título: claro y representativo
# 2. Etiquetas y unidades: explicar qué se mide (ejemplo: "Edad (años)")
# 3. Leyendas y categorías: distinguir variables o grupos

# Un gráfico con estos elementos es comprensible y comunica correctamente.

# ------------------------------------------------------------
# Diferencias entre Matplotlib y Seaborn
# ------------------------------------------------------------
# Matplotlib:
#   - Muy flexible y robusto
#   - Permite gráficos básicos y avanzados
#   - Sintaxis más extensa
#   - Maneja arrays y DataFrames
#
# Seaborn:
#   - Especializado en gráficos estadísticos
#   - Sintaxis más simple y legible
#   - Orientado a trabajar directamente con datasets
#   - Menos flexible que Matplotlib, pero más práctico para análisis rápidos

# ------------------------------------------------------------
# Ejemplo 1: Gráfico de barras con Matplotlib
# ------------------------------------------------------------
import matplotlib.pyplot as plt

categorias = ["A", "B", "C"]
valores = [10, 20, 15]

plt.bar(categorias, valores)
plt.title("Ejemplo Matplotlib")
plt.xlabel("Categorías")
plt.ylabel("Valores")
plt.show()

# ------------------------------------------------------------
# Ejemplo 2: Gráfico de barras con Seaborn
# ------------------------------------------------------------
import seaborn as sns
import pandas as pd

data = pd.DataFrame({
    "Categoría": ["A", "B", "C"],
    "Valor": [10, 20, 15]
})

sns.barplot(x="Categoría", y="Valor", data=data)
plt.title("Ejemplo Seaborn")
plt.show()

#5.2 - Tipos de graficos

# ------------------------------------------------------------
# Tipos de Gráficos
# ------------------------------------------------------------
# Los gráficos son herramientas clave para comunicar información.
# Cada tipo tiene un propósito distinto según los datos que queremos mostrar.

# ------------------------------------------------------------
# Gráficos de Barras
# ------------------------------------------------------------
# - Sirven para comparar cantidades entre categorías.
# - Las barras pueden ser verticales u horizontales.
# - No son adecuados para datos que cambian con el tiempo (usar líneas).
# - Recomendación: etiquetas claras y diferencias visibles.

import matplotlib.pyplot as plt

categorias = ["A", "B", "C"]
valores = [5, 10, 7]

plt.bar(categorias, valores)
plt.title("Gráfico de Barras")
plt.xlabel("Categorías")
plt.ylabel("Valores")
plt.show()

# ------------------------------------------------------------
# Histogramas
# ------------------------------------------------------------
# - Representan la distribución de una variable numérica.
# - Agrupan los datos en intervalos (bins).
# - Son útiles para ver la forma de los datos (normal, sesgo, etc.).

import numpy as np

datos = np.random.randn(1000)  # 1000 valores aleatorios distribuidos normal

plt.hist(datos, bins=20, edgecolor="black")
plt.title("Histograma")
plt.xlabel("Valor")
plt.ylabel("Frecuencia")
plt.show()

# ------------------------------------------------------------
# Diagramas de Dispersión (Scatterplots)
# ------------------------------------------------------------
# - Muestran la relación entre dos variables.
# - Cada punto representa una observación.
# - Útiles para identificar patrones, tendencias o correlaciones.

x = np.random.rand(50)
y = 2 * x + np.random.randn(50) * 0.1  # relación con algo de ruido

plt.scatter(x, y)
plt.title("Diagrama de Dispersión")
plt.xlabel("Variable X")
plt.ylabel("Variable Y")
plt.show()

# ------------------------------------------------------------
# Gráficos de Torta (Pie Charts)
# ------------------------------------------------------------
# - Representan proporciones de un todo.
# - Útiles para mostrar porcentajes de pocas categorías.
# - No recomendados con muchas categorías porque se vuelven confusos.

etiquetas = ["A", "B", "C", "D"]
valores = [40, 30, 20, 10]

plt.pie(valores, labels=etiquetas, autopct="%1.1f%%")
plt.title("Gráfico de Torta")
plt.show()

# ------------------------------------------------------------
# Gráficos de Series Temporales
# ------------------------------------------------------------
# - Muestran cómo cambia una variable a lo largo del tiempo.
# - Ideales para detectar tendencias, ciclos o estacionalidad.
# - Útiles para comparar varias series en paralelo.

fechas = pd.date_range("2025-01-01", periods=10, freq="M")
valores = np.random.randint(50, 100, size=10)

plt.plot(fechas, valores, marker="o")
plt.title("Serie Temporal")
plt.xlabel("Tiempo")
plt.ylabel("Valor")
plt.xticks(rotation=45)
plt.show()

#5.3

# Visualizaciones con Matplotlib
[Diapositivas](https://docs.google.com/presentation/d/1BCmhYqiqKTKSm4hUkzXXuEcBJTAXw25j3oYJyqioTPQ/edit?slide=id.p1#slide=id.p1)

*Esta guía muestra cómo crear gráficos básicos con `matplotlib`, superponer líneas en gráficos de dispersión, combinar múltiples visualizaciones en una misma figura, y crear un gráfico de paleta (lollipop)*

## 1. Gráfico básico con Matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np

# Datos para el gráfico
t = np.linspace(0, 10, 100)  # 100 puntos entre 0 y 10
y = np.sin(t)  # Función seno

# Crear gráfico
plt.figure(figsize=(8, 4))
plt.plot(t, y, label='sin(t)', color='blue', linestyle='-', linewidth=2)
plt.title('Gráfico de la función seno')
plt.xlabel('Tiempo (t)')
plt.ylabel('Amplitud sin(t)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## 2. Superponer líneas sobre un Scatterplot

```python
np.random.seed(42)  # Para reproducibilidad
x = np.random.rand(50)  # 50 valores aleatorios entre 0 y 1
y = 2 * x + 1 + np.random.normal(scale=0.1, size=50)  # Datos con ruido

# Línea de regresión teórica
x_line = np.linspace(0, 1, 100)
y_line = 2 * x_line + 1

# Crear scatterplot con línea superpuesta
plt.figure(figsize=(6, 4))
plt.scatter(x, y, label='Datos observados', color='darkorange', alpha=0.7)
plt.plot(x_line, y_line, color='blue', linewidth=2, label='y = 2x + 1 (teórica)')
plt.title('Scatterplot con línea de regresión')
plt.xlabel('Variable x')
plt.ylabel('Variable y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## 3. Figura con múltiples visualizaciones

```python
# Crear figura con 3 subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# 1. Scatterplot con línea
axs[0].scatter(x, y, color='darkorange', alpha=0.7)
axs[0].plot(x_line, y_line, color='blue', linewidth=2)
axs[0].set_title('Scatterplot con línea')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].grid(True, alpha=0.3)

# 2. Histograma
axs[1].hist(y, bins=10, color='green', alpha=0.7, edgecolor='black')
axs[1].set_title('Distribución de y')
axs[1].set_xlabel('Valores de y')
axs[1].set_ylabel('Frecuencia')
axs[1].grid(True, alpha=0.3)

# 3. Función seno
axs[2].plot(t, np.sin(t), color='purple', linewidth=2)
axs[2].set_title('Función senoidal')
axs[2].set_xlabel('Tiempo (t)')
axs[2].set_ylabel('sin(t)')
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 4. Gráfico de Paleta (Lollipop Graph)

```python
# Datos para el gráfico de paleta
categorias = ['Producto A', 'Producto B', 'Producto C', 'Producto D', 'Producto E']
valores = [23, 45, 56, 78, 32]

# Crear gráfico de paleta
plt.figure(figsize=(10, 6))

# Crear líneas verticales (los "palos" del gráfico)
plt.vlines(x=categorias, ymin=0, ymax=valores, color='skyblue', linewidth=2)

# Crear puntos en los extremos (las "paletas")
plt.scatter(categorias, valores, color='navy', s=100, zorder=3)

# Personalizar el gráfico
plt.title('Gráfico de Paleta - Ventas por Producto')
plt.xlabel('Productos')
plt.ylabel('Ventas (unidades)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

# Añadir valores en los puntos
for i, valor in enumerate(valores):
    plt.text(i, valor + 2, str(valor), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
```

## Explicación de conceptos importantes:

- **plt.figure()**: Crea una nueva figura
- **figsize**: Controla el tamaño de la figura (ancho, alto en pulgadas)
- **plt.subplots()**: Crea múltiples subplots en una figura
- **alpha**: Controla la transparencia (0 = transparente, 1 = opaco)
- **zorder**: Controla el orden de visualización (mayor valor = más arriba)
- **tight_layout()**: Ajusta automáticamente los espacios entre subplots
- **grid()**: Añade una cuadrícula al gráfico
