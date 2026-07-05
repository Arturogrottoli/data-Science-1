# Readme Repaso — Clase 05: Visualizaciones Avanzadas en Data Science

Guia de clase para el profesor. Cada sección incluye qué decir, qué mostrar y qué ejecutar en el notebook.

---

## Antes de empezar — Encuadre de la clase

> "La clase de hoy tiene dos partes. En la primera hacemos un repaso de todo lo visto hasta ahora: variables, control de flujo, NumPy y Pandas. En la segunda parte arrancamos con el tema nuevo: visualizaciones avanzadas. Al final tienen una actividad práctica con un dataset real de tráfico aéreo."

Abrir el notebook: `Repaso_Data_Science_I_Fundamentos_para_la_Ciencia_de_Datos_.ipynb`

---

## PARTE 1 — Repaso Semanas 1 a 4

---

### Semana 1 — Variables y Tipos de Datos

**Qué decir:**
- Una variable es una caja con una etiqueta. El tipo de dato define qué operaciones podemos hacerle.
- Los cuatro tipos básicos: `int` (entero), `float` (decimal), `str` (texto), `bool` (verdadero/falso).
- Las f-strings (`f"texto {variable}"`) son la forma más cómoda de mezclar texto y variables.
- Los diccionarios modelan un "registro" del mundo real: la máquina, el sensor, el cliente. Cada clave es un campo, cada valor es el dato.

**Ejecutar:**
```python
edad = 25
precio = 99.99
nombre = "Planta Alta"
activo = True

print(f"El sensor {nombre} está activo? {activo}")

evento_maquina = {
    "sensor_id": "TERM-04",
    "fecha_hora": "2026-05-17 14:30:00",
    "temperatura": 82.5,
    "estado": "Operando"
}

print(f"El sensor {evento_maquina['sensor_id']} registró {evento_maquina['temperatura']}°C.")
```

**Preguntar a la clase:**
- ¿Por qué usamos un diccionario y no cuatro variables separadas?
  - Porque agrupa toda la información de un solo evento. Si el sensor manda 1.000 registros, manejamos 1.000 diccionarios, no 4.000 variables sueltas.

---

### Semana 2 — Control de Flujo: `if/else` y `for`

**Qué decir:**
- El `for` le pone automatización al script: procesa cada elemento de una lista sin que escribamos una línea por cada uno.
- El `if/else` le da "cerebro": toma decisiones distintas según el dato.
- El patrón "lista vacía + for + append" es uno de los más usados en data science para filtrar o transformar datos.

**Ejecutar:**
```python
historial_temps = [72.0, 85.3, 91.0, 68.4, 88.9]
temps_altas = []

for temp in historial_temps:
    if temp > 80.0:
        temps_altas.append(temp)

print(f"Todas las mediciones: {historial_temps}")
print(f"Mediciones peligrosas (>80): {temps_altas}")
```

**Preguntar a la clase:**
- ¿Qué pasa si sacamos el `else: pass`? — Nada, `pass` es opcional cuando el bloque `else` no hace nada.
- ¿Cómo filtrarían los menores a 70? — Cambiar el `>` por `<` y el umbral.

---

### Semana 3 — NumPy: Cálculos en Bloque

**Qué decir:**
- Con listas de Python puro, para multiplicar cada precio por el tipo de cambio necesitaríamos un `for`.
- NumPy inventa el **array**: aplicás la misma operación a todos los elementos de una vez. Esto se llama **vectorización**.
- Las **máscaras booleanas** permiten filtrar el array usando una condición directamente entre corchetes, sin bucles.

**Ejecutar:**
```python
import numpy as np

precios_usd = np.array([10.0, 25.5, 100.0, 5.25])
tipo_cambio = 1000

precios_ars = precios_usd * tipo_cambio   # vectorización
print(f"En ARS: {precios_ars}")

precios_caros = precios_usd[precios_usd > 20.0]   # máscara booleana
print(f"Más de 20 USD: {precios_caros}")
```

**Preguntar a la clase:**
- ¿Por qué NumPy es más rápido que un `for`? — Porque las operaciones se ejecutan en C por debajo, no en Python puro.

---

### Semana 4 — Pandas: El Excel de Python

**Qué decir:**
- Pandas toma arrays de NumPy y les pone filas y columnas: el **DataFrame**.
- Los tres movimientos más comunes en limpieza de datos: **fillna** (rellenar vacíos), **columna calculada** (nueva columna a partir de otras), **groupby** (agrupar y agregar).
- `np.nan` representa un dato faltante. Es diferente a cero: no sabemos cuánto vale.

**Ejecutar:**
```python
import pandas as pd
import numpy as np

datos_ventas = {
    "Producto": ["Teclado", "Mouse", "Teclado", "Monitor", "Mouse"],
    "Unidades": [10, 15, np.nan, 5, 20],
    "Precio_Unitario": [50, 25, 50, 300, 25]
}

df = pd.DataFrame(datos_ventas)
print("Tabla original:")
print(df)

df["Unidades"] = df["Unidades"].fillna(0)           # rellenar NaN
df["Total_Venta"] = df["Unidades"] * df["Precio_Unitario"]  # columna nueva
reporte = df.groupby("Producto")["Total_Venta"].sum()       # agrupar

print("\nTotal por producto:")
print(reporte)
```

**Preguntar a la clase:**
- ¿Qué pasaría si en vez de `fillna(0)` usáramos `fillna(df["Unidades"].mean())`? — Completaríamos con el promedio del resto.

---

## PARTE 2 — Clase 05: Visualizaciones Avanzadas

Abrir el notebook: `Semana_5_Visualizaciones_Avanzadas_en_Data_Science_.ipynb`

---

## Bloque 1 — Arquitectura de Figuras y Diseño Avanzado

---

### Tema 1 — La Jerarquía de Matplotlib: Figure y Axes

**Qué decir:**
- Matplotlib funciona como un artista. Hay dos objetos principales:
  - **Figure**: el lienzo completo, el cuadro físico colgado en la pared.
  - **Axes**: la pintura individual dibujada sobre ese lienzo.
- Una Figure puede tener uno o muchos Axes (subgráficos).
- `plt.subplots(nrows, ncols)` devuelve el lienzo `fig` y la lista de Axes `axes`.

**Ejecutar:**
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ax1 = axes[0]
ax2 = axes[1]

ax1.plot([1, 2, 3], [10, 20, 30], color="blue", marker="o")
ax1.set_title("Pintura 1 (izquierda)")

ax2.bar(["A", "B", "C"], [5, 15, 10], color="orange")
ax2.set_title("Pintura 2 (derecha)")

fig.suptitle("Un solo lienzo con dos gráficos", fontsize=14)
plt.show()
```

**Qué remarcar:**
- Cuando tienen un solo subplot pueden usar `plt.plot()` directamente. Cuando tienen varios, usan `ax1.plot()`, `ax2.bar()`, etc. para controlar en cuál dibujan.

---

### Tema 2 — Seaborn: Axes-level vs Figure-level

**Qué decir:**
- Seaborn tiene dos tipos de funciones:
  - **Axes-level** (`sns.scatterplot`, `sns.boxplot`, `sns.histplot`): obedecen al Axes que el programador creó. Se les pasa `ax=mi_eje`.
  - **Figure-level** (`sns.displot`, `sns.relplot`, `sns.catplot`): crean su propio lienzo completo y lo administran solas. Devuelven un `FacetGrid`, no un Axes.
- La regla práctica: si querés integrar el gráfico con otros en el mismo subplot, usá Axes-level.

**Ejecutar:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")

# Axes-level: obedece al ax que creamos
fig, mi_eje = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=tips, x="total_bill", y="tip", ax=mi_eje, color="purple")
mi_eje.set_title("Axes-level: yo controlo el eje")
plt.show()

# Figure-level: hace lo suyo sola
g = sns.displot(data=tips, x="total_bill", col="time", kind="kde", fill=True)
plt.show()
```

---

### Tema 3 — GridSpec: Layouts Personalizados

**Qué decir:**
- Por defecto, `plt.subplots()` divide la figura en partes iguales.
- **GridSpec** permite romper esa simetría: que un gráfico ocupe el doble de espacio que otro.
- Útil cuando tenés un gráfico principal grande y uno secundario de resumen pequeño.

**Ejecutar:**
```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(8, 6))

# 2 filas, 1 columna. La segunda fila es 3 veces más alta que la primera.
gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 3], hspace=0.4)

ax_superior = fig.add_subplot(gs[0])
ax_inferior = fig.add_subplot(gs[1])

ax_superior.plot([1, 2, 4], [5, 4, 3], color="red")
ax_superior.set_title("Resumen (ratio 1)")

ax_inferior.scatter([1, 2, 3], [10, 40, 20], color="green", s=100)
ax_inferior.set_title("Detalle principal (ratio 3)")

plt.show()
```

---

## Bloque 2 — Series Temporales y Análisis Multivariado

---

### Tema 4 — Series Temporales: Por Qué Importa el Tipo `datetime`

**Qué decir:**
- Cuando pandas lee una columna de fechas desde un CSV, la lee como texto (`object`) por defecto.
- Si graficamos fechas como texto, el eje X las ordena **alfabéticamente** (Abril, Agosto, Diciembre, Enero...) y destruye la cronología.
- `pd.to_datetime()` convierte el texto en un tipo especial que pandas y matplotlib entienden como tiempo real.

**Ejecutar:**
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {'Año': ['2025', '2023', '2026', '2024'], 'Ventas': [50, 30, 60, 40]}
df = pd.DataFrame(data)

# INCORRECTO: orden de la tabla, no cronológico
plt.figure(figsize=(5, 3))
sns.lineplot(data=df, x="Año", y="Ventas", marker="o")
plt.title("Orden de tabla (INCORRECTO)")
plt.show()

# CORRECTO: convertimos a datetime
df['Año_Correcto'] = pd.to_datetime(df['Año'], format='%Y')
plt.figure(figsize=(5, 3))
sns.lineplot(data=df, x="Año_Correcto", y="Ventas", marker="o")
plt.title("Cronología correcta con datetime")
plt.show()
```

**Tip de profesor:** Mostrar ambos gráficos lado a lado para que el contraste sea visible de golpe.

---

### Tema 5 — Formatear el Eje de Fechas con `matplotlib.dates`

**Qué decir:**
- Cuando el eje X tiene fechas completas (año, mes, día), matplotlib a veces muestra el formato completo y queda ilegible.
- `mdates.DateFormatter('%Y')` le dice que solo muestre el año.
- `mdates.YearLocator()` le dice que ponga una marca por año.

**Ejecutar:**
```python
import matplotlib.dates as mdates

plt.figure(figsize=(10, 3))
sns.lineplot(data=df, x="Año_Correcto", y="Ventas", marker="o")

ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())

plt.title("Solo el año en el eje X")
plt.show()
```

---

### Tema 6 — Análisis Multivariado: Canales Estéticos

**Qué decir:**
- La pantalla es plana: solo tiene X e Y (2 dimensiones).
- Para analizar 3, 4 o más variables al mismo tiempo, mapeamos las variables extra a **canales estéticos**:
  - `hue` → color
  - `style` → forma del marcador o trazo
  - `size` → tamaño del punto
- Con estos tres canales podemos mostrar hasta 5 variables en un solo gráfico 2D.

**Ejecutar:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=tips,
    x="total_bill",  # Variable 1
    y="tip",         # Variable 2
    hue="smoker",    # Variable 3 (color)
    size="size",     # Variable 4 (tamaño)
    sizes=(20, 200)
)
plt.title("4 variables en 2 dimensiones")
plt.show()
```

---

### Tema 7 — Accesibilidad Cromática

**Qué decir:**
- Nunca confíen toda la información al color. Dos razones principales:
  1. **Daltonismo**: el 8% de los hombres no distingue rojo de verde.
  2. **Impresión en blanco y negro**: en un reporte impreso el color desaparece.
- La regla de oro: si cambiás el color, cambiá también la **forma** (`style`, `markers`).
- La paleta `colorblind` de Seaborn está diseñada para ser distinguible con daltonismo.

**Ejecutar:**
```python
flights = sns.load_dataset("flights")
df_meses = flights[flights['month'].isin(['Jan', 'Jul', 'Dec'])]
df_meses['month'] = df_meses['month'].cat.remove_unused_categories()

plt.figure(figsize=(8, 4))
sns.lineplot(
    data=df_meses,
    x="year",
    y="passengers",
    hue="month",
    style="month",       # forma distinta por mes
    markers=True,        # marcadores geométricos distintos
    palette="colorblind" # paleta apta para daltonismo
)
plt.title("Accesible: color + trazo + marcador")
plt.show()
```

**Preguntar a la clase:**
- ¿Por qué no alcanza con `hue` solo? — Porque si alguien imprime en blanco y negro o tiene daltonismo, todas las líneas se ven iguales.

---

## Bloque 3 — Interactividad y Reportes de Alta Calidad

---

### Tema 8 — Plotly Express: Gráficos Interactivos

**Qué decir:**
- Matplotlib y Seaborn generan imágenes estáticas: una foto del dato.
- **Plotly Express** genera gráficos basados en HTML y JavaScript.
- El resultado se puede abrir en el navegador y permite:
  - Hacer **zoom** en cualquier zona
  - Ver **tooltips** con el valor exacto al pasar el cursor
  - **Aislar** una categoría haciendo clic en la leyenda
- Mismo código, experiencia completamente distinta para el usuario final.

**Ejecutar:**
```python
import plotly.express as px
import seaborn as sns

penguins = sns.load_dataset("penguins")

fig = px.scatter(
    penguins,
    x="flipper_length_mm",
    y="body_mass_g",
    color="species",
    title="Interactivo: hover sobre los puntos, clic en la leyenda"
)
fig.show()
```

**Qué demostrar en vivo:**
1. Pasar el cursor sobre un punto → tooltip con valores exactos.
2. Clic en una especie en la leyenda → se aísla ese grupo.
3. Seleccionar una zona con el mouse → zoom automático.

---

### Tema 9 — Exportación: Rasterizado vs Vectorial

**Qué decir:**
- Cuando guardamos un gráfico, el formato define su comportamiento:
  - **Rasterizado** (`.png`, `.jpg`): guardamos píxeles. Si el `dpi` es bajo, se ve borroso al ampliar. Para presentaciones y web siempre usar `dpi=300`.
  - **Vectorial** (`.pdf`, `.svg`): guardamos la fórmula matemática de cada línea. Resolución infinita: podría imprimirse en un cartel de ruta sin pixelarse.
- Regla práctica: `.png` para PowerPoint y web, `.pdf` para documentos impresos o publicaciones.

**Ejecutar:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

penguins = sns.load_dataset("penguins")
sns.scatterplot(data=penguins, x="flipper_length_mm", y="body_mass_g")
plt.title("Gráfico para exportar")

# Rasterizado a alta resolución
plt.savefig("grafico_alta_resolucion.png", dpi=300)

# Vectorial
plt.savefig("grafico_vectorial.pdf")

plt.close()
print("Archivos guardados.")
```

---

### Tema 10 — `bbox_inches='tight'`: El Encuadre Perfecto

**Qué decir:**
- Por defecto, `savefig()` calcula el tamaño de la imagen basándose en el lienzo, pero "se olvida" de medir textos decorativos como títulos largos o etiquetas rotadas.
- El resultado: texto cortado en la imagen guardada, aunque se vea bien en pantalla.
- `bbox_inches='tight'` le ordena recalcular el encuadre incluyendo absolutamente todo el contenido.
- Es un parámetro barato que siempre conviene agregar.

**Ejecutar:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

penguins = sns.load_dataset("penguins")
sns.histplot(data=penguins, x="flipper_length_mm", color="teal")
plt.title("Este título ridículamente largo se suele cortar al exportar", fontsize=14, pad=20)
plt.xlabel("Largo de la Aleta (mm)", fontsize=14, labelpad=20)

# Sin tight: puede cortar
plt.savefig("grafico_mal_encuadrado.png")

# Con tight: perfecto
plt.savefig("grafico_perfecto.png", bbox_inches='tight')

plt.close()
print("Compará ambos archivos.")
```

**Qué demostrar:** Abrir ambas imágenes y comparar el corte del título.

---

## ACTIVIDAD PRÁCTICA — Tráfico Aéreo

Dataset: `vuelos_asientos_pasajeros.csv`

Los tres pasos de la actividad combinan todo lo visto en la clase:

| Paso | Tema que aplica |
|------|----------------|
| 1 — Histograma de vuelos diarios | Bloque 1: Figure + Axes + Axes-level |
| 2 — Evolución de pasajeros 2023 | Bloque 2: datetime + multivariado + accesibilidad |
| 3 — Interactivo + exportación PNG | Bloque 3: Plotly + savefig 300dpi + bbox_inches |

### Paso 1 — Distribución de vuelos

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")
df_vuelos = pd.read_csv("vuelos_asientos_pasajeros.csv")

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(data=df_vuelos, x="vuelos", ax=ax, bins=30, color="#4682B4", kde=True)
ax.set_title("Distribución de Vuelos Diarios", fontsize=12, fontweight='bold')
ax.set_xlabel("Cantidad de vuelos por día")
ax.set_ylabel("Frecuencia (días)")
plt.show()
```

### Paso 2 — Evolución temporal con accesibilidad

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_vuelos = pd.read_csv("vuelos_asientos_pasajeros.csv")
df_vuelos['indice_tiempo'] = pd.to_datetime(df_vuelos['indice_tiempo'])
df_2023 = df_vuelos[df_vuelos['indice_tiempo'].dt.year == 2023]

plt.figure(figsize=(12, 5))
sns.lineplot(
    data=df_2023,
    x="indice_tiempo",
    y="pasajeros",
    hue="clasificacion_vuelo",
    style="clasificacion_vuelo",
    markers=True,
    palette="colorblind"
)
plt.title("Evolución de Pasajeros 2023 — Cabotaje vs Internacional")
plt.xlabel("Fecha")
plt.ylabel("Pasajeros")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
```

### Paso 3 — Interactivo Plotly + exportación

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

df_vuelos = pd.read_csv("vuelos_asientos_pasajeros.csv")

# Gráfico interactivo
fig_interactiva = px.scatter(
    df_vuelos,
    x="asientos",
    y="pasajeros",
    color="clasificacion_vuelo",
    size="vuelos",
    title="Interactivo: Asientos vs Pasajeros",
    template="plotly_white"
)
fig_interactiva.show()

# Exportación de alta calidad
plt.figure(figsize=(7, 4.5))
sns.scatterplot(data=df_vuelos, x="asientos", y="pasajeros",
                hue="clasificacion_vuelo", size="vuelos")
plt.title("Ocupación: Asientos vs Pasajeros")
plt.savefig("reporte_vuelos_300dpi.png", dpi=300, bbox_inches='tight')
plt.close()
print("Exportado como PNG 300dpi.")
```

---

## Resumen de Conceptos Clave

| Concepto | Para qué sirve |
|----------|---------------|
| `Figure` / `Axes` | Jerarquía de Matplotlib: lienzo y gráfico |
| `plt.subplots(nrows, ncols)` | Crear grilla de gráficos |
| `GridSpec(height_ratios)` | Subgráficos con proporciones distintas |
| Axes-level vs Figure-level | Cuándo Seaborn obedece vs cuándo se adueña |
| `pd.to_datetime()` | Convertir texto a fecha para ordenamiento cronológico |
| `mdates.DateFormatter` | Formatear etiquetas del eje de fechas |
| `hue`, `style`, `size` | Canales estéticos para análisis multivariado |
| `palette="colorblind"` | Paleta accesible para daltonismo |
| `markers=True` | Respaldo de formas para impresión B&N |
| `plotly.express` | Gráficos interactivos (hover, zoom, leyenda) |
| `dpi=300` | Alta resolución para exportar PNG |
| `.pdf` / `.svg` | Exportación vectorial con resolución infinita |
| `bbox_inches='tight'` | Encuadre que no corta títulos ni etiquetas |

---

## Errores comunes que suelen aparecer

**"El eje X de mi gráfico temporal está desordenado"**
→ La columna de fechas sigue siendo texto. Aplicar `pd.to_datetime()` antes de graficar.

**"Mi histograma de Seaborn no aparece en el subplot que yo creé"**
→ Están usando una función Figure-level (`sns.displot`) en vez de Axes-level (`sns.histplot`). Cambiar la función y agregar `ax=ax`.

**"El título se corta en la imagen guardada"**
→ Falta `bbox_inches='tight'` en `plt.savefig()`.

**"Los gráficos de Plotly no se muestran en mi entorno"**
→ En Jupyter Lab instalar la extensión de Plotly o usar `fig.show(renderer="browser")`.
