# Clase 07 — Fundamentos de Machine Learning: de la Teoría a Scikit-Learn

**Curso de Data Science I · Clase 07** — del mapa completo de la Inteligencia Artificial al primer contacto con `train_test_split`.

Esta guía es el **libreto completo para dictar la Clase 07**: reúne toda la teoría de `Clase 07.docx` (no solo lo que entra en una diapositiva) organizada en el mismo orden que las 50 filminas de `Clase07.html`. La lógica de cada bloque es siempre la misma: **primero el repaso de estadística** (lo que ya se sabe), **después una introducción apoyada en las filminas** (el mapa visual del tema), y **de ahí en adelante, filmina y teoría del docx intercaladas** — se proyecta la filmina correspondiente y, antes o después de mostrarla, se desarrolla en voz alta el texto de esta guía, que trae la profundidad completa que la filmina por sí sola no alcanza a mostrar.

> **Contexto de la clase anterior**: Clase 06 cerró con Estadística Descriptiva y Preprocesamiento (medidas de tendencia central/dispersión, normalización/estandarización, `StandardScaler`). Hoy no se repite esa teoría — se la repasa rápido al arrancar (Bloque 0 del notebook) y se la da por incorporada: entender un dato con estadística es el prerrequisito para que un algoritmo "aprenda" de ese mismo dato.

---

## Repaso de la Clase 06 — Estadística y Preprocesamiento

Antes de hablar de Machine Learning, conviene tener fresco lo que se vio en Clase 06 — es el terreno sobre el que se apoya todo lo de hoy: **para que un algoritmo "aprenda" de un dataset, primero hace falta poder describirlo con estadística.** Esto es exactamente lo que hace el **Bloque 0** del Colab (`Clase07.ipynb`), sobre el dataset real de natalidad del DEIS (25 años, 25 provincias) — acá va la teoría de cada pilar seguida del código real que lo aplica, para no tener que buscarlo aparte.

**Qué mide exactamente el dataset (importante para leer bien los números que siguen)**: cada columna `natalidad_*` no es una *cantidad* de nacimientos — es la **tasa bruta de natalidad**: nacimientos ocurridos ese año **por cada 1.000 habitantes** de la población de esa provincia. Un valor de, por ejemplo, 17.9 significa "17.9 nacimientos cada 1.000 personas", no "17.9 bebés". Por eso los valores de todas las columnas rondan entre ~14 y ~26 (nunca números en miles o millones) — es la forma estándar de comparar natalidad entre provincias con poblaciones de tamaños muy distintos: si fuera un conteo crudo, Buenos Aires siempre "ganaría" solo por tener más habitantes.

### 1. Limpieza e Integración

**Qué es cada cosa, brevemente**: un valor **nulo** (o `NaN`, *Not a Number*) es una celda del dataset sin ningún dato cargado — no es un cero, es directamente un vacío. **Limpiar** un dataset es decidir qué hacer con esos nulos y con las filas **duplicadas** (registros repetidos que distorsionarían cualquier cálculo si se cuentan dos veces). **Integrar** es el paso siguiente: combinar o derivar columnas nuevas a partir de las que ya existen, para que la información cruda diga algo útil de negocio.

Ningún dataset real llega listo para analizar, y la decisión de qué hacer con un nulo depende del *significado de negocio* detrás de ese vacío, no de una regla mecánica. Ejemplo concreto: en una tabla de clientes, un campo `fecha_baja` vacío no es un error de carga — significa "este cliente todavía está activo". Borrar esa fila, o peor, imputarla con una fecha promedio, inventaría una baja que nunca ocurrió. Por eso la regla no es "todo nulo se imputa" ni "todo nulo se borra": primero hay que preguntarse *por qué* falta el dato.
- Si el nulo es **aleatorio** (un sensor que falló una vez): imputar con media/mediana (numéricas) o moda (categóricas) es razonable, porque no altera la historia real del registro.
- Si el nulo **significa algo** (como el ejemplo de `fecha_baja`): no se imputa con un promedio — se imputa con una etiqueta de negocio que refleje ese significado (ej. "Activo"), o se crea una columna auxiliar booleana que marque explícitamente ese caso.
- **Integrar** es el paso complementario: no alcanza con tener los datos limpios, hay que combinarlos para que digan algo útil — por ejemplo, cruzar fecha de alta y fecha de baja para derivar "antigüedad del cliente", una columna que no existía en el dato crudo pero que sí existe en el negocio.

**👉 En el Colab — Bloque 0, celda 1. Qué hace en general**: carga el dataset, audita si tiene nulos o duplicados, y crea una columna nueva de negocio a partir de una columna numérica existente.
```python
import pandas as pd
import numpy as np

df_raw = pd.read_csv('tasa-natalidad-deis-2000-2024.csv')
print(f"Filas: {df_raw.shape[0]} (años, de 2000 a 2024) | Columnas: {df_raw.shape[1]} (1 índice de tiempo + 25 provincias)")

nulos = df_raw.isnull().sum().sum()
duplicados = df_raw.duplicated().sum()
print(f"Valores nulos totales: {nulos}")
print(f"Filas duplicadas: {duplicados}")

if nulos == 0 and duplicados == 0:
    print("👉 Este dataset del DEIS ya llega limpio (0 nulos, 0 duplicados).")

natalidad_2024 = df_raw[df_raw['indice_tiempo'] == '01-01-2024'].drop(columns='indice_tiempo').T
natalidad_2024.columns = ['natalidad_2024']
natalidad_2024['categoria_natalidad'] = pd.cut(
    natalidad_2024['natalidad_2024'],
    bins=[0, 8.4, 9.7, np.inf],
    labels=['Baja', 'Media', 'Alta']
)
print(natalidad_2024['categoria_natalidad'].value_counts())
```
**Línea por línea:** `pd.read_csv(...)` carga el archivo en `df_raw`; el `print` de "Filas" verifica que cargó bien. `df_raw.isnull().sum()` cuenta nulos por columna y `.sum()` los suma en un total; `.duplicated().sum()` cuenta filas repetidas. El `if` imprime "ya llega limpio" solo si ambos conteos dieron cero (lo que efectivamente pasa acá). La parte de **Integración**: se filtra la fila del último año (`== '01-01-2024'`), se saca la columna de fecha (`.drop`) y se transpone (`.T`) para que cada provincia pase a ser una fila. `pd.cut(..., bins=[0, 8.4, 9.7, np.inf], labels=[...])` crea la columna de categoría de negocio nueva, agrupando el valor continuo en 3 franjas — una columna que no existía en el archivo original. `.value_counts()` cuenta cuántas provincias cayeron en cada categoría.

### 2. Medidas de Tendencia Central y Dispersión

**Qué es cada cosa, brevemente**: la **tendencia central** busca un único número que represente el "centro" de un conjunto de datos. La **media** (o promedio) se calcula sumando todos los valores y dividiendo por la cantidad de valores. La **mediana** es el valor que queda justo en el medio cuando se ordenan todos los datos de menor a mayor (deja 50% de los datos de cada lado). La **moda** es el valor que más se repite. La **dispersión**, en cambio, mide qué tan lejos del centro se alejan los datos: el **desvío estándar** promedia esa distancia respecto a la media, y los **cuartiles** dividen el conjunto ya ordenado en 4 partes iguales — **Q1** es el valor que deja el 25% de los datos por debajo, **Q3** el que deja el 75%, y el **IQR** (Q3 − Q1) es el ancho de ese tramo central donde vive el 50% de los datos.

Antes de la fórmula, un ejemplo mental rápido: los sueldos de 5 personas son $10, $12, $11, $13 y $9.000 (una persona gana mucho más que el resto). La **media** de esos 5 números da casi $2.000 — un número que no describe a *nadie* del grupo real. La **mediana** (ordenando: 9, 10, 11, 12, 9000 → el valor del medio es 11) sí describe fielmente a la mayoría. Ese es el mecanismo exacto detrás de la regla "la media es sensible a extremos, la mediana no": la media "reparte" el efecto de cada valor entre todos los demás — un solo valor gigante puede arrastrar el promedio entero. La mediana ignora la *magnitud* del extremo y solo mira su *posición* en el orden. La dispersión (desvío estándar, IQR) responde la pregunta que la tendencia central no puede: dos grupos pueden tener la misma media y ser radicalmente distintos.

**En qué caso conviene usar cada una (y en cuál no):**

| Medida | Sirve cuando... | No sirve cuando... |
|---|---|---|
| **Media** | Los datos son razonablemente simétricos y sin valores extremos — ej. la altura de un grupo de adultos, la temperatura diaria de un mes. Tiene la ventaja de usar el valor de *todos* los datos en el cálculo. | Hay outliers o la distribución está sesgada — ej. sueldos, precios de propiedades, tiempos de espera con algunos casos excepcionales. Un solo valor extremo la distorsiona por completo. |
| **Mediana** | Hay outliers o asimetría (justo el caso donde la media falla) — es la medida estándar para variables económicas como ingresos o precios, precisamente porque unos pocos casos extremos no la mueven. | Se necesita un cálculo que use la magnitud exacta de cada dato (por ejemplo, para sumar o promediar en una fórmula posterior) — la mediana ignora cuánto más grande o chico es un valor, solo le importa su posición. |
| **Moda** | La variable es **categórica** (color favorito, método de pago, ticker más operado) — ahí "promediar" no tiene sentido alguno, y la moda es la única medida de tendencia central que aplica. También sirve para variables discretas con pocos valores repetidos (ej. la cantidad de habitaciones más común en un barrio). | Los datos son continuos y variados (como precios exactos con decimales) — ahí puede no haber ningún valor que se repita, o la moda puede ser un valor cualquiera sin significado real. |
| **Desvío estándar** | La distribución es razonablemente simétrica — es la medida de dispersión más usada, y la que asumen muchos modelos estadísticos. | Hay outliers marcados: al estar basado en la media, hereda su misma sensibilidad a valores extremos. |
| **IQR** | Hay outliers o sesgo — al basarse en cuartiles (posiciones), es robusto ante valores extremos, igual que la mediana. Es el criterio que se usa para detectar outliers en un boxplot. | Se necesita capturar qué tan dispersos están *todos* los datos, incluidos los extremos — el IQR deliberadamente los deja afuera del cálculo (solo mira el 50% central). |

**Regla práctica rápida**: si media y mediana dan un valor parecido, la distribución es razonablemente simétrica y cualquiera de las dos sirve. Si difieren mucho (como en el ejemplo de los sueldos), hay sesgo u outliers — ahí conviene reportar la mediana y el IQR, no la media y el desvío estándar, porque estos últimos dos van a estar "contaminados" por los valores extremos.

**👉 En el Colab — Bloque 0, celda 2. Qué hace en general**: calcula los 6 estadísticos básicos (media, mediana, desvío, Q1, Q3, IQR) sobre la serie nacional de 25 años.
```python
serie_nacional = df_raw['natalidad_argentina']

media = serie_nacional.mean()
mediana = serie_nacional.median()
std = serie_nacional.std()
q1 = serie_nacional.quantile(0.25)
q3 = serie_nacional.quantile(0.75)
iqr = q3 - q1

print(f"Media (natalidad nacional 2000-2024): {media:.2f}")
print(f"Mediana: {mediana:.2f}")
print(f"Desvío estándar: {std:.2f}")
print(f"IQR (Q3 - Q1): {iqr:.2f} (entre {q1:.2f} y {q3:.2f})")
```
**Línea por línea:** `df_raw['natalidad_argentina']` selecciona la columna nacional como una Serie. `.mean()`, `.median()` y `.std()` calculan los tres estadísticos básicos; `.quantile(0.25)`/`.quantile(0.75)` dan Q1 y Q3, y `iqr = q3 - q1` el ancho de esa caja central. Los `print` muestran cada resultado con 2 decimales. La deducción real: la mediana (17.9) es más alta que la media (16.35) — no por outliers, sino porque la natalidad viene en caída sostenida (más años "altos" al principio de la serie que "bajos" al final).

### 3. Distribuciones y Correlación

**Qué es cada cosa, brevemente**: la **distribución** es la forma que toma un conjunto de datos numéricos al graficarlos en un histograma — puede ser simétrica (una campana, con la misma forma a ambos lados del centro) o **sesgada** hacia un lado (una cola más larga hacia valores altos o hacia valores bajos). La **correlación** (coeficiente de Pearson) es un número entre -1 y 1 que mide qué tan asociadas están linealmente dos variables numéricas: cerca de 1 significa que suben y bajan juntas, cerca de -1 que una sube cuando la otra baja, y cerca de 0 que no hay ninguna relación lineal clara entre ellas.

Sobre la correlación en particular, vale la pena un ejemplo clásico de por qué "no implica causalidad" no es una frase de cajón sino un peligro real: en muchas ciudades, las ventas de helado y los ahogamientos en piletas están altamente correlacionados — suben y bajan juntos durante el año. Nadie diría que el helado *causa* ahogamientos. La variable oculta es el clima: los días de calor generan más consumo de helado *y* más gente en la pileta al mismo tiempo. Cuando dos variables se mueven juntas, siempre conviene preguntarse: ¿hay una tercera variable que explique a ambas por separado? Esa pregunta se repite en el Tema 06 con el ejemplo del protector solar.

**👉 En el Colab — Bloque 0, celda 3. Qué hace en general**: grafica la forma de la serie nacional (histograma) y calcula qué tan correlacionadas están 3 provincias entre sí (heatmap).
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.histplot(serie_nacional, kde=True, bins=10, color='teal')
plt.axvline(media, color='red', linestyle='--', label=f'Media: {media:.1f}')
plt.axvline(mediana, color='green', linestyle='--', label=f'Mediana: {mediana:.1f}')
plt.title('Distribución de la natalidad nacional (2000-2024)')
plt.legend()

plt.subplot(1, 2, 2)
provincias_comparar = df_raw[['natalidad_buenos_aires', 'natalidad_cordoba', 'natalidad_santa_fe']]
matriz_corr = provincias_comparar.corr()
sns.heatmap(matriz_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlación entre provincias')

plt.tight_layout()
plt.show()

skew = serie_nacional.skew()
corr_ba_cba = matriz_corr.loc['natalidad_buenos_aires', 'natalidad_cordoba']
print(f"Asimetría (skew) de la serie nacional: {skew:.2f}")
print(f"Correlación Buenos Aires vs. Córdoba: {corr_ba_cba:.2f}")
```
**Línea por línea:** `plt.figure` abre un lienzo ancho para dos gráficos lado a lado. `plt.subplot(1, 2, 1)` selecciona el primer panel; `sns.histplot(..., kde=True)` dibuja el histograma con una curva suavizada superpuesta; los dos `plt.axvline(...)` marcan media y mediana sobre el mismo gráfico. `plt.subplot(1, 2, 2)` pasa al segundo panel; `.corr()` calcula la matriz de correlación de Pearson entre 3 provincias; `sns.heatmap(..., annot=True, vmin=-1, vmax=1, center=0)` la pinta como cuadrícula de colores con el número exacto adentro y escala fija. `.skew()` calcula la asimetría de la serie; `matriz_corr.loc[...]` extrae puntualmente la correlación Buenos Aires-Córdoba. La deducción: es altísima (>0.95) porque comparten la misma tendencia demográfica nacional — correlación, no causalidad.

**Cómo leer el histograma — sesgo a la izquierda vs. a la derecha**: para diagnosticar el sesgo, hay que fijarse hacia qué lado se estira la **cola larga** de la distribución (la zona con pocas barras, alejada del grueso de los datos), no hacia dónde se amontona la mayoría.

- **Sesgado a la derecha (skew positivo)**: la mayoría de los datos están en valores bajos, con una cola que se estira hacia la derecha (pocos valores altos). Esos pocos valores altos "tiran" de la media hacia arriba, así que queda **media > mediana**. Es el caso típico de los sueldos: la mayoría gana poco, unos pocos ganan mucho, y esa cola de sueldos altos infla el promedio.
- **Sesgado a la izquierda (skew negativo)**: es el espejo — la mayoría de los datos están en valores altos, con una cola que se estira hacia la izquierda (pocos valores bajos). Esos pocos valores bajos "tiran" de la media hacia abajo, así que queda **media < mediana**.
- **Simétrica (skew ≈ 0)**: no hay cola marcada de ningún lado — media y mediana quedan prácticamente iguales.

El número que imprime `.skew()` resume todo esto en una sola cifra, sin tener que mirar el gráfico: positivo = cola a la derecha, negativo = cola a la izquierda, cerca de 0 = simétrica. En este dataset puntual, la mediana (17.9) es mayor que la media (16.35) — eso es exactamente sesgo a la **izquierda** (skew negativo): la mayoría de los años tuvo una natalidad relativamente alta, pero hay una cola de años (los más recientes, tras la caída sostenida) con valores bajos que arrastran el promedio hacia abajo sin mover tanto a la mediana.

**Cómo leer el segundo gráfico — el heatmap de correlación**: es una cuadrícula de 3×3, con las mismas 3 provincias (Buenos Aires, Córdoba, Santa Fe) repetidas en filas y en columnas.

- **La diagonal siempre da 1.00**, con el color más intenso de un extremo — es la correlación de cada provincia consigo misma, siempre perfecta. No aporta información: es la referencia para calibrar el ojo antes de mirar el resto.
- **Es simétrica** respecto a esa diagonal: la celda que cruza Buenos Aires (fila) con Córdoba (columna) muestra el mismo número que la celda que cruza Córdoba con Buenos Aires — alcanza con leer la mitad de la cuadrícula.
- **El color**, con `cmap='coolwarm'` y `center=0`: tonos cálidos (rojos) indican correlación cercana a 1 (fuerte y positiva — suben y bajan juntas), tonos fríos (azules) indican cercana a -1 (fuerte y negativa — una sube cuando la otra baja), y tonos neutros el medio, cerca de 0.
- **Qué se espera ver en este caso concreto**: las 3 celdas fuera de la diagonal en un rojo intenso (correlación por encima de 0.9), porque las 3 son provincias con series de natalidad que vienen cayendo juntas a lo largo de los mismos 25 años — comparten la misma tendencia demográfica del país, no una relación causal entre ellas.

### 4. Transformación y Reducción de Dimensionalidad

**Qué es cada cosa, brevemente**: **transformar** un dato es cambiar su formato o su escala sin cambiar lo que representa — por ejemplo, **escalar** (o **estandarizar**) una columna numérica significa llevarla a una media de 0 y un desvío de 1, para que quede en un rango comparable con las demás columnas, sin alterar el orden relativo de los valores. Matemáticamente es una sola resta y una división: `X_escalado = (X − media) / desvío`. El resultado ya no se lee en las unidades originales (pesos, años, m²), sino en "cuántos desvíos estándar por encima o por debajo de la media" está cada valor — por eso dos columnas con unidades y magnitudes totalmente distintas quedan, después de escalar, en el mismo lenguaje comparable. **Reducir la dimensionalidad** significa resumir muchas columnas en unas pocas columnas **nuevas** — importante: no es elegir un subconjunto de las columnas originales y descartar el resto, es crear combinaciones (sumas ponderadas) de todas ellas, elegidas de forma que esas pocas combinaciones nuevas conserven la mayor parte posible de la variabilidad real del dataset. **PCA** (Análisis de Componentes Principales) es la técnica más común para hacerlo.

**¿Qué es Scikit-Learn, y por qué aparece recién acá?** `StandardScaler` y `PCA` no son fórmulas que se escriben a mano cada vez — son parte de **Scikit-Learn** (`sklearn`), la librería estándar de Python para Machine Learning. En vez de programar la resta y la división de la estandarización, o la descomposición matemática que hace posible PCA (que involucra álgebra lineal bastante más compleja que una resta), Scikit-Learn ya trae esas herramientas implementadas, probadas y optimizadas — solo hay que importarlas y usarlas. Todavía no es el foco de la clase (eso llega formalmente en el Tema 04, "Scikit-Learn por Dentro"), pero vale la pena adelantar el nombre acá porque es la primera vez que aparece en el notebook: cualquier objeto de `sklearn`, sea para escalar, para reducir dimensiones o —más adelante— para entrenar un modelo de predicción, sigue siempre la misma lógica de tres pasos: se **instancia** (se crea el objeto), se **ajusta** con `.fit()` (aprende algo de los datos) y se **aplica** con `.transform()` o `.predict()` (usa lo aprendido). Acá se ve esa lógica por primera vez, aplicada a estadística pura; en el Tema 04 se la va a ver aplicada a un modelo que predice.

**Por qué escalar es obligatorio y no un capricho**: imaginá comparar dos personas usando "edad" (rango típico 0-90) e "ingreso mensual" (rango típico 0-500.000) para decidir cuál es más parecida a un tercer individuo, midiendo la distancia entre sus números tal cual. La diferencia de ingresos (que se mide en decenas de miles) va a dominar completamente el cálculo, aplastando por completo a la diferencia de edad — no porque el ingreso sea más importante, sino porque sus números son más grandes en magnitud. `StandardScaler` neutraliza ese efecto llevando todas las columnas a una escala común.

**Por qué PCA sirve específicamente en este caso**: el dataset tiene 25 provincias, y cada una trae 25 años de datos — pero esas 25 columnas (años) están lejos de ser 25 piezas de información independientes: si la natalidad cae un año, es muy probable que también haya caído el año anterior y el siguiente (es una tendencia, no un salto aleatorio). Esa redundancia es exactamente lo que PCA aprovecha: en vez de necesitar 25 números para describir a cada provincia, encuentra 2 combinaciones nuevas que ya capturan la mayor parte de esa variación real — la caída sostenida en el tiempo, principalmente. Es la misma idea que resumir 25 fotos casi idénticas de una persona parada en el mismo lugar con una sola descripción ("está de pie, mirando a cámara") en vez de describir cada foto por separado.

**👉 En el Colab — Bloque 0, celda 4. Qué hace en general**: estandariza dos columnas con `StandardScaler`, y comprime las 25 provincias en 2 componentes principales con `PCA`.
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

columnas_ejemplo = df_raw[['natalidad_buenos_aires', 'natalidad_cordoba']]
scaler_demo = StandardScaler()
columnas_escaladas = scaler_demo.fit_transform(columnas_ejemplo)

print("Antes de escalar (media original):")
print(columnas_ejemplo.mean().round(2).to_dict())
print("Después de escalar con StandardScaler (media ≈ 0, desvío ≈ 1):")
print(pd.DataFrame(columnas_escaladas, columns=columnas_ejemplo.columns).describe().loc[['mean', 'std']].round(2))

provincias_T = df_raw.drop(columns='indice_tiempo').T
X_scaled_demo = StandardScaler().fit_transform(provincias_T)
pca_demo = PCA(n_components=2)
pca_demo.fit_transform(X_scaled_demo)
varianza_total = pca_demo.explained_variance_ratio_.sum() * 100

print(f"Varianza explicada al comprimir 25 años en 2 componentes principales: {varianza_total:.1f}%")
```
**Línea por línea:** `columnas_ejemplo = df_raw[[...]]` selecciona 2 columnas. `StandardScaler().fit_transform(...)` aprende media/desvío y estandariza en el mismo paso. Los `print` de "antes"/"después" comparan medias originales contra medias ≈0 tras escalar. Para la reducción: `df_raw.drop(columns='indice_tiempo').T` transpone la tabla (cada provincia pasa a ser una fila, cada año una columna); se escala esa tabla transpuesta (obligatorio antes de PCA, por la misma razón de magnitud); `PCA(n_components=2).fit_transform(...)` comprime a 2 componentes; `explained_variance_ratio_.sum() * 100` da el porcentaje de variabilidad conservada. La deducción: escalar es obligatorio antes de PCA o K-Means porque estos algoritmos miden distancias.

**Por qué este repaso es más que un trámite**: el punto 4 —escalar antes de medir distancias, y ajustar el escalador solo con los datos de entrenamiento— es literalmente la misma regla de oro que se retoma formalmente hoy en el Tema 05 (Data Leakage). No es contenido nuevo disfrazado de repaso: es el mismo concepto, primero en estadística pura y después aplicado a Machine Learning. El ejemplo de la edad vs. el ingreso de este punto es, de hecho, el mismo tipo de razonamiento detrás del ejemplo de la Filmina 39 (por qué no se puede calcular un promedio con todo el dataset antes de dividir en train/test).

**Recién ahora, con la estadística repasada, arrancamos con Machine Learning.**

---

## Antes de Arrancar: ¿Qué es Machine Learning, y de qué va esta clase?

**¿Qué es el Machine Learning, en una frase?** Es la rama de la Inteligencia Artificial que, en vez de decirle a la computadora paso a paso qué hacer, le muestra muchos ejemplos y deja que ella misma encuentre el patrón. La computadora no "razona" como una persona — encuentra regularidades estadísticas en los datos que se le dan, y usa esas regularidades para opinar sobre datos nuevos que nunca vio. Ese es el salto que separa "programar" de "entrenar": no se escribe la regla, se muestra el ejemplo.

**El ejemplo que conviene tener siempre a mano hoy: un hospital que quiere detectar una enfermedad rara.**

Imaginá un hospital que atiende a 1.000 pacientes por semana, y de esos, solo 10 (el 1%) tienen una enfermedad grave pero poco frecuente. Se le pide a un equipo de Data Science armar un sistema que, a partir de los datos de cada paciente (síntomas, análisis, antecedentes), diga si tiene la enfermedad o no.

Acá aparece la primera trampa de toda la clase, y es la más importante de las 50 filminas: **un modelo que simplemente respondiera "sano" para TODOS los pacientes, sin mirar un solo dato real, acertaría el 99% de las veces** (990 de 1.000). Si solo se mirara ese número —la Accuracy, o "precisión global"— este modelo, que en realidad no aprendió absolutamente nada, parecería casi perfecto.

Pero ese 1% que falla es exactamente el que importa: son los 10 pacientes enfermos a los que el modelo les dijo que estaban sanos. Un sistema así no sirve para nada, por más espectacular que se vea el número — porque falló justo en el único propósito para el que fue creado.

**El mismo problema, con otro disfraz: el filtro de spam.** Un buzón de correo recibe muchísimos más mails legítimos que spam. Si un filtro decidiera "ningún mail es spam", acertaría la gran mayoría de las veces — pero dejaría pasar el 100% del spam real, que es justo lo que se le pidió que detecte. Es el mismo error de fondo que el del hospital: entrenar (o "resolver") de la forma más fácil posible, apostando a que el caso raro simplemente no importa lo suficiente como para arruinar el promedio.

**Por qué hace falta *entrenar* un modelo, y no programarlo a mano con reglas**: nadie puede escribir a mano una regla que capture todos los síntomas posibles de una enfermedad compleja, o todas las formas posibles de redactar un mail de spam — hay demasiadas combinaciones, y las reglas fijas fallan apenas aparece un caso que el programador no anticipó (esto se retoma en la Filmina 04, con el ejemplo médico exacto de "fiebre + tos"). En cambio, si se le muestran al modelo miles de historiales ya diagnosticados (o miles de mails ya marcados), el algoritmo puede aprender el patrón estadístico que distingue a unos de otros — sin que nadie le diga explícitamente cuál es la regla.

**Por qué esta trampa reaparece durante toda la clase**: que un modelo "perezoso" pueda parecer bueno con la métrica equivocada es una de las razones centrales por las que existe todo lo que se ve hoy — no alcanza con "entrenar un modelo", hace falta saber **medir** si realmente sirve (Tema 05, train/test), entender qué error es más caro para el negocio (Falso Positivo vs. Falso Negativo — un paciente sano al que se alarma innecesariamente no es lo mismo que un paciente enfermo al que se manda a su casa, Tema 03), y no confundir un número prometedor con una decisión correcta (Tema 06). El caso del hospital vuelve explícitamente en la Filmina 18 (Caso E) y en la Filmina 24 (Error 3, "un modelo con 99% de precisión es perfecto") — vale la pena, al llegar ahí, recordar que ya se había anticipado acá mismo, al principio de la clase.

**El mapa del día, en una frase**: primero se ve *qué es* el Machine Learning y cómo se relaciona con la IA y el Deep Learning (Tema 01); después, *cómo* aprende exactamente — con profesor, sin profesor, o por prueba y error (Tema 02); después se conecta todo con casos reales de impacto de negocio, en la Pre-entrega del módulo (Tema 03); recién ahí se entra al código, con la arquitectura de Scikit-Learn (Tema 04) y la forma correcta de evaluar sin hacer trampa (Tema 05); y se cierra viendo el ciclo de vida completo de un proyecto de ML en producción (Tema 06).

---

## Cómo está organizada esta guía

1. **Repaso de Estadística (Clase 06)** — teoría de los 4 pilares + el código del Bloque 0 del Colab, intercalados pilar por pilar.
2. **Introducción con las filminas** — recorrido rápido de portada y mapa general de los 6 temas del día.
3. **Filminas + teoría del docx, intercaladas** — el cuerpo principal de esta guía: cada uno de los 6 Temas, filmina por filmina, con el texto completo del docx desarrollado debajo de cada una.
4. **Código del Colab, intercalado al final de cada Tema** — cada vez que el notebook tiene una celda que corresponde a ese Tema, aparece ahí mismo (marcado con 👉), no en un apéndice aparte.
5. **Referencia Rápida del Notebook** y **Pre-entrega**, al final, para ubicar todo de un vistazo.

---

## Objetivos de la clase

1. Distinguir Inteligencia Artificial, Machine Learning y Deep Learning como una relación de inclusión, no como sinónimos.
2. Reconocer los tres paradigmas de aprendizaje (Supervisado, No Supervisado, Por Refuerzo) y saber diagnosticar cuál aplica a un problema real.
3. Conectar la teoría con aplicaciones reales de la industria (Netflix, Gmail, Uber, banca, salud) para entender el impacto de negocio del ML.
4. Entender la arquitectura interna de Scikit-Learn: Estimators, Transformers y Predictors, y el flujo `fit` → `transform`/`predict`.
5. Aplicar correctamente `train_test_split` y diagnosticar Overfitting, Underfitting y Data Leakage.
6. Completar la Pre-entrega "Aplicaciones Prácticas de ML — Del Algoritmo al Impacto Real".

---

## Introducción con las Filminas

## Filmina 01 — Portada

Apertura de la clase. El subtítulo ya anticipa el arco del día: arrancamos con el mapa conceptual completo de la IA (Tema 01) y terminamos con el primer código ejecutable de Scikit-Learn (Temas 04-05) — de la teoría más abstracta a la herramienta más concreta, en una sola clase. Recorrido rápido de los 6 temas antes de arrancar, para que la clase tenga un mapa mental de adónde va cada bloque: **(1)** el mapa de la IA, **(2)** los tres tipos de aprendizaje, **(3)** aplicaciones prácticas y la Pre-entrega, **(4)** Scikit-Learn por dentro, **(5)** train/test y sobreajuste, **(6)** un segundo repaso de aplicaciones, ahora con el ciclo de vida completo de un proyecto.

**La misma introducción, del lado del notebook**: `Clase07.ipynb` abre con su propio resumen del día — *"Dataset: `propiedades_sueca_ml.csv`, la continuación de la 'valija' de la clase pasada, ya limpia y lista para entrenar un primer modelo"* — y un recorrido en 4 bloques que comprime los 6 Temas de las filminas en una lógica de práctica: **(1)** el mapa de la IA/ML/DL y los tipos de aprendizaje, **(2)** Scikit-Learn por dentro, **(3)** entrenar y evaluar sin trampas, **(4)** consolidación y ciclo de vida completo. Vale la pena mostrar esta misma diapositiva de apertura junto con la introducción del notebook, para que quede clara la correspondencia: el notebook no es "otro tema", es la bajada práctica de las mismas 6 paradas, solo que agrupadas de a dos.

---

# Tema 01 — IA, Machine Learning y Deep Learning: el Mapa Completo (Filminas 02-09)

## Filmina 02 — División de Tema

Divisor de sección. La analogía de las matrioskas que trae la filmina es el ancla de todo el tema — conviene dibujarla en el pizarrón antes de seguir, porque los tres temas siguientes (reglas, ML, DL) son literalmente "abrir" cada muñeca rusa una por una.

**Teoría completa del docx para esta apertura**: imaginá que trabajás en el departamento de atención al cliente de una gran tienda online. Cada día llegan miles de correos electrónicos — algunos son felicitaciones, otros quejas por envíos retrasados, y muchos preguntas sobre devoluciones. Clasificarlos a mano tardaría horas. Se podría escribir un programa con reglas fijas: *"si el correo contiene la palabra 'retraso', enviarlo a Logística"*. Pero, ¿qué pasa si el cliente escribe *"mi paquete no ha llegado"*? La regla falla. Ahí es donde entra la Inteligencia Artificial. La IA no es una "caja negra" mágica, sino un ecosistema de tecnologías organizadas jerárquicamente que permiten resolver problemas complejos de formas que antes eran imposibles.

## Filmina 03 — El Ecosistema de la Inteligencia Artificial

**Teoría completa (1. El Ecosistema de la Inteligencia Artificial, del docx)**: a menudo se usan los términos Inteligencia Artificial (IA), Machine Learning (ML) y Deep Learning (DL) como si fueran intercambiables. Sin embargo, en el mundo profesional de la Ciencia de Datos es fundamental entender que existe una **relación de inclusión**. Imaginá una serie de muñecas rusas o matrioskas: la **IA** es la muñeca más grande (el campo general); el **Machine Learning** es la muñeca que está dentro de la IA; el **Deep Learning** es la muñeca más pequeña, ubicada dentro del Machine Learning. Esta jerarquía significa que **todo el Deep Learning es Machine Learning, y todo el Machine Learning es IA**, pero **no toda la IA es Machine Learning** (existen los sistemas de reglas, que se ven en la próxima filmina).

**Ejemplos reales de cada capa (para bajar la jerarquía a sistemas concretos y conocidos)**:

| Capa | Ejemplo real | Por qué está ahí y no en otra capa |
|---|---|---|
| **IA** (sin ser ML) | El GPS de un auto calculando la ruta más corta | Usa algoritmos de búsqueda y optimización clásicos sobre el mapa (ej. el algoritmo de Dijkstra) — sigue una lógica matemática fija, no "aprende" de viajes anteriores para decidir la ruta. |
| **IA → ML** | El scoring crediticio de un banco | Aprende de miles de préstamos históricos (pagados o no) qué patrones predicen el riesgo de impago — nadie escribió esas reglas a mano, el algoritmo las descubrió en los datos. |
| **IA → ML → DL** | El desbloqueo facial de un celular | Usa redes neuronales profundas entrenadas con millones de fotos para reconocer rasgos faciales — ni un experto humano podría escribir "reglas" de qué combinación exacta de píxeles define una cara. |

**Un poco de historia, como dato para compartir en clase**: la Inteligencia Artificial como campo de estudio existe desde la década de 1950 (con Alan Turing y su pregunta "¿pueden pensar las máquinas?"), pero durante décadas predominaron los sistemas basados en reglas, por la limitación de datos y de poder de cómputo disponible. El Machine Learning recién ganó terreno de forma masiva desde los años 90, cuando empezó a haber suficientes datos digitalizados para entrenar modelos estadísticos a gran escala. El Deep Learning, en particular, tuvo su punto de inflexión en 2012: una red neuronal profunda (AlexNet) superó por un margen enorme a todos los métodos clásicos en un concurso mundial de reconocimiento de imágenes — ese resultado fue lo que llevó a que la industria empezara a invertir masivamente en esta tecnología, hasta llegar a los sistemas de hoy.

**Otro ejemplo de regla fija que falla (para sumar al de los correos de la filmina anterior)**: un sistema de moderación de comentarios que bloquea automáticamente cualquier mensaje con la palabra "estúpido" funciona bien al principio — hasta que un usuario escribe "no seas e5túp1do" (reemplazando letras por números) y esquiva la regla sin esfuerzo. Cada vez que se bloquea una variante nueva, aparece otra distinta. Es el mismo patrón que "retraso" vs. "mi paquete no ha llegado": la regla persigue palabras exactas, mientras que el problema real (el insulto, la queja) puede expresarse de formas prácticamente infinitas.

**Pregunta para tirar a la clase**: ¿alguien puede pensar en un ejemplo de su propio trabajo o vida diaria donde una regla fija "si esto, entonces aquello" dejó de funcionar apenas la situación se volvió un poco más compleja de lo previsto?

## Filmina 04 — IA Simbólica: Sistemas Basados en Reglas

**Teoría completa (2. Inteligencia Artificial: El concepto paraguas, del docx)**: la Inteligencia Artificial es la rama de las ciencias de la computación que busca crear sistemas capaces de realizar tareas que, si fueran hechas por humanos, requerirían inteligencia. Esto incluye razonamiento, aprendizaje, percepción y resolución de problemas. En los inicios de la IA, la mayoría de los sistemas funcionaban con reglas "Si-Entonces" (If-Then) definidas por humanos — la llamada **IA Simbólica**. Ejemplo: un sistema de diagnóstico médico donde un experto humano escribe *"Si el paciente tiene fiebre &gt; 38°C Y tiene tos, entonces sugiere test de gripe"*. **Limitación**: el programador debe prever cada escenario posible. Si el mundo cambia o el problema es muy complejo (como reconocer una cara en una foto), es imposible escribir suficientes reglas manuales.

**Matiz que no está en la filmina**: los sistemas basados en reglas no son "IA vieja e inútil" — siguen siendo la opción correcta cuando el proceso es 100% predecible (un termostato, un menú telefónico "Presione 1 para ventas"). El problema no es la técnica en sí, es usarla para un problema que no es predecible.

**Un ejemplo histórico real, para dar contexto**: uno de los sistemas expertos más conocidos fue **MYCIN**, desarrollado en los años 70 en la Universidad de Stanford — un programa que sugería tratamientos para infecciones bacterianas a partir de varios cientos de reglas "Si-Entonces" que médicos expertos habían escrito a mano. Funcionaba razonablemente bien dentro de ese dominio acotado, pero cada enfermedad nueva o cada síntoma atípico exigía que un experto humano se sentara otra vez a escribir reglas adicionales. Ese cuello de botella —depender de un experto para codificar cada caso nuevo, uno por uno— es la razón concreta por la que este tipo de "sistemas expertos" fue perdiendo terreno frente al Machine Learning.

**La limitación, con números concretos (para que no quede como una frase abstracta)**: pensá en un sistema de aprobación de préstamos que solo tuviera que considerar 5 variables (edad, ingreso, historial crediticio, tipo de empleo, monto solicitado), con apenas 4 categorías posibles cada una. Cubrir a mano todas las combinaciones ya requeriría reglas para 4⁵ = 1.024 casos distintos — y un problema real de negocio suele tener muchas más de 5 variables, y más de 4 categorías cada una. Ese crecimiento explosivo de combinaciones es, en números, por qué "escribir todas las reglas posibles" deja de ser viable apenas el problema se complica un poco.

**Dónde siguen ganando los sistemas de reglas hoy, para no dejar la idea a mitad de camino**: no es una técnica del pasado — se usa todos los días en sistemas donde la lógica es 100% conocida y estable: la validación de un formulario web ("el campo email debe contener un @"), las reglas de un firewall de seguridad informática, o el cálculo de un impuesto según una tabla oficial y fija. La pregunta que separa cuándo conviene usar reglas y cuándo conviene ML no es "¿cuál técnica es mejor?" en abstracto, sino "¿este problema tiene una lógica fija y ya conocida de antemano, o depende de patrones que solo se pueden descubrir mirando datos?".

## Filmina 05 — Machine Learning: el Aprendizaje a través de Datos

**Teoría completa (3. Machine Learning: El aprendizaje a través de datos, del docx)**: el Machine Learning (Aprendizaje Automático) es un subconjunto de la IA que rompe con el paradigma de las reglas manuales. En lugar de decirle a la computadora qué reglas seguir, le damos **datos** (ejemplos) y un **algoritmo** para que ella misma descubra los patrones. En módulos anteriores se aprendió a usar condicionales en Python (`if`, `else`). Si se quisiera detectar correos de spam con Python básico, habría que listar miles de palabras prohibidas. Con Machine Learning, se le entregan al modelo 10.000 correos marcados como "Spam" y 10.000 como "No Spam". El algoritmo analiza la frecuencia de palabras, la hora de envío y la estructura para crear su propia "regla matemática" interna.

**El rol crítico de los datos y el Feature Engineering**: como ya se vio con Pandas y Estadística, la calidad del dato es lo más importante. En el ML clásico se practica el **Feature Engineering** (Ingeniería de Características): el proceso de seleccionar y transformar las variables (columnas) que se le entregan al modelo. Ejemplo: si se quiere predecir el precio de una casa, se decide que las columnas "m2", "barrio" y "número de habitaciones" son las importantes. El modelo no "sabe" qué es una casa, solo procesa los números que se eligieron.

**Diferenciando los conceptos que se acaban de nombrar (para no mezclarlos)**:

- **Regla (Filmina 04) vs. Algoritmo (acá)**: una regla es una instrucción fija escrita por un humano ("si dice X, hacé Y"). Un **algoritmo** de ML, en cambio, es un procedimiento matemático general para *encontrar* esa relación por sí solo, mirando ejemplos — no viene con la respuesta ya escrita adentro.
- **Algoritmo vs. Modelo**: son la confusión más común de esta unidad. El **algoritmo** (ej. "Regresión Logística") es el método genérico, el mismo para cualquier problema. El **modelo** es el resultado concreto de aplicar ese algoritmo a *estos* datos puntuales: son los números ya ajustados (los coeficientes, en el caso de una regresión) que quedan después de entrenar. Es la diferencia entre "la receta" (el algoritmo, siempre igual) y "la torta ya horneada" (el modelo, específico de esta tanda de ingredientes/datos).
- **Dato vs. Feature**: un **dato** es cualquier valor crudo guardado en el dataset (una fecha, un texto, un número). Una **feature** (característica) es un dato — o una transformación de uno o varios datos— que específicamente se decide usar como entrada del modelo. No todo dato termina siendo feature (un ID de cliente es un dato, pero casi nunca una feature útil).

**Otro ejemplo de Feature Engineering, en un dominio distinto al de la casa**: para predecir si un cliente de e-commerce va a darse de baja (*churn*), los datos crudos podrían ser la fecha de cada compra pasada. Como feature no se usa la fecha tal cual — se **transforma** en "días desde la última compra" o "cantidad de compras en los últimos 3 meses", que son los números que realmente le dicen algo al modelo sobre el riesgo de fuga. La fecha cruda, sin ese trabajo de transformación, es casi inútil para el algoritmo.

## Filmina 06 — Deep Learning: la Potencia de las Redes Neuronales

**Teoría completa (4. Deep Learning: La potencia de las Redes Neuronales, del docx)**: el Deep Learning (Aprendizaje Profundo) es una evolución del Machine Learning que utiliza estructuras llamadas **Redes Neuronales Artificiales**. Recibe el nombre de "profundo" porque estas redes tienen muchas capas de procesamiento (decenas o cientos). La gran diferencia con el ML clásico radica en el tipo de datos que maneja y cómo procesa las características:

- **Datos no estructurados**: mientras que el ML clásico brilla con tablas de Excel (datos estructurados), el Deep Learning es el rey de las imágenes, el sonido y el texto libre (datos no estructurados).
- **Extracción automática de características**: a diferencia del ML, donde nosotros elegimos las variables, el Deep Learning puede aprender por sí solo qué partes de una imagen son importantes (bordes, texturas, formas) para identificar que lo que hay en la foto es un gato.

**Diferenciando "capa", "red" y "neurona" (para bajar "profundo" a algo concreto)**: una **Red Neuronal Artificial** está compuesta por **capas** apiladas, y cada capa por muchas **neuronas** (unidades de cálculo muy simples). Lo importante para esta clase no es la matemática interna de cada neurona, sino qué hace cada capa en conjunto: en una red que procesa imágenes, las primeras capas detectan patrones muy simples (bordes, líneas, cambios de color); las capas del medio combinan esos patrones simples en formas más complejas (una oreja, un bigote, una textura de pelaje); y las últimas capas combinan esas formas para reconocer el objeto completo (un gato). Cada capa recibe la salida de la anterior y la transforma un poco más — de ahí "profundo": el conocimiento no está en una sola capa, sino en la cadena completa de transformaciones.

**Más ejemplos de Deep Learning en uso, además de ChatGPT**: el asistente de voz de un celular (Siri, Alexa) usa redes neuronales para convertir audio en texto; Google Translate usa una arquitectura de este tipo para traducir frases completas conservando el sentido, no palabra por palabra; y los autos con conducción autónoma usan redes que procesan video en tiempo real para identificar peatones y semáforos (el mismo ejemplo de Tesla que aparece más adelante, en la Filmina 08).

## Filmina 07 — Tabla Comparativa: ML Tradicional vs. Deep Learning

**Teoría completa (tabla del docx)**:

| Característica | Machine Learning Tradicional | Deep Learning |
|---|---|---|
| Volumen de datos | Funciona bien con conjuntos pequeños/medianos | Requiere cantidades masivas de datos |
| Hardware | Puede correr en una laptop estándar | Suele requerir GPUs (procesadores gráficos potentes) |
| Intervención humana | Mucha (necesita Feature Engineering manual) | Baja (aprende características automáticamente) |
| Tiempo de entrenamiento | De segundos a horas | De días a semanas |
| Ejemplos | Predicción de ventas, scoring bancario | Reconocimiento facial, traducción automática, ChatGPT |

**El porqué detrás de cada fila (esto es lo que la tabla no explica por sí sola):**

- **Volumen de datos**: una red neuronal profunda tiene, típicamente, millones de parámetros internos para ajustar (los "pesos" de cada conexión entre neuronas). Con pocos datos, esos millones de parámetros terminan memorizando en vez de aprender un patrón general — es Overfitting llevado al extremo (el mismo concepto que se ve formalmente en el Tema 05). Un algoritmo de ML clásico, en cambio, suele tener muchísimos menos parámetros (una Regresión Lineal solo tiene un coeficiente por feature), así que puede aprender algo útil con cientos o miles de filas.
- **Hardware**: entrenar una red neuronal es, en esencia, hacer millones de multiplicaciones de matrices una y otra vez. Las GPUs (diseñadas originalmente para gráficos 3D de videojuegos) son extremadamente buenas haciendo muchísimas operaciones matemáticas simples en paralelo — exactamente lo que necesita el Deep Learning. Un CPU normal podría hacerlo también, pero tardaría órdenes de magnitud más tiempo.
- **Intervención humana**: esta fila conecta directo con el Feature Engineering de la Filmina 05 — en ML clásico, un humano decide qué columnas importan (m², barrio, habitaciones); en Deep Learning esa selección la hace la propia red, capa por capa, tal como se explicó arriba con el ejemplo del gato.
- **Tiempo de entrenamiento**: es consecuencia directa de las dos filas anteriores — más datos para procesar, y una arquitectura con millones de parámetros para ajustar, significa más tiempo de cómputo. Modelos muy grandes (como los que están detrás de ChatGPT) pueden tardar semanas entrenando en centros de datos con miles de GPUs funcionando en simultáneo.
- **Ejemplos**: la fila de ML (predicción de ventas, scoring bancario) son todos problemas con datos tabulares, del tipo "filas y columnas en Excel" — el terreno donde el ML clásico sigue siendo el estándar de la industria. La fila de DL son todos problemas con datos no estructurados (imágenes, audio, texto libre) — el terreno donde el Deep Learning es, hoy, insustituible.

**Cómo usar esta tabla en clase**: en vez de leerla fila por fila, conviene pedirle al grupo que la lea al revés — dado un escenario ("tengo 500 filas de ventas en un Excel" vs. "tengo 2 millones de fotos de rayos X"), que decidan qué fila les da la pista de qué técnica conviene.

## Filmina 08 — Aplicaciones en la Industria Actual

**Teoría completa (5. Aplicaciones en la industria actual, del docx)**: para entender cuándo usar cada enfoque, tres casos reales:

- **IA Basada en Reglas**: los sistemas de control de temperatura de una oficina o los menús automáticos de un soporte telefónico ("Presione 1 para ventas"). Son eficientes cuando el proceso es 100% predecible.
- **Machine Learning (Scikit-Learn)**: un banco que quiere predecir si un cliente pagará un préstamo basándose en su historial crediticio, edad e ingresos. Aquí los datos están en tablas y el modelo puede explicar por qué tomó una decisión (interpretabilidad).
- **Deep Learning (Redes Neuronales)**: un sistema de conducción autónoma de Tesla que debe identificar peatones, semáforos y otros autos en milisegundos a partir de cámaras de video.

**Por qué cada caso es el ejemplo "correcto" para su capa, y no otro (lo que la filmina no explica)**:

- **El termostato, en detalle**: acá no hace falta ni conviene usar ML. La regla "si la temperatura baja de 20°, encender la calefacción" no tiene ambigüedad ni casos raros que aprender — es una condición matemática simple, siempre igual. Entrenar un modelo para esto sería un desperdicio: necesitaría datos históricos etiquetados para algo que ya se puede resolver con una comparación (`if temp < 20`). Es el mismo criterio de "¿la lógica es fija y conocida, o hay que descubrirla en datos?" que se vio en la Filmina 04.
- **El banco, en detalle — por qué la interpretabilidad no es un detalle técnico menor**: cuando un banco le niega un préstamo a alguien, en la mayoría de los países existe una obligación (legal o al menos de buena práctica) de poder explicarle al cliente *por qué* — "por su nivel de ingresos" o "por su historial de pagos", por ejemplo. Un modelo de ML clásico como una Regresión Logística permite señalar exactamente qué variable pesó cuánto en la decisión (los coeficientes, como se vio en la Filmina 05). Si el banco usara en cambio una red neuronal profunda, probablemente sería más precisa, pero funcionaría como una "caja negra": sabría *que* el cliente fue rechazado, pero no podría explicar *por qué* con la misma claridad — un problema real, no solo académico, en un rubro regulado.
- **Tesla, en detalle — cómo se conecta con lo visto en la Filmina 06**: la cámara del auto captura video, que no es más que una secuencia de imágenes (píxeles). Una red neuronal profunda (del tipo que procesa imágenes) recorre esas capas que ya se explicaron antes —bordes, formas, objetos— para cada fotograma, decenas de veces por segundo, y le entrega a otro sistema la posición de cada peatón, auto o semáforo detectado, para que el auto decida frenar o girar. Ningún conjunto de reglas fijas podría cubrir todas las formas posibles en que puede verse un peatón (ropa, postura, iluminación, ángulo) — es exactamente el mismo argumento de "por qué no alcanzan las reglas" que abrió el Tema 01, ahora aplicado a un caso de altísimo riesgo si falla.

## Filmina 09 — Errores Comunes y Mejores Prácticas

**Teoría completa (6. Errores comunes y mejores prácticas, del docx)**: es fácil dejarse llevar por el entusiasmo de las nuevas tecnologías, pero un buen Data Scientist debe evitar estos errores:

- **"El Deep Learning siempre es mejor": Falso.** Para datos tabulares (hojas de cálculo), el ML clásico suele ser más rápido, barato y preciso que una red neuronal compleja. No usar un cañón para matar un mosquito.
- **Confundir el modelo con los datos**: un modelo de ML es el "estudiante" y el dataset es el "libro de texto". Si el libro es malo (datos sucios, sesgados o incompletos), el estudiante aprenderá mal por muy inteligente que sea.
- **Pensar que la IA "entiende"**: los modelos no tienen conciencia ni "entienden" conceptos. Son funciones matemáticas muy sofisticadas que encuentran correlaciones estadísticas. Si un modelo de lenguaje dice "Hola", no es porque sea educado, sino porque estadísticamente "Hola" es la respuesta más probable tras un saludo.

**Síntesis del Tema 01 (el docx no trae una filmina de cierre propia, pero conviene resumirlo antes de pasar a la práctica)**: en este tema se armó el mapa completo de la Inteligencia Artificial como una relación de inclusión (IA ⊃ ML ⊃ DL, la analogía de las matrioskas), y no como tres técnicas equivalentes entre sí. Se vio que la IA basada en reglas sigue siendo la opción correcta cuando el proceso es 100% predecible (el termostato); que el Machine Learning entra cuando hace falta aprender un patrón de datos tabulares sin poder escribir esa regla a mano (el banco, con la ventaja extra de la interpretabilidad); y que el Deep Learning es necesario específicamente para datos no estructurados como imágenes o audio, donde ni un experto humano podría enumerar las reglas (Tesla). El hilo conductor de los tres errores comunes de la Filmina 09 es el mismo: no sobrestimar la tecnología — ni asumir que "más complejo es siempre mejor" (Deep Learning), ni que el modelo puede compensar datos malos, ni que hay algo parecido a comprensión real detrás de una predicción.

Con este mapa ya armado, el resto de la clase se apoya en él: el Tema 02 profundiza *cómo* aprende exactamente el Machine Learning (con o sin etiquetas, o por prueba y error), y los Temas 04 y 05 bajan a la arquitectura concreta de Scikit-Learn para implementarlo en código.

**👉 En el Colab — Bloque 1, primeras dos celdas.** Acá arranca la parte práctica: setup del notebook y el "rompehielo" que vive en el dataset de propiedades, antes de seguir con el Tema 02.

**Qué es `propiedades_sueca_ml.csv` (el dataset nuevo de hoy, para tener el contexto antes de ver cualquier número)**: son 300 propiedades en venta en Sueca, una localidad costera de Valencia, España — ya limpio y listo para entrenar (a diferencia del de natalidad, este no se usó en el Repaso, es específico de la parte de Machine Learning). Cada fila es una propiedad, con estas columnas:

| Columna | Qué es |
|---|---|
| `id_propiedad` | Identificador único de la propiedad — no es una feature real, solo sirve para referenciar la fila. |
| `barrio` | La zona dentro de Sueca: Playa, Centro, Mareny, Poble Nou, Els Racons o Estació. |
| `superficie_m2` | Metros cuadrados de la propiedad (van de 35 a 166 en este dataset). |
| `ambientes` | Cantidad de ambientes/habitaciones. |
| `antiguedad_anios` | Antigüedad de la construcción, en años. |
| `con_cochera` / `con_balcon` | Indicadores 0/1: si la propiedad tiene cochera y/o balcón. |
| `score_amenities` | Un puntaje de 0 a 2 que resume comodidades (probablemente la suma de `con_cochera` + `con_balcon`). |
| `precio_eur` | El precio de venta en euros (entre 45.000 y 245.800) — **esta es la columna que se va a predecir** en todo lo que sigue. |
| `precio_por_m2` | Precio dividido por superficie — **ojo con esta**: se calculó *a partir* del precio, así que usarla como feature sería hacer trampa (es justamente la trampa de Data Leakage que se ve más adelante, en el Tema 05). |

**Antes del código, un repaso de qué es Scikit-Learn** (ya se lo nombró de pasada en el Repaso de Clase 06, con `StandardScaler` y `PCA`): es la librería estándar de Python para Machine Learning — reúne, ya implementados, probados y optimizados, los algoritmos de preprocesamiento (escalado), los **modelos** que hacen predicciones (regresión, clasificación, árboles de decisión) y las **métricas** para medir qué tan bien funcionan, todo bajo la misma lógica de tres pasos que se formaliza recién en el Tema 04: **instanciar** el objeto, **ajustarlo** con `.fit()` (aprende de los datos), y **aplicarlo** con `.transform()` o `.predict()`. Hasta ahora solo se había visto la parte de preprocesamiento de esta librería; en el setup de abajo aparecen, por primera vez, los **modelos** (`LinearRegression`, `DecisionTreeRegressor`) y las **métricas** (`mean_absolute_error`, `r2_score`) — el kit completo que se usa en el resto de la clase.

**Setup inicial. Qué hace en general**: importa todas las librerías que se van a usar en el resto del notebook, y carga el dataset de propiedades.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sns.set_style("whitegrid")

df = pd.read_csv("propiedades_sueca_ml.csv")
df.head(10)
```
**Línea por línea:** los `import` traen, en orden, Pandas/NumPy (datos), Matplotlib/Seaborn (gráficos), y de `sklearn` puntualmente lo que se va a usar hoy: `train_test_split` (dividir datos), `StandardScaler` (escalar), `LinearRegression` y `DecisionTreeRegressor` (los dos modelos de la clase), y `mean_absolute_error`/`r2_score` (las métricas). `sns.set_style("whitegrid")` fija un estilo visual con grilla suave para todos los gráficos que vengan después. `pd.read_csv(...)` carga el dataset en `df`; `.head(10)` muestra las primeras 10 filas para chequear que cargó bien.

**El "rompehielo". Qué hace en general**: muestra 5 filas del dataset sin la columna que hay que predecir, para que el grupo vea el problema antes de que se lo expliquen.
```python
df.drop(columns=["precio_eur", "precio_por_m2", "id_propiedad"]).sample(5, random_state=1)
```
**Línea por línea:** `.drop(columns=[...])` saca del DataFrame las columnas que serían la "respuesta" (`precio_eur`, `precio_por_m2`) y el identificador (que no es una feature real); `.sample(5, random_state=1)` muestra 5 filas al azar, pero siempre las mismas 5 gracias a la semilla fija. **Pregunta para el grupo, tal como la trae el notebook**: "si tuviera que escribir un programa con reglas fijas (`SI superficie > 100 Y barrio == Centro, ENTONCES precio > 200.000`) para estimar el precio de estas propiedades, ¿cuántas reglas necesitaría? ¿Alcanzaría alguna vez?" — la misma pregunta que abre este tema, pero vivida en código antes de nombrarla.

---

# Tema 02 — Tipos de Aprendizaje: Supervisado, No Supervisado y por Refuerzo (Filminas 10-18)

## Filmina 10 — División de Tema

**Teoría completa de apertura (del docx)**: imaginá que querés enseñarle a un niño a identificar diferentes tipos de frutas. Hay varias estrategias posibles: podrías mostrarle una manzana y decirle repetidamente "esto es una manzana"; podrías darle una cesta llena de frutas mezcladas y pedirle que las agrupe por su parecido sin decirle qué son; o podrías dejarlo en un huerto y darle un premio cada vez que recoja una fruta madura y deliciosa. En el mundo del Machine Learning, estas tres estrategias representan los tres grandes paradigmas o "modos" en los que una máquina puede aprender de los datos. Entender estos tres tipos de aprendizaje es fundamental porque determina todas las decisiones futuras de un científico de datos: desde qué algoritmo elegir hasta cómo medir si el modelo realmente funciona.

**Por qué son justo estos 3 paradigmas, y no otra clasificación cualquiera**: no es una lista arbitraria — son, hasta hoy, las tres formas fundamentales de diseñar un sistema que aprenda de datos, según qué información hay disponible *antes* de empezar. Si existen ejemplos con la respuesta correcta ya conocida, es Supervisado. Si solo hay datos, sin ninguna respuesta, es No Supervisado. Si no hay ni siquiera un dataset previo, sino un entorno con el que el sistema puede interactuar y recibir una señal de éxito o fracaso, es Por Refuerzo. En la práctica, cualquier proyecto real de Machine Learning empieza respondiendo esta misma pregunta, antes de pensar en qué algoritmo usar.

**Un cuarto ejemplo de la misma analogía, en un dominio bien distinto al de las frutas**: pensá en tres formas de aprender a jugar al ajedrez. **Supervisado** sería estudiar miles de partidas ya jugadas por grandes maestros, viendo qué jugada siguió a cada posición del tablero. **No Supervisado** sería mirar miles de partidas y notar que ciertos patrones de apertura se repiten, sin que nadie te diga cuáles son "buenos" o "malos" movimientos. **Por Refuerzo** sería jugar partida tras partida contra un rival, ganando o perdiendo, y ajustar la estrategia según el resultado — sin estudiar ninguna partida ajena de antemano. Este último es, de hecho, exactamente cómo aprendió AlphaGo a jugar al Go (se retoma con más detalle en la Filmina 14).

**Dos ejemplos más, cortitos, por si con el de las frutas y el ajedrez todavía no termina de cerrar la idea:**

- **Manejar un auto**: Supervisado es un instructor corrigiéndote en cada maniobra, comparando lo que hiciste contra lo que "hay que hacer". No Supervisado es notar, mirando el tráfico de una ciudad nueva sin que nadie te explique nada, que hay patrones de comportamiento que se repiten. Por Refuerzo es aprender a estacionar en paralelo probando una y otra vez, chocando o no un cono, hasta perfeccionar el movimiento.
- **Organizar una biblioteca**: Supervisado es recibir una lista con el género ya asignado a cada libro (Ficción, Historia, Ciencia) y aprender a clasificar libros nuevos según esas mismas categorías. No Supervisado es recibir miles de libros sin ninguna etiqueta y agruparlos vos mismo por temas que notás en común. Por Refuerzo es un sistema que prueba distintos ordenamientos de estantería y recibe una señal positiva cada vez que un usuario encuentra rápido lo que busca.

**Pregunta para tirar a la clase, antes de nombrar los tres paradigmas formalmente**: de las tres estrategias con las frutas, ¿cuál te parece que un niño aprendería más rápido? ¿Y cuál requiere más trabajo de preparación *antes* de siquiera empezar a enseñar?

**Respuesta esperada (para tener un esbozo a mano, no solo para que quede abierta)**: probablemente **aprende más rápido** con la manzana señalada y nombrada una y otra vez (el método Supervisado) — tiene una respuesta correcta explícita en cada ejemplo, así que no tiene que inferir nada por su cuenta, solo memorizar la asociación. Agrupar frutas por parecido (No Supervisado) puede ser rápido para notar que "hay grupos", pero el niño todavía no sabe cómo se *llama* cada fruta — aprendió una estructura, no una respuesta. Aprender en el huerto a pura prueba y error (Por Refuerzo) es probablemente el más lento de los tres: exige cometer errores (agarrar una fruta verde o podrida) antes de descubrir el patrón de "qué hace que una fruta esté madura".

Y acá aparece la tensión interesante, la que vale la pena remarcar: **requiere más preparación previa** justo el método que aprende más rápido — el Supervisado. Alguien (un experto) tuvo que juntar y etiquetar cada ejemplo de antemano, diciendo explícitamente "esto ES una manzana" — es el mismo "costo del etiquetado" que se retoma más adelante, en el Tema 03. El No Supervisado, en cambio, no necesita ninguna etiqueta previa: alcanza con juntar la cesta de frutas mezcladas, sin trabajo extra de preparación. El Por Refuerzo tampoco necesita un dataset preparado de antemano, pero sí exige diseñar con cuidado el "entorno" (el huerto) y la regla de premios ("dale algo rico si la fruta está madura") — un tipo distinto de trabajo previo, de diseño y no de etiquetado.

**La conclusión que vale la pena que se lleve la clase**: el método que aprende más rápido no es gratis — exige más trabajo humano antes de arrancar. No hay una estrategia "mejor" en abstracto, hay un trade-off entre cuánto se tarda en preparar el aprendizaje y cuánto se tarda el aprendizaje en sí — la misma decisión que un Data Scientist enfrenta al elegir entre los tres paradigmas en un proyecto real.

## Filmina 11 — El Concepto de "la Señal de Aprendizaje"

**Teoría completa (1. El concepto de "La Señal de Aprendizaje", del docx)**: antes de profundizar, hace falta entender un concepto clave: la **etiqueta (label)**. En Data Science se suele trabajar con tablas. Imaginá una tabla de datos de departamentos en alquiler: las **Features (Características o Entradas)** son las columnas como metros cuadrados, cantidad de habitaciones, barrio, tiene balcón — la información que se usa para alimentar al modelo. El **Label (Etiqueta o Salida)** es el resultado que se quiere predecir, por ejemplo el precio del alquiler. La presencia o ausencia de esta "etiqueta" es lo que define, en gran medida, ante qué tipo de aprendizaje se está.

**Más ejemplos de Features/Label, en dominios distintos al de los departamentos** (para que el patrón quede claro más allá de un solo caso):

- **Hospital**: Features = edad del paciente, síntomas registrados, resultados de análisis de sangre. Label = si el paciente tiene o no una enfermedad determinada (el caso de la Filmina 19).
- **E-commerce**: Features = historial de compras, tiempo en el sitio, dispositivo usado. Label = si el usuario compra o no en esa visita.
- **Nuestro propio dataset de propiedades (Bloque 1 del Colab)**: Features = `superficie_m2`, `ambientes`, `antiguedad_anios`, `score_amenities`. Label = `precio_eur` — el mismo par que se va a usar en el código, del Bloque 2 en adelante.

**Un ejercicio mental rápido para fijar la idea**: en la tabla de propiedades, si en vez de predecir el precio quisiéramos simplemente **agrupar** propiedades parecidas entre sí (sin decirle al modelo cuál es el precio de ninguna), `precio_eur` dejaría de ser el Label — pasaría a ser una feature más, o directamente se podría sacar de la tabla. El **mismo dataset**, con o sin esa columna marcada como "la respuesta", cambia de Supervisado a No Supervisado. La tabla no cambia; lo que cambia es qué se decide hacer con ella.

## Filmina 12 — Aprendizaje Supervisado: "el Estudiante con Profesor"

**Teoría completa (2. Aprendizaje Supervisado, del docx)**: el Aprendizaje Supervisado es el paradigma más común en la industria. Se llama así porque el modelo cuenta con un "profesor" (el dataset etiquetado) que le proporciona ejemplos de la vida real junto con su respuesta correcta. Al modelo se le entregan miles de ejemplos con sus respectivas soluciones; el algoritmo intenta encontrar la relación matemática entre las features y la etiqueta. Una vez que "aprende" esa relación, se le entregan datos nuevos (sin etiqueta) para que prediga el resultado.

Las dos grandes tareas:
- **Clasificación**: predice una categoría o clase discreta (Sí/No, A/B/C). Ejemplo real: **Detección de Spam en Gmail** — el "profesor" le dio a Google millones de correos marcados manualmente como "Spam" o "No Spam". El modelo aprendió que palabras como "Gratis", "Gane dinero ya" o remitentes extraños suelen ser Spam.
- **Regresión**: predice un valor numérico continuo. Ejemplo real: **Precio de una vivienda** — el modelo analiza datos históricos de casas vendidas (m², ubicación, año) y sus precios finales, y estima el precio de una casa nueva.

**Diferenciando Clasificación de Regresión con una pregunta simple**: no depende del algoritmo ni de lo "difícil" que sea el problema — depende únicamente de **qué tipo de valor es el Label**. Si el Label es una de un conjunto cerrado de categorías (Spam/No Spam, Aprobado/Rechazado), es Clasificación. Si el Label es un número que podría tomar, en principio, cualquier valor dentro de un rango (un precio, una temperatura, un tiempo de entrega), es Regresión. El propio dataset de propiedades de la clase de hoy es un ejemplo de Regresión: `precio_eur` no es "categoría A o B", es un número continuo entre 45.000 y 245.800.

**¿Por qué el Aprendizaje Supervisado es, en la práctica, el paradigma más usado en la industria?** Porque casi cualquier decisión de negocio se puede formular como "dado lo que sé hoy (features), ¿qué va a pasar o qué es cierto (label)?" — y eso es exactamente la definición de un problema supervisado. Prácticamente todos los ejemplos que van a ir apareciendo el resto de la clase (bancos, e-commerce, salud, logística) son variantes de esta misma pregunta.

**¿Por qué importa?** Porque la mayoría de las preguntas de negocio son supervisadas: "¿este cliente se va a dar de baja?", "¿cuánto va a vender mi tienda el próximo mes?", "¿es esta transacción un fraude?".

## Filmina 13 — Aprendizaje No Supervisado: "Buscando Estructura en el Caos"

**Teoría completa (3. Aprendizaje No Supervisado, del docx)**: ¿qué pasa si no hay etiquetas? Imaginá tener los datos de compras de 10 millones de clientes, pero sin saber quiénes son ahorradores, quiénes compran por impulso o quiénes prefieren productos de lujo — no hay una "respuesta correcta" previa. Acá entra el Aprendizaje No Supervisado: el modelo no intenta predecir nada, en su lugar explora los datos para encontrar **patrones ocultos o estructuras intrínsecas**.

Las tareas principales:
- **Clustering (Agrupamiento)**: agrupa los datos en "clusters" donde los elementos de un mismo grupo se parecen mucho entre sí y son muy distintos a los de otros grupos. Ejemplo real: **Segmentación de clientes en una app de música** — Spotify agrupa usuarios no por edad, sino por comportamiento: "usuarios que escuchan podcasts de noche", "usuarios que solo escuchan hits del momento". Esto permite campañas de marketing ultra-específicas sin que nadie haya etiquetado previamente a los usuarios.
- **Reducción de Dimensionalidad**: a veces hay demasiada información (cientos de columnas) y el modelo busca simplificar los datos quedándose solo con lo más importante, sin perder la esencia.

**Diferenciando Clustering de Reducción de Dimensionalidad (las dos tareas de este paradigma)**: aunque las dos son "No Supervisadas", resuelven preguntas distintas. El **Clustering** agrupa **filas** parecidas entre sí (¿qué clientes se parecen?). La **Reducción de Dimensionalidad** combina **columnas** parecidas entre sí (¿qué variables están diciendo, en el fondo, lo mismo?) — es exactamente lo que ya se hizo con `PCA` en el Repaso de Clase 06, comprimiendo 25 columnas de años en 2. Son complementarias: en un proyecto real es común primero reducir dimensiones y después clusterizar sobre ese resultado ya simplificado.

**Otro ejemplo de Clustering, en un dominio distinto a Spotify**: un supermercado con los datos de compra de sus 50.000 clientes (qué compran, cuándo, cuánto gastan) puede agrupar automáticamente a "los que hacen una compra grande semanal", "los que compran de a poco varias veces por semana" y "los que solo compran en oferta" — sin que nadie les haya puesto esas etiquetas de antemano. Recién después, un humano le pone un nombre de negocio a cada grupo que el algoritmo encontró.

**Un error común**: muchos estudiantes creen que el aprendizaje no supervisado no tiene un objetivo. ¡Error! El objetivo es **descubrir**, no predecir. Es como organizar una colección de miles de fotos familiares por colores predominantes sin saber quién aparece en ellas; al final, hay una estructura que antes no se veía.

## Filmina 14 — Aprendizaje por Refuerzo: "Aprender por Ensayo y Error"

**Teoría completa (4. Aprendizaje por Refuerzo, del docx)**: este es el paradigma más distinto de los tres, y es la base de los avances más espectaculares en IA reciente, como los coches autónomos o los sistemas que vencen a campeones mundiales de ajedrez. A diferencia del supervisado (donde hay respuestas) o el no supervisado (donde hay patrones), acá un **Agente** (el algoritmo) interactúa con un **Entorno**.

**¿Cómo funciona?** El agente toma una **Acción**. Dependiendo de si esa acción lo acerca o lo aleja de su objetivo, recibe una **Recompensa** (+) o una **Penalización** (−). El objetivo del agente es maximizar la recompensa acumulada a largo plazo. Es exactamente como se aprende a jugar a un videojuego: no se nace sabiendo que tocar la lava mata; se prueba, se pierden puntos (penalización), y el cerebro aprende a no hacerlo de nuevo.

**Diferenciando los 4 términos clave, con el ejemplo del videojuego**: el **Agente** es el personaje que vos manejás. El **Entorno** es el nivel del juego, con sus reglas y obstáculos. La **Acción** es cada movimiento que el agente decide hacer (saltar, avanzar). La **Recompensa/Penalización** es lo que el juego le devuelve después de esa acción (sumar puntos, perder una vida). Ninguno de los tres paradigmas anteriores tiene estos 4 elementos actuando en conjunto — el Supervisado y el No Supervisado trabajan sobre un dataset ya fijo, mientras que acá el propio agente va generando sus datos de aprendizaje a medida que actúa.

**Ejemplos emblemáticos**:
- **AlphaGo de Google DeepMind**: aprendió a jugar al Go (un juego de estrategia milenario) jugando millones de partidas contra sí mismo. No tenía un archivo CSV con las "mejores jugadas"; aprendió qué movimientos llevaban a la victoria mediante el refuerzo constante.
- **Robótica Industrial**: un brazo robótico en una fábrica puede aprender la trayectoria más eficiente para mover una pieza mediante pequeñas recompensas cada vez que el movimiento es fluido y preciso.
- **Un ejemplo más actual, conectando con la Filmina 06**: ChatGPT (mencionado antes como ejemplo de Deep Learning) no solo se entrenó leyendo texto — en una etapa final se ajustó con Aprendizaje por Refuerzo a partir de feedback humano (la técnica se conoce como RLHF): personas calificaban qué respuestas del modelo eran mejores que otras, y el modelo ajustaba su comportamiento para maximizar esas calificaciones positivas — el mismo patrón Acción → Recompensa, aplicado a generar texto en vez de mover un brazo robótico o jugar al Go.

## Filmina 15 — Cuadro Comparativo: ¿Cuál Elegir?

**Teoría completa (5. Cuadro Comparativo, del docx)**: para un científico de datos principiante, saber distinguir cuál usar es el primer paso de cualquier proyecto.

| Característica | Supervisado | No Supervisado | Por Refuerzo |
|---|---|---|---|
| Datos iniciales | Etiquetados (Entrada + Salida) | No etiquetados (Solo Entrada) | Sin datos previos; aprende interactuando |
| Objetivo | Predecir resultados / Clasificar | Encontrar patrones / Agrupar | Tomar decisiones secuenciales |
| Feedback | Directo (Error vs. Respuesta real) | No tiene feedback explícito | Recompensa o Penalización |
| Analogía | Estudiar con el solucionario | Ordenar un ropero desordenado | Aprender a montar en bicicleta |

**El porqué detrás de cada fila (lo que la tabla no explica por sí sola)**:

- **Datos iniciales**: es la fila que responde la pregunta de diagnóstico más rápida de todas — ¿el dataset tiene una columna marcada como "la respuesta correcta"? Si sí, Supervisado. Si hay datos pero ninguna respuesta marcada, No Supervisado. Si ni siquiera hay un dataset (solo un entorno con el que interactuar), Por Refuerzo.
- **Objetivo**: en Supervisado y No Supervisado el resultado final es algo relativamente estático (una predicción, un conjunto de grupos). En Por Refuerzo el "resultado" es una **estrategia** — una forma de actuar ante distintas situaciones, no un valor único.
- **Feedback**: en Supervisado el feedback es inmediato y exacto (se sabe la respuesta correcta de cada ejemplo apenas se entrena). En Por Refuerzo el feedback puede ser tardío — a veces una buena jugada de ajedrez solo se sabe si fue buena varias jugadas después, cuando se gana o se pierde la partida. En No Supervisado directamente no hay ninguna respuesta "correcta" contra la cual comparar.
- **Analogía**: vale la pena remarcar en voz alta por qué cada analogía es exacta y no solo poética — "estudiar con el solucionario" tiene la respuesta correcta de cada ejercicio a la vista (como el Label); "ordenar un ropero desordenado" no tiene ninguna instrucción de cómo ordenarlo, solo se buscan agrupaciones que tengan sentido (como el Clustering); "aprender a andar en bicicleta" no se resuelve leyendo un manual, sino cayéndose y corrigiendo el equilibrio una y otra vez (como la Recompensa/Penalización).

## Filmina 16 — Errores y Confusiones Comunes

**Teoría completa (6. Errores y Confusiones Comunes, del docx)**:

- **"¿Puedo usar clustering para clasificar correos?"**: No. El clustering agrupará correos parecidos, pero no sabrá cuál es "Spam". Para eso hacen falta etiquetas previas (Supervisado).
- **"El aprendizaje por refuerzo es solo prueba y error"**: No es azar. El algoritmo usa estructuras matemáticas (como Redes Neuronales) para decidir qué camino probar basándose en experiencias pasadas, para ser cada vez más inteligente.
- **"Si tengo muchos datos el modelo será perfecto"**: si las etiquetas en el aprendizaje supervisado están mal (por ejemplo, se marcaron correos buenos como spam por error), el modelo aprenderá a equivocarse. La calidad del dato manda sobre la cantidad.

## Filmina 17 — Síntesis y Conexiones

**Teoría completa (7. Síntesis y Conexiones, del docx)**: el Machine Learning no es una "caja negra" única, sino un conjunto de herramientas adaptables. Se usa **Aprendizaje Supervisado** si se tiene la respuesta histórica y se quiere predecir el futuro. Se usa **Aprendizaje No Supervisado** si se quiere explorar los datos y entender cómo se agrupan. Se usa **Aprendizaje por Refuerzo** si se necesita que un sistema aprenda a tomar decisiones complejas en un entorno dinámico. En las próximas unidades se implementan estos conceptos usando Scikit-Learn, la librería estándar de Python para ML, que utiliza una estructura lógica muy clara para manejar estos tipos de aprendizaje.

## Filmina 18 — Práctica (no entregable): Diagnóstico de 5 Casos

**Instrucciones completas (del docx)**: analizar 5 casos de uso de la industria y, para cada uno, indicar el **Tipo de Aprendizaje** (Supervisado —Clasificación o Regresión—, No Supervisado, o Por Refuerzo), la **Justificación** (¿existen etiquetas? ¿hay respuesta correcta? ¿se busca estructura? ¿hay recompensas?) y una **Métrica sugerida** para medir el éxito.

- **Caso A**: un banco quiere saber si un cliente que solicita un préstamo lo devolverá o no, basándose en el historial de pagos previos de miles de clientes antiguos.
- **Caso B**: una cadena de supermercados tiene datos de 50.000 clientes (compras, horarios, edad) y quiere encontrar grupos de "estilos de vida" para orientar sus folletos de ofertas.
- **Caso C**: una empresa de logística quiere entrenar a un vehículo autónomo para que aprenda a estacionarse solo en un depósito, dándole puntos positivos cuando queda derecho y restando puntos si choca.
- **Caso D**: una inmobiliaria quiere crear una herramienta que estime el valor de mercado de los departamentos basándose en metros cuadrados, ubicación y antigüedad.
- **Caso E**: un hospital tiene miles de imágenes de rayos X marcadas por médicos especialistas como "Normal" o "Infección". Quieren un sistema que ayude a los médicos a priorizar urgencias.

Cierra con una reflexión final: cuál de los tres paradigmas parece más complejo de implementar, y por qué.

**Errores comunes a evitar (del docx)**: confundir Clustering (No Supervisado) con Clasificación (Supervisado) — si el problema menciona datos ya "marcados" o "históricos con resultado", es Supervisado. Olvidar que en el Aprendizaje por Refuerzo no hay un dataset estático inicial, sino un proceso de interacción constante.

**👉 En el Colab — cierre del Bloque 1.** Después del mapa IA → ML → DL y de esta teoría de tipos de aprendizaje, el notebook no agrega código nuevo — son celdas de texto que retoman lo ya visto en las Filminas 03-15 (mapa) y en este Tema 02. Sí cierra con un **mini-quiz oral** para resolver en vivo con el grupo, antes de pasar a Scikit-Learn:

- Agrupar clientes de un supermercado por hábitos de compra, sin categorías previas.
- Predecir si un mail es spam, usando miles de mails ya marcados.
- Un termostato inteligente que aprende a ahorrar energía probando distintas temperaturas.

**Ojo**: estos son exactamente los 3 casos que después reaparecen como "Tarea 1" en el Solucionario del Bloque 4 (al final del Tema 06) — el notebook los plantea acá sin responder, y da la respuesta recién al cierre de la clase. Si se resuelven ya acá, la Tarea 1 del plenario final queda como repaso en vez de ejercicio nuevo — vale la pena decidir en qué momento conviene resolverlos según el ritmo del grupo.

---

# Tema 03 — Aplicaciones Prácticas de ML: Del Algoritmo al Impacto Real (Filminas 19-27)

## Filmina 19 — División de Tema

✅ **Entregable evaluado del Módulo** — ver el detalle completo en la sección "Pre-entrega" al final de esta guía. El resto de las prácticas de la clase son guiadas y no evaluables; esta es la que se corrige y suma al proyecto final.

**Teoría completa de apertura (del docx)**: imaginá ser el dueño de una tienda de comercio electrónico que crece rápidamente. Al principio se podía saludar a cada cliente y recomendarle productos personalmente, pero con 100.000 clientes diarios es físicamente imposible que una persona (o incluso un equipo grande) analice el comportamiento de cada usuario para ofrecerle lo que busca en el momento justo. Ahí entra el Machine Learning: no como un concepto de ciencia ficción, sino como una herramienta práctica que automatiza la toma de decisiones a escala.

## Filmina 20 — El Cambio de Paradigma: de Reglas a Patrones

**Teoría completa (1. El Cambio de Paradigma: De Reglas a Patrones, del docx)**: para entender las aplicaciones prácticas, primero hace falta entender qué problema vino a solucionar el ML. **El Enfoque Tradicional (Basado en Reglas)**: antes del auge del ML, para que una computadora detectara correos de spam había que escribir cientos de reglas manuales ("si el correo contiene la palabra 'GRATIS' en mayúsculas, marcar como spam"; "si el remitente no está en la lista de contactos y pide dinero, marcar como spam"). **El problema**: los estafadores son creativos. Empezarían a escribir "G.R.A.T.I.S" o usar sinónimos; el programador tendría que actualizar las reglas constantemente hasta que el sistema se vuelve tan complejo que se rompe. **El Enfoque de Machine Learning**: en lugar de programar reglas, se le dan a la computadora miles de ejemplos de correos spam y legítimos, y el sistema aprende a identificar los patrones por sí solo. Si los estafadores cambian su táctica, simplemente se alimenta al modelo con los nuevos ejemplos y este se adapta. **Concepto clave**: el Machine Learning es la herramienta ideal cuando las reglas son demasiado numerosas, cambian con el tiempo, o son imposibles de explicar con palabras (como describir cómo se reconoce la cara de un amigo).

## Filmina 21 — El Ciclo de Vida de una Aplicación de ML

**Teoría completa (2. El Ciclo de Vida de una Aplicación de ML, del docx)**: en la práctica, implementar Machine Learning no es solo "entrenar un modelo". Es un proceso sistémico de cuatro grandes etapas:

- **Datos**: la materia prima. Sin datos históricos de calidad (ejemplos de lo que pasó en el pasado), no hay aprendizaje.
- **Entrenamiento**: el proceso donde el algoritmo analiza los datos para encontrar correlaciones. Acá se crea el "Modelo".
- **Evaluación**: antes de lanzar el modelo al mundo, se lo prueba con datos que nunca ha visto, para asegurarse de que realmente aprendió y no solo memorizó.
- **Uso Real (Inferencia)**: el modelo se integra en una aplicación (como una app de banco) para tomar decisiones sobre datos nuevos en tiempo real.

## Filmina 22 — Aplicaciones Reales: ¿Quién lo Usa y para Qué?

**Teoría completa (3. Aplicaciones Reales, del docx)**: para que el ML deje de ser una "caja negra", cuatro ejemplos de empresas conocidas — cada una resuelve un problema de negocio específico mediante la detección de patrones.

- **A. Sistemas de Recomendación — el caso Netflix**: Netflix no muestra películas al azar. Su sistema de ML analiza el historial de visualización, qué géneros se prefieren, a qué hora se conecta el usuario y qué personas con gustos similares han visto. ¿Qué predice? La probabilidad de que el usuario vea al menos el 70% de un título. Valor práctico: mantiene a los usuarios suscritos al reducir la fatiga de decisión.
- **B. Clasificación de Seguridad — Gmail y el Spam**: Google utiliza modelos que analizan el texto, los metadatos y la reputación del remitente para filtrar correos no deseados. ¿Qué predice? Una puntuación del 0 al 1, donde 1 es "definitivamente spam". Valor práctico: ahorra tiempo y protege de estafas (phishing) de forma automática.
- **C. Logística y Movilidad — Uber**: Uber utiliza ML para predecir el futuro cercano en la ciudad. ¿Qué predice? El tiempo estimado de llegada (ETA), la demanda de viajes en una zona específica (para activar precios dinámicos) y la ruta más eficiente. Valor práctico: optimiza el uso de los vehículos y mejora la experiencia del usuario.
- **D. Salud — Diagnóstico por Imagen**: en medicina se entrenan modelos de Deep Learning con miles de radiografías o resonancias marcadas por expertos. ¿Qué predice? La presencia de anomalías, como un tumor o una fractura, a veces con mayor precisión o velocidad que el ojo humano cansado. Valor práctico: sirve como una "segunda opinión" para los doctores, permitiendo detecciones tempranas.

## Filmina 23 — Más Casos de la Industria

**Teoría completa (tabla del docx)**:

| Caso de Uso | Tecnología/Empresa | Función Principal |
|---|---|---|
| Detección de Fraude | BBVA / PayPal | Identifica transacciones inusuales en milisegundos para bloquear robos |
| Predicción de Demanda | Zara / Amazon | Estima cuántas tallas "M" se venderán en una tienda para evitar falta de inventario |
| Mantenimiento Predictivo | General Electric | Predice cuándo fallará una turbina de avión antes de que ocurra, basándose en sensores de vibración |

## Filmina 24 — Errores Comunes y Falsas Expectativas

**Teoría completa (4. Errores Comunes y Falsas Expectativas, del docx)**: cuando un estudiante o una empresa comienza con ML, es fácil caer en trampas conceptuales.

- **Error 1: "El Machine Learning es Magia"**. Realidad: el ML es estadística aplicada a gran escala. No "entiende" conceptos filosóficos. Si se entrena un modelo para predecir ventas usando solo datos de temperatura, el modelo encontrará una relación, aunque no tenga sentido lógico. El ML detecta **correlaciones**, no necesariamente **causalidad**.
- **Error 2: "Más datos siempre es mejor"**. Realidad: los datos malos producen modelos malos (*Garbage In, Garbage Out*). Si los datos están sesgados, el modelo será sesgado. Ejemplo: si un algoritmo de contratación se entrena con datos históricos de una empresa que nunca contrató mujeres, el modelo aprenderá que "ser hombre" es un patrón de éxito — un error grave y discriminatorio.
- **Error 3: "Un modelo con 99% de precisión es perfecto"**. Realidad: depende del contexto. En la detección de una enfermedad rara que afecta a 1 de cada 100 personas, si el modelo siempre dice "estás sano", ¡tendrá un 99% de precisión! Pero habrá fallado en detectar al único enfermo, que era su propósito principal. En la práctica, hay que elegir la métrica que realmente importe para el problema.

## Filmina 25 — Terminología Clave para Profesionales

**Teoría completa (5. Terminología Clave, del docx)**: para hablar el lenguaje del sector, hay que dominar estos términos en su contexto práctico:

- **Features (Características)**: las variables que se le dan al modelo para que aprenda. En el caso de una casa: metros cuadrados, barrio, número de habitaciones.
- **Label (Etiqueta)**: lo que se quiere predecir. En el caso de la casa, el "precio".
- **Inferencia**: el acto de usar el modelo ya entrenado para obtener una respuesta. "Hacer una inferencia" es pedirle al modelo que prediga algo sobre un dato nuevo.
- **Sobreajuste (Overfitting)**: ocurre cuando el modelo memoriza los datos de entrenamiento tan bien que no sabe qué hacer cuando ve algo un poco diferente. Es como un estudiante que memoriza las respuestas del examen pero no entiende la materia: si cambia la pregunta, reprueba.

## Filmina 26 — Síntesis y Conexión

**Teoría completa (6. Síntesis y Conexión, del docx)**: el Machine Learning no es un ente aislado, sino un componente dentro de un sistema más grande. Su valor no reside en la complejidad del algoritmo, sino en la **decisión que ayuda a tomar**: ¿es este correo spam? ¿qué precio debe tener este producto? ¿está este paciente en riesgo? En las próximas unidades se pasa de la teoría a la herramienta: Scikit-Learn, la librería estándar de la industria, con una estructura lógica (estimadores y transformadores) que permite convertir datos en predicciones reales. **Pregunta para reflexionar**: mirá las aplicaciones de tu teléfono ahora mismo — ¿en cuáles creés que hay un modelo de Machine Learning trabajando en segundo plano? Probablemente, en casi todas.

## Filmina 27 — Práctica (no entregable): Perfilado de una Solución de ML

**Instrucciones completas (del docx)**: elegir un proceso del trabajo actual, estudio o vida cotidiana que hoy se haga manualmente o mediante reglas simples (categorizar facturas, decidir qué publicar en redes sociales, predecir cuándo ir al supermercado). Definir los componentes: **La Tarea** (¿qué decisión o predicción se quiere automatizar?), **las Features** (al menos 5 datos de entrada que el modelo necesitaría — fecha, monto, palabra clave, hora del día), **la Label** (¿cuál es la "respuesta correcta" que el modelo debe aprender a predecir?). Anticipar desafíos: ¿de dónde saldrían los datos históricos? ¿qué sesgo (bias) podría tener el modelo si los datos no son representativos? ¿cuál sería el "costo del error"?

---

# Tema 04 — Scikit-Learn por Dentro: Estimators, Transformers y Predictors (Filminas 28-33)

## Filmina 28 — División de Tema

**Teoría completa de apertura (del docx)**: entrar en el mundo de Scikit-Learn (o `sklearn`) es como entrar en una fábrica perfectamente organizada. No importa si se quiere predecir el precio de una casa, clasificar correos como spam o agrupar clientes: **todos los objetos se comportan de la misma manera**. Esta uniformidad es la mayor fortaleza de la librería, y se basa en tres conceptos fundamentales: **Estimators**, **Transformers** y **Predictors**.

## Filmina 29 — Estimators: la Base de Todo (el Alumno)

**Teoría completa (1. Estimators, del docx)**: un **Estimador** es cualquier objeto que aprende de los datos. El método estrella es `.fit()`. Imaginá que el estimador es un alumno: cuando se ejecuta `model.fit(X, y)`, el alumno abre su libro (los datos) y empieza a estudiar. **Parámetros**: son las "instrucciones" que se le dan al alumno antes de estudiar (ej. "estudia rápido" o "fíjate mucho en los detalles"). Se definen al crear el objeto: `Model(opcion=True)`. **Atributos aprendidos**: es lo que el alumno anotó en su cuaderno tras estudiar. En Scikit-Learn, estos atributos siempre terminan en un guion bajo, como `model.coef_` o `scaler.mean_`.

## Filmina 30 — Transformers: Transformando la Realidad (el Filtro)

**Teoría completa (2. Transformers, del docx)**: un **Transformer** es un tipo de estimador que, tras "estudiar" los datos, puede modificarlos. Para ello usa el método `.transform()`. ¿Para qué sirve? Para normalizar datos, rellenar valores faltantes (imputación) o convertir categorías en números. `fit_transform()`: una forma rápida de aprender la receta y aplicarla en el mismo paso.

## Filmina 31 — Predictors: Tomando Decisiones (el Juez)

**Teoría completa (3. Predictors, del docx)**: un **Predictor** es un estimador capaz de hacer pronósticos sobre datos nuevos mediante el método `.predict()`. Recibe datos (`X`) y devuelve una predicción (`y_pred`). **Importante**: para que un predictor funcione bien, los datos nuevos deben tener exactamente la misma forma y escala que los datos con los que el modelo "estudió".

## Filmina 32 — El Flujo de Trabajo Estándar

**Teoría completa (El Flujo de Trabajo Estándar, del docx)**:

1. **Instanciar**: se crea el objeto (ej. `scaler = StandardScaler()`).
2. **Ajustar (Fit)**: el objeto aprende de los datos de entrenamiento.
3. **Transformar o Predecir**: se aplica lo aprendido.

**Recordar**: `fit` es aprender la receta, `transform` es cocinar con ella. No hace falta volver a aprender la receta cada vez que se cocina un plato nuevo.

## Filmina 33 — Un Error Común: ¡Cuidado con el Fit!

**Teoría completa (Un error común: ¡Cuidado con el Fit!, del docx)**: un error muy frecuente de principiante es hacer `.fit()` sobre los datos de prueba o sobre datos nuevos. **¡No hay que hacerlo!** Solo se "estudia" (fit) con el conjunto de entrenamiento. Para los datos nuevos, solo se aplica lo aprendido con `.transform()` o `.predict()`.

**Por qué esto conecta directo con el Tema 05**: este mismo error es, en el fondo, una forma de Data Leakage — dejar que el conjunto de test "contamine" el proceso de aprendizaje. El Tema 05 lo retoma con nombre propio y en más profundidad.

**👉 En el Colab — Bloque 2 completo.** Acá es donde este tema deja de ser solo teoría: las 4 celdas de código de Estimators/Transformers/Predictors, sobre el dataset de propiedades.

**División train/test. Qué hace en general**: separa las 4 features y el target, y divide el dataset en 80% train / 20% test.
```python
features = ["superficie_m2", "ambientes", "antiguedad_anios", "score_amenities"]
X = df[features]
y = df["precio_eur"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Filas de entrenamiento:", X_train.shape[0])
print("Filas de prueba:", X_test.shape[0])
```
**Línea por línea:** `features = [...]` define en una lista los 4 nombres de columna que se van a usar como entrada del modelo. `X = df[features]` arma la tabla de features; `y = df["precio_eur"]` selecciona la columna target como una Serie. `train_test_split(X, y, test_size=0.2, random_state=42)` reparte ambas tablas en 4 piezas a la vez (`X_train`, `X_test`, `y_train`, `y_test`), reservando 20% para test y fijando la semilla para que el split sea siempre el mismo. Los dos `print` muestran cuántas filas quedaron de cada lado, usando `.shape[0]`.

**Transformer en acción: StandardScaler. Qué hace en general**: estandariza las 4 features, aprendiendo la media/desvío únicamente del train.
```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Media aprendida por columna:", scaler.mean_.round(1))
print("Forma de X_train_scaled:", X_train_scaled.shape)
```
**Línea por línea:** `StandardScaler()` instancia el transformador. `scaler.fit_transform(X_train)` aprende (fit) la media y el desvío de cada una de las 4 columnas mirando **solo** train, y en el mismo paso las transforma. `scaler.transform(X_test)` aplica esas mismas medias/desvíos ya aprendidos sobre test — nota que acá es solo `.transform()`, no `.fit_transform()`, porque el test nunca debe "enseñarle" nada al scaler. `scaler.mean_` es el atributo aprendido (la media de cada columna, calculada con train); `.shape` confirma que la forma de la matriz escalada no cambió, solo sus valores.

**Estimator + Predictor: LinearRegression. Qué hace en general**: entrena una Regresión Lineal (sin escalar, a propósito) y predice sobre test.
```python
modelo = LinearRegression()

modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test)

print("Primeras 5 predicciones:", predicciones[:5].round(0))
print("Primeros 5 precios reales:", y_test.values[:5])
```
**Línea por línea:** `LinearRegression()` instancia el modelo. `modelo.fit(X_train, y_train)` es el Estimator "estudiando" la relación entre las 4 features y el precio, usando los datos **sin escalar** (`X_train`, no `X_train_scaled`) — no hace falta escalar para una regresión lineal simple, y así los coeficientes quedan directamente interpretables en euros. `modelo.predict(X_test)` es el Predictor: usa lo aprendido para estimar el precio de las filas de test, que el modelo nunca vio entrenando. Los dos `print` comparan las primeras 5 predicciones contra los 5 precios reales correspondientes.

**Atributos aprendidos. Qué hace en general**: extrae e interpreta los coeficientes que el modelo aprendió.
```python
coeficientes = pd.Series(modelo.coef_, index=features).round(1)
print(coeficientes)

print("\nIntercepto (precio base):", round(modelo.intercept_))
```
**Línea por línea:** `modelo.coef_` es el atributo aprendido (termina en `_`) — un array con un coeficiente por feature, en el mismo orden que la lista `features`. `pd.Series(modelo.coef_, index=features)` le pone el nombre de cada feature a su coeficiente correspondiente. `modelo.intercept_` es el otro atributo aprendido: el valor base de la predicción cuando todas las features valen cero. La lectura de negocio: por cada m² adicional el precio sube esa cantidad de euros, manteniendo todo lo demás constante; la antigüedad debería restar valor (conviene revisar el signo en vivo); `score_amenities` suma directo.

---

# Tema 05 — Entrenar y Evaluar sin Trampas: Train/Test y Sobreajuste (Filminas 34-40)

## Filmina 34 — División de Tema

**Teoría completa de apertura (del docx)**: imaginá estar estudiando para un examen final de matemáticas. El profesor entrega una guía con 50 ejercicios resueltos para practicar. Se pasa toda la semana estudiando esos 50 ejercicios hasta saberlos de memoria. Si el profesor, el día del examen, pone exactamente los mismos 50 ejercicios, ¿realmente se aprendió matemática o simplemente se tiene una excelente memoria? Probablemente lo segundo. Ahora, si el profesor pone ejercicios diferentes pero basados en los mismos conceptos, y se logran resolver, entonces sí se aprendió. En Machine Learning sucede lo mismo: el objetivo no es que el modelo "memorice" los datos que se le dan, sino que "aprenda" los patrones generales para poder predecir resultados en situaciones que nunca ha visto.

## Filmina 35 — Train/Test Split: el "Examen" del Modelo

**Teoría completa (1. El Flujo de Entrenamiento y Evaluación, del docx)**: cuando se entrena un modelo, se quiere que sea capaz de **generalizar**: la capacidad de un modelo de ML de realizar predicciones precisas sobre datos nuevos, que no formaron parte del proceso de entrenamiento. Si se evalúa al modelo con los mismos datos que se usaron para enseñarle, se cae en una trampa: el modelo parecerá perfecto (porque ya conoce las respuestas), pero fallará estrepitosamente cuando se lo lleve al mundo real. **La solución: Train/Test Split** — dividir el dataset original en dos partes disjuntas:

- **Conjunto de Entrenamiento (Train Set)**: el material de estudio. El modelo utiliza estos datos para buscar patrones y ajustar sus parámetros internos. Normalmente representa entre el 70% y el 80% de los datos.
- **Conjunto de Prueba (Test Set)**: el examen final. Son datos que el modelo **nunca ve** durante el entrenamiento. Solo se usan al final para medir qué tan bien aprendió a generalizar.

**¿Por qué es esto fundamental?** Porque en la industria, un modelo que no generaliza no tiene valor. Por ejemplo, en un sistema de detección de fraudes bancarios, no sirve un modelo que reconozca los fraudes del año pasado si no es capaz de detectar un fraude nuevo con un patrón ligeramente distinto hoy.

## Filmina 36 — Los Tres Estados de un Modelo

**Teoría completa (2. Los tres estados de un modelo, del docx)**: al comparar el rendimiento del modelo en el conjunto de entrenamiento versus el de prueba, se puede diagnosticar qué tan bien está funcionando.

- **A. Underfitting (Subajuste): "El estudiante distraído"**. Ocurre cuando el modelo es demasiado simple para capturar la estructura subyacente de los datos. Señal: el error es alto tanto en entrenamiento como en prueba. Analogía: es como intentar explicar la economía global usando solo el precio del pan — faltan variables y complejidad.
- **B. Overfitting (Sobreajuste): "El estudiante que memoriza"**. Es el problema más común y peligroso. El modelo es tan complejo que empieza a aprender el **ruido** y los detalles aleatorios de los datos de entrenamiento, creyendo que son reglas generales. Señal: el error es casi cero en entrenamiento, pero muy alto en prueba. Analogía: es el estudiante que se memoriza que la respuesta a la pregunta 3 es "C", pero no sabe por qué — si en el examen la pregunta 3 cambia de orden, falla.
- **C. El "Sweet Spot" (Punto óptimo)**. Es el equilibrio perfecto: el modelo es lo suficientemente complejo para entender los patrones, pero lo suficientemente robusto para ignorar el ruido aleatorio. Acá, el error en entrenamiento y prueba es bajo y similar.

## Filmina 37 — Implementación con Scikit-Learn

**Teoría completa + código (3. Implementación con Scikit-Learn, del docx)**: en Python, la biblioteca Scikit-Learn facilita enormemente esta tarea con la función `train_test_split`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Parámetros clave**: `test_size=0.2` indica que se quiere reservar el 20% de los datos para la prueba y usar el 80% para entrenar. `random_state=42` es un punto vital para la **reproducibilidad**: al fijar una "semilla" aleatoria, se asegura que cada vez que se corra el código, la división de los datos sea la misma. Si no se hace, los resultados cambiarán cada vez que se ejecute el notebook, haciendo difícil comparar experimentos.

## Filmina 38 — Peligros: Fuga de Información y Series Temporales

**Teoría completa (4. Peligros y Buenas Prácticas, del docx)**:

- **Fuga de Información (Data Leakage)**: uno de los errores más graves para un principiante es permitir que información del conjunto de prueba "se filtre" al entrenamiento. Por ejemplo: si se calcula el promedio de una columna usando **todo** el dataset antes de dividirlo, el conjunto de entrenamiento ya conoce información (el promedio) que incluye datos del futuro (el test). **Regla de oro**: primero se divide en train/test, y cualquier cálculo estadístico (promedios, escalado, limpieza) se calcula **solo** sobre el conjunto de entrenamiento y luego se aplica al de prueba.
- **Series Temporales: la excepción a la regla aleatoria**: si se trabaja con datos que dependen del tiempo (precios de acciones, clima, ventas mensuales), **nunca** se debe usar una división aleatoria. Si se mezclan datos de 2023 en el entrenamiento y de 2021 en la prueba, se estaría usando el futuro para predecir el pasado. En estos casos, se usa una división cronológica: se entrena con los años 1 al 4 y se prueba con el año 5.

## Filmina 39 — Resumen de Conceptos Clave

**Teoría completa (5. Resumen de conceptos clave, del docx)**:

- **Entrenar no es memorizar**: se buscan patrones generales, no ruido.
- **Partición 80/20**: es el estándar inicial para tener suficientes datos para aprender y suficientes para evaluar con rigor.
- **Detectar Overfitting**: si un modelo tiene un "100% de precisión" en entrenamiento pero falla en prueba, hay que sospechar inmediatamente de sobreajuste.
- **Reproducibilidad**: usar siempre `random_state` para que los experimentos sean consistentes.
- **Honestidad intelectual**: el conjunto de prueba es sagrado. No hay que ajustarlo ni "espiar" sus resultados hasta que el modelo esté listo para la evaluación final.

Dominar la separación de datos es lo que separa a un programador que usa herramientas de un verdadero Científico de Datos. Sin una evaluación honesta, cualquier modelo es solo una ilusión tecnológica.

## Filmina 40 — Práctica (no entregable): Evaluación Honesta

**Instrucciones completas (del docx)**: utilizar el dataset con el que se trabajó en las unidades anteriores de Scikit-Learn. El objetivo es transformar el script anterior de entrenamiento simple en un flujo de evaluación profesional. **Preparación del Entorno**: cargar el dataset limpio con Pandas, definir `X` (características) e `y` (target). **La Gran División**: usar `train_test_split` con `test_size=0.2` y `random_state` fijo (ej. 42) para reproducibilidad. **Entrenamiento**: crear una instancia de un modelo (regresor o clasificador simple según el dataset) y entrenarlo **únicamente** con `X_train`/`y_train`. **Evaluación Dual**: calcular el `.score()` del modelo sobre train y sobre test. **Análisis Crítico**: comparar ambos resultados, describir si hay Overfitting (puntaje alto en train, bajo en test) o Underfitting (puntaje bajo en ambos), y proponer una acción correctiva (simplificar el modelo, conseguir más datos).

**Errores comunes a evitar (del docx)**: fuga de datos (no entrenar el modelo con el dataset completo antes de evaluarlo); olvidar el `random_state` (si no se fija, cada vez que se envía el trabajo las métricas podrían variar, dificultando la revisión técnica).

**👉 En el Colab — Bloque 3 completo.** Acá el notebook hace exactamente el ejercicio de la Filmina 40, con las 3 demostraciones centrales del tema.

**Baseline (Regresión Lineal). Qué hace en general**: predice sobre train y test por separado, y compara R²/MAE entre ambos.
```python
pred_train = modelo.predict(X_train)
pred_test = modelo.predict(X_test)

print(f"R² train: {r2_score(y_train, pred_train):.3f}   R² test: {r2_score(y_test, pred_test):.3f}")
print(f"MAE train: {mean_absolute_error(y_train, pred_train):,.0f} €   MAE test: {mean_absolute_error(y_test, pred_test):,.0f} €")
```
**Línea por línea:** `modelo.predict(X_train)` y `modelo.predict(X_test)` generan predicciones **por separado** para cada conjunto — es lo que permite comparar "qué tan bien le fue en lo que ya vio" contra "qué tan bien le fue en lo que nunca vio". `r2_score(y_real, y_predicho)` calcula el R² (cuánta variación explica el modelo, 0 a 1) para cada conjunto; `mean_absolute_error(...)` calcula el error promedio en euros. Train y test dan valores parecidos — señal de que el modelo generalizó bien, no memorizó. Sirve de punto de comparación ("baseline") para lo que sigue.

**Overfitting en acción: árbol sin restricciones. Qué hace en general**: entrena un árbol de decisión que puede crecer sin límite, y mide su R²/MAE en train vs. test.
```python
arbol_libre = DecisionTreeRegressor(random_state=42)
arbol_libre.fit(X_train, y_train)

print(f"R² train: {r2_score(y_train, arbol_libre.predict(X_train)):.3f}   R² test: {r2_score(y_test, arbol_libre.predict(X_test)):.3f}")
print(f"MAE train: {mean_absolute_error(y_train, arbol_libre.predict(X_train)):,.0f} €   MAE test: {mean_absolute_error(y_test, arbol_libre.predict(X_test)):,.0f} €")
```
**Línea por línea:** `DecisionTreeRegressor(random_state=42)` instancia un árbol de decisión **sin** el parámetro `max_depth` — sin ese límite, el árbol puede seguir dividiendo el espacio de datos hasta quedarse con una sola fila por hoja, memorizando el train literalmente. `.fit(X_train, y_train)` lo entrena. Los `print` recalculan R²/MAE igual que en el baseline, pero con `arbol_libre.predict(...)` en vez de `modelo.predict(...)`. Resultado: R² de train prácticamente perfecto (memorizó), pero el error en test mucho más alto que el de la Regresión Lineal — es el "estudiante que se memoriza las respuestas": perfecto en la guía de ejercicios, se derrumba en el examen real.

**La solución: limitar la complejidad. Qué hace en general**: repite el mismo árbol, ahora con profundidad máxima limitada.
```python
arbol_limitado = DecisionTreeRegressor(max_depth=4, random_state=42)
arbol_limitado.fit(X_train, y_train)

print(f"R² train: {r2_score(y_train, arbol_limitado.predict(X_train)):.3f}   R² test: {r2_score(y_test, arbol_limitado.predict(X_test)):.3f}")
print(f"MAE train: {mean_absolute_error(y_train, arbol_limitado.predict(X_train)):,.0f} €   MAE test: {mean_absolute_error(y_test, arbol_limitado.predict(X_test)):,.0f} €")
```
**Línea por línea:** `max_depth=4` es la única diferencia respecto a la celda anterior — limita a 4 la cantidad de veces que el árbol puede dividirse en profundidad, forzándolo a quedarse con los patrones generales en vez de los detalles particulares de cada fila. Resultado esperado: train y test quedan mucho más parecidos entre sí que con el árbol libre.

**Tabla comparativa. Qué hace en general**: arma una tabla con los 3 modelos entrenados hasta acá y calcula el "gap" de cada uno.
```python
comparacion = pd.DataFrame({
    "Modelo": ["Regresión lineal", "Árbol SIN restricción", "Árbol max_depth=4"],
    "R2_train": [
        r2_score(y_train, pred_train),
        r2_score(y_train, arbol_libre.predict(X_train)),
        r2_score(y_train, arbol_limitado.predict(X_train)),
    ],
    "R2_test": [
        r2_score(y_test, pred_test),
        r2_score(y_test, arbol_libre.predict(X_test)),
        r2_score(y_test, arbol_limitado.predict(X_test)),
    ],
})
comparacion["gap"] = (comparacion["R2_train"] - comparacion["R2_test"]).round(3)
comparacion.round(3)
```
**Línea por línea:** `pd.DataFrame({...})` arma una tabla de 3 filas (una por modelo) con el nombre del modelo, su R² de train y su R² de test — reutilizando las predicciones ya calculadas en las celdas anteriores, sin volver a entrenar nada. `comparacion["gap"] = (...).round(3)` agrega una cuarta columna con la diferencia `R2_train - R2_test`: cuanto más grande el gap, más sobreajuste.

**La trampa del Data Leakage. Qué hace en general**: agrega a propósito una feature que "filtra" el target, y muestra cómo el R² se dispara de forma sospechosa.
```python
features_con_leakage = features + ["precio_por_m2"]
X_leak = df[features_con_leakage]

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_leak, y, test_size=0.2, random_state=42)

modelo_leak = LinearRegression()
modelo_leak.fit(X_train_l, y_train_l)

print("R² test CON data leakage:", round(r2_score(y_test_l, modelo_leak.predict(X_test_l)), 4))
```
**Línea por línea:** `features_con_leakage = features + ["precio_por_m2"]` arma una nueva lista de features, agregando `precio_por_m2` a las 4 originales. `X_leak = df[features_con_leakage]` selecciona esa tabla ampliada. `train_test_split(X_leak, y, ...)` divide de nuevo en train/test. `LinearRegression().fit(...)` entrena un modelo nuevo (`modelo_leak`) con esa feature de más. El `print` final calcula el R² en test de este modelo "tramposo". **Por qué es una trampa y no un logro**: `precio_por_m2` se calculó dividiendo `precio_eur` por `superficie_m2` — o sea que contiene casi la respuesta escondida adentro. El R² se dispara de forma sospechosa, y esa sospecha es justamente la señal de alarma a entrenar: en la vida real, ese dato ni siquiera existiría todavía al momento de predecir el precio de una propiedad nueva.

---

# Tema 06 — Aplicaciones Prácticas de ML: del Modelo al Mundo Real (Filminas 41-49)

## Filmina 41 — División de Tema

**Teoría completa de apertura (del docx)**: llegados a la última unidad del módulo, ya se desarmó el motor del Machine Learning: la diferencia entre IA y Deep Learning, los tipos de aprendizaje, la arquitectura de Scikit-Learn, y la importancia crítica de evaluar los modelos sin hacer "trampa". Pero, ¿para qué sirve todo esto en la vida real? Un modelo de Machine Learning no es un fin en sí mismo; es una herramienta para resolver problemas que el software tradicional (basado en reglas fijas) simplemente no puede manejar.

## Filmina 42 — El Porqué del Machine Learning Aplicado

**Teoría completa (1. El Porqué del Machine Learning Aplicado, del docx)**: imaginá trabajar en el equipo de seguridad de un banco, con la tarea de escribir un programa para detectar correos que intentan robar contraseñas (phishing). Con programación tradicional, harían falta miles de reglas manuales: *"SI el correo contiene la palabra 'urgente' Y tiene un enlace sospechoso → MARCAR COMO PHISHING"*; *"SI el remitente es desconocido Y pide datos bancarios → MARCAR COMO PHISHING"*. **El problema**: los atacantes son creativos — mañana cambiarán "urgente" por "prioritario" o usarán una imagen en lugar de texto, y las reglas quedarían obsoletas en horas. Ahí es donde el Machine Learning aplicado brilla: en lugar de programar reglas, se alimenta al sistema con miles de ejemplos de correos reales (benignos y maliciosos), y el modelo aprende a identificar las señales sutiles —el "ruido" y los "patrones"— que un humano o una lista de reglas estáticas pasarían por alto. **ML como un "Filtro Inteligente"**: se pueden ver las aplicaciones de ML como filtros que, dada una entrada compleja (una imagen, un historial de compras, una señal de sensor), producen una salida útil (una categoría, un precio estimado, una alerta).

## Filmina 43 — Casos de Uso: ¿Quién Está Usando ML Hoy?

**Teoría completa (2. Casos de Uso, del docx)**: para entender el impacto del ML, ejemplos concretos de industrias transformadas por esta tecnología.

- **A. Sistemas de Recomendación — el "Efecto Netflix"**: Netflix, Spotify y Amazon son los reyes de este dominio, con un enfoque llamado **Filtro Colaborativo**. El problema: millones de productos y poco tiempo del usuario. La solución ML: el modelo analiza los patrones (qué se vio, qué se saltó, a qué hora se conecta) y los compara con millones de otros usuarios similares. Resultado: no solo se recomiendan "películas de acción", sino "películas de acción que le gustaron a personas que tienen gustos idénticos".
- **B. Detección de Fraude Bancario**: empresas como Mastercard o Visa procesan miles de transacciones por segundo. El problema: es imposible que un humano revise cada compra en tiempo real. La solución ML: modelos de **detección de anomalías**. El sistema conoce el "comportamiento normal" del usuario (dónde suele comprar, montos típicos); si aparece una compra de un reloj de lujo en otro continente, el modelo asigna una "puntuación de riesgo" alta y bloquea la transacción en milisegundos.
- **C. Logística y Movilidad — Uber y la Estimación de Tiempos**: cuando se pide un Uber y dice "llega en 4 minutos", hay un modelo de **Regresión** trabajando. El problema: el tráfico, el clima y los accidentes cambian constantemente. La solución ML: el modelo toma características (hora del día, datos históricos de la ruta, condiciones climáticas) para predecir un valor continuo: el tiempo de llegada (ETA).
- **D. Salud — Diagnóstico por Imagen**: en medicina, el ML ayuda a salvar vidas mediante visión por computadora. El problema: un radiólogo puede estar fatigado después de revisar 100 radiografías. La solución ML: modelos de **Clasificación de Imágenes** entrenados con millones de escaneos pueden resaltar áreas sospechosas de tumores con una precisión que iguala o supera a los expertos, funcionando como un "segundo par de ojos" incansable.

## Filmina 44 — Salud: Diagnóstico por Imagen

**Ampliación de la Filmina 43, punto D**: este caso merece su propio momento porque conecta con la advertencia de la Filmina 09 (pensar que la IA "entiende") — el modelo no "ve" un tumor como lo vería un médico. Detecta patrones estadísticos en píxeles que correlacionan con casos ya diagnosticados por expertos humanos. No reemplaza al radiólogo: funciona como apoyo para detecciones tempranas, en un contexto donde el costo de un error es altísimo.

## Filmina 45 — El Ciclo de Vida de un Proyecto de ML: el Pipeline

**Teoría completa (3. El Ciclo de Vida de un Proyecto de ML: El Pipeline, del docx)**: entrenar el modelo (lo que se hizo con Scikit-Learn en el Tema 04-05) es solo una pequeña parte del trabajo real. En el mundo profesional se sigue un flujo de trabajo o **pipeline**:

1. **Definición del Problema**: ¿qué se quiere predecir? ¿Es una clasificación (Sí/No) o una regresión (Número)?
2. **Recolección y Limpieza de Datos**: el paso más largo. Como se dice en la industria: *"Garbage in, garbage out"* (si entran datos basura, sale un modelo basura).
3. **Ingeniería de Características (Feature Engineering)**: elegir qué datos son relevantes. Para predecir el precio de una casa, el número de baños es clave; el color de la puerta, probablemente no.
4. **Entrenamiento y Evaluación**: acá se usa Scikit-Learn para ajustar el modelo y probarlo con datos que nunca ha visto (test set).
5. **Despliegue (Inferencia)**: se pone el modelo a trabajar en una aplicación real.
6. **Monitoreo**: los modelos pueden "deteriorarse" si el mundo cambia (por ejemplo, un modelo de predicción de ventas antes y después de una pandemia).

## Filmina 46 — Trampas y Errores Comunes: lo que Nadie te Dice

**Teoría completa (4. Trampas y Errores Comunes, del docx)**: incluso con los mejores datos, es fácil cometer errores conceptuales que arruinan una aplicación práctica.

- **Error 1: Confundir Correlación con Causalidad**. Un modelo de ML encuentra correlaciones. Si un modelo nota que "las personas que compran protector solar también compran helados", podría sugerir que el protector solar **causa** hambre de helado. Realidad: hay una variable oculta (el sol/verano). Lección: el modelo no entiende el "porqué", solo el "qué" — las decisiones de negocio deben ser validadas por humanos.
- **Error 2: El Sesgo en los Datos (Bias)**. Si se entrena un modelo de selección de personal usando solo currículums de personas contratadas en los últimos 20 años en una empresa que históricamente favoreció a hombres, el modelo aprenderá que "ser hombre" es una característica de éxito. Realidad: el modelo no es racista ni sexista por sí mismo; simplemente es un espejo de los datos que se le dieron.
- **Error 3: Sobreajuste (Overfitting)**. Un modelo que memoriza los datos de entrenamiento pero falla en la vida real es inútil — es como un estudiante que se memoriza las respuestas del examen pero no entiende la materia: si cambia un número en el examen, reprueba.

## Filmina 47 — Glosario para el Mundo Profesional

**Teoría completa (5. Glosario para el Mundo Profesional, del docx)**:

- **Modelo**: el "cerebro" que ya aprendió y está listo para decidir.
- **Inferencia**: el acto de usar el modelo para predecir algo nuevo (ej. cuando se sube una foto y Facebook sugiere etiquetas).
- **Features (Características)**: las columnas de entrada de los datos.
- **Labels (Etiquetas)**: la respuesta correcta que el modelo intenta aprender en aprendizaje supervisado.
- **Métricas**: los termómetros para saber si el modelo es bueno (Precisión, Recall, Error Cuadrático Medio).

## Filmina 48 — Síntesis y Cierre del Módulo

**Teoría completa (6. Síntesis y Cierre del Módulo, del docx)**: se completó un viaje desde la teoría de la IA hasta la mecánica del entrenamiento de modelos. El Machine Learning no es magia; es **estadística aplicada a gran escala**. La clave para ser un buen científico de datos no es conocer todos los algoritmos del mundo, sino saber: qué problema amerita usar ML (y cuál no); cómo preparar los datos para que el modelo aprenda patrones reales; cómo evaluar el éxito no solo con números, sino con impacto en el mundo real. El Machine Learning es una herramienta poderosa — hay que usarla para construir sistemas que no solo sean precisos, sino también éticos y transparentes.

## Filmina 49 — Práctica (no entregable): Diseño de una Solución de ML

**Instrucciones completas (del docx)**: **Identificar un problema** — pensar en el trabajo actual, un hobby o una empresa admirada, y qué proceso manual o repetitivo podría beneficiarse de una predicción o clasificación automática. **Definir la tarea**: ¿Clasificación (Categorías) o Regresión (Valores numéricos)? ¿Aprendizaje Supervisado o No Supervisado? **Proponer las Características (Features)**: enumerar al menos 5 datos (columnas) que el modelo necesitaría para aprender a tomar esa decisión. **Definir el éxito**: ¿cómo se sabría que el modelo funciona? Elegir una métrica (precisión, tiempo ahorrado, reducción de errores). **Considerar la ética**: ¿qué sesgos potenciales podrían existir en los datos que se recolectarían?

**Qué mirar al corregir (no está en el docx)**: esta práctica final retoma exactamente la misma estructura que la Pre-entrega del Tema 03 (Tarea, Features, Label, métrica, sesgo) — es una buena señal si el alumno ya la resuelve más rápido y con más soltura que la primera vez, porque significa que el vocabulario y el razonamiento quedaron incorporados.

## Filmina 50 (última) — ¿Dudas? ¿Consultas?

Cierre de la clase — espacio abierto antes de que el grupo se ponga a trabajar en la Pre-entrega.

**👉 En el Colab — Bloque 4, el cierre de la clase.** El ciclo de vida completo de un proyecto de ML (Definición → Datos → Entrenamiento → Evaluación → Despliegue/Inferencia → Monitoreo) es solo texto, sin código nuevo. Las Tareas 2 y 3 del plenario sí tienen celda propia:

**Tarea 2 — interpretar coeficientes. Qué hace en general**: reordena los coeficientes ya calculados en el Bloque 2 (Tema 04), de mayor a menor.
```python
coeficientes.sort_values(ascending=False)
```
**Línea por línea:** `coeficientes` es la misma `pd.Series` armada en el Tema 04; `.sort_values(ascending=False)` la reordena de mayor a menor sin modificar los valores — solo cambia el orden en que se muestran, para que sea más fácil leer cuál feature "pesa" más en la predicción.

**Tarea 3 — diagnóstico a partir de números dados. Qué hace en general**: arma una tabla con 3 modelos ficticios (no entrenados en esta celda) para practicar el diagnóstico de over/underfitting mirando solo números.
```python
casos = pd.DataFrame({
    "caso": ["Modelo A", "Modelo B", "Modelo C"],
    "R2_train": [0.95, 0.55, 0.99],
    "R2_test": [0.93, 0.52, 0.61],
})
casos["gap"] = (casos["R2_train"] - casos["R2_test"]).round(2)
casos
```
**Línea por línea:** a diferencia de la tabla `comparacion` del Tema 05 (que usaba resultados reales de modelos ya entrenados), acá los números de `R2_train`/`R2_test` están **escritos a mano** — es un ejercicio de diagnóstico puro, no el resultado de ningún `.fit()` en esta celda. `casos["gap"] = (...).round(2)` calcula la misma columna de diferencia que antes. La consigna para el grupo es decidir, mirando solo esos 3 números por fila, cuál caso es Overfitting, cuál Underfitting y cuál el Sweet Spot — sin correr ningún modelo.

Cierra con un **Solucionario** (uso docente) con las respuestas esperadas de las 4 tareas, para tener a mano mientras se conduce el plenario en vivo:

- **Tarea 1** (los 3 casos del mini-quiz al cierre del Tema 02): agrupar clientes sin categorías previas → **No Supervisado** (clustering); predecir spam con mails ya marcados → **Supervisado** (clasificación); termostato que prueba y aprende de la reacción → **Por Refuerzo**.
- **Tarea 2** (coeficientes): la lectura esperada es que `superficie_m2` y `score_amenities` suman valor, `antiguedad_anios` resta. `ambientes` puede dar un coeficiente chico o levemente negativo controlando por superficie — caso de **multicolinealidad** (una propiedad más grande ya "trae" más ambientes, así que la variable superficie ya captura buena parte de esa información).
- **Tarea 3** (diagnóstico): Modelo A (R² train 0.95, test 0.93, gap 0.02) → **Sweet Spot**, generaliza bien. Modelo B (R² train 0.55, test 0.52, gap 0.03) → **Underfitting** — el gap es chico, pero el error es alto en ambos, señal de que el modelo es demasiado simple. Modelo C (R² train 0.99, test 0.61, gap 0.38) → **Overfitting**, memorizó el train.
- **Tarea 4** (consigna abierta, próximo paso ante overfitting): no hay una única respuesta correcta — opciones válidas que puede proponer el grupo: reducir la complejidad del modelo (bajar `max_depth`, menos features), conseguir más datos de entrenamiento, aplicar regularización (Ridge/Lasso), usar validación cruzada para elegir mejor los hiperparámetros, o eliminar features irrelevantes o correlacionadas entre sí.

**Nota sobre la Pre-entrega y el Podcast**: el notebook cierra mencionando que en la Pre-entrega de esta semana no se programa un modelo, sino que se "piensa como Data Scientist" (exactamente la Filmina 27) — y recomienda escuchar el Podcast del módulo como repaso antes de encararla. Ese podcast es contenido de audio de `Clase 07.docx` que se decidió **no** convertir en filminas (a diferencia de los 6 Temas, que sí están 1 a 1 en `Clase07.html`) — por eso no tiene una sección propia en esta guía.

---

## Guía del Notebook — Referencia Rápida

**Estado del notebook**: [`Clase07.ipynb`](Clase07.ipynb) — Bloque 0 (repaso de Clase 06) + 4 Bloques prácticos, con horarios sugeridos de clase (0:00 a 1:55) y un solucionario para el docente al final. Dos datasets: `tasa-natalidad-deis-2000-2024.csv` para el repaso de estadística (Bloque 0), y `propiedades_sueca_ml.csv` como el dataset nuevo para entrenar el primer modelo real de la clase. El notebook viejo en `material/Viejo/Clase_7_Fundamentos_de_Ciencia_de_Datos_1_.ipynb` (Pipelines + K-Means) queda obsoleto.

**Todo el código ya está explicado, intercalado con la teoría de cada tema** (no repetido acá aparte). Para ubicarlo rápido:

| Bloque del Colab | Dónde está explicado en esta guía |
|---|---|
| Bloque 0 (Repaso 1-4) | Sección "Repaso de la Clase 06", al principio |
| Bloque 1 (Setup + Rompehielo) | Final del Tema 01 |
| Bloque 1 (Mini-quiz) | Final del Tema 02 |
| Bloque 2 (Split, Scaler, Regresión, Coeficientes) | Final del Tema 04 |
| Bloque 3 (Baseline, Árboles, Tabla, Data Leakage) | Final del Tema 05 |
| Bloque 4 (Tareas 2 y 3 + Solucionario) | Final del Tema 06, justo arriba |

---

## Pre-entrega: "Aplicaciones Prácticas de ML"

✅ **Entregable evaluado del Módulo**, anunciado en la Filmina 19 (división del Tema 03). A diferencia de otras clases del curso, `Clase 07.docx` no incluye una sección separada con "Qué tenés que presentar / Criterios de Aceptación / Formato de entrega" para esta Pre-entrega — el contenido más cercano a una consigna es la práctica del Tema 03 (Filmina 27, "Perfilado de una Solución de Machine Learning"), marcada en el propio texto como el ejercicio que **precede** al entregable evaluado.

**Lo que sí está definido explícitamente en el docx**: el módulo completo (Temas 01 a 03) cierra con la nota — *"Entregable de este módulo: Pre-entrega — Aplicaciones Prácticas de ML (Del Algoritmo al Impacto Real), evaluable, suma al proyecto final"* — pero sin una rúbrica propia dentro de este documento.

**Nota para quien dicte la clase**: si existe una consigna formal de esta Pre-entrega en otro documento (una rúbrica separada, un enunciado en el campus), conviene traerla a esta guía para completar la sección — tal como están las fuentes disponibles hoy (docx + html), esto es todo lo que se puede documentar sin inventar criterios que no están en el material original.

---

## Síntesis y Conexión Final

La clase entera se puede resumir en una progresión: primero entendemos el mapa completo de la Inteligencia Artificial y dónde vive el Machine Learning dentro de él (Tema 01); después aprendemos a diagnosticar qué tipo de aprendizaje aplica a un problema según si hay o no una etiqueta (Tema 02); conectamos esa teoría con aplicaciones reales de alto impacto de negocio, en la Pre-entrega del módulo (Tema 03); entramos por primera vez al código con la arquitectura interna de Scikit-Learn (Tema 04); aprendemos a evaluar sin hacer trampa, con train/test y el diagnóstico de sobreajuste (Tema 05); y cerramos viendo cómo todo esto se integra en el ciclo de vida completo de un proyecto de ML real, de punta a punta (Tema 06).

En la próxima unidad se retoman estos mismos conceptos aplicados a modelos concretos — Regresión Lineal, Árboles de Decisión y Regresión Logística — construyendo directamente sobre el flujo `train_test_split` + Estimators/Predictors que hoy se vio por primera vez.
