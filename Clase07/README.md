# Clase 07: Pipelines Reproducibles y Casos de Uso de ML en la Industria — Guía Completa para el Docente


> 📊 **Fuente del dataset**: todo el ejercicio práctico de hoy (repaso, pipeline, K-Means, validación y recomendaciones) se hace sobre un único dataset real, `tasa-natalidad-deis-2000-2024.csv`, publicado por el DEIS (Dirección de Estadísticas e Información de Salud) del Ministerio de Salud de la Nación:
> `https://datos.salud.gob.ar/dataset/tasa-de-natalidad/archivo/0f68d5c6-e667-40ca-90fd-4784336e092e`

---

## Bloque 0 — Un Ejemplito para Repasar 4 Conceptos de la Semana 6

Este documento arranca acá a propósito: es lo primero que corre el notebook, antes de tocar pipelines o Machine Learning, así que es lo primero que conviene tener repasado antes de dar la clase.

No es contenido nuevo de Clase 07: es un **ejemplito rápido** —aplicado sobre el mismo dataset de natalidad que se usa en el resto de la clase— para repasar de forma ágil 4 conceptos de estadística y preprocesamiento vistos la clase pasada. La idea no es volver a dar la teoría completa (eso ya se vio), sino refrescarla en 10 minutos con código concreto, porque el pipeline del Bloque 1 usa los cuatro conceptos sin excepción: se leen datos crudos (**limpieza**), se resumen con estadística (**tendencia y dispersión**), se entiende su forma y sus relaciones (**distribución y correlación**) y finalmente se transforman para que un algoritmo los pueda procesar (**transformación y reducción**).

**Por qué vale la pena este repaso, aunque parezca "perder tiempo" antes de llegar a Machine Learning**: en la práctica, ningún algoritmo de ML trabaja directamente sobre datos crudos — todo pipeline serio pasa primero por estas cuatro etapas, en este mismo orden lógico. Saltearse el repaso y arrancar directo con `KMeans()` es la forma más común de terminar con un modelo que "no anda" sin saber por qué: casi siempre el problema no está en el algoritmo, sino en algo de estadística básica que no se revisó antes (un outlier no detectado, una columna sin escalar, una correlación mal interpretada). Este ejemplito existe justamente para que esos cuatro chequeos queden frescos antes de construir el pipeline real del Bloque 1.

**El dataset que vamos a usar como hilo conductor**: `tasa-natalidad-deis-2000-2024.csv`, publicado por el DEIS (Dirección de Estadísticas e Información de Salud) del Ministerio de Salud de la Nación. Tiene 25 filas (una por año, de 2000 a 2024) y 26 columnas (`indice_tiempo` + la tasa de natalidad de cada una de las 25 provincias argentinas, incluyendo el total nacional). Es un dataset chico y prolijo a propósito — lo suficientemente simple como para repasar los cuatro conceptos sin distracciones, pero real, con una tendencia y unas correlaciones genuinas para analizar.

A continuación, los 4 puntos en el mismo orden en que aparecen en el notebook: primero el concepto en una línea, después el código que lo aplica, después el paso a paso de qué hace cada línea (y **por qué** esa línea está ahí), y por último la profundización teórica para quien quiera más contexto o le surja una pregunta en el momento.

### 1) Limpieza e Integración

**En una línea**: limpiar es decidir qué hacer con nulos y duplicados; integrar es derivar columnas nuevas para que el dato crudo se vuelva accionable.

```python
df_raw = pd.read_csv('tasa-natalidad-deis-2000-2024.csv')
print(f"Filas: {df_raw.shape[0]} ... | Columnas: {df_raw.shape[1]} ...")

nulos = df_raw.isnull().sum().sum()
duplicados = df_raw.duplicated().sum()
print(f"Valores nulos totales: {nulos}")
print(f"Filas duplicadas: {duplicados}")

natalidad_2024 = df_raw[df_raw['indice_tiempo'] == '01-01-2024'].drop(columns='indice_tiempo').T
natalidad_2024.columns = ['natalidad_2024']
natalidad_2024['categoria_natalidad'] = pd.cut(
    natalidad_2024['natalidad_2024'],
    bins=[0, 8.4, 9.7, np.inf],
    labels=['Baja', 'Media', 'Alta']
)
print(natalidad_2024['categoria_natalidad'].value_counts())
```

**Paso a paso**:

1. `pd.read_csv(...)` carga el archivo crudo en un DataFrame, `df_raw`. Es el punto de entrada de **cualquier** análisis: hasta acá el dato es "texto en un archivo", después de esta línea ya es una estructura que Python puede manipular.
2. `df_raw.shape` es un **atributo** (no un método, por eso no lleva paréntesis) que devuelve una tupla `(filas, columnas)` — se imprime para tener una primera dimensión del dataset (25 años de filas, 26 columnas: el índice de tiempo + 25 provincias). Es casi siempre el primer comando que se corre sobre un dataset nuevo, antes de mirar una sola fila.
3. `df_raw.isnull().sum()` primero convierte todo el DataFrame en una grilla de `True`/`False` (`True` donde hay un `NaN`), y el `.sum()` cuenta cuántos `True` hay **por columna** (porque `True` vale 1 y `False` vale 0 para efectos de suma); encadenar un segundo `.sum()` suma esos conteos por columna y da el **total** de nulos en todo el dataset. `df_raw.duplicated().sum()` hace algo parecido: marca con `True` cada fila que es una copia exacta de una fila anterior, y `.sum()` las cuenta.
4. Se imprimen ambos números — acá dan 0 y 0: el dataset llega limpio, así que no hace falta imputar ni eliminar nada (pero el patrón de código —diagnosticar antes de actuar— sería exactamente el mismo si hiciera falta limpiar algo). **¿Y si hubiera dado un número distinto de 0?** Como este dataset es una **serie de tiempo por provincia** (25 años, con una tendencia de caída sostenida — ver punto 2), no cualquier imputación serviría igual: rellenar con la media de la columna (`.fillna(df.mean())`) promediaría *ese año entre todas las provincias*, mezclando la trayectoria propia de cada una con la de las demás. Más apropiado acá sería interpolar **dentro de la misma fila** (`.interpolate(axis=1)`, una vez transpuesto), que estima el valor faltante a partir de los años vecinos de esa misma provincia — o `.ffill()`/`.bfill()` si el hueco cae al principio o al final de la serie, donde falta un vecino de un lado. Para evaluar si la imputación elegida tiene sentido, alcanza con recalcular la media/mediana (punto 2) y volver a mirar el gráfico (punto 3): si el valor imputado se nota como un salto brusco que no sigue la tendencia general, esa técnica no era la adecuada para este caso.
5. **La parte de integración se usa para esto**: convertir un número crudo (la tasa de natalidad, ej. 7.3) en algo directamente accionable para el negocio ("esta provincia está en la categoría Baja"), integrando ese dato con un criterio externo (los puntos de corte). Se hace de la siguiente manera: primero se filtra `df_raw` para quedarse solo con la fila del año 2024 (`df_raw['indice_tiempo'] == '01-01-2024'` genera una máscara booleana, y `df_raw[esa_máscara]` devuelve solo las filas donde la máscara es `True`).
6. Se descarta la columna `indice_tiempo` con `.drop(columns=...)`, porque ya cumplió su función de filtrar y no aporta nada como variable numérica en el paso siguiente.
7. `.T` **transpone** ( intercambia filas por columnas) el resultado: la única fila que quedaba (el año 2024, con una columna por provincia) pasa a ser una columna, y cada provincia pasa a ser una fila — así queda un DataFrame de "una fila por provincia", que es la forma que se necesita para trabajar provincia por provincia.En resumen: transponer no cambia los datos, solo cambia si las provincias son columnas o filas — y acá conviene que sean filas porque el análisis se hace "por provincia".
8. Se renombra esa única columna a `'natalidad_2024'` con `.columns = [...]` para que tenga un nombre claro y no quede con el nombre genérico que dejó la transposición.
9. `pd.cut()` toma esa columna numérica y la corta en 3 rangos (`bins=[0, 8.4, 9.7, np.inf]`) etiquetados como `'Baja'`, `'Media'`, `'Alta'` — cualquier valor entre 0 y 8.4 se convierte en `"Baja"`, entre 8.4 y 9.7 en `"Media"`, y por encima de 9.7 (hasta infinito, de ahí `np.inf`) en `"Alta"`. Los puntos de corte (8.4 y 9.7) no son arbitrarios: normalmente salen de los propios percentiles de los datos o de un criterio de negocio/salud pública ya definido.
10. `.value_counts()` cuenta cuántas provincias cayeron en cada categoría, y se imprime ese resumen — el "producto final" de la integración: pasamos de 25 números decimales a 3 grupos fáciles de comunicar.

**Para profundizar**: un nulo puede aparecer por errores de carga, por integraciones entre fuentes distintas, o porque el campo simplemente no aplica a ese registro (lo cual no es un error, es información). Las estrategias más comunes son **eliminar** la fila/columna (solo si son pocos casos y no introducen sesgo), **imputar** con la media o la mediana si la variable es numérica (la mediana es preferible si hay outliers, porque no se deja arrastrar por ellos), o con la **moda** o una **etiqueta de negocio explícita** si es categórica. Los duplicados exactos se eliminan con `.drop_duplicates()`. Acá el resultado del diagnóstico es "no hace falta hacer nada" — y eso también es una conclusión válida, no una excepción a la regla. Un matiz extra para quien pregunte "¿y si son pocos nulos, los borro nomás?": incluso eliminar unas pocas filas puede introducir sesgo si esos nulos no están distribuidos al azar (por ejemplo, si faltan justo los datos de las provincias más chicas, borrarlas sesga el análisis hacia las provincias grandes) — por eso "pocos" no es sinónimo automático de "seguro".

### 2) Medidas de Tendencia Central y Dispersión

**En una línea**: la tendencia central (media/mediana/moda) ubica el "centro" de los datos; la dispersión (desvío estándar/IQR) mide qué tan esparcidos están.

```python
serie_nacional = df_raw['natalidad_argentina']
media = serie_nacional.mean()
mediana = serie_nacional.median()
std = serie_nacional.std()
q1 = serie_nacional.quantile(0.25)
q3 = serie_nacional.quantile(0.75)
iqr = q3 - q1
```

**Paso a paso**:

1. `df_raw['natalidad_argentina']` extrae **una sola columna** como una `Series` de pandas — los 25 valores de natalidad nacional, uno por año. A partir de acá se trabaja sobre un vector de 25 números, no sobre la tabla completa.
2. `.mean()` calcula la media: suma los 25 valores y divide por 25. Es el resumen más usado, pero también el más fácil de distorsionar con un solo valor raro.
3. `.median()` calcula la mediana: internamente ordena los 25 valores de menor a mayor y toma el que queda en el medio (o el promedio de los dos del medio, si la cantidad es par). Por eso no le importa *cuánto* de extremo es un valor, solo su posición en el orden.
4. `.std()` calcula el desvío estándar: en el fondo, mide la distancia promedio de cada dato respecto a la media, pero elevada al cuadrado y después "des-elevada" con una raíz (para que quede en las mismas unidades que el dato original). No hace falta que la clase memorice la fórmula, alcanza con la idea: "en promedio, ¿cuánto se aleja cada año del promedio general?".
5. `.quantile(0.25)` devuelve el valor por debajo del cual cae el 25% de los datos (primer cuartil, Q1); `.quantile(0.75)` hace lo mismo para el 75% (tercer cuartil, Q3). La mediana, dicho sea de paso, es lo mismo que `.quantile(0.5)`.
6. `iqr = q3 - q1` resta ambos cuartiles para obtener el ancho del 50% central de los datos — cuanto más chico ese número, más apretados están los valores alrededor del centro.

**Para profundizar**: la **media** es sensible a valores extremos (un solo dato muy alto o muy bajo puede "arrastrarla"); la **mediana** no, porque solo le importa la posición central, no cuán extremos son los bordes — por eso es más **robusta**. La **moda** es el valor más frecuente, y la única de las tres que también sirve para variables categóricas (no tiene sentido "promediar" categorías de texto, pero sí contar cuál se repite más). Regla práctica: si media y mediana son parecidas, la distribución es razonablemente simétrica; si difieren mucho, hay dos explicaciones posibles: **outliers** o **asimetría/tendencia** en los datos (y hay que mirar el gráfico del punto 3 para saber cuál de las dos es). El **IQR** es el equivalente robusto del desvío estándar: ignora los valores extremos de los bordes, por eso se usa mucho junto con la mediana cuando se sospecha de outliers, de la misma forma que media y desvío estándar suelen ir juntos.

**El caso concreto para trabajar en el pizarrón**: acá la mediana (17.9) resulta *más alta* que la media (16.35). A primera vista uno esperaría lo contrario si hubiera outliers altos empujando la media hacia arriba, pero la causa real es otra: la natalidad viene en **caída sostenida** durante los 25 años de la serie, así que hay más años "altos" (al principio de la serie) que años "bajos" (al final), y eso corre el centro (mediana) por encima del promedio simple. Es el ejemplo perfecto para instalar la idea de que "media distinta de mediana" no siempre delata outliers: a veces delata una **tendencia** en el tiempo. El IQR de ~3.1 puntos confirma que, aun con esa caída, la dispersión año a año es moderada (no hay saltos bruscos).

La conclusión es: cuando la media y la mediana difieren, no siempre es por outliers — acá es porque los datos tienen una tendencia (la natalidad cae con el tiempo). Como hay más años "altos" al principio que años "bajos" al final, la mediana queda por encima de la media, aunque no haya ningún valor raro o extremo en la serie.



### 3) Distribuciones y Correlación

**En una línea**: la distribución muestra la forma que toman los datos al graficarlos; la correlación mide qué tan asociadas linealmente están dos variables — sin implicar causalidad.

```python
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.histplot(serie_nacional, kde=True, bins=10, color='teal')
plt.axvline(media, color='red', linestyle='--', label=f'Media: {media:.1f}')
plt.axvline(mediana, color='green', linestyle='--', label=f'Mediana: {mediana:.1f}')

plt.subplot(1, 2, 2)
provincias_comparar = df_raw[['natalidad_buenos_aires', 'natalidad_cordoba', 'natalidad_santa_fe']]
matriz_corr = provincias_comparar.corr()
sns.heatmap(matriz_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, center=0)

plt.tight_layout()
plt.show()

skew = serie_nacional.skew()
corr_ba_cba = matriz_corr.loc['natalidad_buenos_aires', 'natalidad_cordoba']
```

**¿Qué es Seaborn?** Es una librería de visualización de Python (se importa como `sns`, por *Samuel Norman Seaborn*, un personaje de la serie *The West Wing*) construida **encima** de Matplotlib (el `plt` que aparece en el resto del código). La diferencia práctica: con Matplotlib "puro" armar un histograma con curva de densidad o un mapa de calor de correlaciones lleva bastantes líneas de código; con Seaborn, `sns.histplot(...)` o `sns.heatmap(...)` lo hacen en una sola línea, con muy buena estética por defecto y pensado específicamente para trabajar directo con DataFrames de pandas. Por eso en este bloque se ven las dos librerías combinadas: Seaborn dibuja los gráficos estadísticos (`histplot`, `heatmap`) y Matplotlib (`plt`) se usa para lo genérico alrededor —el tamaño del lienzo, los títulos, las líneas verticales, mostrar la figura final—.

**Paso a paso**:

1. `plt.figure(figsize=(12, 4))` crea un lienzo ancho (12 pulgadas de ancho por 4 de alto) para poner dos gráficos lado a lado en vez de uno debajo del otro.
2. `plt.subplot(1, 2, 1)` divide ese lienzo en una grilla imaginaria de 1 fila x 2 columnas, y selecciona el primer casillero (el de la izquierda) como destino de lo próximo que se dibuje.
3. `sns.histplot(..., kde=True, bins=10)` dibuja el histograma de la serie nacional agrupando los 25 valores en 10 "cajones" (`bins`) y contando cuántos años caen en cada uno; `kde=True` además superpone una curva suavizada (Kernel Density Estimate) que estima la forma continua de la distribución por debajo de las barras, útil para ver la tendencia general sin el "escalonado" del histograma.
4. Las dos líneas `plt.axvline(...)` (*axis vertical line*) dibujan una línea vertical roja en la posición de la media y una verde en la de la mediana, encima del mismo histograma, para poder **comparar visualmente** dónde cae cada una — si están muy separadas, salta a la vista que hay asimetría o tendencia.
5. `plt.subplot(1, 2, 2)` pasa al segundo casillero de la misma grilla (derecha); todo lo que se dibuje de acá en adelante va ahí, no se pisa con el histograma de la izquierda.
6. Se seleccionan 3 columnas de provincias y `.corr()` calcula, **de a pares**, la correlación de Pearson entre las 3 — el resultado es una tabla cuadrada de 3x3 (siempre simétrica, y con 1.0 en la diagonal, porque toda variable se correlaciona perfectamente consigo misma).
7. `sns.heatmap(...)` dibuja esa matriz como un mapa de calor: cada celda se colorea según su valor (`cmap='coolwarm'`, con `center=0` para que el 0 quede en un color neutro y los extremos en rojo/azul) y además se le superpone el número exacto (`annot=True`, con `fmt='.2f'` para redondear a 2 decimales).
8. `plt.tight_layout()` ajusta automáticamente los márgenes y espacios para que los títulos y ejes de ambos gráficos no se superpongan; `plt.show()` renderiza todo en pantalla.
9. `.skew()` calcula la asimetría (skewness) de la serie nacional con una fórmula estadística estándar — no hace falta memorizarla, alcanza con saber leer el signo del resultado.
10. `matriz_corr.loc[...]` usa el nombre de fila y columna (no la posición numérica) para extraer puntualmente el valor de correlación entre Buenos Aires y Córdoba de la matriz ya calculada en el paso 6.

**Para profundizar**: una distribución puede ser **simétrica** (tipo campana/Normal, los valores se reparten parejo a ambos lados) o **sesgada** a la derecha (cola larga de valores altos poco frecuentes, típico en ingresos o precios) o a la izquierda. El **skew** cuantifica esto: cerca de 0 es simétrica, positivo es sesgo a la derecha, negativo a la izquierda. La **correlación de Pearson** va de -1 a 1: cerca de 1 es relación positiva fuerte (suben juntas), cerca de -1 es negativa fuerte (una sube, la otra baja), cerca de 0 no hay relación lineal. Un detalle importante para aclarar en clase: Pearson solo detecta relaciones **lineales** — dos variables pueden estar fuertemente relacionadas de una forma curva (por ejemplo, en forma de U) y aun así dar una correlación de Pearson cercana a 0, porque esa técnica no "ve" relaciones que no sean rectas.

**El caso concreto para trabajar en el pizarrón**: la correlación entre Buenos Aires y Córdoba da **por encima de 0.95** — altísima. Acá está la mejor oportunidad de la clase para instalar con fuerza el principio de que **correlación no implica causalidad**: no es que una provincia le "contagie" la baja natalidad a la otra. Lo que ocurre es que **ambas comparten la misma tendencia demográfica nacional** — hay una tercera variable de fondo (el fenómeno país, que afecta a todas las provincias por igual) explicando el movimiento conjunto de las dos. Es el mismo tipo de trampa que el clásico ejemplo de "las ventas de helado y los ahogamientos están correlacionados" (ambas suben en verano por el calor, no porque una cause la otra).

Conclusión: "Que dos provincias tengan correlación altísima no significa que se influyan entre sí — acá ambas simplemente están 'surfeando' la misma ola nacional de caída de natalidad."

Para qué sirve: para que no salgan del análisis pensando que correlación = causalidad, y para que aprendan a sospechar de una tendencia compartida (año, tiempo) como explicación alternativa antes de inventar una relación causal entre las variables.



### 4) Transformación y Reducción de Dimensionalidad

**En una línea**: hay que escalar las variables numéricas y, si son muchas, comprimirlas con PCA, porque los algoritmos basados en distancia (como K-Means) son sensibles a la magnitud de cada columna.

**¿Qué es K-Means?** Es un algoritmo de clustering (agrupamiento no supervisado): agarra un conjunto de puntos y los agrupa en *k* grupos, donde cada punto queda asignado al grupo cuyo "centro" (centroide) tiene más cerca. "Más cerca" se mide con distancia euclídea (la distancia geométrica normal, tipo teorema de Pitágoras) entre puntos.

**¿Qué es PCA?** Es una técnica de reducción de dimensionalidad: cuando tenés muchas columnas (variables), PCA las combina en un número menor de "componentes" nuevos que resumen la mayor parte de la variabilidad de los datos originales. Por ejemplo, si tenés 24 provincias como columnas, PCA te puede comprimir eso en 2 o 3 componentes que capturan "lo esencial" de esas 24, sin perder demasiada información.

**¿Por qué K-Means es sensible a la magnitud de cada columna?** Porque calcula distancias, y una columna con números grandes (ej: población en millones) domina esa distancia frente a una columna con números chicos (ej: tasa de natalidad entre 0 y 30), aunque esta última sea igual de importante conceptualmente. Sin escalar, el algoritmo terminaría agrupando casi exclusivamente en base a la variable de mayor magnitud, ignorando a las demás.

**¿Para qué escalamos entonces?** `StandardScaler` transforma cada columna para que tenga media 0 y desvío estándar 1 — así todas las variables "pesan" lo mismo en el cálculo de distancias, independientemente de su unidad original.

**¿Para qué reducimos dimensionalidad con PCA?** Dos motivos prácticos acá: (1) si tenés muchas provincias como columnas, es difícil visualizar o interpretar clusters en un espacio de tantas dimensiones — PCA lo lleva a 2D para poder graficarlo; y (2) reduce ruido y redundancia (columnas muy correlacionadas entre sí, como vimos con Buenos Aires y Córdoba, aportan información repetida).

**En resumen**: escalamos para que K-Means no se deje engañar por la magnitud de las columnas, y reducimos dimensionalidad con PCA para poder visualizar y simplificar los datos antes de agruparlos.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

columnas_ejemplo = df_raw[['natalidad_buenos_aires', 'natalidad_cordoba']]
scaler_demo = StandardScaler()
columnas_escaladas = scaler_demo.fit_transform(columnas_ejemplo)

provincias_T = df_raw.drop(columns='indice_tiempo').T
X_scaled_demo = StandardScaler().fit_transform(provincias_T)
pca_demo = PCA(n_components=2)
pca_demo.fit_transform(X_scaled_demo)
varianza_total = pca_demo.explained_variance_ratio_.sum() * 100
```

**¿Qué son esas dos líneas de `import`?** `scikit-learn` (el paquete que en código se importa como `sklearn`) es la librería estándar de Python para Machine Learning "clásico" — la que se va a usar en todo el resto de la clase (K-Means, métricas de validación, etc.). Está organizada en submódulos temáticos, y estas dos líneas traen herramientas de dos de ellos:

scikit-learn es la librería de referencia en Python para Machine Learning "clásico" (no deep learning). En general sirve para:

Preprocesamiento: escalar/normalizar datos, codificar variables categóricas, imputar valores faltantes (sklearn.preprocessing).
Modelos supervisados: regresión (lineal, logística), árboles de decisión, random forests, SVM, etc. — para predecir una variable objetivo (sklearn.linear_model, sklearn.tree, sklearn.ensemble...).
Modelos no supervisados: clustering (K-Means, jerárquico), reducción de dimensionalidad (PCA) — para encontrar estructura en datos sin etiquetar (sklearn.cluster, sklearn.decomposition).
Validación y métricas: separar train/test, cross-validation, medir qué tan bueno es un modelo (accuracy, silhouette score, etc.) (sklearn.model_selection, sklearn.metrics).
Pipelines: encadenar pasos (preprocesar → entrenar → evaluar) de forma prolija y reproducible.
Es la caja de herramientas estándar para "aprender de los datos" (agrupar, predecir, clasificar) fuera del mundo de redes neuronales — que para eso ya se usan otras librerías como TensorFlow o PyTorch. En esta clase puntualmente la van a usar para K-Means (clustering) y sus métricas de validación.



- `from sklearn.preprocessing import StandardScaler`: el submódulo `preprocessing` agrupa herramientas para **preparar** los datos antes de modelarlos (escalado, codificación de categorías, etc.). `StandardScaler` es específicamente la clase que aplica el z-score que se explica más abajo.
- `from sklearn.decomposition import PCA`: el submódulo `decomposition` agrupa técnicas que **descomponen** una matriz de datos en partes más simples. `PCA` es la clase que implementa el Análisis de Componentes Principales.

El patrón `from paquete.submódulo import Clase` es el mismo que se va a repetir todo el día con scikit-learn (por ejemplo, más adelante aparece `from sklearn.cluster import KMeans`) — conviene que quede instalada esa lectura desde acá: primero el paquete grande (`sklearn`), después el área específica dentro de él (`preprocessing`, `decomposition`, `cluster`...), y por último la herramienta puntual que se necesita.

**Paso a paso**:

1. Se seleccionan 2 columnas de ejemplo (natalidad de Buenos Aires y de Córdoba) para demostrar el escalado en algo simple y visualmente chequeable, antes de aplicarlo a las 25 columnas reales.
2. `StandardScaler()` crea el objeto escalador (todavía no hizo nada, solo está configurado); `.fit_transform(...)` hace dos cosas en un solo paso: **calcula** la media y el desvío de cada columna (`fit`) y **aplica** la fórmula de z-score `(x - media) / desvío` a cada valor (`transform`). El resultado es un array de NumPy con ambas columnas ya en la misma escala (media ≈0, desvío ≈1) — por eso las líneas siguientes del notebook imprimen la media "antes" y "después", para que se vea el cambio con números concretos.
3. Para la reducción de dimensionalidad, ahora se trabaja con **todo** el dataset: `df_raw.drop(columns='indice_tiempo')` descarta la columna de fecha (no es una variable numérica útil acá, y si se dejara, PCA la trataría como una feature más sin sentido), y `.T` transpone para que cada **provincia** sea una fila y cada **año** sea una columna (25 columnas = 25 features) — el mismo truco de transposición que se va a usar en serio en el pipeline del Bloque 1.
4. Se escala esa matriz completa con un nuevo `StandardScaler().fit_transform(...)` — es un escalador distinto al del paso 2 (nuevo objeto), porque ahora son 25 columnas en vez de 2.
5. `PCA(n_components=2)` crea un objeto PCA configurado para comprimir todo en **2 componentes principales** (se elige 2 acá específicamente porque después se puede graficar fácil en un plano; en un caso real, ese número se decide según cuánta varianza se quiere conservar).
6. `.fit_transform(X_scaled_demo)` hace el trabajo matemático pesado: encuentra las 2 combinaciones lineales de las 25 columnas originales que capturan la mayor cantidad de variabilidad posible (los "componentes principales"), y transforma cada provincia a ese nuevo sistema de 2 coordenadas.
7. `pca_demo.explained_variance_ratio_` es un array con la proporción de varianza (información) que capta cada uno de los 2 componentes por separado; sumarlos y multiplicar por 100 da el **porcentaje total** de la variabilidad original que se conserva al comprimir de 25 dimensiones a solo 2 (suele rondar el 90% o más en este dataset, porque los 25 años están muy correlacionados entre sí — la tendencia nacional los mueve a todos juntos).

**Para profundizar**: cuando hay que convertir texto a número existen dos caminos: `LabelEncoder` (asigna un número entero a cada categoría, útil cuando hay un orden implícito, como "Bajo/Medio/Alto") o **one-hot encoding** (crea una columna binaria por categoría, preferible cuando no hay orden, para no inventarle una jerarquía artificial a los datos — por ejemplo, entre provincias no tendría sentido que "Chaco" valga menos que "Córdoba" solo por el orden alfabético). El **escalado** es indispensable para algoritmos que miden distancias (K-Means, PCA, KNN): si una columna tiene valores entre 0 y 100.000 y otra entre 0 y 1, la primera va a dominar completamente el cálculo de distancia solo por su magnitud numérica, sin que eso refleje ninguna importancia real de esa variable. Vale aclarar, para no generalizar de más: no **todos** los algoritmos necesitan escalado — los basados en árboles (como Random Forest) son indiferentes a la escala, porque solo comparan si un valor es mayor o menor que un umbral, no la distancia entre puntos. **PCA**, conceptualmente, combina matemáticamente muchas columnas correlacionadas en un número mucho menor de "componentes principales" (nuevas variables, sin significado directo pero que resumen a las originales) que conservan la mayor parte posible de la variabilidad original.

**La conclusión que cierra el punto 4**: **escalar es obligatorio antes de PCA o K-Means**, porque ambos algoritmos miden distancias, y sin escalar, una columna con números más grandes "pesaría" más en el resultado solo por su magnitud, no porque sea más relevante para el problema.

### Conclusión del repaso: los 4 puntos como un solo checklist

Antes de pasar a pipelines, vale la pena bajar los 4 conceptos a una sola idea, porque en la práctica no se usan por separado: son las **4 preguntas que cualquier dataset real necesita responder antes de tocar un algoritmo**.

| # | Concepto | La pregunta que responde | Qué pasa si te lo saltás |
|---|---|---|---|
| 1 | Limpieza e Integración | ¿Los datos están completos y son confiables? | El modelo aprende de basura sin que nadie se dé cuenta ("garbage in, garbage out"). |
| 2 | Tendencia Central y Dispersión | ¿Dónde está el centro y qué tan esparcidos están los datos? | Se pasan por alto outliers o tendencias que después explican (o rompen) el resultado del modelo. |
| 3 | Distribuciones y Correlación | ¿Cómo se relacionan las variables entre sí? | Se arrastran variables redundantes, o se confunde correlación con causalidad al interpretar resultados. |
| 4 | Transformación y Reducción | ¿Los datos están en un formato que el algoritmo pueda procesar bien? | Algoritmos basados en distancia (K-Means, PCA) dan resultados sesgados por la escala, no por el contenido real de los datos. |

Estos cuatro pasos son, ni más ni menos, las etapas de **ingesta, EDA y feature engineering** de cualquier pipeline — la diferencia es que hoy se hicieron "a mano", en celdas sueltas, sobre un ejemplo chico y ya limpio. Lo que viene ahora en el **Bloque 1** es exactamente ese mismo trabajo, pero empaquetado en una función reutilizable (`pipeline_preprocesamiento()`) para que se pueda correr una y otra vez, con cualquier archivo nuevo, sin repetir el proceso a mano cada vez. Con el repaso fresco, ahora sí: manos a la obra con pipelines.

---

## Índice

00. [Bloque 0 — Repaso de la Semana 6 (con paso a paso del código)](#bloque-0--un-ejemplito-para-repasar-4-conceptos-de-la-semana-6)
0. [Mapa rápido de la clase](#0-mapa-rápido-de-la-clase)
0.1. [Introducción General de la Clase](#introducción-general-de-la-clase)
1. [Módulo 1 — Principios de Diseño de Pipelines Reproducibles](#módulo-1--principios-de-diseño-de-pipelines-reproducibles)
2. [Módulo 2 — Casos de Estudio: Segmentación y Recomendaciones](#módulo-2--casos-de-estudio-segmentación-y-recomendaciones)
3. [Módulo 3 — Casos de Éxito: ML en Acción](#módulo-3--casos-de-éxito-ml-en-acción)
4. [Módulo 4 — Supervisado vs. No Supervisado](#módulo-4--supervisado-vs-no-supervisado)
5. [Break del Coder](#break-del-coder)
6. [Módulo 5 — Métricas y Estrategias de Validación](#módulo-5--métricas-y-estrategias-de-validación)
7. [El ejercicio práctico del notebook, explicado en profundidad](#7-el-ejercicio-práctico-del-notebook-explicado-en-profundidad)
8. [Preguntas frecuentes y errores típicos a anticipar](#preguntas-frecuentes-y-errores-típicos-a-anticipar)
9. [Material de la clase](#material-de-la-clase)

---

## 0. Mapa rápido de la clase

| # | Módulo | Slides | Idea central |
|---|--------|--------|---------------|
| 0 | **Repaso Clase 6** (Bloque 0 del notebook) | — (no está en el PDF/HTML) | Los 4 pilares de estadística/preprocesamiento, repasados sobre el dataset de hoy antes de empezar |
| 1 | Principios de Diseño de Pipelines Reproducibles | 03–08 | Un pipeline que cualquiera pueda replicar y auditar |
| 2 | Casos de Estudio: Segmentación y Recomendaciones | 09–13 | Dos aplicaciones clave de ML en la industria (teoría general) |
| 3 | Casos de Éxito: ML en Acción | 14–29 | 4 casos reales con nombre propio: San Cristóbal, Medplaya, Amazon, Mazda |
| 4 | Supervisado vs. No Supervisado | 30–34 | Cuándo usar cada enfoque |
| — | **Break del Coder** | 35 | Corte de ~10 minutos |
| 5 | Métricas y Estrategias de Validación | 36–42 | Cómo saber si un modelo es realmente bueno |
| — | **Ejercicio práctico (notebook)** | — | Bloque 0 + Pipeline + K-Means sobre natalidad real — aplica los módulos 1, 4 y 5 |

> **Nota sobre el notebook**: arranca con un **Bloque 0 (Repaso de Clase 6)** que no pertenece a esta unidad — es intencional y está bien dejarlo, sirve de entrada en calor. El resto (Bloques 1 a 4) corre en paralelo a los Módulos 1, 4 y 5 de esta guía, pero **no cubre el Módulo 2 ni el Módulo 3** (segmentación/recomendación teórica ni los 4 casos de éxito con nombre propio) — el notebook ya tiene, en su Bloque 1 y Bloque 2, celdas de teoría agregadas que tienden puentes explícitos hacia esos casos para que la clase quede conectada aunque no se repliquen en código.

---

## Introducción General de la Clase

### El disparador teórico del PDF

El material oficial abre la unidad con esta pregunta: *¿Qué hace que un pipeline de ciencia de datos sea realmente reproducible?* Y la plantea con un escenario muy concreto: desarrollaste un modelo predictivo que mejora significativamente la toma de decisiones en tu empresa, pero cuando un colega intenta replicar tu trabajo, los resultados no coinciden. Ese desajuste es, en la práctica, el problema número uno que esta clase busca resolver: **la reproducibilidad es lo que determina si un proyecto de ciencia de datos es confiable y escalable en la industria, o si se queda como un experimento aislado que nadie más puede usar.**

Conviene abrir la clase con esa pregunta tal cual, en voz alta, antes de mostrar ninguna diapositiva — es la misma pregunta que dispara el Módulo 1, y funciona bien como gancho porque casi todos los alumnos ya vivieron una versión de ese problema ("en mi máquina andaba").

### El diagrama que abre el PDF

La primera página del material resume el recorrido completo de la clase con un diagrama de 7 etapas encadenadas — vale la pena dibujarlo en el pizarrón o proyectarlo al empezar:

```
Ingesta de datos → Procesamiento / EDA → Feature Engineering → Modelado → Evaluación → Artefactos Reproducibles → Entrega Mínima
```

Este diagrama es el "mapa madre" de toda la unidad: el **Módulo 1** lo explica en detalle (qué hace reproducible a cada etapa), los **Módulos 2 y 3** muestran ese mismo pipeline aplicado en 4 empresas reales, el **Módulo 4** profundiza en la etapa de "Modelado" (qué tipo de algoritmo elegir según el problema) y el **Módulo 5** profundiza en la etapa de "Evaluación" (cómo saber si el modelo realmente funciona). Todo lo que viene en la clase es, en el fondo, un zoom progresivo sobre distintas partes de este mismo diagrama — es útil volver a él verbalmente entre módulo y módulo ("ahora estamos en la etapa de Evaluación de este mismo pipeline").

### Qué se lleva el alumno al final de la clase

Según el propio material, al cerrar esta unidad el alumno debería poder:

1. Diseñar y explicar la estructura de un pipeline end-to-end reproducible.
2. Comparar segmentación de clientes vs. sistemas de recomendación y elegir cuál aplica a un problema de negocio dado.
3. Identificar, frente a un caso real, si conviene un enfoque supervisado o no supervisado.
4. Elegir la métrica y la estrategia de validación correctas según el tipo de problema (clasificación, regresión o clustering) y las restricciones del negocio.

### La conexión con el repaso del notebook

El notebook no abre directamente con este diagrama: primero hace un ejemplito de 10 minutos que repasa 4 conceptos de la clase pasada (detallado en la [sección 7, Bloque 0](#bloque-0--un-ejemplito-para-repasar-4-conceptos-de-la-semana-6)), porque esos cuatro pilares (limpieza, estadística descriptiva, distribuciones/correlación, transformación/reducción) son insumo directo de las etapas 2 y 3 del pipeline de hoy (Procesamiento/EDA y Feature Engineering). Es una forma de que el repaso no quede "suelto": se siente como el cimiento sobre el que se construye el pipeline reproducible del Bloque 1.

---

## Módulo 1 — Principios de Diseño de Pipelines Reproducibles

### 🗺️ Mapa de filminas de este módulo

Cada fila es una diapositiva (el número es el que ves en el pie de página, ej. `03 / 43`), en el mismo orden en que van pasando. Hacé clic para saltar directo a su explicación extendida.

| Filmina | Título en la diapositiva | Explicación extendida acá |
|---|---|---|
| Slide 02/43 | *(Divisor)* Principios de Diseño de Pipelines Reproducibles | [Intro del módulo](#filmina-02) |
| Slide 03/43 | ¿Qué Hace Reproducible un Pipeline? | [→ Ir a la explicación](#filmina-03) |
| Slide 04/43 | Pipeline End-to-End: Las Etapas | [→ Ir a la explicación](#filmina-04) |
| Slide 05/43 | Componentes Clave para la Reproducibilidad | [→ Ir a la explicación](#filmina-05) |
| Slide 06/43 | Prácticas para Despliegue Mínimo y Compartición | [→ Ir a la explicación](#filmina-06) |
| Slide 07/43 | Reproducibilidad en la Industria | [→ Ir a la explicación](#filmina-07) |

> Una sección de acá abajo (**1.0**) es **contenido extra que no está en ninguna filmina** — profundiza qué significa la palabra "pipeline" antes de entrar en la teoría técnica. Queda marcada como tal para que no la confundas con una diapositiva que no encontrás. El código real del pipeline (Bloque 1 del notebook) se explica al final de este mismo módulo, en [Bloque 1](#bloque-1-pipeline), justo antes de pasar al Módulo 2 — primero las filminas de este módulo, después el ejemplo.

<a id="filmina-02"></a>

#### Pregunta disparadora para abrir la clase (Módulo 1)

Imaginá que desarrollaste un modelo predictivo que mejora significativamente la toma de decisiones en tu empresa. Un colega intenta replicar tu trabajo... y los resultados no coinciden. ¿Qué salió mal? La reproducibilidad es lo que separa un experimento de laboratorio de un proyecto de ciencia de datos confiable y escalable en la industria.

<a id="filmina-03"></a>

### 0.1 Por qué "reproducibilidad" merece un módulo entero (Slide 03/43 — "¿Qué Hace Reproducible un Pipeline?")

La diapositiva lo resume en dos líneas —el problema real y la meta— pero vale la pena desarrollar cada una antes de proyectarla, porque es la idea que sostiene el resto de la clase.

**El problema real, en detalle**: en ciencia de datos, "funcionar una vez, en tu máquina" es la parte fácil. Lo difícil —y lo que realmente se paga en la industria— es que ese mismo resultado se pueda **repetir**: por otra persona, en otra computadora, con el dataset actualizado, seis meses después. Cuando eso no pasa, aparecen síntomas muy concretos y muy comunes:

- *"En mi notebook daba un resultado distinto"* — porque se corrieron las celdas en un orden distinto al que el autor tenía en la cabeza (una celda modificó una variable global que otra celda de más abajo necesitaba con su valor original).
- *"No sé qué versión de la librería usé"* — el mismo código con una versión distinta de `scikit-learn` o `pandas` puede dar un resultado ligeramente distinto, y sin registro de versiones es imposible saber cuál causó el cambio.
- *"Perdimos el dataset que usamos para entrenar ese modelo"* — sin versionado de datos, el archivo original se sobreescribió o se perdió, y el modelo en producción quedó sin forma de auditarse ni re-entrenarse desde cero.
- *"Nadie se anima a tocar ese notebook"* — sin documentación, el código se vuelve una caja negra: nadie sabe qué decisiones de negocio están hardcodeadas ahí adentro (¿por qué el umbral es 8.4 y no 8.5? ¿por qué se excluyó tal provincia?).

**La meta, en detalle**: diseñar el pipeline pensando desde el principio en que **alguien más (o vos mismo, en el futuro) lo va a tener que correr de nuevo**. Eso cambia decisiones concretas de diseño: en vez de escribir código que solo corre "de arriba hacia abajo en este notebook puntual", se escribe como funciones con parámetros explícitos (como `pipeline_preprocesamiento(path_archivo)`, que se ve más abajo); en vez de fijar un dataset "a mano", se versiona; en vez de dejar los comentarios para después, se documenta a medida que se escribe.

**Por qué a la industria le importa tanto**: no es una cuestión de prolijidad — tiene consecuencias de negocio directas. En los 4 casos de éxito del Módulo 3 esto se ve todo el tiempo: un banco necesita poder **auditar** por qué un modelo de fraude (San Cristóbal) marcó una transacción como sospechosa, meses después del hecho; un hotel necesita poder **re-entrenar** su modelo de cancelaciones (Medplaya) cuando cambian las temporadas sin reconstruir todo desde cero; un equipo de científicos de datos necesita poder **traspasar** un proyecto a un colega nuevo sin que el proyecto se vuelva inservible. La reproducibilidad no es un "nice to have" académico, es lo que permite que un modelo sobreviva más tiempo que la persona que lo construyó.

### 1.0 ¿Qué es un "pipeline"? El concepto, antes de la definición técnica (🎁 contenido extra — no está en ninguna filmina)

Antes de entrar en las etapas y las buenas prácticas, vale la pena pararse un minuto en la palabra misma — porque entender de dónde viene ayuda a que el concepto quede grabado mucho mejor que memorizar una lista de pasos.

**El origen de la palabra**: *pipeline* es una palabra compuesta del inglés: *pipe* (tubo, caño, tubería) + *line* (línea, trazado). Literalmente significa **"tubería" o "línea de tuberías"** — el término nació en la industria del petróleo y el gas para nombrar el sistema de caños interconectados que transportan crudo o gas natural a lo largo de cientos o miles de kilómetros, desde el pozo de extracción hasta la refinería o el punto de distribución (en español se lo suele traducir como *oleoducto* o *gasoducto* en ese contexto específico).

**Por qué esa imagen se trasladó a la tecnología**: en un oleoducto real, el petróleo crudo entra por un extremo y va **pasando por una secuencia de estaciones fijas** (bombeo, filtrado, medición, etc.) hasta salir del otro lado convertido en algo listo para usar. Esa misma idea —*algo entra crudo, atraviesa una secuencia fija de etapas conectadas, y sale del otro lado transformado y listo para usar*— es exactamente lo que pasa con los datos en un **pipeline de ciencia de datos**: el dataset crudo "entra" por la ingesta, atraviesa limpieza, feature engineering y modelado, y "sale" del otro lado convertido en un resultado evaluado y listo para compartir. Por eso la industria del software adoptó la misma palabra para nombrar cualquier secuencia automatizada de pasos conectados — hoy se habla de "pipelines" en manufactura (líneas de ensamblaje), en desarrollo de software (pipelines de CI/CD que compilan, testean y despliegan código automáticamente) y, la que nos importa hoy, en ciencia de datos.

**Para qué se usa, en concreto**: un pipeline se usa para **encadenar automáticamente** los pasos que, de otra forma, alguien tendría que ejecutar a mano y en el orden correcto cada vez (cargar datos → limpiarlos → transformarlos → entrenar un modelo → evaluarlo → entregar un resultado). En vez de repetir ese proceso manualmente cada vez que llegan datos nuevos, se escribe una sola vez como una secuencia de funciones conectadas, y después se ejecuta con un solo llamado — como el `pipeline_preprocesamiento()` que se arma más adelante en el notebook de hoy.

**Por qué es tan importante (la razón de fondo de todo este módulo)**:

1. **Consistencia**: cada corrida sigue exactamente los mismos pasos, en el mismo orden — se elimina el riesgo de que alguien se salte un paso o lo haga distinto "a mano".
2. **Reproducibilidad**: es la base técnica de todo el Módulo 1 — si el proceso está encadenado y versionado, cualquier persona (o la misma persona meses después) puede correrlo de nuevo y obtener el mismo resultado.
3. **Ahorro de tiempo y menos errores humanos**: lo que antes tomaba horas de trabajo manual propenso a errores, se ejecuta en segundos y siempre de la misma forma.
4. **Escalabilidad**: un pipeline bien armado no le importa si hoy procesa 100 filas o si mañana el dataset crece a 10 millones — la lógica es la misma, solo cambia el volumen que atraviesa el "caño".
5. **Mantenibilidad**: si hay que corregir un paso (por ejemplo, cambiar cómo se imputan los nulos), se corrige en un solo lugar del pipeline, y el arreglo se aplica automáticamente la próxima vez que se ejecute todo el flujo — no hay que ir a buscar y corregir ese paso en 20 notebooks distintos.

**Para remarcar en clase**: cuando en el Bloque 1 del notebook aparece la función `pipeline_preprocesamiento()`, no es una casualidad de nombre — es literalmente ese "caño" de datos: entra la ruta de un archivo crudo (`path_archivo`) por un extremo, y sale del otro un array ya limpio, transpuesto y escalado (`X_scaled`), listo para entrar directo al modelo de K-Means del Bloque 2.

**Analogía útil para el pizarrón**: un pipeline reproducible es también como una **receta de cocina bien escrita**, no como "cocinar de memoria". Si la receta especifica ingredientes exactos (los datos versionados), pasos numerados (el código modular) y el punto de cocción (las métricas de evaluación), cualquier persona puede reproducir el mismo plato. Si en cambio el chef improvisa "un poco de esto, un poco de aquello" (celdas de Jupyter sueltas, ejecutadas en cualquier orden, con decisiones que solo están en la cabeza de quien las tomó), el resultado depende de quién cocine — y eso es exactamente lo que un pipeline reproducible busca eliminar.

<a id="filmina-04"></a>

### 1.1 ¿Qué es un pipeline end-to-end en ciencia de datos? (Slide 04/43 — "Pipeline End-to-End: Las Etapas")

Un **pipeline end-to-end** es un flujo completo que transforma datos crudos en un modelo funcional y evaluado, listo para su uso o despliegue. La filmina lista seis etapas en una línea cada una; acá va cada una desarrollada, con qué implica en la práctica y dónde se retoma más adelante en la clase.

1. **Ingestión de datos**: recolección y carga desde diversas fuentes — un CSV local, una API, una base de datos, un archivo publicado por un organismo público (como el DEIS en el ejercicio de hoy). En esta etapa la pregunta clave es *"¿de dónde viene el dato y puedo volver a traerlo de la misma fuente más adelante?"* — si la fuente cambia o desaparece, el pipeline entero deja de ser reproducible por más prolijo que sea el código.
2. **Análisis exploratorio de datos (EDA)**: comprensión inicial, detección de patrones y limpieza — es exactamente lo que se practicó en el Bloque 0 del repaso (nulos, duplicados, tendencia central, distribución, correlación). Acá se decide qué está "sano" en el dataset y qué necesita corrección antes de seguir.
3. **Feature engineering**: creación y selección de variables relevantes — transformar lo que ya se tiene en algo que el algoritmo pueda aprovechar mejor (categorizar un número continuo, transponer una matriz, derivar una variable nueva). Los 4 casos de éxito del Módulo 3 dedican una sección entera a esta etapa cada uno, porque suele ser la que más impacto tiene en la calidad final del modelo.
4. **Modelado**: entrenamiento de algoritmos para aprender patrones — acá es donde se elige entre un enfoque supervisado o no supervisado (el Módulo 4 está dedicado enteramente a esa decisión) y se entrena el algoritmo concreto (K-Means en el ejercicio de hoy).
5. **Evaluación**: medición del desempeño con métricas adecuadas — no alcanza con entrenar, hay que poder decir *qué tan bueno* es el resultado con números concretos (el Módulo 5 desarrolla en profundidad qué métrica usar según el tipo de problema).
6. **Entrega mínima**: preparación para compartir o desplegar el modelo — notebooks reproducibles, un archivo de resultados, una demo simple. Es la etapa que más se salta en proyectos de facultad/portfolio, pero es la que determina si el trabajo le sirve a alguien más que a quien lo hizo.

**Para remarcar en clase**: este pipeline debe estar diseñado para que **cualquier persona** pueda seguirlo y obtener resultados consistentes — esa es, literalmente, la definición de reproducibilidad. Y las primeras tres etapas (ingesta, EDA, feature engineering) son, ni más ni menos, lo que hoy se construye en código real en el Bloque 1 del notebook.

> 👉 El código completo (`pipeline_preprocesamiento()`) y su explicación línea por línea están al **[final de este módulo, en Bloque 1](#bloque-1-pipeline)**, justo antes de pasar al Módulo 2. La idea es repasar primero **toda** la teoría de las filminas de este módulo (las que siguen acá abajo) y recién después pasar al ejemplo — así no se mezcla la teoría con el código.

<a id="filmina-05"></a>

### 1.2 Componentes clave para la reproducibilidad (Slide 05/43 — "Componentes Clave para la Reproducibilidad")

La filmina muestra 4 tarjetas con una línea cada una; acá va cada componente desarrollado, con ejemplos de herramientas concretas y qué pasa cuando falta.

**Gestión de Artefactos** — *"Guardar versiones de datasets, modelos entrenados y resultados intermedios."* En la práctica significa no solo generar el modelo, sino **guardarlo de forma que se pueda recuperar exactamente igual** más adelante: con `joblib.dump()` o `pickle` para el modelo entrenado, con un nombre de archivo que incluya fecha o versión, y con el dataset que se usó para entrenarlo guardado aparte (no sobreescrito por la próxima actualización). Herramientas más avanzadas de la industria, como MLflow o DVC, automatizan este registro. **Qué pasa si falta**: cada vez que hace falta reusar un modelo hay que re-entrenarlo desde cero (con el costo de tiempo y cómputo que eso implica), y si el dataset original cambió o se perdió, ni siquiera se puede reproducir el mismo modelo.

**Control de Versiones** — *"Uso de sistemas como Git para rastrear cambios en código y documentación."* Cada cambio en el pipeline queda registrado con quién lo hizo, cuándo y por qué (a través del mensaje de commit), y se puede volver atrás si algo se rompe. En un equipo, permite que varias personas trabajen sobre el mismo pipeline sin pisarse el código entre sí. **Qué pasa si falta**: cambios en el código se pierden o se sobreescriben sin registro, y frases como *"esto andaba la semana pasada, no sé qué le cambiaron"* se vuelven moneda corriente.

**Entornos Reproducibles** — *"Definir dependencias y versiones de librerías para evitar discrepancias."* Concretamente: un archivo `requirements.txt` (o un entorno de `conda`, o una imagen de Docker) que fija exactamente qué versión de `pandas`, `scikit-learn`, etc. se usó. Como se vio en el punto 1.0, hasta una diferencia de versión menor puede cambiar ligeramente un resultado numérico. **Qué pasa si falta**: el clásico "en mi máquina funciona" — el mismo código da error o un resultado distinto en la máquina de otra persona, simplemente porque tiene otra versión de una librería instalada.

**Documentación Clara** — *"Explicar cada paso y decisión tomada en el pipeline."* No es solo comentarios sueltos: es el docstring de cada función (como el que tiene `pipeline_preprocesamiento()` más arriba), explicar *por qué* se tomó una decisión de negocio puntual (por qué esos bins de `pd.cut()`, por qué se imputa con la media y no con la mediana), y dejar registro de qué preguntas de negocio responde el pipeline. **Qué pasa si falta**: el pipeline se vuelve una caja negra — funciona, pero nadie (ni siquiera quien lo escribió, meses después) puede explicar con confianza qué hace cada parte ni por qué.

<a id="filmina-06"></a>

### 1.3 Prácticas para despliegue mínimo y compartición (Slide 06/43 — "Prácticas para Despliegue Mínimo y Compartición")

El **despliegue mínimo** busca entregar una versión funcional del pipeline que permita a otros reproducir y validar resultados sin complejidades innecesarias:

- Notebooks bien estructurados (Jupyter, Google Colab) que integren código, visualizaciones y explicaciones.
- Repositorios organizados con código, datos y documentación.
- Artefactos guardados con nombres y formatos estándar.
- Demos simples con herramientas como **Streamlit** o **Flask** para mostrar resultados interactivos.

<a id="filmina-07"></a>

### 1.4 Por qué importa en la industria (Slide 07/43 — "Reproducibilidad en la Industria")

En un proyecto de **detección de fraude**, un pipeline reproducible permite que el equipo de ingeniería valide el modelo antes de integrarlo en producción, asegurando que los resultados sean consistentes y confiables. La gestión de artefactos y el control de versiones evitan pérdidas de trabajo y facilitan la colaboración entre equipos multidisciplinarios; el despliegue mínimo permite entregar prototipos funcionales que stakeholders pueden evaluar sin infraestructuras complejas.

**Gancho hacia el ejercicio práctico**: todo lo de este módulo deja de ser teoría en la sección que sigue — la función `pipeline_preprocesamiento()`, desarrollada línea por línea en [Bloque 1](#bloque-1-pipeline) a continuación, implementa exactamente las etapas 1-3 (ingesta, limpieza, transformación) sobre datos reales.

<a id="bloque-1-pipeline"></a>

### Bloque 1 — Pipeline de Ingesta y Transformación (20 min)

**El problema real**: en producción, los datos cambian constantemente (ej. todos los años el DEIS publica un nuevo renglón de natalidad). Si se procesan los datos con celdas de Jupyter sueltas o de forma manual, el código se rompe con cada actualización, toma tiempo rehacerlo y es propenso a errores humanos.

**La solución**: un **pipeline reproducible**. Es un conjunto de funciones modulares que automatizan la ingesta, limpieza y transformación. Entra un archivo crudo (raw data) y sale un objeto listo para el modelo de Machine Learning, sin intervención manual — esta es la implementación en código de las 6 etapas y los 4 componentes de reproducibilidad que se acaban de ver en este módulo (ver el [mapa de filminas](#filmina-04) si querés repasarlos antes de seguir).

**Particularidad de este dataset**: el archivo tiene los **años en las filas** y las **provincias en las columnas**. Para aplicar algoritmos de Machine Learning en scikit-learn, las filas deben ser las instancias (las provincias a segmentar) y las columnas deben ser las características (features, es decir, los años). Por lo tanto, el pipeline debe **transponer (`.T`)** la matriz y **escalar** los datos, ya que los modelos basados en distancias son sensibles a las magnitudes.

```python
#https://datos.salud.gob.ar/dataset/tasa-de-natalidad/archivo/0f68d5c6-e667-40ca-90fd-4784336e092e
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =====================================================================
# SECCIÓN 1: El Pipeline de Ingesta y Transformación
# =====================================================================

def pipeline_preprocesamiento(path_archivo):
    """Pipeline reproducible para limpiar y transformar el dataset de natalidad."""
    # 1. Carga de datos crudos
    df = pd.read_csv(path_archivo)

    # 2. Convertir el índice de tiempo a año y setearlo
    df['indice_tiempo'] = pd.to_datetime(df['indice_tiempo']).dt.year
    df.set_index('indice_tiempo', inplace=True)

    # 3. Transposición Crucial: Filas = Provincias (Instancias), Columnas = Años (Features)
    df_provincias = df.T

    # 4. Tratamiento de nulos por imputación matemática (Media por provincia)
    df_provincias = df_provincias.fillna(df_provincias.mean())

    # 5. Escalado de datos para algoritmos de distancia
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_provincias)

    return df_provincias, X_scaled

# Ejecución en vivo:
df_provincias, X = pipeline_preprocesamiento('tasa-natalidad-deis-2000-2024.csv')
print(f"Instancias a segmentar: {df_provincias.shape[0]} provincias.")
print(f"Cantidad de features por provincia: {df_provincias.shape[1]} años analizados.")
```

**Línea por línea** (con más detalle del que trae el propio notebook, que solo tiene comentarios cortos):

- **`def pipeline_preprocesamiento(path_archivo):`** — se define como **función con un parámetro** (`path_archivo`) en lugar de código suelto con el nombre del archivo escrito a mano en el medio. Esto es la reproducibilidad hecha código: la misma función sirve para el dataset de este año o para el del año que viene, sin tocar una sola línea de adentro — solo cambia qué ruta se le pasa al llamarla.
- **`"""Pipeline reproducible para limpiar y transformar..."""`** — el docstring (la cadena de texto justo debajo del `def`) es la documentación mínima de la función: cualquiera que la encuentre en el código (incluso sin leer el resto) sabe qué hace con solo pedir `help(pipeline_preprocesamiento)`. Es la aplicación concreta del componente "Documentación Clara" de la filmina 05.
- **`df = pd.read_csv(path_archivo)`** — la etapa de **ingestión**: carga el archivo crudo tal cual llega, sin modificarlo todavía.
- **`df['indice_tiempo'] = pd.to_datetime(df['indice_tiempo']).dt.year`** — la columna `indice_tiempo` llega como texto con formato de fecha (ej. `"01-01-2024"`). `pd.to_datetime(...)` la convierte a un objeto de fecha real que pandas entiende, y `.dt.year` extrae **solo el número de año** de esa fecha (`2024`), descartando el mes y el día, que acá no aportan nada porque el dataset es anual.
- **`df.set_index('indice_tiempo', inplace=True)`** — convierte esa columna de año en el **índice** del DataFrame (la "etiqueta" de cada fila) en lugar de dejarla como una columna más. `inplace=True` significa que modifica `df` directamente, sin necesidad de reasignarlo (`df = df.set_index(...)` haría lo mismo, pero de forma explícita en vez de "en el lugar").
- **`df_provincias = df.T`** — la **transposición**, ya explicada en el Bloque 0: da vuelta filas y columnas para que cada **provincia** pase a ser una fila (una instancia a segmentar) y cada **año** pase a ser una columna (una feature).
- **`df_provincias = df_provincias.fillna(df_provincias.mean())`** — la etapa de **limpieza** de nulos dentro del pipeline: donde haya un `NaN`, lo reemplaza por el promedio. Matiz técnico para tener claro: `.mean()` sin especificar eje calcula el promedio **por columna** (es decir, acá, el promedio de *ese año* entre todas las provincias), no el promedio histórico propio de cada provincia — el comentario del código dice "media por provincia" pero, estrictamente, es una media por año/columna. En este dataset puntual no cambia nada porque no hay ningún nulo (confirmado en el Bloque 0), pero es un detalle importante si mañana se reutiliza este mismo pipeline con un dataset que sí tenga huecos.
- **`scaler = StandardScaler()` / `X_scaled = scaler.fit_transform(df_provincias)`** — la etapa de **transformación**: crea el escalador y lo aplica de una, calculando media/desvío de cada columna (año) y convirtiendo todo a z-score, tal como se explicó en el punto 4 del Bloque 0.
- **`return df_provincias, X_scaled`** — la función devuelve **dos objetos, no uno**: `df_provincias` es la versión "humana" de los datos (un DataFrame de pandas, con los nombres de provincia como índice, fácil de inspeccionar o graficar) y `X_scaled` es la versión "para la máquina" (un array de NumPy ya escalado, sin nombres, listo para entrar directo a `KMeans`). Devolver ambos evita tener que recalcular uno a partir del otro más adelante en el notebook.
- **`df_provincias, X = pipeline_preprocesamiento('tasa-natalidad-deis-2000-2024.csv')`** — acá se ve el beneficio de haber armado una función: todo el trabajo de las líneas anteriores se dispara con **un solo llamado**, pasándole la ruta del archivo como único dato variable.
- Las dos líneas de `print` finales muestran `df_provincias.shape` (25 provincias, 25 años) para confirmar en pantalla que el pipeline transformó los datos como se esperaba, antes de seguir adelante.

**El detalle no obvio para explicar bien en el pizarrón**: el CSV original tiene los **años en las filas** y las **provincias en las columnas** — el formato natural para leer una serie de tiempo. Pero para que scikit-learn segmente **provincias** (no años), necesitamos que cada fila sea una provincia y cada columna sea una característica (un año) — de ahí la **transposición (`.T`)**. Es un paso conceptual, no solo técnico: cambia qué es "una instancia" para el algoritmo.

Esto implementa en código las etapas 1-3 del Módulo 1 (ingestión, limpieza, feature engineering vía transposición + escalado).

**Las otras piezas de la reproducibilidad (más allá del código)**: la función `pipeline_preprocesamiento()` resuelve la parte de ingesta, limpieza y transformación del pipeline end-to-end. Pero un pipeline reproducible "de verdad" en la industria también incluye piezas que no se ven en el código de una celda, y vale la pena nombrarlas aunque hoy no se implementen:

- **Gestión de artefactos**: guardar versiones del dataset, del modelo de K-Means entrenado y de los resultados intermedios (ej. con `joblib.dump()` o `pickle`), para no tener que re-entrenar desde cero cada vez.
- **Control de versiones**: usar Git para rastrear cambios en el código del pipeline y en la documentación — así cualquier persona del equipo puede ver qué cambió y por qué.
- **Entornos reproducibles**: fijar las versiones de pandas, scikit-learn, etc. (por ejemplo con un `requirements.txt`) para que el pipeline dé el mismo resultado en cualquier máquina.
- **Documentación clara**: el docstring de la función y los comentarios de esta celda cumplen ese rol a pequeña escala.
- **Despliegue mínimo**: la forma más simple de compartir este análisis es el propio notebook reproducible; en un caso más avanzado, se podría envolver en una demo interactiva con Streamlit para que el Ministerio explore los clusters sin tocar código.

En síntesis: hoy se construye la parte de ingesta + transformación del pipeline (la que se ejecuta), pero la reproducibilidad completa de un proyecto real se apoya en estas cinco patas trabajando juntas.

---

## Módulo 2 — Casos de Estudio: Segmentación y Recomendaciones

**Pregunta disparadora**: trabajás en una empresa de comercio electrónico que quiere aumentar sus ventas personalizando la experiencia de sus clientes. ¿Cómo identificar grupos de clientes con comportamientos similares? ¿Cómo recomendar productos que realmente interesen a cada usuario? Estas preguntas son el corazón de dos aplicaciones clave de ML en la industria.

### 2.1 Segmentación de Clientes

Dividir una población de clientes en grupos homogéneos según características o comportamientos similares, para personalizar estrategias de marketing, mejorar la retención y optimizar recursos.

- **Métodos**: clustering no supervisado (K-means, DBSCAN, clustering jerárquico) o segmentación basada en reglas definidas por expertos.
- **Requisitos de datos**: variables relevantes y limpias (demográficas, transaccionales, comportamiento web), con volumen suficiente para detectar patrones significativos.
- **Métricas de éxito**: Silhouette Score (cohesión y separación de clusters) e impacto en KPIs comerciales (ventas, retención, conversión).

### 2.2 Sistemas de Recomendación

Sugerir productos o contenidos personalizados para cada usuario, aumentando satisfacción y ventas.

- **Tipos**: filtrado colaborativo (similitud entre usuarios o ítems), filtrado basado en contenido (características del producto + preferencias del usuario), modelos híbridos (combinan ambos).
- **Requisitos de datos**: historial de interacciones usuario-producto, información contextual (tiempo, ubicación).
- **Métricas de éxito**: precisión y recall, tasa de clics (CTR) y conversión, diversidad y novedad (para evitar recomendaciones repetitivas).

### 2.3 Comparación y Trade-offs

| Aspecto | Segmentación | Recomendaciones |
|---|---|---|
| Tipo de ML | No supervisado | Supervisado / híbrido |
| Objetivo | Agrupar clientes | Personalizar experiencia |
| Datos requeridos | Variables descriptivas | Interacciones usuario-producto |
| Métricas clave | Cohesión de grupos, impacto negocio | Precisión, CTR, diversidad |
| Complejidad | Moderada | Alta (requiere modelado avanzado) |

### 2.4 Aplicación práctica combinada

En la práctica, una empresa puede usar **segmentación** para identificar grupos de clientes con alta propensión a comprar un nuevo producto, y luego aplicar **sistemas de recomendación** para personalizar ofertas dentro de cada segmento. Ejemplo: un retailer online segmenta a sus clientes por frecuencia de compra y categorías preferidas, y luego usa un recomendador híbrido para sugerir productos nuevos o complementarios — maximizando el impacto comercial y el uso de datos y recursos.

**Concepto clave que atraviesa toda la unidad — "Analytics to Action"**: ni la segmentación ni la recomendación generan valor por sí solas. Un cluster de clientes o un score de similitud entre productos es solo el punto medio del pipeline (etapa "Evaluación" del diagrama de la Introducción); el valor de negocio aparece recién cuando ese resultado técnico se traduce en una **decisión o acción concreta**: una campaña dirigida, una oferta personalizada, una alerta. Este es el hilo que conecta este módulo con los 4 casos del Módulo 3 y con el cierre del ejercicio práctico del notebook (Bloque 4).

---

## Módulo 3 — Casos de Éxito: ML en Acción

Cuatro casos reales que muestran cómo se ve todo lo anterior aplicado en la industria. Conviene presentarlos como historias, no como listas de bullets — cada uno tiene un problema de negocio, una técnica y un cierre con impacto medible.

### 3.1 San Cristóbal — Detección de Fraudes

**El problema**: ¿cómo protegerse eficazmente contra fraudes que amenazan la operación y la confianza con los clientes? La detección de fraude es un problema clásico de **clasificación supervisada**: identificar transacciones o eventos fraudulentos entre un gran volumen de datos legítimos.

**El pipeline en 3 etapas**:
1. **Detección inicial**: modelos supervisados entrenados con datos históricos clasifican eventos como fraudulentos o no.
2. **Investigación**: análisis detallado, incluyendo revisión manual y análisis de imágenes de siniestros mediante Deep Learning.
3. **Resolución**: toma de decisiones basada en resultados y métricas para mitigar el fraude.

**El desafío del desbalance**: los fraudes son eventos raros, por lo que los datos están altamente desbalanceados. Se abordan con:
- **Oversampling**: aumentar artificialmente los ejemplos minoritarios (fraudes) — la técnica más usada es **SMOTE**.
- **Undersampling**: reducir los ejemplos de la clase mayoritaria.

**Métricas críticas**: **Precisión** (proporción de predicciones positivas correctas) y **Recall** (proporción de fraudes reales detectados) — en este dominio, **un alto recall es vital** para no dejar pasar fraudes, aunque se sacrifiquen algunos falsos positivos. El **AUC** (área bajo la curva ROC) mide la capacidad de distinguir clases en distintos umbrales, y se usa **cross-validation** para asegurar la robustez del modelo.

**El plus de Deep Learning**: San Cristóbal complementa la detección tabular con **redes neuronales convolucionales (CNN)** que analizan imágenes de siniestros, extrayendo características automáticas y detectando patrones visuales indicativos de fraude. **Matplotlib** se usa para visualizar resultados y comunicar hallazgos a stakeholders.

**Código de referencia de la práctica** (dataset: `csv/creditcardfraud`):

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier

smote = SMOTE(random_state=RANDOM_STATE)
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)

pipeline_smote_rf = ImbPipeline(steps=[
    ('smote', smote),
    ('rf', rf)
])
pipeline_smote_rf.fit(X_train, y_train)
```

**Línea por línea**:

- **`from imblearn.over_sampling import SMOTE`**: `imblearn` (*imbalanced-learn*) es una librería que extiende a scikit-learn específicamente para problemas de **clases desbalanceadas** — como fraude, donde el 99%+ de las transacciones son legítimas y muy pocas son fraude. `SMOTE` (*Synthetic Minority Oversampling Technique*) implementa el sobremuestreo: en vez de duplicar los pocos casos de fraude que hay, **genera ejemplos sintéticos nuevos** interpolando entre casos de fraude reales parecidos entre sí, para que el modelo vea más variedad de "fraude" en vez de la misma fila copiada muchas veces.
- **`from imblearn.pipeline import Pipeline as ImbPipeline`**: no se usa el `Pipeline` normal de `sklearn.pipeline`, sino uno especial de `imblearn`, porque el de scikit-learn no permite pasos que cambien la **cantidad de filas** (como hace SMOTE, que agrega filas sintéticas) — solo pasos que transforman columnas. El `ImbPipeline` sí lo permite. Se importa con alias (`as ImbPipeline`) para no confundirlo si en algún momento también se usa el `Pipeline` común.
- **`from sklearn.ensemble import RandomForestClassifier`**: el submódulo `ensemble` agrupa algoritmos que combinan muchos modelos simples en uno más robusto. `RandomForestClassifier` entrena muchos árboles de decisión distintos (cada uno viendo una muestra distinta de los datos) y combina sus votos, reduciendo el sobreajuste que tendría un solo árbol.
- **`smote = SMOTE(random_state=RANDOM_STATE)`**: crea el objeto SMOTE. `random_state` fija la semilla aleatoria (mismo concepto que en K-Means) para que la generación de ejemplos sintéticos sea reproducible entre corridas.
- **`rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)`**: crea el clasificador. `n_estimators=100` fija el "bosque" en 100 árboles. `n_jobs=-1` le dice que use **todos los núcleos del procesador disponibles** para entrenar los árboles en paralelo.
- **`pipeline_smote_rf = ImbPipeline(steps=[('smote', smote), ('rf', rf)])`**: encadena los dos pasos en un solo objeto — primero SMOTE, después el Random Forest — con nombres de referencia (`'smote'`, `'rf'`) para cada paso. Mismo espíritu de "pipeline reproducible" del Módulo 1, aplicado acá a modelado en vez de a preprocesamiento.
- **`pipeline_smote_rf.fit(X_train, y_train)`**: dispara todo con un solo llamado — `fit` primero aplica SMOTE **solo a los datos de entrenamiento** (`X_train`, `y_train`), generando los ejemplos sintéticos de fraude, y después entrena el Random Forest ya con los datos balanceados. Es clave que SMOTE se aplique solo al *train* y nunca al *test*: si se aplicara también al test, el modelo se evaluaría con datos sintéticos "inventados", dando una métrica de rendimiento falsamente optimista.

La práctica completa pide: carga y EDA del desbalance → oversampling/undersampling → entrenar Random Forest o Logistic Regression → calcular precisión/recall/F1/AUC, matriz de confusión y curva ROC → interpretar y justificar la elección de métricas y técnicas.

### 3.2 Medplaya — Analítica Predictiva en Hotelería

**El problema**: las cancelaciones de reservas afectan la ocupación y los ingresos de una cadena hotelera. ¿Cómo anticiparlas para optimizar la gestión de habitaciones y maximizar el revenue?

**Modelos de clasificación supervisada** para predecir si una reserva será cancelada:
- **Árboles de decisión**: interpretables, segmentan el espacio de características.
- **Random Forest**: ensamble de árboles, mejora precisión y reduce sobreajuste.
- **Regresión logística**: modelo probabilístico para clasificación binaria.
- **Boosting (XGBoost)**: potencia modelos débiles para mejorar rendimiento.

**Selección de features**:
- **Comportamiento histórico**: frecuencia de cancelaciones previas, tiempo entre reserva y llegada.
- **Señales contextuales**: temporada, eventos locales, tipo de habitación, canal de reserva.
- **Feature engineering**: transformaciones numéricas, one-hot encoding, variables derivadas (ej. tasa de cancelación por cliente).

**Desequilibrio**: las cancelaciones son menos frecuentes que las confirmaciones — se aborda con re-muestreo o algoritmos que ponderan clases.

**Métricas de evaluación**:

| Métrica | Descripción | Importancia en cancelaciones |
|---|---|---|
| Precisión | Proporción de predicciones correctas | Evita falsas alarmas de cancelación |
| Recall | Proporción de cancelaciones detectadas | Crucial para anticipar cancelaciones reales |
| F1-Score | Balance entre precisión y recall | Útil en desequilibrio de clases |
| AUC-ROC | Capacidad de distinguir entre clases | Evalúa rendimiento global del modelo |

**Pregunta para lanzar a la clase (viene textual del PDF)**: *¿por qué podría ser más importante maximizar el recall que la precisión en este caso?* (Respuesta esperada: una cancelación no detectada cuesta más que una falsa alarma — se pierde la oportunidad de re-vender esa habitación.)

**El cierre de negocio — Overbooking Controlado**: con predicciones confiables, se acepta más reservas que la capacidad real para compensar cancelaciones esperadas, optimizando ocupación y revenue. Requiere un balance cuidadoso para evitar sobreventa y mala experiencia al cliente. Medplaya aplicó este pipeline completo (limpieza → features → Random Forest + regresión logística → evaluación con F1/AUC-ROC → política de overbooking) y logró aumentar la ocupación promedio y mejorar el revenue.

### 3.3 Amazon — Sistemas de Recomendación

**El problema**: ¿cómo logra Amazon ofrecer recomendaciones precisas entre millones de productos y usuarios?

**Tres enfoques**: filtrado colaborativo, basado en contenido, y sistemas híbridos (la combinación de ambos suele ofrecer mejores resultados, equilibrando precisión y diversidad).

**Matrix Factorization y Embeddings**: la técnica central del filtrado colaborativo. Representa las interacciones usuario-ítem en espacios latentes:
- **SVD** (Singular Value Decomposition): descompone la matriz de interacciones en factores latentes que capturan características implícitas.
- **ALS** (Alternating Least Squares): optimiza iterativamente la factorización, eficiente para grandes conjuntos de datos.

Estas técnicas generan **embeddings** que permiten predecir la afinidad entre usuarios e ítems no observados. *Ejemplo del PDF*: en Amazon, la matriz usuario-producto es enorme y dispersa; ALS permite factorizarla para descubrir patrones latentes y recomendar productos relevantes.

**Impacto en el negocio**: se mide con **experimentación A/B** (comparar grupos con y sin recomendaciones) y métricas de negocio (ROI, tasa de conversión, valor promedio de pedido).

**Consideraciones de despliegue en tiempo real**: minimizar latencia, actualizar modelos periódicamente, integrar pipelines de scoring batch y streaming.

**Práctica: recomendador de películas con NLP + similitud de coseno** (el código que trae el PDF, pensado para correr en Colab):

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF sobre el texto combinado de géneros + keywords de cada película
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

# Similitud de coseno entre todas las películas
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        return ["La película no existe en la base de datos."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # top 10, excluyendo la misma película
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()
```

**Línea por línea**:

- **`from sklearn.feature_extraction.text import TfidfVectorizer`**: el submódulo `feature_extraction.text` convierte **texto** en números. `TfidfVectorizer` implementa **TF-IDF** (*Term Frequency – Inverse Document Frequency*): convierte cada texto (acá, géneros + keywords de cada película) en un vector numérico, dándole más peso a las palabras frecuentes *en esa película puntual* pero raras *en el resto del catálogo* (palabras como "the" o "action" pesan poco por ser comunes; una keyword muy específica pesa mucho).
- **`from sklearn.metrics.pairwise import cosine_similarity`**: el submódulo `metrics.pairwise` calcula distancias/similitudes **entre pares** de vectores. `cosine_similarity` mide el **coseno del ángulo** entre dos vectores — cuanto más chico el ángulo (vectores "apuntando" en la misma dirección), más cercano a 1 el resultado, y más parecidas son las películas en contenido.
- **`tfidf = TfidfVectorizer(stop_words='english')`**: crea el vectorizador. `stop_words='english'` le dice que ignore palabras vacías del inglés (*the*, *and*, *of*...) que no aportan significado.
- **`tfidf_matrix = tfidf.fit_transform(movies['combined_features'])`**: `fit_transform` hace dos cosas en un paso (mismo patrón que `StandardScaler` en el Módulo 1): **aprende** el vocabulario de todas las películas (`fit`) y **convierte** cada película en su vector TF-IDF (`transform`). El resultado es una matriz con una fila por película y una columna por cada palabra única del vocabulario.
- **`cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)`**: calcula la similitud **de cada película contra todas las demás** (incluida ella misma) — el resultado es una matriz cuadrada N×N (N = cantidad de películas), donde la celda `[i, j]` dice qué tan parecida es la película `i` a la película `j`.
- **`def get_recommendations(title, cosine_sim=cosine_sim):`**: función que recibe el título de una película y devuelve recomendaciones. Usar `cosine_sim=cosine_sim` como valor por defecto evita recalcular la matriz cada vez que se llama a la función.
- **`if title not in indices: return [...]`**: chequeo de seguridad — si el título no existe en el dataset, devuelve un mensaje en vez de romper con un error.
- **`idx = indices[title]`**: `indices` es un diccionario (definido en otra celda del notebook) que mapea título → posición numérica de esa película en la matriz.
- **`sim_scores = list(enumerate(cosine_sim[idx]))`**: `cosine_sim[idx]` es la fila de la matriz correspondiente a esa película — un vector con su similitud contra todas las demás. `enumerate(...)` empareja cada valor con su posición, generando pares `(posición, similitud)`.
- **`sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)`**: ordena esos pares de mayor a menor similitud (`x[1]` es la similitud; `reverse=True` para que el más parecido quede primero).
- **`sim_scores = sim_scores[1:11]`**: descarta la posición `0` (la película contra sí misma, similitud 1.0 — no sirve como recomendación) y toma las siguientes 10 más parecidas.
- **`movie_indices = [i[0] for i in sim_scores]`**: extrae solo las posiciones (`i[0]`) de esos 10 resultados, descartando ya el valor de similitud.
- **`return movies['title'].iloc[movie_indices].tolist()`**: usa esas posiciones para buscar los títulos reales en el DataFrame original (`.iloc` porque son posiciones numéricas, no nombres de índice) y devuelve la lista final de 10 títulos recomendados.

**Para explicar el mecanismo en el pizarrón**: TF-IDF convierte el texto de cada película (géneros + keywords) en un vector numérico que pesa cada palabra según su importancia; la similitud de coseno mide el ángulo entre dos vectores — cuanto más chico el ángulo, más parecidas son las películas en contenido. No usa historial de usuarios (no es filtrado colaborativo), es **basado en contenido**.

### 3.4 Mazda — Segmentación de Clientes con Clustering

**El problema**: ¿cómo puede una empresa como Mazda entender mejor a sus clientes para ofrecer productos y servicios personalizados? A partir de un conjunto de **más de 30 variables**, se prepara los datos, se seleccionan características relevantes y se aplican algoritmos de clustering para identificar segmentos significativos.

**Conceptos clave**:
- **Clustering**: agrupamiento de datos sin etiquetas previas, buscando patrones latentes.
- **K-Means**: particiona los datos en *k* clusters minimizando la varianza intra-cluster.
- **Gaussian Mixture Models (GMM)**: modelo probabilístico que asume una mezcla de distribuciones gaussianas, permitiendo clusters con formas elípticas (no solo esféricas como K-Means).

*Cita del PDF*: según Aggarwal (2015), el clustering es efectivo para segmentación cuando existen patrones latentes en atributos de clientes, facilitando la personalización y optimización de campañas.

**Feature engineering y preparación**: limpieza (valores faltantes, errores), selección de variables relevantes, y **escalado** (fundamental — herramientas como `pandas` y `numpy` facilitan estas tareas).

**Pipeline de datos para clustering (6 pasos)**:
1. Ingestión y limpieza de datos
2. Selección y transformación de features
3. Escalado y normalización
4. Aplicación del algoritmo de clustering
5. Evaluación y validación de clusters
6. Interpretación y aplicación de resultados

**Evaluación de clusters**: **Inercia** (suma de distancias cuadradas dentro de clusters) y **Silhouette Score** (medida de separación y cohesión) — ayudan a decidir el número óptimo de segmentos y la robustez del modelo.

**Traducir lo técnico a negocio**:

| Métrica Técnica | Indicador de Negocio |
|---|---|
| Segmentos definidos y estables | Estrategias de marketing dirigidas y efectivas |
| Características distintivas | Personalización de ofertas y comunicación |

> **Este es el caso que el notebook de hoy reproduce en vivo**, con datos de natalidad en vez de clientes de Mazda — mismo pipeline de 6 pasos, mismo algoritmo (K-Means), mismas métricas de validación (Inercia + Silhouette). Ver la [sección 7](#7-el-ejercicio-práctico-del-notebook-explicado-en-profundidad) para el desarrollo completo.

---

## Módulo 4 — Supervisado vs. No Supervisado

### 🗺️ Mapa de filminas de este módulo

Cada fila es una diapositiva real de `Semana 7.html` (el número es el que ves en el pie de página, ej. `30 / 43`), en el mismo orden en que van pasando. Hacé clic para saltar directo a su explicación extendida.

| Filmina | Título en la diapositiva | Explicación extendida acá |
|---|---|---|
| Slide 29/43 | *(Divisor)* Supervisado vs. No Supervisado | [Intro del módulo](#filmina-29) |
| Slide 30/43 | Aprendizaje Supervisado | [→ Ir a la explicación](#filmina-30) |
| Slide 31/43 | Aprendizaje No Supervisado | [→ Ir a la explicación](#filmina-31) |
| Slide 32/43 | Comparación Práctica | [→ Ir a la explicación](#filmina-32) |
| Slide 33/43 | Aplicaciones Combinadas en la Industria | [→ Ir a la explicación](#filmina-33) |
| Slide 34/43 | Break de Coder ("Recreo") | [→ Ver sección](#break-del-coder) |

<a id="filmina-29"></a>

**Pregunta disparadora**: tenés un enorme conjunto de datos de clientes, pero no sabés qué patrones o grupos existen dentro de ellos. ¿Cómo segmentarlos para campañas personalizadas? ¿O cómo predecir si un cliente comprará, en base a su comportamiento previo? Estas preguntas ilustran los dos grandes enfoques de la ciencia de datos.

<a id="filmina-30"></a>

### 4.1 Aprendizaje Supervisado (Slide 30/43 — "Aprendizaje Supervisado")

Los modelos aprenden a partir de **datos etiquetados** (pares entrada-salida conocidos), buscando generalizar para predecir la salida correcta en nuevas entradas.

- **Clasificación**: predice una categoría (ej. ¿es spam?).
- **Regresión**: predice un valor numérico continuo (ej. precio de una vivienda).

| Algoritmo | Descripción breve | Ejemplo de uso en industria |
|---|---|---|
| Regresión lineal | Modela la relación lineal entre variables | Predicción de ventas según inversión publicitaria |
| Random Forest | Conjunto de árboles de decisión | Detección de fraude en transacciones bancarias (San Cristóbal) |

**Supuestos y consideraciones**: requiere datos etiquetados (costoso/difícil de conseguir); el modelo aprende patrones explícitos entre entrada y salida; es fundamental elegir métricas adecuadas (precisión, recall, RMSE, etc.).

**Concepto para reforzar (no viene textual en el PDF, pero es la base de todo lo anterior)**: el objetivo de un modelo supervisado no es memorizar los datos de entrenamiento, sino **generalizar** — funcionar bien con datos nuevos que nunca vio. Cuando un modelo aprende "de memoria" el ruido específico de sus datos de entrenamiento y pierde capacidad de generalizar, se llama **overfitting** (sobreajuste); es la razón por la que en el Módulo 5 se insiste tanto en evaluar siempre con datos separados del entrenamiento (hold-out, K-Fold), y no con las mismas filas que el modelo ya vio.

<a id="filmina-31"></a>

### 4.2 Aprendizaje No Supervisado (Slide 31/43 — "Aprendizaje No Supervisado")

Trabaja con **datos sin etiquetas**, buscando estructuras o patrones ocultos.

- **Clustering**: agrupa datos similares en segmentos (ej. segmentación de clientes — casos Mazda y el ejercicio de hoy).
- **Reducción de dimensionalidad**: simplifica datos complejos conservando la mayor información posible (ej. PCA).

| Algoritmo | Descripción breve | Ejemplo de uso en industria |
|---|---|---|
| K-Means | Agrupa datos en *k* clusters basados en distancia | Segmentación de usuarios en plataformas digitales |
| PCA | Transforma variables correlacionadas en componentes independientes | Visualización y reducción de variables en análisis financiero |

**Supuestos y consideraciones**: no requiere etiquetas, ideal para exploración; los resultados pueden ser menos interpretables que en supervisado; hay que validar la calidad de los clusters o componentes obtenidos.

<a id="filmina-32"></a>

### 4.3 Comparación práctica (Slide 32/43 — "Comparación Práctica")

| Aspecto | Aprendizaje Supervisado | Aprendizaje No Supervisado |
|---|---|---|
| Datos | Etiquetados (entrada-salida) | Sin etiquetas |
| Objetivo | Predecir o clasificar | Encontrar estructura o patrones |
| Ejemplos de aplicación | Detección de fraude, clasificación de imágenes | Segmentación de clientes, reducción de variables |

> La elección entre supervisado y no supervisado depende del problema, la disponibilidad de datos y el objetivo final.

<a id="filmina-33"></a>

### 4.4 Aplicaciones combinadas en la industria (Slide 33/43 — "Aplicaciones Combinadas en la Industria")

En la práctica, los científicos de datos suelen combinar ambos enfoques:

- **Segmentación de clientes**: clustering para identificar grupos, luego modelos supervisados para predecir la respuesta a campañas.
- **Detección de fraude**: Random Forest para clasificar transacciones, apoyado por análisis no supervisado para descubrir patrones nuevos.
- **Optimización logística**: reducción de dimensionalidad para simplificar variables y mejorar la eficiencia de modelos predictivos.

<a id="bloque-2-kmeans"></a>

### Bloque 2 — Supervisado vs. No Supervisado + K-Means (20 min)

**Teoría y escenario real**: si quisiéramos predecir el número exacto de nacimientos en Formosa para 2026, sería un problema **Supervisado (Regresión)**, porque tendríamos una etiqueta histórica (los nacimientos reales de años anteriores) para guiar al modelo.

Pero en nuestro caso real, el Ministerio de Salud no sabe de antemano cuáles son los perfiles demográficos del país. No hay etiquetas de "Provincia con natalidad decreciente" o "Provincia con natalidad estable" — buscamos que el algoritmo encuentre esos patrones ocultos por sí solo. Esto es **Aprendizaje No Supervisado**.

**Caso de estudio (Segmentación)**: usamos el algoritmo **K-Means**. Agrupa las provincias asignando centros geométricos (centroides) e iterando hasta que cada provincia queda asociada al centroide más cercano por distancia euclidiana.

**Ubicando esto en el panorama completo** (retomando la tabla del Módulo 4):

| | Supervisado | No Supervisado (lo que hacemos hoy) |
|---|---|---|
| **Datos** | Etiquetados (entrada → salida conocida) | Sin etiquetas |
| **Objetivo** | Predecir o clasificar | Encontrar estructura o patrones |
| **Algoritmos típicos** | Regresión lineal, Random Forest | K-Means (hoy), PCA |
| **Ejemplo de industria** | Detección de fraude (San Cristóbal), predicción de cancelaciones (Medplaya) | Segmentación de clientes (Mazda) |

Este ejercicio de segmentar provincias por su perfil de natalidad es, en esencia, el mismo tipo de problema que resuelve **Mazda** cuando segmenta clientes con K-Means y GMM a partir de más de 30 variables: agrupar observaciones sin etiquetas previas para después diseñar una estrategia distinta por grupo. La diferencia es el dominio (demografía pública vs. clientes de una automotriz), no la lógica del algoritmo.

En cambio, si más adelante quisiéramos predecir el valor exacto de natalidad del año próximo para una provincia (un número continuo, con datos históricos como "etiqueta"), pasaríamos a un problema de **regresión supervisada** — el mismo enfoque que usan San Cristóbal (clasificación de fraude) o Medplaya (clasificación de cancelaciones), aunque con un algoritmo distinto según si predecimos una categoría o un número.

```python
kmeans_prueba = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_prueba = kmeans_prueba.fit_predict(X)
```

**K-Means en una frase para el pizarrón**: el algoritmo ubica *k* centros geométricos (centroides) y asigna cada provincia al centroide más cercano por distancia euclidiana, iterando hasta que las asignaciones se estabilizan. `random_state=42` fija la semilla aleatoria para que el resultado sea reproducible entre corridas — otro gancho directo al Módulo 1. `X` es el array ya escalado que devolvió `pipeline_preprocesamiento()` en el Bloque 1.

**¿Los 3 parámetros son arbitrarios o tienen una razón?**

- **`n_clusters=3`** — es **arbitrario a propósito**, no tiene todavía una justificación técnica. Es el gancho pedagógico hacia el Bloque 3: primero se prueba con un número cualquiera, y recién ahí se usa el método del codo (inercia) y el Silhouette Score para *justificar* si 3 es realmente el mejor valor de K, o si convendría otro.
- **`random_state=42`** — tampoco tiene significado matemático. Es solo una semilla fija para que la parte aleatoria del algoritmo (la posición inicial de los centroides) dé siempre el mismo resultado entre corridas — el "42" en particular es una convención cultural muy usada en ejemplos de ML (referencia a *La guía del autoestopista galáctico*). Podría ser cualquier otro entero; lo importante es que quede fijo, no cuál sea.
- **`n_init=10`** — esta sí tiene una razón técnica real: K-Means arranca ubicando los centroides en posiciones aleatorias, y según dónde arranquen puede terminar en una solución subóptima (un "mínimo local"). `n_init=10` corre el proceso completo 10 veces, cada una con una inicialización distinta, y se queda con la mejor (menor inercia) — una salvaguarda estándar contra una mala inicialización, no un valor "que da lo mismo".

Este bloque es, en la práctica, una implementación completa del **caso Mazda** (Módulo 3.4): mismo algoritmo, mismo tipo de problema (segmentación sin etiquetas), aplicado a un dominio distinto.

---

## Break del Coder

Corte de ~10 minutos, después del Módulo 4 y antes de arrancar el Módulo 5 (Métricas y Validación) — cierra la parte de "qué algoritmo elegir" y abre la parte de "cómo saber si funcionó".

---

## Módulo 5 — Métricas y Estrategias de Validación

### 🗺️ Mapa de filminas de este módulo

Cada fila es una diapositiva real de `Semana 7.html` (el número es el que ves en el pie de página, ej. `36 / 43`), en el mismo orden en que van pasando. Hacé clic para saltar directo a su explicación extendida.

| Filmina | Título en la diapositiva | Explicación extendida acá |
|---|---|---|
| Slide 35/43 | *(Divisor)* Métricas y Estrategias de Validación | [Intro del módulo](#filmina-35) |
| Slide 36/43 | Métricas de Clasificación | [→ Ir a la explicación](#filmina-36) |
| Slide 37/43 | Métricas de Regresión y Clustering | [→ Ir a la explicación](#filmina-37) (cubre 5.2 y 5.3) |
| Slide 38/43 | Estrategias de Validación | [→ Ir a la explicación](#filmina-38) |
| Slide 39/43 | Trade-offs en la Selección de Métricas | [→ Ir a la explicación](#filmina-39) |
| Slide 40/43 | Aplicación Práctica: Métricas en la Industria | *(sin sub-sección dedicada — ver nota en 5.5)* |
| Slide 41/43 | Para Practicar: Métricas y Validación | [→ Ir a la explicación](#filmina-41) |

<a id="filmina-35"></a>

**Pregunta disparadora**: trabajás en una empresa de logística que quiere optimizar rutas de entrega usando modelos predictivos. ¿Cómo estar seguro de que el modelo realmente mejora la eficiencia y no solo funciona bien con los datos que ya tenés? La respuesta está en evaluar correctamente el modelo.

<a id="filmina-36"></a>

### 5.1 Métricas de Clasificación (Slide 36/43 — "Métricas de Clasificación")

- **Accuracy**: proporción de predicciones correctas sobre el total. Útil cuando las clases están balanceadas.
- **Precision**: proporción de verdaderos positivos sobre todos los positivos predichos. Importante cuando el costo de falsos positivos es alto.
- **Recall (Sensibilidad)**: proporción de verdaderos positivos sobre todos los positivos reales. Clave cuando es crítico detectar todos los casos positivos.
- **F1-Score**: media armónica entre precision y recall.
- **AUC-ROC**: área bajo la curva ROC, mide la capacidad de distinguir clases en diferentes umbrales.

*Ejemplo del PDF*: en detección de fraude, un alto recall es vital para no dejar pasar fraudes, aunque se sacrifiquen algunos falsos positivos (conecta directo con San Cristóbal, Módulo 3).

**La base de todas estas métricas (útil tenerla a mano, aunque el PDF no la despliega explícitamente)**: todas salen de comparar la predicción del modelo contra la realidad en una **matriz de confusión** de 2x2:

| | Predicho Positivo | Predicho Negativo |
|---|---|---|
| **Real Positivo** | Verdadero Positivo (VP) | Falso Negativo (FN) |
| **Real Negativo** | Falso Positivo (FP) | Verdadero Negativo (VN) |

De ahí salen las fórmulas: `Precision = VP / (VP + FP)` (de todo lo que dije que era positivo, ¿cuánto acerté?) y `Recall = VP / (VP + FN)` (de todo lo que era positivo en la realidad, ¿cuánto detecté?). Tenerlas escritas así ayuda mucho cuando un alumno pregunta "¿pero por qué no es lo mismo precision que recall?" — la diferencia está en el denominador: uno mira desde las predicciones, el otro desde la realidad.

<a id="filmina-37"></a>

### 5.2 Métricas de Regresión (Slide 37/43 — "Métricas de Regresión y Clustering")

- **RMSE** (Root Mean Squared Error): raíz del promedio de los errores al cuadrado, penaliza errores grandes.
- **MAE** (Mean Absolute Error): promedio de errores absolutos, más robusto a outliers.
- **R²** (Coeficiente de determinación): proporción de varianza explicada por el modelo.

*Ejemplo del PDF*: para predecir demanda de productos, RMSE ayuda a entender el error típico en unidades vendidas.

### 5.3 Métricas de Clustering (misma Slide 37/43 — "Métricas de Regresión y Clustering")

- **Silhouette Score**: mide qué tan bien separado está cada cluster.
- **Davies-Bouldin Index**: evalúa la separación y compacidad de clusters.
- **Inercia**: suma de distancias cuadradas dentro de clusters, usada en k-means.

*Ejemplo del PDF*: en segmentación de clientes, un buen silhouette indica grupos bien definidos para campañas personalizadas (conecta con Mazda y con el ejercicio del notebook).

<a id="filmina-38"></a>

### 5.4 Estrategias de validación (Slide 38/43 — "Estrategias de Validación")

- **Hold-out**: dividir el dataset en entrenamiento y prueba. Rápido pero puede ser inestable si los datos son pocos.
- **K-Fold**: dividir el dataset en *k* partes; entrenar *k* veces, cada vez con un fold distinto como prueba y el resto para entrenamiento; promediar las métricas. Ventaja: reduce la varianza en la estimación del desempeño.
- **Time-Split**: para datos secuenciales o series temporales — se respeta el orden temporal, entrenando con datos anteriores y probando con datos posteriores. *Ejemplo del PDF*: en predicción de demanda diaria, no se debe usar datos futuros para entrenar.

<a id="filmina-39"></a>

### 5.5 Trade-offs en la selección de métricas y validación (Slide 39/43 — "Trade-offs en la Selección de Métricas")

- **Complejidad vs. interpretabilidad**: métricas simples como accuracy son fáciles de entender, pero pueden ser engañosas en datasets desbalanceados.
- **Tiempo de cómputo**: k-fold es más preciso pero consume más recursos.
- **Naturaleza del problema**: en problemas críticos (fraude, salud), priorizar recall o precisión según el impacto.
- **Datos disponibles**: en series temporales, usar time-split para evitar fugas de información.

> **Reflexión del PDF**: no existe una métrica o estrategia universal; la elección debe alinearse con el contexto y los objetivos del negocio.

**Nota sobre la Slide 40/43 ("Aplicación Práctica: Métricas en la Industria")**: esta filmina no tiene una sub-sección propia acá porque su contenido ya está repartido entre las anteriores — son 3 mini-casos que resumen lo ya visto: **Detección de Fraude** (priorizar recall + validación temporal, ver 5.1), **Segmentación de Clientes** (métricas de clustering, ver 5.3) y **Optimización Logística** (métricas de regresión + K-Fold, ver 5.2 y 5.4). Sirve como cierre/repaso más que como concepto nuevo.

<a id="filmina-41"></a>

### 5.6 Código de práctica (el que trae el PDF completo) (Slide 41/43 — "Para Practicar: Métricas y Validación")

El PDF incluye un script de práctica de 5 partes que vale la pena mostrar o correr en vivo si hay tiempo:

1. **Clasificación**: `make_classification` + `LogisticRegression` + accuracy/precision/recall/F1/AUC-ROC.
2. **Regresión**: `make_regression` + `LinearRegression` + MAE/RMSE/R².
3. **Clustering**: `make_blobs` + `KMeans` + Silhouette/Davies-Bouldin/Inercia + scatter plot.
4. **K-Fold**: `KFold(n_splits=5, shuffle=True)` sobre el dataset de clasificación, promediando accuracy por fold.
5. **Time-Split**: `TimeSeriesSplit(n_splits=5)` sobre una serie temporal simulada (`np.sin(...)` + ruido), midiendo MSE por fold.

```python
# Fragmento representativo (K-Fold)
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_i, test_i in kf.split(Xc):
    model = LogisticRegression()
    model.fit(Xc[train_i], yc[train_i])
    pred = model.predict(Xc[test_i])
    scores.append(accuracy_score(yc[test_i], pred))
print("Promedio K-fold:", np.mean(scores))
```

**Para remarcar en clase**: aunque el ejercicio del día (natalidad + K-Means) solo necesita las métricas de clustering (5.3), es importante que los alumnos vean el panorama completo de las 5 partes — es exactamente el código que van a necesitar apenas trabajen con un problema supervisado.

<a id="bloque-3-metricas"></a>

### Bloque 3 — Métricas y Estrategias de Validación (20 min)

**El dilema a plantear en clase**: elegimos K=3 "porque sí" en el Bloque 2. ¿Cómo justificarlo matemáticamente? Acá no sirven accuracy/precision (no hay etiquetas verdaderas) — se necesitan métricas específicas de clustering:

```python
inercias, siluetas = [], []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inercias.append(km.inertia_)
    siluetas.append(silhouette_score(X, labels))
```

**Línea por línea**:

- **`inercias, siluetas = [], []`** — crea dos listas vacías donde se va a ir guardando, para cada valor de K probado, su inercia y su Silhouette Score respectivamente. Son los dos insumos que después se grafican para "ver" el codo y elegir el mejor K.
- **`for k in range(2, 7):`** — prueba K desde 2 hasta 6 (`range(2, 7)` excluye el 7). **¿Por qué arranca en 2 y no en 1?** Porque con K=1 (todo en un solo cluster) el Silhouette Score no se puede calcular — la métrica necesita al menos 2 grupos para medir separación entre ellos. **¿Por qué llega hasta 6 y no más?** Es un techo razonable para un dataset de 24 provincias: pedirle, por ejemplo, K=20 dejaría casi un cluster por provincia, sin ningún poder de generalización — el rango 2-6 cubre las opciones sensatas para segmentar en "perfiles" con sentido de negocio.
- **`km = KMeans(n_clusters=k, random_state=42, n_init=10)`** — en cada vuelta del loop crea un modelo K-Means nuevo, con el K de esa iteración. `random_state=42` y `n_init=10` cumplen la misma función que en el Bloque 2 (semilla fija para reproducibilidad, y 10 corridas con inicializaciones distintas para evitar un mal mínimo local) — se repiten acá porque cada valor de K necesita su propio modelo entrenado desde cero.
- **`labels = km.fit_predict(X)`** — entrena el modelo con los datos escalados (`X`, el mismo array que devolvió el pipeline del Bloque 1) y devuelve, para cada provincia, el número de cluster al que quedó asignada.
- **`inercias.append(km.inertia_)`** — `km.inertia_` es un atributo que scikit-learn calcula automáticamente después de entrenar: la suma de las distancias al cuadrado de cada punto a su centroide. Se guarda en la lista para poder graficarla después contra cada K (el "método del codo").
- **`siluetas.append(silhouette_score(X, labels))`** — calcula el Silhouette Score de esa partición en particular (comparando los mismos datos `X` contra las etiquetas de cluster que acaba de asignar `labels`) y lo guarda en la otra lista.

- **Método del codo (inercia)**: a más clusters, la inercia siempre baja — se busca el punto donde agregar un cluster más deja de aportar una mejora significativa (el "codo" del gráfico).
- **Silhouette Score**: para cada K probado, mide qué tan bien separados y cohesionados quedan los grupos (rango -1 a 1, más alto es mejor).

Corresponde directamente al Módulo 5.3 (métricas de clustering), que se acaba de repasar arriba. La celda de teoría agregada en el notebook también tiende el puente hacia las métricas de clasificación/regresión y las estrategias hold-out/K-Fold/Time-Split (Módulo 5 completo), aclarando que hoy no hacen falta porque el problema es no supervisado, pero van a ser necesarias en cuanto el proyecto pase a predecir un valor.

---

## 7. El ejercicio práctico del notebook, explicado en profundidad

El notebook `Clase_7_Fundamentos_de_Ciencia_de_Datos_1_.ipynb` construye, sobre el dataset real **`tasa-natalidad-deis-2000-2024.csv`** (Ministerio de Salud, vía datos.salud.gob.ar), un ejercicio completo de segmentación de provincias argentinas según su evolución de natalidad 2000–2024. Es, en esencia, **el caso Mazda hecho en vivo con datos públicos** en lugar de datos de clientes.

### Bloque 0 — Repaso de la Semana 6

Ya desarrollado en detalle, con teoría profunda y el paso a paso de cada línea de código, al [principio de este documento](#bloque-0--un-ejemplito-para-repasar-4-conceptos-de-la-semana-6) — se puso ahí porque es literalmente lo primero que corre el notebook, antes de tocar pipelines o Machine Learning.

### Bloque 1 — Pipeline de Ingesta y Transformación (20 min)

Ya desarrollado en detalle, con teoría, código completo y explicación línea por línea, al [final del Módulo 1](#bloque-1-pipeline) — se puso ahí porque es el pipeline que cierra ese módulo de teoría, justo antes de pasar a los casos de estudio de Segmentación y Recomendaciones (Módulo 2).

### Bloque 2 — Supervisado vs. No Supervisado + K-Means (20 min)

Ya desarrollado en detalle, con teoría, tabla comparativa y código completo, al [final del Módulo 4](#bloque-2-kmeans) — se puso ahí porque es la aplicación en código de la misma distinción teórica (Supervisado vs. No Supervisado) que cierra ese módulo, justo antes del Break del Coder.

### Bloque 3 — Métricas y Estrategias de Validación (20 min)

Ya desarrollado en detalle, con teoría y código completo, al [final del Módulo 5](#bloque-3-metricas) — se puso ahí porque es la aplicación en código de las métricas y estrategias de validación que cierra ese módulo, justo antes de esta sección práctica.

### Bloque 4 — Casos de Estudio y "Recomendaciones" (15 min)

**Teoría y escenario real — de los datos a la acción (*Analytics to Action*)**: un cluster por sí solo no genera valor. El data scientist debe interpretar qué significa cada grupo. Al entender el perfil demográfico de cada cluster (ej. provincias con natalidad en caída libre vs. estables), el pipeline puede disparar de forma automática recomendaciones de políticas públicas o asignación presupuestaria para cada provincia.

**Cómo se relaciona esto con los 4 casos de éxito de la industria**: la función que se arma en esta celda —que traduce cada `cluster_id` en una recomendación de negocio— es exactamente el mismo patrón de cierre que comparten los 4 casos del Módulo 3:

- **Mazda** traduce cada cluster de clientes en una estrategia de marketing distinta (ver la tabla "Métrica Técnica → Indicador de Negocio" de esa teoría) — acá se traduce, del mismo modo, un cluster de provincias en una política pública.
- **Medplaya** traduce una predicción (¿va a cancelar?) en una acción concreta: overbooking controlado.
- **San Cristóbal** traduce una predicción (¿es fraude?) en una acción: investigar el caso con revisión manual y análisis de imágenes.
- **Amazon** traduce una similitud entre productos en una acción: mostrarle al usuario una recomendación puntual.

El principio común a los cuatro —y al ejercicio de hoy— es que **el modelo nunca es el final del pipeline**: el valor de negocio aparece recién cuando el resultado técnico (un cluster, una probabilidad, un score de similitud) se traduce en una decisión o acción concreta y automatizable.

```python
# =====================================================================
# SECCIÓN 4: Modelo Definitivo e Inferencia de Políticas
# =====================================================================

# Asumiendo que el análisis arrojó un K óptimo (ej. K=3)
k_final = 3
modelo_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
df_provincias['cluster_final'] = modelo_final.fit_predict(X)

# Analizamos el promedio de los años extremos para interpretar el comportamiento
perfil_clusters = df_provincias.groupby('cluster_final')[[2000, 2012, 2024]].mean()
print("\nComportamiento histórico promedio por grupo:")
print(perfil_clusters)

# Creamos una lógica automatizada de Recomendación
def sistema_recomendacion_politica(cluster_id):
    if cluster_id == 0:
        return "Alerta Demográfica: Reorientar presupuesto a salud de adultos mayores."
    elif cluster_id == 1:
        return "Prioridad Alta: Planificar construcción de nuevos jardines y escuelas primarias."
    else:
        return "Estable: Mantener subsidios existentes y monitorear tasas de control prenatal."

# Aplicamos la recomendación al dataset original
df_provincias['Accion_Recomendada'] = df_provincias['cluster_final'].apply(sistema_recomendacion_politica)

print("\nVista final del entregable para el Ministerio (Muestra):")
print(df_provincias[['cluster_final', 'Accion_Recomendada']].head(10))
```

**Línea por línea**:

- **`k_final = 3`** — fija el valor de K que salió "ganador" del análisis del Bloque 3 (método del codo + Silhouette Score). A diferencia del `KMeans(n_clusters=3, ...)` de prueba en Bloque 2, acá el 3 ya está **justificado**, no elegido "porque sí".
- **`modelo_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)`** — crea el modelo K-Means definitivo, con los mismos parámetros de reproducibilidad ya vistos en Bloque 2 (`random_state`, `n_init`), pero usando la variable `k_final` en vez de un número hardcodeado — si mañana cambiara el K óptimo, alcanza con cambiar esa única variable.
- **`df_provincias['cluster_final'] = modelo_final.fit_predict(X)`** — entrena el modelo con los datos escalados (`X`) y agrega el resultado directamente como una **columna nueva** en `df_provincias` (el DataFrame "humano", con nombres de provincia, no el array escalado) — así cada provincia queda con su número de cluster asociado, fácil de leer.
- **`perfil_clusters = df_provincias.groupby('cluster_final')[[2000, 2012, 2024]].mean()`** — agrupa las provincias por su cluster asignado y calcula el promedio de natalidad en 3 años puntuales (2000, 2012, 2024): el inicio, la mitad y el final de la serie. Es una forma rápida de "leer" qué caracteriza a cada grupo sin mirar los 25 años completos.
- **`print("\nComportamiento histórico promedio por grupo:")` / `print(perfil_clusters)`** — muestran esa tabla resumen en pantalla: es el momento en que un humano (el docente, el alumno, el Ministerio) le pone un significado a cada número de cluster (0, 1, 2) antes de seguir.
- **`def sistema_recomendacion_politica(cluster_id):`** — función que traduce un número de cluster en una recomendación de política pública en texto plano: el paso de "resultado técnico" a "acción de negocio" del que habla la teoría.
- **`if cluster_id == 0: ...` / `elif cluster_id == 1: ...` / `else: ...`** — tres ramas simples, una por cluster. Qué mensaje le corresponde a cada número **no sale solo del código**: se decide mirando la tabla `perfil_clusters` de arriba (por ejemplo, el cluster con natalidad más baja y con mayor caída se asocia a "Alerta Demográfica"). Es una decisión de negocio, no un cálculo automático.
- **`df_provincias['Accion_Recomendada'] = df_provincias['cluster_final'].apply(sistema_recomendacion_politica)`** — `.apply(...)` ejecuta la función anterior **una vez por cada fila** de la columna `cluster_final`, y guarda el resultado (el texto de la recomendación) en una columna nueva.
- **`print(df_provincias[['cluster_final', 'Accion_Recomendada']].head(10))`** — muestra las primeras 10 provincias con su cluster y su recomendación: el "entregable final" que un Ministerio podría leer directamente, sin necesitar entender nada de K-Means por debajo.

**Para leer el resultado con la clase**: la tabla `perfil_clusters` (promedio de natalidad en 2000, 2012 y 2024 por cluster) permite nombrar cada grupo con criterio propio antes de mostrar las recomendaciones — es un buen momento para pedirle a los alumnos que interpreten los tres clusters *antes* de revelar las etiquetas que puso la función.

---

## Preguntas frecuentes y errores típicos a anticipar

- **"¿Por qué transponemos el DataFrame en el pipeline?"** → porque necesitamos que las provincias sean las filas (instancias) y los años las columnas (features) para que K-Means las segmente correctamente. Ver Bloque 1.
- **"¿Por qué hay que escalar antes de K-Means?"** → porque el algoritmo mide distancias euclidianas; sin escalar, una columna con valores más grandes domina el resultado solo por su magnitud, no por su relevancia real (Módulo 3.4 / Bloque 0-1).
- **"¿Por qué no usamos accuracy para evaluar los clusters?"** → porque no hay etiquetas verdaderas contra las cuales comparar; accuracy es una métrica de clasificación supervisada. Se usan Inercia y Silhouette Score en su lugar (Módulo 5.3 / Bloque 3).
- **"¿Cuándo usar oversampling y cuándo undersampling?"** → depende de cuántos datos hay disponibles: con pocos datos de la clase minoritaria conviene oversampling (SMOTE); con abundancia de datos de la clase mayoritaria, undersampling puede ser más eficiente sin perder información relevante (Módulo 3.1, San Cristóbal).
- **"¿Por qué el recall importa más que la precisión en fraude/cancelaciones?"** → porque el costo de no detectar un caso positivo real (un fraude que pasa, una cancelación no anticipada) suele ser mayor que el costo de una falsa alarma (Módulo 3.1 y 3.2).
- **"¿Por qué no se puede usar K-Fold normal en series temporales?"** → porque mezclaría datos del futuro en el entrenamiento (data leakage); hace falta Time-Split, que respeta el orden cronológico (Módulo 5.4).

---

## Material de la clase

| Archivo | Qué es |
|---|---|
| `Clase 07.pdf` | Material teórico oficial de la unidad (fuente original de esta guía). |
| `Semana 7.html` | Diapositivas para proyectar en clase (44 filminas). Navegación con flechas del teclado o los botones inferiores. |
| `Clase_7_Fundamentos_de_Ciencia_de_Datos_1_.ipynb` | Notebook con el ejercicio práctico completo (repaso + pipeline + K-Means + validación + recomendaciones) sobre datos reales de natalidad. |
| `tasa-natalidad-deis-2000-2024.csv` | Dataset real usado en el notebook (Ministerio de Salud, DEIS). |
| `material/` | Carpeta con recursos adicionales: teoría de aprendizaje supervisado/no supervisado, paso a paso, PPTs, notebooks de referencia adicionales. |

**Cómo usar esta guía durante la clase**: los Módulos 1 a 5 siguen el mismo orden que las diapositivas; la sección 7 sigue el orden del notebook. Podés alternar entre proyectar la filmina/notebook correspondiente y volver acá si necesitás un dato de contexto, una analogía o una pregunta frecuente para anticipar.

---

## Conclusión final de la clase

**El recorrido completo, en una sola línea**: partimos de un problema de negocio real (el Ministerio de Salud no sabe cómo agrupar provincias por su evolución de natalidad), lo resolvimos con un pipeline reproducible de punta a punta, y cerramos traduciendo el resultado técnico en una recomendación de política pública — exactamente el mismo patrón que Mazda, Medplaya, San Cristóbal y Amazon aplican en la industria, cada uno en su dominio.

**Los tres hilos que atravesaron toda la clase** (vale la pena nombrarlos explícitamente al cerrar, porque son los que quedan cuando se olviden los detalles de sintaxis):

1. **Reproducibilidad** (Módulo 1 → Bloque 1): un pipeline solo tiene valor real si otra persona (o vos mismo, meses después) puede correrlo de nuevo y obtener el mismo resultado. Cada `random_state=42` de la clase es una instancia concreta de esta idea.
2. **Supervisado vs. No Supervisado** (Módulo 4 → Bloque 2): la pregunta que separa ambos mundos es siempre la misma — *¿el dataset ya trae la respuesta correcta, o el algoritmo tiene que descubrirla solo?* Hoy trabajamos del lado no supervisado porque nadie le puso una etiqueta de "provincia con natalidad decreciente" a los datos.
3. **Analytics to Action** (Módulo 2/3 → Bloque 4): ni un cluster, ni una predicción, ni un score de similitud generan valor por sí solos. El pipeline termina cuando ese resultado técnico se convierte en una decisión o acción concreta — una recomendación de política, un overbooking, una alerta de fraude, una sugerencia al usuario.

**Lo que un alumno debería poder responder al salir de esta clase** (retomando los 4 objetivos de la Introducción): diseñar un pipeline end-to-end reproducible, elegir entre segmentación y recomendación según el problema de negocio, decidir si un caso es supervisado o no supervisado, y elegir la métrica/estrategia de validación correcta según el tipo de problema. El ejercicio de natalidad tocó los cuatro puntos en una sola corrida de principio a fin.

**Cierre sugerido para el pizarrón**: *"Hoy no aprendimos un algoritmo — K-Means es solo una herramienta más. Aprendimos un método: cómo llevar un dataset crudo, real y públicamente disponible hasta una recomendación accionable, de forma que cualquier colega pueda reproducir cada paso del camino."*
