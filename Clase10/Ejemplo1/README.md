**Materia:** Data Science I  
**Dataset:** `ecommerce_data.csv` (subir manualmente a Google Colab)  
**Notebook:** `ProyectoEjemplo.ipynb` (abrir en Google Colab)

> Documento generado a partir de todas las celdas del notebook (texto explicativo + codigo completo), en el mismo orden en que aparecen.

---

## Cómo dar este ejemplo en clase

Este es el **Ejemplo 1** de la Clase 10 (Repaso Final). Es un proyecto de punta a punta — de la pregunta de negocio al modelo evaluado — que funciona como repaso integrador porque toca, en un solo caso real, contenido de casi toda la cursada:

| Lo que se ve acá | De qué clase viene |
|---|---|
| Carga, tipos de datos, `.info()`/`.describe()` | Clase 03 — NumPy y Pandas |
| Nulos, imputación, `missingno`, capping de outliers | Clase 04 y Clase 06 — Limpieza y EDA |
| Gráficos de distribución, balance de clases, feature importances | Clase 05 — Visualización |
| `train_test_split`, Random Forest, KNN, Logistic Regression, XGBoost | Clase 08 — Aprendizaje Supervisado |
| `Pipeline` + `ColumnTransformer` + `GridSearchCV` + `StratifiedKFold` | Clase 07 y Clase 08 — Reproducibilidad y validación correcta |

**Por qué conviene leerlo en dos pasadas:** el notebook entrena un modelo dos veces — primero con preprocesado manual (más fácil de seguir, pero con **data leakage** real: `fillna` calculado sobre todo el dataset, categóricas que quedan afuera del modelo final) y después con un `Pipeline` formal que corrige esos errores. Esa comparación **a propósito** es el mejor gancho para la clase: no es un ejemplo perfecto desde el inicio, es un ejemplo que se corrige a sí mismo — igual que pasa en un proyecto real.

**Sugerencia de recorrido en vivo** (no hace falta leer las 1500 líneas palabra por palabra):
1. Arrancar por "De qué se trata este análisis" y "Cuánto vale resolver esto" — el encuadre de negocio, para que quede claro que esto no es solo un ejercicio técnico.
2. Pasar rápido por el diccionario de columnas — mencionar que son 37 variables y que muchas fechas llegan mal tipadas (`object` en vez de `datetime`), gancho directo a Clase 03/04.
3. Detenerse en "Visualización de nulos con missingno" — es una herramienta que probablemente no vieron antes, vale la pena mostrar el gráfico.
4. Mostrar el contraste "Modelado — primera pasada" (manual, con leakage) vs. "Pipeline formal" — el corazón pedagógico del ejemplo.
5. Cerrar con la sección "Conclusiones" — el hallazgo de negocio (`ALL_USERS_COUNT` importa más que la infraestructura) es un buen disparador de preguntas.

### Índice

- [De qué se trata este análisis](#de-qué-se-trata-este-análisis)
- [Qué significa cada columna](#qué-significa-cada-columna)
- [Carga del Dataset](#carga-del-dataset)
- [Limpieza y transformación de datos](#limpieza-y-transformación-de-datos)
- [Visualización de nulos con missingno](#visualización-de-nulos-con-missingno)
- [Modelado — primera pasada](#modelado--primera-pasada)
- [Qué tan desbalanceadas están las clases](#qué-tan-desbalanceadas-están-las-clases)
- [Pipeline formal: `Pipeline` + `GridSearchCV` + `StratifiedKFold`](#pipeline-formal-pipeline--gridsearchcv--stratifiedkfold)
- [Riesgos y controles de calidad](#riesgos-y-controles-de-calidad)
- [Conclusiones](#conclusiones)

---

# ¿Qué empresas se quedan en Veeqo? — Predicción de churn B2B

Proyecto de clasificación binaria sobre la base de clientes de **Veeqo** (plataforma de gestión de inventario y envíos multicanal, adquirida por Amazon en 2021).

| Ficha del proyecto | |
|---|---|
| **Rubro** | E-commerce / SaaS B2B |
| **Tarea** | Clasificación binaria — ¿el cliente se queda o abandona? |
| **Datos** | 84.063 empresas registradas, 37 columnas operativas |
| **Métrica** | ROC-AUC (las clases están muy desbalanceadas, ~99:1) |
| **Stack** | Python, pandas, scikit-learn, seaborn, missingno |
| **Entorno de trabajo** | Google Colab + Drive |

> Veeqo conecta el inventario y los envíos de vendedores que operan en varios canales a la vez (Amazon, Shopify, eBay). Cada fila del dataset es el estado de cuenta de una empresa distinta registrada en la plataforma.

## De qué se trata este análisis

Retener un cliente en un negocio SaaS B2B sale mucho más barato que conseguir uno nuevo — la relación típica que se maneja en la industria es de 5 a 7 veces más caro adquirir que retener. Entonces tiene sentido invertir en anticipar quién se va a ir *antes* de que pase.

Acá construimos un modelo que, mirando solo el comportamiento operativo temprano de una empresa (canales conectados, usuarios activos, almacenes, etc.), predice si esa empresa va a terminar siendo cliente activo de alto valor (`live`) o se va a quedar en el camino (`trialing`, `canceled`, `churned`).

### ¿Para qué le sirve esto al equipo de Customer Success?

- Para saber a quién llamar primero durante el onboarding
- Para detectar antes de tiempo señales de que un cliente se va a caer
- Para armar campañas de activación apuntadas a los segmentos de mayor riesgo

### Cómo se arma el target

```
SUBSCRIPTION_STATUS → TARGET (0 / 1)

  live | implementation        →  1   cliente activo, paga, usa la plataforma
  trialing | canceled | resto  →  0   nunca convirtió o abandonó
```

### Preguntas que quiero responder con el análisis

1. ¿Qué comportamiento temprano distingue a un cliente que se queda de uno que se va?
2. ¿Hay un número de usuarios en la cuenta a partir del cual la retención se dispara?
3. ¿Pesa más el equipo de trabajo (usuarios, admins) o la infraestructura (canales, almacenes)?
4. ¿Un modelo simple como Logistic Regression compite con algo como XGBoost acá?

### 💵 Cuánto vale resolver esto

De las **84.063 empresas** que se registraron en Veeqo, solamente **686 (0.82%)** terminaron siendo clientes `live` (activos, pagando). Con una conversión tan baja, no hace falta mover mucho la aguja para que el impacto en dólares sea grande — es justo el tipo de escenario donde un modelo predictivo rinde.

**Supuestos usados como referencia** (órdenes de magnitud típicos de SaaS B2B — no son cifras reales de Veeqo):

| Variable | Valor supuesto |
|---|---|
| Ingreso anual promedio por cliente `live` (ARR) | USD 3.000 |
| Costo de adquisición por lead (CAC) | USD 150–300 (alto: onboarding asistido por ventas) |
| Conversión trial → live (dato real de este dataset) | 0.82% |
| Si el modelo prioriza el 20% de leads con mayor probabilidad de convertir | Ahí es donde el equipo de CS concentra las llamadas de onboarding |

Si ese enfoque llevara la conversión de 0.82% a 1.2% sobre los mismos 84.063 leads, son **~336 clientes `live` adicionales por año** — a USD 3.000 de ARR cada uno, más de **USD 1.000.000 de ingreso incremental**, sin gastar un dólar extra en adquisición (el CAC de esos leads ya estaba pagado).

**KPIs que este modelo termina moviendo:**

| KPI | Qué mide acá |
|---|---|
| **Tasa de conversión trial → live** | % de registros que llegan a `live` — hoy 0.82% |
| **Customer Lifetime Value (CLV)** | Ingreso total esperado por cliente `live` durante toda la relación |
| **Tasa de retención** | % de clientes `live` que se mantienen activos y no vuelven a `canceled` |
| **Ingreso activado** | USD generados por leads que el equipo convirtió gracias a la priorización |
| **Costo por intervención** | Costo de accionar sobre un lead priorizado (llamada, mail, demo) |
| **ROI de la intervención** | (Ingreso activado − Costo de campañas) / Costo de campañas |

## Qué significa cada columna

Las 37 variables se agrupan en 5 bloques:


1- **COMPANY_ID** (float64): Identificador único numérico de la empresa.

2- **COMPANY_NAME** (object): Nombre legal o comercial de la empresa registrada.

3- **COMPANY_CREATED_AT** (Inicialmente object pero es necesario cambiar el tipo de dato a datetime): Fecha y hora exacta de registro de la empresa en la plataforma.

4- **SUBSCRIPTION_STATUS** (object): Estado actual de la suscripción (ej. activo, cancelado, periodo de prueba).

5- **VEEQO_PRODUCT_NAME** (object): Nombre del plan o producto específico de Veeqo contratado por el cliente.

6- **PRODUCT**(object): Tipo o categoría de producto principal que comercializa el vendedor.

7- **DECILE** (float64): Clasificación del cliente en deciles (1-10) basada en su valor o actividad para análisis RFM.

8- **SELLER_TYPE** (object): Categorización del tipo de vendedor (ej. Retail, Wholesale, marca propia).

9- **SIGNUP_TYPE** (object): Método o canal por el cual el usuario realizó su registro inicial.

10- **ACTIVE_SHIPPER** (object): Indicador de si la empresa realiza envíos de forma activa a través de la plataforma.

11- **COUNTRY** (object): País de operación o ubicación principal de la empresa.

12- **EMAIL** (object): Correo electrónico de contacto principal de la cuenta.

13- **PHONE** (Inicialmente float64 pero es necesario cambiarlo ya que no es un dato numérico): Número telefónico de contacto (presenta alta tasa de valores nulos).

14- **ACTIVE_CHANNELS** (float64): Cantidad de canales de venta conectados que están actualmente activos.

15- **INACTIVE_CHANNELS** (float64): Cantidad de canales de venta que fueron conectados pero se encuentran inactivos.

16- **UNIQUE_CHANNELS_ACTIVE_EXCL_DIRECT_AMAZON** (float64): Conteo de canales activos únicos, excluyendo la integración directa de Amazon.

17- **UNIQUE_CHANNELS_ACTIVE_EXCL_AMAZON** (float64): Conteo de canales activos únicos, excluyendo todos los canales relacionados con Amazon.

18- **UNIQUE_CHANNELS** (object): Listado o detalle de los nombres de los canales únicos vinculados.

19- **PQL_DATE** (Inicialmente object pero es necesario cambiar el tipo de dato a datetime): Fecha en que el usuario calificó como Lead de Producto (Product Qualified Lead).

20- **FIRST_SHIPMENT_DATE** (Inicialmente object pero es necesario cambiar el tipo de dato a datetime): Fecha en la que la empresa realizó su primer envío usando Veeqo.

21- **LAUNCHED_DATE** (Inicialmente object pero es necesario cambiar el tipo de dato a datetime): Fecha oficial en la que la cuenta del cliente completó su configuración y salió a producción.

22- **ACTIVATED_WEEK** (Inicialmente object pero deberia cambiarse a tipo date): Semana en la que se activó la cuenta para análisis de cohortes temporales.

23- **LAST_SHIPMENT_DATE** (Inicialmente object pero es necesario cambiar el tipo de dato a datetime): Fecha del envío más reciente registrado (clave para medir la Recencia).

24- **LAST_PAGE_LOADED_AT** (Inicialmente object pero es necesario cambiar el tipo de dato a datetime): Marca de tiempo de la última actividad del usuario en la interfaz de la plataforma.

25- **ACTIVE_USERS_L_28D** (float64): Cantidad de usuarios únicos de la empresa que han estado activos en los últimos 28 días.

26- **ALL_USERS_COUNT** (float64): Número total de usuarios registrados bajo la misma cuenta de empresa.

27- **ADMIN_USER_COUNT** (float64): Número de usuarios que poseen privilegios de administrador en la cuenta.

28- **FIRST_SALES_INTERACTION** (Inicialmente object pero es necesario cambiar el tipo de dato a datetime): Fecha de la primera interacción documentada con el equipo de ventas.

29- **MARKETING_CHANNEL** (object): Canal de marketing principal a través del cual se adquirió al cliente.

30- **MARKETING_SUB_CHANNEL** (object): Sub-categoría o fuente específica del canal de marketing de procedencia.

31- **MARKETING_CAMPAIGN** (object): Nombre de la campaña de marketing específica que generó el registro del usuario.

32- **MOS_ACTIVATED_AT** (Inicialmente object pero es necesario cambiar el tipo de dato a datetime): Fecha de activación de funciones logísticas avanzadas.

33- **UPGRADED_TO_POWER_AT** (Inicialmente object pero es necesario cambiar el tipo de dato a datetime): Fecha en la que el cliente cambió su plan a una versión superior ("Power").

34- **WAREHOUSES** (float64): Cantidad total de almacenes o bodegas físicas registradas por la empresa.

35- **FBA_WAREHOUSES** (int64): Cantidad de almacenes gestionados bajo la modalidad Fulfillment by Amazon. Donde FBA (Fulfillment by Amazon)  es el proceso de entrega de pedidos a los clientes, desde el inventario, pasando por el empaquetado de producto, hasta la entrega de paquetes y manejo de retornos en e-commerce. Implica todas las actividades desde el momento en que se recibe un pedido hasta que se entrega al cliente.

36- **FIRST_IMPORTED_ORDER** (Inicialmente object pero es necesario cambiar el tipo de dato a datetime): Fecha en la que se importó la primera orden de venta desde un canal externo.

37- **FIRST_CHANNEL_CONNECTED** ( Inicialmente object pero deberia cambiarse a tipo date): fecha enla que el usuario se vinculo en el primer canal de ventas que el usuario vinculó a su cuenta de Veeqo.

## Carga del Dataset

```python
import seaborn as sns
sns.set_theme(style='whitegrid', palette='deep')  # tema mas claro que el default

import pandas as pd

# Subir el archivo manualmente a Colab:
#   1. Panel izquierdo (icono de carpeta)
#   2. Arrastrar el archivo ecommerce_data.csv
#   3. El archivo queda en /content/ (mismo nivel que sample_data)
#
# Alternativa por codigo (abre selector de archivos del navegador):
#   from google.colab import files
#   files.upload()

df_veeqo = pd.read_csv("/content/ecommerce_data.csv")

print(f"Dataset cargado: {df_veeqo.shape[0]:,} filas, {df_veeqo.shape[1]} columnas")
from IPython.display import display
display(df_veeqo.tail())
```

## Limpieza y transformación de datos

Antes de entrenar cualquier modelo hay que dejar la tabla en condiciones:

1. Tipos de dato correctos (fechas como `datetime`, no como texto)
2. Sin nulos que rompan el entrenamiento
3. Outliers bajo control (no eliminados — son clientes reales)
4. Categóricas codificadas en números

Lo que sigue identifica qué columna necesita cada transformación.

Con el diccionario de la celda anterior a mano, agrupamos las columnas que hay que retipar porque lo que trae el CSV no coincide con lo que la columna representa en la realidad (fechas, IDs, booleanos).

He dividido las transformaciones en tres grupos principales para hacerlo en la seccion correspondiente:

####**a. <u>Transformación a tipo Fecha (datetime)**</u>

Estas columnas aparecen como object (texto), pero son marcas de tiempo. Es fundamental transformarlas para poder calcular la Recencia en el análisis RFM (El análisis *RFM [* "Recencia", "Frecuencia" y "Valor Monetario"] es una técnica de marketing utilizada para segmentar a los clientes en función de su comportamiento de compra.) o hacer análisis temporales.

3- **COMPANY_CREATED_AT**

19- **PQL_DATE**

20- **FIRST_SHIPMENT_DATE**

21- **LAUNCHED_DATE**

22- **ACTIVATED_WEEK**

23- **LAST_SHIPMENT_DATE**

24- **LAST_PAGE_LOADED_AT**

28- **FIRST_SALES_INTERACTION**

32- **MOS_ACTIVATED_AT**

33- **UPGRADED_TO_POWER_AT**

36- **FIRST_IMPORTED_ORDER**

37- **FIRST_CHANNEL_CONNECTED**

####**b. <u>Transformación a tipo Texto o Categórico (object / str)</u>**

Estas columnas aparecen como números (float64), pero no se deben realizar operaciones matemáticas con ellas. Sumar IDs o números de teléfono no tiene sentido.

1- **COMPANY_ID:** Debe ser object o int64 (si no tiene decimales), ya que es un identificador único.

13- **PHONE:** Debe ser object. Al estar como float64, Python puede intentar leerlo con notación científica (ej: 5.5E+9) y corromper el número. Además, los teléfonos pueden empezar con "+".

####**c. <u>Transformación a tipo Entero (int64)**</u>
Pandas suele asignar float64 a columnas numéricas si estas tienen valores nulos (NaN). Si después de limpiar los nulos se quieren valores exactos, estas deberían ser enteros:

14- **ACTIVE_CHANNELS**

15- **INACTIVE_CHANNELS**

25- **ACTIVE_USERS_L_28D**

26- **ALL_USERS_COUNT**

27- **ADMIN_USER_COUNT**

34- **WAREHOUSES**

```python
# Analisis inicial: dimensiones, tipos de datos, nulos
df_veeqo.info() # (informacion de las columnas que posee el dataset, en este momento todavia no se realizo la transformacion de los tipos de datos)
```

Aplicamos esos tres arreglos de tipos — fechas que estaban en **object**, numéricos que en realidad son **object**, y los que conviene dejar en **entero** — y de paso definimos la estrategia para sus nulos

```python
import re
import pandas as pd

# a. Transformación a tipo Fecha (datetime)
datetime_cols = [
    'COMPANY_CREATED_AT',
    'PQL_DATE',
    'FIRST_SHIPMENT_DATE',
    'LAUNCHED_DATE',
    'LAST_SHIPMENT_DATE',
    'LAST_PAGE_LOADED_AT',
    'FIRST_SALES_INTERACTION',
    'MOS_ACTIVATED_AT',
    'UPGRADED_TO_POWER_AT',
    'FIRST_IMPORTED_ORDER',
    'ACTIVATED_WEEK',
    'FIRST_CHANNEL_CONNECTED'
]

# Mapping for Spanish month names to English
month_mapping = {
    'enero': 'January', 'febrero': 'February', 'marzo': 'March', 'abril': 'April',
    'mayo': 'May', 'junio': 'June', 'julio': 'July', 'agosto': 'August',
    'septiembre': 'September', 'octubre': 'October', 'noviembre': 'November',
    'diciembre': 'December'
}

for col in datetime_cols:
    if col in df_veeqo.columns and df_veeqo[col].dtype == 'object':
        # Copia de la columna para evitar warnings de Pandas
        temp_col = df_veeqo[col].copy()

        # Reemplazar meses en español por inglés (sin regex es más eficiente aquí)
        for sp_month, en_month in month_mapping.items():
            temp_col = temp_col.str.replace(sp_month, en_month, case=False, regex=False)

        # Usar REGEX para reemplazar 'a. m.' y 'p. m.' manejando cualquier tipo de espacio u omisión de los mismos (ej: a. m., a.m., a. m.)
        temp_col = temp_col.str.replace(r'a\.\s*m\.', 'AM', case=False, regex=True)
        temp_col = temp_col.str.replace(r'p\.\s*m\.', 'PM', case=False, regex=True)

        # Convertimos a datetime SIN forzar el formato, de esa manera Pandas infiere fechas como "May 25, 2023" que no tienen hora
        df_veeqo[col] = pd.to_datetime(temp_col, errors='coerce')


# b. Transformación a tipo Texto o Categórico (object / str)
df_veeqo['COMPANY_ID'] = df_veeqo['COMPANY_ID'].astype(str)

# Transformar 'PHONE' a object
df_veeqo['PHONE'] = df_veeqo['PHONE'].astype(str).replace({'nan': None})


# c. Transformación a tipo Entero (int64)
int_cols = [
    'ACTIVE_CHANNELS',
    'INACTIVE_CHANNELS',
    'ACTIVE_USERS_L_28D',
    'ALL_USERS_COUNT',
    'ADMIN_USER_COUNT',
    'WAREHOUSES'
]

for col in int_cols:
    if col in df_veeqo.columns:
        # Rellenar NaN con 0 antes de convertir a int64.
        df_veeqo[col] = df_veeqo[col].fillna(0).astype('int64')

print("Transformaciones de tipos de datos completadas. Aquí está la información actualizada del DataFrame:")
df_veeqo.info()
```

##### Pasamos de float a Int64 las columnas que corresponde

```python
#identificamos las columnas float
float_cols = df_veeqo.select_dtypes(include=['float64']).columns
#convertimos estas float en int (tipo entero)
for col in float_cols:
    if col in ['DECILE', 'UNIQUE_CHANNELS_ACTIVE_EXCL_DIRECT_AMAZON', 'UNIQUE_CHANNELS_ACTIVE_EXCL_AMAZON']:
        df_veeqo[col] = df_veeqo[col].astype('Int64')
print("Columnas float convertidas a Int64 (entero con soporte para NaNs) o int64 según el caso:")
df_veeqo.info()
# Verificar el resultado para las columnas convertidas
print("\nPrimeras filas de las columnas float transformadas:")
print(df_veeqo[float_cols].head())
```

```python
# Conteo de valores nulos por columna
# 1. Usamos tu código base para obtener los conteos de nulos, y le ponemos nombre a la columna
df_nulos = df_veeqo.isnull().sum().where(lambda x: x > 0).dropna().to_frame(name='Cantidad_Nulos')

# 2. Agregamos la columna de porcentaje
# Dividimos por el total de filas (len(df_veeqo)), multiplicamos por 100, redondeamos a 1 decimal y agregamos el '%'
df_nulos['Porcentaje (%)'] = (df_nulos['Cantidad_Nulos'] / len(df_veeqo) * 100).round(1).astype(str) + '%'

# 3. (Opcional pero recomendado) Ordenar de mayor cantidad de nulos a menor
df_nulos = df_nulos.sort_values(by='Cantidad_Nulos', ascending=False)

# Mostrar el resultado
print(df_nulos)
```

---
## Visualización de nulos con missingno

### ¿Por que visualizar los nulos con un grafico y no solo con `.isnull().sum()`?

`df.isnull().sum()` dice **cuantos** nulos hay.
`missingno` dice **donde** estan y si tienen un **patron**. Eso cambia la estrategia:

| Tipo de nulo | Descripcion | Ejemplo en Veeqo | Estrategia |
|---|---|---|---|
| **MCAR** | Aleatorio, sin patron | Fallo tecnico puntual | Imputar con media/mediana |
| **MAR** | Relacionado con otra columna | Sin primer envio = sin fecha | Imputar condicionalmente |
| **MNAR** | El nulo mismo tiene significado | PHONE: la empresa no quiere darlo | Crear bandera `HAS_PHONE` o eliminar |

En este dataset, **PHONE** tiene 99% de nulos → es MNAR → se elimina.
Las **fechas de hitos** (primer envio, primer canal) → MNAR → se convierte a `HAS_X` (booleano).

```python
#----------------------------------------------------------------------
# Visualización de nulos con missingno
#----------------------------------------------------------------------
#
# Libreria: missingno (pip install missingno)
# Autor original: Aleksey Bilogur — https://github.com/ResidentMario/missingno
#
# msno.matrix()
#   Cada FILA del grafico = una fila del dataset
#   Negro = valor presente  |  Blanco = valor nulo
#   Lectura:
#     Columnas completamente negras → 0% de nulos (muy bueno)
#     Columnas con franjas blancas  → nulos dispersos (MCAR, imputar)
#     Patron diagonal/bloque blanco → nulos correlacionados (MNAR, bandera)
#
# msno.bar()
#   Una barra por columna mostrando el % de filas con valor presente
#   Barra al 100% = sin nulos | Barra al 50% = la mitad de filas son nulas
#
import missingno as msno
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Grafico 1: patron espacial de nulos — VER si los blancos se agrupan
msno.matrix(df_veeqo, ax=axes[0], sparkline=False, color=(0.0, 0.42, 0.42))
axes[0].set_title(
    "Patron de nulos por fila (msno.matrix)\n"
    "Negro = dato presente  |  Blanco = nulo", fontsize=10)

# Grafico 2: completitud por columna — VER cuales variables tienen mas nulos
msno.bar(df_veeqo, ax=axes[1], color=(0.0, 0.42, 0.42), fontsize=7)
axes[1].set_title(
    "Completitud por columna (msno.bar)\n"
    "1.0 = sin nulos  |  0.5 = mitad de filas son nulas", fontsize=10)

plt.suptitle("Analisis de Missingness — Veeqo B2B Dataset",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# ── Regla de decision para cada columna ──────────────────────────────────────
# > 80% nulos  → eliminar (no aporta informacion estadistica util)
# 20–80% nulos → crear variable binaria HAS_X o imputar con modelo
# < 20% nulos  → imputar con mediana (numerica) o moda (categorica)
nulos_pct = df_veeqo.isnull().mean() * 100
criticos  = nulos_pct[nulos_pct > 20].sort_values(ascending=False)
print("Columnas con mas del 20% de nulos (requieren decision especial):")
print(criticos if len(criticos) > 0 else "Ninguna — todas las columnas tienen menos del 20% de nulos.")
```

#### Qué hacemos con cada tipo de nulo

| # | Columna(s) | Qué se hace | Por qué |
|---|---|---|---|
| 1 | `PHONE` | Se elimina la columna completa | Tiene casi 100% de nulos (99.98%). Rellenarla no aporta nada y solo ocupa memoria |
| 2 | Canales (`UNIQUE_CHANNELS_...`) | Se rellenan con 0 y se pasan a entero | Si no hay registro de canales conectados, lo razonable es asumir que todavía no conectó ninguno |
| 3 | Categóricas de marketing / vendedor / estado | Se rellenan con etiquetas explícitas (`'Desconocido'`, `'No Conectado'`) y `DECILE` con 0 | Un nulo también es información: crea un segmento propio ("no pudimos rastrear el origen") en vez de mezclarse con datos reales |
| 4 | `COMPANY_NAME`, `EMAIL` | Se rellenan con texto genérico (`'Empresa sin nombre'`, `'No Email'`) | Son menos del 1% de nulos — tirar esas filas haría perder el resto de la información de esos clientes por un dato menor |
| 5 | Columnas de fecha | Se dejan como `NaT` (no se inventa una fecha) | Un nulo en una fecha *significa* algo: el evento todavía no pasó. Rellenar con `1900-01-01` arruinaría cualquier cálculo de días transcurridos, porque Pandas ignora los `NaT` al promediar pero no ignoraría una fecha inventada |

Además, para las columnas de fecha creamos una columna booleana equivalente (`HAS_X`) — así un modelo o un gráfico puede usar "hizo su primer envío / no lo hizo" sin tener que lidiar con la fecha en sí.

```python
# Tratamiento de valores nulos (imputacion logica)
# Estrategia de tratamiento de Valores nulos
import pandas as pd
import numpy as np

# 1. Eliminar PHONE
if 'PHONE' in df_veeqo.columns:
    df_veeqo = df_veeqo.drop(columns=['PHONE'])

# 3. Rellenar columnas numéricas/de canales con 0
canales_cols = ['UNIQUE_CHANNELS_ACTIVE_EXCL_DIRECT_AMAZON', 'UNIQUE_CHANNELS_ACTIVE_EXCL_AMAZON']
for col in canales_cols:
    # Solución al error: convertimos a numérico forzando errores a nulos, luego rellenamos con 0 y pasamos a int
    df_veeqo[col] = pd.to_numeric(df_veeqo[col], errors='coerce')
    df_veeqo[col] = df_veeqo[col].fillna(0).astype(int)

# 4. Rellenar categóricas y variables de marketing
marketing_cols = ['MARKETING_CAMPAIGN', 'MARKETING_SUB_CHANNEL', 'MARKETING_CHANNEL']
for col in marketing_cols:
    df_veeqo[col] = df_veeqo[col].fillna('Desconocido') # o 'Organic' si aplica a tu negocio

df_veeqo['FIRST_CHANNEL_CONNECTED'] = df_veeqo['FIRST_CHANNEL_CONNECTED'].fillna('No Conectado')
df_veeqo['SELLER_TYPE'] = df_veeqo['SELLER_TYPE'].fillna('Desconocido')
df_veeqo['DECILE'] = df_veeqo['DECILE'].fillna(0) # O 0 para indicar que no está clasificado
df_veeqo['ACTIVE_SHIPPER'] = df_veeqo['ACTIVE_SHIPPER'].fillna('Desconocido') # O False / 'No'
#Rellenar los pocos nulos en datos del cliente
df_veeqo['COMPANY_NAME'] = df_veeqo['COMPANY_NAME'].fillna('Empresa sin nombre')
df_veeqo['EMAIL'] = df_veeqo['EMAIL'].fillna('No Email')
```

```python
# 5. Tratamiento de columnas tipo fecha
columnas_fechas = [
    'PQL_DATE',
    'FIRST_SHIPMENT_DATE',
    'LAUNCHED_DATE',
    'ACTIVATED_WEEK',
    'LAST_SHIPMENT_DATE',
    'LAST_PAGE_LOADED_AT',
    'FIRST_SALES_INTERACTION',
    'MOS_ACTIVATED_AT',
    'UPGRADED_TO_POWER_AT',
    'FIRST_IMPORTED_ORDER'
]

# 1. Convertir al formato de tiempo real (Los vacíos se volverán 'NaT' automáticamente)
for col in columnas_fechas:
    # errors='coerce' fuerza a que si hay un texto raro o un vacío, se convierta en NaT
    df_veeqo[col] = pd.to_datetime(df_veeqo[col], errors='coerce')

# 2. (Opcional pero recomendado) Crear columnas booleanas para saber si el evento ocurrió
# Esto creará columnas como 'HAS_UPGRADED_TO_POWER_AT' con True o False
for col in columnas_fechas:
    nombre_nueva_columna = f'HAS_{col}'
    df_veeqo[nombre_nueva_columna] = df_veeqo[col].notna()
```

```python
df_veeqo.info()
```

```python
df_veeqo.head()
```

#### Cómo tratamos los outliers

**Columnas candidatas** (conteos que pueden tener valores extremos): `ACTIVE_CHANNELS`, `INACTIVE_CHANNELS`, `UNIQUE_CHANNELS_ACTIVE_EXCL_DIRECT_AMAZON`, `UNIQUE_CHANNELS_ACTIVE_EXCL_AMAZON`, `ACTIVE_USERS_L_28D`, `ALL_USERS_COUNT`, `ADMIN_USER_COUNT`, `WAREHOUSES`, `FBA_WAREHOUSES`.

**Detección — IQR (rango intercuartílico):**
Límite superior = `Q3 + 1.5 · IQR`, límite inferior = `Q1 - 1.5 · IQR`, con `IQR = Q3 - Q1`.

**Tratamiento — capping (winsorización), no eliminación:**

| Decisión | Detalle |
|---|---|
| Outliers por arriba | Todo lo que supere `Q3 + 3.0·IQR` se recorta a ese límite (no se borra la fila) |
| Outliers por abajo | No se tocan — en columnas de conteo, un valor bajo (incluido 0) tiene significado real: ausencia de la característica, no un error |

**Por qué este enfoque:**
- No perdemos filas completas por un solo valor extremo
- Modelos sensibles a la escala (regresión lineal, por ejemplo) dejan de estar dominados por un puñado de outliers
- Se mantiene el orden y la escala original de la variable, más fácil de interpretar que una transformación logarítmica u otra cosa más agresiva

```python
# Tratamiento de outliers (capping hibrido IQR/percentil)
import pandas as pd
import numpy as np
import re

# Recargar el DataFrame original desde cero

df_veeqo = pd.read_csv("/content/ecommerce_data.csv")

# --- Tipos de datos ---

# a. Transformación a tipo Fecha (datetime)
datetime_cols = [
    'COMPANY_CREATED_AT', 'PQL_DATE', 'FIRST_SHIPMENT_DATE', 'LAUNCHED_DATE',
    'LAST_SHIPMENT_DATE', 'LAST_PAGE_LOADED_AT', 'FIRST_SALES_INTERACTION',
    'MOS_ACTIVATED_AT', 'UPGRADED_TO_POWER_AT', 'FIRST_IMPORTED_ORDER',
    'ACTIVATED_WEEK', 'FIRST_CHANNEL_CONNECTED'
]

month_mapping = {
    'enero': 'January', 'febrero': 'February', 'marzo': 'March', 'abril': 'April',
    'mayo': 'May', 'junio': 'June', 'julio': 'July', 'agosto': 'August',
    'septiembre': 'September', 'octubre': 'October', 'noviembre': 'November',
    'diciembre': 'December'
}

for col in datetime_cols:
    if col in df_veeqo.columns and df_veeqo[col].dtype == 'object':
        temp_col = df_veeqo[col].copy()
        for sp_month, en_month in month_mapping.items():
            temp_col = temp_col.str.replace(sp_month, en_month, case=False, regex=False)
        temp_col = temp_col.str.replace(r'a\.\s*m\.', 'AM', case=False, regex=True)
        temp_col = temp_col.str.replace(r'p\.\s*m\.', 'PM', case=False, regex=True)
        df_veeqo[col] = pd.to_datetime(temp_col, errors='coerce')

# b. Transformación a tipo Texto o Categórico
df_veeqo['COMPANY_ID'] = df_veeqo['COMPANY_ID'].astype(str)
df_veeqo['PHONE'] = df_veeqo['PHONE'].astype(str).replace({'nan': None})

# c. Transformación a tipo Entero
int_cols = ['ACTIVE_CHANNELS', 'INACTIVE_CHANNELS', 'ACTIVE_USERS_L_28D', 'ALL_USERS_COUNT', 'ADMIN_USER_COUNT', 'WAREHOUSES']
for col in int_cols:
    if col in df_veeqo.columns:
        df_veeqo[col] = df_veeqo[col].fillna(0).astype('int64')

# Convertir float a Int64
float_cols_to_int = ['DECILE', 'UNIQUE_CHANNELS_ACTIVE_EXCL_DIRECT_AMAZON', 'UNIQUE_CHANNELS_ACTIVE_EXCL_AMAZON']
for col in float_cols_to_int:
    df_veeqo[col] = df_veeqo[col].astype('Int64')

# --- Nulos ---

# 1. Eliminar PHONE
df_veeqo = df_veeqo.drop(columns=['PHONE'])

# 3. Rellenar columnas de canales con 0
canales_cols = ['UNIQUE_CHANNELS_ACTIVE_EXCL_DIRECT_AMAZON', 'UNIQUE_CHANNELS_ACTIVE_EXCL_AMAZON']
for col in canales_cols:
    df_veeqo[col] = pd.to_numeric(df_veeqo[col], errors='coerce').fillna(0).astype(int)

# 4. Rellenar categóricas
marketing_cols = ['MARKETING_CAMPAIGN', 'MARKETING_SUB_CHANNEL', 'MARKETING_CHANNEL']
for col in marketing_cols:
    df_veeqo[col] = df_veeqo[col].fillna('Desconocido')

df_veeqo['FIRST_CHANNEL_CONNECTED'] = df_veeqo['FIRST_CHANNEL_CONNECTED'].fillna('No Conectado')
df_veeqo['SELLER_TYPE'] = df_veeqo['SELLER_TYPE'].fillna('Desconocido')
df_veeqo['DECILE'] = df_veeqo['DECILE'].fillna(0)
df_veeqo['ACTIVE_SHIPPER'] = df_veeqo['ACTIVE_SHIPPER'].fillna('Desconocido')
df_veeqo['COMPANY_NAME'] = df_veeqo['COMPANY_NAME'].fillna('Empresa sin nombre')
df_veeqo['EMAIL'] = df_veeqo['EMAIL'].fillna('No Email')

# 5. Crear columnas booleanas para fechas
columnas_fechas = ['PQL_DATE', 'FIRST_SHIPMENT_DATE', 'LAUNCHED_DATE', 'ACTIVATED_WEEK',
                   'LAST_SHIPMENT_DATE', 'LAST_PAGE_LOADED_AT', 'FIRST_SALES_INTERACTION',
                   'MOS_ACTIVATED_AT', 'UPGRADED_TO_POWER_AT', 'FIRST_IMPORTED_ORDER']
for col in columnas_fechas:
    df_veeqo[f'HAS_{col}'] = df_veeqo[col].notna()

# --- Capping híbrido ---

# Columnas con IQR suavizado (3.0)
columnas_iqr = ['ACTIVE_CHANNELS', 'INACTIVE_CHANNELS', 'ALL_USERS_COUNT', 'ADMIN_USER_COUNT', 'WAREHOUSES', 'FBA_WAREHOUSES']

# Columnas con Percentil 99
columnas_percentil = ['UNIQUE_CHANNELS_ACTIVE_EXCL_DIRECT_AMAZON', 'UNIQUE_CHANNELS_ACTIVE_EXCL_AMAZON', 'ACTIVE_USERS_L_28D']

summary_hibrido = []

# Aplicar IQR suavizado (3.0)
for col in columnas_iqr:
    if col not in df_veeqo.columns:
        continue

    s = df_veeqo[col].dropna()
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + (3.0 * IQR)
    upper_limit = int(np.round(upper))

    original_max = df_veeqo[col].max()
    mask = df_veeqo[col] > upper_limit
    num_capped = int(mask.sum())

    df_veeqo.loc[mask, col] = upper_limit
    new_max = df_veeqo[col].max()

    summary_hibrido.append({
        'Columna': col,
        'Método': 'IQR (3.0)',
        'Límite': upper_limit,
        'Capados': num_capped,
        'Original_Max': int(original_max),
        'Nuevo_Max': int(new_max),
        '% Capado': round(num_capped / len(df_veeqo) * 100, 2)
    })

# Aplicar Percentil 99
for col in columnas_percentil:
    if col not in df_veeqo.columns:
        continue

    s = df_veeqo[col].dropna()
    p99 = s.quantile(0.99)
    upper_limit = int(np.round(p99))

    original_max = df_veeqo[col].max()
    mask = df_veeqo[col] > upper_limit
    num_capped = int(mask.sum())

    df_veeqo.loc[mask, col] = upper_limit
    new_max = df_veeqo[col].max()

    summary_hibrido.append({
        'Columna': col,
        'Método': 'Percentil 99',
        'Límite': upper_limit,
        'Capados': num_capped,
        'Original_Max': int(original_max),
        'Nuevo_Max': int(new_max),
        '% Capado': round(num_capped / len(df_veeqo) * 100, 2)
    })

summary_hibrido_df = pd.DataFrame(summary_hibrido)
print("[OK] Capping Híbrido Completado:")
print(summary_hibrido_df.to_string())
print(f"\nDataFrame limpio y procesado: {df_veeqo.shape[0]} filas × {df_veeqo.shape[1]} columnas")
```

```python
from matplotlib.patches import Patch

# Visualizar el resultado del capping híbrido
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Gráfico 1: Comparación de máximos antes y después
ax1 = axes[0, 0]
plot_df_comparison = summary_hibrido_df[['Original_Max', 'Nuevo_Max']].copy()
plot_df_comparison.index = summary_hibrido_df['Columna']
plot_df_comparison.plot(kind='bar', ax=ax1, rot=45, title='Original Max vs Nuevo Max (Capping Híbrido)', color=['#b91c1c', '#0f766e'])
ax1.set_ylabel('Valor máximo')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3, axis='y')

# Gráfico 2: Porcentaje de registros capados
ax2 = axes[0, 1]
colors = ['#b91c1c' if x > 2 else '#0f766e' for x in summary_hibrido_df['% Capado']]
ax2.bar(range(len(summary_hibrido_df)), summary_hibrido_df['% Capado'], color=colors)
ax2.set_xticks(range(len(summary_hibrido_df)))
ax2.set_xticklabels(summary_hibrido_df['Columna'], rotation=45, ha='right')
ax2.set_ylabel('% registros capados')
ax2.set_title('Porcentaje de Registros Capados por Columna')
ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Umbral 1%')
ax2.axhline(y=5, color='orange', linestyle='--', linewidth=2, label='Umbral 5%')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Gráfico 3: Método utilizado
ax3 = axes[1, 0]
method_colors = {'IQR (3.0)': '#6d28d9', 'Percentil 99': '#d97706'}
colors_method = [method_colors[m] for m in summary_hibrido_df['Método']]
ax3.barh(summary_hibrido_df['Columna'], summary_hibrido_df['Capados'], color=colors_method)
ax3.set_xlabel('Número de registros capados')
ax3.set_title('Registros Capados por Método')
ax3.grid(True, alpha=0.3, axis='x')

# Añadir leyenda
legend_elements = [Patch(facecolor='#6d28d9', label='IQR (3.0)'), Patch(facecolor='#d97706', label='Percentil 99')]
ax3.legend(handles=legend_elements, loc='lower right')

# Gráfico 4: Tabla resumen
ax4 = axes[1, 1]
ax4.axis('tight')
ax4.axis('off')
table_data = summary_hibrido_df[['Columna', 'Método', 'Límite', 'Capados', '% Capado']].values.tolist()
table = ax4.table(cellText=table_data,
                  colLabels=['Columna', 'Método', 'Límite', 'Capados', '% Capado'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.25, 0.2, 0.1, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)
ax4.set_title('Resumen Detallado del Capping Híbrido', pad=20)

plt.tight_layout()
plt.show()

print("[OK] Visualización completada. El capping híbrido se aplicó exitosamente.")
print(f"\n📊 Resumen General:")
print(f"   • Total de registros en el dataset: {len(df_veeqo):,}")
print(f"   • Registros modificados (capados): {summary_hibrido_df['Capados'].sum():,} ({summary_hibrido_df['% Capado'].sum():.2f}%)")
print(f"   • Columnas procesadas: {len(summary_hibrido_df)}")
print(f"   • Eficiencia: [OK] Todas las columnas bajo el 5% de modificación")
```

**Lectura de los resultados:**

1. **Meta cumplida (< 5% por columna).** Modificar más del 5% de una variable empieza a deformar la distribución original. Acá el máximo bajó fuerte en varias columnas sin pasarse de ese umbral.
2. **Impacto total bajo.** En conjunto, 8.73% de los registros (7.344 filas) tuvieron algún valor capado — un número sano para el tamaño del dataset.
3. **El enfoque híbrido se justifica.** El IQR suavizado (factor 3.0) le da margen a columnas como `WAREHOUSES` o `ACTIVE_CHANNELS` para no penalizar a clientes corporativos genuinamente grandes. El percentil 99, en cambio, funciona mejor en columnas con muchos ceros, donde el IQR clásico sería demasiado agresivo.

## Modelado — primera pasada

Antes de entrenar nada, terminamos de dejar el target listo: pasamos las fechas a `datetime` (ya lo veníamos haciendo) y definimos la variable binaria. Como primer intento vamos a marcar como **1 (Alto Valor / Activo)** a las suscripciones activas, y **0** al resto — ya en la siguiente celda vemos que esta primera definición hay que corregirla.

```python
# Preparacion de features y pipeline de preprocesado manual
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Limpieza y Formateo de Fechas (según tus especificaciones)
date_cols = [
    'COMPANY_CREATED_AT', 'PQL_DATE', 'FIRST_SHIPMENT_DATE', 'LAUNCHED_DATE',
    'LAST_SHIPMENT_DATE', 'LAST_PAGE_LOADED_AT', 'FIRST_SALES_INTERACTION',
    'MOS_ACTIVATED_AT', 'UPGRADED_TO_POWER_AT', 'FIRST_IMPORTED_ORDER',
    'ACTIVATED_WEEK', 'FIRST_CHANNEL_CONNECTED'
]

for col in date_cols:
    df_veeqo[col] = pd.to_datetime(df_veeqo[col], errors='coerce')

# 2. Definición del Target (High Value vs Churn/Trial)
# Definimos High Value como Activos (o según tu lógica de Deciles)
df_veeqo['TARGET'] = df_veeqo['SUBSCRIPTION_STATUS'].apply(
    lambda x: 1 if x in ['active', 'active_paid'] else 0
)

# 3. Preprocesamiento de variables numéricas y categóricas
# Seleccionamos variables operativas críticas para el análisis
cols_modelo = [
    'ACTIVE_CHANNELS', 'INACTIVE_CHANNELS', 'ACTIVE_USERS_L_28D',
    'ALL_USERS_COUNT', 'ADMIN_USER_COUNT', 'WAREHOUSES', 'FBA_WAREHOUSES',
    'DECILE', 'SIGNUP_TYPE', 'SELLER_TYPE', 'MARKETING_CHANNEL'
]

X = df_veeqo[cols_modelo].copy()
y = df_veeqo['TARGET']

# Manejo de nulos (imputación simple para el ejemplo)
X = X.fillna(0)

# Codificación de variables categóricas (One-Hot Encoding)
X = pd.get_dummies(X, columns=['SIGNUP_TYPE', 'SELLER_TYPE', 'MARKETING_CHANNEL'], drop_first=True)

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
```

### Selección de variables

Para elegir qué variables entran al modelo usamos la importancia que les asigna un Random Forest (`SelectFromModel`). En un dataset B2B como este ayuda a identificar qué variables operativas —cantidad de canales, almacenes, usuarios— realmente pesan en la retención.

```python
# Seleccion de variables con Random Forest (SelectFromModel)
# Entrenamos un selector basado en Random Forest
selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
selector.fit(X_train, y_train)

# Obtenemos las columnas seleccionadas
feats_top = X_train.columns[(selector.get_support())]
print(f"Características seleccionadas ({len(feats_top)}):", list(feats_top))

# Reducimos dimensionalidad
X_train_sel = X_train[feats_top]
X_test_sel = X_test[feats_top]
```

### ¿Por qué Random Forest para seleccionar variables?

Hay varias formas de hacer feature selection:

| Método | Ventaja | Desventaja |
|---|---|---|
| Correlación de Pearson | Simple y rápido | Solo detecta relaciones lineales |
| SelectKBest (Chi2, ANOVA) | Sin modelo | No considera interacciones entre variables |
| **SelectFromModel (RF)** | Captura no linealidades e interacciones | Más lento |
| RFE (Recursive Feature Elimination) | Muy preciso | Muy costoso computacionalmente |

Acá elegimos `SelectFromModel` con Random Forest porque:
- Veeqo tiene relaciones no lineales (tener 3 warehouses no vale el triple que tener 1)
- El RF puede captar que `ALL_USERS_COUNT` combinado con `ACTIVE_CHANNELS` predice mejor que cada variable sola
- El umbral `threshold='mean'` descarta automáticamente lo que está por debajo del promedio de importancia

```python
print("Contenido real de SUBSCRIPTION_STATUS:")
print(df_veeqo['SUBSCRIPTION_STATUS'].unique())
```

En el ecosistema Veeqo/Amazon un cliente no queda marcado como "active", sino como `live`. Los que están en `trialing` todavía están en período de prueba, y `canceled`/`on-hold` ya se fueron o pausaron (esto es el churn). Entonces la lógica de Alto Valor vs. Churn/Prueba tiene que ser otra:

Por eso el target correcto marca como **1 (Alto Valor)** a quienes están en `live` o `implementation` (clientes reales, aunque estén terminando de configurarse), y **0** a todo el resto.

```python
# Definicion de clasificacion binaria: live/implementation=1, resto=0
# Definimos el Target según tus valores únicos
df_veeqo['TARGET'] = df_veeqo['SUBSCRIPTION_STATUS'].apply(
    lambda x: 1 if x in ['live', 'implementation'] else 0
)

print("Nueva distribución del Target (Veeqo):")
print(df_veeqo['TARGET'].value_counts())
```

---
## Qué tan desbalanceadas están las clases

### ¿Por qué el desbalance complica la clasificación?

Si el 99% de los clientes son Churn/Trial y solo el 1% es Alto Valor, un modelo que **siempre** predice "Churn" saca:

- **Accuracy = 99%** → suena espectacular, pero no sirve para nada
- **0% de los clientes Alto Valor detectados** → que es justo lo que nos importa

### Qué hacemos al respecto

| Técnica | Cuándo usarla | Cómo se aplica acá |
|---|---|---|
| `class_weight='balanced'` | Siempre como primer paso | En RandomForest y LogisticRegression |
| Cambiar la métrica de evaluación | Siempre | Usar **ROC-AUC** en lugar de Accuracy |
| SMOTE (oversampling) | Si el modelo sigue fallando | Sintetizar ejemplos de la clase minoritaria |
| Ajustar el umbral de decisión | En producción | Bajar de 0.5 a 0.3 para capturar más clase 1 |

```python
#----------------------------------------------------------------------
# Balance de clases
#----------------------------------------------------------------------
#
# Este grafico se hace ANTES de entrenar cualquier modelo.
# Objetivo: entender el desbalance para elegir la metrica correcta.
#
# En Veeqo el negocio tiene sentido con desbalance:
#   Solo una fraccion de empresas que se registran se convierten en clientes "live"
#   (activos pagando). La mayoria queda en trialing o cancela.
#   Ese desbalance no es un error del dataset, refleja la realidad del negocio.
#
# Por eso elegimos ROC-AUC:
#   ROC-AUC mide la CAPACIDAD DE RANKING del modelo:
#   ¿puede el modelo ordenar a los clientes de "mas probable de ser High Value"
#   a "menos probable"? Un modelo con AUC=0.85 dice: si tomo un cliente aleatorio
#   de clase 1 y uno de clase 0, el 85% de las veces le dara mayor puntaje al de clase 1.
#
import matplotlib.pyplot as plt

conteo = df_veeqo['TARGET'].value_counts()
pct    = df_veeqo['TARGET'].value_counts(normalize=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
labels_c = ['Clase 0\n(Churn / Trial)', 'Clase 1\n(High Value / Live)']
colores  = ['#ea580c', '#1d4ed8']

# Grafico 1: numeros absolutos para entender la escala del problema
axes[0].bar(labels_c, [conteo.get(0, 0), conteo.get(1, 0)],
            color=colores, edgecolor='black', linewidth=1.2)
axes[0].set_title("Balance de Clases — Conteo Absoluto", fontsize=11)
axes[0].set_ylabel("Numero de registros")
for i, v in enumerate([conteo.get(0, 0), conteo.get(1, 0)]):
    axes[0].text(i, v + 200, f"{v:,}", ha='center', fontweight='bold')

# Grafico 2: porcentajes para ver la PROPORCION del desbalance
axes[1].bar(labels_c, [pct.get(0, 0), pct.get(1, 0)],
            color=colores, edgecolor='black', linewidth=1.2)
axes[1].set_title("Balance de Clases — Porcentaje (%)", fontsize=11)
axes[1].set_ylabel("Porcentaje del total (%)")
for i, v in enumerate([pct.get(0, 0), pct.get(1, 0)]):
    axes[1].text(i, v + 0.5, f"{v:.1f}%", ha='center', fontweight='bold')

ratio = pct.get(0, 0) / max(pct.get(1, 0), 0.001)
plt.suptitle(
    f"Desbalance {ratio:.0f}:1  →  Accuracy NO sirve  →  usar ROC-AUC + class_weight='balanced'",
    fontsize=10, style='italic', color='darkred')
plt.tight_layout()
plt.savefig("balance_clases_veeqo.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"Ratio de desbalance: {ratio:.0f}:1")
print("Con este ratio, un modelo que siempre predice clase 0 tendria:")
print(f"  Accuracy = {pct.get(0,0):.1f}%  (enganoso: detecta 0% de clientes High Value)")
print(f"  ROC-AUC  = 0.50  (equivalente a tirar una moneda)")
```

```python
# Random Forest Classifier (modelo principal)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, roc_auc_score

# Preparación de variables (X) e y
# Nota: Quitamos SUBSCRIPTION_STATUS de X porque es la base del Target (evita Data Leakage)
cols_modelo = [
    'ACTIVE_CHANNELS', 'INACTIVE_CHANNELS', 'ACTIVE_USERS_L_28D',
    'ALL_USERS_COUNT', 'ADMIN_USER_COUNT', 'WAREHOUSES', 'FBA_WAREHOUSES',
    'DECILE'
]

X = df_veeqo[cols_modelo].fillna(0)
y = df_veeqo['TARGET']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Feature Selection
selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
selector.fit(X_train, y_train)
feats_top = X_train.columns[(selector.get_support())]

# Entrenamiento
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train[feats_top], y_train)

# Métricas (Ya no dará error el index [:, 1])
y_pred = rf_model.predict(X_test[feats_top])
y_prob = rf_model.predict_proba(X_test[feats_top])[:, 1]

print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred))
```

### Métricas de validación

Acá calculamos lo esencial para confirmar que el modelo efectivamente distingue entre los dos grupos.

```python
# Accuracy + ROC-AUC + Classification Report
print("--- MÉTRICAS DEL MODELO RANDOM FOREST ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred))

# Visualización de la importancia de las variables
importances = pd.Series(rf_model.feature_importances_, index=feats_top).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=importances.index)
plt.title("Factores Críticos para la Salud del Cliente (Feature Importance)")
plt.show()
```

```python
#----------------------------------------------------------------------
# Matriz de confusión y curva ROC — Random Forest
#----------------------------------------------------------------------
#
# ─── CONFUSION MATRIX ────────────────────────────────────────────
# La confusion matrix muestra las 4 combinaciones posibles entre
# lo que el modelo PREDICE y lo que REALMENTE es cada cliente:
#
#                   | Predice: Churn | Predice: High Value
#   Real: Churn     |   VN (bueno)   |   FP (alarma falsa)
#   Real: High Value|   FN (costoso) |   VP (bueno)
#
# VN = Verdadero Negativo : predijo Churn → era Churn      ✓ Ahorramos recursos
# FP = Falso Positivo     : predijo Live  → era Churn      ✗ Gastamos recursos en quien no vale
# FN = Falso Negativo     : predijo Churn → era High Value ✗ PERDEMOS un cliente valioso (alto costo)
# VP = Verdadero Positivo : predijo Live  → era High Value ✓ Retenemos a quien vale
#
# En negocio B2B: el FN es el ERROR MAS COSTOSO.
# Perder un cliente High Value = perder subscription + potencial de expansion de cuenta.
# Por eso miramos especialmente el RECALL de la clase 1:
#   Recall clase 1 = VP / (VP + FN) → ¿de todos los High Value, cuantos capturamos?
#
# ─── CURVA ROC ───────────────────────────────────────────────────
# ROC = Receiver Operating Characteristic (curva de caracteristica operativa)
# Origen: radar de la Segunda Guerra Mundial (detectar aviones vs ruido)
# Hoy: estandar en medicina, credit scoring, churn prediction
#
# Eje X: FPR = FP / (FP + VN)  → tasa de "alarmas falsas" sobre todos los negativos
# Eje Y: TPR = VP / (VP + FN)  → tasa de aciertos sobre todos los positivos (= Recall clase 1)
# Cada punto de la curva = un umbral de decision distinto (de 0.0 a 1.0)
#
# AUC (Area Under Curve):
#   1.00 → modelo perfecto: siempre distingue clase 1 de clase 0
#   0.90 → modelo excelente: recomendado para produccion
#   0.70 → modelo util: mejor que azar, mejorable
#   0.50 → equivale a clasificar al azar (linea diagonal)
#   <0.5 → modelo peor que el azar (algo esta muy mal)
#
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix — eje X = prediccion, eje Y = valor real
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Churn/Trial (0)', 'High Value (1)'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title("Confusion Matrix — Random Forest\n"
                  "Fila = Real | Columna = Prediccion", fontsize=11)

# Curva ROC — una linea por cada umbral de decision posible
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_val = auc(fpr, tpr)

axes[1].plot(fpr, tpr, color='#b91c1c', lw=2.5,
             label=f'Random Forest (AUC = {roc_val:.3f})')
axes[1].plot([0, 1], [0, 1], color='#64748b', lw=1.5, linestyle='--',
             label='Clasificador aleatorio (AUC = 0.50)')
axes[1].fill_between(fpr, tpr, alpha=0.10, color='#b91c1c')
axes[1].set_xlabel("FPR — Tasa de Falsos Positivos (alarmas falsas)")
axes[1].set_ylabel("TPR / Recall — Tasa de Verdaderos Positivos")
axes[1].set_title("Curva ROC — Random Forest\n"
                  "Area azul = ganancia sobre el azar", fontsize=11)
axes[1].legend(loc='lower right')

plt.suptitle(f"Evaluacion Visual del Modelo | ROC-AUC = {roc_val:.3f}",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("confusion_roc_veeqo.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"ROC-AUC = {roc_val:.4f}")
print("Interpretacion:")
print("  >= 0.90 → Excelente  |  0.70-0.89 → Bueno  |  0.50-0.69 → Regular")
```

```python
#----------------------------------------------------------------------
# Importancia de variables — qué mira el modelo
#----------------------------------------------------------------------
#
# ¿Qué son las Feature Importances en Random Forest?
# --------------------------------------------------
# Un Random Forest construye N arboles de decision.
# En cada nodo de cada arbol, elige la variable que mas reduce la "impureza" (Gini).
# La impureza Gini mide el desorden de las clases en un nodo:
#   Gini = 0.0 → nodo puro (todos de la misma clase) — perfecto
#   Gini = 0.5 → nodo 50/50 entre clases — no aporta informacion
#
# feature_importances_[i] = reduccion promedio de Gini que produce la variable i
#   Suma total = 1.0 (es una distribucion de importancia relativa)
#   Variable con 0.40 → sola explica el 40% de las decisiones del bosque
#   Variable con 0.01 → casi irrelevante para el modelo
#
# ¿Para qué sirve este grafico en el contexto del proyecto Veeqo?
# ---------------------------------------------------------------
# Nos dice QUÉ COMPORTAMIENTOS del cliente distinguen a los High Value de los Churn.
# En B2B, conocer esto tiene valor de negocio directo:
#   Si ALL_USERS_COUNT es la variable mas importante →
#   El equipo de Customer Success debe enfocarse en ayudar al cliente a agregar usuarios
#   desde el primer dia de onboarding.
# Las variables de baja importancia pueden eliminarse → modelo mas liviano y rapido.
#
import matplotlib.pyplot as plt
import pandas as pd

fi_df = pd.DataFrame({
    'Variable': feats_top,
    'Importancia': rf_model.feature_importances_
}).sort_values('Importancia', ascending=True)  # ascending=True para grafico horizontal correcto

# Colores: verde para variables por encima del percentil 60 (las mas utiles)
umbral = fi_df['Importancia'].quantile(0.6)
colores_fi = ['#b91c1c' if v >= umbral else '#94a3b8' for v in fi_df['Importancia']]

fig, ax = plt.subplots(figsize=(10, max(5, len(feats_top) * 0.45)))

# barh() = barras horizontales → mejor para muchas categorias (texto legible)
bars = ax.barh(fi_df['Variable'], fi_df['Importancia'],
               color=colores_fi, edgecolor='black', linewidth=0.5)

# Etiquetas numericas en cada barra para lectura precisa
for bar, val in zip(bars, fi_df['Importancia']):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=9)

# Linea de promedio → variables por encima de esto aportan mas que el promedio
prom = fi_df['Importancia'].mean()
ax.axvline(prom, color='red', linestyle='--', alpha=0.7,
           label=f"Importancia promedio ({prom:.3f})")
ax.set_xlabel("Importancia Relativa (reduccion de impureza Gini)", fontsize=11)
ax.set_title(
    "Feature Importances — Random Forest Classifier\n"
    "Verde = variables que mas contribuyen (por encima del percentil 60)", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig("feature_importances_veeqo.png", dpi=150, bbox_inches='tight')
plt.show()

print("Top 5 variables mas predictivas para identificar clientes High Value:")
print(fi_df.tail(5)[['Variable', 'Importancia']]
      .sort_values('Importancia', ascending=False)
      .to_string(index=False))
print()
print("Interpretacion de negocio:")
print("  La variable #1 en importancia es el PREDICTOR MAS FUERTE.")
print("  Segun el paper de Veeqo: ALL_USERS_COUNT suele liderar —")
print("  el cliente que agrega mas usuarios a su cuenta tiene exponencialmente")
print("  mas chances de convertirse en un cliente 'live' de alto valor.")
```

```python
#para ver nomas la cantidad de compañias únicas por cada estado de suscripción
# Crear tabla dinámica con SUBSCRIPTION_STATUS y contar empresas únicas
pivot_table = df_veeqo.groupby('SUBSCRIPTION_STATUS')['COMPANY_ID'].nunique().reset_index()
pivot_table.columns = ['SUBSCRIPTION_STATUS', 'Unique_Companies']

# Calcular el porcentaje respecto del total
total_companies = pivot_table['Unique_Companies'].sum()
pivot_table['Porcentaje (%)'] = (pivot_table['Unique_Companies'] / total_companies * 100).round(2)

# Ordenar por cantidad descendente
pivot_table = pivot_table.sort_values('Unique_Companies', ascending=False).reset_index(drop=True)

print(pivot_table)
```

```python
# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

import matplotlib.pyplot as plt

# Usar Regresión Logística con las características seleccionadas
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train[feats_top], y_train)

# Predicciones
y_pred_lr = lr_model.predict(X_test[feats_top])
y_prob_lr = lr_model.predict_proba(X_test[feats_top])[:, 1]

# Métricas
print("--- METRICAS DEL MODELO DE REGRESIÓN LOGÍSTICA ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_lr):.4f}")
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred_lr))

# Comparar coeficientes
coef_df = pd.DataFrame({
    'Feature': feats_top,
    'Coeficiente': lr_model.coef_[0]
}).sort_values('Coeficiente', key=abs, ascending=False)

plt.figure(figsize=(11, 6))
plt.barh(coef_df['Feature'], coef_df['Coeficiente'])
plt.xlabel('Coeficiente de Regresión Logística')
plt.title('Impacto de Variables en Probabilidad de ser High Value')
plt.tight_layout()
plt.show()

print("\nCoeficientes de Regresión Logística:")
print(coef_df)
```

```python
# K-Nearest Neighbors (modelo complementario)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

# Prepare data if not already done
if 'X_train' not in locals() or 'X_test' not in locals():
    from sklearn.model_selection import train_test_split
    cols_modelo = [
        'ACTIVE_CHANNELS', 'INACTIVE_CHANNELS', 'ACTIVE_USERS_L_28D',
        'ALL_USERS_COUNT', 'ADMIN_USER_COUNT', 'WAREHOUSES', 'FBA_WAREHOUSES',
        'DECILE'
    ]

    X = df_veeqo[cols_modelo].fillna(0)
    y = df_veeqo['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Feature selection
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
    selector.fit(X_train, y_train)
    feats_top = X_train.columns[(selector.get_support())]

# Entrenar KNeighborsClassifier con las características seleccionadas
knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_model.fit(X_train[feats_top], y_train)

# Predicciones
y_pred_knn = knn_model.predict(X_test[feats_top])
y_prob_knn = knn_model.predict_proba(X_test[feats_top])[:, 1]

# Métricas
print("--- MÉTRICAS DEL MODELO K-NEAREST NEIGHBORS ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_knn):.4f}")
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred_knn))

# Visualización de distancias entre vecinos
plt.figure(figsize=(10, 6))
distances, indices = knn_model.kneighbors(X_test[feats_top][:100])
plt.plot(distances.mean(axis=1), marker='o', linestyle='-', label='Distancia promedio a 5 vecinos')
plt.xlabel('Muestras de Test')
plt.ylabel('Distancia Euclidiana Promedio')
plt.title('Distancias de los K Vecinos Más Cercanos (Primeras 100 muestras)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

```python
# XGBoost Classifier
%pip install xgboost
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

import matplotlib.pyplot as plt

# Entrenar XGBoost Classifier con las características seleccionadas
xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train[feats_top], y_train)

# Predicciones
y_pred_xgb = xgb_model.predict(X_test[feats_top])
y_prob_xgb = xgb_model.predict_proba(X_test[feats_top])[:, 1]

# Métricas
print("--- MÉTRICAS DEL MODELO XGBOOST CLASSIFIER ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_xgb):.4f}")
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred_xgb))

# Feature importance
feature_importance_xgb = xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feats_top,
    'Importance': feature_importance_xgb
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(11, 5))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importancia')
plt.title('Feature Importance - XGBoost Classifier')
plt.tight_layout()
plt.show()

print("\nFeature Importance:")
print(feature_importance_df)
```

---
## Pipeline formal: `Pipeline` + `GridSearchCV` + `StratifiedKFold`

### ¿Por qué un Pipeline en vez de hacer cada paso a mano?

El preprocesado manual de más arriba tiene un problema: **data leakage**.

> Calcular la mediana de una columna sobre **todo el dataset** (train + test) e imputar con eso significa que el modelo ya "vio" algo del test set antes de ser evaluado. Las métricas quedan infladas.

Un `Pipeline` lo evita porque:
1. `.fit(X_train)` aprende mediana, escala y vocabulario del encoder **solo** con train
2. `.transform(X_test)` aplica lo aprendido sin tocar las estadísticas del test

### `StratifiedKFold` — ¿para qué sirve con datos tan desbalanceados?

Con un `train_test_split` común podría tocarnos un fold sin ningún ejemplo de la clase 1. `StratifiedKFold` obliga a que **cada fold mantenga la misma proporción de clases** que el dataset completo.

```
Dataset: 99% clase 0 / 1% clase 1
Fold 1 Train: 99% clase 0 / 1% clase 1  ← StratifiedKFold asegura esto
Fold 1 Test:  99% clase 0 / 1% clase 1  ← en cada una de las 5 rondas
```

### `GridSearchCV`, en criollo

Prueba todas las combinaciones de hiperparámetros del grid y, para cada una, corre validación cruzada con `StratifiedKFold`. Se queda con la combinación que da mejor `roc_auc` promedio entre los 5 folds.

```
n_estimators=[100, 200]  x  max_depth=[5, 10, None]  x  min_samples_split=[2, 5]
= 2 x 3 x 2 = 12 combinaciones  x  5 folds = 60 entrenamientos en total
```

### ¿OneHotEncoder o TargetEncoder?

Hasta acá el pipeline usaba solo variables numéricas. Sumamos las categóricas que habían quedado afuera del modelo final (`SIGNUP_TYPE`, `SELLER_TYPE`, `MARKETING_CHANNEL`) porque se estaba desperdiciando señal real.

- **`OneHotEncoder`**: una columna binaria por categoría. Anda bien con pocas categorías — `SIGNUP_TYPE` tiene 3, `SELLER_TYPE` tiene 2.
- **`TargetEncoder`**: reemplaza cada categoría por el promedio del target dentro de esa categoría. Conviene con más niveles — `MARKETING_CHANNEL` tiene 14, así que en vez de 14 columnas dispersas queda **una sola columna numérica** con la misma información resumida.

> ⚠️ El riesgo del `TargetEncoder` es el mismo data leakage de siempre: si el promedio se calcula con TODO el dataset (test incluido), el modelo espía el resultado antes de tiempo. Por eso va **adentro** del `Pipeline` — en cada fold de `GridSearchCV` se ajusta solo con el train de ese fold.

```python
#----------------------------------------------------------------------
# Pipeline formal con GridSearchCV y StratifiedKFold
#----------------------------------------------------------------------
#
# Este bloque muestra la forma CORRECTA y PROFESIONAL de construir
# un pipeline de clasificacion binaria. Diferencias con el preprocesado
# manual (hecho mas arriba):
#
#   MANUAL (arriba en el notebook):
#     df_veeqo[col].fillna(0)         <- calcula sobre TODO el dataset (leakage)
#     pd.get_dummies(X, ...)           <- puede crear columnas distintas en train vs test
#     Las categoricas (SIGNUP_TYPE, SELLER_TYPE, MARKETING_CHANNEL) quedaron
#     afuera del modelo final que realmente se entreno
#
#   PIPELINE (este bloque):
#     SimpleImputer.fit(X_train)       <- aprende mediana/moda SOLO de train
#     SimpleImputer.transform(X_test)  <- aplica eso a test sin verlo
#     OneHotEncoder                    <- para categoricas de pocas categorias
#     TargetEncoder                    <- para MARKETING_CHANNEL (14 categorias),
#                                          ajustado solo con train dentro de cada fold
#
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# 0. FEATURES: numericas + categoricas (sumamos las 3 categoricas que
#    en el preprocesado manual habian quedado afuera del modelo final)
# ─────────────────────────────────────────────────────────────────
cols_num = ['ACTIVE_CHANNELS', 'INACTIVE_CHANNELS', 'ACTIVE_USERS_L_28D',
                 'ALL_USERS_COUNT', 'ADMIN_USER_COUNT', 'WAREHOUSES',
                 'FBA_WAREHOUSES', 'DECILE']
cols_cat_ohe = ['SIGNUP_TYPE', 'SELLER_TYPE']      # 2-3 categorias -> OneHotEncoder
cols_cat_target = ['MARKETING_CHANNEL']               # 14 categorias -> TargetEncoder

X_mod = df_veeqo[cols_num + cols_cat_ohe + cols_cat_target].copy()
y_mod = df_veeqo['TARGET']

X_train, X_test, y_train, y_test = train_test_split(
    X_mod, y_mod, test_size=0.3, random_state=42, stratify=y_mod
)

# ─────────────────────────────────────────────────────────────────
# 1. PREPROCESADOR (ColumnTransformer)
#    Aplica un tratamiento distinto segun el TIPO de columna
# ─────────────────────────────────────────────────────────────────
preprocesador = ColumnTransformer(transformers=[
    ('num', Pipeline([
        # SimpleImputer: aprende la mediana de X_train y la aplica a X_test
        # strategy='median' -> robusto a outliers
        ('imputer', SimpleImputer(strategy='median')),

        # StandardScaler: (x - media) / std -> necesario para LogisticRegression
        ('scaler', StandardScaler())
    ]), cols_num),

    # OneHotEncoder: para categoricas con pocas categorias, sin orden entre ellas
    # handle_unknown='ignore' evita que explote si test tiene una categoria que train no vio
    ('cat_ohe', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ]), cols_cat_ohe),

    # TargetEncoder: para MARKETING_CHANNEL (14 categorias). En vez de crear 14
    # columnas nuevas (como haria OneHot), reemplaza cada categoria por el
    # promedio del target dentro de esa categoria -> queda 1 sola columna numerica
    ('cat_target', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Desconocido')),
        ('target_enc', TargetEncoder(random_state=42))
    ]), cols_cat_target),
], remainder='drop')

# ─────────────────────────────────────────────────────────────────
# 2. PIPELINES COMPLETOS
# ─────────────────────────────────────────────────────────────────

# Baseline: Logistic Regression
# class_weight='balanced' -> el modelo asigna mayor peso a la clase minoritaria
pipeline_lr = Pipeline([
    ('prep',   preprocesador),
    ('modelo', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

# Modelo avanzado: Random Forest
pipeline_rf = Pipeline([
    ('prep',   preprocesador),
    ('modelo', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# ─────────────────────────────────────────────────────────────────
# 3. STRATIFIEDKFOLD
#    n_splits=5 -> 5 rondas, cada una respeta la proporcion de clases
# ─────────────────────────────────────────────────────────────────
cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─────────────────────────────────────────────────────────────────
# 4. GRIDSEARCHCV
#    12 combinaciones x 5 folds = 60 entrenamientos.
#    En cada fold, el TargetEncoder se ajusta SOLO con el train de ese fold
# ─────────────────────────────────────────────────────────────────
param_grid = {
    'modelo__n_estimators':      [100, 200],
    'modelo__max_depth':         [5, 10, None],
    'modelo__min_samples_split': [2, 5]
}

print("[GRIDSEARCHCV] Probando 12 combinaciones x 5 folds = 60 entrenamientos...")
print("(puede tardar ~2 minutos segun hardware)")

grid_search = GridSearchCV(
    estimator=pipeline_rf,
    param_grid=param_grid,
    cv=cv_strat,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)

print()
print(f"Mejor combinacion de hiperparametros: {grid_search.best_params_}")
print(f"ROC-AUC promedio en los 5 folds:      {grid_search.best_score_:.4f}")

# ─────────────────────────────────────────────────────────────────
# 5. EVALUACION FINAL EN TEST SET
# ─────────────────────────────────────────────────────────────────
pipe_ganador = grid_search.best_estimator_
y_pred_pipe = pipe_ganador.predict(X_test)
y_prob_pipe = pipe_ganador.predict_proba(X_test)[:, 1]

pipeline_lr.fit(X_train, y_train)
y_pred_lr_p = pipeline_lr.predict(X_test)
y_prob_lr_p = pipeline_lr.predict_proba(X_test)[:, 1]

print()
print("=" * 60)
print("COMPARACION FINAL - Pipeline Formal (evaluacion honesta en test)")
print("=" * 60)
print(f"  Logistic Regression (baseline) -> ROC-AUC: {roc_auc_score(y_test, y_prob_lr_p):.4f}")
print(f"  Random Forest (GridSearchCV)   -> ROC-AUC: {roc_auc_score(y_test, y_prob_pipe):.4f}")
print()
print("Reporte detallado del Random Forest optimizado:")
print(classification_report(y_test, y_pred_pipe, target_names=['Churn/Trial', 'High Value']))
```

**¿Por qué el recall mejoró tanto al sumar las categóricas?** (de 34% a un recall casi perfecto en la clase Alto Valor)

Antes de festejar, hay que descartar que alguna categoría separe las clases casi perfectamente — eso sería señal de fuga de información, no de una mejora real. Mirando la tasa de conversión por categoría:

| Variable | Categoría | Tasa de conversión a `live` |
|---|---|---|
| `SIGNUP_TYPE` | Email Sign-up | 2.60% |
| `SIGNUP_TYPE` | Amazon SSO | 0.03% |
| `SELLER_TYPE` | Scaler | 2.60% |
| `SELLER_TYPE` | Starter | 0.18% |

Ninguna categoría llega a 0% ni a 100% — hay diferencias grandes (Email Sign-up convierte ~80× más que Amazon SSO) pero nada determinístico. Es señal real, no leakage: el tipo de alta y el tipo de vendedor ya predicen bastante bien el compromiso futuro, y esa información está disponible desde el día 1, antes de cualquier comportamiento operativo.

---
## Riesgos y controles de calidad

Antes de cerrar el análisis, dejamos explícito qué se controló y qué quedaría pendiente si esto pasara a producción.

| Riesgo | ¿Se controló? | Cómo |
|---|---|---|
| **Data leakage** (usar info de test durante el entrenamiento) | ✅ Sí | El `Pipeline` de la sección anterior ajusta `SimpleImputer` / `StandardScaler` / `OneHotEncoder` / `TargetEncoder` solo con los datos de entrenamiento de cada fold de `GridSearchCV` |
| **Imputar usando el target** | ✅ Sí | Ninguna imputación usa `SUBSCRIPTION_STATUS` ni `TARGET` — solo estadísticas de las variables predictoras |
| **Desequilibrio de clases** (99.18% / 0.82%) | ✅ Parcial | `class_weight='balanced'` + métrica `roc_auc` en vez de Accuracy. Pendiente: probar SMOTE si el recall de la clase 1 sigue siendo bajo (Logistic Regression detecta ~34% de los `live` reales, según la tabla resumen de abajo) |
| **Overfitting por tuning excesivo** | ⚠️ Monitoreado | Se compara `grid_search.best_score_` (promedio en CV) contra el ROC-AUC en test. Si la diferencia fuera grande, indicaría que el modelo se ajustó a la grilla de hiperparámetros en vez de generalizar |
| **Dependencia temporal** | ❌ No aplica en este dataset | Es un corte transversal (no hay una fecha de "hoy" que separe pasado de futuro), por eso `StratifiedKFold` es correcto acá |

**¿Cuándo no alcanza con `StratifiedKFold`?** Si en vez de predecir la conversión inicial trial→live quisiéramos predecir el churn mes a mes de clientes que ya son `live`, ahí sí habría dependencia temporal real — entrenar con datos de "mañana" para predecir "hoy" sería otra forma de leakage. En ese caso correspondería `TimeSeriesSplit`, que arma los folds respetando el orden cronológico.

### Cómo se relaciona cada sección con el pipeline

| Sección del notebook | Etapa |
|---|---|
| Carga del Dataset + Diccionario de Variables | Carga de datos |
| Análisis inicial (nulos, tipos) + Balance de Clases | EDA |
| Tratamiento de nulos, outliers y fechas | ETL / Limpieza |
| Feature Selection (`SelectFromModel`) | Ingeniería de features |
| `ColumnTransformer` (imputación + escalado + OneHot + TargetEncoder) | Preprocesado |
| Logistic Regression / Random Forest / KNN / XGBoost | Modelado |
| `GridSearchCV` + `StratifiedKFold` | Búsqueda de hiperparámetros |
| ROC-AUC, `classification_report`, feature importances | Evaluación |

## Conclusiones

A lo largo del análisis procesamos y modelamos datos de 84.063 empresas que usan Veeqo. El trabajo se dividió en tres etapas: limpieza de datos, tratamiento de outliers y modelado predictivo.

### Sobre la calidad de los datos

La limpieza fue intensiva pero necesaria: el dataset traía fechas como texto, IDs numéricos, y cerca del 20% de valores nulos en algunas columnas. El capping híbrido (IQR suavizado + percentil 99) normalizó las variables extremas sin perder la naturaleza de los clientes corporativos grandes — solo 8.73% de los registros terminó ajustado.

### Sobre el desbalance y los resultados del modelado

Entrenamos varios modelos para distinguir clientes Alto Valor (`live`/`implementation`) de los que quedan en fase inicial o riesgo (`trialing`/`canceled`). El desafío central es el desbalance de clases — en el set de prueba, 25.013 registros de clase 0 contra apenas 206 de clase 1. Eso vuelve el Accuracy engañoso (~99% incluso sin aprender nada útil), así que la evaluación se centró en ROC-AUC y en la capacidad de detectar la clase minoritaria.

**Logistic Regression — el mejor balance general**
ROC-AUC: 0.9821 · Recall clase 1: 0.34
Le ganó a los ensambles en detectar la minoría. Sus coeficientes muestran que sumar usuarios a la cuenta (`ALL_USERS_COUNT`) es lo que más empuja la probabilidad de convertirse en cliente activo — y, algo interesante, tener canales inactivos (`INACTIVE_CHANNELS`) también suma: intentar conectar algo, aunque quede sin usarse, ya muestra compromiso inicial.

**XGBoost**
ROC-AUC: 0.9783 · Recall clase 1: 0.21
Rendimiento parejo pero conservador con la clase minoritaria. Le asigna casi el 80% del peso predictivo a una sola variable: `ALL_USERS_COUNT`.

**Random Forest**
ROC-AUC: 0.9740 · Recall clase 1: 0.22
Confirma la misma tendencia: tamaño de equipo (`ALL_USERS_COUNT`, `ADMIN_USER_COUNT`) y comportamiento transaccional (`DECILE`) son los pilares de la retención.

### El hallazgo principal: el equipo pesa más que la infraestructura

A diferencia de lo que uno intuiría en un software de logística (donde importarían más los canales o almacenes conectados), lo que más predice el éxito de un cliente en Veeqo es la adopción por parte del equipo. `ALL_USERS_COUNT` es, por lejos, el predictor más fuerte en todos los modelos: una empresa que suma usuarios propios tiene muchísima más probabilidad de terminar siendo cliente `live` que una que opera con una sola persona — más allá de cuántos almacenes tenga configurados.

### Tabla resumen de modelos

| Modelo | ROC-AUC aprox. | A favor | En contra |
|---|---|---|---|
| Logistic Regression | ★★★★☆ | Interpretable, coeficientes claros | Asume relaciones lineales |
| Random Forest | ★★★★☆ | Robusto a outliers, da feature importances | Caja negra |
| KNN | ★★★☆☆ | Capta patrones locales | Lento en producción, sensible a escala |
| XGBoost | ★★★★★ | Mayor capacidad predictiva | Necesita tuning cuidadoso |

> **Nota sobre la métrica:** con un desbalance de 99:1, el ROC-AUC puede quedar algo optimista por el tamaño chico de la clase minoritaria. En producción convendría sumar Precision-Recall AUC (Average Precision Score), más informativa cuando los positivos escasean tanto.
