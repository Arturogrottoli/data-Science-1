# Análisis 360 del Ciclo de Vida del Cliente B2B en E-commerce — Veeqo (Amazon)

**Materia:** Data Science I  
**Dataset:** `ecommerce_data.csv` (subir manualmente a Google Colab)  
**Notebook:** `ProyectoEjemplo.ipynb` (abrir en Google Colab)

---

## ¿De qué trata este proyecto?

Este proyecto analiza datos reales de **Veeqo**, una plataforma de gestión de inventarios y envíos adquirida por Amazon, orientada a pequeñas y medianas empresas que venden en múltiples canales (Shopify, eBay, Amazon, etc.).

El dataset contiene **84.063 registros** de empresas que se registraron en la plataforma, con información sobre su comportamiento operativo, los canales que conectaron, cuántos usuarios crearon y su estado final de suscripción.

### El problema de negocio

En el ecosistema B2B de SaaS (Software as a Service), el mayor costo no es adquirir un cliente sino **retenerlo**. La pregunta que guía este proyecto es:

> ¿Qué características operativas y de comportamiento distinguen a un cliente que se convierte en usuario activo y de alto valor (`live`) de uno que abandona la plataforma o se queda en fase de prueba (`trialing`, `canceled`)?

Poder predecir esto permite al equipo de Customer Success de Veeqo saber **a cuáles clientes dedicarle recursos de onboarding** antes de que abandonen.

---

## Dataset

- **Archivo:** `ecommerce_data.csv` (subir a `/content/` en Colab)
- **Filas:** 84.063 empresas registradas
- **Columnas:** 37 variables operativas, de suscripción y de comportamiento
- **Período:** Sin fecha fija, corte transversal del estado actual de cada cuenta

### Variable objetivo (TARGET)

```
SUBSCRIPTION_STATUS → TARGET binario

  live | implementation → 1  (High Value: cliente activo, paga y usa la plataforma)
  trialing | canceled | resto → 0  (Churn/Trial: abandonó o nunca convirtió)
```

El desbalance de clases es severo: **~99% clase 0 / ~1% clase 1**. Esto es esperable en B2B SaaS: de cada 100 empresas que se registran, solo una pequeña fracción se convierte en cliente real. Por eso se usa **ROC-AUC como métrica guía** en lugar de Accuracy.

---

## Estructura del Notebook

### 1. Lectura y análisis inicial (EDA)

- `df.info()` para ver dimensiones, tipos de datos y nulos
- Conteo de nulos por columna con tabla de porcentaje
- **Visualización de missingness con `missingno`** para detectar patrones de nulos

#### Hallazgos principales del EDA

| Columna | % de nulos | Tipo | Decisión tomada |
|---|---|---|---|
| `PHONE` | ~99% | MNAR | Eliminar (empresas no dan su teléfono voluntariamente) |
| Fechas de hitos (`PQL_DATE`, `FIRST_SHIPMENT_DATE`, etc.) | Variable | MNAR | Convertir a `HAS_X` (booleano: ¿ocurrió o no?) |
| Canales inactivos (`INACTIVE_CHANNELS`) | ~20% | MAR | Rellenar con 0 (ausencia = no tiene canales inactivos) |
| Campos de marketing (`MARKETING_CHANNEL`) | ~10% | MAR | Rellenar con "Desconocido" |

**¿Qué es MNAR?** Un nulo que tiene significado propio. En Veeqo, que una empresa no tenga `FIRST_SHIPMENT_DATE` no es un error del sistema, sino que **nunca realizó su primer envío**. Imputar esa fecha con la mediana sería incorrecto; en cambio, se crea `HAS_FIRST_SHIPMENT_DATE = 0` (variable booleana de presencia).

### 2. Limpieza y transformación de datos

Se corrigieron los tipos de dato de 37 columnas:

- **Fechas:** 11 columnas en formato `object` → `datetime64` con `pd.to_datetime(errors='coerce')`
- **Enteros:** Columnas `float64` que representaban conteos (usuarios, warehouses) → `int64`
- **Booleanos:** Fechas de hitos → `HAS_X` (1 si ocurrió, 0 si no)
- **Strings:** Columnas de texto normalizadas (espacios, mayúsculas)

### 3. Tratamiento de outliers (Capping Híbrido)

El dataset tiene clientes corporativos grandes con valores muy atípicos en variables como `ALL_USERS_COUNT` o `ACTIVE_CHANNELS`. Eliminarlos sería un error (son clientes reales, no errores de carga). En cambio, se aplicó **capping híbrido**:

- Si la columna tiene distribución normal/simétrica → **IQR suavizado × 3.0** como límite
- Si la columna tiene distribución sesgada → **Percentil 99** como límite

**Resultado obtenido:** Solo el **8.73% de los registros** fueron ajustados, y ninguna columna supera el **5% de registros modificados** individualmente. Esto cumple el "estándar de oro analítico" (< 5% por columna).

> El capping preserva la información (no elimina filas) pero evita que los outliers dominen los coeficientes del modelo.

### 4. Feature Engineering y Feature Selection

**Feature Engineering aplicado:**
- Variables `HAS_X` (booleanas) para 8 fechas de hitos operativos
- Conversión de fechas a días transcurridos desde la creación de la cuenta
- Imputación con 0 para variables de canales (ausencia = 0 activos)

**Feature Selection:**
Se entrenó un `RandomForestClassifier` preliminar sobre todas las variables disponibles y se usó `SelectFromModel` para conservar únicamente las **variables con importancia por encima de la media**. Esto reduce el ruido y hace el modelo más interpretable.

Variables finales seleccionadas:
```
ACTIVE_CHANNELS, INACTIVE_CHANNELS, ACTIVE_USERS_L_28D,
ALL_USERS_COUNT, ADMIN_USER_COUNT, WAREHOUSES, FBA_WAREHOUSES, DECILE
```

### 5. Modelado Predictivo

Se entrenaron **4 modelos de clasificación binaria**, comparados por ROC-AUC:

| Modelo | Rol | Por qué se eligió |
|---|---|---|
| **Logistic Regression** | Baseline | Modelo más simple; si el RF no lo supera, hay un problema |
| **Random Forest** | Principal | Robusto a outliers, no requiere normalización, da feature importances |
| **K-Nearest Neighbors** | Complementario | Detecta patrones locales no lineales |
| **XGBoost** | Complementario avanzado | Gradient boosting, generalmente el más potente |

**División de datos:**
```python
train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
```
`stratify=y` garantiza que el 1% de clase 1 se mantenga tanto en train como en test.

**Pipeline formal (sección de requisito):**  
Adicionalmente se construyó un `sklearn.Pipeline` con `SimpleImputer + StandardScaler` combinado con `GridSearchCV + StratifiedKFold(n_splits=5)` para demostrar la metodología correcta de búsqueda de hiperparámetros sin data leakage.

### 6. Evaluación del Modelo

**¿Por qué ROC-AUC y no Accuracy?**

Con 99% de clase 0, un modelo que siempre predice "Churn" tiene 99% de accuracy pero detecta 0 clientes High Value. El ROC-AUC mide la **capacidad de ranking**: de cada par (cliente High Value, cliente Churn), ¿cuántas veces el modelo le asigna mayor puntaje al High Value?

Visualizaciones incluidas en el notebook:
- **Balance de clases** (barras de conteo y porcentaje)
- **Confusion Matrix** con etiquetas de negocio (VN, VP, FP, FN y su costo)
- **Curva ROC** comparando todos los modelos
- **Feature Importances** del Random Forest (barras horizontales con importancia Gini)

---

## Resultado y Conclusión Principal

**"La adopción de equipo mata a la infraestructura"**

Todos los modelos coinciden en que el predictor más fuerte para identificar un cliente High Value es `ALL_USERS_COUNT`, la cantidad de usuarios registrados en la misma cuenta.

> Un cliente que ingresa a Veeqo y crea usuarios adicionales para su equipo tiene una probabilidad **exponencialmente mayor** de convertirse en cliente `live`, independientemente de la complejidad de su infraestructura (cantidad de warehouses, canales conectados, etc.).

**Implicación de negocio:** El equipo de Customer Success debería priorizar, en la primera semana de onboarding, ayudar al cliente a agregar al menos un usuario adicional a su cuenta. Ese es el comportamiento más predictivo del éxito a largo plazo.

---

## Cómo abrir y ejecutar este proyecto

Este proyecto está diseñado para correr en **Google Colab** con el dataset subido localmente.

### Pasos

1. Abrir el archivo `ProyectoEjemplo.ipynb` en Google Colab
2. En el panel izquierdo (ícono de carpeta), arrastrar el archivo `ecommerce_data.csv`
3. El archivo queda disponible en `/content/ecommerce_data.csv` y el notebook lo lee automáticamente

```python
df_retail = pd.read_csv("/content/ecommerce_data.csv")
```

> **Nota:** los archivos subidos directamente a Colab se borran al cerrar la sesión. Hay que volver a subirlos si se reinicia el runtime.

### Librerías necesarias

Todas están disponibles en Google Colab sin instalación adicional, excepto:

```python
pip install missingno   # visualización de nulos
pip install xgboost     # modelo XGBoost
```

Estas líneas ya están incluidas en el notebook.

---

## Diccionario de Variables Clave

| Variable | Tipo | Descripción |
|---|---|---|
| `SUBSCRIPTION_STATUS` | object | Estado de la suscripción → base del TARGET |
| `ALL_USERS_COUNT` | int | Total de usuarios registrados en la cuenta |
| `ACTIVE_CHANNELS` | int | Canales de venta conectados y activos |
| `INACTIVE_CHANNELS` | int | Canales conectados pero sin actividad reciente |
| `ACTIVE_USERS_L_28D` | int | Usuarios que usaron la plataforma en los últimos 28 días |
| `WAREHOUSES` | int | Almacenes propios del cliente conectados |
| `FBA_WAREHOUSES` | int | Almacenes gestionados por Amazon (Fulfillment by Amazon) |
| `DECILE` | int | Clasificación RFM del cliente (1 = más valioso, 10 = menos) |
| `HAS_FIRST_SHIPMENT_DATE` | int (0/1) | ¿El cliente realizó su primer envío? |
| `HAS_PQL_DATE` | int (0/1) | ¿El cliente alcanzó el hito de Product Qualified Lead? |
| `MARKETING_CHANNEL` | object | Canal por el que llegó el cliente (orgánico, pagado, etc.) |
| `TARGET` | int (0/1) | Variable objetivo: 1=High Value (live), 0=Churn/Trial |
