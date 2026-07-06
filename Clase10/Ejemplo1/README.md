# DS
📦 Análisis 360 del Ciclo de Vida del Cliente B2B en E-commerce (Veeqo - Amazon) (link del dataset: https://docs.google.com/spreadsheets/d/1lh5AACg42dKT8NWyR-Gi_pV9rPNqdMEFqniMVNA4jRw/edit?gid=573541234#gid=573541234)
📖 Abstracto y Objetivo del Proyecto
Este proyecto analiza un dataset transaccional y operativo de 84,063 registros de empresas que utilizan Veeqo, una plataforma de gestión de inventarios y envíos perteneciente a Amazon.

A diferencia de un análisis B2C clásico, este estudio se enfoca en la "salud" de la cuenta del vendedor (Business-to-Business). El objetivo principal es identificar qué factores operativos, tecnológicos y de comportamiento distinguen a los usuarios de alto valor y activos (Live) de aquellos que abandonan la plataforma o se estancan en la fase de prueba (Churn/Trialing).

🛠️ Procesamiento y Limpieza de Datos
El dataset original constaba de 37 variables que requerían un tratamiento profundo para asegurar la calidad del modelo:

Transformación de Tipos: Conversión de fechas a formato fecha (datetime), corrección de IDs y métricas numéricas.

Tratamiento de Nulos (Imputación Lógica): * Eliminación de variables irrecuperables (ej. PHONE con 99% de nulos).

Imputación de canales inactivos con 0.

Creación de variables booleanas de presencia/ausencia para eventos temporales (ej. HAS_FIRST_SHIPMENT_DATE).

Tratamiento de Outliers (Capping Híbrido): Se implementó una estrategia de winsorización para domar valores extremos sin perder clientes corporativos grandes.

Se usó IQR Suavizado (3.0) y Percentil 99 según la distribución de la columna.

Resultado: Solo el 8.73% de los registros fueron ajustados (manteniendo las modificaciones individuales de cada columna por debajo del 5%), logrando el estándar de oro analítico.

🤖 Modelado Predictivo (Machine Learning)
Se estructuró un modelo de Clasificación Binaria enfrentando un fuerte desbalance de clases (83,376 en Trial/Churn vs 687 Activos).

Target (TARGET): 1 (High Value: live, implementation) vs 0 (Churn/Trial: trialing, canceled, etc.).

Feature Selection: Mediante Random Forest, aislando las 26 características más predictivas.

Algoritmos Entrenados: 1. Random Forest Classifier
2. Logistic Regression
3. K-Nearest Neighbors (KNN)
4. XGBoost Classifier

📊 Desempeño de los Modelos
Debido al desbalance de clases, la métrica guía fue ROC-AUC en lugar de Accuracy (el cual rondaba engañosamente el 99%).

La Regresión Logística demostró ser el mejor modelo general, superando a los ensambles en la detección de la clase minoritaria (empresas exitosas).

💡 Conclusión Principal del Negocio
"La adopción de equipo mata a la infraestructura"

Contrario a la intuición en un software logístico (donde se esperaría que la cantidad de almacenes conectados dicte el éxito), todos los modelos coincidieron en un hallazgo fundamental: el predictor más fuerte para la retención es ALL_USERS_COUNT.

Un cliente que ingresa a la plataforma y crea usuarios adicionales para su equipo de trabajo tiene una probabilidad exponencialmente mayor de convertirse en un usuario "High Value" (Live) frente a un usuario solitario, independientemente de la complejidad de sus almacenes (Warehouses/FBA).

🗂️ Diccionario de Datos (Variables Clave)
SUBSCRIPTION_STATUS: Estado de la suscripción (Target base).

ALL_USERS_COUNT: Total de usuarios registrados en la misma cuenta.

INACTIVE_CHANNELS: Cantidad de canales conectados pero inactivos.

ACTIVE_CHANNELS: Cantidad de canales de venta conectados.

DECILE: Clasificación RFM del cliente.

WAREHOUSES / FBA_WAREHOUSES: Almacenes propios vs gestionados por Amazon.

🚀 Instalación (Para reproducir el código)
Si querés correr este análisis en tu máquina:

Cloná este repositorio.

Asegurate de tener instaladas las librerías: pandas, numpy, scikit-learn, xgboost, matplotlib y seaborn.

Ejecutá el script principal o Jupyter Notebook asociado.
