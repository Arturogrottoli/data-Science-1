"""
----------------------------------------------------------
📘 CURSO DE CIENCIA DE DATOS - REPASO GENERAL
----------------------------------------------------------

Clase 7 - Aprendizaje Supervisado
Clase 8 - Aprendizaje No Supervisado
----------------------------------------------------------
"""

# ==========================================================
# 🧠 CLASE 7 - APRENDIZAJE SUPERVISADO
# ==========================================================

"""
En el aprendizaje supervisado, el modelo aprende a partir de datos etiquetados,
es decir, ejemplos donde conocemos tanto las variables de entrada (X)
como las de salida (y).

📌 Objetivo: predecir la salida para nuevos datos.

Ejemplos comunes:
- Clasificación (spam / no spam, enfermedad / no enfermedad)
- Regresión (predicción de precios, demanda, temperatura)

"""

# Ejemplo simple de clasificación con árboles de decisión
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset de flores Iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test)

print("🌸 Precisión Árbol de Decisión:", accuracy_score(y_test, predicciones))


# ==========================================================
# 🤖 CLASE 8 - APRENDIZAJE NO SUPERVISADO
# ==========================================================

"""
En el aprendizaje no supervisado no hay etiquetas o resultados conocidos.
El objetivo es **descubrir patrones o estructuras ocultas** en los datos.

Ejemplos:
- Agrupar clientes por comportamiento
- Reducir la cantidad de variables
- Encontrar asociaciones entre productos
"""

# ----------------------------------------------------------
# 8.1 Introducción al Aprendizaje No Supervisado
# ----------------------------------------------------------
"""
El modelo explora los datos sin una variable objetivo.
Busca similitudes, relaciones o patrones automáticamente.
"""

# ----------------------------------------------------------
# 8.2 Algoritmos de Clustering
# ----------------------------------------------------------
"""
📍 El clustering agrupa observaciones similares entre sí.

El algoritmo más popular: K-Means.
Divide los datos en K grupos, minimizando la distancia dentro de cada grupo.
"""

from sklearn.cluster import KMeans
import numpy as np

# Datos simulados (clientes con ingresos y gastos)
X = np.array([[15, 39], [16, 81], [17, 6], [18, 77], [19, 40],
              [20, 76], [25, 6], [40, 77], [42, 40], [50, 76]])

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

print("🏷️ Etiquetas de los grupos:", kmeans.labels_)
print("📍 Centros de los clusters:\n", kmeans.cluster_centers_)

# ----------------------------------------------------------
# 8.3 Reducción de Dimensionalidad
# ----------------------------------------------------------
"""
📉 Objetivo: simplificar los datos reduciendo el número de variables.

Ejemplo común: PCA (Análisis de Componentes Principales)
Permite visualizar datos complejos en 2D o 3D.
"""

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reducido = pca.fit_transform(X)
print("🔻 Datos reducidos a 2 dimensiones:\n", X_reducido[:5])

# ----------------------------------------------------------
# 8.4 Reglas de Asociación
# ----------------------------------------------------------
"""
💡 Buscan relaciones entre ítems en transacciones (como en un supermercado).
Ejemplo: “Si compra pan y manteca, probablemente compre leche”.

Algoritmos: Apriori, FP-Growth
"""

# Ejemplo teórico (sin ejecutar):
# Transacciones = [
#     ["pan", "manteca", "leche"],
#     ["pan", "manteca"],
#     ["leche", "galletitas"],
#     ["pan", "galletitas", "leche"],
# ]

# ----------------------------------------------------------
# 8.5 Evaluación y Comparación
# ----------------------------------------------------------
"""
A diferencia del aprendizaje supervisado, acá no hay etiquetas verdaderas.
Se usan métricas como:
- Inercia (K-Means)
- Silhouette Score (cohesión y separación de grupos)
"""

from sklearn.metrics import silhouette_score
sil = silhouette_score(X, kmeans.labels_)
print("✨ Silhouette Score:", round(sil, 3))

# ----------------------------------------------------------
# 8.6 Actividad práctica
# ----------------------------------------------------------
"""
👉 Agrupar datos de clientes según variables de comportamiento
   (ejemplo: frecuencia de compra, gasto promedio, visitas mensuales).
"""

# ----------------------------------------------------------
# 8.7 Recursos complementarios
# ----------------------------------------------------------
"""
- Documentación oficial de Scikit-learn: https://scikit-learn.org/stable/
- Libro: “Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow”
- Dataset recomendado: Mall Customers (Kaggle)
"""

# ----------------------------------------------------------
# 8.8 Glosario
# ----------------------------------------------------------
"""
Cluster: grupo de elementos similares.
Centroide: punto medio de un cluster.
Inercia: medida de qué tan cerca están los puntos de su centroide.
Silhouette: evalúa la separación entre grupos.
PCA: técnica de reducción de dimensiones.
"""

# ----------------------------------------------------------
# 8.9 Agrupación y Segmentación con IA
# ----------------------------------------------------------
"""
Los algoritmos de clustering permiten segmentar clientes, productos o usuarios
de manera automática, ayudando a personalizar estrategias de marketing,
ofertas o análisis de comportamiento.

🔍 Ejemplo de uso real:
Segmentar clientes según sus hábitos de compra para campañas publicitarias.
"""

# ==========================================================
# ✅ FIN DEL REPASO CLASES 7 y 8
# ==========================================================



"""
## Clase 9.1: Introducción a la Inteligencia Artificial

### Diapositivas
https://docs.google.com/presentation/d/1zmIN5N8NCY3Z9f_dm-tkEm3vj387oHs4Jwb2mtpCce8/edit#slide=id.p1

### Objetivos de la clase
- Comprender qué es la Inteligencia Artificial (IA).
- Conocer la historia y evolución de la IA.
- Identificar las principales clasificaciones de IA.
- Distinguir entre los distintos tipos de IA existentes.
- Experimentar con ejemplos prácticos de IA.

---

### 1. Introducción y disparador
**Pregunta inicial para los estudiantes**:
- ¿Cuántas veces interactuaron hoy con alguna forma de inteligencia artificial sin notarlo?

**Objetivo**: generar participación y conectar con la vida cotidiana.
Ejemplos posibles:
- Recomendaciones en plataformas como Netflix o Spotify
- Autocompletado en el celular
- Chatbots de atención al cliente
- Asistentes de voz como Siri, Alexa o Google Assistant
- Reconocimiento de imágenes en redes sociales (Facebook, Instagram)
- Filtros automáticos de spam en el correo electrónico
- Traducción automática (Google Translate, DeepL)

---

### 2. ¿Qué es la Inteligencia Artificial?
La inteligencia artificial (IA) se define como la capacidad de las máquinas para aprender a partir de datos y tomar decisiones o ejecutar tareas sin intervención humana directa, de manera similar a un ser humano.

**Subcampos principales:**
- Machine Learning (Aprendizaje Automático)
- Deep Learning (Aprendizaje Profundo, con redes neuronales)

**Diferencias respecto a los humanos:**
- No requieren descanso
- Pueden procesar grandes volúmenes de información rápidamente
- Menor tasa de error en tareas repetitivas

**Ejemplo visual:**  
- Entrada de datos → Entrenamiento del modelo → Predicción  
- Ejemplo: recomendación de películas según historial del usuario.

---

### 3. Historia de la IA
Presentar a través de una línea de tiempo o storytelling:

| Año  | Evento                                                          |
| ---- | --------------------------------------------------------------- |
| 1943 | McCulloch y Pitts: primer modelo de una neurona artificial      |
| 1949 | Donald Hebb y su teoría del aprendizaje sináptico               |
| 1950 | Alan Turing propone el Test de Turing                           |
| 1956 | John McCarthy acuña el término "Inteligencia Artificial"        |
| 1966 | Creación de ELIZA en el MIT (procesamiento de lenguaje natural) |
| 1997 | Deep Blue vence a Garri Kaspárov en ajedrez                     |
| 2011 | Watson (IBM) gana en Jeopardy!                                  |
| 2016 | AlphaGo (DeepMind) vence al campeón mundial de Go               |

---

### 4. Clasificación de la IA según Arend Hintze
**Cuatro tipos de IA según Hintze:**
1. Máquinas Reactivas: no usan memoria ni experiencia previa. Ejemplo: Deep Blue.
2. Memoria Limitada: usan datos recientes para tomar decisiones. Ejemplo: autos autónomos.
3. Teoría de la Mente: comprenden emociones y estados mentales. En etapa experimental.
4. Autoconciencia: IA que se reconoce a sí misma. Aún no alcanzada.

Es útil asociar cada nivel con ejemplos concretos o contextos para facilitar la comprensión.

---

### 5. Tipos de IA: Débil vs Fuerte

| Tipo      | Características                                    | Ejemplos                                                 |
| --------- | -------------------------------------------------- | -------------------------------------------------------- |
| IA Débil  | Diseñada para tareas específicas                   | Chatbots, motores de recomendación, asistentes virtuales |
| IA Fuerte | Capaz de realizar cualquier tarea cognitiva humana | Hipotética. Aún no desarrollada.                         |

**Reflexión sugerida para los estudiantes:**
- ¿Creen que llegaremos a desarrollar una IA fuerte? ¿Qué implicancias éticas tendría?  

**Discusión rápida:**
- Sesgos en algoritmos, privacidad, autonomía y empleo.

---

### 6. Miniactividad práctica
**Objetivo:** experimentar con IA de manera tangible.
1. Entrar a Teachable Machine: https://teachablemachine.withgoogle.com/
2. Crear un modelo sencillo de clasificación de imágenes o sonidos
3. Probar cómo el modelo aprende y predice nuevas entradas
4. Compartir la experiencia con el resto de la clase

---

### Recursos adicionales

**Videos:**
- ¿Qué es la inteligencia artificial? – IBM: https://www.youtube.com/watch?v=2ePf9rue1Ao
- AlphaGo - El documental completo (DeepMind): https://www.youtube.com/watch?v=WXuK6gekU1Y

**Artículos y lecturas recomendadas:**
- "A Brief History of AI" – Medium o Towards Data Science
- Entrada “Artificial Intelligence” en la Stanford Encyclopedia of Philosophy: https://plato.stanford.edu/entries/artificial-intelligence/

**Libros recomendados:**
- “Life 3.0: Being Human in the Age of Artificial Intelligence” – Max Tegmark
- “Inteligencia Artificial” – Stuart Russell y Peter Norvig (lectura técnica avanzada)

**Herramientas para exploración:**
- Teachable Machine de Google: https://teachablemachine.withgoogle.com/
- Experimentos con IA en Google AI Experiments: https://experiments.withgoogle.com/collection/ai
"""
