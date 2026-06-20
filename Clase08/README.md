[Material Diapositivas](https://docs.google.com/presentation/d/14g5eAuGgTasx_D4dJ7w7txt2qOZEY7EqvrTVTMddh74/edit#slide=id.p44)

---

# Clase 8: Aprendizaje Supervisado en Práctica

Esta clase cubre el flujo completo de modelos supervisados: desde la preparación de datos hasta la evaluación robusta, pasando por regresión lineal, árboles de decisión, Random Forest, pipelines, imputación y métricas.

---

## Contenido (PDF)

| # | Tema |
|---|------|
| 1 | Laboratorio: regresión lineal |
| 2 | Laboratorio: árboles de decisión y Random Forest |
| 3 | Pipelines en scikit-learn y buenas prácticas |
| 4 | Principios de limpieza e imputación |
| 5 | Laboratorio: imputación y limpieza con pandas y scikit-learn |
| 6 | ¿Qué es el aprendizaje supervisado? |
| 7 | Visión general de modelos supervisados y cuándo usarlos |
| 8 | Métricas en profundidad: clasificación y regresión |
| 9 | Laboratorio: cross-validation y estimación robusta |
| 10 | Visualización para diagnóstico y métricas (matplotlib, seaborn) |
| 11 | Escalado, normalización y codificación de variables |

---

## Notebook (`Unidad_8.ipynb`)

El notebook cubre los siguientes ejemplos prácticos:

### 1. Árbol de Decisión
- Dataset: Iris (sklearn)
- `DecisionTreeClassifier` con visualización via `tree.plot_tree` y `dtreeviz`

### 2. KNN (K-Nearest Neighbors)
- Datasets: ushape, concéntrico, XOR, linealmente separable, outliers
- Función `knn_comparison` para explorar el efecto de distintos valores de `k`
- Visualización de fronteras de decisión con `mlxtend`

### 3. Regresión Logística
- Dataset: Breast Cancer (sklearn)
- `LogisticRegression` + `accuracy_score` + matriz de confusión con seaborn

### 4. Regresión Lineal con statsmodels (datos de bolsa)
- Empresas: D, EXC, NEE, SO, DUK — concatenadas con `pd.concat`
- Escalado min-max manual
- Dos modelos OLS: `VolStat` y `Return` como variables objetivo
- Interpretación de coeficientes y R²

### 5. Aprendizaje Supervisado — Teoría
- Definición y flujo típico: `f(X) → y`
- Clasificación vs Regresión (tabla comparativa)
- Comparación de modelos clásicos: interpretabilidad, no-linealidad, robustez

### 6. Random Forest
- Dataset: Breast Cancer
- Comparación de accuracy: Random Forest vs Árbol de Decisión vs Regresión Logística
- Gráfico de importancia de variables (Gini importance, top 10)

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
top10_idx = np.argsort(importances)[::-1][:10]
```

### 7. Cross-Validation
- `StratifiedKFold` con 5 folds (mantiene proporción de clases)
- `cross_val_score` para Random Forest, Árbol y Regresión Logística
- Gráfico de accuracy por fold con medias superpuestas

```python
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(RandomForestClassifier(), X, y, cv=kfold, scoring='accuracy')
print(f"Media: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## Conceptos clave

### Clasificación vs Regresión

| | Clasificación | Regresión |
|-|:---:|:---:|
| **Variable objetivo** | Categórica discreta | Numérica continua |
| **Ejemplo** | ¿Tumor maligno? | ¿Precio de la vivienda? |
| **Métricas** | Accuracy, F1, AUC-ROC | MAE, RMSE, R² |
| **Modelos** | Logistic Regression, Decision Tree | Linear Regression, Random Forest |

### Métricas de clasificación

| Métrica | Qué mide | Cuándo usarla |
|---------|----------|---------------|
| **Accuracy** | % predicciones correctas | Clases balanceadas |
| **Precision** | De los que predije positivo, ¿cuántos lo son? | Minimizar falsas alarmas |
| **Recall** | De los positivos reales, ¿cuántos detecté? | Minimizar casos perdidos |
| **F1-Score** | Media armónica Precision/Recall | Clases desbalanceadas |
| **AUC-ROC** | Capacidad discriminativa global | Comparar modelos |

### Métricas de regresión

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **MAE** | mean(\|y - ŷ\|) | Error promedio en unidades reales |
| **RMSE** | sqrt(mean((y - ŷ)²)) | Penaliza errores grandes |
| **R²** | 1 - SS_res/SS_tot | Proporción de varianza explicada (0–1) |

### Pipelines y Data Leakage

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])

# La clave: fit solo sobre train, transform aplicado a ambos
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
```

> **Data Leakage**: ocurre cuando el preprocesamiento (escalado, imputación) se ajusta con datos de test.
> Usar `Pipeline` garantiza que cada fold de cross-validation sea independiente.

### Estrategias de imputación

| Método | Ventaja | Limitación |
|--------|---------|------------|
| `SimpleImputer(media/moda)` | Rápido, baseline | Distorsiona varianza |
| `KNNImputer` | Aprovecha correlaciones | Costoso computacionalmente |
| `IterativeImputer` (MICE) | Más preciso | Mayor complejidad |

### Random Forest vs Árbol de Decisión

| | Árbol | Random Forest |
|-|:---:|:---:|
| **Overfitting** | Alto riesgo | Reducido (bagging) |
| **Estabilidad** | Baja | Alta |
| **Interpretabilidad** | Alta | Media |
| **Feature importance** | Sí | Sí (más confiable) |

### Cross-Validation

```
Fold 1: [TEST ] [train] [train] [train] [train]
Fold 2: [train] [TEST ] [train] [train] [train]
Fold 3: [train] [train] [TEST ] [train] [train]
...
```

- Usar `StratifiedKFold` en clasificación para mantener proporción de clases
- Reportar `mean ± std` de las K métricas, no solo el promedio
- Menor varianza entre folds = modelo más estable
