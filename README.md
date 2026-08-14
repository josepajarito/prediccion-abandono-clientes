# Predicción de Abandono de Clientes (Churn) — Machine Learning End-to-End

Segundo proyecto del portafolio de Data Science. Continuación directa del
**Proyecto 1** (procesamiento a gran escala con PySpark): aquí se convierten
las transacciones en una tabla de clientes con variables **RFM** (Recencia,
Frecuencia, Valor Monetario) y se entrena un modelo de clasificación para
predecir abandono.

## Contexto del problema

Con 300k clientes activos, no todos van a seguir comprando. El objetivo es
predecir, con anticipación, qué clientes tienen mayor probabilidad de dejar
de comprar (**churn**), para que el equipo de marketing pueda enfocar
campañas de retención en ellos.

## Sobre la variable objetivo

Los datos transaccionales originales (Proyecto 1) son sintéticos y sin un
patrón real de abandono. La etiqueta de `churn` se construyó combinando
recencia, frecuencia y valor monetario con una función logística + ruido
aleatorio — un enfoque transparente y documentado en el notebook, que
permite practicar el flujo completo de ML supervisado con un problema
realista (no perfectamente separable, sin fuga de datos).

## Stack técnico

- **PySpark** — feature engineering a escala (agregación de 5M transacciones a nivel cliente)
- **Pandas** — manipulación de la tabla agregada (~300k clientes)
- **Scikit-learn** — pipelines de preprocesamiento, Regresión Logística, Random Forest
- **Matplotlib / Seaborn** — EDA y visualización de resultados
- **Joblib** — serialización del modelo

## Qué se practica en este proyecto

- Feature engineering a escala con Spark (RFM: recencia, frecuencia, valor monetario)
- Construcción de una variable objetivo con criterio de negocio
- EDA orientado a hipótesis (boxplots por clase, matriz de correlación)
- Pipelines de preprocesamiento reproducibles (`ColumnTransformer`, `Pipeline`)
- Entrenamiento y comparación de modelos (Regresión Logística vs Random Forest)
- Evaluación con métricas correctas para clases desbalanceadas (precision, recall, F1, AUC-ROC)
- Interpretación de importancia de variables
- Serialización del modelo para producción (`joblib`)

## Resultados principales

- Tasa de churn: 25%
- Regresión Logística: AUC-ROC = 0.96
- Random Forest: AUC-ROC = 0.95
- Variables más influyentes: **recencia** y **frecuencia de compra**

## Estructura del repositorio

```
proyecto2_churn/
├── churn_prediction_analysis.ipynb   # notebook principal, ya ejecutado
├── modelo_churn_random_forest.pkl    # modelo entrenado serializado
├── eda_boxplots.png
├── eda_correlacion.png
├── matrices_confusion.png
├── curva_roc.png
├── importancia_variables.png
└── README.md
```

## Cómo ejecutarlo

Requiere haber generado antes los datos del Proyecto 1
(`../proyecto1_spark/data/transactions/`).

```bash
pip install pyspark pandas numpy scikit-learn matplotlib seaborn joblib
jupyter notebook churn_prediction_analysis.ipynb
```

## Recomendación de negocio

Priorizar campañas de retención en clientes con recencia alta y frecuencia
baja de compra, antes de que crucen el umbral de abandono: retener a un
cliente existente es más barato que adquirir uno nuevo.

## Próximos pasos del portafolio

El siguiente proyecto explorará **segmentación de clientes con aprendizaje
no supervisado** (clustering) para complementar este modelo de churn con
perfiles de cliente accionables para marketing.
