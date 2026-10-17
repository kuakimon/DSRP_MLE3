# Documentación del proyecto — Detección de fraude (PaySim) con CatBoost

> Proyecto del curso *DSRP – ML Engineering* (`DSRP_MLE3`). Esta documentación describe el flujo completo:
> preprocesamiento → exploración → AutoML → selección de variables y optimización → evaluación en producción → monitoreo.
>
> Todas las cifras provienen de las salidas guardadas en los notebooks (`notebooks/*.ipynb`) y de los
> archivos de `optuna/` y `reports/monitoreo/`. Las rutas son relativas a la raíz del repositorio.

## Índice

1. [Objetivo y alcance](#1-objetivo-y-alcance)
2. [Estructura del repositorio](#2-estructura-del-repositorio)
3. [Entorno y dependencias](#3-entorno-y-dependencias)
4. [Datos](#4-datos)
5. [Notebooks](#5-notebooks)
6. [Módulo `sbp_modelo.py`](#6-módulo-sbp_modelopy)
7. [Optimización con Optuna y MLflow](#7-optimización-con-optuna-y-mlflow)
8. [Modelo final](#8-modelo-final)
9. [Monitoreo con Evidently](#9-monitoreo-con-evidently)
10. [Cómo reproducir](#10-cómo-reproducir)
11. [Problemas conocidos y deuda técnica](#11-problemas-conocidos-y-deuda-técnica)
12. [Convenciones de Git](#12-convenciones-de-git)

---

## 1. Objetivo y alcance

Construir un clasificador binario que identifique transacciones fraudulentas (`isFraud = 1`) en el log
sintético **PaySim**, validarlo fuera de tiempo (OOT) y dejar un pipeline de monitoreo reproducible.

- **Problema:** clasificación binaria con desbalance extremo (≈0.08 % de fraude en desarrollo).
- **Métrica de optimización principal:** PR-AUC (Estudios 1–2) y F1 (Estudios 3–4). La exactitud (accuracy) no es informativa con este desbalance.
- **Métricas reportadas:** AUC, PR-AUC, Accuracy, F1, Precision, Recall, Gini (`2·AUC − 1`) y KS (`max(TPR − FPR)`).

## 2. Estructura del repositorio

```
.
├── README.md                    # Resumen del proyecto y scorecard del modelo
├── pyproject.toml / uv.lock     # Dependencias (gestionadas con uv, Python 3.11)
├── requirements.txt             # Lista corta heredada (ver §11)
├── data/                        # raw/ processed/ logs/  (ignorados por git; solo .gitkeep)
├── models/                      # .pkl de modelos (ignorado por git; solo .gitkeep)
├── mlruns/                      # artefactos MLflow (ignorado por git)
├── mlflow.db                    # backend SQLite de MLflow
├── optuna/                      # estudios (.db) y resultados (.csv)
├── notebooks/
│   ├── 00.Preprocessing.ipynb
│   ├── 01.Exploracion.ipynb
│   ├── 02.AutoML.ipynb
│   ├── 03.Optimizacion.ipynb
│   ├── 04.Implementacion.ipynb
│   ├── sbp_modelo.py            # funciones de apoyo (EDA, métricas, PSI, RFE)
│   └── eda/                     # 01.Analisis_Exploratorio.xlsx, 02.Analisis_Bivariado.pdf
├── src/monitoreo_evidently.py   # pipeline crudo → score → PSI → reportes Evidently
├── reports/monitoreo/           # salidas del monitoreo
└── docs/                        # esta documentación, estrategia de git, imágenes
```

## 3. Entorno y dependencias

- **Python 3.11** (`requires-python = ">=3.11,<3.12"`), gestor **uv**, kernel de Jupyter `dsrp3`.
- Versiones fijadas por compatibilidad (cada una resolvió un error real durante el desarrollo):

| Paquete | Versión | Motivo |
|---|---|---|
| `pycaret` | `==3.3.2` | AutoML (3.x requiere Python < 3.12) |
| `mlflow` | `==2.8.0` | versiones más nuevas rompen con pycaret (`ThreadLocalVariable`) |
| `joblib` | `==1.3.2` | compatibilidad con pycaret (`_format_load_msg`) |
| `scikit-learn` | `>=1.4.1,<1.5` | pycaret usa APIs internas retiradas en versiones nuevas |
| `setuptools` | `<81` | `pkg_resources` requerido por `dython` |
| `catboost`, `optuna`, `optuna-dashboard`, `evidently`, `dython`, `shap`, `lightgbm`, `xgboost` | ver `pyproject.toml` | modelado, optimización, monitoreo y EDA |

```bash
uv sync
uv run python -m ipykernel install --user --name dsrp3
```

## 4. Datos

- **Fuente:** PaySim, archivo `PS_20174392719_1491204439457_log.csv` (6,362,620 filas × 11 columnas). No se versiona.
- **Target:** `isFraud`.

### 4.1 Diccionario de variables

| Variable | Tipo | Descripción |
|---|---|---|
| `step` | int | Unidad de tiempo (1 paso = 1 hora). Se usa para el corte temporal. |
| `type` | cat | `PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEBIT`, `CASH_IN` |
| `amount` | num | Monto de la transacción |
| `nameOrig` / `nameDest` | str | Cliente que origina / destinatario (prefijo `C` cliente, `M` comercio) |
| `oldbalanceOrg`, `newbalanceOrig` | num | Saldo del originador antes / después |
| `oldbalanceDest`, `newbalanceDest` | num | Saldo del destinatario antes / después |
| `isFlaggedFraud` | 0/1 | Bandera del sistema de reglas original |
| `isFraud` | 0/1 | **Target** |

### 4.2 Variables construidas (notebook 00)

| Variable | Definición |
|---|---|
| `entTrxType` | Primera letra de `nameOrig` + primera de `nameDest` (`CM` / `CC`). Se descarta en la exploración (ver §5.2). |
| `origTrxAmount` | `newbalanceOrig − oldbalanceOrg` |
| `destTrxAmount` | `newbalanceDest − oldbalanceDest`; `NaN` cuando `entTrxType == 'CM'` (comercios no reportan saldo) |

### 4.3 Diseño de bases

- Se eliminan las filas con `isFlaggedFraud == 1`. Todas eran fraude, así que dejarlas filtraría el target (quedan 6,362,604 filas).
- Variable `base`: `0` = Desarrollo (`step ≤ 400`), `2` = Producción / OOT (`step > 400`).
- Tasa de fraude: base 0 → **0.000774**; base 2 → **0.006491** (≈8.4× mayor en producción).
- Desarrollo (5,787,026) → **Train 4,629,620 / Test 1,157,406** (80/20 estratificado, semilla 666). Producción: 575,578.
- Exportaciones: `data/raw/train.csv`, `data/raw/prod.csv`, `data/processed/processed.parquet`.

## 5. Notebooks

### 5.1 `00.Preprocessing.ipynb`
Lectura del log crudo, verificación de valores únicos de `isFraud`/`isFlaggedFraud` y `type`, construcción de `entTrxType`, `origTrxAmount`, `destTrxAmount`, filtrado de `isFlaggedFraud == 1`, creación de `base` y exportación de las bases.
**Salidas clave:** forma cruda (6362620, 11); tasa de fraude por base (0.000774 / 0.006491).

### 5.2 `01.Exploracion.ipynb`
- Particiones: `df` (6362604, 9), Desarrollo (5787026, 9), Train (4629620, 9), Test (1157406, 9), Producción (575578, 9).
- EDA univariado y bivariado con `sbp.eda_vf2` (≈2 min 8 s) → `notebooks/eda/01.Analisis_Exploratorio.xlsx` y `02.Analisis_Bivariado.pdf`.
- Ninguna variable supera 95 % de nulos. Tabla de percentiles de outliers (p.ej. 463 registros sobre p99.99 en `amount`); boxplots en la figura siguiente.
- Asociaciones (Cramér's V, Spearman, razón de correlación con `dython`): `type` y `entTrxType` correlacionan 1.00 entre sí y `type` tiene mayor asociación con `isFraud` (0.05 vs 0.02), por eso se descarta `entTrxType`.
- Exporta `train.parquet` (4629620, 8) y `test.parquet` (1157406, 8).

![Boxplots de outliers](img/boxplot_outliers.png)
![Asociaciones](img/asociaciones.png)

### 5.3 `02.AutoML.ipynb`
PyCaret 3.3.2: `normalize=zscore`, `numeric_imputation=0`, `fold=5` (StratifiedKFold), `session_id=666`, logging a MLflow y métrica PR-AUC personalizada; `compare_models(sort='F1')`.
La tabla de setup indica 5 variables numéricas + 1 categórica y 33.7 % de filas con algún nulo.

| Modelo | Acc | AUC | Recall | Prec | F1 | PR-AUC | TT (s) |
|---|---|---|---|---|---|---|---|
| **CatBoost** | 0.9997 | 0.9985 | 0.7277 | 0.9273 | **0.8150** | **0.8636** | 193.3 |
| KNN | – | – | – | – | 0.7896 | 0.7198 | 546.5 |
| Decision Tree | – | – | – | – | 0.7806 | 0.6099 | – |
| XGBoost | – | – | – | – | 0.6029 | 0.4865 | – |
| Regresión logística | – | – | – | – | 0.5403 | 0.5317 | – |
| Gradient Boosting | – | – | – | – | 0.2261 | 0.1327 | – |

(Solo se muestran las columnas disponibles en el resumen; Random Forest y LightGBM no aparecen en la tabla final.)
Ganador: `CatBoostClassifier(border_count=254, random_state=666)`.

### 5.4 `03.Optimizacion.ipynb`
1. **RFE con CatBoost** (experimento MLflow `rfe_catboost`, 1 h 55 min 31 s). Métricas de validación:

| Variables | Val F1 | Val PR-AUC | Recall | Comentario |
|---|---|---|---|---|
| 1 | – | 0.0092 | – | |
| 2 | 0.5548 | 0.6300 | – | |
| 3 | 0.5776 | 0.6869 | – | |
| 4 | 0.7912 | 0.8543 | 0.7011 | |
| **5** | **0.7908** | **0.8356** | **0.7179** | Precision 0.8801, KS 0.9841, Gini 0.9952, AUC 0.9976 |

   Decisión: se conservan **5 variables** (ver §8 por la lista exacta usada en los modelos finales, que incluye `type`).
2. **Optuna** — cuatro estudios (§7).
3. Se guardan tres modelos: `prauc_best_model.pkl`, `f1prauc_best_model.pkl` y `f1_best_model.pkl`.

### 5.5 `04.Implementacion.ipynb`
Carga `processed.parquet`, toma `base == 2` (producción), rellena nulos con 0 y evalúa los tres modelos con `sbp.metricas_clasificacion_binaria(..., umbral=0.5)`.

| Modelo | AUC* | PR-AUC* | F1 | Precision | Recall | KS* |
|---|---|---|---|---|---|---|
| `f1prauc_best_model.pkl` | 0.7039 | 0.4115 | 0.5793 | 0.9993 | 0.4079 | 0.4079 |
| `prauc_best_model.pkl` | 0.6779 | 0.3600 | 0.5249 | 1.0000 | 0.3558 | 0.3558 |
| **`f1_best_model.pkl`** | **0.7493** | **0.4990** | **0.6642** | 0.9941 | 0.4987 | 0.4986 |

\* En este notebook se pasa a la función la predicción **binaria** (`model.predict`), no la probabilidad; por eso AUC, PR-AUC, Gini y KS quedan subestimados. Las métricas basadas en probabilidad están en el monitoreo (§9).
Conclusión del notebook: `f1_best_model.pkl` es el mejor; podría mejorar con más trials.

## 6. Módulo `sbp_modelo.py`

| Grupo | Funciones |
|---|---|
| EDA | `eda_vf2`, `eda_bivariado`, `descripcion_variable`, `bin_variable`, `bin_variable_categorica`, `plot_null_percentage`, `porcentaje_categorias`, `box_plot_outliers`, `obtener_var_constantes`, `correlacion` |
| Métricas | `metricas_clasificacion`, `metricas_clasificacion_binaria` (AUC, PR-AUC, Accuracy, F1, Precision, Recall, Gini, KS) |
| PSI | `calcular_casos_x_deciles`, `calcular_psi_x_intervalo`, `obtener_psi_variables`, `graficar_psi_variables` |
| Selección de variables | `eliminacion_recursiva_de_variables` (RFE con registro opcional en MLflow) |
| Utilidades | `MedirTiempoEjecucion` |
| Heredadas (regresión) | `metricas_regresion*`, `calcular_coincidencia*`, `calcular_precision_x_*`, `Arbol_Regresion`, `importancias_lgbm*` — de otro proyecto; no se usan en el flujo de fraude |

## 7. Optimización con Optuna y MLflow

- Modelo: `CatBoostClassifier`, validación cruzada estratificada con **MedianPruner** y sampler **TPE**; almacenamiento SQLite en `optuna/*.db`; semilla 666.
- Cada trial registra en MLflow (`mlruns/`) AUC, PR-AUC, Accuracy, F1, Precision, Recall, Gini y KS.
- Dashboards: `optuna-dashboard` apuntando a la carpeta `optuna/`.

| Estudio | Objetivo | Experimento MLflow | Resultado |
|---|---|---|---|
| 1 | PR-AUC (CV), 50 trials | `optuna_catboost_fraude` | mejor trial 40: **0.8690** |
| 2 | PR-AUC, rangos acotados, `cat_features=['type']` | `optuna_catboost_fraude2` | 75 trials; mejor trial 53: **0.8695** |
| 3 | F1, rangos amplios | `optuna_catboost_fraude_f1` | interrumpido; mejor parcial F1 ≈ 0.81 (trial 15: 0.8105) |
| 4 | F1, rangos acotados | `optuna_catboost_fraude_f1_2` | mejor trial 57: **F1 CV 0.8129**, umbral 0.50 |

Nota: en el Estudio 2 el mejor PR-AUC usa `scale_pos_weight ≈ 27`, lo que desplaza las probabilidades (F1 CV 0.628 con umbral 0.5, Recall 0.885, Precision 0.487). Un `scale_pos_weight` alto exige recalibrar el umbral.

Resultados guardados: `optuna/optuna_results.csv` (Estudios 1–2) y `optuna/optuna_results_f1.csv` (Estudio 4; 61 filas: 35 completos, 22 podados, 3 fallidos, 1 en ejecución al guardar).

## 8. Modelo final

**`models/f1_best_model.pkl`** — Estudio 4, trial 57.

- Variables: `type` (categórica nativa de CatBoost) y numéricas `amount`, `oldbalanceOrg`, `origTrxAmount`, `oldbalanceDest`, `destTrxAmount`. Los nulos de `destTrxAmount` se rellenan con 0 en el notebook 04.
- Hiperparámetros:

```python
{'depth': 6, 'iterations': 801, 'l2_leaf_reg': 18.342, 'learning_rate': 0.1172,
 'max_ctr_complexity': 2, 'one_hot_max_size': 2, 'random_strength': 1.7036,
 'rsm': 0.9773, 'scale_pos_weight': 1.0347, 'subsample': 0.5872,
 'bootstrap_type': 'Bernoulli', 'border_count': 128, 'loss_function': 'Logloss',
 'random_seed': 666, 'thread_count': -1}
```
- Umbral de decisión: 0.50.
- Métricas CV (5 folds, Estudio 4): AUC 0.9984, PR-AUC 0.832, Accuracy 0.9998, F1 0.8129, Precision 0.9664, Recall 0.7016, Gini 0.9968, KS 0.9784.

## 9. Monitoreo con Evidently

`src/monitoreo_evidently.py` ejecuta: lectura de datos crudos → `transformar_crudo` (mismas variables que el notebook 00) → `predict_proba` con el `.pkl` → PSI por variable → reportes Evidently (calidad/drift con `DataSummaryPreset` + `DataDriftPreset`, y desempeño con `ClassificationPreset`).
Configuración en el diccionario `CONFIG` (modelo, bases, umbral, columnas, carpeta de salida).

```bash
python src/monitoreo_evidently.py
```

Salidas en `reports/monitoreo/`: `psi_variables.csv`, `resumen_modelo.csv`, `reporte_calidad_drift.html/.json`, `reporte_desempeno.html`.

Resultados: ver el scorecard del [README](../README.md#scorecard-del-modelo). Conclusiones del monitoreo:
- PSI de todas las variables y del score < 0.01 → **sin drift de población** de entrada.
- Evidently marca 1 de 9 columnas con drift (`prediction_proba`, Wasserstein normada 0.126 > umbral 0.1), coherente con la mayor prevalencia de fraude en producción.
- 881 filas duplicadas y 35.18 % de nulos en `destTrxAmount` (los pagos a comercios), igual que en entrenamiento (33.68 %).

## 10. Cómo reproducir

1. Colocar `PS_20174392719_1491204439457_log.csv` en `data/raw/`.
2. `uv sync` y registrar el kernel `dsrp3`.
3. Ejecutar en orden `notebooks/00` → `04`. Los notebooks esperan las carpetas `data/`, `models/`, `mlruns/`, `optuna/` (existen con `.gitkeep`).
4. Interfaz MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db`. Dashboard Optuna: `optuna-dashboard sqlite:///optuna/estudio_catboost_fraude2.db`.
5. Monitoreo: `python src/monitoreo_evidently.py` desde la raíz.

## 11. Problemas conocidos y deuda técnica

- **`save_model` de PyCaret** sin nombre genera `models/.pkl` (el notebook 02 imprime `'../models/.pkl'`). Pasar un nombre explícito.
- **Discrepancia de scoring:** el monitoreo da PR-AUC 0.498 en la base de desarrollo, por debajo del 0.83 de CV. El script deja los `NaN` de `destTrxAmount` y convierte `type` a `category`, mientras que el notebook 04 rellena con 0. CatBoost trata los `NaN` de forma distinta a 0, y la base de desarrollo incluye datos de entrenamiento. Conviene unificar la ruta de scoring antes de comparar.
- El notebook 04 evalúa con predicciones binarias (ver §5.5).
- `docs/model_card_template.md` pertenece a un proyecto anterior (Fake News) y debe reescribirse; el README anterior también (queda en el historial de git).
- `requirements.txt` está desactualizado respecto a `pyproject.toml` (usar `uv sync`).
- El notebook 03 contiene funciones heredadas de LightGBM-regresión sin uso.
- `.gitignore` tiene el typo `!smodels/.gitkeep` (debería ser `!models/.gitkeep`).
- `catboost_info/` y `logs.log` están versionados; deberían ignorarse.

## 12. Convenciones de Git

Ver [`git_strategy.md`](git_strategy.md): GitHub Flow con `main` (estable) y `development`, ramas `feature/*`, merge commits y versionado semántico.
