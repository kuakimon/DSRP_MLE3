from __future__ import annotations
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURACIÓN  
# =============================================================================
CONFIG = dict(
    ruta_modelo = "models/f1_best_model.pkl",
    ruta_train = "data/raw/train.csv",
    ruta_prod = "data/raw/prod.csv",
    ruta_ref_desempeno = None, # opcional: validación/test CON target para medir rendimiento. None -> se usa train
    target = "isFraud",
    umbral = 0.5,
    columnas_modelo = ['type','amount','oldbalanceOrg','origTrxAmount','oldbalanceDest','destTrxAmount'],
    columnas_cat = ['type'],
    relleno_numerico = None,
    relleno_categorico = "Missing",
    carpeta_salida = "reports/monitoreo",
    muestra_max_drift = None, # 300_000, # filas por base para el reporte de drift
    guardar_scoring = False, # True -> guarda train/prod con score en parquet
    semilla = 666,
)

# =============================================================================
# PASO 1. DATA CRUDA -> FEATURES DEL MODELO
# =============================================================================
def leer_datos(ruta) -> pd.DataFrame:
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo: {ruta.resolve()}")
    ext = ruta.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(ruta)
    if ext in (".csv", ".txt"):
        return pd.read_csv(ruta)
    if ext in (".pkl", ".pickle"):
        return pd.read_pickle(ruta)
    raise ValueError(f"Formato no soportado: {ext} (usa .parquet, .csv o .pkl)")


def transformar_crudo(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    
    df['entTrxType'] = df['nameOrig'].str[0] + df['nameDest'].str[0]
    df['origTrxAmount'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['destTrxAmount'] = np.where(df['entTrxType'] == "CM", np.nan, df['newbalanceDest'] - df['oldbalanceDest'])

    lista_var_nominales = ['type']
    for i in lista_var_nominales:
        df[i] = df[i].fillna('Missing')
        df[i] = df[i].astype('category')

    df = df.drop(columns={'entTrxType'})
    
    return df


def cargar_modelo(ruta):
    try:
        return joblib.load(ruta)
    except Exception as e1:
        import pickle
        try:
            with open(ruta, "rb") as f:
                return pickle.load(f)
        except Exception as e2:
            raise RuntimeError(
                f"No se pudo cargar {ruta}. joblib: {e1} | pickle: {e2}. "
                "Usa las mismas versiones de Python, scikit-learn y catboost con las que lo guardaste."
            )


def columnas_del_modelo(modelo, cfg) -> list[str]:
    if cfg["columnas_modelo"]:
        return list(cfg["columnas_modelo"])
    for attr in ("feature_names_", "feature_names_in_"):
        try:
            v = getattr(modelo, attr, None)
        except Exception:
            v = None
        if v is not None and len(v) > 0:
            return [str(c) for c in v]
    raise ValueError("El modelo no informa sus columnas. Define CONFIG['columnas_modelo'] en el orden del entrenamiento.")


def columnas_categoricas(modelo, cols, df, cfg) -> list[str]:
    if cfg["columnas_cat"] is not None:
        return [c for c in cfg["columnas_cat"] if c in cols]
    if hasattr(modelo, "get_cat_feature_indices"):
        try:
            return [cols[i] for i in modelo.get_cat_feature_indices()]
        except Exception:
            pass
    return [c for c in cols if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]


def preparar_X(df: pd.DataFrame, cols, cat_cols, cfg) -> pd.DataFrame:
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas que el modelo necesita: {faltan}")
    X = df[cols].copy()
    for c in cols:
        s = X[c]
        if c in cat_cols:
            X[c] = s.astype("object").where(s.notna(), cfg["relleno_categorico"]).astype(str)
        else:
            X[c] = pd.to_numeric(s, errors="coerce").astype("float64")
            if cfg["relleno_numerico"] is not None:
                X[c] = X[c].fillna(cfg["relleno_numerico"])
    return X


# =============================================================================
# PASO 2. PREDICCIÓN
# =============================================================================
def puntuar(modelo, df_listo: pd.DataFrame, cols, cat_cols, cfg, nombre="datos", lote=200_000) -> pd.DataFrame:
    """Devuelve features + prediction_proba + prediction (+ target si existe)."""
    X = preparar_X(df_listo, cols, cat_cols, cfg)
    probas = np.empty(len(X), dtype="float64")
    for i in range(0, len(X), lote):
        # Se asume que la columna 1 de predict_proba es la clase positiva (fraude)
        probas[i:i + lote] = modelo.predict_proba(X.iloc[i:i + lote])[:, 1]
    out = X.copy()
    out["prediction_proba"] = probas
    out["prediction"] = (probas >= cfg["umbral"]).astype(int)
    t = cfg["target"]
    if t in df_listo.columns:
        out[t] = pd.to_numeric(df_listo[t], errors="coerce").to_numpy()
    print(f"[{nombre}] {len(out):,} filas puntuadas | score medio {probas.mean():.5f} | "
          f"alertas {out['prediction'].mean():.3%}")
    return out


# =============================================================================
# PASO 3. PSI Y MÉTRICAS PROPIAS
# =============================================================================
def _psi(p_ref, p_cur, eps=1e-6) -> float:
    p_ref = np.clip(np.asarray(p_ref, dtype=float), eps, None)
    p_cur = np.clip(np.asarray(p_cur, dtype=float), eps, None)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def psi_numerica(ref: pd.Series, cur: pd.Series, bins=10) -> float:
    ref = pd.to_numeric(ref, errors="coerce")
    cur = pd.to_numeric(cur, errors="coerce")
    r, c = ref.dropna().to_numpy(), cur.dropna().to_numpy()
    if len(r) == 0 or len(c) == 0:
        return np.nan
    cortes = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
    if len(cortes) >= 2:
        edges = np.concatenate([[-np.inf], cortes[1:-1], [np.inf]])
    else:
        edges = np.array([-np.inf, cortes[0], np.inf])
    cnt_r, _ = np.histogram(r, bins=edges)
    cnt_c, _ = np.histogram(c, bins=edges)
    cnt_r = np.append(cnt_r, ref.isna().sum()) # los nulos cuentan como un tramo más
    cnt_c = np.append(cnt_c, cur.isna().sum())
    return _psi(cnt_r / cnt_r.sum(), cnt_c / cnt_c.sum())


def psi_categorica(ref: pd.Series, cur: pd.Series) -> float:
    r = ref.astype(str).value_counts(normalize=True)
    c = cur.astype(str).value_counts(normalize=True)
    cats = r.index.union(c.index)
    return _psi(r.reindex(cats, fill_value=0).to_numpy(), c.reindex(cats, fill_value=0).to_numpy())


def _estado_psi(v: float) -> str:
    if not np.isfinite(v):
        return "sin dato"
    return "estable" if v < 0.1 else ("alerta" if v < 0.25 else "cambio")


def tabla_psi(ref: pd.DataFrame, cur: pd.DataFrame, num_cols, cat_cols) -> pd.DataFrame:
    filas = []
    for c in num_cols + ["prediction_proba"]:
        v = psi_numerica(ref[c], cur[c])
        filas.append(dict(variable=c, tipo="score" if c == "prediction_proba" else "numérica", psi=v,
                          estado=_estado_psi(v),
                          nulos_ref_pct=ref[c].isna().mean() * 100, nulos_cur_pct=cur[c].isna().mean() * 100,
                          media_ref=ref[c].mean(), media_cur=cur[c].mean(), categorias_nuevas=np.nan))
    for c in cat_cols:
        v = psi_categorica(ref[c], cur[c])
        nuevas = len(set(cur[c].astype(str).unique()) - set(ref[c].astype(str).unique()))
        filas.append(dict(variable=c, tipo="categórica", psi=v, estado=_estado_psi(v),
                          nulos_ref_pct=np.nan, nulos_cur_pct=np.nan,
                          media_ref=np.nan, media_cur=np.nan, categorias_nuevas=nuevas))
    return pd.DataFrame(filas).sort_values("psi", ascending=False).reset_index(drop=True)


def _seguro(f):
    try:
        return float(f())
    except Exception:
        return np.nan


def _ks(metrics, y, p) -> float:
    fpr, tpr, _ = metrics.roc_curve(y, p)
    return np.max(tpr - fpr)


def resumen_modelo(bases: dict, cfg) -> pd.DataFrame:
    from sklearn import metrics
    t, u = cfg["target"], cfg["umbral"]
    filas = []
    for nombre, df in bases.items():
        p = df["prediction_proba"].to_numpy()
        fila = dict(base=nombre, filas=len(df), score_medio=p.mean(), score_p95=np.quantile(p, 0.95),
                    tasa_alertas=(p >= u).mean())
        if t in df.columns and df[t].notna().any():
            d = df[df[t].notna()]
            y, pp = d[t].astype(int).to_numpy(), d["prediction_proba"].to_numpy()
            yh = (pp >= u).astype(int)
            prev = y.mean()
            pr = _seguro(lambda: metrics.average_precision_score(y, pp))
            auc = _seguro(lambda: metrics.roc_auc_score(y, pp))
            fila.update(positivos=int(y.sum()), prevalencia=prev, PR_AUC=pr,
                        lift_vs_azar=pr / prev if prev > 0 else np.nan, AUC=auc, Gini=2 * auc - 1,
                        KS=_seguro(lambda: _ks(metrics, y, pp)),
                        Precision=_seguro(lambda: metrics.precision_score(y, yh, zero_division=0)),
                        Recall=_seguro(lambda: metrics.recall_score(y, yh, zero_division=0)),
                        F1=_seguro(lambda: metrics.f1_score(y, yh, zero_division=0)))
        filas.append(fila)
    return pd.DataFrame(filas)


# =============================================================================
# PASO 4. REPORTES DE EVIDENTLY
# =============================================================================
def reportes_evidently(ref, cur, num_cols, cat_cols, cfg, ref_desemp=None):
    """Cada reporte va en su propio try/except: si uno falla, el resto de salidas se conserva."""
    try:
        import evidently
        from evidently import Dataset, DataDefinition, Report
        from evidently.presets import DataDriftPreset, DataSummaryPreset
        try:
            from evidently import BinaryClassification
        except ImportError:
            from evidently.core.datasets import BinaryClassification
    except ImportError as e:
        print(f"\n[Evidently] No se pudo importar ({e}). Instala/actualiza con `uv add evidently` "
              "(API actual: Dataset / DataDefinition / Report). Se omiten los reportes HTML.")
        return
    print(f"\n[Evidently] versión {getattr(evidently, '__version__', '?')}")

    out = Path(cfg["carpeta_salida"])
    t, n = cfg["target"], cfg["muestra_max_drift"]
    hay_target = t in ref.columns and t in cur.columns

    def muestra(df):
        return df if (n is None or len(df) <= n) else df.sample(n, random_state=cfg["semilla"])

    # ---- Reporte 1: calidad de datos + data drift (variables, score y predicción) ----
    try:
        cols_num = num_cols + ["prediction_proba"]
        cols_cat = cat_cols + ["prediction"] + ([t] if hay_target else [])
        definicion = DataDefinition(numerical_columns=cols_num, categorical_columns=cols_cat)
        ds_ref = Dataset.from_pandas(muestra(ref)[cols_num + cols_cat], data_definition=definicion)
        ds_cur = Dataset.from_pandas(muestra(cur)[cols_num + cols_cat], data_definition=definicion)
        snap = Report([DataSummaryPreset(), DataDriftPreset()]).run(ds_cur, ds_ref)  # primero current, luego reference
        snap.save_html(str(out / "reporte_calidad_drift.html"))
        try:
            (out / "reporte_calidad_drift.json").write_text(snap.json(), encoding="utf-8")
        except Exception:
            pass
        print("[Evidently] guardado reporte_calidad_drift.html")
    except Exception as e:
        print(f"[Evidently] Falló el reporte de calidad/drift: {type(e).__name__}: {e}")

    # ---- Reporte 2: rendimiento de clasificación (necesita target en ambas bases) ----
    try:
        from evidently.presets import ClassificationPreset
        base_ref = ref_desemp if ref_desemp is not None else ref
        if not (t in base_ref.columns and hay_target):
            print("[Evidently] Sin target en ambas bases: se omite el reporte de desempeño.")
            return
        d_ref, d_cur = base_ref[base_ref[t].notna()], cur[cur[t].notna()]
        definicion = DataDefinition(
            numerical_columns=num_cols, categorical_columns=cat_cols,
            classification=[BinaryClassification(target=t, prediction_labels="prediction",
                                                 prediction_probas="prediction_proba", pos_label=1)])
        ds_ref = Dataset.from_pandas(d_ref, data_definition=definicion)
        ds_cur = Dataset.from_pandas(d_cur, data_definition=definicion)
        snap = Report([ClassificationPreset()]).run(ds_cur, ds_ref)
        snap.save_html(str(out / "reporte_desempeno.html"))
        print("[Evidently] guardado reporte_desempeno.html")
    except Exception as e:
        print(f"[Evidently] Falló el reporte de desempeño: {type(e).__name__}: {e}")


# =============================================================================
# ORQUESTADOR
# =============================================================================
def main(**overrides):
    cfg = {**CONFIG, **overrides}
    out = Path(cfg["carpeta_salida"])
    out.mkdir(parents=True, exist_ok=True)

    print("1) Cargando modelo y datos...")
    modelo = cargar_modelo(cfg["ruta_modelo"])
    train_t = transformar_crudo(leer_datos(cfg["ruta_train"]))
    prod_t = transformar_crudo(leer_datos(cfg["ruta_prod"]))
    desemp_t = (transformar_crudo(leer_datos(cfg["ruta_ref_desempeno"]))
                if cfg["ruta_ref_desempeno"] else None)

    cols = columnas_del_modelo(modelo, cfg)
    cat_cols = columnas_categoricas(modelo, cols, train_t, cfg)
    num_cols = [c for c in cols if c not in cat_cols]
    print(f"   {len(cols)} variables del modelo ({len(num_cols)} numéricas, {len(cat_cols)} categóricas)")

    print("2) Puntuando...")
    ref = puntuar(modelo, train_t, cols, cat_cols, cfg, "train")
    cur = puntuar(modelo, prod_t, cols, cat_cols, cfg, "producción")
    ref_d = puntuar(modelo, desemp_t, cols, cat_cols, cfg, "ref. desempeño") if desemp_t is not None else None

    print("3) PSI y métricas...")
    psi = tabla_psi(ref, cur, num_cols, cat_cols)
    psi.to_csv(out / "psi_variables.csv", index=False)
    bases = {"train": ref, "producción": cur}
    if ref_d is not None:
        bases["ref_desempeño"] = ref_d
    resumen = resumen_modelo(bases, cfg)
    resumen.to_csv(out / "resumen_modelo.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print("\nResumen del modelo:\n", resumen.to_string(index=False))
        print("\nTop 15 variables por PSI (<0.1 estable, 0.1-0.25 alerta, >0.25 cambio):\n",
              psi.head(15)[["variable", "tipo", "psi", "estado"]].to_string(index=False))

    if cfg["guardar_scoring"]:
        ref.to_parquet(out / "scoring_train.parquet")
        cur.to_parquet(out / "scoring_produccion.parquet")

    print("4) Reportes Evidently...")
    reportes_evidently(ref, cur, num_cols, cat_cols, cfg, ref_d)

    print(f"\nListo. Salidas en: {out.resolve()}")
    return dict(ref=ref, cur=cur, psi=psi, resumen=resumen)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()