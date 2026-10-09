import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None 
import os
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import lightgbm as lgb
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from sklearn.feature_selection import RFE
from IPython.display import display
from tqdm import tqdm_notebook
from dython.nominal import associations
import re
import mlflow
import mlflow.catboost

seed = 666

class MedirTiempoEjecucion:
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = time.time() - self.start_time
        print(f"Tiempo transcurrido: {int(elapsed_time // 3600)} horas, {int((elapsed_time % 3600) // 60)} minutos y {elapsed_time % 60:.3f} segundos")


#Resumen de las métricas de rendimiento de un modelo de regresión
def metricas_regresion(y_train_real,y_train_predicho,y_test_real,y_test_predicho,plot_actual_vs_predicted=False):
    
    def rmspe(y_true, y_pred):
        return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

    #Train
    mae_train = metrics.mean_absolute_error(y_train_real, y_train_predicho)
    mape_train = metrics.mean_absolute_percentage_error(y_train_real, y_train_predicho)
    mse_train = metrics.mean_squared_error(y_train_real, y_train_predicho)
    rmse_train = metrics.root_mean_squared_error(y_train_real, y_train_predicho)
    rmspe_train = rmspe(y_train_real, y_train_predicho)
    r2_train = metrics.r2_score(y_train_real, y_train_predicho)

    #Test
    mae_test = metrics.mean_absolute_error(y_test_real, y_test_predicho)
    mape_test = metrics.mean_absolute_percentage_error(y_test_real, y_test_predicho)
    mse_test = metrics.mean_squared_error(y_test_real, y_test_predicho)
    rmse_test = metrics.root_mean_squared_error(y_test_real, y_test_predicho)
    rmspe_test = rmspe(y_test_real, y_test_predicho)
    r2_test = metrics.r2_score(y_test_real, y_test_predicho)
    #Resumen
    resultados = { 'MAE': [mae_train, mae_test],'MAPE': [mape_train, mape_test], 'MSE': [mse_train, mse_test], 'RMSE': [rmse_train, rmse_test], 'RMSPE': [rmspe_train, rmspe_test], 'R2': [r2_train, r2_test] }
    resultados = pd.DataFrame(resultados, index=['Train', 'Test'])
    if plot_actual_vs_predicted:
        fig, axs = plt.subplots(1, 2, figsize=(6,3))
        # Primer gráfico:
        display_train = metrics.PredictionErrorDisplay(y_true=y_train_real, y_pred=y_train_predicho)
        display_train.plot(ax=axs[0], kind="actual_vs_predicted")   # actual_vs_predicted / residual_vs_predicted
        axs[0].set_title('Train')
        # Segundo gráfico:
        display_test = metrics.PredictionErrorDisplay(y_true=y_test_real, y_pred=y_test_predicho)
        display_test.plot(ax=axs[1], kind="actual_vs_predicted")    # actual_vs_predicted / residual_vs_predicted
        axs[1].set_title('Test')
        plt.tight_layout()
    return resultados


def metricas_regresion_v2(y_train_real,y_train_predicho,y_test_real,y_test_predicho,y_real_oot1, y_pred_oot1,y_real_oot2, y_pred_oot2,y_real_oot3, y_pred_oot3):

    def rmspe_metric(y_true, y_pred):
        return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

    #   def r2_adj(y_true, y_pred, p):
    #       n = len(y_true)  # asumimos n > p + 1
    #       R2 = metrics.r2_score(y_true, y_pred)
    #       return 1 - (1 - R2) * (n - 1) / (n - p - 1)

    # Mapeo de datasets en el orden requerido
    datasets = [
        ('Train', y_train_real, y_train_predicho),
        ('Test', y_test_real, y_test_predicho),
        ('OOT1', y_real_oot1, y_pred_oot1),
        ('OOT2', y_real_oot2, y_pred_oot2),
        ('OOT3', y_real_oot3, y_pred_oot3),
    ]

    # Calcula métricas en un bucle
    filas = []
    indices = []
    for nombre, y_true, y_pred in datasets:
        mae = metrics.mean_absolute_error(y_true, y_pred)
        mape = metrics.mean_absolute_percentage_error(y_true, y_pred)
        mse = metrics.mean_squared_error(y_true, y_pred)
        rmse = metrics.root_mean_squared_error(y_true, y_pred)
        rmspe = rmspe_metric(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)
        #   r2aj = r2_adj(y_true, y_pred, p)

        filas.append({
            'MAE': mae,
            'MAPE': mape,
            'MSE': mse,
            'RMSE': rmse,
            'RMSPE': rmspe,
            'R2': r2,
            #   'R2_adj': r2aj
        })
        indices.append(nombre)

    resultados = pd.DataFrame(filas, index=indices)

    return resultados


#Obtiene la lista con las variables constantes
def obtener_var_constantes(df, var_excl=[], umbral_const=0.9999):
    list_var_const = []
    for col in df.columns:
        if col in var_excl:
            continue
        if (df[col].dropna().nunique() == 1) or (df[col].value_counts().iloc[0]/len(df) >= umbral_const):
            list_var_const.append(col)
    return list_var_cons


def porcentaje_categorias(data):
    # Lista de variables cualitativas
    variables_with_non_representative_categories = []

    # Bucle para analizar cada variable cualitativa
    for var in data.columns:
        if var in data.columns:  
            freq = data[var].value_counts(normalize=True)  
            dominant_category_percentage = freq.max() 
            non_representative_categories = freq[freq < 0.05]  
    
            if not non_representative_categories.empty:
                num_categories = freq.count()
                num_non_representative_categories = non_representative_categories.count()
                variables_with_non_representative_categories.append((var, num_categories, num_non_representative_categories, 
                                                                    round(dominant_category_percentage*100,2), 
                                                                    round(num_non_representative_categories/num_categories*100,2)))
    
    non_representative = pd.DataFrame(variables_with_non_representative_categories, columns=['Variable', 'Total categorias',
                                                                                            'Total cat. no representativas','% Clase predominante',
                                                                                            '% Cat. no representativas'])
    return non_representative.sort_values(by=['% Clase predominante', '% Cat. no representativas'], ascending=False)


# Muestra las importancias gain y split de un modelo de regresión LightGBM ya entrenado
def importancias_lgbm(modelo,datos):
    split_importance = modelo.feature_importances_
    gain_importance = modelo.booster_.feature_importance(importance_type='gain')

    f_imp = pd.DataFrame({
        'Var': datos.columns,
        'Split_Importance': split_importance,
        'Gain_Importance': gain_importance
    })

    # Calcular el peso de cada importancia
    f_imp['Split_Weight'] = f_imp['Split_Importance'] / f_imp['Split_Importance'].sum() 
    f_imp['Gain_Weight'] = f_imp['Gain_Importance'] / f_imp['Gain_Importance'].sum()
    f_imp['Avg_Weight'] = ( f_imp['Gain_Weight'] + f_imp['Split_Weight'] )/2

    # Ordenar el DataFrame por la importancia de 'gain'
    return f_imp.sort_values("Avg_Weight", ascending=False,ignore_index=True)


def importancias_lgbm_v2(modelo,datos):
    split_importance = modelo.feature_importances_
    gain_importance = modelo.booster_.feature_importance(importance_type='gain')

    f_imp = pd.DataFrame({
        'Var': datos.columns,
        'Split_Importance': split_importance,
        'Gain_Importance': gain_importance
    })

    # Calcular el peso de cada importancia
    f_imp['Split_Weight'] = f_imp['Split_Importance'] / f_imp['Split_Importance'].sum() 
    f_imp['Gain_Weight'] = f_imp['Gain_Importance'] / f_imp['Gain_Importance'].sum()
    f_imp['Avg_Weight'] = ( f_imp['Gain_Weight'] + f_imp['Split_Weight'] )/2
    f_imp['H_Weight'] = 2*f_imp['Gain_Weight']*f_imp['Split_Weight']/( f_imp['Gain_Weight'] + f_imp['Split_Weight'] )
    f_imp['H_Avg_Weight'] = f_imp['H_Weight'] / f_imp['H_Weight'].sum()
    f_imp = f_imp.drop(columns=['H_Weight'])

    # Ordenar el DataFrame por la importancia de 'gain'
    return f_imp.sort_values("H_Avg_Weight", ascending=False,ignore_index=True)


def plot_null_percentage(df):
    """
    Esta función genera una gráfica de barras horizontales que muestra el porcentaje de valores nulos
    por cada variable en un DataFrame dado.
    
    Parámetros:
    df (pandas.DataFrame): El DataFrame para el cual se calculará y visualizará el porcentaje de valores nulos.
    """
    # Calcular el porcentaje de valores nulos por columna
    porcentaje_nulos = df.isnull().mean() * 100
    porcentaje_nulos = porcentaje_nulos.sort_values(ascending=True)

    # Visualizar el porcentaje de valores nulos
    #plt.figure(figsize=(8,6))
    bars = plt.barh(porcentaje_nulos.index, porcentaje_nulos)
    plt.xlabel('Porcentaje')
    plt.title('Porcentaje de valores nulos por cada variable')
    plt.xlim(0, 100)
    
    # Añadir etiquetas con el valor del porcentaje en cada barra
    for bar in bars:
        xval = bar.get_width()
        plt.text(xval + 1, bar.get_y() + bar.get_height()/2, round(xval, 2), va='center', ha='left')
    
    plt.show()


def correlacion(df, cols, lista_var_cat=None, target=None):
    
    print(".Correlacion")
    
    # Target
    if target is not None:
        cols_total = cols + [target]
    else:
        cols_total = cols

    # Correlacion
    assoc = associations(df[cols_total], 
                        nominal_columns=lista_var_cat,
                        numerical_columns=list(set(cols_total).difference(set(lista_var_cat))),
                        plot=False,num_num_assoc='spearman',nom_num_assoc='correlation_ratio',nom_nom_assoc='cramer'
                        )#,cmap='coolwarm',title='Correlación',figsize=(9,6)

    return assoc['corr']

    
def extrae_cortes_arbol(df,cols,arbol):
    if df[cols].dtype == "int64":
        cuts = df[cols].drop_duplicates().to_list()
    else:
        cuts = arbol.tree_.threshold
    cuts = np.unique(cuts)
    #print(cuts)
    #cuts = np.insert(cuts,0,0)
    #cuts = np.insert(cuts,0,0)
    cuts = np.insert(cuts,0,-9999999)
    cuts = np.insert(cuts,0,-9999998)
    cuts = np.insert(cuts,0,-9999990)
    cuts = np.insert(cuts,0,-9000000)
    cuts = np.insert(cuts,0,np.max(cuts))
    cuts = np.sort(cuts)
    cuts = np.delete(cuts, np.where(cuts == -2))
    return cuts


def bin_variable_regresion(df,col,cuts,target,estimado=None):
    n = len(cuts)
    variable = []
    orden_cota = []
    cotas = []
    cota_inf = []
    cota_sup = []
    sql = []
    sql_num = []
    media = []
    if estimado is not None: media_est = []
    conteo = []

    df_count = df[col]
    aux = 1
    #print(n)
    for c in range(n):
        #print(c)
        if aux < 100: orden = ("00"+str(aux))[-2:]
        else: orden = str(aux)
        
        if c == 0: #Primer valor
            df_count = df[df[col] <= cuts[c]]
            cotas_c = orden+". (-, "+str(round(cuts[c],3))+"]"
            cota_inf_c = float('-inf')
            cota_sup_c = cuts[c]
            sql_c = "when "+col+" <= "+str(cuts[c])+" then '"+cotas_c+"'"
            sql_n = "when "+col+" <= "+str(cuts[c])+" then "+str(aux)
        elif c == n-1: #Ultimo valor
            df_count = df[df[col] > cuts[c]] 
            cotas_c = orden+". ("+str(round(cuts[c],3))+", +)"
            cota_inf_c = cuts[c]
            cota_sup_c = float('inf')
            sql_c = "when "+col+" > "+str(cuts[c])+" then '"+cotas_c+"' end ["+col+"_cat]"
            sql_n = "when "+col+" > "+str(cuts[c])+" then "+str(aux)+" end ["+col+"_cat]"
        else: #Valores intermedios
            if (cuts[c-1] < cuts[c]) & (cuts[c] == 0):
                df_count = df[(df[col] < cuts[c])&(df[col] > cuts[c-1])] 
                cotas_c = orden+". ("+str(round(cuts[c-1],3))+", "+str(round(cuts[c],3))+")"
            elif (cuts[c-1] == cuts[c]) & (cuts[c] == 0):
                df_count = df[df[col] == cuts[c]]
                cotas_c = orden+". ("+str(round(cuts[c-1],3))+", "+str(round(cuts[c],3))+"]"
            else:
                df_count = df[(df[col] <= cuts[c])&(df[col] > cuts[c-1])] 
                cotas_c = orden+". ("+str(round(cuts[c-1],3))+", "+str(round(cuts[c],3))+"]"
            sql_c = "when "+col+" <= "+str(cuts[c])+" then '"+cotas_c+"'"
            sql_n = "when "+col+" <= "+str(cuts[c])+" then "+str(aux)
            cota_inf_c = cuts[c-1]
            cota_sup_c = cuts[c]
        
        if "-9000000.0]" in cotas_c: 
            cotas_c = "Missing"
            sql_c = "when "+col+" is null then '"+cotas_c+"'"
            sql_n = "when "+col+" is null then "+str(aux)

        if aux == 1:
            sql_c = ",case " + sql_c
            sql_n = ",case " + sql_n
            
        media_c = df_count[target].mean()
        if estimado is not None: media_est_c = df_count[estimado].mean()
        conteo_c = len(df_count.index)
        
        if conteo_c > 0:
            variable.append(col)
            orden_cota.append(aux)
            cotas.append(cotas_c)
            cota_inf.append(cota_inf_c)
            cota_sup.append(cota_sup_c)
            sql.append(sql_c)
            sql_num.append(sql_n)
            media.append(media_c)
            if estimado is not None: media_est.append(media_est_c)
            conteo.append(conteo_c)
            aux = aux + 1
        
        if conteo_c == 0 and c == n-1:
            sql[-1] += " end ["+col+"_cat]"
    
    media = [round(x,6) for x in media]
    if estimado is not None: media_est = [round(x,6) for x in media_est]
    cota_inf = [round(x,3) for x in cota_inf]
    cota_sup = [round(x,3) for x in cota_sup]
    conteo_pct = [round(x/sum(conteo),3) for x in conteo]
    
    dict_resumen = ""
    dict_resumen = dict_resumen + "'Variable':variable,'Orden_Cota':orden_cota,'Cotas':cotas,'Cota_Inf':cota_inf,'Cota_Sup':cota_sup,'Media':media"
    if estimado is not None: dict_resumen = dict_resumen + ",'Media_Est':media_est"
    dict_resumen = dict_resumen + ",'Total':conteo,'Total_Pct':conteo_pct,'Sql':sql,'Sql_num':sql_num"
    dict_resumen = "{" + dict_resumen + "}"
    dict_resumen = eval(dict_resumen)
    
    df_resumen = pd.DataFrame(dict_resumen)
    
    return df_resumen


def bin_variable(cuts,df_subset,target,col,orden):
    n = len(cuts)
    variable = []
    orden_var = []
    orden_cota = []
    cotas = []
    sql = []
    media = []
    conteo = []

    df_count = df_subset[col]
    aux = 1
    #print(n)
    for c in range(n):
        #print(c)
        if c == 0: #Primer valor
            df_count = df_subset[df_subset[col] <= cuts[c]]
            cotas_c = str(aux)+". (-, "+str(round(cuts[c],3))+"]"
            sql_c = "when "+col+" <= "+str(cuts[c])+" then '"+cotas_c+"'"
        elif c == n-1: #Ultimo valor
            df_count = df_subset[df_subset[col] > cuts[c]] 
            cotas_c = str(aux)+". ("+str(round(cuts[c],3))+", +)"
            sql_c = "when "+col+" > "+str(cuts[c])+" then '"+cotas_c+"' end ["+col+"_cat]"
        else: #Valores intermedios
            if (cuts[c-1] < cuts[c]) & (cuts[c] == 0):
                df_count = df_subset[(df_subset[col] < cuts[c])&(df_subset[col] > cuts[c-1])] 
                cotas_c = str(aux)+". ("+str(round(cuts[c-1],3))+", "+str(round(cuts[c],3))+")"
            elif (cuts[c-1] == cuts[c]) & (cuts[c] == 0):
                df_count = df_subset[df_subset[col] == cuts[c]]
                cotas_c = str(aux)+". ("+str(round(cuts[c-1],3))+", "+str(round(cuts[c],3))+"]"
            else:
                df_count = df_subset[(df_subset[col] <= cuts[c])&(df_subset[col] > cuts[c-1])] 
                cotas_c = str(aux)+". ("+str(round(cuts[c-1],3))+", "+str(round(cuts[c],3))+"]"
            sql_c = "when "+col+" <= "+str(cuts[c])+" then '"+cotas_c+"'"
        
        if "-9000000.0]" in cotas_c: 
            cotas_c = cotas_c[:cotas_c.find(".")+1]+" Missing"
            sql_c = "when "+col+" is null then '"+cotas_c+"'"
        
        if aux == 1: sql_c = ",case " + sql_c
        media_c = df_count[target].mean()
        conteo_c = len(df_count.index)
        
        if conteo_c > 0:
            variable.append(col)
            orden_cota.append(aux)
            orden_var.append(orden)
            cotas.append(cotas_c)
            sql.append(sql_c)
            media.append(media_c)
            conteo.append(conteo_c)
            aux = aux + 1
        
        if conteo_c == 0 and c == n-1:
            sql[-1] += " end ["+col+"_cat]"
    
    df_resumen = pd.DataFrame({"Orden":orden_var,"Variable":variable,"Orden_Cota":orden_cota,"Cotas":cotas,"Sql":sql,"Media":media,"Total":conteo})
    df_resumen["Total_Pct"] = (df_resumen["Total"] / df_resumen["Total"].sum()).round(3)
    
    return df_resumen


def bin_variable_categorica(df_subset, target, col, orden, max_top=10):
    # Conteos ordenados de mayor a menor
    conteos = df_subset[col].value_counts()
    top = conteos.index[:max_top].tolist()
    hay_otros = len(conteos) > max_top

    variable, orden_var, orden_cota, cotas, sql, media, total = [], [], [], [], [], [], []

    # Top 10 categorías
    for aux, cat in enumerate(top, start=1):
        mask = df_subset[col] == cat
        cota_c = str(aux) + ". " + str(cat)
        cat_sql = str(cat).replace("'", "''")
        sql_c = "when " + col + " = '" + cat_sql + "' then '" + cota_c.replace("'", "''") + "'"

        variable.append(col)
        orden_var.append(orden)
        orden_cota.append(aux)
        cotas.append(cota_c)
        sql.append(sql_c)
        media.append(df_subset.loc[mask, target].mean())
        total.append(int(mask.sum()))

    # Categoría 11: acumulado del resto
    if hay_otros:
        aux = max_top + 1
        mask = ~df_subset[col].isin(top)
        lista_sql = ",".join("'" + str(c).replace("'", "''") + "'" for c in top)
        cota_c = str(aux) + ". Otros"
        sql_c = "when " + col + " not in (" + lista_sql + ") then '" + cota_c + "'"

        variable.append(col)

        orden_var.append(orden)
        orden_cota.append(aux)
        cotas.append(cota_c)
        sql.append(sql_c)
        media.append(df_subset.loc[mask, target].mean())
        total.append(int(mask.sum()))

    # Mismo formato SQL que bin_variable: abre con ",case" y cierra con "end [col_cat]"
    sql[0] = ",case " + sql[0]
    sql[-1] = sql[-1] + " end [" + col + "_cat]"

    df_resumen = pd.DataFrame({"Orden": orden_var, "Variable": variable, "Orden_Cota": orden_cota,
                               "Cotas": cotas, "Sql": sql, "Media": media, "Total": total})
    df_resumen["Total_Pct"] = (df_resumen["Total"] / df_resumen["Total"].sum()).round(3)

    return df_resumen


def descripcion_variable(df,col,orden,redondeo=None):
    
    if redondeo is None: redondeo = 6
    
    # Medidas
    describe = df[col].describe()
    
    if df[col].dtype in (["int64","float64","int32"]):
        iqr = describe["75%"] - describe["25%"]
        outlo = describe["25%"] - 1.5*iqr
        outhi = describe["75%"] + 1.5*iqr
        dict_descripcion = {"Orden":orden
                                    ,"Variable":col
                                    ,"Type":df[col].dtype
                                    ,"Freq":df.shape[0]
                                    ,"Unico":len(df[col].unique())
                                    ,"Media":describe["mean"]
                                    ,"Desv":describe["std"]
                                    ,"Min":describe["min"]
                                    ,"Q1":describe["25%"]
                                    ,"Mediana":describe["50%"]
                                    ,"Q3":describe["75%"]
                                    ,"Max":describe["max"]
                                    ,"Miss":df[col].isna().sum()
                                    ,"Cero":df[col].eq(0).sum()
                                    ,"Pos":df[col].gt(0).sum()
                                    ,"Neg":df[col].lt(0).sum()
                                    ,"OutLo":df[col].lt(outlo).sum()
                                    ,"OutHi":df[col].gt(outhi).sum()
                                    ,"pMiss":(df[col].isna().sum() / df.shape[0]).round(redondeo)
                                    ,"pCero":(df[col].eq(0).sum() / df.shape[0]).round(redondeo)
                                    ,"pPos":(df[col].gt(0).sum() / df.shape[0]).round(redondeo)
                                    ,"pNeg":(df[col].lt(0).sum() / df.shape[0]).round(redondeo)
                                    ,"pOutLo":(df[col].lt(outlo).sum() / df.shape[0]).round(redondeo)
                                    ,"pOutHi":(df[col].gt(outhi).sum() / df.shape[0]).round(redondeo)
                                    ,"Moda":np.nan
                                    ,"nModa":np.nan
                                    ,"pModa":np.nan}
    else:
        dict_descripcion = {"Orden":orden
                                    ,"Variable":col
                                    ,"Type":df[col].dtype
                                    ,"Freq":df.shape[0]
                                    ,"Unico":len(df[col].unique())
                                    ,"Media":np.nan
                                    ,"Desv":np.nan
                                    ,"Min":np.nan
                                    ,"Q1":np.nan
                                    ,"Mediana":np.nan
                                    ,"Q3":np.nan
                                    ,"Max":np.nan
                                    ,"Miss":df[col].isna().sum()
                                    ,"Cero":np.nan
                                    ,"Pos":np.nan
                                    ,"Neg":np.nan
                                    ,"OutLo":np.nan
                                    ,"OutHi":np.nan
                                    ,"pMiss":(df[col].isna().sum() / df.shape[0]).round(redondeo)
                                    ,"pCero":np.nan
                                    ,"pPos":np.nan
                                    ,"pNeg":np.nan
                                    ,"pOutLo":np.nan
                                    ,"pOutHi":np.nan
                                    ,"Moda":describe["top"]
                                    ,"nModa":describe["freq"]
                                    ,"pModa":round((describe["freq"] / df.shape[0]),redondeo)}
    
    df_descripcion = pd.DataFrame([dict_descripcion])
    
    return df_descripcion


def extrae_cortes_int(df,col):
    cuts = []
    if df[col].dtype == "int64":
        cuts = df[col].drop_duplicates().to_list()
        cuts = np.unique(cuts)
        cuts = np.insert(cuts,0,-9999999)
        cuts = np.insert(cuts,0,-9999998)
        cuts = np.insert(cuts,0,-9999990)
        cuts = np.insert(cuts,0,-9000000)
        cuts = np.insert(cuts,0,np.max(cuts))
        cuts = np.sort(cuts)
        cuts = np.delete(cuts, np.where(cuts == -2))
    return cuts


def eda_bivariado(df_in,cols,target,ruta_eda,name=None):
    df = df_in.copy()
    df_target = df[target]
    n = df.shape[0]
    
    # df_variables = pd.DataFrame(columns=["Variable","Orden_Cota","Cotas","Sql","Media","Total","Total_Pct"])
    df_variables = pd.DataFrame()
    
    if name is None:  
        ruta_pdf = ruta_eda+"//02.Analisis_Bivariado.pdf"
    else:
        ruta_pdf = ruta_eda+"//02.Analisis_Bivariado"+name+".pdf"
        
    with PdfPages(ruta_pdf) as pdf:    
        for i in range(len(cols)):
            print("...."+str(i+1)+". "+cols[i])
            
            if df[cols[i]].dtype in (["int64","float64","int32"]):
                try:
                    # Base inicial
                    df_X = pd.DataFrame(df[cols[i]])
                    df_X.loc[df_X[cols[i]].isna(),cols[i]] = -9000000
                    df.loc[df[cols[i]].isna(),cols[i]] = -9000000

                    # Arbol
                    arbol = DecisionTreeClassifier(min_samples_split = 0.20,min_samples_leaf = int(n*0.05))#,max_depth = 3
                    arbol.fit(df_X,df_target)

                    # Cortes del Arbol
                    if df[cols[i]].dtype == "int64":
                        cuts = extrae_cortes_int(df,cols[i])
                    else:
                        cuts = extrae_cortes_arbol(df,cols[i],arbol)
                    
                    # Resumen de Variables
                    df_bin_variable = bin_variable_regresion(df,cols[i],cuts,target)
                    df_variables = pd.concat([df_variables,df_bin_variable])

                    # Graficos
                    # Grafico de Barras
                    plt.figure(figsize=(15, 6), dpi=100)

                    plt.subplot(121)
                    plt.bar(df_bin_variable["Orden_Cota"],df_bin_variable["Total"],width=.5)
                    plt.xticks(df_bin_variable["Orden_Cota"])
                    plt.title(str(i+1)+". "+cols[i])
                    axes2 = plt.twinx()
                    axes2.plot(df_bin_variable["Orden_Cota"], df_bin_variable['Media'], color='Orange',marker = "o")

                    # Tabla resumen
                    plt.subplot(122)
                    cell_text = []
                    df_X = df_bin_variable[["Cotas", "Media", "Total","Total_Pct"]]
                    df_X.loc[:,"Media"] = df_X["Media"].apply(lambda x: round(x,6))
                    df_X.loc[:,"Total_Pct"] = df_X['Total_Pct'].apply(lambda x: str(round(x*100,1))+"%")
                    #df_X["Media"] = df_X["Media"].apply(lambda x: round(x,1))
                    #df_X["Total_Pct"] = df_X['Total_Pct'].apply(lambda x: str(round(x*100,1))+"%")
                    for row in range(len(df_X)):
                        cell_text.append(df_X.iloc[row])

                    tab = plt.table(cellText=cell_text, colLabels=df_X.columns, cellLoc='center',loc='center',colWidths=[0.6, 0.2,0.2,0.2])
                    tab.auto_set_font_size(False)
                    #tab.set_fontsize(12)
                    #tab.scale(2, 2)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.25,hspace=0.4)

                    pdf.savefig()
                    plt.close()

                    # Arbol
                    fig = plt.figure(figsize=(25,20))
                    _ = tree.plot_tree(arbol, 
                                    feature_names=[cols[i]],
                                    #class_names=iris.target_names,
                                    filled=True)
                    pdf.savefig()
                    plt.close()
                    
                except Exception as e:
                    print("...."+str(i+1)+". Error: ",e)
            
            else:
                try:
                    # Base inicial
                    df_X = pd.DataFrame(df[cols[i]])
                    df_X.loc[df_X[cols[i]].isna(),cols[i]] = "Missing"
                    df.loc[df[cols[i]].isna(),cols[i]] = "Missing"
    
                    # Histograma: top 10 categorias + acumulado del resto (categoria 11)
                    df_bin_variable = bin_variable_categorica(df,target,cols[i],i+1)
                    df_variables = pd.concat([df_variables, df_bin_variable])
    
                    # Graficos
                    # Grafico de Barras
                    plt.figure(figsize=(15, 6), dpi=100)
    
                    plt.subplot(121)
                    plt.bar(df_bin_variable["Orden_Cota"],df_bin_variable["Total"],width=.5)
                    plt.xticks(df_bin_variable["Orden_Cota"])
                    # plt.xticks(Orden_Cota, labels, rotation=45, ha="right")
                    plt.title(str(i+1)+". "+cols[i])
                    axes2 = plt.twinx()
                    axes2.plot(df_bin_variable["Orden_Cota"], df_bin_variable['Media'], color='Orange',marker = "o")
    
                    # Tabla resumen
                    plt.subplot(122)
                    cell_text = []
                    df_X = df_bin_variable[["Cotas", "Media", "Total","Total_Pct"]].copy()
                    df_X.loc[:,"Cotas"] = df_X["Cotas"].apply(lambda x: x if len(x) <= 40 else x[:37]+"...")
                    df_X.loc[:,"Media"] = df_X["Media"].apply(lambda x: round(x,6))
                    df_X.loc[:,"Total_Pct"] = df_X['Total_Pct'].apply(lambda x: str(round(x*100,1))+"%")
                    for row in range(len(df_X)):
                        cell_text.append(df_X.iloc[row])
    
                    tab = plt.table(cellText=cell_text, colLabels=df_X.columns, cellLoc='center',loc='center',colWidths=[0.6, 0.2,0.2,0.2])
                    tab.auto_set_font_size(False)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.25,hspace=0.4)
    
                    pdf.savefig()
                    plt.close()
    
                except Exception as e:
                    print("...."+str(i+1)+". Error: ",e)
                    plt.close()
            
    df_variables = df_variables.reset_index()
    
    return df_variables


def eda_vf2(df, target, var_excluidas, var_categoricas, ruta_eda):
    
    print("..Eda")
    start_time = time.time()
    
    # Quitando variables
    df = df.drop(columns=var_excluidas)

    # Lista de columnas
    cols = df.columns.to_list()
    cols.sort()
    cols.remove(target)
    
    # Correlación
    df_cor = correlacion(df, cols, var_categoricas, target)
    
    # Descripcion Variables
    df_descripcion = pd.DataFrame()
    for i in range(len(cols)):
        df_descripcion = pd.concat([df_descripcion, descripcion_variable(df, cols[i], i, redondeo=6)])
    
    # Inicia EDA bivariado
    df_variables = eda_bivariado(df, cols, target, ruta_eda)
    
    with pd.ExcelWriter(ruta_eda+"\\01.Analisis_Exploratorio.xlsx") as writer:  
        df_cor.to_excel(writer, sheet_name='Correlacion')
        df_descripcion.to_excel(writer, sheet_name='EDA',index=False)
        df_variables.to_excel(writer, sheet_name='Variables',index=False)
        #df.to_excel(writer,sheet_name='diccionario',index=False)
    
    elapsed_time = time.time() - start_time
    print(f"Tiempo transcurrido: {int(elapsed_time // 60)} minutos y {elapsed_time % 60:.3f} segundos")
    start_time = time.time()

    
def coincidencia(y_real_train, y_pred_train, umbral, index):
    
    porc_pres_train = ((y_pred_train - y_real_train)/y_real_train)

    # Definir las condiciones para la clasificación
    condiciones_train = [
        (porc_pres_train >= -umbral) & (porc_pres_train <= umbral),  # Coincidencia
        porc_pres_train > umbral,  # Sobreestimación
        porc_pres_train < -umbral  # Subestimación
    ]
    clases = ['Coincidencia', 'Sobreestimacion', 'Subestimacion']

    # Crea el vector de clasificación
    clasificacion_train = np.select(condiciones_train, clases, default='No clasificado')

    # Crear DataFrames con los resultados
    df_precision_train = pd.DataFrame({'y_real': y_real_train, 'y_predicho': y_pred_train, 'porc_pres': porc_pres_train, 'clas_pres': clasificacion_train})
    
    # Calcular las proporciones de cada clase
    presc_train = df_precision_train['clas_pres'].value_counts(normalize=True)

    # Crear un DataFrame con los resultados finales
    resultados = {
        'Subestimacion': [presc_train.get('Subestimacion', 0)],
        'Coincidencia': [presc_train.get('Coincidencia', 0)],
        'Sobreestimacion': [presc_train.get('Sobreestimacion', 0)]
    }
    df_resultados_precs = pd.DataFrame(resultados, index=[index])
    
    return df_resultados_precs

def calcular_coincidencia(y_pred_train, y_pred_test, y_real_train, y_real_test, umbral):
    # Evitar errores dividiendo por cero
    #y_real_train = np.where(y_real_train == 0, np.nan, y_real_train)
    #y_real_test = np.where(y_real_test == 0, np.nan, y_real_test)
    
    # Calcular el porcentaje de error
    porc_pres_train = ((y_pred_train - y_real_train)/y_real_train)
    porc_pres_test = ((y_pred_test - y_real_test)/y_real_test)

    # Definir las condiciones para la clasificación
    condiciones_train = [
        (porc_pres_train >= -umbral) & (porc_pres_train <= umbral),  # Coincidencia
        porc_pres_train > umbral,  # Sobreestimación
        porc_pres_train < -umbral  # Subestimación
    ]
    condiciones_test = [
        (porc_pres_test >= -umbral) & (porc_pres_test <= umbral),  # Coincidencia
        porc_pres_test > umbral,  # Sobreestimación
        porc_pres_test < -umbral  # Subestimación
    ]
    clases = ['Coincidencia', 'Sobreestimacion', 'Subestimacion']

    # Crea el vector de clasificación
    clasificacion_train = np.select(condiciones_train, clases, default='No clasificado')
    clasificacion_test = np.select(condiciones_test, clases, default='No clasificado')

    # Crear DataFrames con los resultados
    df_precision_train = pd.DataFrame({'y_real': y_real_train, 'y_predicho': y_pred_train, 'porc_pres': porc_pres_train, 'clas_pres': clasificacion_train})
    df_precision_test = pd.DataFrame({'y_real': y_real_test, 'y_predicho': y_pred_test, 'porc_pres': porc_pres_test, 'clas_pres': clasificacion_test})

    # Calcular las proporciones de cada clase
    presc_train = df_precision_train['clas_pres'].value_counts(normalize=True)
    presc_test = df_precision_test['clas_pres'].value_counts(normalize=True)

    # Crear un DataFrame con los resultados finales
    resultados = {
        'Subestimacion': [presc_train.get('Subestimacion', 0), presc_test.get('Subestimacion', 0)],
        'Coincidencia': [presc_train.get('Coincidencia', 0), presc_test.get('Coincidencia', 0)],
        'Sobreestimacion': [presc_train.get('Sobreestimacion', 0), presc_test.get('Sobreestimacion', 0)]
    }
    df_resultados_precs = pd.DataFrame(resultados, index=['Train', 'Test'])
    
    return df_resultados_precs

def calcular_coincidencia_v2(y_real_train,y_pred_train, y_real_test,y_pred_test, y_real_oot1,y_pred_oot1, y_real_oot2,y_pred_oot2, y_real_oot3,y_pred_oot3, umbral):

    # Calcular el porcentaje de error
    porc_pres_train = ((y_pred_train - y_real_train)/y_real_train)
    porc_pres_test = ((y_pred_test - y_real_test)/y_real_test)
    porc_pres_oot1 = ((y_pred_oot1 - y_real_oot1)/y_real_oot1)
    porc_pres_oot2 = ((y_pred_oot2 - y_real_oot2)/y_real_oot2)
    porc_pres_oot3 = ((y_pred_oot3 - y_real_oot3)/y_real_oot3)

    # Definir las condiciones para la clasificación
    condiciones_train = [
        (porc_pres_train >= -umbral) & (porc_pres_train <= umbral),  # Coincidencia
        porc_pres_train > umbral,  # Sobreestimación
        porc_pres_train < -umbral  # Subestimación
    ]
    condiciones_test = [
        (porc_pres_test >= -umbral) & (porc_pres_test <= umbral),  # Coincidencia
        porc_pres_test > umbral,  # Sobreestimación
        porc_pres_test < -umbral  # Subestimación
    ]
    condiciones_oot1 = [
        (porc_pres_oot1 >= -umbral) & (porc_pres_oot1 <= umbral),  # Coincidencia
        porc_pres_oot1 > umbral,  # Sobreestimación
        porc_pres_oot1 < -umbral  # Subestimación
    ]
    condiciones_oot2 = [
        (porc_pres_oot2 >= -umbral) & (porc_pres_oot2 <= umbral),  # Coincidencia
        porc_pres_oot2 > umbral,  # Sobreestimación
        porc_pres_oot2 < -umbral  # Subestimación
    ]
    condiciones_oot3 = [
        (porc_pres_oot3 >= -umbral) & (porc_pres_oot3 <= umbral),  # Coincidencia
        porc_pres_oot3 > umbral,  # Sobreestimación
        porc_pres_oot3 < -umbral  # Subestimación
    ]
    clases = ['Coincidencia', 'Sobreestimacion', 'Subestimacion']

    # Crea el vector de clasificación
    clasificacion_train = np.select(condiciones_train, clases, default='No clasificado')
    clasificacion_test = np.select(condiciones_test, clases, default='No clasificado')
    clasificacion_oot1 = np.select(condiciones_oot1, clases, default='No clasificado')
    clasificacion_oot2 = np.select(condiciones_oot2, clases, default='No clasificado')
    clasificacion_oot3 = np.select(condiciones_oot3, clases, default='No clasificado')

    # Crear DataFrames con los resultados
    df_precision_train = pd.DataFrame({'y_real': y_real_train, 'y_predicho': y_pred_train, 'porc_pres': porc_pres_train, 'clas_pres': clasificacion_train})
    df_precision_test = pd.DataFrame({'y_real': y_real_test, 'y_predicho': y_pred_test, 'porc_pres': porc_pres_test, 'clas_pres': clasificacion_test})
    df_precision_oot1 = pd.DataFrame({'y_real': y_real_oot1, 'y_predicho': y_pred_oot1, 'porc_pres': porc_pres_oot1, 'clas_pres': clasificacion_oot1})
    df_precision_oot2 = pd.DataFrame({'y_real': y_real_oot2, 'y_predicho': y_pred_oot2, 'porc_pres': porc_pres_oot2, 'clas_pres': clasificacion_oot2})
    df_precision_oot3 = pd.DataFrame({'y_real': y_real_oot3, 'y_predicho': y_pred_oot3, 'porc_pres': porc_pres_oot3, 'clas_pres': clasificacion_oot3})

    # Calcular las proporciones de cada clase
    presc_train = df_precision_train['clas_pres'].value_counts(normalize=True)
    presc_test = df_precision_test['clas_pres'].value_counts(normalize=True)
    presc_oot1= df_precision_oot1['clas_pres'].value_counts(normalize=True)
    presc_oot2 = df_precision_oot2['clas_pres'].value_counts(normalize=True)
    presc_oot3 = df_precision_oot3['clas_pres'].value_counts(normalize=True)

    # Crear un DataFrame con los resultados finales
    resultados = {
        'Subestimacion': [presc_train.get('Subestimacion', 0), presc_test.get('Subestimacion', 0), presc_oot1.get('Subestimacion', 0), presc_oot2.get('Subestimacion', 0), presc_oot3.get('Subestimacion', 0)],
        'Coincidencia': [presc_train.get('Coincidencia', 0), presc_test.get('Coincidencia', 0), presc_oot1.get('Coincidencia', 0), presc_oot2.get('Coincidencia', 0), presc_oot3.get('Coincidencia', 0)],
        'Sobreestimacion': [presc_train.get('Sobreestimacion', 0), presc_test.get('Sobreestimacion', 0), presc_oot1.get('Sobreestimacion', 0), presc_oot2.get('Sobreestimacion', 0), presc_oot3.get('Sobreestimacion', 0)]
    }
    df_resultados_precs = pd.DataFrame(resultados, index=['Train', 'Test', 'OOT1', 'OOT2', 'OOT3'])
    
    return df_resultados_precs

def calcular_coincidencia_v0(y_pred_train, y_real_train, umbral):
    
    # Calcular el porcentaje de error
    porc_pres_train = ((y_pred_train - y_real_train)/y_real_train)

    # Definir las condiciones para la clasificación
    condiciones_train = [
        (porc_pres_train >= -umbral) & (porc_pres_train <= umbral),  # Coincidencia
        porc_pres_train > umbral,  # Sobreestimación
        porc_pres_train < -umbral  # Subestimación
    ]
    clases = ['Coincidencia', 'Sobreestimacion', 'Subestimacion']

    # Crea el vector de clasificación
    clasificacion_train = np.select(condiciones_train, clases, default='No clasificado')

    # Crear DataFrames con los resultados
    df_precision_train = pd.DataFrame({'y_real': y_real_train, 'y_predicho': y_pred_train, 'porc_pres': porc_pres_train, 'clas_pres': clasificacion_train})

    # Calcular las proporciones de cada clase
    presc_train = df_precision_train['clas_pres'].value_counts(normalize=True)

    # Crear un DataFrame con los resultados finales
    resultados = {
        'Subestimacion': [presc_train.get('Subestimacion', 0)],
        'Coincidencia': [presc_train.get('Coincidencia', 0)],
        'Sobreestimacion': [presc_train.get('Sobreestimacion', 0)]
    }
    df_resultados_precs = pd.DataFrame(resultados, index=['Train'])
    
    return df_resultados_precs

def calcular_precision_x_intervalos(y_pred, y_real, umbral, bins):
    import numpy as np
    import pandas as pd

    porc_pres = (y_pred - y_real) / y_real
    condiciones = [
        (porc_pres < -umbral),              # Subestimación
        (porc_pres >= -umbral) & (porc_pres <= umbral),  # Coincidencia
        (porc_pres > umbral)                # Sobreestimación
    ]
    clases = ['Subestimacion', 'Coincidencia', 'Sobreestimacion']

    clasificacion = np.select(condiciones, clases, default='No clasificado')
    df_precision = pd.DataFrame({
        'y_real': y_real,
        'y_predicho': y_pred,
        'porc_pres': porc_pres,
        'clas_pres': clasificacion
    })

    df_precision['Intervalos'] = pd.cut(df_precision['y_real'], bins)

    frecuencia = df_precision['Intervalos'].value_counts().sort_index()
    porcentaje = frecuencia / frecuencia.sum()
    prom_real= df_precision.groupby('Intervalos')['y_real'].mean()
    prom_pred = df_precision.groupby('Intervalos')['y_predicho'].mean()

    resultado = pd.DataFrame({
        'Casos': frecuencia,
        'Casos (%)': porcentaje,
        'Prom. real': prom_real,
        'Prom. pred': prom_pred
    })

    distribucion = df_precision.groupby(['Intervalos', 'clas_pres']).size().unstack(fill_value=0)

    # Asegurar el orden deseado de las columnas
    for col in ['Subestimacion', 'Coincidencia', 'Sobreestimacion']:
        if col not in distribucion.columns:
            distribucion[col] = 0
    distribucion = distribucion[['Subestimacion', 'Coincidencia', 'Sobreestimacion']]

    distribucion = distribucion.div(distribucion.sum(axis=1), axis=0)

    resultado_final = resultado.join(distribucion, on='Intervalos')

    return resultado_final

def calcular_precision_x_intervalos_v2(y_pred, y_real, umbral, bins):

    def rmspe(y_true, y_pred):
        return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

    porc_pres = (y_pred - y_real) / y_real
    condiciones = [
        (porc_pres < -umbral),              # Subestimación
        (porc_pres >= -umbral) & (porc_pres <= umbral),  # Coincidencia
        (porc_pres > umbral)                # Sobreestimación
    ]
    clases = ['Subestimacion', 'Coincidencia', 'Sobreestimacion']

    clasificacion = np.select(condiciones, clases, default='No clasificado')
    df_precision = pd.DataFrame({
        'y_real': y_real,
        'y_predicho': y_pred,
        'porc_pres': porc_pres,
        'clas_pres': clasificacion
    })

    df_precision['Intervalos'] = pd.cut(df_precision['y_real'], bins)

    frecuencia = df_precision['Intervalos'].value_counts().sort_index()
    porcentaje = frecuencia / frecuencia.sum()
    prom_real= df_precision.groupby('Intervalos')['y_real'].mean()
    prom_pred = df_precision.groupby('Intervalos')['y_predicho'].mean()
    rmspe_cat = df_precision.groupby('Intervalos').apply(lambda df: rmspe(df['y_real'], df['y_predicho']) )

    resultado = pd.DataFrame({
        'Casos': frecuencia,
        'Casos (%)': porcentaje,
        'Prom. real': prom_real,
        'Prom. pred': prom_pred,
        'RMSPE': rmspe_cat
    })

    distribucion = df_precision.groupby(['Intervalos', 'clas_pres']).size().unstack(fill_value=0)

    # Asegurar el orden deseado de las columnas
    for col in ['Subestimacion', 'Coincidencia', 'Sobreestimacion']:
        if col not in distribucion.columns:
            distribucion[col] = 0
    distribucion = distribucion[['Subestimacion', 'Coincidencia', 'Sobreestimacion']]

    distribucion = distribucion.div(distribucion.sum(axis=1), axis=0)

    resultado_final = resultado.join(distribucion, on='Intervalos')

    return resultado_final

def calcular_precision_x_intervalos_v3(y_pred, y_real, umbral, bins):

    from sklearn import metrics

    def rmspe(y_true, y_pred):
        return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

    porc_pres = (y_pred - y_real) / y_real
    condiciones = [
        (porc_pres < -umbral),              # Subestimación
        (porc_pres >= -umbral) & (porc_pres <= umbral),  # Coincidencia
        (porc_pres > umbral)                # Sobreestimación
    ]
    clases = ['Subestimacion', 'Coincidencia', 'Sobreestimacion']

    clasificacion = np.select(condiciones, clases, default='No clasificado')
    df_precision = pd.DataFrame({
        'y_real': y_real,
        'y_predicho': y_pred,
        'porc_pres': porc_pres,
        'clas_pres': clasificacion
    })

    df_precision['Intervalos'] = pd.cut(df_precision['y_real'], bins)

    frecuencia = df_precision['Intervalos'].value_counts().sort_index()
    porcentaje = frecuencia / frecuencia.sum()
    prom_real= df_precision.groupby('Intervalos')['y_real'].mean()
    prom_pred = df_precision.groupby('Intervalos')['y_predicho'].mean()
    mae_cat = df_precision.groupby('Intervalos').apply(lambda df: metrics.mean_absolute_error(df['y_real'], df['y_predicho']) )
    mape_cat = df_precision.groupby('Intervalos').apply(lambda df: metrics.mean_absolute_percentage_error(df['y_real'], df['y_predicho']) )
    rmspe_cat = df_precision.groupby('Intervalos').apply(lambda df: rmspe(df['y_real'], df['y_predicho']) )

    resultado = pd.DataFrame({
        'Casos': frecuencia,
        'Casos (%)': porcentaje,
        'Prom. real': prom_real,
        'Prom. pred': prom_pred,
        'MAE': mae_cat,
        'MAPE': mape_cat,
        'RMSPE': rmspe_cat
    })

    distribucion = df_precision.groupby(['Intervalos', 'clas_pres']).size().unstack(fill_value=0)

    # Asegurar el orden deseado de las columnas
    for col in ['Subestimacion', 'Coincidencia', 'Sobreestimacion']:
        if col not in distribucion.columns:
            distribucion[col] = 0
    distribucion = distribucion[['Subestimacion', 'Coincidencia', 'Sobreestimacion']]

    distribucion = distribucion.div(distribucion.sum(axis=1), axis=0)

    resultado_final = resultado.join(distribucion, on='Intervalos')

    return resultado_final

def calcular_precision_x_categorias_segmento(y_pred, y_real, df_segmento, umbral):
    # Calcular el porcentaje de precisión
    porc_pres = (y_pred - y_real) / y_real

    # Definir condiciones para clasificar
    condiciones = [
        (porc_pres >= -umbral) & (porc_pres <= umbral),  # Coincidencia
        (porc_pres > umbral),                            # Sobreestimación
        (porc_pres < -umbral)                            # Subestimación
    ]
    clases = ['Coincidencia', 'Sobreestimacion', 'Subestimacion']

    # Clasificar según las condiciones
    clasificacion = np.select(condiciones, clases, default='No clasificado')

    # Crear DataFrame con resultados
    df_precision = pd.DataFrame({
        'y_real': y_real,
        'y_predicho': y_pred,
        'porc_pres': porc_pres,
        'clas_pres': clasificacion,
        'Categoría': df_segmento
    })

    # Calcular frecuencia y promedio por categoría
    frecuencia = df_precision['Categoría'].value_counts().sort_index()
    frecuencia_porc = frecuencia / frecuencia.sum()
    prom_real = df_precision.groupby('Categoría')['y_real'].mean().sort_index()
    prom_pred = df_precision.groupby('Categoría')['y_predicho'].mean().sort_index()

    resultado = pd.DataFrame({
        'Casos': frecuencia,
        'Casos (%)': frecuencia_porc,
        'Prom. real': prom_real,
        'Prom. pred.': prom_pred
    })

    # Distribución por clase de precisión
    distribucion = df_precision.groupby(['Categoría', 'clas_pres']).size().unstack(fill_value=0)

    # Asegurar columnas en orden deseado
    for col in ['Subestimacion', 'Coincidencia', 'Sobreestimacion']:
        if col not in distribucion.columns:
            distribucion[col] = 0
    distribucion = distribucion[['Subestimacion', 'Coincidencia', 'Sobreestimacion']]

    # Calcular porcentajes
    distribucion = distribucion.div(distribucion.sum(axis=1), axis=0)

    # Unir resultados
    resultado_final = resultado.join(distribucion, on='Categoría')

    return resultado_final

def Arbol_Regresion(df,X_names,y_name,porc_min_samples_leaf,omitir_nulos=True,mostrar_prop=True):
    """
    Procesa un Arbol de Regresion.
    
    :param df: DataFrame a procesar.
    :param X_names: Nombre de la/s variable/s independientes.
    :param y_name: Nombre de la variable objetivo.
    :param porc_min_samples_leaf: Porcentaje mínimo del total de las muestras en las hojas.
    :param omitir_nulos: Booleano que indica si se deben omitir los valores nulos.
    :param mostrar_prop: Booleano que indica si se quiere mostrar las muestras como proporcion.
    :return: Grafico del arbol de regresion.
    """
    df_ar = df

    if omitir_nulos:
        df_ar = df_ar.dropna(subset=X_names)
    else:
        df_ar

    X = df_ar[X_names]
    y = df_ar[y_name]

    regressor = DecisionTreeClassifier(random_state = seed,max_depth=5, min_samples_leaf = int(len(X) * porc_min_samples_leaf))
    regressor.fit(X, y)
    # Gráfico
    fig, ax = plt.subplots(figsize=(20,10))

    print(f"Profundidad del árbol: {regressor.get_depth()}")
    print(f"Número de nodos terminales: {regressor.get_n_leaves()}")

    plot = plot_tree(
                decision_tree = regressor,
                feature_names = X.columns,
                class_names   = [y.name],
                proportion    = mostrar_prop,
                filled        = True,
                impurity      = False,
                fontsize      = 10,
                precision     = 8,
                ax            = ax
            )   

def box_plot_outliers(data,list_var_num):
    # Crear subplots para mostrar gráficos al lado de otro con un límite de hasta seis boxplots por fila
    df = data[list_var_num]
    variables = df.columns
    num_vars = len(variables)
    num_cols = min(6, num_vars)
    num_rows = (num_vars + num_cols - 1) // num_cols

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, num_rows * 3.5))

    # Asegurarse de que axes sea un array bidimensional
    axes = np.array(axes).reshape(num_rows, num_cols)

    for i, variable in enumerate(variables):
        row = i // num_cols
        col = i % num_cols
        df.boxplot(column=variable, ax=axes[row, col])
        axes[row, col].set_title(f'{variable}')
        axes[row, col].set_xlabel('Variables')
        axes[row, col].set_ylabel('Valores')

    # Eliminar subplots vacíos si hay menos variables que subplots
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.show()

def metricas_clasificacion(y_train_real, y_train_predicho, y_test_real, y_test_predicho, plot_confusion_matrix=False):
    # Train
    accuracy_train = metrics.accuracy_score(y_train_real, y_train_predicho)
    auc_train = roc_auc_score(y_train_real, y_train_predicho)
    precision_train = metrics.precision_score(y_train_real, y_train_predicho, average='weighted')
    recall_train = metrics.recall_score(y_train_real, y_train_predicho, average='weighted')
    f1_train = metrics.f1_score(y_train_real, y_train_predicho, average='weighted')
    kappa_train = cohen_kappa_score(y_train_real, y_train_predicho)
    gini_train = 2 * auc_train - 1
    
    # Test
    accuracy_test = metrics.accuracy_score(y_test_real, y_test_predicho)
    auc_test = roc_auc_score(y_test_real, y_test_predicho)
    precision_test = metrics.precision_score(y_test_real, y_test_predicho, average='weighted')
    recall_test = metrics.recall_score(y_test_real, y_test_predicho, average='weighted')
    f1_test = metrics.f1_score(y_test_real, y_test_predicho, average='weighted')
    kappa_test = cohen_kappa_score(y_test_real, y_test_predicho)
    gini_test = 2 * auc_test - 1
    
    # Resumen
    resultados = {
        'Accuracy': [accuracy_train, accuracy_test],
        'AUC': [auc_train, auc_test],
        'Recall': [recall_train, recall_test],
        'Precision': [precision_train, precision_test],
        'F1 Score': [f1_train, f1_test],
        'Kappa': [kappa_train, kappa_test],
        'Gini': [gini_train, gini_test]
    }
    resultados = pd.DataFrame(resultados, index=['Train', 'Test'])
    
    if plot_confusion_matrix:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        # Primer gráfico: Matriz de confusión para el conjunto de entrenamiento
        metrics.plot_confusion_matrix(y_train_real, y_train_predicho, ax=axs[0], cmap='Blues')
        axs[0].set_title('Train Confusion Matrix')
        
        # Segundo gráfico: Matriz de confusión para el conjunto de prueba
        metrics.plot_confusion_matrix(y_test_real, y_test_predicho, ax=axs[1], cmap='Blues')
        axs[1].set_title('Test Confusion Matrix')
        
        plt.tight_layout()
    
    return resultados

def calcular_casos_x_deciles(df, variable, puntos_de_corte=None):
    tiene_nulos = df[variable].isnull().any()
    
    # Verificar si la variable es categórica
    if df[variable].dtype == 'object' or df[variable].dtype.name == 'category':
        puntos_de_corte = df[variable].unique().tolist()
        cantidad_por_intervalo = df[variable].value_counts().reset_index()
        cantidad_por_intervalo.columns = [variable, 'Cantidad']
    else:
        # Calcular los puntos de corte para los deciles si no se proporcionan
        if puntos_de_corte is None:
            puntos_de_corte = df[variable].quantile([0.1 * i for i in range(11)]).tolist()
            puntos_de_corte = list(set(puntos_de_corte))
            puntos_de_corte.sort()
        
        # Calcular la cantidad de casos por cada intervalo
        intervalos = pd.cut(df[variable], bins=puntos_de_corte, right=False)
        cantidad_por_intervalo = df.groupby(intervalos).size().reset_index(name='Cantidad')
    
    df_dec = pd.DataFrame({
        'Intervalo': cantidad_por_intervalo[variable].astype(str),
        'Casos': cantidad_por_intervalo['Cantidad'],
        'Casos(%)': cantidad_por_intervalo['Cantidad'] / len(df),
    })
    
    if tiene_nulos:
        cantidad_nulos = df[variable].isnull().sum()
        df_nulos = pd.DataFrame({'Intervalo': ['Nulos'], 'Casos': cantidad_nulos, 'Casos(%)': cantidad_nulos / len(df)})
        df_dec = pd.concat([df_nulos, df_dec], ignore_index=True)
    
    return df_dec

def calcular_psi_x_intervalo(df1, df2, variable, puntos_de_corte=None):
    # Verificar si la variable es categórica
    if df1[variable].dtype == 'object' or df1[variable].dtype.name == 'category':
        puntos_de_corte = df1[variable].unique().tolist()
    else:
        # Calcular los puntos de corte para los deciles si no se proporcionan
        if puntos_de_corte is None:
            puntos_de_corte = df1[variable].quantile([0.1 * i for i in range(11)]).tolist()
            puntos_de_corte = list(set(puntos_de_corte))  # Para eliminar posibles duplicados
            puntos_de_corte.sort()

    prop_1 = calcular_casos_x_deciles(df1, variable, puntos_de_corte)['Casos(%)']
    prop_2 = calcular_casos_x_deciles(df2, variable, puntos_de_corte)['Casos(%)']

    df_psi = pd.DataFrame({
        'Intervalo': calcular_casos_x_deciles(df1, variable, puntos_de_corte)['Intervalo'].astype(str),
        'PSI': (prop_1 - prop_2) * np.log(prop_1 / prop_2)
    })

    suma_psi = df_psi['PSI'].sum()

    total_row = pd.DataFrame({'Intervalo': ['Total'], 'PSI': [suma_psi]})
    df_psi = pd.concat([df_psi, total_row], ignore_index=True)

    return df_psi

def obtener_psi_variables(df1, df2, lista_var):
    lista_psi = []
    for variable in lista_var:
        psi = calcular_psi_x_intervalo(df1, df2, variable)['PSI'].iloc[-1]
        lista_psi.append(psi)

    return pd.DataFrame({'Variable':df1[lista_var].columns,'PSI':lista_psi})

def graficar_psi_variables(df1, df2, lista_var):
    lista_psi = []
    for variable in lista_var:
        psi = calcular_psi_x_intervalo(df1, df2, variable)['PSI'].iloc[-1]
        lista_psi.append(psi)

    psi_df = pd.DataFrame({'Variable':df1[lista_var].columns,'PSI':lista_psi})
    #psi_df['PSI'] = pd.to_numeric(psi_df['PSI'], errors='coerce').fillna(0)

    # Grafico 
    max_psi = psi_df['PSI'].max()
    if np.isfinite(max_psi):
        x_limit = max(0.3, max_psi * 1.1)
    else:
        x_limit = 0.3

    plt.figure(figsize=(10, 6))
    bars = plt.barh(psi_df['Variable'], psi_df['PSI'], color='skyblue')
    plt.axvline(x=0.1, color='orange', linestyle='--', label='Umbral Moderado')
    plt.axvline(x=0.25, color='red', linestyle='--', label='Umbral Alto')
    plt.title('Population Stability Index (PSI) por Variable')
    plt.xlabel('PSI')
    plt.ylabel('Variables')
    plt.legend(loc='upper right')
    plt.xlim(0, x_limit)

    # Agregar etiquetas a las barras
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f'{width:.4f}', ha='center', va='center')

    plt.tight_layout()
    plt.show()
        
    #return lista_psi

# def eliminacion_recursiva_de_variables(X, y, model, list_num_features, umbral_coinc):
    
#     rfe_0 = RFE(estimator=model, n_features_to_select = max(list_num_features))
#     rfe_0.fit(X, y)

#     variables_seleccionadas_0 = list(X.columns[rfe_0.support_])

#     X = X[variables_seleccionadas_0]

#     list_num_features_ord = sorted(list_num_features, reverse=True)
#     list_num_features_ord.remove(max(list_num_features))
#     lista_var_elim = []

#     df_total = []

#     for i in tqdm_notebook(list_num_features_ord):

#         rfe = RFE(estimator=model, n_features_to_select=i)

#         rfe.fit(X, y)

#         X_train_selected = rfe.transform(X)

#         model.fit(X_train_selected, y)

#         y_pred_train = np.exp(model.predict(X_train_selected))

#         df_metricas_rfe = metricas_regresion(np.exp(y), y_pred_train,np.exp(y), y_pred_train).loc[['Train']]
#         df_coincidencias_rfe = calcular_coincidencia(y_pred_train, y_pred_train, np.exp(y), np.exp(y), umbral_coinc ).loc[['Train']]

#         variables_eliminadas = list(X.columns[~rfe.support_])
#         ultima_variable_eliminada = list(set(variables_eliminadas) - set(lista_var_elim))
#         lista_var_elim.extend(ultima_variable_eliminada)

#         df_num_var = pd.DataFrame({"Num_Variables":i,"Var_Eliminada":ultima_variable_eliminada}, index = ['Train'])
#         df_concat_metricas = pd.concat([df_num_var,df_metricas_rfe, df_coincidencias_rfe], axis=1)

#         df_total.append(df_concat_metricas)
    
#     # Concatenando todos los dataframes de la lista
#     df_total_df = pd.concat(df_total, axis = 0, ignore_index=True)
#     # Invirtiendo el orden de las filas
#     df_total_df = df_total_df.iloc[::-1].reset_index(drop=True)

#     print(f'Número de variables seleccionadas: {rfe.n_features_}')
#     variables_seleccionadas = list(X.columns[rfe.support_])
#     print('\nVariables seleccionadas:', variables_seleccionadas)
#     display(df_total_df)


def metricas_clasificacion_binaria(y_real, y_proba, umbral=0.5):
    y_real = np.asarray(y_real)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= umbral).astype(int)

    def _calc(nombre, f):
        try:
            return float(f())
        except Exception as e:
            warnings.warn(f"No se pudo calcular {nombre}: {e}", RuntimeWarning)
            return np.nan

    auc = _calc("AUC", lambda: metrics.roc_auc_score(y_real, y_proba))

    def _ks():
        fpr, tpr, _ = metrics.roc_curve(y_real, y_proba)
        return np.max(tpr - fpr)

    return {
        "AUC": auc,
        "PR_AUC": _calc("PR_AUC", lambda: metrics.average_precision_score(y_real, y_proba)),
        "Accuracy": _calc("Accuracy", lambda: metrics.accuracy_score(y_real, y_pred)),
        "F1": _calc("F1", lambda: metrics.f1_score(y_real, y_pred, zero_division=0)),
        "Precision": _calc("Precision", lambda: metrics.precision_score(y_real, y_pred, zero_division=0)),
        "Recall": _calc("Recall", lambda: metrics.recall_score(y_real, y_pred, zero_division=0)),
        "Gini": 2 * auc - 1,   # si el AUC falla, el Gini también queda en NaN
        "KS": _calc("KS", _ks),
    }


def eliminacion_recursiva_de_variables(X, y, model, list_num_features, umbral=0.5, X_val=None, y_val=None,
                                       experiment_name=None, run_name="RFE", tracking_uri=None):

    registrar = experiment_name is not None
    if registrar:
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)  # run padre
        mlflow.log_params({"umbral": umbral,
                           "n_variables_inicial": X.shape[1],
                           "n_filas": X.shape[0],
                           "con_validacion": X_val is not None})
    
    try:
        rfe_0 = RFE(estimator=model, n_features_to_select=max(list_num_features))
        rfe_0.fit(X, y)
        X = X[list(X.columns[rfe_0.support_])]

        list_num_features_ord = sorted(list_num_features, reverse=True)
        list_num_features_ord.remove(max(list_num_features))
        lista_var_elim = []
        df_total = []

        for i in tqdm_notebook(list_num_features_ord):

            rfe = RFE(estimator=model, n_features_to_select=i)
            rfe.fit(X, y)

            cols_sel = list(X.columns[rfe.support_])
            model.fit(X[cols_sel], y)

            # Metricas
            metr = {f"Train_{k}": v for k, v in metricas_clasificacion_binaria(y, model.predict_proba(X[cols_sel])[:, 1], umbral).items()}
            if X_val is not None:
                metr.update({f"Val_{k}": v for k, v in metricas_clasificacion_binaria(y_val, model.predict_proba(X_val[cols_sel])[:, 1], umbral).items()})
            
            variables_eliminadas = list(X.columns[~rfe.support_])
            ultima_variable_eliminada = list(set(variables_eliminadas) - set(lista_var_elim))
            lista_var_elim.extend(ultima_variable_eliminada)

            df_total.append(pd.DataFrame([{"Num_Variables": i,
                                           "Var_Eliminada": ", ".join(ultima_variable_eliminada),
                                           **metr}]))

            # Registro en MLflow
            if registrar:
                with mlflow.start_run(run_name=f"{i}_variables", nested=True):
                    mlflow.log_param("num_variables", i)
                    mlflow.log_param("var_eliminada", ", ".join(ultima_variable_eliminada))
                    mlflow.log_dict({"variables": cols_sel}, "variables_seleccionadas.json")
                    #mlflow.log_metrics({re.sub(r"[^\w\-\. :/]", "_", k): float(v) for k, v in metr.items()})
                    mlflow.log_metrics({re.sub(r"[^\w\-\. :/]", "_", k): float(v) for k, v in metr.items() if np.isfinite(v)})
                    mlflow.catboost.log_model(model, artifact_path="model")
                    
        df_total_df = pd.concat(df_total, axis=0, ignore_index=True)
        df_total_df = df_total_df.iloc[::-1].reset_index(drop=True)

        print(f'Número de variables seleccionadas: {rfe.n_features_}')
        print('\nVariables seleccionadas:', cols_sel)
        display(df_total_df)
        return df_total_df

    finally:
        if registrar:
            mlflow.end_run()  # cierra el run padre

    