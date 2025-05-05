import numpy as np
import pandas as pd
import lightgbm as lgb
import sklearn
import os
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

#os.chdir("/home/sagemaker-user/")  # 🔥 Mueve el directorio de trabajo
print("✅ Ahora transform_functions está en:", os.getcwd())  # Confirma el cambio

script_path = os.path.abspath(__file__)  # 🔥 Ruta absoluta del script
script_dir = os.path.dirname(script_path)  # 🔥 Directorio del script

print("📌 Este script está en:", script_dir)

def NonNullValues(df):
    
    df['PQT_MAS_COMPRADO'] = df['PQT_MAS_COMPRADO'].astype('category')
    df = df.replace([None, 'nan', 'NaN'], np.nan)
    
    Null_Cols = ['DIAS_INACTIVIDAD', 'QTY_PQT', 'PQT_MAS_COMPRADO', 'qty_pqt_M.1', 'pqt_mas_comprado_M.1', 'qty_pqt_M.2', 'pqt_mas_comprado_M.2', 'qty_pqt_M.3', 'pqt_mas_comprado_M.3']
    
    null_numeric_cols = [col for col in Null_Cols if df[col].dtypes in ['float64', 'int64']]
    null_category_cols = [col for col in Null_Cols if df[col].dtypes in ['category']]
    
    for col_categorica in null_category_cols:
        df[col_categorica] = df[col_categorica].cat.add_categories(['MISSING'])
        df[col_categorica] = df[col_categorica].fillna('MISSING')

    for col_numerica in null_numeric_cols:
        df[col_numerica] = df[col_numerica].fillna(0)
    
    # Seleccionar las columnas de tipo 'object'
    df_churn_object_cols = [col for col in df.columns if df[col].dtypes == 'object']

    # Convertir las columnas seleccionadas a tipo 'category'
    for col in df_churn_object_cols:
        df[col] = df[col].astype('category')

    ACTIVATION_CHANNEL_SELCATS = ['DISTRIBUIDORES', 'TIENDAS_PROPIAS', 'RETAIL', 'FUERZA_VD', 'AUTOACTIVACION', 'CORPORATIVO', 'TO BE DETERMINED', 'INSTITUCIONAL', 'E-COMMERCE', 'TELEVENTAS', 'TIENDA_FVD']
    
    # Reemplazar valores nulos con 'MISSING'
    df['ACTIVATION_CHANNEL'] = df['ACTIVATION_CHANNEL'].cat.add_categories(['MISSING'])
    df['ACTIVATION_CHANNEL'] = df['ACTIVATION_CHANNEL'].fillna('MISSING')
    
    # Creacion de la nueva variable ACTIVATION_CHANNEL_MOD
    df['ACTIVATION_CHANNEL_MOD'] = df['ACTIVATION_CHANNEL'].apply(lambda x: x if x in ACTIVATION_CHANNEL_SELCATS else 'OTROS')
    df['ACTIVATION_CHANNEL_MOD'] = df['ACTIVATION_CHANNEL_MOD'].astype('category')

    return df

# Crear una función de tendencia
def new_vars_trend(df, col_trs, var_list):

    # A las variables de paquetes adquiridos se pone todo en minúsculas
    var_list = ['PQT_MAS_COMPRADO', 'pqt_mas_comprado_M.1', 'pqt_mas_comprado_M.2', 'pqt_mas_comprado_M.3']
    
    for var in var_list:
        df[var] = df[var].astype(str).str.lower()

    vars_PQT_MAS_COMPRADO = ['missing', ' fb 7d atck -$5000', 'voz ilimitada ', ' fb 30d -$15000']
    vars_PQT_MAS_COMPRADO_M1 = ['missing', 'voz ilimitada ', ' fb 5d -$3000', ' fb 7d atck -$5000', ' fb 30d -$15000']
    vars_PQT_MAS_COMPRADO_M2 = ['missing', 'voz ilimitada ', ' fb 7d atck -$5000']
    vars_PQT_MAS_COMPRADO_M3 = ['missing', 'voz ilimitada ', ' fb 7d atck -$5000', ' fb 5d -$3000', ' fb 30d -$15000']
    
    # Diccionario con las variables y listas correspondientes
    vars_dict = {
        'PQT_MAS_COMPRADO': vars_PQT_MAS_COMPRADO,
        'pqt_mas_comprado_M.1': vars_PQT_MAS_COMPRADO_M1,
        'pqt_mas_comprado_M.2': vars_PQT_MAS_COMPRADO_M2,
        'pqt_mas_comprado_M.3': vars_PQT_MAS_COMPRADO_M3
    }
    
    # Construcción de las variables de servicios adquiridos por paquete
    for var, elementos in vars_dict.items():
        for elemento in elementos:
            df[f'{elemento}_{var.split(".")[1] if "." in var else "M"}'] = df[var].apply(lambda x: 1 if elemento in str(x) else 0).astype(bool)

    
    for col in col_trs:
        # Valores de tiempo (independientes) y lista de variables
        time = [3, 2, 1]
        vars_time = [f'{col}_M-1', f'{col}_M-2', f'{col}_M-3']
        
        # Valores de [f'PCA_{num}' for num in Vars_PCA]
        rev_total = [df[f'{col}_M-1'], df[f'{col}_M-2'], df[f'{col}_M-3']]
        # Calcular la pendiente usando np.polyfit
        coeficientes = np.polyfit(time, rev_total, 1)
        
        df[f'{col}_PEND'] = coeficientes[0]
        df['ratio_M2_M3'] = np.nan_to_num(df[f'{col}_M-2'] / df[f'{col}_M-3'], nan=0, posinf=0, neginf=0)
        df['ratio_M1_M2'] = np.nan_to_num(df[f'{col}_M-1'] / df[f'{col}_M-2'], nan=0, posinf=0, neginf=0)
        df[f'{col}_TEND'] = np.where((df[f'{col}_PEND'] > 0) & (df['ratio_M2_M3'] > 1.4) | (df['ratio_M1_M2']  > 1.4), 'AUMENTA', 
                                        np.where((df[f'{col}_PEND'] < 0) & ((df['ratio_M2_M3'] > 0) & (df['ratio_M2_M3'] < 0.6)) | ((df['ratio_M1_M2'] > 0) & (df['ratio_M1_M2']  < 0.6)), 'DISMINUYE', 'IGUAL')
                                       )
        df[f'{col}_MEAN_3M'] = df[vars_time].mean(axis = 1)
        df[f'{col}_MIN_3M'] = df[vars_time].min(axis = 1)
        df[f'{col}_MAX_3M'] = df[vars_time].max(axis = 1)
        
        df = df.drop(columns=[f'{col}_PEND', 'ratio_M2_M3', 'ratio_M1_M2'])
    
    return df

#def custom_combiner(feature_name, category):
#    return f"{feature_name}.{category}"

def encoder_scaler_loading(df_churn_new_vars): 
    
    Selected_Cols_loaded = np.loadtxt("/opt/ml/code/Listas_mod/Selected_Cols.csv", delimiter=",", dtype=str)
    df_churn_red = df_churn_new_vars[Selected_Cols_loaded] 
    
    # Carga de los escaladores para variables numericas
    
    # Crear un diccionario para almacenar los escaladores cargados
    scalers_loaded = {}
    
    # Ruta de la carpeta donde se guardaron los escaladores
    scalers_folder = "/opt/ml/code/scalers_folder"
    
    # Cargar cada escalador guardado
    for file in os.listdir(scalers_folder):
        if file.endswith("_scaler.pkl"):  # Asegurar que solo se cargan archivos de escaladores
            col_name = file.replace("_scaler.pkl", "")  # Obtener el nombre de la columna
            scalers_loaded[col_name] = joblib.load(os.path.join(scalers_folder, file))
    
    print("Escaladores cargados correctamente.")

    # Crear un diccionario para almacenar los codificadores cargados
    encoders_loaded = {}
    
    # Ruta de la carpeta donde se guardaron los encoders
    encoders_folder = "/opt/ml/code/encoders_folder"
    
    # Cargar cada codificador guardado
    for file in os.listdir(encoders_folder):
        if file.endswith("_encoder.pkl"):  # Asegurar que solo se cargan archivos de encoders
            col_name = file.replace("_encoder.pkl", "")  # Obtener el nombre de la columna
            encoders_loaded[col_name] = joblib.load(os.path.join(encoders_folder, file))
    
    print("Codificadores cargados correctamente.")

    # Escalar una nueva columna usando el escalador guardado
    df_churn_sc = df_churn_red.copy()
    for col, scaler in scalers_loaded.items():
        if col in df_churn_sc.columns:  # Verificar que la columna exista en los nuevos datos
            df_churn_sc[col] = scaler.transform(df_churn_sc[[col]])
    
    # Codificar una nueva columna usando el encoder guardado

    for col, encoder in encoders_loaded.items():
        if col in df_churn_sc.columns:
            encoded_values = encoder.transform(df_churn_sc[[col]].astype(str))
            encoded_df = pd.DataFrame(encoded_values, columns=encoder.get_feature_names_out([col]), index=df_churn_sc.index)
            df_churn_sc = df_churn_sc.drop(columns=[col]).join(encoded_df)
    
    print("Transformaciones aplicadas correctamente.")
    return df_churn_red, df_churn_sc

def fun_final_base(DF_Original):

    var_list_pqt_loaded = np.loadtxt("/opt/ml/code/Listas_mod/var_list_pqt.csv", delimiter=",", dtype=str)
    col_trans_loaded = np.loadtxt("/opt/ml/code/Listas_mod/col_trans.csv", delimiter=",", dtype=str)
    
    df_new_vars = new_vars_trend(DF_Original, col_trans_loaded, var_list_pqt_loaded)
    df_new_vars = df_new_vars.drop(var_list_pqt_loaded, axis = 1)

    X_red, X_sc = encoder_scaler_loading(df_new_vars)
    X_sc_PCA_Pol = transf_PCA_POL(X_sc)

    X_Final = XD_Train_Final(X_sc, X_sc_PCA_Pol)

    Orden_Columnas_loaded = np.loadtxt("/opt/ml/code/Listas_mod/Orden_Columnas.csv", delimiter=";", dtype=str, usecols=[0], encoding="utf-8")
    XDF_Final = X_Final[Orden_Columnas_loaded]

    return X_red, X_sc, XDF_Final

def XD_Train_Final(XDF_sc, XDF_sc_PCA_Pol):

    XDF_columns_Final_loaded = np.loadtxt("/opt/ml/code/Listas_mod/XDF_columns_Final.csv", delimiter=",", dtype=str, usecols=[0], encoding="utf-8")

    # Igualar los indixes
    XDF_sc_PCA_Pol.index = XDF_sc.index 
    
    # Identificar columnas repetidas
    common_columns = XDF_sc.columns.intersection(XDF_sc_PCA_Pol.columns)
    
    # Eliminar las columnas duplicadas de XTrain_sc_PCA_Pol
    XDF_sc_PCA_Pol_clean = XDF_sc_PCA_Pol.drop(columns=common_columns)
    
    # Concatenar los DataFrames sin repetir columnas
    XDFTrain_Final = pd.concat([XDF_sc, XDF_sc_PCA_Pol_clean], axis=1)
    
    print("DataFrame unido sin columnas duplicadas:", XDFTrain_Final.shape)

    return XDFTrain_Final

def const_vars_serv(df, var):

    # Separar el contenido de la variable 'PQT_MAS_COMPRADO' por el signo "+"
    df_prueba = df[[var, 'CHURN']].copy()
    
    df_prueba[var] = df_prueba[var].str.lower()
    
    df_split = df_prueba[var].str.split('+', expand=True)
    
    df_split.columns = ['Componente_1', 'Componente_2', 'Componente_3', 'Componente_4', 'Componente_5']
    
    # Crear un diccionario para almacenar los resultados
    conteos_categorias = {}
    
    # Iterar sobre las columnas y guardar los conteos en el diccionario
    for col in ['Componente_1', 'Componente_2', 'Componente_3', 'Componente_4', 'Componente_5']:
        print(f"Conteo de categorías para {col}:")
        conteos_categorias[col + '_list'] = df_split[col].value_counts()  # Guardar el conteo en el diccionario
        print(conteos_categorias[col + '_list'])  # Mostrar el conteo
        print("\n")
    
    # Crear una única lista con todos los elementos de las 5 listas
    todos_los_elementos = []
    
    for key, conteo in conteos_categorias.items():
        todos_los_elementos.extend(conteo.index)  # Agregar los índices (categorías) de cada serie a la lista
    
    # Remover duplicados, si es necesario
    todos_los_elementos = list(set(todos_los_elementos))
    
    # Mostrar la lista resultante
    todos_los_elementos
        
    for elemento in todos_los_elementos:
        df_prueba[elemento] = df_prueba[var].apply(lambda x: 1 if elemento in str(x) else 0)

    # Modificaciones inciales para lectura correcta

    XTrain_mod_arbol = df_prueba.drop([var, 'CHURN'], axis = 1)
    yTrain = df_prueba['CHURN']
    
    # Guardar nombres originales de las columnas
    nombres_originales = XTrain_mod_arbol.columns.tolist()
    
    # Crear un DataFrame a partir de la lista nombres_originales
    df_nombres = pd.DataFrame(nombres_originales, columns=['Nombre'])
    
    # Agregar una columna 'Col_Indice' con números del 0 al total de columnas
    df_nombres['Feature'] = range(len(XTrain_mod_arbol.columns))
    
    # Asegúrate de que las longitudes coincidan
    if len(df_nombres) == len(XTrain_mod_arbol.columns):
        # Reemplazar 'Col_Indice' con los nombres de las columnas de XTrain_mod_arbol
        XTrain_mod_arbol.columns = df_nombres['Feature']
    else:
        print("Error: Las longitudes de df_nombres y las columnas de XTrain_mod_arbol no coinciden.")

    # Aplicacion del modelo LightGBM para encontrar los mejores parametros y variables
    print("---------- Comienza Ejecucion modelo LightGBM ----------")

    # Identificar columnas con tipo 'object'

    grid_search_lgbm = modelo_LightGBM(XTrain_mod_arbol, yTrain)

    print("---------- Finaliza Ejecucion modelo LightGBM ----------")

    # Obtener el modelo óptimo ajustado por GridSearchCV
    best_lgbm_model = grid_search_lgbm.best_estimator_
    
    # Extraer las importancias de las características
    feature_importances = best_lgbm_model.feature_importances_
    
    # Suponiendo que tienes una lista de nombres de las características
    feature_names = XTrain_mod_arbol.columns if isinstance(XTrain_mod_arbol, pd.DataFrame) else [f"Feature {i}" for i in range(len(feature_importances))]
    
    # Crear un DataFrame para ordenar y visualizar
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)
    
    importance_df = importance_df.merge(df_nombres, on = 'Feature', how = 'left')
    corte = (importance_df[importance_df['Importance'] != 0]['Importance']).mean()
    
    selvars = importance_df[importance_df['Importance'] >= corte]
    selvars = selvars['Nombre'].tolist()

    print("---------- Las variables selecionadas por servicios adquiridos son ----------")

    print(selvars)

    importance_df_red = importance_df[importance_df['Nombre'].isin(selvars)]
    
    # Visualizar con un gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df_red["Nombre"], importance_df_red["Importance"], color="skyblue")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance (LightGBM)")
    plt.gca().invert_yaxis()  # Invertir el eje para mostrar la más importante arriba
    plt.show()
    
    return selvars

def transf_PCA_POL(XDF_sc):
    
    # Carga de listas de variables y transformadores
    pf = joblib.load("/opt/ml/code/Listas_mod/polyFeature_model.pkl")
    num_cols_loaded = np.loadtxt("/opt/ml/code/Listas_mod/num_cols.csv", delimiter=",", dtype=str)
    
    VarSel_PCA_F_loaded = np.loadtxt("/opt/ml/code/Listas_mod/VarSel_PCA_F.csv", delimiter=",")
    VarSel_PCA_F_loaded = VarSel_PCA_F_loaded.astype(int)
    VarSel_PCA_F_loaded = np.array(VarSel_PCA_F_loaded)
    
    # Asegurar que `VarSel_PCA_F_loaded` sea un array NumPy
    if isinstance(VarSel_PCA_F_loaded, (int, float)):  # Si es un número escalar, conviértelo a una lista
        VarSel_PCA_F_loaded = [VarSel_PCA_F_loaded]
    elif isinstance(VarSel_PCA_F_loaded, np.ndarray) and VarSel_PCA_F_loaded.ndim == 0:  # Si es un array de dimensión 0, conviértelo en un array de dimensión 1
        VarSel_PCA_F_loaded = np.array([VarSel_PCA_F_loaded])
    
    selvars_pol_loaded = np.loadtxt("/opt/ml/code/Listas_mod/selvars_pol.csv", delimiter=",", dtype=str)

    pca_loaded = joblib.load("/opt/ml/code/Listas_mod/pca_model.pkl")
    pf_loaded = joblib.load("/opt/ml/code/Listas_mod/polyFeature_model.pkl")

    # Construcción de base con las variables de PCA
    
    XDF_sc_num = XDF_sc[num_cols_loaded]

    #X_pca_test = pca_transf.transform(XTest_sc_num)
    #ZTest = X_pca_test[:, :n_componentes_opt_lr]
    
    X_pca = pca_loaded.transform(XDF_sc_num)
    X_pca_selected = X_pca[:, VarSel_PCA_F_loaded - 1]
    XDF_PCA = pd.DataFrame(X_pca_selected)
    column_names = [f'PCA_{num}' for num in VarSel_PCA_F_loaded]
    XDF_PCA.columns = column_names

    # Construcción de base con las variables de PCA
    
    X_pol = pf_loaded.transform(XDF_sc_num)
    XDF_pol = pd.DataFrame(X_pol, columns=pf.get_feature_names_out())
    XDF_pol = XDF_pol[selvars_pol_loaded]

    XDF_Final = pd.concat([XDF_PCA, XDF_pol], axis=1)
    
    return XDF_Final