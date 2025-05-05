import joblib
import os
import json
import numpy as np
import pandas as pd
from flask import Flask

from transform_functions import fun_final_base, NonNullValues, new_vars_trend, new_vars_trend, encoder_scaler_loading, const_vars_serv, transf_PCA_POL

def custom_combiner(feature_name, category):
    return f"{feature_name}.{category}"

# Lista de columnas esperadas por el modelo
FEATURES = ['REV_OUT_INFORMATION_MAX_3M', 'CONSUMO_DATOS', 'REV_OUT_INFORMATION', 'MINUTES_IN', 'DIAS_INACTIVIDAD_DIAS_INACTIVIDAD_M-1', 'DIAS_INACTIVIDAD_AGEING', 'AGEING_REV_TOTAL', 'REV_OUT_COMMUNICATION_QTY_RCHG', 'MINUTES_OUT', 'REV_OUT_INFORMATION_QTY_RCHG', 'REV_OUT_COMMUNICATION', 'CALLS_IN', 'CONSUMO_DATOS_MAX_3M', 'REV_IN', 'MSG_OUT', 'REV_OUT_COMMUNICATION_DIAS_INACTIVIDAD_M-3', 'MSG_OUT_M-3', 'DIAS_INACTIVIDAD', 'DIAS_INACTIVIDAD_M-1', 'REV_OUT_INFORMATION_AMNT_RCHG', 'CONSUMO_DATOS_M-1', 'DIAS_INACTIVIDAD_M-2', 'MSG_IN', 'CALLS_OUT', 'QTY_PQT', 'AGEING_REV_OUT_SOLUTIONS', 'REV_OUT_ENTERTAIMENT', 'QTY_RCHG', 'QTY_RCHG_REV_OUT', 'PCA_1', 'REV_TOTAL', 'REV_OUT_INFORMATION_REV_OUT_COMMUNICATION', 'qty_pqt_M.2', 'qty_pqt_M.1', 'REV_OUT_SOLUTIONS', 'qty_pqt_M.3', 'REV_OUT_INFORMATION_M-3', 'MSG_OUT_M-1', 'AGEING', 'REV_OUT']

var_list_pqt = ['PQT_MAS_COMPRADO', 'pqt_mas_comprado_M.1', 'pqt_mas_comprado_M.2', 'pqt_mas_comprado_M.3']

col_trans = ['QTY_RCHG', 'AMNT_RCHG', 'REV_TOTAL', 'REV_IN', 'REV_OUT', 'CALLS_IN', 'MINUTES_IN', 'MSG_IN', 'CALLS_OUT', 'MINUTES_OUT', 'MSG_OUT', 'REV_OUT_COMMUNICATION', 'REV_OUT_ENTERTAIMENT', 'REV_OUT_INFORMATION', 'REV_OUT_SOLUTIONS', 'CONSUMO_DATOS', 'DIAS_INACTIVIDAD']

# Def de un PING 

from flask import Flask

app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


def model_fn(model_dir):
    """Carga el modelo cuando se inicia el endpoint"""
    model_path = os.path.join(model_dir, "best_model.pkl")
    globals()["custom_combiner"] = custom_combiner
    model = joblib.load(model_path)
    print("✅ Modelo cargado correctamente en SageMaker")
    print(model.booster_.feature_name())
    return model

def preprocess_input(data):
    """Transforma los datos de entrada para que el modelo reciba solo las variables correctas"""
    #df = pd.DataFrame(data)  # 🔥 Convierte el JSON en DataFrame
    df = NonNullValues(data)
    df_red, df_sc, df_Final = fun_final_base(df)
    df_Final_1 = df_Final[FEATURES] # 🔥 Filtra solo las columnas esperadas por el modelo 
    return df_Final_1.values  # 🔥 Convierte a matriz numpy  

def input_fn(serialized_input_data, content_type):
    """Convierte la entrada JSON en formato adecuado"""
    if content_type == "application/json":
        data = json.loads(serialized_input_data)
        df = pd.DataFrame(data)
        transformed_data = preprocess_input(df)  # 🔥 Aplica transformación antes de la inferencia  
        return transformed_data
    raise ValueError(f"Tipo de datos no soportado: {content_type}")

def predict_fn(input_data, model):
    """Hace predicciones con el modelo"""
    prediction = model.predict(input_data)
    return prediction.tolist()

def output_fn(prediction, accept):
    """Convierte la salida en formato JSON"""
    if accept == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Formato de salida no soportado: {accept}")
