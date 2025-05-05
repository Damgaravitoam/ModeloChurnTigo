import json
import pandas as pd
from inference import model_fn, input_fn, predict_fn, output_fn

# Cargar el modelo
model = model_fn(".")  # 🔥 Asegúrate de que `best_model.pkl` está en la ruta correcta

# Leer el CSV y convertirlo a JSON
df = pd.read_csv("df_muestra.csv")  # 🔥 Cambia la ruta si es necesario
payload = df.to_json(orient="records")  # 🔥 Convierte el DataFrame en formato JSON

# Procesar entrada
input_data = input_fn(payload, "application/json")

# Obtener predicción
prediction = predict_fn(input_data, model)

# Formatear salida
output = output_fn(prediction, "application/json")

print("🔮 Predicciones:", output)
