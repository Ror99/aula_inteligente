import joblib
import os

modelo_path = '../ia_modelo/modelo_rendimiento.pkl'
scaler_path = '../ia_modelo/scaler.pkl'

try:
    model = joblib.load(modelo_path)
    scaler = joblib.load(scaler_path)
    print("✅ Modelo y escalador cargados correctamente")
except Exception as e:
    print("❌ Error al cargar:", str(e))