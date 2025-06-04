import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import joblib
import numpy as np
import requests
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración desde .env
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
MODEL_PATH = "modelo_rendimiento.pkl"
SCALER_PATH = "scaler.pkl"

# Cargar modelo y escalador
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Error al cargar los modelos: {e}")

# Inicializar app
app = FastAPI(
    title="Aula Inteligente - Mejorado",
    description="Con alertas tempranas, predicciones por materia y recomendaciones personalizadas",
    version="1.0"
)

# Esquema de respuesta
class PrediccionMateria(BaseModel):
    materia: str
    promedio_predicho: float
    alerta: bool

class PrediccionDetalladaResponse(BaseModel):
    student_id: str
    promedio_global_predicho: float
    alerta_riesgo: bool
    predicciones_por_materia: List[PrediccionMateria]
    recomendaciones: List[str]


def obtener_datos_alumno(student_id: str) -> dict:
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}"
    }

    profile_url = f"{SUPABASE_URL}/rest/v1/profiles?id=eq.{student_id}"
    profile_res = requests.get(profile_url, headers=headers, verify=False).json()

    if not profile_res:
        raise Exception("Alumno no encontrado")

    profile = profile_res[0]

    grades_url = f"{SUPABASE_URL}/rest/v1/grades?student_id=eq.{student_id}"
    grades_res = requests.get(grades_url, headers=headers, verify=False).json()
    promedio_anterior = round(sum(g['grade'] for g in grades_res) / max(len(grades_res), 1), 2) if grades_res else 7.0

    attendance_url = f"{SUPABASE_URL}/rest/v1/attendance?student_id=eq.{student_id}"
    attendance_res = requests.get(attendance_url, headers=headers, verify=False).json()
    asistencia = len([a for a in attendance_res if a['present']]) / max(len(attendance_res), 1)

    participation_url = f"{SUPABASE_URL}/rest/v1/participation?student_id=eq.{student_id}"
    participation_res = requests.get(participation_url, headers=headers, verify=False).json()
    participacion = round(sum(p['score'] for p in participation_res) / max(len(participation_res), 1), 2) if participation_res else 6

    # Obtener materias del alumno
    subjects_url = f"{SUPABASE_URL}/rest/v1/student_subjects?student_id=eq.{student_id}"
    subjects_res = requests.get(subjects_url, headers=headers, verify=False).json()

    subject_ids = [s['subject_id'] for s in subjects_res if subjects_res]
    subject_names = {}
    if subject_ids:
        subject_url = f"{SUPABASE_URL}/rest/v1/subjects?id=in.({','.join(map(str, subject_ids))})"
        subject_name_res = requests.get(subject_url, headers=headers, verify=False).json()
        subject_names = {s['id']: s['name'] for s in subject_name_res}

    return {
        "profile": profile,
        "promedio_anterior": promedio_anterior,
        "asistencia": round(asistencia, 2),
        "participacion": int(round(participacion, 0)),
        "materias": subject_names
    }


@app.get("/predict/student/{student_id}/detallado", response_model=PrediccionDetalladaResponse)
def predecir_rendimiento_detallado(student_id: str):
    datos = obtener_datos_alumno(student_id)

    input_data = np.array([[
        datos["asistencia"],
        datos["participacion"],
        datos["promedio_anterior"],
        4,  # horas_estudio simuladas
        8   # actividades_entregadas simuladas
    ]])

    input_scaled = scaler.transform(input_data)
    prediccion_global = model.predict(input_scaled)[0]

    # Predicciones por materia
    predicciones_por_materia = []
    for subject_id, subject_name in datos["materias"].items():
        # Simulamos una ligera variación por materia
        prediccion_materia = prediccion_global * (0.95 + np.random.uniform(-0.05, 0.05))
        alerta = prediccion_materia < 6.0
        predicciones_por_materia.append({
            "materia": subject_name,
            "promedio_predicho": round(prediccion_materia, 2),
            "alerta": alerta
        })

    # Alerta global
    alerta_riesgo = prediccion_global < 6.0

    # Generar recomendaciones personalizadas
    recomendaciones = []

    if datos["asistencia"] < 0.7:
        recomendaciones.append("Mejorar la asistencia es fundamental para mejorar el desempeño académico.")
    if datos["participacion"] < 5:
        recomendaciones.append("Es importante aumentar la participación en clase.")
    if datos["promedio_anterior"] < 6.0:
        recomendaciones.append("Se recomienda repasar los temas anteriores y buscar apoyo adicional.")

    if alerta_riesgo:
        recomendaciones.append("Se recomienda atención temprana por bajo rendimiento predictivo.")

    return {
        "student_id": student_id,
        "promedio_global_predicho": round(prediccion_global, 2),
        "alerta_riesgo": alerta_riesgo,
        "predicciones_por_materia": predicciones_por_materia,
        "recomendaciones": recomendaciones
    }