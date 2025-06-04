from pydantic import BaseModel
from typing import Optional

class PrediccionRequest(BaseModel):
    asistencia: float
    participacion: int
    promedio_anterior: float
    horas_estudio: int
    actividades_entregadas: int

class PrediccionResponse(BaseModel):
    promedio_predicho: float