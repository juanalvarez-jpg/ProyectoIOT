from fastapi.middleware.cors import CORSMiddleware
import pickle
from pydantic import BaseModel
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse


class SolicitudClasificacion(BaseModel):
    modelo: str
    valores: list[float]


class RespuestaClasificacion(BaseModel):
    modelo: str
    clase_predicha: int | str


servicio = FastAPI(title="API Clasificador de Cosechas")

servicio.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# Carga de clasificadores entrenados
DIRECTORIO_BASE = Path(__file__).parent

ARCHIVOS_CLASIFICADORES = {
    "svm": "SVMCosecha.pkl",
    "random_forest": "RFCosecha.pkl",
}

NUM_CARACTERISTICAS_ESPERADAS = 7


def cargar_clasificador(nombre_archivo: str):
    ruta = DIRECTORIO_BASE / nombre_archivo
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo del clasificador: {ruta}")
    with open(ruta, "rb") as archivo:
        return pickle.load(archivo)


clasificadores = {
    nombre: cargar_clasificador(archivo)
    for nombre, archivo in ARCHIVOS_CLASIFICADORES.items()
}


@servicio.get("/")
async def index():
    return FileResponse(DIRECTORIO_BASE / "index.html")


@servicio.post("/predict", response_model=RespuestaClasificacion)
async def predict(solicitud: SolicitudClasificacion):

    if solicitud.modelo not in clasificadores:
        opciones_validas = list(clasificadores.keys())
        raise HTTPException(
            status_code=400,
            detail=f"El modelo '{solicitud.modelo}' no existe. Modelos disponibles: {opciones_validas}",
        )

    if len(solicitud.valores) != NUM_CARACTERISTICAS_ESPERADAS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Se esperaban {NUM_CARACTERISTICAS_ESPERADAS} características, "
                f"pero se recibieron {len(solicitud.valores)}"
            ),
        )

    clf_elegido = clasificadores[solicitud.modelo]
    muestra = [solicitud.valores]

    etiqueta = clf_elegido.predict(muestra)[0]

    return RespuestaClasificacion(
        modelo=solicitud.modelo,
        clase_predicha=etiqueta,
    )
