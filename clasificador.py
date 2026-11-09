"""Clasificación de comentarios con el modelo ONNX INT8 del proyecto.

Ejemplo desde Python:
    resultado = clasificar_comentario("Esta aplicación es muy útil")

Ejemplo desde la terminal:
    python clasificador.py "Esta aplicación es muy útil"
"""

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np


RUTA_MODELO = Path(os.environ.get("MODEL_DIR") or Path(__file__).resolve().parent / "modelo")
RUTA_ONNX = RUTA_MODELO / "modelo_int8.onnx"
MAXIMO_TOKENS = 128
ID_A_ETIQUETA = {0: "negative", 1: "neutral", 2: "positive"}
SENTIMIENTOS = {
    "negative": "negativo",
    "neutral": "neutral",
    "positive": "positivo",
}


def limpiar_texto(texto):
    """Normalizar espacios como en el notebook, conservando signos y emojis."""
    if not isinstance(texto, str):
        raise TypeError("El comentario debe ser un texto.")

    texto = re.sub(r"[\r\n\t]+", " ", texto)
    texto = re.sub(r" {2,}", " ", texto).strip()
    if not texto:
        raise ValueError("Escribe un comentario antes de analizar.")
    return texto


@lru_cache(maxsize=1)
def cargar_modelo():
    """Cargar el tokenizador y una sesión ONNX para reutilizarla."""
    if not RUTA_ONNX.is_file():
        raise FileNotFoundError(f"No se encontró el modelo ONNX en: {RUTA_ONNX}")

    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizador = AutoTokenizer.from_pretrained(
        str(RUTA_MODELO), local_files_only=True
    )
    opciones = ort.SessionOptions()
    opciones.intra_op_num_threads = int(os.environ.get("ONNX_INTRA_OP_THREADS", "2"))
    opciones.inter_op_num_threads = 1
    opciones.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sesion = ort.InferenceSession(
        str(RUTA_ONNX),
        sess_options=opciones,
        providers=["CPUExecutionProvider"],
    )
    return tokenizador, sesion


def clasificar_comentario(comentario):
    """Devolver sentimiento y confianza sin volver a entrenar la red."""
    texto = limpiar_texto(comentario)
    tokenizador, sesion = cargar_modelo()
    nombres_entrada = {entrada.name for entrada in sesion.get_inputs()}
    codificado = tokenizador(
        texto, truncation=True, max_length=MAXIMO_TOKENS, return_tensors="np"
    )
    entradas = {
        nombre: np.ascontiguousarray(codificado[nombre], dtype=np.int64)
        for nombre in nombres_entrada
        if nombre in codificado
    }
    logits = sesion.run(None, entradas)[0][0].astype(np.float64)
    probabilidades = np.exp(logits - np.max(logits))
    probabilidades /= probabilidades.sum()

    indice = int(np.argmax(probabilidades))
    etiqueta = ID_A_ETIQUETA[indice]
    return {
        "sentimiento": SENTIMIENTOS[etiqueta],
        "etiqueta_modelo": etiqueta,
        "confianza": float(probabilidades[indice]),
    }


if __name__ == "__main__":
    import argparse

    argumentos = argparse.ArgumentParser(description=__doc__)
    argumentos.add_argument("comentario", help="Comentario que se desea analizar.")
    opciones = argumentos.parse_args()
    try:
        resultado = clasificar_comentario(opciones.comentario)
    except (TypeError, ValueError, FileNotFoundError) as error:
        argumentos.error(str(error))
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
