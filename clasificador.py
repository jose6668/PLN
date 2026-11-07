"""Clasificación con el modelo seleccionado en la sección 12 del notebook.

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


RUTA_MODELO = Path(os.environ.get("MODEL_DIR") or Path(__file__).resolve().parent / "modelo")
MAXIMO_TOKENS = 128
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
    """Cargar los archivos locales una vez y reutilizarlos en cada consulta."""
    if not (RUTA_MODELO / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"No se encontró model.safetensors en: {RUTA_MODELO}"
        )

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    tokenizador = AutoTokenizer.from_pretrained(
        str(RUTA_MODELO), local_files_only=True
    )
    red = AutoModelForSequenceClassification.from_pretrained(
        str(RUTA_MODELO), local_files_only=True
    )
    # La aplicación usa CPU; la GPU se utilizó para el entrenamiento.
    return pipeline(
        "sentiment-analysis", model=red, tokenizer=tokenizador, device=-1
    )


def clasificar_comentario(comentario):
    """Devolver sentimiento y confianza, sin volver a entrenar la red.

    La confianza está entre 0 y 1 y no equivale a la exactitud del modelo.
    Los comentarios largos se truncan a 128 tokens, como en la evaluación.
    """
    texto = limpiar_texto(comentario)
    modelo = cargar_modelo()
    prediccion = modelo(
        texto, truncation=True, max_length=MAXIMO_TOKENS
    )[0]
    etiqueta = prediccion["label"].lower()
    if etiqueta not in SENTIMIENTOS:
        raise ValueError(f"El modelo devolvió una etiqueta inesperada: {etiqueta}")

    return {
        "sentimiento": SENTIMIENTOS[etiqueta],
        "etiqueta_modelo": etiqueta,
        "confianza": float(prediccion["score"]),
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
