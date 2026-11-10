"""Aplicación web local para el proyecto de análisis de sentimientos."""

from flask import Flask, render_template, request

from clasificador import clasificar_comentario, limpiar_texto


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
MAXIMO_CARACTERES = 5000


@app.route("/", methods=["GET", "POST"])
def inicio():
    comentario = ""
    resultado = None
    error = None
    estado = 200

    if request.method == "POST":
        comentario = request.form.get("comentario", "")
        try:
            limpiar_texto(comentario)
            if len(comentario) > MAXIMO_CARACTERES:
                raise ValueError("El comentario debe tener como máximo 5.000 caracteres.")
        except ValueError as detalle:
            error = str(detalle)
            estado = 400

        if error is None:
            try:
                resultado = clasificar_comentario(comentario)
            except (OSError, RuntimeError, ImportError, ValueError):
                app.logger.exception("No se pudo ejecutar el clasificador local.")
                error = "No pudimos analizar el comentario. Intenta nuevamente en unos momentos."
                estado = 503

    return render_template(
        "index.html", comentario=comentario, resultado=resultado,
        error=error, maximo_caracteres=MAXIMO_CARACTERES
    ), estado


@app.errorhandler(413)
def solicitud_demasiado_grande(error):
    return render_template(
        "index.html", comentario="", resultado=None,
        error="El texto enviado es demasiado grande. Usa hasta 5.000 caracteres.",
        maximo_caracteres=MAXIMO_CARACTERES
    ), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
