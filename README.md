# Análisis de sentimientos multilingüe

Aplicación Flask que clasifica comentarios como positivos, neutrales o negativos. Usa la versión ONNX INT8 del modelo XLM-RoBERTa entrenado en el proyecto académico y muestra las tres gráficas guardadas. El servidor solo realiza inferencia, no vuelve a entrenar el modelo.

## Preparar el servidor

Esta configuración está pensada para Linux con Python 3.14 y ONNX Runtime para CPU. Desde la raíz de este repositorio, crea el entorno e instala las dependencias.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Transfiere `modelo/modelo_int8.onnx` desde tu copia local al mismo directorio del servidor. El archivo pesa aproximadamente 266 MiB y está excluido de Git. En esa carpeta también deben estar `config.json`, `tokenizer.json` y `tokenizer_config.json`, que sí están incluidos en el repositorio. Si el proveedor usa un volumen aparte, configura `MODEL_DIR` con la ruta absoluta de la carpeta que contiene estos archivos.

Comprueba que la transferencia esté completa con `sha256sum modelo/modelo_int8.onnx` y compara el resultado con la huella SHA-256 indicada abajo.

## Iniciar la aplicación

```bash
gunicorn --config gunicorn.conf.py app:app
```

La configuración escucha en el puerto `PORT` que entregue el proveedor, o en `8000` si no está definido. Usa un proceso y limita ONNX Runtime a dos hilos por defecto para compartir la CPU disponible. El proveedor o su proxy se encarga de la URL HTTPS.

Abre la URL del servicio, envía un comentario y comprueba que aparezcan el sentimiento y la confianza. Desplázate hasta «Resultados del proyecto» para revisar las tres gráficas. La primera predicción carga los pesos en memoria. Mide la memoria y los tiempos de respuesta en el servidor elegido.

En Windows puedes probar la copia local con `python app.py` desde esta carpeta y abrir <http://127.0.0.1:5000>. Si ya tienes las dependencias instaladas en Python 3.14, puedes usar ese mismo entorno.

## Archivos de ejecución

La aplicación usa el código Python, las plantillas de `templates`, los recursos de `static`, el resumen de `resultados_web`, el modelo ONNX INT8 y el tokenizador. No necesita los datos de entrenamiento ni el modelo `model.safetensors` original en el servidor.

Modelo desplegable SHA-256: `165ff9ce7937c15e73abb33b21db59ebbee924bd6fa29c1459bc68e61ecafa62`.
