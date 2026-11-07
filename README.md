# Análisis de sentimientos multilingüe

Aplicación Flask que clasifica comentarios como positivos, neutrales o negativos. Usa XLM-RoBERTa con la capa clasificadora entrenada en el proyecto académico y muestra las tres gráficas de evaluación guardadas.

## Preparar el servidor

Esta configuración está pensada para Linux con Python 3.14 y PyTorch para CPU. Desde la raíz de este repositorio, crea el entorno e instala las dependencias.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
python -m pip install -r requirements.txt
```

Transfiere `model.safetensors` desde la carpeta local `modelo` a la carpeta `modelo` del servidor. El archivo pesa 1,11 GB y está excluido de Git. En esa misma carpeta deben quedar `config.json`, `tokenizer.json` y `tokenizer_config.json`, incluidos en el repositorio. Si el proveedor usa un volumen aparte, configura `MODEL_DIR` con la ruta absoluta de la carpeta que contiene los cuatro archivos.

El campo `modelo_sha256` de `resultados_web/estadisticas.json` contiene la huella de los pesos con los que se generaron las gráficas. Comprueba el archivo transferido con `sha256sum modelo/model.safetensors` y compara los valores.

## Iniciar la aplicación

```bash
gunicorn --config gunicorn.conf.py app:app
```

La configuración escucha en el puerto `PORT` que entregue el proveedor, o en `8000` si no está definido. Usa un proceso para cargar una sola copia del modelo. El proveedor o su proxy se encarga de la URL HTTPS.

Abre la URL del servicio, envía un comentario y comprueba que aparezcan el sentimiento y la confianza. Desplázate hasta «Resultados del proyecto» para revisar las tres gráficas. La primera predicción carga los pesos en memoria. Mide la memoria y los tiempos de respuesta en el servidor elegido.

En Windows puedes probar la copia local con `python app.py` desde esta carpeta y abrir <http://127.0.0.1:5000>. Si ya tienes las dependencias instaladas en Python 3.14, puedes usar ese mismo entorno.

## Archivos de ejecución

La aplicación usa el código Python, las plantillas de `templates`, los recursos de `static`, el resumen de `resultados_web` y el modelo con su tokenizador. Los datos de entrenamiento y los experimentos se mantienen en el proyecto académico.
