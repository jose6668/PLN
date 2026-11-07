"""Configuración del servidor Linux para una instancia de inferencia."""

import os


bind = f"0.0.0.0:{int(os.environ.get('PORT', '8000'))}"
workers = 1
timeout = 120
accesslog = "-"
errorlog = "-"
