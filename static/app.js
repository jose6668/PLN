const formulario = document.querySelector('#formulario-analisis');
const comentario = document.querySelector('#comentario');
const boton = document.querySelector('#boton-analizar');
const textoBoton = document.querySelector('#texto-boton');
const estado = document.querySelector('#estado-carga');

comentario.addEventListener('input', () => {
  document.querySelector('#contador').textContent = `${comentario.value.length} / 5.000`;
});

formulario.addEventListener('submit', () => {
  boton.disabled = true;
  textoBoton.textContent = 'Analizando…';
  estado.textContent = 'Procesando el comentario. La primera consulta puede tardar mientras se carga el modelo.';
  formulario.setAttribute('aria-busy', 'true');
});

window.addEventListener('pageshow', () => {
  boton.disabled = false;
  textoBoton.textContent = 'Analizar comentario';
  estado.textContent = '';
  formulario.removeAttribute('aria-busy');
});
