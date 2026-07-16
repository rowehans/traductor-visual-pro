> 📖 **Antes de tocar `app.js` o `server.py`, lee [`AGENTS.md`](./AGENTS.md) / [`CLAUDE.md`](./CLAUDE.md).**

# Traductor visual de PDF e imagenes

App local con backend Python para cargar paginas de PDF o imagenes, detectar texto con OCR, traducirlo y reemplazarlo visualmente dentro de burbujas o zonas marcadas.

## Iniciar la app

Haz doble clic en `start-app.bat` o ejecuta:

```powershell
powershell -ExecutionPolicy Bypass -File .\start-app.ps1
```

La app abre en:

http://127.0.0.1:5174/

## Funciones

- Carga PDF, PNG, JPG y WebP.
- Detecta texto automaticamente en toda la pagina con OCR.
- Detecta texto solo dentro de una burbuja seleccionada.
- Traduce con backend Python usando `deep-translator`.
- Soporta espanol, ingles, portugues, frances, aleman, italiano, japones, coreano y chino.
- Permite corregir manualmente el texto traducido.
- Permite mover, redimensionar y estilizar burbujas.
- Exporta la pagina editada como PNG o PDF.

## Notas

- El OCR sigue usando Tesseract.js desde el navegador, por lo que requiere internet la primera vez que carga sus archivos.
- La traduccion usa servicios online desde Python; si el servicio limita o falla, la app cae al traductor web anterior y luego al diccionario basico local.
- Python y las dependencias quedaron instalados en `env/`.

## Edicion profesional de burbujas

- Al exportar, la app borra el texto original rellenando solo el interior de la zona marcada con un color claro muestreado desde la propia burbuja. Esto ayuda a conservar el contorno y reducir dano al dibujo.
- El texto traducido se recompone automaticamente: ajusta tamano de fuente, interlineado, espaciado entre letras y saltos de linea para caber en la burbuja.
- Para mejores resultados, marca la burbuja dejando un pequeno margen interior y evitando cubrir el borde negro del dibujo.

## Borrado avanzado con OpenCV

- La exportacion ahora intenta usar OpenCV.js con `cv.inpaint` para reconstruir el interior de las zonas marcadas antes de escribir la traduccion.
- Si OpenCV.js no carga o falla, la app usa automaticamente el borrado inteligente anterior por color muestreado.
- Este modo mejora fondos con textura ligera, tramas simples y burbujas no totalmente blancas. Para fondos muy complejos, un modelo IA tipo LaMa seguiria dando mejores resultados.

## PDF completo

- La app ahora agrega el boton `Descargar PDF completo` desde JavaScript.
- En PDFs grandes, ese boton recorre todas las paginas, renderiza cada una, aplica las ediciones guardadas por pagina y genera un unico PDF final.
- `Descargar pagina PDF` sigue exportando solo la pagina visible.

## Traduccion automatica total

- La app agrega el boton `Traducir todo automatico`.
- Tambien agrega la casilla `Traducir automaticamente al cargar`.
- El modo automatico recorre todas las paginas, detecta texto con OCR aunque no este en burbujas, traduce cada bloque y crea cajas editables respetando posicion, tamano aproximado y saltos.
- La tipografia exacta no siempre puede recuperarse desde imagenes escaneadas, pero la app intenta conservar el aspecto usando tamano y ubicacion inferidos desde el OCR.
