# INVESTIGACIÓN EXHAUSTIVA — Mejora de la detección de texto en manga

**Fecha**: 2026-08-06 · **Estado**: investigación (sin código aplicado aún)
**Contexto**: los 3 motores (EasyOCR GPU + RapidOCR CPU + Unlimited-OCR VLM 4-bit) ya están fusionados en `OCRManager` con trigger selectivo v4.2. Este documento ordena TODAS las técnicas de mejora de detección de manga investigadas, de la **más eficiente/mejor** a la **menos eficiente/peor** (ratio calidad ganada vs costo computacional y de integración).

---

## 0. Resumen ejecutivo (TL;DR)

| Posición | Técnica | Ganancia estimada | Costo | Ratio E/C |
|:-:|:--|:--|:--|:--|
| 🥇 1 | **Detector YOLO fine-tuned para burbujas de manga** (nano/small) | Recupera globos que ningún OCR detecta como "texto" | Bajo (CPU 200-400ms, 3M params) | **Excelente** |
| 🥈 2 | **Real-ESRGAN anime upscale (6B) pre-OCR** | +texto en scans de baja resolución/JPEG | Medio (CPU ~2-5s/pág) | **Muy alto** |
| 🥉 3 | **Comic Text Detector (CTD/dmMaze) como tier de detección** | Detección específica de líneas de texto manga (ya se probó en el proyecto) | Medio (ONNX CPU) | **Muy alto** |
| 4 | **Clasificador de dirección de texto (horizontal/vertical) + rotación** | Recupera diálogo vertical japonés/CJK que EasyOCR pierde | Bajo (ya existe `_classify_rotate_crop` 0°/180°) | Alto |
| 5 | **Detección de paneles (morfología/proyecciones) → orden de lectura + región** | Mejor agrupación de bloques y orden; base para re-OCR dirigido | Muy bajo (OpenCV, ~20ms) | Alto |
| 6 | **Descreening (eliminar medios tonos)** | Crítico SOLO en scans de revista impresa | Bajo (OpenCV) | Alto (condicional) |
| 7 | **PaddleOCR-VL-For-Manga / manga-ocr (kha-white) como 4º motor** | CER japonés muy superior | Alto (nuevo venv/GPU) | Medio |
| 8 | **Manga Whisperer / Magi (diarización completa)** | Panel+texto+personaje+orden en UNA red | Muy alto (DETR ~1.5M params+dataset) | Medio |
| 9 | **SAM/U-Net para segmentación de burbujas** | Contornos perfectos de globo | Alto (VRAM) | Bajo-Medio |
| 10 | **SWT / MSER (heurísticas clásicas)** | Casi nula en manga (trama confunde) | Bajo | **Malo — evitar** |
| 11 | **Binarización dura pre-OCR** | Contraproducente con deep learning | Bajo | **Negativo — evitar** |

---

## 1. 🥇 Detector YOLO fine-tuned para burbujas de manga (MEJOR ratio)

**Qué es**: Modelos YOLOv8n-seg/YOLOv11n-seg (~3M params) fine-tuned sobre datasets de manga/webtoon/comic para detectar *speech bubbles*, cartelas y cajas de texto como OBJETOS (no como texto).

**Por qué funciona**: Los OCR (EasyOCR/RapidOCR) solo ven "texto" si detectan glifos. Pero un globo de diálogo puede estar vacío de caracteres detectables (texto artístico, bajo contraste, o el detector falla). Un detector de burbujas localiza la REGIÓN blanca elíptica — independiente del texto — y luego se re-OCRea el crop con upscale 3-4x (exactamente la Ruta C ya existente, pero con regiones MUCHO mejores).

**Métricas reportadas** (community bubble detectors):
- YOLOv8 bubble detector: precisión 95.9%, recall 95.1%, mAP@50 98.4%, mAP@50-95 85.5%
- YOLOv11n-seg (MangaLens): box precision 97.55%, mask precision 97.66%, mAP@50 99.13%, mAP@50-95 94.69%
- **Velocidad**: 8-25ms en GPU (T4/V100); 200-400ms en CPU — compatible con GTX 1050 Ti y hasta CPU pura

**Integración en el proyecto**:
1. `pip install ultralytics` (solo en server, no en el .exe — se carga en runtime como EasyOCR)
2. Nuevo tier "bubble detector" en `_detect_bubble_regions_in_panel()`: reemplaza/aumenta el OpenCV blob actual con el modelo YOLO
3. Las regiones detectadas alimentan la **Ruta C existente** (`_recover_regions_with_easyocr` con upscale 3.5 + cls de rotación + degradación CPU)
4. Se ejecuta SIEMPRE en CPU (YOLO nano) o GPU cuando no infiere U-OCR

**Costo**: ~200-400ms/pág CPU. Modelos públicos: `ogkalu/yolo-manga-bubble-detector`, `huyvux3005`, Roboflow Universe.

**Ganancia esperada**: recupera los globos perdidos (el problema central de las páginas artísticas 3/11/12) SIN la inferencia VLM de 2-8 min/pág. Directamente ataca el 12.2% de bloques no traducidos por fragmentos no detectados.

**Fase sugerida**: Fase 6 — el mayor impacto por hora de trabajo.

---

## 2. 🥈 Real-ESRGAN anime upscale pre-OCR

**Qué es**: Red de super-resolución entrenada para contenido anime/ilustrado (`RealESRGAN_x4plus_anime_6B`). Escala 4x restaurando bordes de glifos.

**Por qué funciona**: Los scans viejos/JPEG pierden trazos finos (crítico en kanji/furigana). El OCR moderno espera definición clara; el upscale reconstruye la tipografía sin desenfocar.

**Métricas/evidencia**: Es el factor de mayor mejora cuantitativa documentado en flujos manga-ocr. Alternativa más ligera: `cv2.resize` con `INTER_CUBIC` (ya se usa en la Ruta C con 3.5x) — el salto es usar el modelo IA.

**Integración**: Aplicar SOLO en crops de globos (no página completa — 2-5s/pág en CPU con modelos ONNX) o en páginas donde el trigger detecta baja calidad. El modelo `RealESRGAN_x4plus_anime_6B` tiene versión ONNX/ONNXRuntime que corre en CPU.

**Costo**: ~2-5s por crop grande en CPU; menos si se limita a los globos de la Ruta C.

**Ganancia**: sube el CER en scans de baja calidad; combinado con el cls de rotación (ya implementado) es la pareja ideal pre-re-OCR.

**Fase sugerida**: Fase 6 (junto con YOLO) o Fase 7 standalone.

---

## 3. 🥉 Comic Text Detector (CTD / dmMaze) como detector dedicado

**Qué es**: Detector entrenado con ~13K imágenes de manga (1/3 Manga109, 1/3 DCM, 1/3 sintético con weak supervision U-Net/DBNet). Arquitectura YOLOv5 + cabezas U-Net/DBNet para segmentar LÍNEAS de texto dentro de bloques. Es el detector estándar de manga-image-translator y BallonsTranslator.

**Historial del proyecto**: ⚠️ **Ya se probó y se ELIMINÓ** (sesión 2026-07-25): `ctd_lib/` + `ocr_ctd_fallback.py` + `comictextdetector.pt` (77MB) + deps (pyclipper, shapely, einops) ≈84MB, 2690 líneas. Se eliminó por fragilidad de dependencias y porque EasyOCR+RapidOCR cubrían la mayoría.

**Por qué volvería a evaluarse**: El CTD está específicamente entrenado para texto manga vertical/horizontal/SFX, mientras que EasyOCR es genérico. Si el YOLO bubble detector (posición 1) se adopta, el CTD podría complementarlo como detector de LÍNEAS dentro del globo (no solo burbuja). También existen pesos ONNX (`mayocream/comic-text-detector`) que evitan la fragilidad PyTorch original.

**Costo**: ONNX CPU ~1-2s/pág. ~84MB de modelos.

**Ganancia**: detección de líneas de texto densas y SFX estilizados que EasyOCR/RapidOCR pierden.

**Fase sugerida**: Fase 7-8, como tier opcional detrás del trigger v4.2 (solo páginas difíciles), NO en el camino caliente.

---

## 4. Clasificador de dirección (horizontal/vertical) + rotación

**Qué es**: Extender el `_classify_rotate_crop` actual (0°/180° ya implementado en Fase 3 pt.3) con clasificación **90°** y detección de texto **vertical** (japonés: tategaki). RapidOCR tiene `cls_image_shape` que podría extenderse; o usar el ángulo de los blobs de EasyOCR (`rotation_info=['0','90','180','270']` ya soportado por la librería pero no activado).

**Por qué funciona**: El manga japonés se lee verticalmente. EasyOCR con `rotation_info` puede rotar el crop automáticamente. El proyecto ya tiene la infraestructura (cls + des-rotación de coords + upscale).

**Integración**: Activar `rotation_info` en `_get_ocr_reader()` (parámetro de EasyOCR ya soportado) — un cambio de 1 línea. O extender `_classify_rotate_crop` para clasificar 4 ángulos.

**Costo**: ~0ms extra (EasyOCR lo hace inline con el batch de rotaciones) o ~0.1-0.3s/globo con el cls.

**Ganancia**: Recupera el diálogo vertical — probablemente una parte del 17.5% UNTRANSLATED actual.

**Fase sugerida**: Fase 7 — cambio mínimo, impacto directo en el capítulo actual.

---

## 5. Detección de paneles (layout) → orden de lectura + re-OCR dirigido

**Qué es**: Segmentar la página en paneles (morfología: dilatar líneas de gutters; o proyecciones; o `MangaPanelSegmentation` DL). Sirve para: (1) orden de lectura RTL correcto, (2) saber qué región pertenece a qué globo, (3) descartar márgenes/headers con precisión de región (no solo de bloque).

**Integración**: Función nueva `_detect_panels(img)` en ocr_utils (OpenCV: umbral → invertir → dilatar mucho → componentes conexos → filtrar por área). ~20-50ms. Usar los rects de panel para: alimentar `_detect_bubble_regions_in_panel` (mejor contexto), y para el re-OCR dirigido de la Ruta C.

**Costo**: Muy bajo (solo OpenCV).

**Ganancia**: Moderada pero barata; mejora coherencia de fusión y elimina falsos bloques de gutter.

**Fase sugerida**: Fase 7 (con el clasificador de dirección).

---

## 6. Descreening (eliminar medios tonos)

**Qué es**: Eliminar el patrón de puntos de semitono (moiré) de scans de revistas impresas físicas. Técnica clásica: filtro de mediana grande + resta, o `cv2.morphologyEx` con kernel ~7-9px.

**Por qué funciona**: La trama de puntos confunde a la binarización y a los detectores (falsos trazos). **Solo necesario si el PDF viene de revista escaneada** — si es digital (como el capítulo actual de "Cómo criar villanos"), NO aporta.

**Integración**: En `_pre_filter_image` (ya existe el pre-filtro morfológico), añadir paso condicional detectando periodicidad de trama.

**Costo**: ~50-100ms/pág OpenCV.

**Ganancia**: Condicional (solo scans físicos). Alto si aplica, cero si no.

**Fase sugerida**: Fase 8 — con flag `--descreen`.

---

## 7. PaddleOCR-VL-For-Manga / manga-ocr (kha-white) como 4º motor

**Qué es**: 
- **manga-ocr** (kha-white): Transformer Vision-Encoder-Decoder entrenado en Manga109. OCR end-to-end multilínea para japonés. `pip install manga-ocr`. CER muy inferior a Tesseract/Cloud Vision en manga. ~400MB en GPU, ONNX para CPU.
- **PaddleOCR-VL-For-Manga**: fine-tune de PaddleOCR-VL (~0.9B params) sobre Manga109-s + sintético; 70-90% exact match en validación, texto vertical nativo.

**Por qué**: Son los MEJORES en CER para japonés manga — pero el proyecto traduce español→inglés (manga español/scanlation), donde su ventaja sobre EasyOCR es menor. Útil solo si el usuario lee manga japonés/raw.

**Integración**: Nuevo venv (conflicto CUDA con EasyOCR/CT2 como el daemon U-OCR) o proceso daemon separado. Costo alto.

**Ganancia**: Solo para japonés raw. Para el capítulo actual (español), marginal.

**Fase sugerida**: Fase 9 — solo si se amplía a manga japonés.

---

## 8. Manga Whisperer / Magi (diarización completa de manga)

**Qué es**: Sistema CVPR 2024 (Sachdeva & Zisserman, Oxford VGG). Un modelo DETR-like (backbone ResNet-50 + encoder-decoder con tokens `[OBJ]`, `[C2C]`, `[T2C]`) que detecta **paneles + texto + personajes** y asocia diálogo a personaje, con orden de lectura por grafos. Versiones Magiv2 (nombres de personajes, ACCV 2024) y Magiv3 (2025).

**Disponibilidad**: código y pesos públicos (`ragavsachdeva/magi`, Hugging Face). Datasets: PopManga (55K anotaciones, 80+ mangas); Manga109 integrado.

**Por qué importa**: Es el estado del arte absoluto en comprensión de manga. La detección de texto sale "gratis" con el layout y el orden.

**Costo**: Muy alto — modelo grande (GPU ~4-8GB VRAM, la GTX 1050 Ti de 4GB queda justa), entrenar/fine-tune requiere dataset; inferencia por página más lenta que YOLO.

**Ganancia**: Cobertura total de panel+texto+orden, pero el ROI por hora de trabajo es bajo frente a YOLO bubble (posición 1) que resuelve el 80% del problema.

**Fase sugerida**: Investigación futura (Q4 2026+), no inmediata.

---

## 9. SAM / U-Net para segmentación de burbujas

**Qué es**: Segment Anything Model (SAM) o U-Net afinado para segmentar siluetas de burbuja a nivel de píxel.

**Por qué**: Contorno perfecto del globo (colas curvas), útil para inpainting preciso y para saber exactamente qué recortar.

**Costo**: SAM pesado (VRAM alta); variantes mobile/quantizadas corren en CPU pero lentas. Post-procesamiento adicional.

**Ganancia**: El inpainting por glifos actual ya preserva el arte; la segmentación de burbuja añade precisión marginal sobre el detector YOLO-seg (que YA produce máscaras de burbuja — posición 1 cubre esto sin SAM).

**Fase sugerida**: No recomendado — YOLOv11n-seg ya da máscaras.

---

## 10. ❌ SWT / MSER (heurísticas clásicas) — EVITAR

**Qué es**: Stroke Width Transform (Epshtein 2010) y MSER detectan texto por anchura de trazo uniforme y regiones estables.

**Por qué falla en manga**: La trama/retícula y las líneas de acción del dibujo comparten grosor de trazo con las letras → falsos positivos masivos. Investigación específica (Piriyothinkul et al. 2019 "Detecting Text in Manga Using SWT") confirma que necesita parches (SVM, filtros de burbuja) y aun así queda muy por detrás del deep learning. Velocidad aparente, pero el post-procesado heurístico la anula.

**Veredicto**: No usar. Cualquier tiempo invertido aquí rinde menos que YOLO nano.

---

## 11. ❌ Binarización dura pre-OCR — CONTRAINDICADA

**Qué es**: Umbralizado (Sauvola/Niblack/OTSU) a blanco y negro puro antes del OCR.

**Por qué es negativa**: Los modelos modernos (EasyOCR, RapidOCR, VLM) se entrenaron con niveles de gris y anti-aliasing. La binarización agresiva corta trazos finos, produce bordes dentados y destruye el texto estilizado. La regla es: **solo útil para Tesseract clásico; dañina para deep learning** (confirmado en flujos manga-ocr).

**Veredicto**: Mantener la imagen en gris restaurrada; nunca binarizar para los 3 motores.

---

## Anexo A — Datasets públicos para entrenar/evaluar

| Dataset | Tamaño | Contenido | Uso |
|:--|:--|:--|:--|
| **Manga109** | ~10.6-11K páginas, 109 volúmenes | Paneles, texto, personajes, rostros, onomatopeyas | Referencia académica #1 |
| **DCM772** (Digital Comic Museum) | 772 páginas | Cómic occidental Golden Age, paneles/personajes/texto | Contraste estilos |
| **comic-text-detector data** | ~13K imágenes (1/3 Manga109 + 1/3 DCM + 1/3 sintético) | Cajas de texto, líneas, máscaras binarias | Entrenar detectores de texto manga |
| **PopManga** | ~2K páginas + 55K anotaciones (80+ mangas) | Diálogo/personaje/panel | Benchmark Magi |
| **eBDtheque** | 100 páginas multiestilo | Paneles, texto, personajes | Evaluación genérica |
| **Mangadex-1.5M** (via Magi tools) | ~1.5M páginas | Preentrenamiento (no redistribuido) | Fine-tuning a escala |

**Métricas de evaluación**: mAP@50 / mAP@50-95 (COCO, detección de cajas) y F1 (segmentación binaria). El proyecto debería añadir un benchmark local: páginas anotadas a mano del capítulo actual + script de CER.

---

## Anexo B — Herramientas open source de referencia (qué usan los mejores)

| Herramienta | Detector | OCR | Inpainting | Notas |
|:--|:--|:--|:--|:--|
| **manga-image-translator** (MIT) | CTD (default) | modelos propios 32/48px_ctc | AOT-GAN, LaMa manga | El estándar de la comunidad |
| **BallonsTranslator-Pro** | ysgyolo (YOLOv11 x), RF-DETR Seg, PP-OCRv5/DocLayout, magi_det, CTD | PaddleOCR, manga-ocr | LaMa manga | El detector ysgyolo supera a CTD en muchos casos (2025) |
| **Koharu** (Rust) | CTD/ysgyolo | manga-ocr ONNX | — | Ligero |
| **ImageTrans** | CTD | manga-ocr | — | Soporte "strip furigana" |

**Insight clave**: La comunidad converge en **detector dedicado de burbujas/líneas (CTD o ysgyolo) + OCR especializado (manga-ocr) + inpainting LaMa**. El proyecto ya tiene el OCR fusionado (superior en español) y el inpainting por glifos; le falta el **detector de burbujas moderno** (posición 1) — exactamente la brecha.

---

## Anexo C — Roadmap recomendado (fases)

| Fase | Contenido | Esfuerzo | Impacto |
|:--|:--|:--|:--|
| **F6** | YOLOv11n-seg bubble detector → regiones de la Ruta C (reemplaza/aumenta blobs OpenCV) | 1-2 días | **Alto** (recupera globos perdidos, sin VLM) |
| **F7** | `rotation_info` de EasyOCR (0/90/180/270) + detección de paneles OpenCV | 0.5-1 día | Medio-Alto (texto vertical + orden) |
| **F8** | Real-ESRGAN anime ONNX en crops de la Ruta C + descreening condicional | 1 día | Medio (scans malos) |
| **F9** | CTD ONNX como tier opcional en páginas trigger (sin volver al bundle pesado) | 1-2 días | Medio (SFX/denso) |
| **F10** | Benchmark local con anotaciones del capítulo (mAP/F1/CER) | 1 día | Infraestructura (mide todo lo anterior) |
| — | Magi / manga-ocr / PaddleOCR-VL | Semanas | Solo si se amplía a japonés raw |

---

## Conclusión

El orden de inversión óptimo es: **1) YOLO bubble detector → 2) Real-ESRGAN en crops → 3) CTD ONNX como complemento**, reutilizando toda la infraestructura ya construida (Ruta C, cls de rotación, trigger v4.2, degradación CPU, fusión multi-motor). Las técnicas clásicas (SWT/MSER/binarización) están **descartadas por evidencia** para manga moderno con deep learning. El detector de burbujas es la brecha central: el proyecto sabe leer el texto que encuentra, pero encuentra poco texto artístico — YOLO cierra exactamente esa brecha en CPU/GPU ligera.
