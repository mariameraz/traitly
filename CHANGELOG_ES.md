# Registro de Cambios

*Todos los cambios significativos de Traitly están documentados aquí.*

## [0.2.2] - 2026-10-06

Parche rápido

### Corregido
- Se incluyen los pesos de los modelos YOLO (`size_reference.pt`, `label.pt`) en el wheel y el sdist. Estaban excluidos en el `.gitignore`, por lo que las instalaciones desde PyPI no podían cargarlos y las mediciones volvían silenciosamente a píxeles.
    - Versiones afectadas: `0.2.0` y `0.2.1`. Recomendamos actualizar a `0.2.2`. 

Documentación: sin cambios respecto a [v0.2.1](https://traitly.readthedocs.io/en/v0.2.1/)

---

## v0.2.1 - 2026-10-06

### Correcciones
- `apply_color_correction` regresa error porque `PolynomialFeatures` no estaba siendo importado, y es necesario para la parte de la regresión polinomial.

Documentación: [https://traitly.readthedocs.io/en/v0.2.1/](https://traitly.readthedocs.io/en/v0.2.1/)

----

## v0.2.0 – 2026-10-05

### Correcciones

- En `setup_label` de `FruitInternalAnalyzer` y `FruitExternalAnalyzer`:
    - Si el código QR era detectado, la detección de la región de interés (ROI) de la etiqueta era saltada, y `label_roi = None`
    - Ahora, la detección del ROI de la etiqueta se ejecuta de forma independiente a la detección del código QR
- Se corrigió `outer_pericarp_mean_thickness`: el límite entre el pericarpio interno y externo se establecía previamente en el primer píxel con valor 255 sobre el rayo, el cual, dado que el pericarpio interno también es 255, siempre coincidía con el centroide, midiendo así el radio del fruto en lugar del pericarpio. Ahora el límite utiliza el último píxel con valor 255, marcando correctamente la transición de pericarpio interno a externo.
- Se corrigió una fuga de memoria durante el análisis por lote: las imágenes de entrada se almacenaban previamente en una caché LRU, manteniendo en RAM cada imagen procesada. Esto provocaba que el uso de memoria creciera sin control al analizar lotes grandes. Se eliminó la caché por completo, de modo que cada imagen ahora se carga, procesa y libera de forma independiente por cada proceso trabajador.
- Se corrigieron incompatibilidades de los módulos `mcc` y `wechat_qrcode` con OpenCV >= 5.0:
    - `cv2.mcc.CCheckerDetector.process()` movió el argumento de tipo de tabla de color (color chart) al nuevo método `setColorChartType()`; se añadió detección de versión para invocar la API correcta según la disponibilidad
    - `cv2.mcc.CCheckerDraw` fue eliminado; el método de dibujo se movió al propio `CCheckerDetector`; se añadió una alternativa acorde a esto
    - El constructor heredado de `cv2.wechat_qrcode_WeChatQRCode` ya no acepta rutas personalizadas de modelos Caffe; se añadió una alternativa al nuevo detector WeChat integrado y, en su defecto, a `cv2.QRCodeDetector`
- Se eliminó la binarización de Otsu previa a la alternativa clásica `cv2.QRCodeDetector`, la cual causaba que los códigos QR en las etiquetas no fueran detectados

### Cambios

- Se encapsularon atributos que solo son relevantes para procesos internos en `FruitExternalAnalyzer` y `FruitInternalAnalyzer` para mantener más limpia la interfaz del usuario.

#### *Cambios que rompen compatibilidad:*

- Se eliminó el soporte para Python 3.9; ahora se requiere Python 3.10 o superior.
- La selección de modo en la CLI cambió de banderas mutuamente excluyentes a subcomandos:
  - Antes: `traitly --fruit_internal -i PATH` / `traitly --fruit_external -i PATH`
  - Ahora: `traitly fruit_internal -i PATH` / `traitly fruit_external -i PATH`
- Se incluyó un método dedicado (`detect_color_checker()`) en `FruitInternalAnalyzer` y `FruitExternalAnalyzer` para detectar tarjetas de color en una imagen. Por lo tanto, `setup_measurements()` ya no acepta los argumentos `detect_color_checker` y `scale_factor`.

### Nuevo

- Se añadió el nuevo comando de CLI `traitly info` para imprimir metadatos del paquete, el sistema y las dependencias
- Se añadió el nuevo módulo `traitly.utils.metadata` con `get_package_versions()` para obtener las versiones instaladas de todas las dependencias del paquete y la versión de Python
- Se añadió el nuevo módulo `traitly.color_correction` con la clase `ColorCorrection` para corregir el color de imágenes o carpetas completas utilizando una tarjeta Macbeth Color Checker (24 parches)

Documentación: [https://traitly.readthedocs.io/en/v0.2.0/](https://traitly.readthedocs.io/en/v0.2.0/)

----

## v0.1.2 – 2026-05-18

### Correcciones
- Se corrigió la salida de `edit_mask` en la terminal (antes solo funcionaba en Jupyter) (reportado por @AlvaroGuerrero)
- Se añadió la dependencia `IPython` para poder abrir ventanas interactivas con `edit_mask` en CLI (reportado por @AlvaroGuerrero)
- Se corrigió un crash en `annotate_all_fruits` cuando los frutos no tienen lóculos detectados
- Se corrigió un crash en `detect_color_checker` cuando `cv2.mcc.CCheckerDetector` no está disponible
- Se corrigió un error de certificado SSL cuando easyocr intenta descargar modelos por primera vez
- Se corrigió versión hardcodeada en cli.py
- Se parchó `_load_img_cached`, el cual lanza `FileNotFoundError` en lugar de `None` en Windows (Ref. upstream error: [ultralytics#24405](https://github.com/ultralytics/ultralytics/issues/24405))
- Se corrigió el problema al renombrar imágenes con nombres duplicados cuando mas de una imagen tiene el mismo QR en el PDF cuando se utiliza `pdf_to_img`. 
  - Solo la primer imagen se renombraba con el QR.
  - Las imagenes ahora se renombran como `<texto_qr>.jpg`, `<texto_qr>_1.jpg`, `<texto_qr>_2.jpg`, etcétera.


### Nuevo
- Se mejoró la detección de QR con dos nuevas funciones:
  - Se añadió `cv2.wechat_qrcode_WeChatQRCode` como método principal para una detección mas robusta de códigos QR pequeños o inclinados
  - Se añadió `detectAndDecodeCurved` como alternativa cuando la función estandar `detectAndDecode` falla

Documentación: [https://traitly.readthedocs.io/en/v0.1.2/](https://traitly.readthedocs.io/en/v0.1.2/)

---

## v0.1.1 – 2026-05-04

### Correcciones
- Se renombró `fast_calibration` a `skip_yolo` en los archivos de ejemplo JSON para que coincida con los parámetros del código (reportado por @Hector-LM)
- Shiny App:
	- Se corrigió la ruta de imágenes de ejemplo en la documentación de la página principal
	- Se corrigió el reinicio de los pasos del pipeline en la barra lateral al regresar desde otra pestaña

### Cambios
- Se estandarizó el valor predeterminado de `min_fruit_area` a 1000 $px^2$ en todas las clases (reportado por @Hector-LM)
- El tiempo total de sesión ahora se muestra en segundos o minutos según la duración en los reportes de análisis por lote
- Se movió `convert_pdf` de `utils` a un nuevo módulo `pdf`:
  - Importación anterior: `from traitly.utils.convert_pdf import pdf_to_img`
  - Importación actual: `from traitly.pdf import pdf_to_img`
- Se renombró la dependencia opcional `traitly[all]` a `traitly[app]`
- Shiny App:
	- Se optimizaron las exportaciones de morfología y color eliminando escrituras temporales en disco
	- Se mejoró el uso de memoria en exportaciones por lote escribiendo archivos ZIP en disco en lugar de mantenerlos en RAM
	- Se adoptaron directorios temporales para manejar el procesamiento por lote y PDF con limpieza automática

### Nuevo
- Se agregó el parámetro `erosion_px` en `analyze_folder()` para las clases `FruitInternalAnalyzer` y `FruitExternalAnalyzer`

### Documentación
- Se fijaron las versiones de las dependencias

Documentación: [https://traitly.readthedocs.io/en/v0.1.1/](https://traitly.readthedocs.io/en/v0.1.1/)

---

## v0.1.0 – 2026-04-07
Lanzamiento inicial.

### Funcionalidades
- Análisis interno de frutos, lóculos y estampas con `FruitInternalAnalyzer`
- Análisis de morfología y color de frutos enteros con `FruitExternalAnalyzer`
- Procesamiento por lote con multiprocesamiento opcional (`analyze_folder`)
- Conversión de píxeles a centímetros usando referencias de tamaño
- Detección de códigos QR, etiquetas de texto y tarjeta de color
- Interfaz de línea de comandos (`traitly`)
- Aplicación web interactiva (`traitly-app`)

### Mediciones
- Rasgos morfológicos: área, perímetro, ejes, índices de forma, grosor del pericarpio, simetría
- Rasgos de color: RGB, HSV, Lab y Escala de grises por región de tejido

### Salidas
- Imágenes anotadas, resultados en CSV, reportes de sesión y errores, y archivos de parámetros

Documentación: [https://traitly.readthedocs.io/en/v0.1.0/](https://traitly.readthedocs.io/en/v0.1.0/)
