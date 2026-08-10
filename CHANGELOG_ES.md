# Registro de Cambios

*Todos los cambios significativos de Traitly están documentados aquí.*

## v.0.1.3 – En desarrollo

### Correcciones

- En `setup_label`:
    - Si el código QR era detectado, la dtection de la región de interés (ROI) de la etiqueta era saltada, y `label_roi = None`
    - Ahora, la detecctión del ROI de la etiqueta y la detección del código QR son dos pasos independientes

### Cambios

- Se encapsularon atributos que solo son relevantes para procesos internos en `FruitExternalAnalyzer` y `FruitInternalAnalyzer` para mantener más limpia la interfáz del usuario.
- Se movió `detect_color_checker` de `FruitInternalAnalyzer` al nuevo módulo `color_correction`
- `setup_measurements` ya no acepta los argumentos `detect_color_checker` and `scale_factor`. Usar el método `detect_color_checker()` en su lugar.


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
