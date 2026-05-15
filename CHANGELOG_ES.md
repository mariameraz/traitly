# Registro de Cambios

*Todos los cambios notables de Traitly están documentados aquí:*

## v0.1.2 – En desarrollo

### Correcciones
- Corregir el mensaje de salida de `edit_mask` en la terminal (anteriormente solo funcionaba en Jupyter) (reportado por @AlvaroGuerrero)

## v0.1.1 – 2026-05-04

### Correcciones
- Renombrar `fast_calibration` a `skip_yolo` en los archivos de ejemplo JSON para que coincida con los parámetros del código (reportado por @Hector-LM)
- Shiny App:
	- Corregir ruta de imágenes de ejemplo rota en la documentación de la página principal
	- Corregir el reinicio de los pasos del pipeline en la barra lateral al regresar desde otra pestaña

### Cambios
- Estandarizar el valor predeterminado de `min_fruit_area` a 1000 $px^2$ en todas las clases (reportado por @Hector-LM)
- Mostrar el tiempo total de sesión en segundos o minutos según la duración en los reportes de análisis por lote
- Mover `convert_pdf` de `utils` a un módulo dedicado `pdf`:
  - Importación anterior: `from traitly.utils.convert_pdf import pdf_to_img`
  - Importación actual: `from traitly.pdf import pdf_to_img`
- Renombrar la dependencia opcional `traitly[all]` a `traitly[app]`
- Shiny App:
	- Optimizar las exportaciones de morfología y color eliminando escrituras temporales en el disco y usando procesamiento en la memoria
	- Mejorar el uso de memoria en exportaciones por lote escribiendo archivos ZIP en disco en lugar de mantenerlos en RAM
	- Usar directorios temporales administrados para el procesamiento por lote y PDF con limpieza adecuada de datos temporales

### Agregado
- Agregar el parámetro `erosion_px` en `analyze_folder()` para las clases `FruitInternalAnalyzer` y `FruitExternalAnalyzer`

### Documentación
- Fijar versiones de dependencias

----

## v0.1.0 — 2026-04-07

Lanzamiento inicial.

### Funcionalidades
- Análisis interno de fruto, lóculo y estampa con `FruitInternalAnalyzer`
- Análisis de morfología y color del fruto completo con `FruitExternalAnalyzer`
- Procesamiento por lote con multiprocesamiento opcional (`analyze_folder`)
- Conversión de píxeles a centímetros usando referencias de tamaño
- Detección de códigos QR, etiquetas de texto y verificadores de color
- Interfaz de línea de comandos (`traitly`)
- Aplicación web interactiva (`traitly-app`)

### Mediciones
- Rasgos morfológicos: área, perímetro, ejes, índices de forma, grosor del pericarpio, simetría
- Rasgos de color: RGB, HSV, Lab y Escala de grises por región de tejido

### Salidas
- Imágenes anotadas, resultados en CSV, reportes de sesión y errores, y archivos de parámetros
