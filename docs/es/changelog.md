# Historial de Cambios ⊹ ࣪ ˖

Todos los cambios notables de Traitly están documentados aquí:

---

## v0.1.0 — Marzo 2026

Versión inicial.

**Incluye:**

- `FruitInternalAnalyzer`: análisis interno de frutos y lóculos
- `FruitExternalAnalyzer`: análisis de frutos enteros
- Análisis de estampas de frutos con `FruitInternalAnalyzer`
- Procesamiento por lotes con multiprocesamiento opcional (`analyze_folder`)
- Conversión pixel a centímetros mediante referencias de tamaño
- Detección de códigos QR y etiquetas de texto
- Detección de tarjetas de corrección de color
- Traits morfológicos: área, perímetro, ejes, índices de forma, grosor del pericarpio, simetría
- Traits de color: canales RGB, HSV, Lab, y Gray por región de tejido
- Imágenes anotadas como salida
- Resultados de los análisis como archivos CSV
- Reporte de sesión y reporte de errores para procesamiento por lotes
- Exporta parámetros de sesión como archivos TXT y JSON
- Módulo para la exploración de los resultados de color
- Extracción de imágenes desde PDF
