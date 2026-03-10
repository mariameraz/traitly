# Historial de Cambios ⊹ ࣪ ˖

Todos los cambios notables de Traitly están documentados aquí.

---

## v0.1.0 — Febrero 2026

Versión inicial.

### Añadido
- `FruitInternalAnalyzer` — análisis interno de fruta con segmentación de lóculos
- `FruitExternalAnalyzer` — análisis de fruta completa sin segmentación de lóculos
- Procesamiento por lotes con multiprocesamiento opcional (`analyze_folder`)
- Calibración píxel-a-métrica mediante marcadores de referencia
- Detección de códigos QR y etiquetas de texto
- Traits morfológicos: área, perímetro, ejes, índices de forma, pericarpio, simetría
- Traits de color: canales RGB, HSV, Lab por región de tejido
- Imágenes anotadas como salida
- Reporte de sesión y reporte de errores para procesamiento por lotes
- Dataclass `AnalysisParameters` para trazabilidad de parámetros
