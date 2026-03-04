# Inicio Rápido

*Un análisis completo corriendo en minutos.*

**Requisitos:** Traitly instalado ([Guía de instalación](../installation.md)) y una imagen escaneada de un corte transversal de fruta.

---

## Análisis interno

Usa `FruitInternalAnalyzer` cuando tus imágenes contengan lóculos visibles (estructura interna).

```python
from traitly.fruit_phenotyping import FruitInternalAnalyzer

analyzer = FruitInternalAnalyzer("mi_imagen.jpg")
analyzer.load_image()
analyzer.setup_measurements()
analyzer.generate_fruit_mask()
analyzer.detect_fruits()

# Traits morfológicos → retorna un DataFrame
df = analyzer.analyze_morphology()

# Traits de color (opcional)
df_color = analyzer.analyze_color()
```

---

## Análisis externo

Usa `FruitExternalAnalyzer` para análisis de fruta completa sin segmentación de lóculos.

```python
from traitly.fruit_phenotyping import FruitExternalAnalyzer

analyzer = FruitExternalAnalyzer("mi_imagen.jpg")
analyzer.load_image()
analyzer.setup_measurements()
analyzer.generate_fruit_mask()
analyzer.detect_fruits()

df = analyzer.analyze_morphology()
```

---

## Procesamiento por lotes

Procesa una carpeta completa de imágenes automáticamente.

```python
analyzer.analyze_folder(
    folder_path="mis_imagenes/",
    output_path="resultados/",
    analyze_morphology=True,
    analyze_color=True
)
```

Esto genera en `resultados/`:

- `morphology_results.csv`
- `color_results.csv` *(si `analyze_color=True`)*
- `*_annotated.jpg` por cada imagen
- `session_report.txt`
- `error_report.txt` *(si alguna imagen falló)*

---

## ¿Qué sigue?

- [Tutorial de Análisis Interno](internal.md) — guía detallada con todos los parámetros
- [Tutorial de Análisis Externo](external.md) — análisis de fruta completa paso a paso
- [Referencia API](../api/internal_analysis.md) — documentación completa de parámetros
- [Tabla de Traits](../traits.md) — qué significa cada columna del CSV
