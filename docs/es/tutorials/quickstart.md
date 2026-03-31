---
hide:
  - navigation
  - toc
---
<div style="text-align: center;" markdown>

# Inicio Rápido: análisis completo corriendo en minutos.

<p style="color:gray; margin-top: -35px; margin-bottom: 55px;" markdown>**Traitly v0.1.0 – Marzo, 2026**</p>

</div>

**Requisitos:** Traitly instalado ([Guía de instalación](../installation.md)).

!!! tip "Imágenes de muestra"
    :fontawesome-solid-file-code: Los ejemplos de este tutorial usan imágenes de **cranberry**. Si no cuentas con tus propias imágenes, puedes descargar las imágenes de muestra [aquí](LINK). Para frutos con una estructura interna más compleja, explora el tutorial [Segmentación de Lóculos](segmentate_locules.md).

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

- [Guía para el Análisis Interno](../user_guide/internal_class.md) — guía detallada con todos los parámetros y métodos disponibles para `FruitInternalAnalyzer`.
- [Guía para el Análisis Externo](../user_guide/external_class.md) — guía detallada con todos los parámetros y métodos disponibles para `FruitExternalAnalyzer`.
- [Tutorial de Análisis Externo](external.md) — analizando una imagen paso a paso.
- [Tabla de Traits](../user_guide/results/measurements.md) — qué significa cada columna del CSV.

<div style="text-align: center;" markdown>

[← Volver a Tutoriales](overview.md){ .md-button style="background-color: black; color: white; border-color: black;" }

</div>
