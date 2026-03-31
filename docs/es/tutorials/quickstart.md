---
hide:
  - navigation
  - toc
---
<div style="text-align: center;" markdown>

# Inicio Rápido: análisis completo de cranberry corriendo en minutos.

<p style="color:gray; margin-top: -35px; margin-bottom: 55px;" markdown>**Traitly v0.1.0 – Marzo, 2026**</p>

</div>

**Requisitos:** Traitly instalado ([Guía de instalación](../installation.md)).

!!! tip "Imágenes de muestra"
    :fontawesome-solid-file-code: Los ejemplos de este tutorial usan imágenes de **cranberry**. Si no cuentas con tus propias imágenes, puedes descargar las imágenes de muestra [aquí](https://github.com/mariameraz/traitly/tree/main/tutorials_data/images). Para frutos con una estructura interna más compleja (ej., tomate, naranja o pepino), explora el tutorial [Segmentación de Lóculos](segmentate_locules.md).

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

# Análisis morfológico y de color
analyzer.analyze_morphology()
analyzer.analyze_color()

# Guarda resultados -> retorna un CSV y una imagen anotada
analyzer.results.save_all()

# Opcionalmente, guarda los parámetros usados en la sesión
analyzer.save_paramenters()
```

---

## Análisis externo

Usa `FruitExternalAnalyzer` para análisis del fruto completo (sin segmentación de lóculos y otras estructuras internas).

```python
from traitly.fruit_phenotyping import FruitExternalAnalyzer

analyzer = FruitExternalAnalyzer("mi_imagen.jpg")
analyzer.load_image()
analyzer.setup_measurements()
analyzer.generate_fruit_mask()
analyzer.detect_fruits()

# Análisis morfológico y de color
analyzer.analyze_morphology()
analyzer.analyze_color()

# Guarda resultados -> retorna un CSV y una imagen anotada
analyzer.results.save_all()

# Opcionalmente, guarda los parámetros usados en la sesión
analyzer.save_paramenters()
```

---

## Procesamiento por lotes

Procesa una carpeta completa de imágenes automáticamente.

```python
# Analisis interno
from traitly.fruit_phenotyping import FruitInternalAnalyzer

analyzer = FruitInternalAnalyzer("PATH_FOLDER/")

analyzer.analyze_folder(
    folder_path="mis_imagenes/",
    output_path="resultados/",
    analyze_morphology=True,
    analyze_color=True,
    json_path="path/file.json" # Opcional, útil para definir parametros 
)

# Analisis externo

from traitly.fruit_phenotyping import FruitExternalAnalyzer

analyzer = FruitExternalAnalyzer("PATH_FOLDER/")

analyzer.analyze_folder(
    folder_path="mis_imagenes/",
    output_path="resultados/",
    analyze_morphology=True,
    analyze_color=True,
    json_path="path/file.json" # Opcional, útil para definir parametros 
)
```

Esto genera una carpeta llamada `resultados/` con:

- `morphology_results.csv` *(si `analyze_morphology=True`)* 
- `color_results.csv` *(si `analyze_color=True`)*
- `*_annotated.jpg` por cada imagen analizada **exitosamente**
- `session_report.txt`
- `error_report.txt` *(solo si alguna imagen falló)*

---

## ¿Qué sigue?

- [Guía para el Análisis Interno](../user_guide/internal_class.md) — guía detallada con todos los parámetros y métodos disponibles para `FruitInternalAnalyzer`.
- [Guía para el Análisis Externo](../user_guide/external_class.md) — guía detallada con todos los parámetros y métodos disponibles para `FruitExternalAnalyzer`.
- [Tutorial de Análisis Externo](individual_img_tutorial.md) — analizando una imagen paso a paso.
- [Tabla de Traits](../user_guide/results/measurements.md) — qué significa cada columna del CSV.

<div style="text-align: center;" markdown>

[← Volver a Tutoriales](overview.md){ .md-button }

</div>
