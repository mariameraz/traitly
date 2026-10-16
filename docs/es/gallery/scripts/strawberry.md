---
hide:
  - navigation
  - toc

---

# Análisis de color de fresas 

Descarga la imagen de ejemplo [aquí](https://swiftimg.com/i/3DXKzt9Z).

```python
from traitly.fruit_phenotyping import FruitExternalAnalyzer

straw = FruitExternalAnalyzer("Test_02.jpg")
straw.load_image()
straw.generate_fruit_mask()
straw.detect_fruits()
straw.analyze_color(plot = True, 
                    display_table = False, 
                    font_size= 5, 
                    pericarp_ext_thickness= 10)
```

<br>

<div style="text-align: center;" markdown>

[← Regresar a la Galería](../overview.md){ .md-button }

</div>
