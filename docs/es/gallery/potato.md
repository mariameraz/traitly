---
hide:
  - navigation
  - toc

---

# Análisis morfológico de papas

Descarga la imagen de ejemplo [aquí](https://swiftimg.com/i/f2zgHX9U).

```python
from traitly.fruit_phenotyping import FruitExternalAnalyzer

straw = FruitExternalAnalyzer("Test_40.jpg")

straw.load_image()

straw.setup_measurements()

straw.generate_fruit_mask(lower_hsv= (0,0,50), 
                          upper_hsv = (180,30,225), 
                          plot_size = (20,20))

straw.detect_fruits(min_fruit_circularity= 0.3)

straw.analyze_morphology(pericarp_ext_thickness=5, 
                         pericarp_ext_color=(66, 228, 240),
                         font_size= 3, plot_size = (20,20))
```

<br>

<div style="text-align: center;" markdown>

[← Regresar a la Galería](./overview.md){ .md-button }

</div>
