---
hide:
  - navigation
  - toc

---
# Analizando rodajas de pepino

Descarga la imagen de ejemplo [aquí](https://i.swiftimg.com/originals/DWRm7Rt2.jpg).

```python

from traitly.fruit_phenotyping import FruitInternalAnalyzer

cc = FruitInternalAnalyzer("IMG_20250725_0001.jpg")

cc.load_image()

cc.generate_fruit_mask()

cc.enhance_locule_contrast(contrast_method= "gamma", gamma= 2)
cc.generate_locule_mask(erosion_px=35)

cc.detect_fruits(plot = True, min_locule_area = 200)

cc.analyze_morphology(contour_mode = "hull")
```

<br>

<div style="text-align: center;" markdown>

[← Regresar a la Galería](../overview.md){ .md-button }

</div>
