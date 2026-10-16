---
hide:
  - navigation
  - toc

---

Download this example's image [here](https://swiftimg.com/i/3DXKzt9Z).

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

[← Back to Gallery](../overview.md){ .md-button }

</div>
