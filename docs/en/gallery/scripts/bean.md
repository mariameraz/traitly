---
hide:
  - navigation
  - toc

---

Download this example's image [here](https://swiftimg.com/i/KsyBWIBs).

```python
from traitly.fruit_phenotyping import FruitExternalAnalyzer

straw = FruitExternalAnalyzer("Test_38.jpg")
straw.load_image()
straw.setup_measurements()
straw.generate_fruit_mask(background_color = 'gray')
straw.detect_fruits()
straw.analyze_morphology(
                    pericarp_ext_thickness=5, 
                    pericarp_ext_color=(66, 228, 240),
                    font_size= 3, plot_size = (20,20)
                    )
```


<br>


<div style="text-align: center;" markdown>

[← Back to Gallery](../overview.md){ .md-button }

</div>
