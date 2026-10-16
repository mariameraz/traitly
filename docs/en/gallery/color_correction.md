---
hide:
  - navigation
  - toc

---

This example demonstrates color correction applied to an image from the Gehler dataset.<sup>1</sup>


```python
from traitly.color_correction import ColorCorrection

cc = ColorCorrection("8D5U5577.png")
cc.load_image()
cc.detect_color_checker()
cc.apply_color_correction(num_components= 12, 
                          scaler = "standard")
```

References:

<sup>1</sup>  Peter Gehler, Carsten Rother, Andrew Blake, Tom Minka, and Toby Sharp. “Bayesian Color Constancy Revisited.” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2008. [Dataset website](http://www.kyb.mpg.de/bs/people/pgehler/colour/index.html).

<br>

<div style="text-align: center;" markdown>

[← Back to Gallery](./overview.md){ .md-button }

</div>
