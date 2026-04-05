---
hide:
  - navigation
  - toc
---

<div style="text-align: center;" markdown>

# Análisis de apariencia externa — Imagen individual

<p style="color:gray; margin-top: -35px; margin-bottom: 55px;" markdown>**Creado por: Héctor López-Moreno; Traitly v0.1.0 – Marzo, 2026**</p>

</div>

En este tutorial, demostraremos cómo realizar el análisis de apariencia externa de frutos utilizando `FruitExternalAnalyzer`, una herramienta para extraer medidas de morfología y color a partir de una sola imagen.

!!! tip "Sigue el tutorial"
    :fontawesome-solid-file-code: Descarga el cuaderno de Jupyter y las imágenes de muestra para este tutorial [aquí](https://github.com/mariameraz/traitly/tree/main/tutorials_data/ext_analysis_ind_img_sample_es).

!!! info "Referencia de métodos y parámetros"
    A lo largo de este tutorial se utilizan métodos como `setup_measurements()`, `generate_fruit_mask()`, `detect_fruits()`, `analyze_morphology()` y `analyze_color()`. Para una descripción completa de cada método y sus parámetros disponibles, consulta la [Clase External Analyzer](../user_guide/external_class.md).

Primero, cargamos la clase `FruitExternalAnalyzer` desde Traitly y la imagen a analizar.

```python
from traitly.fruit_phenotyping import FruitExternalAnalyzer


input_path_img = '~/ext_analysis_sample1.jpg'

pic_test = FruitExternalAnalyzer(path_img)

pic_test.load_image()
```
    
![png](individual_img_tutorial_es_files/individual_img_tutorial_es_4_0.png)
    

Después ejecutamos `setup_measurements()` para detectar las referencias de tamaño en la imagen (círculos negros).

```python
pic_test.setup_measurements()
```

    
    =======================================================
    ★ LABEL DETECTION:
    =======================================================
    
    =======================================================
    ★ REFERENCE SIZE:
    =======================================================
    > Reference size detected:
      - Processing reference box(es) with a confidence threshold >=0.6:
                Ref 1: 453x2532 px, conf: 0.948
                Ref 2: 452x2547 px, conf: 0.942
    
      - Total circles detected: 12
                Range: [310.2, 314.1] px
                Filtered circles: 11/12 (std > 2)
    
            . ݁₊ ⊹ . ݁ ⟡ ݁ px/cm density: 124.9 (diameter_cm: 2.5 cm) ⟡ ݁ . ⊹ ₊ ݁.
    
    Note: Default reference diameter (2.5 cm) applied.
            Specify diameter_cm to override this value.


Ahora generemos las máscaras de frutos con `generate_fruit_mask()` usando el valor de color de fondo predeterminado `white` y veamos qué objetos en la imagen fueron detectados como frutos con `detect_fruits()`.

!!! note "Sobre el color del fondo"
    Si necesitas segmentar frutos sobre un fondo de color diferente a los predeterminados – `white`, `blue` y `black` – puedes consultar el tutorial de [Segmentación de Fondo](background_segmentation.md) para más detalles.

Como podemos ver en los gráficos generados a continuación, la mayoría de los frutos han sido segmentados efectivamente. Sin embargo, notamos que algunos frutos no fueron detectados (no tienen el contorno verde) por `detect_fruits()`. En estos casos, necesitamos modificar algunos parámetros para mejorar tanto las máscaras como la detección; abordemos eso a continuación.

```python
pic_test.generate_fruit_mask(background_color='white')

pic_test.detect_fruits(
    plot=True, plot_size=(5,5),
    contour_thickness=7,
)    
```
    
![png](individual_img_tutorial_es_files/individual_img_tutorial_es_8_0.png)
    
![png](individual_img_tutorial_es_files/individual_img_tutorial_es_8_1.png)
    


    
    =====================================
            . ݁₊ ⊹ . ݁ ⟡ ݁ Detected fruits: 28 ⟡ ݁ . ⊹ ₊ ݁.
    
     > Parameters used:
            - min_fruit_circularity: 0.5
            - min_fruit_area: 500
    =====================================


En `generate_fruit_mask()`, las máscaras de los frutos a veces pueden mostrar hendiduras o regiones de las orillas sin segmentar correctamente. Para cerrar estos espacios, podemos usar el parámetro `apply_convex_hull=True`, que aplica un [convex hull](https://www.geeksforgeeks.org/dsa/convex-hull-algorithm/) alrededor del contorno del fruto, garantizando un resultado más suave y cerrado. Además, el parámetro `kernel_blur=5` también ayuda a definir mejor los contornos de los frutos, difuminando y simplificando los colores de la imágen, lo que facilita la segmentación. Tener una buena definición de los contornos es fundamental, ya que los cálculos se basan en ellos, y cualquier hueco en el contorno impacta directamente en los análisis posteriores. Opcionalmente, se puede aplicar `erosion_px=3` para eliminar algunos píxeles de alrededor del contorno cuyo color podría verse afectado por el reflejo del fondo. La erosión también ayuda a eliminar porciones del fondo que podrían estar incluidas en la máscara y que sesgarían las estimaciones de color del fruto.

En `detect_fruits()`, `min_fruit_circularity=0.4` garantiza que capturemos todos los frutos al reducir el umbral de circularidad, ya que algunos tienen una forma más alargada.


```python
pic_test.generate_fruit_mask(background_color='white',
                             apply_convex_hull=True,
                             kernel_blur=5,
                             erosion_px=3)

pic_test.detect_fruits(
    plot=True, plot_size=(5,5),
    contour_thickness=7,
    min_fruit_circularity=0.4
)
```

![png](individual_img_tutorial_es_files/individual_img_tutorial_es_10_0.png)
    
![png](individual_img_tutorial_es_files/individual_img_tutorial_es_10_1.png)
    


    
    =====================================
            . ݁₊ ⊹ . ݁ ⟡ ݁ Detected fruits: 30 ⟡ ݁ . ⊹ ₊ ݁.
    
     > Parameters used:
            - min_fruit_circularity: 0.4
            - min_fruit_area: 500
    =====================================


Ahora que los frutos han sido correctamente segmentados y detectados, podemos realizar los análisis morfológicos y de color.

```python
pic_test.analyze_morphology(display_table=False,
                            plot=True,
                            plot_size=(5,5))

pic_test.analyze_color(display_table=False,
                       plot=False)
```

![png](individual_img_tutorial_es_files/individual_img_tutorial_es_12_0.png)

Al ejecutar `analyze_morphology()` y `analyze_color()`, se generará el objeto `resultados` que contiene el método `save_all()`, el cual podemos llamar de la siguiente manera para guardar tanto los archivos CSV con los resultados de cada análisis como la imagen anotada. En la imagen anotada, podemos verificar que tanto los frutos como las referencias de tamaño han sido correctamente detectadas.

```python
pic_test.results.save_all()
```

    Image saved at: ./ext_analysis_sample1_processed.jpg
    Morphology CSV saved at: ./ext_analysis_sample1_morphology_results.csv
    Color CSV saved at: ./ext_analysis_sample1_color_results.csv

Opcionalmente, puedes guardar los parámetros e información de la sesión de tu análisis con `save_parameters()` para garantizar la reproducibilidad de tus análisis futuros o para utilizarlos en el procesamiento por lotes.

```python
pic_test.save_parameters()
```

    
    > Parameters saved:
      - TXT:  ./ext_analysis_sample1_parameters.txt
      - JSON: ./ext_analysis_sample1_parameters.json


## ¿Qué sigue?

- [Cómo realizar un procesamiento por lotes](batch_tutorial.md) – análisis de la apariencia externa de múltiples imágenes.
- [Tabla de Traits](../user_guide/results/measurements.md) – qué significa cada columna del CSV


<div style="text-align: center;" markdown>

[← Volver a Tutoriales](overview.md){ .md-button }

</div>
