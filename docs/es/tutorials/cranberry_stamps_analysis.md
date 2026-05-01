---
hide:
  - navigation
  - toc
---

<div style="text-align: center;" markdown>

# Análisis de la morfología interna del fruto en Cranberry – Estampas

<p style="color:gray; margin-top: -35px; margin-bottom: 55px;" markdown>*Creado por: María A. Torres-Meraz; Traitly v0.1.0 – Abril, 2026*</p>

</div>


En este tutorial, demostraremos cómo realizar el análisis de la morfología interna de frutos de cranberry usando imágenes de estampas mediante `FruitInternalAnalyzer`. Aquí nos enfocaremos en qué parámetros ajustar cuando tenemos este tipo de imágenes. Para revisar qué hace cada método con más detalle y cómo se realiza un análisis completo con `FruitInternalAnalyzer`, ver el tutorial de [Análisis de morfología y color en Cranberry](./cranberry_internal_analysis.md).

!!! tip "Sigue el tutorial"
    :fontawesome-solid-file-code: Descarga el cuaderno de Jupyter y las imágenes de muestra para este tutorial [aquí](https://github.com/mariameraz/traitly/tree/main/tutorials/cranberry_internal_analysis).


El primer paso es inicializar la clase `FruitInternalAnalyzer` y cargar nuestra imagen.

```python
# Importar la clase
from traitly.fruit_phenotyping import FruitInternalAnalyzer
```

```python
# Crear el objeto `cranberry`
input_path = "./cranberry_stamps.jpg"
cranberry = FruitInternalAnalyzer(input_path)

# Cargar nuestra imagen
cranberry.load_image()
```
    
![png](cranberry_stamps_analysis_files/cranberry_stamps_analysis_2_0.png)

Al momento de realizar las estampas, puede ser que algunas de ellas tengan errores que dificulten el análisis. Hay varias maneras de eliminar esos frutos de la imagen: una de ellas sería eliminar manualmente de la máscara las estampas con `cranberry.edit_mask()`, o más sencillamente, marcar con una X nuestra estampa. Esto se hace con la intención de filtrar posteriormente estos frutos según su circularidad en `cranberry.detect_fruits`. En este ejemplo, seguimos esta última estrategia.

A pesar de que la imagen no incluye ninguna referencia de tamaño, aún podemos convertir píxeles a centímetros si conocemos el tamaño de la hoja escaneada en centímetros. Las mediciones se pasan como se muestra a continuación:

```python
cranberry.setup_measurements(width_cm = 21.6, 
                             length_cm = 27.9)
```

    
    =======================================================
    ★ LABEL DETECTION:
    =======================================================
    > Label detection: SKIPPED (detect_label=False)
    
    =======================================================
    ★ REFERENCE SIZE:
    =======================================================
    > Size reference detection: SKIPPED (skip_yolo=True).
    
    > Using provided physical dimensions:
        - width_cm:  21.6 cm
        - length_cm: 27.9 cm
    
            . ݁₊ ⊹ . ݁ ⟡ ݁ px/cm density: 115.26 ⟡ ݁ . ⊹ ₊ ݁.


Ahora procederemos a crear la máscara binaria, en donde separaremos el fondo de los demás objetos en la imagen. Por default, `generate_fruit_mask` espera un fondo de color negro, pero en las imágenes de estampas el fondo es de color blanco. En este caso, utilizaremos el parámetro `stamp=True`, lo que le indica a `generate_fruit_mask` que debe invertir el color de la imagen antes de generar la máscara, de modo que el color blanco del papel pase a ser negro y la máscara pueda segmentar correctamente el fondo de las estampas.

```python
cranberry.generate_fruit_mask(stamp = True)
```

![png](cranberry_stamps_analysis_files/cranberry_stamps_analysis_6_0.png)
    
Con la máscara lista, podemos proceder como con cualquier otro tipo de imagen, detectando frutos con `detect_fruits`. Si las estampas contienen pequeños espacios sin tinta, estos podrían causar ruido en la detección de los lóculos, para lo cual podemos ajustar `min_locule_area`.

Como podemos ver en los resultados, las estampas marcadas con una cruz no fueron detectadas como frutos, gracias a que filtramos contornos según su circularidad con `min_fruit_circularity=0.5`.


```python
cranberry.detect_fruits(plot = True, 
                        plot_size = (15,15),
                        min_locule_area = 300)
```

    
    =====================================
            . ݁₊ ⊹ . ݁ ⟡ ݁ Detected fruits: 21 ⟡ ݁ . ⊹ ₊ ݁.
    
     > Parameters used:
            - min_fruit_circularity: 0.5
            - min_locule_area: 300
            - min_locule_per_fruit: 1
            - min_fruit_area: 5000
    =====================================

    
![png](cranberry_stamps_analysis_files/cranberry_stamps_analysis_8_1.png)

Con los frutos detectados, realizamos el análisis morfológico con `analyze_morphology`.

En ocasiones, los contornos de las estampas no se encuentran bien definidos, lo que puede impactar en los resultados de morfología subestimando mediciones como el área, circularidad, perímetro, etc. Vemos un ejemplo más detallado en el fruto 15.


```python
# Analizar morfologia de los frutos
cranberry.analyze_morphology(display_table = False, plot_size = (15,15))

# Visualizar estampa num. 15
cranberry.generate_single_fruit_masks(fruit_id = 15)
```
    
![png](cranberry_stamps_analysis_files/cranberry_stamps_analysis_10_0.png)

    
![png](cranberry_stamps_analysis_files/cranberry_stamps_analysis_10_1.png)
  
Para este tipo de situaciones, en lugar de utilizar el contorno original de la estampa (`contour_mode='raw'`), podemos aplicar una transformación con `contour_mode='hull'`, el cual aplica un convex hull con ayuda de la librería [cv2](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga014b28e56cb8854c0de4a211cb2be656), corrigiendo las hendiduras de las estampas. Este método toma todos los puntos que definen el contorno original de un fruto y busca un nuevo contorno que los envuelva de manera convexa. Puedes pensarlo como si ajustaras una liga alrededor del fruto, "rellenando" los huecos en el perímetro de la estampa, tal como se puede ver en detalle en la estampa 15 revisada previamente.

```python
cranberry.analyze_morphology(display_table = False,
                             contour_mode = 'hull', 
                             plot_size = (15,15))
```

    
![png](cranberry_stamps_analysis_files/cranberry_stamps_analysis_12_0.png)
    

¡Con esto concluímos el análisis! así que procedemos a guardar los resultados.

```python
cranberry.results.save_all()
```

    > Results saved at:
        – Image: /Users/traitly/tutorials/cranberry_stamp_analysis/cranberry_stamps_processed.jpg
        – Morphology CSV: /Users/traitly/tutorials/cranberry_stamp_analysis/cranberry_stamps_morphology_results.csv


## ¿Qué sigue?

- [Cómo realizar un procesamiento por lotes](batch_tutorial.md) – análisis de la apariencia externa de múltiples imágenes.
- [Tabla de Traits](../user_guide/results/measurements.md) – qué significa cada columna del CSV
- [Guía para el Análisis Interno](../user_guide/internal_class.md) — guía detallada con todos los parámetros y métodos disponibles para `FruitInternalAnalyzer`.

<div style="text-align: center;" markdown>

[← Volver a Tutoriales](overview.md){ .md-button }

</div>
