---
hide:
  - navigation
  - toc
---

<div style="text-align: center;" markdown>

# Procesamiento por lotes

<p style="color:gray; margin-top: -35px; margin-bottom: 55px;" markdown>**Creado por: Héctor López-Moreno; Traitly v0.1.0 – Marzo, 2026**</p>

</div>

En este tutorial demostraremos cómo utilizar `FruitExternalAnalyzer` para procesar de manera automatizada múltiples imágenes.

!!! note
    Aunque este tutorial se enfoca en `FruitExternalAnalyzer`, el mismo flujo de trabajo aplica para el procesamiento por lotes con `FruitInternalAnalyzer`.
  
!!! tip "Sigue el tutorial"
    :fontawesome-solid-file-code: Descarga el cuaderno de Jupyter, el folder con las imágenes de muestra y el archivo `.json` para este tutorial [aquí](https://github.com/mariameraz/traitly/tree/main/tutorials/ext_analysis_batch_sample).

A diferencia del [análisis en imágenes individuales](individual_img_tutorial.md), el procesamiento por lotes se realiza corriendo solamente `analyze_folder()`, el cual itera todos los pasos del análisis individual para cada imagen encontrada en una carpeta. 

Existen algunos parámetros predeterminados (por ejemplo, el color del fondo, tamaño mínimo de los frutos, etcétera) que podrían funcionar o no dependiendo de las características de tus imágenes. Por esta razón, se recomienda explorar la efectividad de los valores predeterminados en imágenes individuales y ajustarlos si es necesario antes de procesar toda la carpeta, como se muestra en el [tutorial de apariencia externa](individual_img_tutorial.md). Lo recomendable es probar los parámetros con algunas (~2-3) imágenes diferentes, especialmente si hay mucha variación (diferente exposición a la luz, colores o formas de fruto contrastantes, etc.). Una vez conforme con los ajustes, guarda la información de tu sesión con `save_parameters()`, el cual genera un archivo `.json` con toda la información necesaria para replicar el análisis en el procesamiento por lotes.

En este tutorial ejecutaremos el análisis por lotes en una carpeta con 7 imágenes, incluyendo la que se utilizó en el tutorial de análisis individual. Usaremos el archivo `.json` generado al final de dicho tutorial, el cual contiene los parámetros optimizados para nuestras imágenes. Con este archivo, el análisis se completa en un solo paso con `analyze_folder()`, como se muestra a continuación.

Primero, cargamos la clase `FruitExternalAnalyzer()` desde Traitly, y definimos el path a la carpeta a analizar y el archivo `.json` con los parametros predefinidos que vamos a utilizar.

```python
from traitly.fruit_phenotyping import FruitExternalAnalyzer 

img_folder = '~/ext-analysis_batch_sample'

parameters = '~/ext_analysis_batch_sample/ext_analysis_sample1_parameters.json'

```

Ahora inicializamos la clase `FruitExternalAnalyzer()` para cargar la carpeta y después analizamos las imágenes con `analyze_folder()`.

??? note "Acerca de la carpeta de resultados"
    En este caso utilizamos el parámetro `output_path` para especificar el directorio donde se guardarán los resultados. Sin embargo, este parámetro es *opcional*. Si no se indica, se generará automáticamente una carpeta llamada `Results` dentro de la carpeta donde están tus imágenes.

```python
folder_test = FruitExternalAnalyzer(img_folder)

folder_test.analyze_folder( 
    json_path = parameters,
    output_path = '~ext-analysis_batch_sample/ext_analysis_results',
    analyze_morphology = True,
    analyze_color = True
    )  
```

    ============================================================
     Traitly running ⋆✧｡٩(ˊᗜˋ )و✧*｡   
    ============================================================
        > Input folder: ~/ext-analysis_batch_sample
        > Image(s) detected: 7
        > analyze_morphology: True
        > analyze_color: True
        > num_cores: 1
        > Parameters loaded from: ~/ext_analysis_ind_img_sample/ext_analysis_sample1_parameters.json
    


    Processing images: 100%|██████████| 7/7 [00:10<00:00,  1.50s/img]

    
    ( ദ്ദി ˙ᗜ˙ ) Finished ===============================================
        > Image(s) processed:
            - Successfully: 7/7 img(s)
            - Total fruits: 195
            - Total time: 10.5s  (avg 1.5s/img)
        > Files saved:
            - 7 annotated image(s)
            - morphology_results.csv
            - color_results.csv
            - session_report.txt
            - Results folder: ~/ext-analysis_batch_sample/ext_analysis_results


    


¡Nuestro análisis está completado! Como se puede ver en el resultado anterior, se ofrece un resumen de las características más relevantes del análisis: el input analizado, los parámetros utilizados, las características del proceso y el output obtenido.


## ¿Qué sigue?

- [Archivos de resultados](../user_guide/results/overview.md) – descripción de los archivos generados por Traitly
- [Tabla de Traits](../user_guide/results/measurements.md) – qué significa cada columna del CSV

<div style="text-align: center;" markdown>

[← Volver a Tutoriales](overview.md){ .md-button }

</div>
