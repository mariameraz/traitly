# Análisis de apariencia externa — procesamiento por lotes

*Versión de Traitly utilizada en este tutorial: 0.1.0*

En este tutorial demostraremos cómo realizar el análisis de apariencia externa en una carpeta con varias fotos de manera automatizada. A diferencia del análisis en imágenes individuales, el procesamiento por lotes se realiza con `analyze_folder()`, el cual conserva todos los argumentos ajustables del análisis individual. Como se mostró en el análisis individual, existen algunos parámetros predeterminados (por ejemplo, el color del fondo) que podrían funcionar o no dependiendo de las características de tus imágenes. Por esta razón, se recomienda que antes de iniciar el análisis de todas las fotos en la carpeta se explore la efectividad de los valores predeterminados en imágenes individuales y se ajusten si es necesario, como se muestra en el [tutorial](https://github.com/mariameraz/traitly/blob/main/docs/es/tutorials/individual_img_tutorial_es.md). Lo recomendable es probar los parámetros con algunas (~2-3) imágenes diferentes de tu carpeta, especialmente si hay mucha variación (diferente exposición a la luz, colores o formas de fruto contrastantes, etc.). Una vez que estés conforme con los ajustes, guarda la información de tu sesión y los parámetros utilizados con `save_parameters()`, el cual genera un archivo `.json` con toda la información necesaria para replicar el análisis en el procesamiento por lotes.

En este tutorial ejecutaremos el análisis por lotes en una carpeta con 7 imágenes, incluyendo la que se utilizó en el tutorial de análisis individual, por lo que usaremos el archivo `.json` generado al final de dicho tutorial, el cual contiene los parámetros optimizados para las imágenes de nuestra carpeta. Así, contando con un archivo `.json`, el análisis se puede completar en un solo paso con `analyze_folder()`, como se muestra a continuación.

!!! tip "Sigue el tutorial"
    :fontawesome-solid-file-code: Descarga el cuaderno de Jupyter, el folder con las imágenes de muestra y el archivo `.json` para este tutorial [aquí]().


Primero, cargamos la clase `FruitExternalAnalyzer()` desde traitly, la imagen a analizar y el archivo `.json` con los parametros predefinidos


```python
from traitly.fruit_phenotyping import FruitExternalAnalyzer 

img_folder = '~/ext-analysis_batch_sample'

parameters = '~/ext_analysis_batch_sample/ext_analysis_sample1_parameters.json'

```

Ahora inicializamos la clase `FruitExternalAnalyzer()` para cargar la carpeta y después analizamos las imágenes con `analyze_folder()`.


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


    


¡Nuestro análisis está completado! Como se puede ver en el resultado anterior, se ofrece un resumen de las características más relevantes del análisis: el input analizado, los parámetros utilizados, las características del proceso y el output obtenido. Los resultados fueron guardados dentro de la carpeta ext_analysis_results, ubicada dentro de la carpeta donde se encontraban las imágenes analizadas, la cual contiene imágenes anotadas de cada una de las imágenes analizadas, un `.csv` con los resultados de color, otro con los resultados de morfología y un archivo `.txt` con el reporte de la sesión.
