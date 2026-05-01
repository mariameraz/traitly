---
hide:
  - navigation
  - toc
---

<div style="text-align: center;" markdown>

# Análisis de la morfología interna del fruto en Cranberry

<p style="color:gray; margin-top: -35px; margin-bottom: 55px;" markdown>*Creado por: María A. Torres-Meraz; Traitly v0.1.0 – Abril, 2026*</p>

</div>

En este tutorial, demostraremos cómo realizar el análisis de la morfología interna de frutos utilizando `FruitInternalAnalyzer`, una herramienta para extraer medidas de morfología y color a partir de una imagenes de rodajas transversales de cranberry.

!!! tip "Sigue el tutorial"
    :fontawesome-solid-file-code: Descarga el cuaderno de Jupyter y las imágenes de muestra para este tutorial [aquí](https://github.com/mariameraz/traitly/tree/main/tutorials/cranberry_internal_analysis).

!!! info "Referencia de métodos y parámetros"
    A lo largo de este tutorial se utilizan métodos como `setup_measurements()`, `generate_fruit_mask()`, `detect_fruits()`, `analyze_morphology()` y `analyze_color()`. Para una descripción completa de cada método y sus parámetros disponibles, consulta la [Clase Internal Analyzer](../user_guide/internal_class.md).


El primer paso es cargar la clase `FruitInternalAnalyzer` desde Traitly para crear el objeto `cranberry`, que contendrá todo lo necesario para el análisis. Puedes elegir el nombre de objeto que prefieras.

A continuación, inicializamos la clase indicando la ubicación de nuestra imagen mediante el parámetro `path`.  

```python
from traitly.fruit_phenotyping import FruitInternalAnalyzer
```

```python
input_path = "./cranberry_slices.jpg"
cranberry = FruitInternalAnalyzer(path = input_path)
```

Primero, cargamos la imagen en el objeto mediante `load_image()`. Por defecto, la imagen se mostrará en pantalla (`plot=True`). Una vez cargada, puede accederse a ella a través de `cranberry.img`. Para más detalles sobre los datos almacenados en el objeto, consulta el apartado de [atributos de la clase].


```python
cranberry.load_image() 
```


    
![png](../../en/tutorials/cranberry_internal_analysis_files/cranberry_internal_analysis_4_0.png)
    


A continuación, ejecutamos `setup_measurements()` para establecer el diámetro de las referencias de tamaño (círculos negros) y, opcionalmente, extraer la información del código QR de la etiqueta. Activando `plot_reference=True` podemos inspeccionar en detalle la detección de la referencia y el diámetro en píxeles de cada círculo.

Como indican los resultados, se detectó una tira de círculos (`Ref 1`) compuesta por 6 círculos. Antes de calcular el promedio, `setup_measurements()` elimina los círculos cuya desviación estándar es mayor a 2, para evitar ruido por círculos mal detectados o muy diferentes al resto. En este caso, se usaron 5 de los 6 círculos para obtener un diámetro promedio de 218 px. Este valor se divide entre el diámetro real promedio en centímetros para obtener la densidad de píxeles por cm, que se usará para convertir píxeles a centímetros en análisis posteriores.


```python
cranberry.setup_measurements(detect_label = True,
                            diameter_cm = 1.7, 
                            plot_reference = True)
```

    
    =======================================================
    ★ LABEL DETECTION:
    =======================================================
    > QR Code detected: DP14-313 (0.14s)
    
    =======================================================
    ★ REFERENCE SIZE:
    =======================================================
    > Reference size detected:
      - Processing reference box(es) with a confidence threshold >=0.6:
                Ref 1: 1904x297 px, conf: 0.821
    
      - Total circles detected: 6
                Filtered circles: 5/6 (std > 2)
                Mean diameter: 218.0 ± 0.0 px
    
            . ݁₊ ⊹ . ݁ ⟡ ݁ px/cm density: 128.2 (diameter_cm: 1.7 cm) ⟡ ݁ . ⊹ ₊ ݁.


    
![png](cranberry_internal_analysis_files/cranberry_internal_analysis_6_1.png)
    

A continuación, generamos una máscara binaria de frutos y lóculos con `generate_fruit_mask()`, donde los lóculos aparecerán en negro y el resto del fruto en blanco.


```python
cranberry.generate_fruit_mask()
```


    
![png](cranberry_internal_analysis_files/cranberry_internal_analysis_8_0.png)
    


Después, detectamos los frutos en la máscara con `detect_fruits()` y hacemos una inspección rápida con `plot=True`. La imagen muestra para cada fruto su contorno en verde, los contornos de sus lóculos en rosa y el área del pericarpio interno en cyan. Esta visualización permite corroborar la detección y determinar si es necesario ajustar los parámetros de segmentación.


```python
cranberry.detect_fruits(plot = True,
                       plot_size = (10,10))
```

    
    =====================================
            . ݁₊ ⊹ . ݁ ⟡ ݁ Detected fruits: 25 ⟡ ݁ . ⊹ ₊ ݁.
    
     > Parameters used:
            - min_fruit_circularity: 0.5
            - min_locule_area: 50
            - min_locule_per_fruit: 1
            - min_fruit_area: 5000
    =====================================



    
![png](cranberry_internal_analysis_files/cranberry_internal_analysis_10_1.png)
    


Ahora, llevamos a cabo el analisis morfológico con `analyze_morphology`, el cual genera una copia de la imagen original con anotaciones para cada fruto y un DataFrame con los resultados. En ambos archivos, cada fruto es asignado a un identificador único (`id`) el cual es útil para comparar visualizar los datos numéricos.

```python
cranberry.analyze_morphology()
```
    
![png](cranberry_internal_analysis_files/cranberry_internal_analysis_12_0.png)
    

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_name</th>
      <th>label</th>
      <th>fruit_id</th>
      <th>n_locules</th>
      <th>unit</th>
      <th>fruit_area_cm2</th>
      <th>fruit_perimeter_cm</th>
      <th>fruit_circularity</th>
      <th>fruit_solidity</th>
      <th>fruit_convexity</th>
      <th>...</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cranberry_slices.jpg</td>
      <td>DP14-313</td>
      <td>1</td>
      <td>4</td>
      <td>cm</td>
      <td>1.688945</td>
      <td>5.019779</td>
      <td>0.842280</td>
      <td>0.988715</td>
      <td>0.927457</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cranberry_slices.jpg</td>
      <td>DP14-313</td>
      <td>2</td>
      <td>4</td>
      <td>cm</td>
      <td>1.491642</td>
      <td>4.738720</td>
      <td>0.834742</td>
      <td>0.987420</td>
      <td>0.921867</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cranberry_slices.jpg</td>
      <td>DP14-313</td>
      <td>3</td>
      <td>4</td>
      <td>cm</td>
      <td>1.637134</td>
      <td>4.970773</td>
      <td>0.832619</td>
      <td>0.988616</td>
      <td>0.920782</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cranberry_slices.jpg</td>
      <td>DP14-313</td>
      <td>4</td>
      <td>4</td>
      <td>cm</td>
      <td>1.831669</td>
      <td>5.227883</td>
      <td>0.842181</td>
      <td>0.988546</td>
      <td>0.925857</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cranberry_slices.jpg</td>
      <td>DP14-313</td>
      <td>5</td>
      <td>4</td>
      <td>cm</td>
      <td>2.280032</td>
      <td>5.840573</td>
      <td>0.839924</td>
      <td>0.988831</td>
      <td>0.924711</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cranberry_slices.jpg</td>
      <td>DP14-313</td>
      <td>6</td>
      <td>4</td>
      <td>cm</td>
      <td>2.128824</td>
      <td>5.763834</td>
      <td>0.805242</td>
      <td>0.986585</td>
      <td>0.907714</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>cranberry_slices.jpg</td>
      <td>DP14-313</td>
      <td>7</td>
      <td>4</td>
      <td>cm</td>
      <td>1.692107</td>
      <td>5.080596</td>
      <td>0.823774</td>
      <td>0.987263</td>
      <td>0.916677</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cranberry_slices.jpg</td>
      <td>DP14-313</td>
      <td>8</td>
      <td>4</td>
      <td>cm</td>
      <td>1.726800</td>
      <td>5.223315</td>
      <td>0.795352</td>
      <td>0.982850</td>
      <td>0.907571</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>cranberry_slices.jpg</td>
      <td>DP14-313</td>
      <td>9</td>
      <td>4</td>
      <td>cm</td>
      <td>2.144787</td>
      <td>5.860413</td>
      <td>0.784761</td>
      <td>0.986642</td>
      <td>0.894509</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>cranberry_slices.jpg</td>
      <td>DP14-313</td>
      <td>10</td>
      <td>4</td>
      <td>cm</td>
      <td>2.161905</td>
      <td>5.699882</td>
      <td>0.836209</td>
      <td>0.989355</td>
      <td>0.922177</td>
      <td>...</td>
    </tr>
    <th>...</th>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
  </tbody>
</table>
<p>25 rows × 38 columns</p>
</div>


Alternativamente podemos explorar a detalle la segmentación de los diferentes tejidos de un fruto con `generate_single_fruit_masks`. Mediante el parámetro `fruit_id` podemos indicar exactamente cual fruto de la imagen o la tabla queremos visualizar. 

```python
cranberry.generate_single_fruit_masks(fruit_id = 2)
```

![png](cranberry_internal_analysis_files/cranberry_internal_analysis_14_0.png)

Finalmente, analizaremos el color de cada fruto. Por default, `analyze_color()` extrae información de color en los canales `rgb`, `lab`, `hsv` y `gray` para los tejidos `total_pericarp`, `outer_pericarp`, `internal_pericarp` y `locules`.

En este caso, excluiremos los lóculos del análisis, ya que al estar huecos, solo capturan el color negro del fondo. Para seleccionar tejidos específicos utilizamos el parámetro `tissue`, y para seleccionar canales de color específicos utilizamos el parámetro `color_space`.

Al pasar múltiples valores, estos deben escribirse separados por comas, en minúsculas y con los espacios reemplazados por `_`. Por ejemplo:

- Canales RGB y HSV: `"rgb, hsv"`
- Tejidos lóculos, pericarpio total y pericarpio externo: `"locules, total_pericarp, outer_pericarp"`

Si necesitamos mayor control sobre qué tejidos analizar, las máscaras obtenidas con `generate_single_fruit_masks()` nos permiten seleccionar únicamente los tejidos más relevantes para el análisis.


```python
tissues_ext = "total_pericarp, outer_pericarp, internal_pericarp"

cranberry.analyze_color(tissue = tissues_ext,
                       color_space = "rgb")
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_name</th>
      <th>label</th>
      <th>fruit_id</th>
      <th>tissue</th>
      <th>R_mean</th>
      <th>G_mean</th>
      <th>B_mean</th>
      <th>R_std</th>
      <th>G_std</th>
      <th>B_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cranberry_slices.jpg</td>
      <td>DP14-313</td>
      <td>1</td>
      <td>total_pericarp</td>
      <td>140.147522</td>
      <td>107.556854</td>
      <td>102.661514</td>
      <td>29.093695</td>
      <td>35.751049</td>
      <td>31.698030</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cranberry_slices.jpg</td>
      <td>DP14-313</td>
      <td>1</td>
      <td>internal_pericarp</td>
      <td>139.475250</td>
      <td>100.753677</td>
      <td>95.946632</td>
      <td>29.493685</td>
      <td>43.233368</td>
      <td>39.668324</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cranberry_slices.jpg</td>
      <td>DP14-313</td>
      <td>1</td>
      <td>outer_pericarp</td>
      <td>142.934784</td>
      <td>111.056740</td>
      <td>105.959541</td>
      <td>26.142126</td>
      <td>32.451111</td>
      <td>28.258091</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cranberry_slices.jpg</td>
      <td>DP14-313</td>
      <td>2</td>
      <td>total_pericarp</td>
      <td>148.301102</td>
      <td>119.346420</td>
      <td>105.754677</td>
      <td>27.769527</td>
      <td>30.698967</td>
      <td>26.121008</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cranberry_slices.jpg</td>
      <td>DP14-313</td>
      <td>2</td>
      <td>internal_pericarp</td>
      <td>145.619812</td>
      <td>122.564034</td>
      <td>102.932617</td>
      <td>19.196663</td>
      <td>18.638998</td>
      <td>24.833715</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>cranberry_slices.jpg</td>
      <td>DP14-313</td>
      <td>24</td>
      <td>internal_pericarp</td>
      <td>147.718323</td>
      <td>114.503822</td>
      <td>107.229797</td>
      <td>28.367800</td>
      <td>36.551113</td>
      <td>33.617943</td>
    </tr>
    <tr>
      <th>71</th>
      <td>cranberry_slices.jpg</td>
      <td>DP14-313</td>
      <td>24</td>
      <td>outer_pericarp</td>
      <td>147.333191</td>
      <td>117.123642</td>
      <td>111.132996</td>
      <td>29.128061</td>
      <td>32.934532</td>
      <td>28.674442</td>
    </tr>
    <tr>
      <th>72</th>
      <td>cranberry_slices.jpg</td>
      <td>DP14-313</td>
      <td>25</td>
      <td>total_pericarp</td>
      <td>147.192123</td>
      <td>114.394127</td>
      <td>100.128860</td>
      <td>29.666571</td>
      <td>34.832909</td>
      <td>29.787176</td>
    </tr>
    <tr>
      <th>73</th>
      <td>cranberry_slices.jpg</td>
      <td>DP14-313</td>
      <td>25</td>
      <td>internal_pericarp</td>
      <td>147.941788</td>
      <td>121.066788</td>
      <td>101.173103</td>
      <td>24.562567</td>
      <td>26.238270</td>
      <td>26.743542</td>
    </tr>
    <tr>
      <th>74</th>
      <td>cranberry_slices.jpg</td>
      <td>DP14-313</td>
      <td>25</td>
      <td>outer_pericarp</td>
      <td>150.031143</td>
      <td>115.447060</td>
      <td>102.027443</td>
      <td>27.291199</td>
      <td>34.938839</td>
      <td>28.972141</td>
    </tr>
  </tbody>
</table>
<p>75 rows × 10 columns</p>
</div>



Al ejecutar `analyze_morphology()` y/o `analyze_color()`, se genera el objeto `results`, que contiene los resultados del análisis junto con los métodos necesarios para exportarlos. Para guardar todos los resultados en un solo paso, utilizamos `save_all()` como se muestra a continuación.


```python
cranberry.results.save_all()

## Alternativas:
#cranberry.results.save_img() # Guarda solo la imagen anotada
# cranberry.results.save_csv() # Guarda solo el o los CSV generados

```

    > Results saved at:
        – Image: /Users/traitly/tutorials/internal_analysis/cranberry_slices_processed.jpg
        – Morphology CSV: /Users/traitly/tutorials/internal_analysis/cranberry_slices_morphology_results.csv
        – Color CSV: /Users/traitly/tutorials/internal_analysis/cranberry_slices_color_results.csv


También podemos exportar los parámetros utilizados en la sesión con `save_parameters()`, que genera dos archivos: uno `.txt` y uno `.json`. El archivo `.txt` está pensado para el usuario y contiene los parámetros de cada método, la versión de Traitly, la fecha y hora del análisis, y el nombre de la imagen. El archivo `.json`, en cambio, es útil para replicar el análisis, por ejemplo al procesar múltiples imágenes con `analyze_folder()` o al usar Traitly desde la terminal.

```python
cranberry.save_parameters()
```

    
    > Parameters saved at:
      - TXT:  /Users/traitly/tutorials/internal_analysis/cranberry_slices_parameters.txt
      - JSON: /Users/traitly/tutorials/internal_analysis/cranberry_slices_parameters.json



## ¿Qué sigue?

- [Cómo realizar un procesamiento por lotes](batch_tutorial.md) – análisis de la apariencia externa de múltiples imágenes.
- [Tabla de Traits](../user_guide/results/measurements.md) – qué significa cada columna del CSV
- [Guía para el Análisis Interno](../user_guide/internal_class.md) — guía detallada con todos los parámetros y métodos disponibles para `FruitInternalAnalyzer`.

<div style="text-align: center;" markdown>

[← Volver a Tutoriales](overview.md){ .md-button }

</div>
