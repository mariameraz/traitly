<div class="animate" markdown>

# Análisis Externo: Clase y Métodos

En esta sección encontrarás todo lo que necesitas para usar `FruitExternalAnalyzer`, la clase principal para analizar imágenes de frutos completos. Aquí se explica cada método, sus parámetros y cómo utilizarlos en tu flujo de trabajo.

---

## Clase `FruitExternalAnalyzer`

`FruitExternalAnalyzer` es la herramienta principal para analizar la morfología y el color de frutos completos a partir de imágenes segmentadas de frutos enteros (sin incluir segmentación de tejidos internos). Puedes utilizarla para procesar una sola imagen o una carpeta completa con cientos de imágenes.

```python
from traitly.fruit_phenotyping import FruitExternalAnalyzer

# Para analizar una imagen
analyzer = FruitExternalAnalyzer("ruta/a/imagen.jpg")

# Para analizar varias imágenes en carpeta
analyzer = FruitExternalAnalyzer("ruta/de/mi/carpeta/con/imagenes/")
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `image_path` | `str` | Ruta a la imagen o carpeta que quieres analizar |


!!! tip "Recomendación"
    Cuando tengas una carpeta con varias imágenes para analizar, te sugerimos:
    
    1. **Comienza con una imagen representativa** para ajustar los parámetros
    2. Experimenta con los métodos paso a paso hasta obtener buenos resultados
    3. Guarda la configuración ideal con `save_parameters()`
    4. Usa `analyze_folder(json_path="tu_archivo.json")` para procesar todo el lote automáticamente con los mismos parámetros
    
    Para ver ejemplos prácticos de este flujo de trabajo, consulta los [Tutoriales](tutorials/quickstart.md).


<br>

</div>

---

## Cómo se organiza el análisis

Cuando trabajas con `FruitExternalAnalyzer`, el análisis sigue este orden lógico:

```python
from traitly.fruit_phenotyping import FruitExternalAnalyzer

analyzer = FruitExternalAnalyzer('ruta/a/mi/imagen.jpg')

analyzer.load_image()                     # Cargar imagen
analyzer.setup_measurements()             # Configurar calibración y etiquetas
analyzer.generate_fruit_mask()            # Separar frutos del fondo
analyzer.detect_fruits()                  # Identificar frutos individuales
analyzer.analyze_morphology()             # Obtener medidas morfológicas
analyzer.analyze_color()                  # (Opcional) Obtener medidas de color

## Guardar resultados
analyzer.results.save_all()               # Guardar todos los resultados (CSV e imagen anotada)
analyzer.save_parameters()                # (Opcional) Guardar los parámetros usados en la sesión
```

Si trabajas con lotes de imágenes, no necesitas ejecutar estos pasos uno por uno, `analyze_folder()` lo hace todo automáticamente:

```python
# Analizar múltiples imágenes
analyzer = FruitExternalAnalyzer('ruta/a/mi/carpeta')
analyzer.analyze_folder(json_path='ruta/a/mi/parameters.json')
```


<br>

---

## Lo que puedes obtener del analizador

Después de ejecutar los métodos, el analizador guarda los resultados en atributos que puedes consultar:

| Atributo | Qué contiene |
|----------|--------------|
| `img_path` | Ruta de la imagen que estás analizando |
| `img`, `img_rgb`, `img_hsv` | La imagen en diferentes formatos de color |
| `mask_fruit` | Máscara donde los frutos aparecen en blanco y el fondo en negro |
| `contours` | Lista de contornos de todos los frutos detectados |
| `fruit_locule_map` | Mapeo de frutos (por compatibilidad mantiene el mismo nombre que en `FruitInternalAnalyzer`, pero aquí cada fruto se mapea a una lista vacía de lóculos) |
| `px_per_cm` | Factor de conversión de píxeles a centímetros (si calibraste) |
| `label_text` | Texto de la etiqueta detectada (si usaste detección) |
| `results` | Todos los resultados del análisis (tablas + imagen anotada) |
| `parameters` | Parámetros que usaste en esta sesión |

<br>

---

## Métodos:

!!! example ""
    Aquí explicamos cada método con ejemplos prácticos. Todos los parámetros tienen valores por defecto, así que puedes empezar con lo básico e ir ajustando según tus necesidades.



### `load_image`

Carga la imagen y prepara las representaciones internas (BGR, RGB, HSV).
Opcionalmente puede recortar una región de interés con `x`, `y`, `w`, `h`.

```python
analyzer.load_image(plot=True, plot_size=(5, 5))
analyzer.load_image(plot=True, show_axis=True, x=1500, y=0, w=2600, h=2700)
```

<br>

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `plot` | `bool` | `True` | Muestra la imagen cargada |
| `plot_size` | `tuple` | `(5, 5)` | Tamaño de la figura |
| `show_axis` | `bool` | `False` | Muestra los ejes en el plot |
| `x` | `int` | `None` | Coordenada izquierda del recorte |
| `y` | `int` | `None` | Coordenada superior del recorte |
| `w` | `int` | `None` | Ancho del recorte en píxeles |
| `h` | `int` | `None` | Alto del recorte en píxeles |

<br>

---

### `setup_measurements`

Realiza la detección de etiqueta y la referencia de tamaño y calcula el factor de escala píxel/cm.

??? note "Notas"

    - Cuando `detect_label=True`, la etiqueta se detecta en orden: primero QR y si no se encuentra, recurre a OCR. Para saltar la detección de QR e ir directo a OCR, activar `skip_qr=True`.

    - Cuando `fast_calibration=False` (default), la referencia de tamaño se detecta primero con YOLO y si falla, recurre a las medidas físicas de la imagen proporcionadas (`width_cm`, `length_cm`). Si no se encuentra referencia y `width_cm` y `length_cm` son `None`, los resultados se expresan en píxeles.
    
    - Para la detección de la referencia de tamaño, se asume que los círculos de la referencia son de color negro y que el fondo de la referencia es blanco.

    - Cuando se utiliza la referencia de tamaño para la calibración, el factor píxel/cm se calcula a partir del diámetro promedio de todos los círculos detectados. Por defecto, se descartan los círculos cuyo diámetro se desvía más de 2 desviaciones estándar respecto al promedio, con el fin de evitar sesgos en la estimación de la escala.

    - Cuando `detect_color_checker=True`, la carta de color se detecta usando el módulo MCC de OpenCV (cv2.mcc), compatible con tarjetas estándar de 24 colores (estilo Macbeth). La detección se realiza sobre una versión reducida de la imagen según `scale_factor`, lo que acelera el proceso pero puede afectar la precisión del área detectada para cada cuadro de color. Puedes revisar la detección a detalle con `plot_color_checker=True`.

```python
# Usando medidas físicas y detectando etiqueta
analyzer.setup_measurements(
    width_cm=29.7,
    length_cm=21.0,
    detect_label=True
)

# Usando referencia de tamaño y detectando etiqueta solo con OCR
analyzer.setup_measurements(
    diameter_cm=1.7,
    detect_label=True,
    skip_qr=True
)
```

<br>

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `detect_label` | `bool` | `False` | Si `True`, activa detección de etiqueta (QR -> OCR) |
| `width_cm` | `float` | `None` | Ancho conocido de la imagen en cm |
| `length_cm` | `float` | `None` | Largo conocido de la imagen en cm |
| `diameter_cm` | `float` | `None` | Diámetro conocido del círculo de referencia en cm; si no se proporciona, usa 2.5 cm por defecto |
| `fast_calibration` | `bool` | `False` | Si `True`, omite YOLO y calibra usando `width_cm` y `length_cm`; si no se proporcionan, los resultados se expresan en píxeles |
| `confidence` | `float` | `0.6` | Confianza mínima para detección YOLO de la referencia |
| `skip_qr` | `bool` | `False` | Si `True`, omite detección de QR e intenta OCR directamente |
| `gpu` | `bool` | `False` | Si `True`, usa GPU para OCR; solo compatible con NVIDIA. Si falla, continúa con CPU |
| `detect_color_checker` | `bool` | `False` | Si `True`, detecta carta de color (24 colores, estilo Macbeth) después de la calibración |
| `scale_factor` | `float` | `0.5` | Factor de reducción de imagen para detección de carta de color; debe estar entre 0.1 y 1.0 |
| `language_label` | `list` | `["es", "en"]` | Idiomas para OCR |
| `font_size` | `int` | `3` | Tamaño de fuente para anotaciones sobre los círculos de la referencia |
| `plot_reference` | `bool` | `False` | Si `True`, muestra recorte de la referencia de tamaño detectada y anotada |
| `plot_color_checker` | `bool` | `False` | Si `True`, muestra recorte de la tarjeta de color detectada y anotada |
| `plot_size` | `tuple` | `(5, 5)` | Tamaño de figura para los plots |
| `verbose` | `bool` | `True` | Si `True`, imprime resultados en consola |


<br>

---

### `generate_color_scatterplot`

*Opcional*

Muestra un scatterplot de los colores de los píxeles de la **imagen completa** (frutos, fondo, referencias, etcétera) en el espacio HSV. Es útil para seleccionar umbrales adecuados antes de crear la máscara (parámetros `lower_hsv` y `upper_hsv` en `generate_fruit_mask()`).

```python
analyzer.generate_color_scatterplot(sample_size=10000)
```
<br>

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `sample_size` | `int` | `10000` | Número de píxeles a muestrear para el plot |
| `plot_size` | `tuple` | `(18, 5)` | Tamaño de la figura |


<br>

---

### `generate_fruit_mask`

Genera una máscara binaria segmentando el fondo de la imagen en el espacio HSV y detectando todo lo que no corresponde al fondo (frutos, referencia de tamaño, etiqueta, etcétera).

Cuando `background_color=None`, el fondo se detecta automáticamente asumiendo que es de color azul. Para fondos de otro color, puedes indicarlo explícitamente con `'black'` o `'white'`, o definir los rangos manualmente con `lower_hsv` y `upper_hsv`. En la máscara resultante, el fondo se representa en negro (0) y los frutos en blanco (1).

Si se detectan regiones correspondientes a la referencia de tamaño, a la tarjeta de color o a la etiqueta en `setup_measurements()`, estas áreas se enmascaran en negro en la máscara final. No obstante, pueden permanecer contornos residuales, los cuales pueden ser descartados posteriormente durante el filtrado de contornos en `detect_fruits()`. Si dichas regiones no se detectan previamente, aparecerán como blanco en la máscara, al ser clasificadas como no-fondo.

```python
# Usando rangos HSV personalizados
analyzer.generate_fruit_mask(
    lower_hsv=[20, 30, 30],
    upper_hsv=[80, 255, 255]
)

# Usando rangos predefinidos
analyzer.generate_fruit_mask(background_color='white')
```

<br>

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `lower_hsv` | `list[int]` | `None` | Umbral HSV inferior `[H, S, V]` para seleccionar el color del fondo; si `None`, se aplica umbralización automática |
| `upper_hsv` | `list[int]` | `None` | Umbral HSV superior `[H, S, V]` para seleccionar el color del fondo; si `None`, se aplica umbralización automática |
| `background_color` | `str` | `None` | Opciones predefinidas: `'black'`, `'white'`, `'blue'`. Se utiliza para definir umbrales predeterminados de HSV del fondo |
| `n_iteration` | `int` | `1` | Número de iteraciones para las operaciones morfológicas (solo aplica si `kernel_open` y/o `kernel_close` están definidos) |
| `kernel_blur` | `int` | `None` | Tamaño de kernel Gaussian blur |
| `kernel_open` | `int` | `None` | Tamaño de kernel apertura morfológica |
| `kernel_close` | `int` | `None` | Tamaño de kernel cierre morfológico |
| `canny_min` | `int` | `None` | Umbral mínimo Canny |
| `canny_max` | `int` | `None` | Umbral máximo Canny |
| `remove_roi` | `bool` | `True` | Si `True`, elimina regiones de etiqueta, referencia y carta de color de la máscara |
| `roi_expansion` | `int` | `10` | Margen en píxeles alrededor de las ROIs antes de eliminarlas |
| `fill_holes` | `bool` | `False` | Si `True`, rellena huecos cerrados en la máscara binaria |
| `apply_convex_hull` | `bool` | `False` | Si `True`, aplica convex hull a los contornos externos del fruto |
| `erosion_px` | `int` | `3` | Radio en píxeles de la erosión elíptica en la máscara final |
| `stamp` | `bool` | `False` | Si `True`, invierte los colores de la imagen antes del enmascaramiento; asume un fondo original blanco |
| `plot` | `bool` | `True` | Muestra la máscara generada |
| `plot_size` | `tuple` | `(5, 5)` | Tamaño de la figura |


<br>

---

### `detect_fruits`

Detecta frutos individuales a partir de la máscara binaria generada por `generate_fruit_mask()`.

La detección se basa en contornos y en criterios morfológicos de **tamaño** y **forma** (área y circularidad), permitiendo filtrar objetos indeseados. A diferencia de `FruitInternalAnalyzer`, no se realiza detección ni mapeo de lóculos.

Como resultado, genera dos estructuras principales:

* `analyzer.contours`: lista de contornos de frutos detectados.
* `analyzer.fruit_locule_map`: diccionario que asocia cada fruto con una lista vacía de lóculos, manteniendo coherencia con el resto del pipeline.

??? note "Notas"
    - Cuando se trabaja con imágenes muy grandes, puede utilizarse `rescale_factor` para reducir temporalmente la escala durante la detección de contornos. Una vez finalizada, los contornos se re-escalan automáticamente al tamaño original. Esto puede mejorar el rendimiento computacional, aunque en imágenes con frutos muy pequeños o de baja calidad puede afectar la precisión de la detección.
    - Antes de continuar con el análisis, puedes observar rápidamente los contornos detectados con `plot=True`.

```python
analyzer.detect_fruits(
    min_fruit_circularity=0.5,
    min_fruit_area=500
)
```

<br>

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `min_fruit_circularity` | `float` | `0.5` | Circularidad mínima `[0, 1]` para aceptar un contorno como fruto |
| `min_fruit_area` | `int` | `500` | Área mínima (px) del fruto |
| `max_fruit_area` | `int` | `None` | Área máxima (px) del fruto; si `None`, no se aplica límite superior |
| `rescale_factor` | `float` | `None` | Factor para reescalar contornos antes de la detección |
| `verbose` | `bool` | `True` | Imprime un resumen de detección y parámetros utilizados |
| `plot` | `bool` | `False` | Muestra los contornos de frutos detectados sobre la imagen |
| `plot_size` | `tuple[int, int]` | `(5, 5)` | Tamaño de la figura (solo si `plot=True`) |
| `contour_color` | `tuple[int, int, int]` | `(0, 255, 0)` | Color BGR para dibujar los contornos detectados (solo si `plot=True`) |
| `contour_thickness` | `int` | `2` | Grosor de línea para dibujar los contornos (solo si `plot=True`) |

!!! warning "Importante"
    **Requiere** que exista una máscara (`generate_fruit_mask()`).

<br>

---

### `analyze_morphology`

Extrae métricas morfológicas de los frutos detectados a nivel de **fruto completo**, sin incluir métricas de lóculos, pericarpio interno ni simetría.

Los resultados se almacenan en `analyzer.results` como una instancia de `ResultsImage`. Esta clase contiene:

* `analyzer.results.morphology_results`: `pd.DataFrame` con las métricas morfológicas de cada fruto.
* `analyzer.results.annotated_img`: imagen anotada para inspección visual.

Además, `analyzer.results` incluye métodos para guardar los resultados:

```python
analyzer.results.save_all() # Guarda la imagen anotada y el archivo CSV
analyzer.results.save_csv() # Guarda únicamente el CSV
analyzer.results.save_img() # Guarda únicamente la imagen
```

Por defecto, los archivos se guardan en la misma carpeta que la imagen de entrada, utilizando como base el nombre del archivo original. El directorio de salida y un nombre base alternativo pueden especificarse mediante `output_dir='RUTA/'` y `base_name='nuevo_nombre'`. Para más detalles, consultar la documentación de la clase `ResultsImage`.

En la imagen anotada se indica un **ID único para cada fruto** y se resaltan los siguientes elementos:

* contorno del **fruto** (cian),
* **rectángulo del *bounding box***,
* **eje mayor** y **eje menor**.

??? note "Notas sobre modos de contorno"

    Para análisis de frutos con bordes muy irregulares, puede ser útil probar distintos `contour_mode` para suavizar el contorno:

    - **`'raw'`** (default): Usa el contorno original sin modificaciones. Es el más preciso pero también el más sensible a irregularidades del borde.
    - **`'hull'`**: Calcula el polígono convexo que envuelve el fruto, rellenando las entradas o hendiduras.
    - **`'approx'`**: Simplifica el contorno reduciendo el número de vértices, suavizando pequeñas irregularidades sin perder la forma general.
    - **`'ellipse'`**: Ajusta una elipse al contorno del fruto.
    - **`'circle'`**: Ajusta un círculo al contorno. Si se usa `'circle'`, la circularidad del fruto será `1` (círculo perfecto) para todos los frutos.

```python
analyzer.analyze_morphology(
    contour_mode="hull",
    label_position="bottom",
    label_color=(255, 255, 0)
)
```

<br>

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `contour_mode` | `str` | `'raw'` | Modo de contorno usado para las métricas: `'raw'`, `'hull'`, `'approx'`, `'ellipse'`, `'circle'` |
| `epsilon` | `float` | `0.001` | Factor de aproximación (solo si `contour_mode='approx'`) |
| `display_table` | `bool` | `True` | Si `True`, retorna el `DataFrame` con los resultados |
| `plot` | `bool` | `True` | Si `True`, muestra la imagen anotada |
| `plot_size` | `tuple[int, int]` | `(10, 10)` | Tamaño de la figura (solo si `plot=True`) |
| `font_size` | `float` | `1.5` | Tamaño del texto en la anotación |
| `font_thickness` | `int` | `2` | Grosor del texto en la anotación |
| `font_color` | `tuple[int,int,int]` | `(0, 0, 0)` | Color del texto (BGR) |
| `label_position` | `str` | `'top'` | Posición de la etiqueta (`'top'`, `'bottom'`, `'left'`, `'right'`) |
| `label_color` | `tuple[int,int,int]` | `(255, 255, 255)` | Color de fondo de la etiqueta (BGR) |
| `pericarp_ext_color` | `tuple[int,int,int]` | `(0, 240, 240)` | Color del contorno del fruto (BGR) |
| `pericarp_ext_thickness` | `int` | `2` | Grosor del contorno del fruto |

!!! warning "Importante"
    **Requiere** que exista una máscara (`generate_fruit_mask()`) y que `detect_fruits()` haya sido ejecutado.

<br>

---

### `analyze_color`

Extrae características de color del **pericarpio total** de los frutos detectados a partir de la imagen original y la máscara generada en el pipeline. A diferencia de `FruitInternalAnalyzer`, no se segmentan tejidos internos; el color se extrae únicamente sobre la región completa del fruto.

La extracción de color siempre utiliza los contornos originales en modo `'raw'`, independientemente del `contour_mode` seleccionado en `analyze_morphology()`. Esto garantiza que el área de extracción de color corresponda fielmente a la región segmentada en la máscara, sin verse afectada por simplificaciones geométricas del contorno.

Los resultados se almacenan en `analyzer.results` como una instancia de `ResultsImage`. Esta clase contiene:

* `analyzer.results.color_results`: `pd.DataFrame` con las métricas de color de cada fruto.
* `analyzer.results.annotated_img`: imagen anotada para inspección visual.

Además, `analyzer.results` incluye métodos para guardar los resultados:

```python
analyzer.results.save_all() # Guarda la imagen anotada y el archivo CSV
analyzer.results.save_csv() # Guarda únicamente el CSV
analyzer.results.save_img() # Guarda únicamente la imagen
```

Por defecto, los archivos se guardan en la misma carpeta que la imagen de entrada. El directorio de salida y nombre base pueden especificarse mediante `output_dir='RUTA/'` y `base_name='nuevo_nombre'`.

??? note "Notas"
    * `analyze_color()` es **independiente** de `analyze_morphology()`. Si se ejecuta únicamente `analyze_color()`, se genera una imagen anotada básica con el **ID del fruto** y el **contorno del fruto** en verde.
    * Si `analyze_morphology()` fue ejecutada previamente, al guardar resultados se **reutiliza** la imagen anotada de morfología, ya que contiene una anotación más completa.
    * Por defecto, la función calcula un estadístico resumen (`'mean'` o `'median'`) por canal. Alternativamente, puede calcular histogramas de color por píxel activando `get_color_histogram=True`, lo cual devuelve distribuciones completas por canal en lugar de un solo valor resumen.

```python
df = analyzer.analyze_color(
    stat='median',
    color_space='hsv, lab',
    plot=False
)
```

<br>

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `stat` | `str` | `'mean'` | Estadístico: `'mean'` o `'median'` (ignorado si `get_color_histogram=True`) |
| `color_space` | `str` | `'all'` | Espacios: `'all'`, `'rgb'`, `'lab'`, `'hsv'`, `'gray'` |
| `display_table` | `bool` | `True` | Si `True`, retorna el `DataFrame` con resultados |
| `plot` | `bool` | `False` | Si `True`, muestra la imagen anotada usada para la extracción de color |
| `plot_size` | `tuple[int, int]` | `(10, 10)` | Tamaño de la figura (solo si `plot=True`) |
| `font_size` | `int` | `2` | Tamaño del texto en la anotación |
| `font_thickness` | `int` | `2` | Grosor del texto en la anotación |
| `font_color` | `tuple[int,int,int]` | `(0, 0, 0)` | Color del texto (BGR) |
| `label_position` | `str` | `'top'` | Posición de la etiqueta (`'top'`, `'bottom'`, `'left'`, `'right'`) |
| `label_color` | `tuple[int,int,int]` | `(255, 255, 255)` | Color de fondo de la etiqueta (BGR) |
| `pericarp_ext_color` | `tuple[int,int,int]` | `(0, 255, 0)` | Color del contorno del fruto (BGR) |
| `pericarp_ext_thickness` | `int` | `2` | Grosor del contorno del fruto |
| `label_opacity` | `float` | `0.7` | Opacidad del fondo de la etiqueta `[0, 1]` |
| `get_color_histogram` | `bool` | `False` | Si `True`, retorna histogramas por píxel en lugar de estadísticos resumen |

!!! warning "Importante"
    **Requiere** que exista una máscara (`generate_fruit_mask()`) y que `detect_fruits()` haya sido ejecutada.

<br>

---

### `generate_single_fruit_masks`

*Opcional*

Genera y visualiza la máscara de fruto completo para un fruto específico, útil para inspeccionar en detalle los resultados de la segmentación antes de ejecutar `analyze_color()`.

El fruto se recorta a su *bounding box* con un margen opcional. El parámetro `fruit_id` corresponde al identificador del fruto en la imagen anotada o la tabla de resultados, tal como aparece en las salidas generadas por `analyze_morphology()` o `analyze_color()`.

```python
analyzer.generate_single_fruit_masks(fruit_id=3)
```

<br>

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `fruit_id` | `int` | `None` | ID del fruto a visualizar; si `None`, usa el primer fruto detectado |
| `plot_size` | `tuple[int, int]` | `(7, 5)` | Tamaño de la figura |
| `margin` | `int` | `5` | Margen (px) alrededor del recorte del fruto |

!!! warning "Importante"
    **Requiere** que exista una máscara y que `detect_fruits()` haya sido ejecutado.

<br>

---

### `save_parameters`

*Opcional*

Exporta los **parámetros de análisis de la sesión actual** en formato `.txt` y `.json`, listos para su inspección, reutilización y reproducibilidad.

Los parámetros almacenados en `analyzer.parameters` se exportan usando como nombre base el de la imagen cargada, generando automáticamente dos archivos:

* `<nombre_imagen>_parameters.txt`: versión legible para inspección humana.
* `<nombre_imagen>_parameters.json`: versión estructurada para uso programático.

Ambos se guardan por defecto en la misma carpeta que la imagen de entrada, o en el directorio indicado por `output_path`. Son especialmente útiles para:

* reutilizar configuraciones en análisis por lote con `analyze_folder()`,
* ejecutar análisis reproducibles desde la terminal con Traitly,
* archivar y compartir pipelines de análisis.

??? note "Notas"
    * Solo se exportan los parámetros de las funciones ejecutadas durante la sesión (segmentación, detección, morfología, color).
    * No retorna ningún valor; imprime en la consola las rutas de los archivos generados.

```python
analyzer.save_parameters()
```

<br>

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `output_path` | `str` | `None` | Directorio de salida. Si `None`, se usa el mismo directorio de la imagen de entrada. |

<br>

---

### `plot_image`

*Opcional*

Muestra la imagen original o la imagen **anotada con resultados**, según el valor de `annotated`, reutilizando las imágenes ya almacenadas en memoria sin recargarlas ni regenerarlas.

* Cuando `annotated=False`, se muestra la **imagen original** cargada.
* Cuando `annotated=True`, se muestra la **imagen anotada** generada durante `analyze_morphology()` o `analyze_color()`.

```python
analyzer.plot_image(annotated=True)
```

<br>

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `annotated` | `bool` | `True` | Si `True`, muestra la imagen anotada; si `False`, muestra la imagen original |
| `plot_size` | `tuple[int, int]` | `(10, 10)` | Tamaño de la figura |

<br>

---

### `analyze_folder`

Procesa por lotes todas las imágenes de la carpeta indicada al inicializar `FruitExternalAnalyzer`, ejecutando el pipeline completo sobre cada imagen de forma secuencial (`num_cores=1`) o en paralelo (`num_cores` > 1). Por defecto se ejecutan tanto el análisis morfológico como el de color; cada uno puede desactivarse de forma independiente con `analyze_morphology=False` o `analyze_color=False`.

Por cada imagen analizada se genera una **imagen anotada** con los identificadores y anotaciones visuales del análisis. Los resultados de todas las imágenes se consolidan en un único archivo CSV por tipo de análisis:

* `morphology_results.csv`: métricas morfológicas de todos los frutos detectados.
* `color_results.csv`: métricas de color de todos los frutos detectados.

Adicionalmente, se genera siempre un `session_report.txt` con un resumen de la sesión (imágenes procesadas, frutos detectados, tiempos, parámetros utilizados y dependencias). Si alguna imagen falla durante el procesamiento, se genera también un `error_report.txt` detallando qué ocurrió en cada caso.

Todos los archivos se guardan en el directorio indicado por `output_path`, o en una subcarpeta `Results/` dentro de la carpeta de entrada si no se especifica.

??? note "Nota"
    Esta función acepta individualmente todos los parámetros de los pasos del pipeline. Sin embargo, para mayor practicidad y reproducibilidad, se recomienda explorar y estandarizar los parámetros sobre una imagen representativa con `save_parameters()` y luego pasar el archivo `.json` generado mediante el parámetro `json_path`.

```python
# Usando parámetros individuales
analyzer.analyze_folder(
    lower_hsv=[0, 0, 0],
    upper_hsv=[180, 80, 80],
    min_fruit_area=500,
    analyze_color=True
)

# Usando archivo de parámetros guardado
analyzer.analyze_folder(json_path="imagen_parameters.json")
```

<br>

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `analyze_morphology` | `bool` | `True` | Si `True`, ejecuta análisis morfológico sobre cada imagen |
| `analyze_color` | `bool` | `True` | Si `True`, ejecuta análisis de color sobre cada imagen |
| `json_path` | `str` | `None` | Ruta a un archivo `.json` de parámetros generado por `save_parameters()` |
| `config` | `dict` | `None` | Configuración base como diccionario; los parámetros individuales tienen prioridad |
| `output_path` | `str` | `None` | Directorio de salida. Si `None`, se crea una subcarpeta `Results/` dentro de la carpeta de entrada |
| `num_cores` | `int` | `1` | Número de procesos en paralelo. Se limita automáticamente a los núcleos disponibles |
| `verbose` | `bool` | `True` | Si `True`, imprime progreso y resumen de la sesión |
| `width_cm` | `float` | `None` | Ancho conocido de la imagen en cm -> `setup_measurements` |
| `length_cm` | `float` | `None` | Largo conocido de la imagen en cm -> `setup_measurements` |
| `diameter_cm` | `float` | `None` | Diámetro conocido de la referencia en cm -> `setup_measurements` |
| `fast_calibration` | `bool` | `None` | Si `True`, omite YOLO y calibra con dimensiones físicas -> `setup_measurements` |
| `skip_qr` | `bool` | `None` | Si `True`, omite detección de QR -> `setup_measurements` |
| `detect_label` | `bool` | `None` | Si `True`, activa detección de etiqueta con OCR -> `setup_measurements` |
| `confidence` | `float` | `None` | Confianza mínima para detección YOLO -> `setup_measurements` |
| `detect_color_checker` | `bool` | `None` | Si `True`, detecta y elimina carta de color -> `setup_measurements` |
| `scale_factor` | `float` | `None` | Factor de reducción para detección de carta de color -> `setup_measurements` |
| `lower_hsv` | `list[int]` | `None` | Umbral HSV inferior para segmentación -> `generate_fruit_mask` |
| `upper_hsv` | `list[int]` | `None` | Umbral HSV superior para segmentación -> `generate_fruit_mask` |
| `background_color` | `str` | `None` | Color de fondo predefinido -> `generate_fruit_mask` |
| `n_iteration` | `int` | `None` | Iteraciones de operaciones morfológicas -> `generate_fruit_mask` |
| `kernel_blur` | `int` | `None` | Tamaño de kernel Gaussian blur -> `generate_fruit_mask` |
| `kernel_open` | `int` | `None` | Tamaño de kernel apertura morfológica -> `generate_fruit_mask` |
| `kernel_close` | `int` | `None` | Tamaño de kernel cierre morfológico -> `generate_fruit_mask` |
| `canny_min` | `int` | `None` | Umbral mínimo Canny -> `generate_fruit_mask` |
| `canny_max` | `int` | `None` | Umbral máximo Canny -> `generate_fruit_mask` |
| `fill_holes` | `bool` | `None` | Si `True`, rellena huecos en la máscara -> `generate_fruit_mask` |
| `apply_convex_hull` | `bool` | `None` | Si `True`, aplica convex hull a cada fruto -> `generate_fruit_mask` |
| `remove_roi` | `bool` | `None` | Si `True`, elimina regiones de referencia y etiqueta -> `generate_fruit_mask` |
| `roi_expansion` | `int` | `None` | Margen en píxeles alrededor de las ROIs -> `generate_fruit_mask` |
| `stamp` | `bool` | `None` | Si `True`, invierte colores antes del enmascaramiento -> `generate_fruit_mask` |
| `min_fruit_area` | `int` | `None` | Área mínima para aceptar un contorno como fruto -> `detect_fruits` |
| `max_fruit_area` | `int` | `None` | Área máxima para aceptar un contorno como fruto -> `detect_fruits` |
| `min_fruit_circularity` | `float` | `None` | Circularidad mínima para aceptar un fruto -> `detect_fruits` |
| `rescale_factor` | `float` | `None` | Factor de reescalado de contornos -> `detect_fruits` |
| `contour_mode` | `str` | `None` | Modo de contorno para métricas morfológicas -> `analyze_morphology` |
| `epsilon` | `float` | `None` | Factor de aproximación de contorno -> `analyze_morphology` |
| `stat` | `str` | `None` | Estadístico de color: `'mean'` o `'median'` -> `analyze_color` |
| `color_space` | `str` | `None` | Espacios de color a extraer -> `analyze_color` |
| `get_color_histogram` | `bool` | `None` | Si `True`, calcula histogramas por píxel -> `analyze_color` |

!!! warning "Importante"
    **Requiere** que `FruitExternalAnalyzer()` haya sido inicializado con una ruta de carpeta, no de archivo.