<div class="animate" markdown>

# Mediciones 

Esta sección lista todos los traits que devuelven `FruitInternalAnalyzer` y `FruitExternalAnalyzer`. Los resultados se almacenan en dos DataFrames separados: uno para **morfología** y otro para **color**.

---

## Traits de morfología

Obtenidos por `analyze_morphology()` y almacenados en `results.morphology_results`.

Los nombres de columna que incluyen un sufijo de unidad de medición (p. ej. `fruit_area_cm2`) reflejarán la unidad utilizada:

- `cm` o `cm2` cuando se detecta o proporciona una referencia de tamaño
- `px` o `px2` cuando no hay calibración disponible

!!! info "Regiones de los tejido"
    Para ejemplos visuales de las regiones de los tejidos mencionados en esta sección, consulta [Regiones de los tejidos y extracción de color](#regiones-de-tejido-y-extraccion-de-color).

### Metadatos de la imagen

| Columna | Descripción | Interna | Externa |
|---------|-------------|:-------:|:-------:|
| `image_name` | Nombre de la imagen | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-check:{ .icon-green } |
| `label` | Texto de la etiqueta detectada por QR u OCR | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-check:{ .icon-green } |
| `fruit_id` | ID secuencial del fruto | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-check:{ .icon-green } |
| `n_locules` | Número de lóculos detectados en el fruto | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |
| `unit` | Unidad de medición usada: `cm` o `px` | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-check:{ .icon-green } |

### Morfología del fruto

| Columna | Descripción | Interna | Externa |
|---------|-------------|:-------:|:-------:|
| `fruit_area_cm2` / `fruit_area_px2` | Área total del contorno del fruto (medida directa del tamaño del fruto) | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-check:{ .icon-green } |
| `fruit_perimeter_cm` / `fruit_perimeter_px` | Perímetro del contorno del fruto (perímetros más largos indican formas más irregulares) | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-check:{ .icon-green } |
| `fruit_circularity` | `4π·área / perímetro²` -> [0, 1]. Mide qué tan cercano es el fruto a un círculo perfecto; valores cercanos a 1 indican frutos redondos | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-check:{ .icon-green } |
| `fruit_solidity` | `área / área_convexa` -> [0, 1]. Mide qué tan bien el área del fruto llena su área convexa envolvente; valores bajos indican frutos cóncavos o irregulares | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-check:{ .icon-green } |
| `fruit_convexity` | `perímetro_convexo / perímetro_contorno` -> [0, 1]. Mide qué tan suave es el borde del fruto respecto a su área convexa envolvente; valores bajos indican superficies rugosas o lobuladas | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-check:{ .icon-green } |
| `fruit_major_axis_cm` / `fruit_major_axis_px` | Distancia máxima en línea recta entre dos puntos del contorno del fruto (útil para estimar la longitud del fruto) | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-check:{ .icon-green } |
| `fruit_minor_axis_cm` / `fruit_minor_axis_px` | Ancho máximo del fruto medido perpendicularmente al eje mayor (útil para estimar el ancho del fruto) | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-check:{ .icon-green } |
| `fruit_box_length_cm` / `fruit_box_length_px` | Lado más largo de la caja delimitadora (estimación alternativa de la longitud del fruto) | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-check:{ .icon-green } |
| `fruit_box_width_cm` / `fruit_box_width_px` | Lado más corto de la caja delimitadora (estimación alternativa del ancho del fruto) | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-check:{ .icon-green } |
| `fruit_aspect_ratio` | `ancho_caja / largo_caja` -> [0, 1]. Valores cercanos a 1 indican frutos redondos; valores más bajos indican frutos alargados | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-check:{ .icon-green } |
| `fruit_compactness` | `área_fruto / área_caja` -> [0, 1]. Qué tan eficientemente rellena el fruto su caja delimitadora; valores más altos indican formas más compactas | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-check:{ .icon-green } |
| `fruit_lobedness_cm` / `fruit_lobedness_px` | Desviación estándar de las distancias radiales desde el centroide del fruto hasta el contorno externo del fruto. Proxy de irregularidad superficial: valores más altos indican una superficie más lobulada o irregular | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-check:{ .icon-green } |

### Pericarpio

| Columna | Descripción | Interna | Externa |
|---------|-------------|:-------:|:-------:|
| `total_outer_pericarp_area_cm2` / `total_outer_pericarp_area_px2` | Área total de la región del pericarpio externo (`área total del fruto` – `área interna del fruto`) | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |
| `outer_pericarp_mean_thickness_cm` / `outer_pericarp_mean_thickness_px` | Grosor promedio de la pared del pericarpio, estimado como la distancia desde el contorno externo del fruto hasta el límite de la cavidad interna a lo largo de rayos radiales emitidos desde el centroide del fruto | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |
| `outer_pericarp_std_thickness_cm` / `outer_pericarp_std_thickness_px` | Desviación estándar del grosor del pericarpio entre todos los rayos (indica qué tan uniforme es la pared) | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |
| `outer_pericarp_cv_thickness` | Coeficiente de variación del grosor del pericarpio (%) (permite comparar la uniformidad de la pared entre frutos de distintos tamaños) | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |

### Áreas internas

| Columna | Descripción | Interna | Externa |
|---------|-------------|:-------:|:-------:|
| `total_internal_fruit_area_cm2` / `total_internal_fruit_area_px2` | Área interna total del fruto (`pericarpio interno` + `lóculos`) | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |
| `total_internal_pericarp_area_cm2` / `total_internal_pericarp_area_px2` | Área del tejido del pericarpio interno (`área interna del fruto` – `área de lóculos`) (refleja el tamaño del tejido alrededor de los lóculos) | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |
| `total_locules_area_cm2` / `total_locules_area_px2` | Área total de todos los lóculos detectados | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |

### Lóculos

| Columna | Descripción | Interna | Externa |
|---------|-------------|:-------:|:-------:|
| `locules_mean_area_cm2` / `locules_mean_area_px2` | Área media por lóculo individual | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |
| `locules_std_area_cm2` / `locules_std_area_px2` | Desviación estándar del área de los lóculos (indica variabilidad de tamaño entre lóculos) | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |
| `locules_cv_area` | Coeficiente de variación del área de los lóculos (%) (permite comparar la homogeneidad de tamaño entre frutos de distintas dimensiones) | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |
| `locules_mean_circularity` | Circularidad media de todos los lóculos (indica qué tan redondos son los compartimentos) | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |
| `locules_std_circularity` | Desviación estándar de la circularidad de los lóculos | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |
| `locules_cv_circularity` | Coeficiente de variación de la circularidad de los lóculos (%) | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |
| `locules_angular_symmetry` | Error angular promedio entre las posiciones observadas de los lóculos y una distribución ideal equiespaciada (valores más bajos indican mayor simetría angular) | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |
| `locules_radial_symmetry` | Coeficiente de variación de las distancias radiales de los lóculos desde el centroide del fruto (%) (valores más bajos indican mayor uniformidad radial) | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |

### Radios

Todos los radios entre areas de tejidos son adimensionales, por lo que permiten comparar frutos de distintos tamaños.

| Columna | Numerador | Denominador | Interna | Externa |
|---------|-----------|-------------|:-------:|:-------:|
| `outer_pericarp_to_fruit_ratio` | Área del pericarpio externo | Área total del fruto | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |
| `internal_pericarp_to_fruit_ratio` | Área del pericarpio interno | Área total del fruto | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |
| `locules_to_fruit_ratio` | Área total de lóculos | Área total del fruto | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |
| `locules_to_total_internal_ratio` | Área total de lóculos | Área interna total del fruto | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |
| `internal_pericarp_to_total_internal_ratio` | Área del pericarpio interno | Área interna total del fruto | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |

---

## Traits de color

Obtenidos con `analyze_color()` y almacenados en `results.color_results`.

Por defecto, Traitly extrae la **media** (o la **mediana**, opcionalmente) de cada canal de color sobre todos los píxeles de la región de interés. Además, para cada canal de color se reporta su desviación estándar y su coeficiente de variación.  

Sí `get_color_histogram=True`, Traitly devuelve el conteo de píxeles por bin de intensidad para cada canal, donde cada bin corresponde a un valor de intensidad. Por ejemplo, `R_128` contiene el número de píxeles con un valor de rojo igual a 128. Con `normalize=True`, cada conteo se divide entre el total de píxeles válidos, devolviendo proporciones en lugar de conteos absolutos.

### Opciones de tejido

El color puede extraerse de forma independiente para diferentes regiones del fruto. En `FruitExternalAnalyzer`, debido a que los frutos no presentan cavidades internas, por defecto solo `total_pericarp` está disponible y la columna `tissue` no se incluye en los resultados.

| Tejido | Descripción | Interna | Externa |
|--------|-------------|:-------:|:-------:|
| `total_pericarp` | Área total del fruto, excluyendo lóculos | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-check:{ .icon-green } |
| `outer_pericarp` | Pared del pericarpio externo (área total del fruto menos región interna) | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |
| `internal_pericarp` | Tejido del pericarpio interno entre la pared externa y los lóculos | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |
| `locules` | Regiones de lóculos únicamente | :fontawesome-solid-check:{ .icon-green } | :fontawesome-solid-minus:{ .icon-red } |

### Estructura de la tabla

Para `FruitInternalAnalyzer`, cada fila representa un tejido de un fruto:

| `image_name` | `label` | `fruit_id` | `tissue` | `R_mean` | `G_mean` | … |
---------------|---------|------------|----------|----------|----------|---|
| img_01.jpg | TOM-001 | 1 | total_pericarp | 185.3 | 52.1 | … |
| img_01.jpg | TOM-001 | 1 | outer_pericarp | 190.7 | 48.6 | … |

Para `FruitExternalAnalyzer`, la columna `tissue` se omite, por lo que tendremos solo una fila por fruto:

| `image_name` | `label` | `fruit_id` | `R_mean` | `G_mean` | … |
|--------------|---------|------------|----------|----------|---|
| img_01.jpg | TOM-001 | 1 | 185.3 | 52.1 | … |
| img_01.jpg | TOM-001 | 2 | 190.7 | 48.6 | … |

### Canales de color

### Canales de color

| Columna | Espacio de color | Rango | Descripción |
|---------|-----------------|-------|-------------|
| `R_mean` / `R_median` | RGB | 0–255 | Canal rojo |
| `R_std` | RGB | ≥ 0 | Desviación estándar del canal rojo |
| `G_mean` / `G_median` | RGB | 0–255 | Canal verde |
| `G_std` | RGB | ≥ 0 | Desviación estándar del canal verde |
| `B_mean` / `B_median` | RGB | 0–255 | Canal azul |
| `B_std` | RGB | ≥ 0 | Desviación estándar del canal azul |
| `L_mean` / `L_median` | L\*a\*b\* | 0–100 | Luminosidad (perceptualmente uniforme, independiente del tono) |
| `L_std` | L\*a\*b\* | ≥ 0 | Desviación estándar de la luminosidad |
| `a_mean` / `a_median` | L\*a\*b\* | –128 a +127 | Eje verde–rojo (valores positivos indican tonos rojos, negativos indican verde) |
| `a_std` | L\*a\*b\* | ≥ 0 | Desviación estándar del eje verde–rojo |
| `b_mean` / `b_median` | L\*a\*b\* | –128 a +127 | Eje azul–amarillo (valores positivos indican amarillo, negativos indican azul) |
| `b_std` | L\*a\*b\* | ≥ 0 | Desviación estándar del eje azul–amarillo |
| `H_mean` / `H_median` | HSV | 0–360° | Tono (media circular) (representa el color dominante; p. ej., ~0/360°=rojo, ~120°=verde, ~240°=azul) |
| `H_std` | HSV | ≥ 0° | Desviación estándar circular del tono (indica qué tan variable es el tono dentro de la región) |
| `S_mean` / `S_median` | HSV | 0–100 | Saturación — qué tan vibrante o puro es el color |
| `S_std` | HSV | ≥ 0 | Desviación estándar de la saturación |
| `V_mean` / `V_median` | HSV | 0–100 | Valor (brillo) |
| `V_std` | HSV | ≥ 0 | Desviación estándar del valor |
| `Gray_mean` / `Gray_median` | Escala de grises | 0–255 | Intensidad media de píxeles — medida simple de luminancia |
| `Gray_std` | Escala de grises | ≥ 0 | Desviación estándar de la intensidad de píxeles |

!!! tip "¿Por qué usar estadísticas circulares para el tono?"
    El tono es una variable circular: 0° y 360° representan el mismo color (rojo). Usar media y desviación estándar convencionales en valores de tono cercanos a 0°/360° produciría resultados incorrectos (p. ej., una "media" cercana a 180°). Traitly usa estadísticas circulares para manejar correctamente esta periodicidad.

---

## Notas

### ¿Por qué incluir el CV junto con la media y la desviación estándar?

La desviación estándar (DE) se calcula respecto a la media de cada fruto. Esto significa que el mismo valor de DE tiene un impacto diferente dependiendo de la escala de la medición. Por ejemplo, una DE de 5 px en un fruto con un grosor promedio del pericarpio de 100 px representa mucha menos variación que la misma DE en un fruto con una media de 20 px. El **coeficiente de variación** (`CV = DE / media × 100`) corrige esto expresando la variabilidad como porcentaje de la media, lo que permite comparar la homogeneidad entre frutos de distintos tamaños.

### Grosor del pericarpio y lobedness

Tanto `outer_pericarp_mean_thickness` como `fruit_lobedness` se estiman mediante rayos radiales emitidos desde el centroide del fruto hacia el contorno del fruto. La siguiente imagen muestra este método sobre una sección transversal real.

<figure style="text-align: center; margin: 0 auto;">
  <img src="../../../assets/images/radial_rays.png" alt="Rayos radiales para grosor de pericarpio y lobedness"
       style="height: 250px; width: auto;">
  <figcaption><em>Rayos radiales emitidos desde el centroide del fruto (punto cyan) hasta el contorno externo del fruto (verde) y el límite de su región interna (magenta).</em></figcaption>
</figure>

- **`outer_pericarp_mean_thickness`**: media de las longitudes de los segmentos de rayo entre el contorno externo del fruto y el contorno de la región interna. La desviación estándar y el coeficiente de variación de esas longitudes describen qué tan uniforme es la pared alrededor del fruto.
- **`fruit_lobedness`**: desviación estándar de las longitudes completas de los rayos desde el centroide del fruto hasta su contorno externo. Un fruto perfectamente redondo tendrá longitudes casi idénticas en todas las direcciones, lo que da una DE baja. Un fruto lobulado o irregular tendrá más variación, resultando en un valor más alto.

### Interpretación de la simetría

Para `locules_angular_symmetry` y `locules_radial_symmetry`, **valores más bajos indican mayor simetría**. Ambas métricas solo son significativas cuando `n_locules ≥ 2`.

Cada lóculo se describe con dos coordenadas polares relativas al centroide del fruto: su **posición angular (θ)** y su **distancia radial (r)**.

<figure style="text-align: center; margin: 0 auto;">
  <img src="../../../assets/images/symmetry_diagram.png" alt="Coordenadas polares de los lóculos"
       style="height: 350px; width: auto;">
  <figcaption><em>Centroide de los lóculos (círculos verdes), descritos por sus ángulos θ y sus distancias radiales r desde el centroide del fruto (círculo azul).</em></figcaption>
</figure>

- **`locules_angular_symmetry`**: mide qué tan uniformemente están distribuidos los lóculos alrededor del centro del fruto. Para un fruto con *n* lóculos, una simetría angular perfecta los distribuiría exactamente a 360°/n entre sí. Es la desviación absoluta promedio entre los ángulos observados y esa distribución ideal. Un valor cercano a 0 indica lóculos equiespaciados; valores más altos indican una distribución angular desigual.

- **`locules_radial_symmetry`**: mide qué tan similares son las distancias radiales entre lóculos. Es el coeficiente de variación (%) de todos los valores *r*. Un valor cercano a 0 indica que todos los lóculos están aproximadamente a la misma distancia del centro; valores más altos indican que algunos lóculos están más cerca del centro que otros.

<figure style="text-align: center; margin: 0 auto;">
  <img src="../../../assets/images/symmetry_examples.png" alt="Ejemplos de simetría"
       style="width: 100%; max-width: 1200px;">
  <figcaption><em>Ejemplos de resultados de simetría angular y radial.</em></figcaption>
</figure>

### Regiones de los tejido y extracción de color { #regiones-de-tejido-y-extraccion-de-color }

Las siguientes imágenes ilustran cómo Traitly segmenta cada región de los tejidos del fruto y qué información de color se extraen de ellas.

Las máscaras binarias muestran exactamente qué píxeles se incluyen por tejido. Los paneles grises indican tejidos que no se seleccionaron para el análisis de color debido a su redundancia. Estas máscaras corresponden a las regiones segmentadas tanto en `analyze_morphology()` como en `analyze_color()`: `total_pericarp`, `outer_pericarp`, `internal_pericarp` y `locules`.

En `analyze_morphology()`, `total_internal_fruit` se refiere al área combinada de `internal_pericarp` + `locules`, y `total_fruit_area` se refiere a `total_pericarp` + `locules`.

<figure style="text-align: center; margin: 0 auto;">
  <img src="../../../assets/images/tissue_masks.png" alt="Máscaras binarias por región de tejido"
       style="width: 100%; max-width: 700px;">
  <figcaption><em>Máscaras binarias para cada región de los tejidos de frutos de distintas especies.</em></figcaption>
</figure>

`analyze_color()` puede extraer estadísticas de color de forma independiente para cada una de estas regiones, como se muestra a continuación. Nótese que los lóculos del arándano aparecen casi negros ya que son cavidades vacías, por lo que el color extraído refleja el fondo oscuro y no el tejido del fruto.

<figure style="text-align: center; margin: 0 auto;">
  <img src="../../../assets/images/tissue_colors.png" alt="Color extraído por tejido"
       style="width: 100%; max-width: 650px;">
  <figcaption><em>Color RGB promedio extraído de cada región de los tejidos de tomate y arándano.</em></figcaption>
</figure>

Distintas especies de frutos tienen estructuras internas muy diferentes, por lo que a veces no todas las regiones extraídas por Traitly son relevantes. En esos casos, `analyze_color()` te permite seleccionar solo las regiones que tienen sentido para tus frutos, como se muestra en la siguiente imagen.

<figure style="text-align: center; margin: 0 auto;">
  <img src="../../../assets/images/tissue_color_examples.png" alt="Ejemplos de extracción de color por especie"
       style="width: 100%; max-width: 400px;">
  <figcaption><em>Extracción de color para las áreas de pericarpio total y lóculos en distintas especies.</em></figcaption>
</figure>

</div>