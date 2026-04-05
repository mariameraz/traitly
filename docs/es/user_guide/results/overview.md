<div class="animate" markdown>

# Resultados ⊹ ࣪ ˖

Cada análisis en Traitly genera hasta cuatro tipos de salida: dos tablas de resultados (morfología y color), imágenes anotadas, un reporte de sesión y un reporte de errores. Esta sección describe cada uno.

---

## Tablas de resultados

Traitly devuelve las mediciones como [pandas DataFrames](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html), accesibles a través del objeto `results` después de ejecutar `analyze_morphology()` y/o `analyze_color()`:

```python
analyzer.results.morphology_results  # DataFrame de morfología
analyzer.results.color_results       # DataFrame de color
```

Al usar `analyze_folder()` para procesamiento por lotes, ambas tablas se guardan automáticamente como archivos CSV en la carpeta de salida. `analyze_folder()` también guarda un **`session_report.txt`** junto con los CSV. Este incluye la versión de Traitly y sus dependencias, la fecha y hora del análisis, la carpeta de entrada y todos los parámetros utilizados en cada paso del procesamiento. Si alguna imagen no pudo procesarse, se genera adicionalmente un **`error_report.txt`** con la lista de archivos fallidos y el motivo del error:

```
Results/
├── morphology_results.csv
├── color_results.csv
├── session_report.txt
└── error_report.txt      <- solo si ocurrieron errores
```

Al correr Traitly desde la CLI se genera la misma estructura de archivos que con `analyze_folder()`.

!!! example ""
    La descripción completa de todas las columnas disponibles se encuentra en la sección de [Mediciones](measurements.md).

---

## Imágenes anotadas

### Anotación de frutos

Por cada imagen procesada, Traitly guarda una versión anotada que muestra los contornos detectados y los IDs de los frutos directamente sobre la imagen original. Esto es útil para verificar visualmente que la detección y segmentación funcionaron correctamente antes de interpretar los datos.

```
Results/
└── nombre_imagen_processed.jpg
```

<div style="display: flex; gap: 16px; justify-content: center; align-items: flex-start;">
  <figure style="text-align: center; margin: 0;">
    <img src="../../../../assets/images/ext_annotation.png" alt="Ejemplo de imagen anotada — análisis externo"
         style="height: 300px; width: auto;">
    <figcaption><em>Análisis externo: contorno del fruto, ejes, caja delimitadora y etiqueta con ID.</em></figcaption>
  </figure>
  <figure style="text-align: center; margin: 0;">
    <img src="../../../../assets/images/int_annotation.png" alt="Ejemplo de imagen anotada — análisis interno"
         style="height: 300px; width: auto;">
    <figcaption><em>Análisis interno: contornos del fruto y lóculos, límite de la cavidad interna, ejes, caja delimitadora, centroides y etiqueta con ID y número de lóculos.</em></figcaption>
  </figure>
</div>

Las anotaciones varían según el analizador (clase) utilizado y los pasos ejecutados:

`FruitExternalAnalyzer`:

- Etiqueta con el ID del fruto (`id 1`)
- Contorno del fruto en **amarillo**
- Eje mayor en **verde**, eje menor en **azul**
- Caja delimitadora en **azul claro**

`FruitInternalAnalyzer`:

- Etiqueta con ID y número de lóculos (`id 5: 4 loc`)
- Contorno externo del fruto en **verde**
- Límite de la región interna del fruto en **amarillo**
- Contornos de los lóculos en **magenta**
- Eje mayor en **verde**, eje menor en **azul**
- Caja delimitadora en **azul claro**
- Centroide del fruto como punto **cyan**
- Centroides de los lóculos como puntos **amarillos**

!!! note "Análisis solo de color"
    Si `analyze_morphology()` no se ejecuta y solo se corre `analyze_color()` con `FruitInternalAnalyzer`, la imagen anotada mostrará una versión simplificada: contorno del fruto, contornos de los lóculos, límite de la cavidad interna y etiquetas con el ID — sin ejes, caja delimitadora ni centroides.

### Referencia de tamaño

Si la imagen incluye una referencia de tamaño, la anotación muestra también el resultado de la detección: una caja delimitadora en **azul claro** alrededor de la tira de referencia con su coeficiente de confianza YOLO, cada círculo contorneado en **rojo**, y el diámetro medido en píxeles marcado con una línea **azul**. Esto permite verificar que la referencia fue detectada correctamente antes de confiar en las mediciones calibradas.

<figure style="text-align: center; margin: 0 auto;">
  <img src="../../../../assets/images/size_reference.png" alt="Caja y círculos de la referencia de tamaño detectados"
       style="height: 400px; width: auto;">
  <figcaption><em>Caja y círculos de la referencia de tamaño detectados</em></figcaption>
</figure>

### Referencia de color

Si se detecta una carta de color Macbeth, la anotación dibuja un rectángulo **verde** sobre cada parche de color, marcando el área exacta de la que se extraerán los valores de color.

<figure style="text-align: center; margin: 0 auto;">
  <img src="../../../../assets/images/color_card.png" alt="Carta de color y parches detectados"
       style="height: 300px; width: auto;">
  <figcaption><em>Carta de color y parches detectados</em></figcaption>
</figure>

---

## Reporte de sesión

En cualquier momento después de ejecutar el análisis, es posible guardar los parámetros utilizados en la sesión actual con `save_parameters()`:

```python
analyzer.save_parameters(output_path="Results/")
```

Esto genera dos archivos:

```
Results/
├── nombre_imagen_parameters.txt
└── nombre_imagen_parameters.json
```

El archivo **`.txt`** registra todos los parámetros utilizados en cada paso del análisis — creación de la máscara, detección de frutos, análisis de morfología y/o análisis de color — junto con las versiones de todas las dependencias. Es útil para compartir, reportar o documentar el análisis realizado.

El archivo **`.json`** contiene la misma información en un formato pensado para ser usado directamente por Traitly. Puede pasarse a `analyze_folder()` o a la CLI para aplicar exactamente los mismos parámetros a un nuevo conjunto de imágenes:

```python
# Python
analyzer.analyze_folder(
    "ruta/a/carpeta",
    json_path="Results/nombre_imagen_parameters.json"
)
```

```bash
# CLI
traitly --fruit_internal -i imagenes/ --json parametros.json
traitly --fruit_external -i imagenes/ --json parametros.json
```

</div>