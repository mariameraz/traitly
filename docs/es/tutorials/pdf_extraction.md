---
hide:
  - navigation
  - toc
---

<div style="text-align: center;" markdown>

# Extracción de imagenes desde un PDF

<p style="color:gray; margin-top: -35px; margin-bottom: 55px;" markdown>**Creado por: María A. Torres-Meraz; Traitly v0.1.0 – Abril, 2026**</p>

</div>

!!! tip ""
    **Requisitos:** Es necesario tener instaladas las dependencias opcionales para la manipulación de PDF de Traitly. Para más detalles, consulta la [Guía de Instalación](../installation.md#dependencias-opcionales).

!!! tip "Sigue el tutorial"
    :fontawesome-solid-file-code: Descarga el cuaderno de Jupyter y las imágenes de muestra para este tutorial [aquí](https://github.com/mariameraz/traitly/tree/main/tutorials_data/cranberry_internal_analysis).

En este tutorial aprenderemos cómo extraer imágenes desde un PDF utilizando Traitly.

Cuando se escanean múltiples muestras, puede resultar más práctico exportarlas todas en un único archivo PDF. En estos casos, la función `pdf_to_img` de Traitly es especialmente útil, ya que permite extraer las imágenes del archivo para continuar con su análisis.

El primer paso es importar la función desde Traitly a nuestro espacio de trabajo, como se muestra a continuación:


```python
from traitly.utils.convert_pdf import pdf_to_img
```

A continuación, debemos indicar la ubicación de nuestros archivos mediante el parámetro `pdf_path`. Por defecto, las imágenes se renombrarán tomando el nombre del PDF como base, añadiendo el sufijo `_page1` y así sucesivamente. No obstante, si las imágenes contienen códigos QR, podemos especificar `detect_qr = True`: esto activará la detección y lectura automática de los QR, y las imágenes serán renombradas de acuerdo con su contenido. Solo se utilizará la primera palabra del código (sin espacios) como nombre del archivo.

También es posible definir el formato de salida deseado; por defecto, las imágenes se exportan en formato JPG.

Al finalizar el proceso, se imprimirá en pantalla un mensaje que indica cuántos archivos fueron analizados y cuántas imágenes se extrajeron. Si deseas suprimir este mensaje, puedes utilizar el parámetro `verbose = False`.

Cabe destacar que la función retorna las rutas de cada imagen generada. Si no deseas que estos valores aparezcan en pantalla, puedes redirigirlos a una variable temporal como `temp`.


```python
path = "./cranberry_slices.pdf"  # Path del archivo PDF

temp = pdf_to_img(pdf_path = path,
                  dpi = 150, 
                  detect_qr = True, 
                  output_format = 'png')
```

    =================================================================
    Extracting images may take a few minutes... ⋆✧｡٩(ˊᗜˋ )و✧*｡
    =================================================================
    > Processing 1 PDF file:
        – Images extracted: 2
        – QR detected: 2/2 img(s)
        – Results folder: /Users/traitly/tutorials_data/pdf_extraction/Images_from_PDF

Podemos explorar `temp` para ver con detalle que regresa `pdf_to_img`:

```python
print(temp)
```

    ['/Users/traitly/tutorials_data/pdf_extraction/Images_from_PDF/SW-1073.png', '/Users/traitly/tutorials_data/pdf_extraction/Images_from_PDF/DP14-497.png']


Es importante tener en cuenta que un DPI bajo puede interferir con la detección correcta de los códigos QR, como se ilustra en el siguiente ejemplo. Al extraer las imágenes con `dpi=70`, el mensaje de salida indica que no fue posible detectar ningún código QR para el mismo PDF. Esto no representa un problema ya que cuando no se detectan códigos QR, las imágenes simplemente se renombran con base en el nombre del PDF, tal como se mencionó anteriormente. Por ello, recomendamos ajustar el DPI en función del tamaño de los objetos presentes en la imagen, asegurando que estos sean nítidos y legibles.


```python
path = "./cranberry_slices.pdf"  # Path del archivo PDF

temp = pdf_to_img(pdf_path = path,
                  dpi = 70, 
                  detect_qr = True, 
                  output_format = 'png')
```

    =================================================================
    Extracting images may take a few minutes... ⋆✧｡٩(ˊᗜˋ )و✧*｡
    =================================================================
    > Processing 1 PDF file:
        – Images extracted: 2
        – QR detected: 0/2 img(s)
        – Results folder: /Users/traitly/tutorials_data/pdf_extraction/Images_from_PDF


Por último, la función también es capaz de procesar múltiples PDFs contenidos en una carpeta. Basta con indicar la ruta a dicha carpeta y la función buscará automáticamente todos los archivos PDF disponibles en ella.

Cuando se trabaja con un gran número de archivos, el proceso de extracción puede volverse lento. Para reducir el tiempo de procesamiento, es posible paralelizar la tarea mediante el argumento `num_cores`, el cual permite distribuir el trabajo entre múltiples núcleos del procesador. Si no se indica, por default, `num_cores = 1`. 


```python
path = "./" # Path de la carpeta que contiene los PDFs

temp = pdf_to_img(pdf_path = path,
                  dpi = 150, 
                  num_cores = 2,
                  detect_qr = True)
```

    =================================================================
    Extracting images may take a few minutes... ⋆✧｡٩(ˊᗜˋ )و✧*｡
    =================================================================
    > Processing 1 PDF file(s):
        – Images extracted: 2
        - QR detected: 2/2 img(s)
        – num_cores: 2
        – Results folder: /Users/traitly/tutorials_data/pdf_extraction/Images_from_PDF


!!! note ""
    En ambos casos, ya sea que se procese un único archivo o una carpeta completa, las imágenes extraídas se guardarán automáticamente en una carpeta llamada `Images_from_PDF`. Si deseas especificar una ubicación diferente, puedes hacerlo mediante el argumento `output_path`.

## ¿Qué sigue?

- [Guía para el Análisis Interno](../user_guide/internal_class.md) — guía detallada con todos los parámetros y métodos disponibles para `FruitInternalAnalyzer`.
- [Guía para el Análisis Externo](../user_guide/external_class.md) — guía detallada con todos los parámetros y métodos disponibles para `FruitExternalAnalyzer`.
- [Tutorial de Análisis Externo](individual_img_tutorial.md) — analizando una imagen paso a paso.
- [Tutorial de Análisis Interno en Cranberry](cranberry_internal_analysis.md) — analizando una imagen paso a paso.

<div style="text-align: center;" markdown>

[← Volver a Tutoriales](overview.md){ .md-button }

</div>
