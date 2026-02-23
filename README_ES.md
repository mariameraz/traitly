# Welcome to Traitly 

Available in: [![English](https://img.shields.io/badge/Language-English-purple)](README.md)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-green.svg)](https://www.gnu.org/licenses/agpl-3.0) [![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://www.python.org/) [![Version](https://img.shields.io/badge/Version-0.1.0--beta-orange)]() [![DOI](https://zenodo.org/badge/1122521844.svg)](https://doi.org/10.5281/zenodo.18058712)

***Traitly*** es una herramienta de código abierto en Python para el fenotipado de frutos de alto rendimiento, que extrae automáticamente rasgos cuantitativos a partir de imágenes digitales de frutos completos o en rodajas.

Se centra en el fenotipado de estructuras internas y la morfología externa del fruto, utilizando métodos basados en visión por computadora para cuantificar rasgos de morfología, anatomía, simetría y color.

La herramienta admite flujos de trabajo tanto de imágenes individuales como de procesamiento por lotes, lo que permite a los usuarios analizar grandes conjuntos de imágenes con solo unas pocas líneas de código, haciéndola adecuada para programas de mejoramiento vegetal e investigación.

</br>

> **Nota:**  
> Actualmente se está preparando un manuscrito que describe este software y su uso, y se espera su envío en **primavera–verano de 2026**. Mientras tanto, si utilizas Traitly en tu investigación, por favor cítalo como:
>
> Torres-Meraz, M. A., & Lopez-Moreno, H. (2026). Traitly: A Python Tool for High-Throughput Fruit Phenotyping. Zenodo. https://doi.org/10.5281/zenodo.18738367

</br>

### ¿Qué puede hacer Traitly?

Traitly procesa imágenes de frutos para medir:

* **Morfología del fruto**: área, perímetro, circularidad, relación de aspecto y dimensiones de la caja delimitadora  
* **Anatomía de los lóculos**: número de lóculos, distribución de tamaños y disposición espacial  
* **Estructura del pericarpo**: grosor, uniformidad (CV) e irregularidad de la superficie (lobulación)  
* **Cuantificación de color**: análisis multicanal (RGB, HSV, Lab) en diferentes regiones del fruto  

**👉 Para consultar la lista completa de rasgos extraídos, ver:**
- [![Documentation_EN](https://img.shields.io/badge/Documentation-English-lightblue)](docs/documentation.md)
- [![Documentation_ES](https://img.shields.io/badge/Documentaci%C3%B3n-Espa%C3%B1ol-red)](docs/documentation_ES.md)

</br>

## Estatus del proyecto

**Traitly se encuentra actualmente en fase beta y en proceso de pruebas en diferentes sistemas y entornos.**

El código fuente ya está disponible públicamente. La arquitectura del proyecto y su lógica central están definidas, y un grupo de early testers está evaluando la herramienta en diferentes sistemas, flujos de trabajo y casos de uso.

La documentación está en construcción activa, y se irán añadiendo más detalles, ejemplos y aclaraciones conforme avancen las pruebas.

Actualmente se está desarrollando una aplicación web con Streamlit, con el objetivo de ofrecer una interfaz amigable para ejecutar Traitly sin necesidad de escribir código.

Las actualizaciones se anunciarán a través de este repositorio y en [LinkedIn](https://www.linkedin.com/in/alemeraz/).  
Se recomienda a las personas interesadas seguir el repositorio para mantenerse informadas.

</br>

## Publicaciones y presentaciones

Los pósters relacionados con Traitly pueden encontrarse en esta carpeta:

- [Pósters](https://drive.google.com/drive/folders/1AvlHWKcDvoE9m9QcmCJ5o-ma9W-LNQMe?usp=share_link) ★ˎˊ˗

Estos materiales proporcionan detalles metodológicos adicionales y resultados de investigaciones relacionadas.

</br>

## Uso

A continuación se muestra un ejemplo básico de cómo utilizar **traitly**.

### Uso con Python

#### Análisis interno
```python
from traitly.fruit_phenotyping import FruitInternalAnalyzer

##########################
# Análisis de una imagen #
##########################
path = 'PATH/my_image.jpg'
analyzer = FruitInternalAnalyzer(path)  # Inicializar la clase FruitInternalAnalyzer
analyzer.load_image()                   # Leer la imagen
analyzer.setup_measurements()           # Obtener información de etiquetas y tamaño de referencia
analyzer.generate_fruit_mask()          # Crear una máscara binaria para segmentar frutos y lóculos
analyzer.detect_fruits()                # Filtrar los frutos detectados
analyzer.analyze_morphology()           # Ejecutar el análisis morfológico
analyzer.analyze_color()                # Ejecutar el análisis de color
analyzer.results.save_all()             # Guardar los archivos .csv de color y morfología, y la imagen anotada
analyzer.save_parameters()              # Guardar los parámetros de sesión como archivos .txt y .json

######################
# Análisis por lotes #
######################
path = 'PATH/my_folder'
json = 'my_parameters.json'
analyzer = FruitInternalAnalyzer(path)          # Inicializar la clase FruitInternalAnalyzer
analyzer.analyze_folder(json_path = json)       # Ejecutar el análisis en todas las imágenes válidas de la carpeta
# Se guardará un único archivo CSV y las imágenes anotadas correspondientes.
```

#### Análisis externo
```python
from traitly.fruit_phenotyping import FruitExternalAnalyzer

##########################
# Análisis de una imagen #
##########################
path = 'PATH/my_image.jpg'
analyzer = FruitExternalAnalyzer(path)  # Inicializar la clase FruitExternalAnalyzer
analyzer.load_image()                   # Leer la imagen
analyzer.setup_measurements()           # Obtener información de etiquetas y tamaño de referencia
analyzer.generate_fruit_mask()          # Crear una máscara binaria para segmentar frutos
analyzer.detect_fruits()                # Filtrar los frutos detectados
analyzer.analyze_morphology()           # Ejecutar el análisis morfológico
analyzer.analyze_color(stat='median',
    color_channel='RGB')                # Extraer los valores medianos de los canales RGB para cada fruto
analyzer.results.save_all()             # Guardar los archivos .csv de color y morfología, y la imagen anotada
analyzer.save_parameters()              # Guardar los parámetros de sesión como archivos .txt y .json

######################
# Análisis por lotes #
######################
path = 'PATH/my_folder'
json = 'my_parameters.json'
analyzer = FruitExternalAnalyzer(path)          # Inicializar la clase FruitExternalAnalyzer
analyzer.analyze_folder(json_path = json)       # Ejecutar el análisis en todas las imágenes válidas de la carpeta
# Se guardará un único archivo CSV y las imágenes anotadas correspondientes.
```

### Uso desde la línea de comandos
```bash
# Análisis de estructura interna (imagen individual o carpeta)
traitly --fruit_internal -i tests/sample_data/
traitly --fruit_internal -i tests/sample_data/ -o results/ --num_cores 4
traitly --fruit_internal -i tests/sample_data/ --json config.json

# Análisis externo (imagen individual o carpeta)
traitly --fruit_external -i tests/sample_data/
traitly --fruit_external -i tests/sample_data/ -o results/ --json config.json --num_cores 4
```

</br>

Ejemplos más detallados:
👉 [https://github.com/mariameraz/traitly/blob/main/docs/traitly-examples.ipynb](https://github.com/mariameraz/traitly/blob/main/docs/traitly-examples.ipynb)

</br>

## Contacto ˖᯽ ݁˖

Para consultas sobre el proyecto o posibles colaboraciones, por favor envíe un mensaje a:

* [ma.torresmeraz@gmail.com](mailto:ma.torresmeraz@gmail.com)
* [torresmeraz@wisc.edu](mailto:torresmeraz@wisc.edu)

Estamos abiertos a colaboraciones, incluyendo el desarrollo de pipelines para especies específicas, la incorporación de nuevos rasgos o mediciones, y la creación de tutoriales o flujos de trabajo adaptados a cultivos o tejidos vegetales específicos.
