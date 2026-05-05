<div align="center">
<h1>Traitly</h1>

[![PyPI](https://img.shields.io/pypi/v/traitly?logo=pypi&logoColor=white)](https://pypi.org/project/traitly)
[![Python 3.9+](https://github.com/mariameraz/traitly/actions/workflows/python_compatibility.yml/badge.svg)](https://github.com/mariameraz/traitly/actions/workflows/python_compatibility.yml)
[![MultiOS](https://github.com/mariameraz/traitly/actions/workflows/pytest_multi_os.yml/badge.svg)](https://github.com/mariameraz/traitly/actions/workflows/pytest_multi_os.yml)
[![Testing](https://github.com/mariameraz/traitly/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/mariameraz/traitly/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/mariameraz/traitly/graph/badge.svg?token=ZDT6RBAGZJ)](https://codecov.io/gh/mariameraz/traitly)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20020292.svg)](https://doi.org/10.5281/zenodo.20020292)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-green?logo=gnu&logoColor=white)](https://github.com/mariameraz/traitly/blob/main/LICENSE)


<a href="https://traitly.readthedocs.io/en/latest/es/tutorials/overview/">Documentación</a> ⋆
<a href="https://traitly.readthedocs.io/en/latest/es/installation/">Instalación</a> ⋆
<a href="https://traitly.readthedocs.io/en/latest/es/user_guide/overview/">Guía de usuario</a> ⋆
<a href="https://traitly.readthedocs.io/en/latest/es/tutorials/overview/">Tutoriales</a>

</div>

<br>

Available in: [![English](https://img.shields.io/badge/Language-English-purple)](README.md)


**Traitly** es una herramienta de código abierto en Python que permite el fenotipado automatizado y de alto rendimiento de frutos a partir de imágenes digitales. Mediante métodos de visión por computadora, cuantifica rasgos morfológicos, de simetría y color tanto en estructuras internas como en la apariencia externa del fruto.

Admite tanto el análisis de imágenes individuales como el procesamiento por lotes, lo que facilita el análisis de grandes conjuntos de datos con pocas líneas de código, lo cual es especialmente util en el contexto de programas de mejoramiento de plantas e investigación.
</br>

### ¿Qué puede hacer Traitly?

Traitly procesa imágenes de frutos para medir:

* **Morfología del fruto**: área, perímetro, circularidad, dimensión, y relación de su aspecto
* **Anatomía de los lóculos**: número de lóculos, distribución de tamaños y disposición espacial
* **Estructura del pericarpo**: grosor, uniformidad (CV) e irregularidad de la superficie (lobulación)
* **Cuantificación de color**: análisis multicanal (RGB, HSV, Lab) en diferentes regiones del fruto


<br>

## Publicaciones y presentaciones

Los pósters relacionados con Traitly pueden encontrarse en esta carpeta:

- [Pósters](https://drive.google.com/drive/folders/1AvlHWKcDvoE9m9QcmCJ5o-ma9W-LNQMe?usp=share_link) ★ˎˊ˗

Estos materiales proporcionan detalles metodológicos adicionales y resultados de investigaciones derivadas de nuestro paquete.

</br>

## Uso

Traitly puede ejecutarse de diferentes formas:

| Entorno                    | Estado         |
| -------------------------- | -------------- |
| Jupyter Notebook           | ✔ Disponible   |
| Línea de comandos (CLI)    | ✔ Disponible   |
| Aplicación web (Shiny)     | ✔ Disponible   |

 ⤷ También puedes probar nuestro [demo interactivo](https://huggingface.co/spaces/mariameraz/traitly) en líneaˎˊ˗

---

A continuación se muestra un ejemplo básico de cómo utilizar **Traitly**.

### ⋆ Uso con Python

#### Análisis de morfología interna
```python
from traitly.fruit_phenotyping import FruitInternalAnalyzer

# Análisis de una imagen
path = 'PATH/my_image.jpg'
analyzer = FruitInternalAnalyzer(path)  # Inicializar la clase FruitInternalAnalyzer
analyzer.load_image()                   # Leer la imagen
analyzer.setup_measurements()           # Obtener información de etiquetas y referencia de tamaño
analyzer.generate_fruit_mask()          # Crear una máscara binaria para segmentar frutos y lóculos
analyzer.detect_fruits()                # Filtrar los frutos detectados
analyzer.analyze_morphology()           # Ejecutar el análisis morfológico
analyzer.analyze_color()                # Ejecutar el análisis de color
analyzer.results.save_all()             # Guardar los resultados (archivos .csv e imagen anotada)
analyzer.save_parameters()              # Guardar los parámetros de sesión como .txt y .json

# Análisis por lotes
path = 'PATH/my_folder'
json = 'my_parameters.json'
analyzer = FruitInternalAnalyzer(path)
analyzer.analyze_folder(json_path = json)
```

#### Análisis de morfología externa
```python
from traitly.fruit_phenotyping import FruitExternalAnalyzer

# Análisis de una imagen
path = 'PATH/my_image.jpg'
analyzer = FruitExternalAnalyzer(path)       # Inicializar la clase FruitExternalAnalyzer
analyzer.load_image()                        # Leer la imagen
analyzer.setup_measurements()                # Obtener información de etiquetas y tamaño de referencia
analyzer.generate_fruit_mask()               # Crear una máscara binaria para segmentar frutos
analyzer.detect_fruits()                     # Filtrar los frutos detectados
analyzer.analyze_morphology()                # Ejecutar el análisis morfológico
analyzer.analyze_color(color_channel='RGB')  # Extraer valores medios de canales RGB por fruto
analyzer.results.save_all()                  # Guardar los resultados (archivos .csv e imagen anotada)
analyzer.save_parameters()                   # Guardar los parámetros de sesión como .txt y .json

# Análisis por lotes
path = 'PATH/my_folder'
json = 'my_parameters.json'
analyzer = FruitExternalAnalyzer(path)
analyzer.analyze_folder(json_path = json)
```

### ⋆ Uso desde la línea de comandos
```bash
# Iniciar la aplicación web
traitly-app

# Análisis de morfología interna (imagen individual o carpeta)
traitly --fruit_internal -i tests/sample_data/
traitly --fruit_internal -i tests/sample_data/ -o results/ --num_cores 4
traitly --fruit_internal -i tests/sample_data/ --json config.json

# Análisis de morfología externa (imagen individual o carpeta)
traitly --fruit_external -i tests/sample_data/
traitly --fruit_external -i tests/sample_data/ -o results/ --json config.json --num_cores 4
```

</br>

Para ejemplos más detallados, [consulta nuestros tutoriales](https://traitly.readthedocs.io/en/latest/es/tutorials/overview/) ᯓ★

</br>

## Cómo citar

Estamos trabajando en un manuscrito que describe este trabajo y su uso, y se espera que esté listo en **primavera–verano de 2026**. Mientras tanto, si utilizas Traitly, puedes cítarlo como:

> Torres-Meraz, M. A., Lopez-Moreno, H. & Zalapa, J. (2026). Traitly: A Python Toolkit for High-Throughput Fruit Phenotyping. Zenodo. 10.5281/zenodo.18738366

</br>

## Contacto

Para preguntas o comentarios sobre el proyecto, puedes escribirnos a:

* [ma.meraz@proton.me](mailto:ma.meraz@proton.me)
* [torresmeraz@wisc.edu](mailto:torresmeraz@wisc.edu)

Estamos abiertos a colaboraciones, incluyendo la incorporación de nuevos rasgos, y la creación de tutoriales o flujos de trabajo para cultivos o tejidos vegetales específicos.

</br>

## Contribuciones 

<!-- CONTRIBUTORS-START -->
| Contribuidor | Rol |
|-------------|------|
| [<img src="https://github.com/mariameraz.png" width="44" height="44" valign="middle">&nbsp;María Meraz](https://github.com/mariameraz) | 💻 📆 🚧 📓 ✅ 🐛 📖 ⚠️ 🤔 🌍 |
| [<img src="https://github.com/hector-LM.png" width="44" height="44" valign="middle">&nbsp;Héctor López](https://github.com/hector-LM) | 📖 📓 ✅ 🤔 🐛 🔣 🌍 |
| Juan Zalapa | 🔣 |
<!-- CONTRIBUTORS-END -->

</br>

## Agradecimientos ♡

Agradecemos a los desarrolladores de [OpenCV](https://opencv.org/), [Ultralytics](https://github.com/ultralytics/ultralytics), [EasyOCR](https://github.com/JaidedAI/EasyOCR), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/) y [Shiny](https://shiny.posit.co/py/), así como a todas las librerías de código abierto que hicieron posible este proyecto.
