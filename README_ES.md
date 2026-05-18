<div align="center">
<h1>Traitly</h1>

[![PyPI](https://img.shields.io/pypi/v/traitly?logo=pypi&logoColor=white)](https://pypi.org/project/traitly)
[![Python 3.9+](https://github.com/mariameraz/traitly/actions/workflows/python_compatibility.yml/badge.svg)](https://github.com/mariameraz/traitly/actions/workflows/python_compatibility.yml)
[![Testing](https://github.com/mariameraz/traitly/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/mariameraz/traitly/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/mariameraz/traitly/graph/badge.svg?token=ZDT6RBAGZJ)](https://codecov.io/gh/mariameraz/traitly)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-green?logo=gnu&logoColor=white)](https://github.com/mariameraz/traitly/blob/main/LICENSE)


<a href="https://traitly.readthedocs.io/en/latest/es/tutorials/overview/">Documentación</a> ⋆
<a href="https://traitly.readthedocs.io/en/latest/es/installation/">Instalación</a> ⋆
<a href="https://traitly.readthedocs.io/en/latest/es/user_guide/overview/">Guía de usuario</a> ⋆
<a href="https://traitly.readthedocs.io/en/latest/es/tutorials/overview/">Tutoriales</a>

</div>

<br>

Available in: [![English](https://img.shields.io/badge/Language-English-purple)](https://github.com/mariameraz/traitly/blob/main/README.md)


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

## Guía de uso

Traitly puede usarse desde Python, línea de comandos (CLI) o como aplicación web (Shiny App). Para más detalles:

- [Especificaciones de las imágenes de entrada](https://traitly.readthedocs.io/en/latest/es/user_guide/int_image_requirements/)
- [Arquítectura de Traitly](https://traitly.readthedocs.io/en/latest/es/user_guide/overview/)
- [Inicio rápido en Python](https://traitly.readthedocs.io/en/latest/es/tutorials/quickstart/)
- [Cómo usar Traitly con Línea de Comandos y acceso a Shiny App](https://traitly.readthedocs.io/en/latest/es/user_guide/cli/)
- [Descripción de los resultados](https://traitly.readthedocs.io/en/latest/es/user_guide/results/overview/)
- Prueba nuestro [demo interactivo](https://huggingface.co/spaces/mariameraz/traitly) en líneaˎˊ˗


</br>

## Publicaciones y presentaciones

Los pósters relacionados con Traitly pueden encontrarse en esta carpeta:

- [Pósters](https://drive.google.com/drive/folders/1AvlHWKcDvoE9m9QcmCJ5o-ma9W-LNQMe?usp=share_link) ★ˎˊ˗

Estos materiales proporcionan detalles metodológicos adicionales y resultados de investigaciones derivadas de nuestro paquete.

</br>

## Cómo citar

Estamos trabajando en un manuscrito que describe este trabajo y su uso, y se espera que esté listo en **primavera–verano de 2026**. Mientras tanto, si utilizas Traitly, puedes cítarlo como:

> Torres-Meraz, M. A., Lopez-Moreno, H. & Zalapa, J. (2026). Traitly: A Python Toolkit for High-Throughput Fruit Phenotyping. Zenodo. 10.5281/zenodo.18738366

</br>

## Contacto

Para preguntas o comentarios sobre el proyecto, puedes escribirnos a:

* [ma.meraz@proton.me](mailto:ma.meraz@proton.me)
* [torresmeraz@wisc.edu](mailto:torresmeraz@wisc.edu)

Estamos abiertos a colaboraciones, incluyendo la incorporación de nuevos rasgos creación de tutoriales o flujos de trabajo para cultivos o tejidos vegetales específicos.

</br>

## Contribuciones 

<!-- CONTRIBUTORS-START -->
| Contribuidor | Rol |
|-------------|------|
| [<img src="https://github.com/mariameraz.png" width="44" height="44" valign="middle">&nbsp;María Meraz](https://github.com/mariameraz) | 💻 📆 🚧 📓 ✅ 🐛 📖 ⚠️ 🤔 🌍 |
| [<img src="https://github.com/hector-LM.png" width="44" height="44" valign="middle">&nbsp;Héctor López](https://github.com/hector-LM) | 📖 📓 ✅ 🤔 🐛 🔣 🌍 |
| Juan Zalapa | 🔣 |
| [<img src="https://github.com/AlvaroGuerrero.png" width="44" height="44" valign="middle">&nbsp;Álvaro Guerrero](https://github.com/AlvaroGuerrero) | 🐛 |
<!-- CONTRIBUTORS-END -->

</br>

## Agradecimientos ♡

Agradecemos a los desarrolladores de [OpenCV](https://opencv.org/), [Ultralytics](https://github.com/ultralytics/ultralytics), [EasyOCR](https://github.com/JaidedAI/EasyOCR), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/) y [Shiny](https://shiny.posit.co/py/), así como a todas las librerías de código abierto que hicieron posible este proyecto.
