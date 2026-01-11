Disponible en: [![Español](https://img.shields.io/badge/Language-English-purple)](README.md)


***Traitly*** es una herramienta de código abierto en Python para el fenotipado de frutos de alto rendimiento, que extrae automáticamente rasgos cuantitativos a partir de imágenes digitales de rebanadas de fruta. 
Se centra en el fenotipado de estructuras internas del fruto, utilizando métodos basados en visión por computadora para cuantificar rasgos de morfología, anatomía, simetría y color.

La herramienta admite flujos de trabajo tanto de imágenes individuales como de procesamiento por lotes, lo que permite a los usuarios analizar grandes conjuntos de imágenes con solo unas pocas líneas de código, haciéndola adecuada para programas de mejoramiento vegetal e investigación.


</br>

> **Nota:**  
> Actualmente se está preparando un manuscrito que describe este software, y se espera su publicación en **primavera–verano de 2026**.

</br>

### ¿Qué puede hacer Traitly?

Traitly procesa imágenes de frutos para medir:

* **Morfología del fruto**: área, perímetro, circularidad, relación de aspecto y dimensiones de la caja delimitadora  
* **Anatomía de los lóculos**: número de lóculos, distribución de tamaños y disposición espacial  
* **Estructura del pericarpo**: grosor, uniformidad (CV) e irregularidad de la superficie (lobulación)  
* **Cuantificación de color**: análisis multicanal (RGB, HSV, Lab) en diferentes regiones del fruto  


**👉 Para consultar la lista completa de rasgos extraídos, ver:** [Tablas de rasgos](docs/documentation.md)

</br>

## Estatus del proyecto

**Traitly se encuentra en fase de pre-lanzamiento y en desarrollo activo.**  
El código fuente aún no está disponible públicamente.

La documentación actual corresponde a una **versión preliminar del manual** y está sujeta a cambios.  
Se proporcionarán más detalles, ejemplos y aclaraciones en futuras actualizaciones.

Las actualizaciones sobre el lanzamiento público se anunciarán a través de este repositorio y en [LinkedIn](https://www.linkedin.com/in/alemeraz/).  
Se recomienda a las personas interesadas seguir el repositorio para mantenerse informadas.

</br>

## Publicaciones y presentaciones

Los pósters relacionados con Traitly pueden encontrarse en esta carpeta:

- [Pósters](https://drive.google.com/drive/folders/1AvlHWKcDvoE9m9QcmCJ5o-ma9W-LNQMe?usp=share_link) ★ˎˊ˗

Estos materiales proporcionan detalles metodológicos adicionales y resultados de investigaciones relacionadas.

</br>

## Uso

A continuación se muestra un ejemplo básico de cómo utilizar **traitly**:

Uso con Python

```python
from traitly.internal_structure import FruitAnalyzer

##########################
# Análisis de una imagen #
##########################
path = 'PATH/my_image.jpg'

analyzer = FruitAnalyzer(path)  # Inicializar la clase FruitAnalyzer

analyzer.read_image()           # Leer la imagen
analyzer.setup_measurements()   # Obtener información de etiquetas y tamaño de referencia
analyzer.create_mask()          # Crear una máscara binaria para segmentar frutos y lóculos
analyzer.find_fruits()          # Filtrar los frutos detectados
analyzer.analyze_image()        # Ejecutar el análisis del fruto
analyzer.results.save_all()     # Guardar el archivo CSV y la imagen anotada

######################
# Análisis por lotes #
######################
path = 'PATH/my_folder'

analyzer = FruitAnalyzer(path)  # Inicializar la clase FruitAnalyzer
analyzer.analyze_folder()       # Ejecutar el análisis en todas las imágenes válidas de la carpeta.
                                # Se guardará un único archivo CSV y las imágenes anotadas correspondientes.
````

Uso desde la línea de comandos 

```bash
traitly internal_structure -i PATH/my_folder
```

</br>

Ejemplos más detallados:
👉 [https://github.com/mariameraz/traitly/blob/main/docs/traitly-examples.ipynb](https://github.com/mariameraz/traitly/blob/main/docs/traitly-examples.ipynb)

</br>

## Contacto ˖᯽ ݁˖

Para consultas sobre el proyecto o posibles colaboraciones, por favor envíe un mensaje a:

* [ma.torresmeraz@gmail.com](mailto:ma.torresmeraz@gmail.com)
* [torresmeraz@wisc.edu](mailto:torresmeraz@wisc.edu)


