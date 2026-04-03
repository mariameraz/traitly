---
hide:
  - navigation
  - toc
---

<div class="animate" markdown>

# Bienvenido a Traitly

**Traitly** es una librería en Python diseñada para **automatizar el análisis de imágenes de frutos**, desde una sola muestra hasta cientos de frutos en una sola ejecución. A partir de imágenes RGB estándar, extrae rasgos de **color, forma y tamaño**, tanto de imágenes internas (cortes transversales) como externas (superficie del fruto).

Traitly apuesta por la **ciencia abierta y reproducible**: cada análisis genera automáticamente un reporte de sesión con todos los parámetros y versiones utilizadas, garantizando trazabilidad completa de los resultados. 

!!! tip ""
    Puedes encontrar nuestra documentación tanto en **inglés** como en **español** :fontawesome-regular-face-smile-beam:. Cambia entre lenguajes mediante el ícono que se encuentra a un lado de la barra de búsqueda :fontawesome-regular-hand-point-up:.

---

## Primeros pasos ⊹ ࣪ ˖


<div class="grid cards" markdown>

-   :simple-rocket: __**Instalación**__

    ---

    Instala Traitly y sus dependencias.

    [:octicons-arrow-right-24: Comenzar instalación](installation.md)

-   :material-star-shooting: __**Inicio Rápido**__

    ---

    Ejecuta tu primer análisis en minutos.

    [:octicons-arrow-right-24: Ver tutorial](tutorials/quickstart.md)

-   :fontawesome-solid-code: __**Tutoriales**__

    ---

    Guías paso a paso para diferentes flujos de trabajo.

    [:octicons-arrow-right-24: Explorar guías](tutorials/overview.md)

-   :material-sitemap: __**Arquitectura de Traitly**__

    ---

    Estructura de clases, flujo del análisis y entornos de uso disponibles.

    [:octicons-arrow-right-24: Consultar referencia](user_guide/overview.md)

-   :material-table-heart: __**Tabla de Rasgos**__

    ---

    Descripción detallada de todos los rasgos extraídos.

    [:octicons-arrow-right-24: Consultar referencia](user_guide/results/measurements.md)

</div>



---

## ¿Qué analiza Traitly?

Traitly trabaja con dos tipos principales de imágenes de frutos:

### **Imágenes internas (corte transversal)**

* Morfología interna
* Número y distribución de lóculos
* Grosor del pericarpio
* Simetría
* Color de los tejidos internos

### **Imágenes externas (superficie)**

* Forma general del fruto
* Tamaño
* Color superficial

En ambos casos, los rasgos se extraen a partir de imágenes RGB estándar.
Opcionalmente, Traitly puede **convertir píxeles a unidades métricas reales** mediante la detección automática de un marcador de referencia de tamaño presente en la imagen.

---

## Enfoque metodológico

El núcleo del análisis en Traitly se basa principalmente en **segmentación clásica y procesamiento de imágenes tradicional**, complementado con modelos preentrenados para tareas auxiliares como la detección de etiquetas o referencias de tamaño.

Este diseño prioriza la **robustez, interpretabilidad y reproducibilidad**, y permite que el método sea **fácilmente adaptable** más allá de frutos. Con ajustes mínimos de parámetros, el mismo enfoque puede aplicarse a otros tejidos como **semillas u hojas**, sin necesidad de redefinir la arquitectura del pipeline.

---

## Características principales

* **Procesamiento individual o por lotes**:
  Analiza una sola imagen o carpetas completas en una sola ejecución.

* **Mediciones por fruto**:
  Cada fruto detectado recibe un ID único y se mide de forma independiente.
  Por ejemplo, una imagen con 25 frutos genera 25 filas en la salida.

* **Totalmente automatizado**:
  Detección, segmentación, calibración y extracción de rasgos sin mediciones manuales, reduciendo sesgos y tiempo de fenotipado.

* **Modelos preentrenados incluidos**:
  Detección automática de marcadores de tamaño y etiquetas de muestra, sin configuración adicional.

* **Corrección de color**:
  Detección de **Macbeth Color Checker** para estandarizar color entre experimentos.

* **Identificación automática de muestras**:
  Detección de **códigos QR** y **etiquetas de texto**.

* **Soporte para PDF**:
  Conversión directa de archivos PDF escaneados a imágenes.

* **Reportes de sesión**:
  Guarda automáticamente parámetros, versiones de dependencias y metadatos de cada ejecución.

---

## ¿Dónde puedes usar Traitly?

| Entorno                    | Estado          |
| -------------------------- | --------------- |
| Jupyter Notebook           | :fontawesome-solid-square-check:{.icon-green} Disponible    |
| Línea de comandos (CLI)    | :fontawesome-solid-square-check:{.icon-green} Disponible    |
| Aplicación web (Shiny)     | :fontawesome-solid-square-check:{.icon-green} Disponible |

!!! info ""
    Traitly también está disponible en línea a través de su 
    [demo interactivo](https://huggingface.co/spaces/mariameraz/traitly), 
    sin necesidad de instalación local.
   
    :fontawesome-regular-hand-point-right: Consulta la [Guía de Uso](../user_guide/overview.md#como-usar-traitly) para encontrar el entorno más adecuado según tu caso.
    
---

## Proyecto en crecimiento y colaboraciones

Traitly es un **proyecto en desarrollo**, diseñado para crecer junto con la comunidad científica que lo utiliza. Su arquitectura modular facilita la incorporación de nuevas ideas sin comprometer la consistencia ni la reproducibilidad del análisis.

Todas las contribuciones son bienvenidas, incluyendo:

* :fontawesome-brands-readme: **Creación de tutoriales y documentación**:
  Nuevos tutoriales o flujos de trabajo.

* :fontawesome-solid-seedling: **Propuestas de nuevos traits**:
  Ideas sobre nuevos rasgos morfológicos, geométricos o de color según distintas necesidades experimentales.

* :fontawesome-solid-globe: **Traducción**:
  Expansión de la documentación a nuevos idiomas.

* :fontawesome-solid-heart: **Extensiones metodológicas**:
  Adaptaciones a otros tejidos, especies o contextos.

Nuestra meta es que Traitly crezca como una herramienta colaborativa, flexible y científicamente sólida, guiada por el uso real en investigación.

---

## Construido sobre bases sólidas

Traitly se apoya en librerías consolidadas del ecosistema científico de Python. El procesamiento principal utiliza **OpenCV (contrib)**, **NumPy**, **SciPy**, **pandas** y **matplotlib**, todas con backends en C/C++ que garantizan un alto rendimiento incluso en análisis por lotes de gran escala.

Esto hace que Traitly sea especialmente adecuado para **experimentos de fenotipado masivo** en mejoramiento vegetal y genética, donde es común analizar grandes poblaciones.

</div>
