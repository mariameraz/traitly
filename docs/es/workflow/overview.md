<div class="animate" markdown>

# Arquitectura de Traitly ⊹ ࣪ ˖

Traitly está construido sobre una estructura de clases que separa claramente dos tipos de análisis fenotípicos: el análisis interno y el análisis externo de frutos. Esta organización te permite elegir el nivel de detalle que necesitas sin cargar con funcionalidades que no vas a utilizar.

El módulo principal `traitly.fruit_phenotyping` contiene dos clases principales:

```python
# Análisis de cortes transversales (con lóculos)
from traitly.fruit_phenotyping import FruitInternalAnalyzer

# Análisis de frutos enteros (solo contorno externo)
from traitly.fruit_phenotyping import FruitExternalAnalyzer
```

### ¿Qué hace cada clase?

| Clase | Enfoque | ¿Qué detecta? | Aplicación típica |
|--------|---------|---------------|-------------------|
| `FruitInternalAnalyzer` | **Morfología interna** | Contornos del fruto completo y de cada lóculo individual | Cortes transversales donde interesa cuantificar la organización interna (número de lóculos, área relativa, simetría, grosor del pericarpio, etcétera) y el color de sus diferentes tejidos (pericarpio y lóculos) |
| `FruitExternalAnalyzer` | **Morfología superficial** | Únicamente el contorno exterior del fruto | Frutos enteros para estudios de forma, tamaño y color externo |

Ambas clases comparten la misma lógica de pipeline – procesamiento de imagen, segmentación, extracción de contornos y cálculo de rasgos — pero están optimizadas para sus respectivos objetivos:

- **`FruitInternalAnalyzer`** busca relaciones jerárquicas: un contorno de fruto que contiene múltiples contornos de lóculos en su interior.
- **`FruitExternalAnalyzer`** se enfoca en la silueta completa del fruto, ignorando estructuras internas.

![Analyzer pipelines](../assets/images/workflow.png)
*Flujo general para cada análisis. **A)** `FruitExternalAnalyzer`: pipeline para el análisis de la apariencia externa de frutos enteros. **B)** `FruitInternalAnalyzer`: pipeline extendido para la detección y segmentación de frutos **y** lóculos para cortes transversales de los frutos.*

---

## ¿Cómo usar Traitly?

Traitly puede usarse de dos formas, según el contexto de trabajo:

**Desde Python**: recomendado para explorar parámetros sobre imágenes individuales antes de procesar un lote completo y para analizar múltiples imágenes usando `analyze_folder()` con un archivo `.json` de parámetros.

**Desde la terminal (CLI)**: para ejecutar el análisis directamente desde la terminal sin necesidad de un script de Python, especialmente útil en servidores o entornos de cómputo sin interfaz gráfica. Consulta la sección [CLI](cli.md) para más detalles.

!!! info ""
    En las siguientes secciones encontrarás la documentación detallada de cada clase, incluyendo los rasgos específicos que extraen, ejemplos de uso, parámetros configurables y los requisitos de las imágenes para cada tipo de análisis.

</div>