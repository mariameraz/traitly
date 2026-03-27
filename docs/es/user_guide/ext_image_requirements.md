<div class="animate" markdown>

# Especificaciones de las Imágenes

La calidad de los análisis depende directamente de la calidad de las imágenes. Traitly está diseñado para ser robusto, pero seguir estas recomendaciones garantizará los mejores resultados.

---

## 1. Adquisición de las imágenes

### 1.1 Equipo recomendado

Para el análisis externo de los frutos, las imágenes se pueden capturar con una **cámara fotográfica**, ya sea de un teléfono inteligente o una cámara profesional. Sin importar el tipo de dispositivo que se elija, la consistencia es clave. Usa la **misma cámara** para tomar todas las imágenes de un experimento, ya que los ajustes de color, balance de blancos y procesamiento interno varían entre fabricantes y modelos, y estas diferencias pueden introducir sesgos en las mediciones de color. 

### 1.2 Preparación de las muestras

- **Posición de la cámara**: Fija la cámara de forma que quede **paralela y perpendicular** a los frutos, sin ángulo de inclinación. Tomar imágenes desde un ángulo introduce distorsión geométrica que puede afectar las mediciones morfológicas. También recomendamos mantener la cámara en la **misma posición y lugar** durante todo el experimento para asegurar la misma distancia entre la lente y fondo en cada imágen.
- **Objetos ajenos al fruto**: Procura que las imágenes contengan únicamente los objetos de interés. La presencia de tallos, hojas, semillas sueltas o suciedad, aunque puede filtrarse en pasos posteriores del análisis, incrementa el tiempo de procesamiento, ya que este escala con el número de contornos detectados por imagen.

### 1.3 Iluminación

La iluminación es uno de los factores más críticos para obtener mediciones de color reproducibles. Recomendamos ampliamente el uso de una **fuente de luz controlada y estable** (ej., paneles LED). 

Siempre que sea posible, utiliza **difusores** entre la fuente de luz y los frutos. La luz directa genera reflejos que son especialmente problemáticos en frutos cerosos o brillantes (como uva, ciruela o arándano), y puede alterar tanto la segmentación como las mediciones de color. 

!!! tip "Configuración de la cámara"
    Si el dispositivo lo permite, fija manualmente los parámetros de captura: exposición, ISO, balance de blancos y apertura. Evita los modos automáticos o el uso de flash, ya que estos ajustan las condiciones entre toma y toma.

    Independientemente de la configuración elegida, es fundamental capturar **todas las imágenes de un mismo experimento bajo las mismas condiciones**: mismo dispositivo, misma fuente de luz, misma distancia y mismos ajustes de software. Cualquier variación entre sesiones puede introducir inconsistencias en las mediciones de color y morfología.

### 1.4 Configuración del fondo

La elección del fondo es especialmente importante en el análisis externo, ya que determina tanto la calidad de la segmentación como la ausencia de artefactos de color en los bordes de los frutos.

**Material**: Usa un material **mate, sin textura ni relieve**, para evitar sombras y reflexiones que puedan confundirse con contornos de fruto o alterar el color percibido.

**Color**: Elige un color que **contraste claramente con el color de tus frutos**. Como se muestra en las imágenes a continuación, la elección del fondo tiene un impacto directo en la calidad de la segmentación:

- Fondos **blancos** no funcionan bien con frutos de colores claros (amarillos, blancos, rosados)
- Fondos **negros** no funcionan bien con frutos oscuros (moras, arándanos negros, ciruelas)
- Fondos **azules o verdes** pueden funcionar, pero en algunos frutos el color del fondo puede reflejarse en los bordes, lo que puede afectar las mediciones de color
- Para la mayoría de los frutos, recomendamos un **gris neutro de saturación media-baja**, ya que ofrece buen contraste con una amplia gama de colores de fruto y minimiza los artefactos de reflexión

<div style="display: flex; gap: 16px; justify-content: center; align-items: flex-start;">
  <figure style="text-align: center; margin: 0;">
    <img src="../../../assets/images/cranberry_background_example.jpg" alt="Ejemplos de fondo para cranberry"
         style="height: 400px; width: auto;">
    <figcaption><em>Arándano sobre distintos fondos. En frutos de colores claros, el fondo blanco reduce el contraste y dificulta la segmentación</em></figcaption>
  </figure>
  <figure style="text-align: center; margin: 0;">
    <img src="../../../assets/images/blackberry_background_example.jpg" alt="Ejemplos de fondo para mora"
         style="height: 400px; width: auto;">
    <figcaption><em>Mora sobre distintos fondos. En frutos oscuros, el fondo negro no genera suficiente contraste para una segmentación confiable</em></figcaption>
  </figure>
  
</div>

Traitly soporta los fondos predefinidos `'black'`, `'white'`, `'blue'` y `'gray'`, o permite definir rangos HSV personalizados para cualquier otro color.

### 1.5 Formato y resolución

**Formatos soportados:**

- `.jpg`, `.jpeg`
- `.png`
- `.tif`, `.tiff`
- `.bmp`

**Resolución (DPI / megapíxeles):**

- No hay un requisito mínimo estricto
- La resolución adecuada depende del tamaño y complejidad de las estructuras a medir: frutos pequeños requieren mayor resolución que frutos grandes
- **Recomendación clave**: Usa la **misma resolución** para todas las imágenes de un mismo experimento
- Consistencia > resolución absoluta

---

## 2. Referencias de tamaño

Traitly ofrece dos formas de convertir píxeles a unidades métricas reales en el análisis externo.

### 2.1 Métodos de calibración

| Método | Cuándo usarlo | Reproducibilidad |
|--------|---------------|----------------|
| **Referencia circular** | Incluir una tira de círculos de diámetro conocido en la imagen | :fontawesome-solid-star:{ .icon-yellow }:fontawesome-solid-star:{ .icon-yellow }:fontawesome-solid-star:{ .icon-yellow } Entre equipos y experimentos |
| **Sin calibración** | Solo necesitas medidas relativas o comparar dentro de la misma imagen | :fontawesome-solid-star:{ .icon-yellow } Mismo lote con configuración idéntica (ej., resolución y distancia de captura) |

!!! tip "Recomendación"
    Usa siempre que sea posible la **tira de círculos de referencia negros sobre fondo blanco**. Traitly detecta automáticamente estos círculos usando un modelo YOLO entrenado específicamente para este propósito.

    Cuando uses la plantilla, verifica el diámetro real de los círculos impresos con una regla. Las impresoras pueden escalar el documento al imprimir, por lo que el tamaño final puede diferir. Usa siempre el valor medido, no el del archivo.

    [:octicons-download-24: Descargar plantilla de referencia circular](../../assets/templates/size_reference_template.pdf)

### 2.2 ¿Por qué usar referencia circular?

Las cámaras pueden variar en su resolución efectiva dependiendo de la distancia de captura y el lente utilizado. La referencia circular:

- Proporciona una calibración por imagen independiente
- Es más precisa que asumir una resolución fija
- Permite comparar directamente resultados entre diferentes experimentos, datasets o dispositivos, ya que todas las medidas están calibradas contra un estándar común


---

## 3. Identificación de muestras

Traitly puede extraer automáticamente información sobre las muestras mediante la lectura de códigos QR o el reconocimiento de texto (OCR), almacenando la información en las tablas de resultados. Recomendamos usar **códigos QR** siempre que sea posible, ya que son más rápidos de detectar, toleran peor iluminación y no dependen tanto de la calidad de la imagen ni del tipo de fuente como con OCR.

Para generar códigos QR puede usarse cualquier herramienta disponible. Si necesitas crear múltiples etiquetas desde un archivo de texto (`.txt`, `.csv` o `.tsv`), recomendamos el software en línea **[QRLabel](https://github.com/mariameraz/qrlabel)**, con el que generamos las etiquetas de la imagen de ejemplo.

### 3.1 Detección de etiquetas

La detección sigue este orden:

1. Intenta detectar un código QR
2. Si no encuentra QR, aplica OCR

!!! warning "OCR es sensible a la calidad de la imagen y la fuente"
    A diferencia del QR, la tasa de detección del OCR depende directamente de la resolución de la imagen, el contraste de la etiqueta y el tipo de fuente utilizada. Una etiqueta mal diseñada o una imagen de baja calidad puede resultar en texto detectado incorrectamente o no detectado. Si usas OCR, sigue las recomendaciones de la tabla a continuación.

Para una detección óptima con OCR:

| Recomendación | Ejemplo bueno :octicons-check-circle-fill-24:{ .icon-green } | Ejemplo malo :octicons-x-circle-fill-24:{ .icon-red } |
|--------------|---------------|--------------|
| **Fondo claro, texto oscuro** | Texto negro sobre blanco | Texto gris sobre fondo gris |
| **Fuente clara y sans-serif** | Arial, Helvetica, Consolas, Roboto, Verdana | Fuentes decorativas, cursivas |
| **Evitar caracteres ambiguos** | Usar dígitos y mayúsculas: `TOM-001` | `I`, `l`, `O`, `0` son fácilmente confundidos entre sí |
| **Usar separadores entre campos** | `TOM-001`, `MANZ-02-A` | `TOM001`, `MANZA02` |
| **Tamaño de fuente suficiente** | ≥ 14 pt | Texto muy pequeño reduce la tasa de detección |
| **Suficiente contraste** | Alto contraste | Bajo contraste |

**Ejemplo de etiquetas bien diseñadas:**
```
Buenas:   TOM-001      CHILE-02       MANZ-123
          TOM-001-A    CHILE-02-REP1

Evitar:   TOM-00I      CHlLE-02       MANZ-I23   <- I/l/1 ambiguos
          TOM001       CHILE02        MANZ123    <- sin separadores
          Tom-001      chile-02       Manz-123   <- mezcla mayúsculas/minúsculas
```

---

## 4. Resumen de buenas prácticas

:octicons-check-circle-fill-24:{ .icon-green } **Hacer:**

- Usar el mismo dispositivo para todo el experimento
- Colocar la cámara paralela y perpendicular a los frutos
- Usar fuente de luz controlada y estable
- Usar difusores para evitar reflejos en frutos cerosos
- Usar fondo mate, sin textura, que contraste con el color del fruto
- Incluir referencia circular para calibración precisa
- Incluir QR para identificación de muestras
- En caso de usar OCR, crear etiquetas con alto contraste y fuentes claras
- Mantener misma resolución y distancia de captura en todo el experimento

:octicons-x-circle-fill-24:{ .icon-red } **Evitar:**

- Mezclar dispositivos dentro de un mismo experimento
- Tomar imágenes en ángulo
- Iluminación variable, natural no controlada o con flash automático
- Luz directa sin difusión sobre frutos cerosos o brillantes
- Fondos con textura, relieve o saturación alta
- Fondos que no contrasten con el color del fruto
- Objetos ajenos al fruto (tallos, hojas, semillas sueltas) y suciedad en las imágenes
- Etiquetas con caracteres ambiguos (`I`/`l`, `O`/`0`)
- Asumir el tamaño de los círculos de la referencia sin verificar con regla

</div>
