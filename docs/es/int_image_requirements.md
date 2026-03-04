<div class="animate" markdown>

# Especificaciones de las Imágenes

La calidad de los análisis depende directamente de la calidad de las imágenes. Traitly está diseñado para ser robusto, pero seguir estas recomendaciones garantizará los mejores resultados.

---

## Adquisición de las imágenes

### Equipo recomendado

Para obtener resultados consistentes, recomendamos usar un **escáner plano** convencional. Este método permite controlar las condiciones de iluminación y mantener la misma distancia de captura entre muestras.

### Preparación de las muestras

- **Corte de frutos**: Utiliza una navaja o cuchillo afilado para obtener cortes transversales limpios.
- **Mantenimiento**: Cambia o afila la navaja periódicamente para evitar que el desgaste afecte la calidad del corte.
- **Importancia**: Cortes irregulares pueden introducir sesgos en las mediciones morfológicas.
- **Frutos jugosos**: En frutos con alto contenido de jugo, retira el exceso suavemente con un paño antes de colocarlos sobre el escáner. Puedes limpiar la superficie del escáner con alcohol entre tomas para eliminar residuos que puedan ser detectados como contornos de fruto.
- **Objetos ajenos al fruto**: Procura que las imágenes contengan únicamente los objetos de interés. La presencia de tallos, hojas, semillas sueltas o suciedad, aunque puede filtrarse en pasos posteriores del análisis, incrementa el tiempo de procesamiento, ya que este escala con el número de contornos detectados por imagen.

### Configuración del fondo

Traitly asume por defecto que el **fondo es negro**. Para lograr esto:

1. Coloca los frutos directamente sobre el escáner
2. Cubre el escáner con una caja de cartón o cualquier otro material que bloquee la luz exterior
3. Esto garantiza un fondo uniforme y consistente en todas las imágenes

??? tip "Configuración del escáner"
    Muchos escáneres permiten ajustar parámetros como el perfil de color, la corrección de blancos y otros ajustes de imagen. Para obtener mediciones de color reproducibles, se recomienda configurar el escáner para que capture los colores tal como los registra el sensor, sin aplicar correcciones automáticas de color ni balance de blancos.

    Independientemente de la configuración elegida, es fundamental escanear **todas las imágenes de un mismo experimento bajo las mismas condiciones**: mismo escáner, misma resolución y mismos ajustes de software. Cualquier variación entre sesiones puede introducir inconsistencias en las mediciones de color y morfología.

<br>

<div style="display: flex; gap: 16px; justify-content: center; align-items: flex-start;">
  <figure style="text-align: center; margin: 0;">
    <img src="../../assets/images/scanner_box.jpg" alt="Caja negra y escáner"
         style="height: 600px; width: auto;">
    <figcaption><em>Ejemplo de caja negra y escáner</em></figcaption>
  </figure>
  <figure style="text-align: center; margin: 0;">
    <img src="../../assets/images/slices_image.jpg" alt="Ejemplo de imágen obtenida con el escáner"
         style="height: 600px; width: auto;">
    <figcaption><em>Ejemplo de imágen escaneada</em></figcaption>
  </figure>
</div>



### Formato y resolución

**Formatos soportados:**

- `.jpg`, `.jpeg`
- `.png`
- `.tif`, `.tiff`
- `.bmp`

**PDF:**

Durante la toma de imágenes, es más práctico configurar el escáner para guardar todas las capturas en un único archivo PDF con múltiples páginas, en lugar de manejar archivos individuales. Traitly incluye funciones para extraer automáticamente cada página como una imagen independiente, lista para el análisis. Ver detalles en [Tutoriales](tutorials/quickstart.md).

**Resolución (DPI):**

- No hay un requisito mínimo estricto
- La resolución adecuada depende del tamaño y complejidad de las estructuras a medir: frutos o lóculos pequeños requieren mayor resolución que frutos o lóculos grandes
- **Recomendación clave**: Usa la **misma resolución (DPI)** para todas las imágenes de un mismo experimento
- Consistencia > resolución absoluta

---

## Referencias de tamaño

Traitly ofrece múltiples formas de convertir píxeles a unidades métricas reales.

### Métodos de calibración

| Método | Cuándo usarlo | Reproducibilidad |
|--------|---------------|----------------|
| **Referencia circular** | Incluir una tira de círculos de diámetro conocido | :fontawesome-solid-star:{ .icon-yellow }:fontawesome-solid-star:{ .icon-yellow }:fontawesome-solid-star:{ .icon-yellow } Entre equipos y experimentos |
| **Dimensiones conocidas** | Conoces el área de captura del escáner (ej. 21×29.7 cm) | :fontawesome-solid-star:{ .icon-yellow }:fontawesome-solid-star:{ .icon-yellow } Mismo escáner y resolución |
| **Sin calibración** | Solo necesitas medidas relativas o comparar dentro de la misma imagen | :fontawesome-solid-star:{ .icon-yellow } Mismo lote con configuración idéntica (ej., DPI y tamaño) |

!!! tip "Recomendación"
    Usa siempre que sea posible la **tira de círculos de referencia negros sobre fondo blanco**. Traitly detecta automáticamente estos círculos usando un modelo YOLO entrenado específicamente para este propósito.

    Cuando uses la plantilla, verifica el diámetro real de los círculos impresos con una regla. Las impresoras pueden escalar el documento al imprimir, por lo que el tamaño final puede diferir.

    [:octicons-download-24: Descargar plantilla de referencia circular](../../assets/templates/size_reference_template.pdf)

### ¿Por qué usar referencia circular?

Los escáneres pueden presentar pequeñas variaciones entre sus dimensiones declaradas y las reales de captura. 

La referencia circular:

- Corrige estas variaciones internas del escáner
- Proporciona una calibración por imagen independiente
- Es más precisa que asumir las dimensiones declaradas del escáner
- Al derivar la escala del diámetro promedio de múltiples círculos detectados, el método amortigua el efecto de pequeñas distorsiones geométricas del escáner
- Al derivar la escala de la referencia y no de las dimensiones de la imagen, permite procesar en lote imágenes de distintos tamaños, ya que la conversión píxel/cm es invariable al tamaño o recorte de la imagen

---

## Identificación de muestras

Traitly puede extraer automáticamente información sobre las muestras mediante la lectura de códigos QR o el reconocimiento de texto (OCR), almacenando la información en las tablas de resultados. Recomendamos usar **códigos QR** siempre que sea posible, ya que son más rápidos de detectar, toleran peor iluminación y no dependen tanto de la calidad de la imagen ni del tipo de fuente como con OCR.

Para generar códigos QR puede usarse cualquier herramienta disponible. Si necesitas crear múltiples etiquetas desde un archivo de texto (`.txt`, `.csv` o `.tsv`), recomendamos el software en línea **[QRLabel](https://github.com/mariameraz/qrlabel)**, con el que generamos las etiquetas de la imagen de ejemplo.

### Detección de etiquetas

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

## Resumen de buenas prácticas

:octicons-check-circle-fill-24:{ .icon-green } **Hacer:**

- Usar escáner con fondo negro (caja cubierta)
- Cambiar navaja periódicamente para cortes limpios
- Retirar el exceso de jugo con un paño en frutos muy jugosos
- Limpiar el escáner con alcohol entre tomas
- Incluir referencia circular para calibración precisa
- Incluir QR para mayor velocidad
- En caso de usar OCR para detectar texto, crear etiquetas con alto contraste y fuentes claras
- Mantener mismo DPI en todo el experimento

:octicons-x-circle-fill-24:{ .icon-red } **Evitar:**

- Iluminación variable o reflejos
- Objetos ajenos al fruto (tallos, hojas, semillas sueltas) y suciedad en las imágenes
- Cortes con navajas desgastadas
- Mezclar resoluciones en un mismo lote
- Etiquetas con caracteres ambiguos (`I`/`l`, `O`/`0`)
- Asumir dimensiones del escáner y los círculos de la referencia de tamaño sin verificar

</div>
