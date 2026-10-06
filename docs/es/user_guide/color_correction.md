<div class="animate" markdown>

# Corrección de color: clase y métodos

Esta sección explica todo lo que necesitas para trabajar con `ColorCorrection`, la clase que corrige los colores de tus imágenes usando una tarjeta de color. Cada método se explica junto con sus parámetros y la forma en que encaja en tu flujo de trabajo.

---

## 1. Clase principal

`ColorCorrection` ajusta los colores de una imagen para que coincidan con los valores de referencia de una tarjeta de color de 24 parches que aparece en la misma foto. Así se reducen las diferencias causadas por la iluminación, la cámara o sus ajustes, y las mediciones de color se pueden comparar entre imágenes y sesiones. Puedes usarla para corregir una sola imagen o una carpeta completa.

```python
from traitly.color_correction import ColorCorrection

# Para corregir una sola imagen
cc = ColorCorrection(path = "ruta/a/mi/imagen.jpg")

# Para corregir varias imágenes en una carpeta
cc = ColorCorrection(path = "ruta/a/mi/carpeta/con/imagenes/")
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `path` | `str` | Ruta de la imagen o de la carpeta que contiene las imágenes que quieres corregir |

!!! info "Lo que necesitas"
    - Cada imagen debe incluir una **tarjeta de color de 24 parches** (X-Rite ColorChecker Classic, también conocida como tarjeta Macbeth), completamente visible.
    - Solo se detecta una tarjeta por imagen.
    - La detección de la tarjeta depende del módulo `mcc` de `opencv-contrib-python`, que no está disponible en la [instalación para Mac Intel / macOS antiguo](../installation.md). En esos sistemas, **la corrección de color no es compatible**.

!!! tip "Recomendación"
    Cuando tengas una carpeta de imágenes por procesar, te sugerimos:

    1. **Empezar con una imagen representativa** para comprobar que la tarjeta se detecta y que el resultado se ve bien
    2. Evaluar la corrección con `calculate_delta_e_stats()`
    3. Guardar la configuración con `save_parameters()`
    4. Usar `analyze_folder(json_path="tu_archivo.json")` para corregir todo el lote con los mismos parámetros

    Una vez guardadas tus imágenes corregidas, puedes analizarlas con [`FruitInternalAnalyzer`](internal_class.md) o [`FruitExternalAnalyzer`](external_class.md).

### Cómo funciona la corrección

La tarjeta tiene 24 parches de color cuyos [valores de referencia bajo luz de día](https://www.xrite.com/service-support/new_color_specifications_for_colorchecker_sg_and_classic_charts) son conocidos. La clase mide el color de cada parche en tu imagen, lo compara con su valor de referencia y construye un modelo (PLSR) que describe cómo la cámara y la iluminación alteraron esos colores. Después, ese modelo se aplica a **todos los píxeles de la imagen**.

<br>

</div>

---

## 2. Cómo se organiza la corrección

Al trabajar con `ColorCorrection`, el proceso sigue este orden:

```python
from traitly.color_correction import ColorCorrection

# Corregir una sola imagen
cc = ColorCorrection('ruta/a/mi/imagen.jpg')

cc.load_image()                 # Carga la imagen
cc.detect_color_checker()       # Busca la tarjeta de color en la imagen
cc.apply_color_correction()     # Corrige los colores de toda la imagen
cc.calculate_delta_e_stats()    # (Opcional) Mide cuánto mejoraron los colores

## Guardar resultados
cc.save_img()                   # Guarda la imagen corregida
cc.save_csv()                   # (Opcional) Guarda el error de color por parche
cc.save_parameters()            # (Opcional) Guarda los parámetros usados en la sesión
```

Si trabajas con multiples imágenes, no necesitas ejecutar cada paso por separado: `analyze_folder()` se encarga de todo automáticamente:

```python
# Corregir varias imágenes
cc = ColorCorrection('ruta/a/mi/carpeta')                           # Inicializa con la ruta de tu carpeta
cc.analyze_folder(json_path = 'ruta/a/mis/parametros.json')         # Ejecuta la corrección, usando opcionalmente parámetros guardados
```

<br>

---

## 3. Qué puedes obtener del objeto creado

Después de ejecutar los métodos, `cc` guarda resultados en atributos que puedes consultar:

| Atributo | Contenido |
|----------|-----------|
| `input_path` | Ruta de la imagen (o carpeta) que se está procesando |
| `original_img` | Imagen original en formato de color BGR |
| `corrected_img` | Imagen corregida en formato de color BGR (disponible después de `apply_color_correction()`) |

<br>

---

## 4. Métodos

!!! example ""
    Todos los métodos tienen valores por defecto para sus parámetros, así que puedes empezar de forma sencilla y ajustar sobre la marcha.

### `load_image`

Carga la imagen que quieres corregir.

```python
cc.load_image()
cc.load_image(plot=True, plot_size=(8, 8), show_axis=True)
```

<br>

| Parámetro | Tipo | Valor por defecto | Descripción |
|-----------|------|-------------------|-------------|
| `plot` | `bool` | `True` | Muestra la imagen cargada |
| `plot_size` | `tuple[int, int]` | `(5, 5)` | Tamaño de la figura |
| `show_axis` | `bool` | `False` | Muestra los ejes en la gráfica |

<br>

---

### `detect_color_checker`

Busca la tarjeta de color en la imagen y lee el color de cada uno de sus 24 parches. Estos colores medidos son la base sobre la que se construye la corrección.

??? note "Notas"

    - Si la tarjeta no se detecta, revisa que esté completamente visible, enfocada y sin reflejos fuertes ni sombras.

    - Si en la imagen aparece más de una tarjeta, solo se usa la primera que se encuentre.

```python
# Detectar la tarjeta y mostrarla
cc.detect_color_checker(plot=True)

# Ejecutar sin mensajes
cc.detect_color_checker(verbose=False)
```

<br>

| Parámetro | Tipo | Valor por defecto | Descripción |
|-----------|------|-------------------|-------------|
| `plot` | `bool` | `False` | Si es `True`, muestra un recorte de la tarjeta detectada |
| `plot_size` | `tuple[int, int]` | `(5, 5)` | Tamaño de la figura (solo si `plot=True`) |
| `verbose` | `bool` | `True` | Si es `True`, imprime el resultado de la detección y las coordenadas de la tarjeta |

!!! warning "Importante"
    **Requiere** haber ejecutado `load_image()` antes.

<br>

---

### `apply_color_correction`

Corrige los colores de toda la imagen usando la tarjeta de color detectada por `detect_color_checker()`. La imagen corregida se guarda en `cc.corrected_img`.

Los valores por defecto funcionan bien en la mayoría de los casos. Si cambias `degree` o `num_components`, ten en cuenta que el modelo se construye con solo 24 parches: los ajustes más complejos pueden hacer que la tarjeta se vea mejor mientras el resto de la imagen se distorsiona. Cuando los modifiques, revisa siempre el resultado a simple vista y con `calculate_delta_e_stats()`.

??? note "Cómo elegir `degree` y `num_components`"

    - `degree` controla qué tan flexible es la corrección. Los valores altos pueden corregir cambios de color más complejos, pero también aumentan el riesgo de sobreajuste a la tarjeta.

    - `num_components` no puede ser mayor que el número de términos que genera `degree`: **10** con `degree=2` y **20** con `degree=3`. Con el valor por defecto `num_components=11` debes usar `degree=3` o más; si usas `degree=2`, baja `num_components` a 10 o menos.

```python
# Corrección con valores por defecto
cc.apply_color_correction()

# Sin gráficas ni mensajes en consola
cc.apply_color_correction(plot=False, verbose=False)

# Corrección más simple
cc.apply_color_correction(degree=2, num_components=8)
```

<br>

| Parámetro | Tipo | Valor por defecto | Descripción |
|-----------|------|-------------------|-------------|
| `degree` | `int` | `3` | Flexibilidad de la corrección; los valores altos permiten correcciones más complejas |
| `num_components` | `int` | `11` | Número de componentes que usa el modelo; no puede superar 10 con `degree=2` ni 20 con `degree=3` |
| `max_iterations` | `int` | `1000` | Número máximo de iteraciones al ajustar el modelo |
| `plot` | `bool` | `True` | Si es `True`, muestra la imagen original y la corregida lado a lado |
| `plot_size` | `tuple[int, int]` | `(8, 5)` | Tamaño de la figura (solo si `plot=True`) |
| `verbose` | `bool` | `True` | Si es `True`, imprime mensajes de progreso |

!!! warning "Importante"
    **Requiere** haber ejecutado `load_image()` y `detect_color_checker()` antes. Las imágenes grandes pueden tardar unos segundos en procesarse.

<br>

---

### `calculate_delta_e_stats`

*Opcional*

Mide qué tan cerca están los colores de la tarjeta de sus valores de referencia, antes y después de la corrección. La medida es la **diferencia de color ΔE (CIE 2000)**: los valores bajos indican una mejor coincidencia, y 0 es una coincidencia perfecta.

El resultado se calcula para cada uno de los 24 parches y queda guardado para exportarlo con `save_csv()`. Con `verbose=True`, se imprime en consola una tabla resumen con el ΔE medio antes y después, y la mejora por parche.

??? note "Notas"
    - La tarjeta también debe detectarse en la imagen **corregida** para calcular los valores "después". Si la corrección es muy fuerte y la tarjeta ya no se encuentra, este paso fallará.

    - Úsalo para comparar distintos ajustes de `apply_color_correction()` sobre la misma imagen.

```python
cc.calculate_delta_e_stats(verbose=True)
```

<br>

| Parámetro | Tipo | Valor por defecto | Descripción |
|-----------|------|-------------------|-------------|
| `verbose` | `bool` | `True` | Si es `True`, imprime en consola el ΔE medio y una tabla por parche |

!!! warning "Importante"
    **Requiere** haber ejecutado `apply_color_correction()` antes.

<br>

---

### `save_img`

Guarda la imagen corregida. Por defecto, el archivo se guarda en la misma carpeta que la imagen de entrada, con el nombre original más el sufijo `_corrected`.

```python
cc.save_img()
cc.save_img(output_path='ruta/de/salida/', base_name='mi_imagen', format='png')
```

<br>

| Parámetro | Tipo | Valor por defecto | Descripción |
|-----------|------|-------------------|-------------|
| `output_path` | `str` | `None` | Carpeta de salida. Si es `None`, usa la misma carpeta que la imagen de entrada |
| `base_name` | `str` | `None` | Nombre base del archivo. Si es `None`, usa el nombre original |
| `format` | `str` | `'png'` | Formato de la imagen guardada |
| `quality` | `int` | `100` | Calidad de la imagen; relevante para formatos comprimidos como JPEG |
| `verbose` | `bool` | `True` | Si es `True`, imprime la ruta del archivo guardado |

!!! warning "Importante"
    **Requiere** haber ejecutado `apply_color_correction()` antes.

<br>

---

### `save_csv`

*Opcional*

Guarda en un archivo CSV las diferencias de color (ΔE) de cada parche. Por defecto, el archivo se guarda en la misma carpeta que la imagen de entrada, con el nombre original más el sufijo `_delta_e_stats`.

El archivo tiene una fila por parche con las siguientes columnas:

| Columna | Descripción |
|---------|-------------|
| `Patch` | Posición del parche en la tarjeta (por ejemplo, `A1`) |
| `Color` | Nombre del parche (por ejemplo, `dark skin`) |
| `DeltaE_Before` | Diferencia de color con la referencia antes de la corrección |
| `DeltaE_After` | Diferencia de color con la referencia después de la corrección |
| `DeltaE_Improvement` | `DeltaE_Before` menos `DeltaE_After`; los valores positivos indican que el color se acercó a la referencia |

```python
cc.save_csv()
cc.save_csv(output_path='ruta/de/salida/', sep=';')
```

<br>

| Parámetro | Tipo | Valor por defecto | Descripción |
|-----------|------|-------------------|-------------|
| `output_path` | `str` | `None` | Carpeta de salida. Si es `None`, usa la misma carpeta que la imagen de entrada |
| `base_name` | `str` | `None` | Nombre base del archivo. Si es `None`, usa el nombre original |
| `sep` | `str` | `','` | Separador de columnas |
| `verbose` | `bool` | `True` | Si es `True`, imprime un mensaje cuando no hay resultados que guardar |

!!! warning "Importante"
    **Requiere** haber ejecutado `calculate_delta_e_stats()` antes.

<br>

---

### `save_parameters`

*Opcional*

Exporta los **parámetros de corrección de la sesión actual** en formato `.txt` y `.json`, listos para revisar, reutilizar y reproducir.

* `<nombre_imagen>_parameters.txt`: versión legible para revisar.
* `<nombre_imagen>_parameters.json`: versión estructurada para uso programático.

Ambos se guardan por defecto en la misma carpeta que la imagen de entrada, o en la carpeta indicada con `output_path`. El archivo `.json` se puede reutilizar en el procesamiento por lotes con `analyze_folder(json_path=...)`.

```python
cc.save_parameters()
```

<br>

| Parámetro | Tipo | Valor por defecto | Descripción |
|-----------|------|-------------------|-------------|
| `output_path` | `str` | `None` | Carpeta de salida. Si es `None`, usa la misma carpeta que la imagen de entrada |

!!! warning "Importante"
    **Requiere** haber ejecutado `apply_color_correction()` antes.

<br>

---

### `analyze_folder`

Corrige por lotes todas las imágenes de la carpeta indicada al inicializar `ColorCorrection`, ya sea de forma secuencial (`num_cores=1`) o en paralelo (`num_cores` > 1). Cada imagen se corrige con **su propia** tarjeta de color, así que todas las imágenes de la carpeta deben incluir una.

Por cada imagen se guarda una copia corregida con el sufijo `_corrected`. Cuando `delta_e=True`, las diferencias de color de todas las imágenes se reúnen en un solo archivo:

* `delta_e_results.csv`: ΔE por parche, antes y después de la corrección, de todas las imágenes.

Siempre se genera un `session_report.txt` con el resumen de la sesión (imágenes procesadas, tiempos y parámetros usados). Si alguna imagen falla durante el procesamiento (por ejemplo, porque no se detectó la tarjeta), también se genera un `error_report.txt` que detalla qué pasó en cada caso.

Todos los archivos se guardan en la carpeta indicada con `output_path`. Si no se indica, se guardan en una subcarpeta `Results/` dentro de la carpeta de entrada.

??? note "Nota"
    Para mayor comodidad y reproducibilidad, te recomendamos probar los parámetros en una imagen representativa, guardarlos con `save_parameters()` y luego pasar el archivo `.json` generado con `json_path`.

```python
# Usando los parámetros por defecto
cc.analyze_folder()

# Usando parámetros individuales
cc.analyze_folder(num_cores=4, delta_e=False)

# Usando un archivo de parámetros guardado
cc.analyze_folder(json_path="parametros_imagen.json")
```

<br>

| Parámetro | Tipo | Valor por defecto | Descripción |
|-----------|------|-------------------|-------------|
| `delta_e` | `bool` | `True` | Si es `True`, calcula las diferencias de color (ΔE) de cada imagen y las guarda en `delta_e_results.csv` |
| `json_path` | `str` | `None` | Ruta de un archivo de parámetros `.json` generado con `save_parameters()` |
| `output_path` | `str` | `None` | Carpeta de salida. Si es `None`, se crea una subcarpeta `Results/` dentro de la carpeta de entrada |
| `num_cores` | `int` | `1` | Número de procesos en paralelo. Se limita automáticamente a los núcleos disponibles |
| `verbose` | `bool` | `True` | Si es `True`, imprime el progreso y el resumen de la sesión |
| `degree` | `int` | `None` | Flexibilidad de la corrección -> `apply_color_correction` |
| `num_components` | `int` | `None` | Número de componentes que usa el modelo -> `apply_color_correction` |
| `max_iterations` | `int` | `None` | Número máximo de iteraciones al ajustar el modelo -> `apply_color_correction` |

!!! warning "Importante"
    **Requiere** que `ColorCorrection()` se haya inicializado con la ruta de una carpeta, no de un archivo.
