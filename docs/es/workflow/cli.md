<div class="animate" markdown>

# CLI: Interfaz de Línea de Comandos

Traitly incluye una interfaz de línea de comandos que permite ejecutar los análisis directamente desde la terminal, sin necesidad de escribir código en Python. Es especialmente útil para procesar lotes de imágenes en servidores o entornos de cómputo donde no se trabaja con notebooks o scripts interactivos.

---

## Uso básico

```bash
traitly --fruit_internal -i RUTA [-o RUTA] [--json RUTA] [--num_cores N]
traitly --fruit_external -i RUTA [-o RUTA] [--json RUTA] [--num_cores N]
```

La entrada (`-i`) puede ser una imagen individual o una carpeta con múltiples imágenes. En ambos casos, se delega automáticamente al método correspondiente (`process_single_file` o `analyze_folder`).

!!! tip "Un archivo JSON es requerido para configurar el análisis"
    
    **Toda la configuración del análisis debe pasarse mediante un archivo `.json` con `--json`.** 

    Hay dos formas de obtenerlo:

    **Opción 1 – Desde Python:**

    1. Ajusta los parámetros sobre una imagen representativa
    2. Guarda la configuración con `analyzer.save_parameters()`
    3. Pasa el archivo `.json` generado con `--json` en la terminal
   
    **Opción 2 — Desde una plantilla:**

    Copia el archivo base, modifica los parámetros que necesites en cualquier editor de texto y guárdalo como `.json`. Luego pásalo con `--json`.

    [:octicons-file-code-24: Ver plantilla JSON](../assets/templates/parameters_template.json)

    Para conocer qué hace cada parámetro, consulta la documentación de [`FruitInternalAnalyzer`](internal_class.md) o [`FruitExternalAnalyzer`](external_class.md).

<br>

---

## Argumentos

| Argumento | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--fruit_internal` | `flag` | — | Activa el análisis de estructura interna (`FruitInternalAnalyzer`) |
| `--fruit_external` | `flag` | — | Activa el análisis de apariencia externa (`FruitExternalAnalyzer`) |
| `-i`, `--input` | `str` | — | Ruta a la imagen o carpeta a analizar (**requerido**) |
| `-o`, `--output` | `str` | `None` | Directorio de salida; si `None`, se crea una subcarpeta `Results/` dentro de la carpeta de entrada |
| `--json` | `str` | `None` | Ruta al archivo `.json` de parámetros generado por `save_parameters()` |
| `--num_cores` | `int` | `1` | Número de núcleos de CPU para procesamiento en paralelo |
| `--no_morphology` | `flag` | — | Omite el análisis morfológico |
| `--no_color` | `flag` | — | Omite el análisis de color |
| `--version` | `flag` | — | Muestra la versión instalada de Traitly |
| `--help` | `flag` | — | Muestra información sobre los parámetros disponibles y ejemplos de uso |

!!! warning "Importante"
    `--fruit_internal` y `--fruit_external` son mutuamente excluyentes, solo puede usarse uno por llamada.

<br>

---

## Ejemplos

**Análisis interno sobre una carpeta, usando parámetros guardados:**
```bash
traitly --fruit_internal -i datos/frutos_cortes/ --json config.json
```

**Análisis externo con salida personalizada y procesamiento en paralelo:**
```bash
traitly --fruit_external -i datos/frutos_enteros/ -o resultados/ --num_cores 4
```

**Análisis sobre una imagen individual:**
```bash
traitly --fruit_internal -i datos/imagen_001.jpg --json config.json
```

**Solo morfología, sin color:**
```bash
traitly --fruit_internal -i datos/cortes/ --json config.json --no_color
```

**Verificar versión instalada:**
```bash
traitly --version
```

**Ver parámetros disponibles:**
```bash
traitly --help
```

<br>

---

## Salidas

El CLI genera los mismos archivos que `analyze_folder()` desde Python:

| Archivo | Descripción |
|---------|-------------|
| `morphology_results.csv` | Métricas morfológicas de todos los frutos detectados |
| `color_results.csv` | Métricas de color de todos los frutos detectados |
| `session_report.txt` | Resumen de la sesión: imágenes procesadas, frutos detectados, tiempos y parámetros utilizados |
| `error_report.txt` | Detalle de errores por imagen (solo si alguna imagen falló) |
| Imágenes anotadas | Una imagen anotada por cada imagen procesada |

Todos los archivos se guardan en el directorio indicado por `-o`, o en una subcarpeta `Results/` dentro de la carpeta de entrada si no se especifica.

</div>