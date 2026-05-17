<!-- ---
hide:
  - navigation
  - toc
--- -->

<div class="animate" markdown>

#  Instalando Traitly ⊹ ࣪ ˖

*Última actualización: Marzo 2026 - Traitly v0.1.0*

---

Antes de comenzar, te recomendamos ampliamente crear un **entorno limpio** para evitar conflictos con librerías preexistentes. Esto garantiza una instalación reproducible. Si ya tienes experiencia con entornos virtuales, puedes saltar directamente a [Instalación](#instalacion).

Para los propósitos de este tutorial, crearemos un **entorno virtual de Python (venv)**, pero puedes usar cualquier otro gestor de entornos de tu preferencia (conda, pixi, etc.).

---

## Requisitos

- Python 3.9 o superior ([descargar aquí](https://www.python.org/downloads/))
- RAM: 4 GB (recomendado 8 GB)
- Espacio libre en el disco: ~2 GB para la instalación completa

!!! note "Nota"
    En **Windows**, al ejecutar el instalador de Python, asegúrate de marcar la opción **"Add Python to PATH"** para evitar problemas en los pasos siguientes.

Opcionalmente, si prefieres no instalar nada en tu equipo o no cuentas con los requisitos necesarios, también puedes usar Traitly de manera remota en [Google Colab](https://colab.research.google.com/) (requiere 
cuenta de Google). La cuenta gratuita generalmente incluye ~12 GB de RAM y ~100 GB de espacio local en el disco. En ese caso, puedes saltar directamente a la pestaña **Google Colab** en la sección de [instalación](#instalacion).

---

## Antes de Comenzar

Todos los comandos de este tutorial deben ejecutarse en una **terminal** (también llamada línea de comandos o consola). Aquí te mostramos cómo accesar según tu sistema operativo:

- **MacOS:** Abre la app `Terminal` (encuéntrala con Spotlight usando `Cmd + Space` y escribiendo "Terminal")
- **Linux:** Abre la app `Terminal` desde el menú de aplicaciones, o presiona `Ctrl + Alt + T`
- **Windows:** Abre `PowerShell` (búscalo en el menú de inicio)

??? info "¿Cómo usar la terminal?"
    Para más información sobre cómo usar la terminal, aquí hay dos tutoriales para comenzar:

    - [Introducción a PowerShell](https://programminghistorian.org/en/lessons/intro-to-powershell) (Windows)
    - [Introducción a la línea de comandos](https://ryanstutorials.net/linuxtutorial/) (MacOS y Linux)


Una vez abierta la terminal, ve a la carpeta donde deseas alojar tu proyecto.
Si aún no existe, créala con:
```bash
# Crear carpeta con mkdir 
mkdir ~/Documentos/mi-proyecto

# Ir a la carpeta
cd ~/Documentos/mi-proyecto
```

Aquí es donde guardarás tu entorno virtual y tus archivos.

La carpeta de tu proyecto se verá algo así:
```
mi-proyecto/
├── traitly-env/       <- entorno virtual (se crea en los siguientes pasos)
├── mi_notebook.ipynb
├── imagenes/
└── ...
```

!!! tip "Directorio en Windows"
    Reemplaza `~` con la ruta completa de tu usuario, por ejemplo `C:\Users\TuNombre\Documentos\mi-proyecto`, o simplemente haz clic derecho en la carpeta en el Explorador de archivos y selecciona **"Abrir en Terminal"** para saltarte el paso de navegación.

---

## Instalación

=== ":fontawesome-solid-terminal: MacOS y Linux"

    ??? warning "Usuarios con Mac Intel (anteriores a 2020)"
              
        PyTorch (requerido por EasyOCR) ya no publica paquetes precompilados para 
        Macs con procesador Intel, y las versiones recientes de OpenCV requieren 
        macOS 14 o superior. Por esta razón, la instalación estándar no funcionará en Macs con Intel más antiguos.
    
        ??? tip "¿No sabes si tienes Intel?"
            *Para verificar si tienes una Mac con procesador Intel, ejecuta en la terminal:*

            ```bash
            uname -m 
            ```
            o haz click en  -> **Acerca de esta Mac**. Si imprime `x86_64` o dice **Intel** en Procesador, sigue estos pasos en lugar de la instalación estándar.
            
        **1. Crea el ambiente con Python 3.11**
    
        Asegúrate de tener Python 3.11 instalado ([descárgalo aquí](https://www.python.org/downloads/release/python-3111/)). Luego ejecuta:
        
        ```bash
        python3.11 -m venv traitly-env
        source traitly-env/bin/activate
        ```
              
        **2. Instala las dependencias en orden:**
    
        ```bash
        pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
        pip install opencv-python==4.8.1.78
        pip install opencv-python-headless==4.8.1.78
        pip install opencv-contrib-python==4.9.0.80
        pip install "numpy<2"
        ```
        **3. Instala traitly**
    
        Obtén la última versión:
        ```bash
        pip install traitly
        ```
    
        O una versión específica:
        ```bash
        pip install traitly==0.1.1
        ```

    **1. Crear un nuevo entorno de Python:**

    Puedes reemplazar `traitly-env` con el nombre que prefieras.
    ```bash
    python3 -m venv traitly-env
    ```

    **2. Activar el entorno:**
    ```bash
    source traitly-env/bin/activate
    ```

    Si la activación fue exitosa, aparecerá `(traitly-env)` al inicio de tu terminal.

    **3. Instalar Traitly:**
  
    Obtén la última versión:
    ```bash
    pip install traitly
    ```

    O una versión específica:
    ```bash
    pip install traitly==0.1.1
    ```

=== ":fontawesome-brands-windows:{.icon-blue} Windows"

    **1. Crear un nuevo entorno de Python:**
    
    Puedes reemplazar `traitly-env` con el nombre que prefieras.
    ```bash
    py -m venv traitly-env
    ```

    **2. Activar el entorno:**
    ```bash
    traitly-env\Scripts\activate
    ```

    Si la activación fue exitosa, aparecerá `(traitly-env)` al inicio de tu terminal.

    **3. Instalar Traitly:**

    Obtén la última versión:
    ```bash
    pip install traitly
    ```

    O una versión específica:
    ```bash
    pip install traitly==0.1.1
    ```

=== ":simple-googlecolab:{.icon-orange} Google Colab"

    !!! warning "Nota importante"
        Google Colab no guarda sesiones. Cada vez que cierres el navegador o la sesión expire, tendrás que volver a instalar Traitly desde el paso 3.

    **1. Crear una cuenta o iniciar sesión:**
    Ve a [colab.research.google.com](https://colab.research.google.com/) e inicia sesión con tu cuenta de Google. Si no tienes una, puedes crear una gratis en [accounts.google.com](https://accounts.google.com/).

    **2. Crear un nuevo notebook:**
    Una vez dentro, ve a `File -> New notebook in Drive`. Esto abrirá un nuevo documento de Jupyter en tu navegador.

    **3. Instalar Traitly:**
    En la primera celda, copia y ejecuta el siguiente comando:

    Obtén la última versión:
    ```bash
    !pip install traitly
    ```

    O una versión específica:
    ```bash
    !pip install traitly==0.1.1
    ```

---

## Dependencias Opcionales

Por defecto, Traitly instalará únicamente el **paquete base**. Dependiendo de tu caso de uso, puede que necesites instalar dependencias adicionales:

| Extra | Incluye                             | Comando                           |
|-------|-------------------------------------|-----------------------------------|
| `pdf` | Convertir archivos PDF a imágenes   | `pip install "traitly[pdf]"` |
| `app` | Convertir PDF + Shiny app           | `pip install "traitly[app]"` |

!!! tip "¿No sabes cuál elegir?"
    Si tienes dudas, te recomendamos instalar todas las dependencias con `app`, con las cuales puedes hacer uso de todas las funcionalidades de Traitly.
    
!!! warning "Importante"
    Las comillas alrededor de la URL de Git son necesarias en MacOS y Linux para evitar que la terminal interprete incorrectamente los caracteres `[` y `]`. En Windows son opcionales, pero se recomiendan por consistencia.

---

## Verificar la Instalación

Una vez instalado, puedes confirmar que todo funciona correctamente ejecutando:

**Desde la terminal:**
```bash
traitly --version
```

**Desde Python o Jupyter Notebook:**
```python
import traitly
print(traitly.__version__)
```

Si la instalación fue exitosa, deberás ver el número de versión instalada en la consola.

---

## Desactivar el Entorno

Cuando termines de trabajar, puedes desactivar el entorno con:
```bash
deactivate
```

---

## Solución de Problemas

**1. `pip` no encontrado después de activar el entorno:**
Intenta usar `pip3` en lugar de `pip`:

```bash
pip3 install traitly
```

O reinstala Python asegurándote de marcar la opción "Add to PATH" durante la instalación (**Windows**).


**2. Errores de permisos en MacOS/Linux:**
Evita usar `sudo pip install`. En cambio, asegúrate de que tu entorno virtual esté correctamente activado antes de instalar.

</div>
