

#  Instalación de Traitly
*Última actualización: Febrero 2026 - Traitly v0.1.0*

Available in: [![English](https://img.shields.io/badge/Language-English-purple)](installation.md)

---

Antes de comenzar, te recomendamos ampliamente crear un **entorno limpio** para evitar conflictos con librerías preexistentes. Esto garantiza una instalación reproducible.

Para los propósitos de este tutorial, crearemos un **entorno virtual de Python (venv)**, pero puedes usar cualquier otro gestor de entornos de tu preferencia (conda, pixi, etc.).

---

## Requisitos

- Python 3.8 o superior ([descargar aquí](https://www.python.org/downloads/))
- `pip` (incluido con Python)
- `git` instalado en tu sistema ([descargar aquí](https://git-scm.com/downloads))
- RAM: 4 GB (recomendado 8 GB)
- Espacio libre en el disco: ~2 GB para la instalación completa

> **💡 Nota (solo Windows):** Al ejecutar el instalador de Python, asegúrate de marcar la opción **"Add Python to PATH"** para evitar problemas en los pasos siguientes.

Opcionalmente, si prefieres no instalar nada en tu equipo o no cuentas con los requisitos necesarios, también puedes usar Traitly de manera remota en [Google Colab](https://colab.research.google.com/) (requiere cuenta de Google). La cuenta gratuita generalmente incluye ~12 GB de RAM y ~100 GB de espacio local en el disco. Puedes ver mas detalles en la sección [Google Colab](#google-colab).

---

## Antes de Comenzar: 

Todos los comandos de este tutorial deben ejecutarse en una **terminal** (también llamada línea de comandos o consola). Aquí te mostramos cómo accesar según tu sistema operativo:

- **MacOS:** Abre la app `Terminal` (encuéntrala con Spotlight usando `Cmd + Space` y escribiendo "Terminal")
- **Linux:** Abre la app `Terminal` desde el menú de aplicaciones, o presiona `Ctrl + Alt + T`
- **Windows:** Abre `PowerShell` (búscalo en el menú de inicio)

> Para más información sobre cómo usar la terminal, aquí hay dos tutoriales para comenzar:
> - [Introducción a PowerShell](https://programminghistorian.org/en/lessons/intro-to-powershell) (Windows)
> - [Introducción a la línea de comandos](https://ryanstutorials.net/linuxtutorial/) (MacOS y Linux)

<br>

Una vez abierta la terminal, ve a la carpeta donde deseas alojar tu proyecto. Aquí es donde te recomendamos guardar tu entorno virtual y tus archivos. Por ejemplo:

```bash
cd ~/Documents/my-project
```

> Si la carpeta aún no existe, puedes crearla primero:
> ```bash
> mkdir ~/Documents/my-project
> cd ~/Documents/my-project
> ```

La carperta de tu proyecto se verá algo así:
```
mi-proyecto/
├── traitly-env/       <- entorno virtual (se crea en los siguientes pasos)
├── mi_notebook.ipynb
├── imagenes/
└── ...
```

> **💡 Tip (Windows):** Reemplaza `~` con la ruta completa de tu usuario, por ejemplo `C:\Users\TuNombre\Documents\my-project`, o simplemente haz clic derecho en la carpeta en el Explorador de archivos y selecciona **"Abrir en Terminal"** para saltarte el paso de navegación.

---

## MacOS y Linux

**1. Crear un nuevo entorno de Python:**
```bash
python -m venv traitly-env
```

**2. Activar el entorno:**
```bash
source traitly-env/bin/activate
```

> Si la activación fue exitosa, aparecerá `(traitly-env)` al inicio de tu terminal.

**3. Instalar Traitly:**
```bash
pip install git+https://github.com/mariameraz/traitly.git
```

---

## Windows


**1. Crear un nuevo entorno de Python:**
```bash
python -m venv traitly-env
```

**2. Activar el entorno:**
```bash
traitly-env\Scripts\activate
```

> Si la activación fue exitosa, aparecerá `(traitly-env)` al inicio de tu terminal.

**3. Instalar Traitly:**
```bash
pip install git+https://github.com/mariameraz/traitly.git
```

---

## Google Colab

> ⚠️ **Nota importante:** Google Colab no guarda sesiones. Cada vez que cierres el navegador o la sesión expire, tendrás que volver a instalar Traitly desde el paso 3.

**1. Crear una cuenta o iniciar sesión:**
Ve a [colab.research.google.com](https://colab.research.google.com/) e inicia sesión con tu cuenta de Google. Si no tienes una, puedes crear una gratis en [accounts.google.com](https://accounts.google.com/).

**2. Crear un nuevo notebook:**
Una vez dentro, ve a `File -> New notebook in Drive`. Esto abrirá un nuevo documento de Jupyter en tu navegador.

**3. Instalar Traitly:**
En la primera celda, copia y ejecuta el siguiente comando:
```python
!pip install "git+https://github.com/mariameraz/traitly.git"
```

---

## Dependencias Opcionales

Por defecto, Traitly instalará únicamente el **paquete base**. Dependiendo de tu caso de uso, puede que necesites instalar dependencias adicionales:

| Extra | Uso | Comando |
|-------|-----|---------|
| `app` | Ejecutar la aplicación de Streamlit | `pip install "git+...traitly.git[app]"` |
| `pdf` | Convertir archivos PDF a imágenes | `pip install "git+...traitly.git[pdf]"` |
| `all` | Instalar todas las dependencias opcionales | `pip install "git+...traitly.git[all]"` |

Por ejemplo, si quieres usar la aplicación de Streamlit:
```bash
pip install "git+https://github.com/mariameraz/traitly.git[app]"
```

Si quieres instalar todo de una vez:
```bash
pip install "git+https://github.com/mariameraz/traitly.git[all]"
```

> **⚠️ Importante:** Las comillas alrededor de la URL son necesarias en MacOS y Linux para evitar que la terminal interprete incorrectamente los caracteres `[` y `]`. En Windows son opcionales, pero se recomiendan por consistencia.

---

## Verificar la Instalación

Una vez instalado, puedes confirmar que todo funciona correctamente ejecutando en la terminal:

```bash
python -c "import traitly; print(traitly.__version__)"
```
o
```bash
traitly --version
```

Si la instalación fue exitosa, deberás ver el número de versión instalada en la terminal.

---

## Desactivar el Entorno

Cuando termines de trabajar, puedes desactivar el entorno con:
```bash
deactivate
```

---

## Solución de Problemas

**1. `python` no encontrado:**
Intenta usar `python3` en su lugar (común en MacOS y Linux):
```bash
python3 -m venv traitly-env
```
En Windows, también puedes intentar con `py` (disponible a través del Python Launcher en la mayoría de instalaciones modernas):
```bash
py -m venv traitly-env
```
> Nota: Puede que `py` no esté disponible en versiones antiguas (por ejemplo, Python 3.6 en Windows 7). En ese caso, recomendamos reinstalar Python desde [python.org](https://www.python.org/downloads/) y marcar **"Add Python to PATH"** durante la instalación.

<br>

**2. `git` no encontrado:**
Asegúrate de que `git` esté instalado y disponible en tu PATH. Puedes verificarlo con:
```bash
git --version
```

<br>

**3. `pip` no encontrado después de activar el entorno:**
Intenta usar `pip3` en lugar de `pip`, o reinstala Python asegurándote de marcar la opción "Add to PATH" durante la instalación (Windows).

<br>

**4. Errores de permisos en MacOS/Linux:**
Evita usar `sudo pip install`. En cambio, asegúrate de que tu entorno virtual esté correctamente activado antes de instalar.