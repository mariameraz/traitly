# How to Install Traitly

*Last updated: February 2026 - Traitly v0.1.0*

Disponible en: [![Spanish](https://img.shields.io/badge/Idioma-Espa%C3%B1ol-pink)](installation_ES.md)

---

Before getting started, it is strongly recommended to create a **clean environment** to avoid conflicts with pre-existing libraries. This ensures a reproducible setup.

For the purposes of this tutorial, we will be using a **Python virtual environment (venv)**, but feel free to use any other environment manager of your choice (conda, pixi, etc.).

---

## Requirements

- Python 3.8 or higher ([download here](https://www.python.org/downloads/))
- `pip` (comes included with Python)
- `git` installed on your system ([download here](https://git-scm.com/downloads))
- RAM: 4 GB (8 GB recommended)
- Disk space: ~2 GB for the full installation

> **💡 Note (Windows only):** When running the Python installer, make sure to check the **"Add Python to PATH"** option to avoid issues with the following steps.

Optionally, if you prefer not to install anything on your computer or do not meet the system requirements, you can also use Traitly remotely on [Google Colab](https://colab.research.google.com/) (requires a Google account). Usually, the free tier includes ~12 GB of RAM and ~100 GB of local disk space. See the [Google Colab](#google-colab) section for more details.

---

## Before You Start:

All the commands in this tutorial need to be run in a **terminal** (also called command line or console). Here is how to open it depending on your OS:

- **MacOS:** Open the `Terminal` app (find it via Spotlight with `Cmd + Space` and typing "Terminal")
- **Linux:** Open the `Terminal` app from your applications menu, or press `Ctrl + Alt + T`
- **Windows:** Open `PowerShell` (search for it in the Start menu)

> For more information on how to work in the terminal, here are two tutorials to get you started:
> - [Introduction to PowerShell](https://programminghistorian.org/en/lessons/intro-to-powershell) (Windows)
> - [Introduction to the command line](https://ryanstutorials.net/linuxtutorial/) (MacOS and Linux)

<br> 

Once the terminal is open, navigate to the folder where you want to set up your project. This is where we recommend keeping your virtual environment and the files you will work with. For example:

```bash
cd ~/Documents/my-project
```

> If the folder doesn't exist yet, you can create it first:
> ```bash
> mkdir ~/Documents/my-project
> cd ~/Documents/my-project
> ```

Your project folder will end up looking like this:
```
my-project/
├── traitly-env/       <- virtual environment (created in the next steps)
├── my_notebook.ipynb
├── my_images/
└── ...
```

> **💡 Tip (Windows):** Replace `~` with your full user path, e.g. `C:\Users\YourName\Documents\my-project`, or simply right-click the folder in File Explorer and select **"Open in Terminal"** to skip the navigation step entirely.

---

## MacOS and Linux

**1. Create an empty Python environment:**
```bash
python -m venv traitly-env
```

**2. Activate the environment:**
```bash
source traitly-env/bin/activate
```

> You should see `(traitly-env)` at the beginning of your terminal prompt, confirming the environment is active.

**3. Install Traitly:**
```bash
pip install git+https://github.com/mariameraz/traitly.git
```

---

## Windows

**1. Create an empty Python environment:**
```bash
python -m venv traitly-env
```

**2. Activate the environment:**
```bash
traitly-env\Scripts\activate
```

> You should see `(traitly-env)` at the beginning of your terminal prompt, confirming the environment is active.

**3. Install Traitly:**
```bash
pip install git+https://github.com/mariameraz/traitly.git
```

--- 

## Google Colab

> ⚠️ **Important note:** Google Colab does not save sessions. Every time you close the browser or the session expires, you will need to reinstall Traitly starting from step 3.

**1. Create an account or sign in:**
Go to [colab.research.google.com](https://colab.research.google.com/) and sign in with your Google account. If you don't have one, you can create one for free at [accounts.google.com](https://accounts.google.com/).

**2. Create a new notebook:**
Once inside, go to `File -> New notebook in Drive`. This will open a new Jupyter document in your browser.

**3. Install Traitly:**
In the first cell, copy and run the following command:
```python
!pip install "git+https://github.com/mariameraz/traitly.git"
```

---

## Optional Dependencies

By default, Traitly installs only the **core package**. Depending on your use case, you may need to install additional extras:

| Extra | Use case | Command |
|-------|----------|---------|
| `app` | Run the Streamlit application | `pip install "git+...traitly.git[app]"` |
| `pdf` | Convert PDF files to images | `pip install "git+...traitly.git[pdf]"` |
| `all` | Install all optional dependencies | `pip install "git+...traitly.git[all]"` |

For example, if you want to use the Streamlit app:
```bash
pip install "git+https://github.com/mariameraz/traitly.git[app]"
```

If you want to install everything at once:
```bash
pip install "git+https://github.com/mariameraz/traitly.git[all]"
```

> **⚠️ Important:** The quotes around the URL are required on MacOS and Linux to prevent the shell from misinterpreting the `[` and `]` characters. On Windows they are optional but recommended for consistency.

---

## Verify the Installation

Once installed, confirm everything is working correctly by running in the terminal:
```bash
python -c "import traitly; print(traitly.__version__)"
```
or

```bash
traitly --version
```

If the installation was successful, you should see the installed version number printed in the terminal.

---

## Deactivating the Environment

When you are done working, you can deactivate the environment with:
```bash
deactivate
```

---

## Troubleshooting

**1. `python` not found:**
Try using `python3` instead (common on MacOS and Linux):
```bash
python3 -m venv traitly-env
```
On Windows, you may also try `py` (available via the Python Launcher on most modern installations):
```bash
py -m venv traitly-env
```
> Note: `py` may not be available on older setups (e.g. Python 3.6 on Windows 7). In that case, reinstalling Python from [python.org](https://www.python.org/downloads/) and checking **"Add Python to PATH"** during setup is recommended.

<br>

**2. `git` not found:**
Make sure `git` is installed and available in your PATH. You can verify with:
```bash
git --version
```
<br>

**3. `pip` not found after activating the environment:**
Try using `pip3` instead of `pip`, or reinstall Python making sure to check the "Add to PATH" option during setup (Windows).

<br>

**4. Permission errors on MacOS/Linux:**
Avoid using `sudo pip install`. Instead, make sure your virtual environment is properly activated before installing.
