<div class="animate" markdown>

# Installating Traitly ⊹ ࣪ ˖

*Last updated: March 2026 - Traitly v0.1.0*

---

Before you begin, we highly recommend creating a **clean environment** to avoid conflicts with pre-existing libraries. This ensures a reproducible installation. If you're already familiar with virtual environments, you can skip directly to [Installation](#installation).

For this tutorial, we'll create a **Python virtual environment (venv)**, but you can use any other environment manager of your choice (conda, pixi, etc.).

---

## Requirements

- Python 3.8 or higher ([download here](https://www.python.org/downloads/))
- `pip` (included with Python)
- `git` installed on your system ([download here](https://git-scm.com/downloads))
- RAM: 4 GB (8 GB recommended)
- Free disk space: ~2 GB for full installation

!!! note "Note"
    On **Windows**, when running the Python installer, make sure to check **"Add Python to PATH"** to avoid issues in the following steps.

Optionally, if you prefer not to install anything on your computer or don't meet the requirements, you can also use Traitly remotely on [Google Colab](https://colab.research.google.com/) (requires a Google account). The free tier typically includes ~12 GB RAM and ~100 GB of local disk space. In that case, you can skip directly to the **Google Colab** tab in the [installation](#installation) section.

---

## Before You Begin

All commands in this tutorial should be run in a **terminal** (also called command line or console). Here's how to access it on different operating systems:

- **macOS:** Open the `Terminal` app (find it with Spotlight using `Cmd + Space` and typing "Terminal")
- **Linux:** Open the `Terminal` app from the applications menu, or press `Ctrl + Alt + T`
- **Windows:** Open `PowerShell` (search for it in the Start menu)

??? info "How to use the terminal?"
    For more information on using the terminal, here are two tutorials to get started:
    
    - [Introduction to PowerShell](https://programminghistorian.org/en/lessons/intro-to-powershell) (Windows)
    - [Introduction to the command line](https://ryanstutorials.net/linuxtutorial/) (macOS and Linux)

Once the terminal is open, navigate to the folder where you want to host your project.
If it doesn't exist yet, create it with:
```bash
# Create folder with mkdir
mkdir ~/Documents/my-project

# Navigate to the folder
cd ~/Documents/my-project
```

This is where you'll save your virtual environment and your files.

Your project folder will look something like this:
```
my-project/
├── traitly-env/       <- virtual environment (created in the following steps)
├── my_notebook.ipynb
├── images/
└── ...
```

!!! tip "Windows Directory"
    Replace `~` with your full user path, for example `C:\Users\YourName\Documents\my-project`, or simply right-click on the folder in File Explorer and select **"Open in Terminal"** to skip the navigation step.

---

## Installation

=== ":fontawesome-solid-terminal: macOS and Linux"

    **1. Create a new Python environment:**

    You can replace `traitly-env` with any name you prefer.

    ```bash
    python -m venv traitly-env
    ```

    **2. Activate the environment:**
    ```bash
    source traitly-env/bin/activate
    ```

    If activation was successful, `(traitly-env)` will appear at the beginning of your terminal prompt.

    **3. Install Traitly:**
    ```bash
    pip install git+https://github.com/mariameraz/traitly.git
    ```

=== ":fontawesome-brands-windows:{.icon-blue} Windows"

    **1. Create a new Python environment:**
    
    You can replace `traitly-env` with any name you prefer.
    ```bash
    python -m venv traitly-env
    ```

    **2. Activate the environment:**
    ```bash
    traitly-env\Scripts\activate
    ```

    If activation was successful, `(traitly-env)` will appear at the beginning of your terminal prompt.

    **3. Install Traitly:**
    ```bash
    pip install git+https://github.com/mariameraz/traitly.git
    ```

=== ":simple-googlecolab:{.icon-orange} Google Colab"

    !!! warning "Important Note"
        Google Colab does not save sessions. Every time you close your browser or your session expires, you'll need to reinstall Traitly starting from step 3.

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

By default, Traitly will only install the **base package**. 
Depending on your use case, you might need to install additional dependencies:

| Extra | Includes                   | Command                           |
|-------|----------------------------|-----------------------------------|
| `pdf` | PDF to image conversion    | `pip install "git+https://github.com/mariameraz/traitly.git[pdf]"` |
| `all` | PDF conversion + Shiny app | `pip install "git+https://github.com/mariameraz/traitly.git[all]"` |

!!! tip "Not sure which to choose?"
    If you're unsure, we recommend installing `all`, since it includes everything you need to use Traitly's full functionality, including the interactive app.

!!! warning "Important"
    The quotes around the URL are necessary on macOS and Linux to prevent 
    the terminal from incorrectly interpreting the `[` and `]` characters. 
    On Windows they are optional, but recommended for consistency.
    
---

## Verify Installation

Once installed, you can confirm everything is working correctly by running in the terminal:

```bash
python -c "import traitly; print(traitly.__version__)"
```
or
```bash
traitly --version
```

If the installation was successful, you should see the installed version number in the terminal.

---

## Deactivate the Environment

When you finish working, you can deactivate the environment with:
```bash
deactivate
```

---

## Troubleshooting

**1. `python` not found:**
Try using `python3` instead (common on macOS and Linux):
```bash
python3 -m venv traitly-env
```
On Windows, you can also try with `py` (available through the Python Launcher in most modern installations):
```bash
py -m venv traitly-env
```

!!! Note ""
    `py` might not be available on older versions (e.g., Python 3.6 on Windows 7). In that case, we recommend reinstalling Python from [python.org](https://www.python.org/downloads/) and checking **"Add Python to PATH"** during installation.

**2. `git` not found:**
Make sure `git` is installed and available in your PATH. You can verify it with:
```bash
git --version
```

**3. `pip` not found after activating the environment:**
Try using `pip3` instead of `pip`, or reinstall Python making sure to check the "Add to PATH" option during installation (Windows).

**4. Permission errors on macOS/Linux:**
Avoid using `sudo pip install`. Instead, make sure your virtual environment is properly activated before installing.

</div>
