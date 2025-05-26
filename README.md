# Project Installation & Test Instructions

This project is distributed as a `.zip` archive and uses a `requirements.txt` file to manage dependencies. Follow the instructions below to install the environment and run the tests on **Windows, macOS, or Linux**.

## Prerequisites

- Python **3.7 or higher**
- `pip` (Python package installer)

Check that Python and pip are available:

```bash
python --version
pip --version
```

> On some systems, use `python3` and `pip3` instead of `python` and `pip`.

---

## Installation Steps

### 1. Extract the ZIP Archive

- **Windows**: Right-click the `.zip` → “Extract All…”
- **macOS/Linux**:

```bash
unzip your_project.zip -d your_project_folder
cd your_project_folder
```

### 2. (Optional but recommended) Create and Activate a Virtual Environment

#### Windows

```cmd
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Tests

Run the following test commands **in order**, each corresponding to a different section:

```bash
python3 -m pytest -v -k "forward"
python3 -m pytest -l -v -k "backward"
python3 -m pytest -k "topo_sort"
python3 -m pytest -k "compute_gradient"
```

> On **Windows**, use `python` instead of `python3` if `python3` is not recognized.

---

## Troubleshooting

- If you see "command not found" or similar errors, ensure Python and pip are added to your system's PATH.
- If dependencies are missing, make sure the virtual environment is activated and `pip install -r requirements.txt` ran successfully.
- If you encounter permission issues on Unix systems, prepend commands with `python3 -m` rather than calling `pytest` directly.

---
