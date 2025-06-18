# Deep Learning Automatic Differentiation Engine

A minimal, extensible **automatic differentiation engine** for deep learning, machine learning, AI engineering, and scientific computing. This project implements **reverse-mode autodiff (backpropagation)** from scratch in Python and NumPy, inspired by the core principles of **PyTorch** and **TensorFlow**. It demonstrates differentiable programming, computational graph construction, and gradient-based optimization—key skills for AI, ML, and data engineering roles.

---

##  Key Features

- **Reverse-mode automatic differentiation** (backpropagation) for neural networks and general computational graphs
- **Custom Tensor and Operation classes** (similar to PyTorch/TensorFlow autograd)
- **Matrix operations, broadcasting, and advanced tensor manipulations**
- **NumPy-based backend** for high performance and easy integration
- **Modular, extensible, and production-ready design**
- **Educational value:** Understand how modern deep learning frameworks work internally

---

##  Why This Project?

Modern AI, ML, and data engineering workflows rely on automatic differentiation for model training, optimization, and scientific computing. This project demonstrates:

- Differentiable programming and autodiff
- Computational graph construction and traversal
- Gradient-based optimization algorithms
- Software engineering best practices in Python
- Deep understanding of neural network internals

---

##  Use Cases

- Deep learning research and prototyping
- Educational purposes: understanding PyTorch/TensorFlow internals
- Custom AI/ML model development
- Data engineering pipelines requiring differentiable computations
- Scientific computing and optimization

---

##  Keywords

**automatic differentiation, autodiff, deep learning, machine learning, AI engineering, data engineering, computational graph, backpropagation, PyTorch, TensorFlow, NumPy, Python, tensor operations, gradient computation, differentiable programming, scientific computing, neural networks, software engineering**

---

##  Installation & Test Instructions

This project is distributed as a `.zip` archive and uses a `requirements.txt` file to manage dependencies. Follow the instructions below to install the environment and run the tests on **Windows, macOS, or Linux**.

### Prerequisites

- Python **3.7 or higher**
- `pip` (Python package installer)

Check that Python and pip are available:

```bash
python --version
pip --version
```

> On some systems, use `python3` and `pip3` instead of `python` and `pip`.

---

### Installation Steps

1. **Extract the ZIP Archive**
   - **Windows**: Right-click the `.zip` → “Extract All…”
   - **macOS/Linux**:
     ```bash
     unzip your_project.zip -d your_project_folder
     cd your_project_folder
     ```
2. **(Optional but recommended) Create and Activate a Virtual Environment**
   - **Windows**:
     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

### Running the Tests

Run the following test commands **in order**, each corresponding to a different section:

```bash
python3 -m pytest -v -k "forward"
python3 -m pytest -l -v -k "backward"
python3 -m pytest -k "topo_sort"
python3 -m pytest -k "compute_gradient"
```

> On **Windows**, use `python` instead of `python3` if `python3` is not recognized.

---

## ⚠️ Note

This is an individual, open-source project developed for professional and educational purposes. It is not affiliated with any university or institution.

---

## Troubleshooting

- If you see "command not found" or similar errors, ensure Python and pip are added to your system's PATH.
- If dependencies are missing, make sure the virtual environment is activated and `pip install -r requirements.txt` ran successfully.
- If you encounter permission issues on Unix systems, prepend commands with `python3 -m` rather than calling `pytest` directly.

---
