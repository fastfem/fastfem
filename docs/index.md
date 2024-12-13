# FastFEM

[![](https://github.com/fastfem/fastfem/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/fastfem/fastfem/actions/workflows/test.yaml)
[![](https://coverage-badge.samuelcolvin.workers.dev/fastfem/fastfem.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/fastfem/fastfem)
[![](<https://img.shields.io/badge/docs-mkdocs-rgb(0%2C79%2C144)>)](https://fastfem.com)
[![](<https://img.shields.io/pypi/v/fastfem?label=PyPI%20version&color=rgb(0%2C79%2C144)>)](https://pypi.python.org/pypi/fastfem)
[![](<https://img.shields.io/pepy/dt/fastfem?label=PyPI%20downloads&color=rgb(0%2C%2079%2C%20144)>)](https://pypistats.org/packages/fastfem)

FastFEM is planned to be a general-purpose finite element method (FEM) library with a focus on great Python interface. Currently, it can only solve

$$
\begin{equation}
    \frac{\partial^2 f(x,y,t)}{\partial x^2} + \frac{\partial^2 f(x,y,t)}{\partial y^2}
    =
    h(f) \frac{\partial f(x,y,t)}{\partial t} + g(x,y)
    \label{pde}
\end{equation}
$$

where $f(x,y,t)$, $h(f)$, and $g(x,y)$ are scalar functions, $x$ and $y$ are spatial coordinates, and $t$ is time.

## Installation

1. Install Python 3.12.
2. Install FastFEM using pip:

```bash
pip install fastfem
```

## Usage

Check out the [examples](https://github.com/fastfem/fastfem/tree/main/examples) directory for usage examples.