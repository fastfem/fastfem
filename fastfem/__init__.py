"""
FastFEM is planned to be a general-purpose finite element method (FEM) library with a
focus on great Python interface. Currently, it can only solve

$$
\\frac{\\partial^2 f(x,y,t)}{\\partial x^2} + \\frac{\\partial^2 f(x,y,t)}{\\partial y^2}
=
h(f) \\frac{\\partial f(x,y,t)}{\\partial t} + g(x,y)
$$

where $f(x,y,t)$, $h(f)$, and $g(x,y)$ are scalar functions, $x$ and $y$ are spatial
coordinates, and $t$ is time.
"""

__version__ = "0.0.1"

from . import mesh

__all__ = ["mesh"]
