"""
The `fastfem.mesh` package contains utilities for generating and reading meshes.
"""

from .fundamentals import (
    RectangleMesh,
    SquareMesh,
    create_a_rectangle_mesh,
    create_a_square_mesh,
)
from .generator import (
    Domain,
    Line,
    Mesh,
    OneDElementType,
    Point,
    Submesh,
    Surface,
    TwoDElementType,
    ZeroDElementType,
    mesh,
)

__all__ = [
    "Domain",
    "Line",
    "Mesh",
    "OneDElementType",
    "Point",
    "RectangleMesh",
    "SquareMesh",
    "Submesh",
    "Surface",
    "TwoDElementType",
    "ZeroDElementType",
    "create_a_rectangle_mesh",
    "create_a_square_mesh",
    "mesh",
]
