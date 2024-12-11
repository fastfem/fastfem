import fastfem.mesh as m

points = [
    m.Point(0, 0),
    m.Point(1, 0),
    m.Point(1, 1),
    m.Point(0.5, 1.5),
    m.Point(0, 1),
    m.Point(-0.5, 0.5),
    m.Point(0, 0),
]
lines = [m.Line(points[i], points[i + 1], number_of_nodes=10) for i in range(6)]
surface = m.Surface(lines, element_type="quadrangle")
mesh = m.mesh()
