import fastfem.mesh as m

square_mesh = (
    m.create_a_square_mesh(
        side_length=1.0,
        nodes_in_horizontal_direction=10,
        nodes_in_vertical_direction=10,
        element_type="quadrangle",
    ),
)

create_a_rectangle_mesh = (
    m.create_a_rectangle_mesh(
        horizontal_length=1.0,
        vertical_length=0.5,
        nodes_in_horizontal_direction=10,
        nodes_in_vertical_direction=10,
        element_type="triangle",
    ),
)
