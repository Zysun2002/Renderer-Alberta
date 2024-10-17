import numpy as np

def opengl_to_blender_coordinates(vertices):
    """
    Convert a list of vertices from OpenGL coordinate system to Blender coordinate system.

    Args:
    - vertices: List of 3D coordinates (x, y, z) in OpenGL coordinate system.

    Returns:
    - A list of 3D coordinates in Blender coordinate system.
    """
    converted_vertices = []

    for vertex in vertices:
        x, y, z = vertex
        # Swap Y and Z, and invert the new Z axis
        blender_vertex = (x, -z, y)
        converted_vertices.append(blender_vertex)

    return converted_vertices

# Example usage
opengl_vertices = [
    (0, -30, 30),
]

blender_vertices = opengl_to_blender_coordinates(opengl_vertices)

# Print the converted vertices
print("Converted vertices to Blender coordinate system:")
for vertex in blender_vertices:
    print(vertex)
