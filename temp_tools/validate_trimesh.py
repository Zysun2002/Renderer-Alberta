import trimesh
import numpy as np
from PIL import Image

# Load the OBJ file
mesh = trimesh.load('/root/autodl-tmp/tiny-renderer/assets/pumpkin/pumpkin_.obj')

# Define the camera parameters
camera_position = np.array([0, 0, 5])  # Camera position in 3D space
look_at = np.array([0, 0, 0])  # Where the camera is looking
up_vector = np.array([0, 1, 0])  # Defines the 'up' direction for the camera
fov = 60  # Field of view in degrees
image_size = (800, 600)  # Output image size (width, height)

# Function to create a "look-at" matrix
def look_at_matrix(camera_pos, target, up):
    # Compute the forward, right, and up vectors
    forward = (camera_pos - target)
    forward /= np.linalg.norm(forward)
    
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    
    up_corrected = np.cross(forward, right)
    
    # Create the rotation and translation components
    rotation = np.array([right, up_corrected, forward])
    translation = -rotation @ camera_pos
    
    # Build the view matrix (4x4)
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = rotation
    view_matrix[:3, 3] = translation
    return view_matrix

# Create the camera view matrix
camera_transform = look_at_matrix(camera_position, look_at, up_vector)

# Create a perspective projection matrix
aspect_ratio = image_size[0] / image_size[1]
fov_rad = np.deg2rad(fov)
projection_matrix = trimesh.transformations.perspective_matrix(fov_rad, aspect_ratio, 0.1, 100.0)

# Combine the view and projection matrices
view_projection = projection_matrix @ camera_transform

# Project vertices to 2D
vertices = mesh.vertices.copy()
vertices_homogeneous = np.hstack([vertices, np.ones((vertices.shape[0], 1))])  # Convert to homogeneous coordinates
vertices_2d = (view_projection @ vertices_homogeneous.T).T

# Convert homogeneous coordinates back to 2D, with explicit casting to float
vertices_2d = vertices_2d[:, :2] / vertices_2d[:, 3][:, None]
vertices_2d = vertices_2d.astype(np.float64)  # Ensure float casting for division

# Scale and translate to image space
vertices_2d[:, 0] = (vertices_2d[:, 0] + 1) * image_size[0] / 2
vertices_2d[:, 1] = (1 - vertices_2d[:, 1]) * image_size[1] / 2

# Create a blank image
image = Image.new('RGB', image_size, (255, 255, 255))

# Draw the projected vertices onto the image
for vertex in vertices_2d:
    x, y = map(int, vertex)  # Convert floating-point values to integers for pixel coordinates
    if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
        image.putpixel((x, y), (0, 0, 0))  # Drawing points (simplified)

# Save or display the image
image.show()
