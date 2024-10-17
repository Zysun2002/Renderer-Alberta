import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to read the vertices from the OBJ file
def read_vertices_from_obj(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("v "):  # Only process lines that define vertices
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
    return vertices

# Function to plot and save the vertices as an image
def plot_vertices_and_save(vertices, image_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Unpack the vertices into x, y, z components
    xs, ys, zs = zip(*vertices)
    
    # Create a 3D scatter plot
    ax.scatter(xs, ys, zs, c='r', marker='o', s=1)
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of OBJ Vertices')
    
    # Save the plot as an image
    plt.savefig(image_path, dpi=300)  # Save with high resolution
    plt.close()  # Close the plot to avoid display if running in script mode

# File path to your OBJ file and output image
obj_file = '/root/autodl-tmp/tiny-renderer/assets/pumpkin/pumpkin_.obj'
image_file = 'output_image.png'  # Can also use .jpg, .svg, etc.

# Read vertices from OBJ file and save the plot as an image
vertices = read_vertices_from_obj(obj_file)
plot_vertices_and_save(vertices, image_file)

print(f"3D plot saved as {image_file}")
