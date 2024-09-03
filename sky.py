import trimesh
import ipdb 
# 假设有点和面的信息

class OBJParser:
    def __init__(self, filepath):
        self.vertices = []
        self.texcoords = []
        self.normals = []
        self.faces = []
        self.parse_obj(filepath)

    def parse_obj(self, filepath):
        with open(filepath, 'r') as file:
            for line in file:
                if line.startswith('v '):  # Vertex position
                    self.vertices.append(self.parse_vertex(line))
                elif line.startswith('vt '):  # Texture coordinate
                    self.texcoords.append(self.parse_texcoord(line))
                elif line.startswith('vn '):  # Vertex normal
                    self.normals.append(self.parse_normal(line))
                elif line.startswith('f '):  # Face
                    self.faces.append(self.parse_face(line))

    def parse_vertex(self, line):
        # Example: "v 1.000000 2.000000 3.000000"
        parts = line.strip().split()
        return list(map(float, parts[1:4]))

    def parse_texcoord(self, line):
        # Example: "vt 0.500000 1.000000"
        parts = line.strip().split()
        return list(map(float, parts[1:3]))

    def parse_normal(self, line):
        # Example: "vn 0.000000 1.000000 0.000000"
        parts = line.strip().split()
        return list(map(float, parts[1:4]))

    def parse_face(self, line):
        # Example: "f 1/1/1 2/2/2 3/3/3"
        parts = line.strip().split()[1:]
        face = []
        for part in parts:
            indices = list(map(int, part.split('/')))
            face.append({
                'vertex_index': indices[0] - 1,  # OBJ indices are 1-based, convert to 0-based
                'texcoord_index': indices[1] - 1 if len(indices) > 1 and indices[1] else None,
                'normal_index': indices[2] - 1 if len(indices) > 2 and indices[2] else None
            })
        return face

    def get_faces_as_vertex_indices(self):
        return [[vertex['vertex_index'] for vertex in face] for face in self.faces]

# Usage
mesh = trimesh.load('/root/autodl-tmp/tiny-renderer/assets/african_head.obj')

obj_parser = OBJParser('/root/autodl-tmp/tiny-renderer/assets/african_head.obj')

ipdb.set_trace()

# Access parsed data
print("Vertices:", obj_parser.vertices)
print("Texture Coordinates:", obj_parser.texcoords)
print("Normals:", obj_parser.normals)
print("Faces (with indices):", obj_parser.get_faces_as_vertex_indices())
