import trimesh
from PIL import Image
import numpy as np
import ipdb

class Model():
    def __init__(self, obj_path):
        mesh = trimesh.load(obj_path)
        self.verts = mesh.vertices
        self.faces = mesh.faces
        self.v_normals = mesh.vertex_normals
        self.texture_coords = mesh.visual.uv
        self.texture_mapping = None
        self.normal_mapping = None

    def load_texture(self, tga_path):
        texture_mapping = Image.open(tga_path)
        self.texture_mapping = texture_mapping.transpose(Image.FLIP_TOP_BOTTOM)
        
    def load_normal_mapping(self, tga_path):
        normal_mapping = Image.open(tga_path)
        # ipdb.set_trace()
        self.normal_mapping = normal_mapping.transpose(Image.FLIP_TOP_BOTTOM)
        # self.uv_map.save("texture.png")        

    def retrieve_texture(self, u, v):
        x, y = int(u * self.texture_mapping.size[0]), int(v * self.texture_mapping.size[1])
        return self.texture_mapping.getpixel((x, y))
    
    def retrieve_normal(self, u, v):
        x, y = int(u * self.normal_mapping.size[0]), int(v * self.normal_mapping.size[1])
        c = self.normal_mapping.getpixel((x, y))
        res = np.zeros(3)
        for i in range(3):
            res[2-i] = float(c[i]/255.* 2.-1.)
        # ipdb.set_trace()
        # normal = np.array(c) / 255. * 2 - 1
        return res


def norm(x):
    return x / np.linalg.norm(x)
    

def embed(x):
    assert x.shape == (3,)
    return np.append(x, 1.0)

def embed_v(x):

    return (np.append(x, 1.0))

    


