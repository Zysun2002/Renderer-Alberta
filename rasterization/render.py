import numpy as np
import sys
from PIL import Image
from tqdm import tqdm
import ipdb
import trimesh  # only used for loading obj file
import argparse
import os
import re

class Config():
    def __init__(self, args):
        self.obj_path = args.obj_path
        self.mtl_path = args.mtl_path
        self.output_path = args.output_path

        self.eye = np.array(args.camera_position)
        self.center = np.array(args.lookat)
        self.up = np.array(args.up_direction)
        self.light_dir = norm(args.light_direction)

        self.depth = 255
        self.width, self.height = args.width, args.height
        self.background = tuple(args.background)
        self.Ca, self.Cs, self.Cd = args.Ca, args.Cs, args.Cd
        
        self.allow_vis = args.allow_vis
        self.vis_iter = args.vis_iter
        self.vis_path = args.vis_path
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)

def get_numeric_suffix(filename):
    match = re.search(r'real_time_(\d+)', filename)
    return int(match.group(1)) if match else None
def create_video(config):
    images = [img for img in os.listdir(config.vis_path) if img.endswith(".png")]
    images.sort(key=get_numeric_suffix)  # Sort images by numeric suffix

    # Load the images into a list
    frames = [Image.open(os.path.join(config.vis_path, image)) for image in images]
    frames[0].save("{}/video.gif".format(config.vis_path), format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)


def norm(x):
    return x / np.linalg.norm(x)
    

def embed_v(x):
    
    assert x.shape == (3,)
    return np.append(x, 1.0)

def embed_vert(x):

    return (np.append(x, 1.0))

def embed_vec(x):

    return (np.append(x, 0.))


def create_model_view(eye, center, up):
    z = norm(eye-center)
    x = norm(np.cross(up, z))
    y = norm(np.cross(z, x)) 

    return np.array(((x[0], x[1], x[2], -center[0]),
                    (y[0], y[1], y[2], -center[1]),
                    (z[0], z[1], z[2], -center[2]),
                    (0.,   0.,   0.,   1.)))

def create_viewport(x, y, w, h, depth):
    return np.array(((w/2., 0, 0, x+w/2.), 
                     (0, h/2., 0, y+h/2.), 
                     (0, 0, depth/2., depth/2.), 
                     (0, 0, 0, 1)))

def create_projection(f):
    return np.array(((1, 0, 0, 0), 
                     (0, 1, 0, 0), 
                     (0, 0, 1, 0), 
                     (0, 0, -1./f, 1)))

class Object():
        def __init__(self, name):
            print("{} is created!".format(name))
        
        def create_material(self):
            self.material = {}
        
        def add_material(self, name, value):
            self.material[name] = value

        def load_texture(self, tga_path):
            texture_mapping = Image.open(tga_path)
            self.texture_mapping = texture_mapping.transpose(Image.FLIP_TOP_BOTTOM)

        def retrieve_texture(self, uv):
            x, y = int(uv[0] * self.texture_mapping.size[0]), int(uv[1] * self.texture_mapping.size[1])
            return np.array(self.texture_mapping.getpixel((x, y)))

class Model():
    def __init__(self, obj_path):
        scene = trimesh.load(obj_path)
        self.objects = {}
        # ipdb.set_trace()

        if isinstance(scene, trimesh.Trimesh):
            name = "Single Object"
            object = Object(name)
            object.verts = scene.vertices
            object.v_normals = scene.vertex_normals
            object.faces = scene.faces
            object.uv = scene.visual.uv % 1
            self.objects[name] = object

        else:
            for name, geometry in tqdm(scene.geometry.items()):
                object = Object(name)
                object.verts = geometry.vertices  
                object.v_normals = geometry.vertex_normals
                object.faces = geometry.faces
                object.uv = geometry.visual.uv % 1
                self.objects[name] = object

    def load_mtl(self, stl_path):
        self.materials = {}
        current_material = None

        with open(stl_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split(maxsplit=1)
                keyword = parts[0]

                if keyword == 'newmtl':
                    # Start a new material definition
                    current_material = parts[1].strip()
                    if "Single Object" in self.objects.keys():
                        current_material = "Single Object"
                    self.objects[current_material].create_material()
                elif keyword in ['Ka', 'Kd', 'Ks']:
                    # Ambient, diffuse, specular colors
                    color = parts[1].split()
                    color = np.array(color, dtype=float) * 255
                    self.objects[current_material].add_material(keyword, color) 
                elif keyword == 'Ns':
                    # Specular exponent
                    self.objects[current_material].add_material('Ns', float(parts[1])) 
                elif keyword == 'd':
                    # Transparency
                    self.objects[current_material].add_material('d', float(parts[1]))
                elif keyword == 'illum':
                    # Illumination model
                    self.objects[current_material].add_material('illum', int(parts[1]))
                elif keyword.startswith('map_'):
                    # Textures
                    self.objects[current_material].add_material(keyword, parts[1].strip())
                    self.objects[current_material].load_texture(parts[1].strip())
    
    

def triangle(verts, shader, image, z_buffer):
    def barycentric(pts, p):
        u = np.cross(np.array((pts[2, 0]-pts[0, 0], pts[1, 0]-pts[0, 0], pts[0, 0]-p[0])),
                    np.array((pts[2, 1]-pts[0, 1], pts[1, 1]-pts[0, 1], pts[0, 1]-p[1])))
        if(abs(u[2]) < 1): return np.array((-1, 1, 1))
        return np.array((float(1) - float(u[0]+u[1])/u[2], float(u[1])/u[2], float(u[0])/u[2]))

    # ipdb.set_trace()
    verts = (verts / (verts[:, -1])[:,np.newaxis]).astype(int)
    
    bbox_min = np.full((2,), np.inf)
    bbox_max = np.full((2, ), -np.inf)
    # ipdb.set_trace()
    for i in range(3):
        bbox_min[0] = np.minimum(bbox_min[0], verts[i][0])
        bbox_min[1] = np.minimum(bbox_min[1], verts[i][1])
        bbox_max[0] = np.maximum(bbox_max[0], verts[i][0])
        bbox_max[1] = np.maximum(bbox_max[1], verts[i][1])
    
    for x in range(max(int(bbox_min[0]), 0), min(int(bbox_max[0]) + 1,image.width)):
        for y in range(max(int(bbox_min[1]), 0), min(int(bbox_max[1]) + 1, image.height)):
            P = np.array((x, y, 0)).astype(float)
            bc_screen = barycentric(verts, P)
            if bc_screen[0] < 0 or bc_screen[1] < 0 or bc_screen[2] < 0: continue
            for i in range(3): P[2] += verts[i][2] * bc_screen[i]
            if (z_buffer[x, y] < P[2]):
                z_buffer[x, y] = P[2]
                color = shader.fragment(bc_screen)
                image.putpixel((x, y), color)

class Gouraud_Shader():
    def __init__(self, model, config):
        self.model = model
        self.light_dir, self.viewport = config.light_dir, config.viewport
        self.projection, self.model_view = config.projection, config.model_view
        self.view = config.view
        self.Ca, self.Cd, self.Cs = config.Ca, config.Cd, config.Cs
        self.varying_intensity = np.zeros((3)).astype(float)
        self.varying_normal = np.zeros((3, 3)).astype(float)
        self.varying_uv = np.zeros((3, 2)).astype(float)

    def vertex(self, face, object, nth_vert):
        self.object = object
        self.material = object.material
        
        vertex = embed_v(object.verts[face][nth_vert])
        vertex = (self.viewport @ self.projection @ self.model_view @ vertex)
        
        normal = object.v_normals[face][nth_vert]
        # intensity = np.dot(normal, self.light_dir)
        # self.varying_intensity[nth_vert] = max(0., intensity)
        self.varying_normal[nth_vert] = normal
        self.varying_uv[nth_vert] = object.uv[face][nth_vert]
        
        return vertex

    def fragment(self, bar):
        # intensity = np.dot(self.varying_intensity, bar) 
        normal = self.varying_normal.T @ bar
        uv = self.varying_uv.T @ bar

        ambient = self.material['Ka'] if 'Ka' in self.material.keys() else 0

        if 'map_Kd' in self.material.keys():
            diffuse = self.object.retrieve_texture(uv)
        else: diffuse =  self.material['Kd']
        
        reflect = norm(2 * np.dot(self.light_dir, normal) * normal - self.light_dir)
        # ipdb.set_trace()
        ns = self.material['Ns'] if 'Ns' in self.material.keys() else 100
        spec = pow(max(np.dot(reflect, self.view), 0.), ns)
        ks = self.material['Ks'] if 'Ks' in self.material.keys() \
            else np.array((255, 255, 255)) 
        specular = ks * spec 

        color = self.Ca * ambient + self.Cd * diffuse + self.Cs * specular

        return tuple(int(c) for c in color)

def render(config):

    config.view = norm(config.center - config.eye)
    config.f = np.linalg.norm(config.eye-config.center)
    
    config.model_view = create_model_view(config.eye, config.center, config.up)
    config.viewport = create_viewport(config.width/8, config.height/8, \
                               config.width*3/4, config.height*3/4, config.depth)
    config.projection = create_projection(config.f)

    image = Image.new("RGB", (config.width, config.height), config.background)
    z_buffer = np.full((config.width, config.height), -np.inf)

    model = Model(config.obj_path)
    model.load_mtl(config.mtl_path)

    shader = Gouraud_Shader(model, config)
    render_iter = 0


    for object in model.objects.values():
        # ipdb.set_trace()
        for face in tqdm(object.faces, desc='rendering current object'):
            screen_coords = np.zeros((3, 4))
            for i in range(3):
                screen_coords[i] = shader.vertex(face, object,  i)

            triangle(screen_coords, shader, image, z_buffer)
            render_iter += 1
            if config.allow_vis and render_iter % config.vis_iter == 0:
                real_time_image = image.transpose(Image.FLIP_TOP_BOTTOM)
                real_time_image.save("./{}/real_time_{}.png".format(config.vis_path, render_iter))
        
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save("{}".format(config.output_path))
    if config.allow_vis:
        create_video(config)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", type=str, required=True)
    parser.add_argument("--mtl_path", type=str)
    parser.add_argument("--output_path", type=str, default="./output.png")

    parser.add_argument("--camera_position", type=float, nargs="+", required=True)
    parser.add_argument("--lookat", type=float, nargs="+", required=True)
    parser.add_argument("--up_direction", type=float, nargs="+", required=True)
    parser.add_argument("--light_direction", type=float, nargs="+", required=True)


    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--background", type=int, nargs="+", default=[0, 0, 0])

    parser.add_argument("--Ca", type=float, default=0.2)
    parser.add_argument("--Cd", type=float, default=0.9)
    parser.add_argument("--Cs", type=float, default=0.8)

    parser.add_argument("--allow_vis", action="store_true")
    parser.add_argument("--vis_iter", type=int, default=100)
    parser.add_argument("--vis_path", type=str, default="output")

    args = parser.parse_args()

    config = Config(args)

    render(config)