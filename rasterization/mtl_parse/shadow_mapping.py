import numpy as np
import sys
sys.path .append("/root/autodl-tmp/tiny-renderer")
from PIL import Image
from tqdm import tqdm
import ipdb
import trimesh

def norm(x):
    return x / np.linalg.norm(x)
    

def embed_v(x):
    assert x.shape == (3,)
    return np.append(x, 1.0)

def embed_vert(x):

    return (np.append(x, 1.0))

def embed_vec(x):

    return (np.append(x, 0.))


def lookat(eye, center, up):
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

def create_projection(f_):
    # ipdb.set_trace()
    return np.array(((1, 0, 0, 0), 
                     (0, 1, 0, 0), 
                     (0, 0, 1, 0), 
                     (0, 0, f_, 1)))
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
            object.uv = scene.visual.uv
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
                    # Skip empty lines and comments
                    continue

                parts = line.split(maxsplit=1)
                keyword = parts[0]

                if keyword == 'newmtl':
                    # Start a new material definition
                    current_material = parts[1].strip()
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
    
    

def triangle(verts, shader, image, z_buffer, depth):
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
    def __init__(self, model, light_dir, viewport, projection, model_view, view, shadow_buffer, trans_shadow):
        self.model = model
        self.light_dir = light_dir
        self.viewport = viewport
        self.projection = projection
        self.model_view = model_view
        self.view = view
        self.shadow_buffer = shadow_buffer
        self.frame2shadow = trans_shadow @ np.linalg.inv(viewport @ projection @ model_view)
        # self.varying_intensity = np.zeros((3)).astype(float)
        self.varying_normal = np.zeros((3, 3)).astype(float)
        self.varying_uv = np.zeros((3, 2)).astype(float)
        self.varying_vertex = np.zeros((3,3)).astype(float)


    def vertex(self, face, object, nth_vert):
        self.object = object
        self.material = object.material

        vertex = embed_v(object.verts[face][nth_vert])
        vertex = (self.viewport @ self.projection @ self.model_view @ vertex)
    
        normal = object.v_normals[face][nth_vert]
        # intensity = np.dot(normal, self.light_dir)
        self.varying_vertex[nth_vert] = (vertex / vertex[3])[:3]
        self.varying_normal[nth_vert] = normal
        self.varying_uv[nth_vert] = object.uv[face][nth_vert]
        
        return vertex

    def fragment(self, bar):
        # intensity = np.dot(self.varying_intensity, bar) 
        shadow_coords = self.frame2shadow @ embed_vert(bar @ self.varying_vertex)
        shadow_coords = shadow_coords / shadow_coords[3]
        id_x, id_y = int(shadow_coords[0]), int(shadow_coords[1])
        shadow = .3 + .7 * int(self.shadow_buffer[id_x, id_y] < shadow_coords[2]+ 43.34)
        # print(shadow_coords[2])
        normal = self.varying_normal.T @ bar
        uv = self.varying_uv.T @ bar

        ambient = self.material['Ka']

        if uv[1] < 0:
            ipdb.set_trace()
        # diffuse =  self.material['Kd'] * intensity
        diffuse = self.object.retrieve_texture(uv)

        reflect = norm(2 * np.dot(self.light_dir, normal) * normal - self.light_dir)
        spec = pow(max(np.dot(reflect, self.view), 0.), self.material['Ns'])
        specular = self.material['Ks'] * spec

        color = 0.2 * ambient + (0.9 * diffuse + 0.5 * specular) * shadow

        contains_nan = np.isnan(color).any()
        if contains_nan:
            ipdb.set_trace()

        return tuple(int(c) for c in color)

class Depth_Shader():
    def __init__(self, model, light_dir, viewport, projection, model_view, view, depth):
        self.model = model
        self.light_dir = light_dir
        self.viewport = viewport
        self.projection = projection
        self.model_view = model_view
        self.view = view
        self.varying_vertex = np.zeros((3,3)).astype(float)
        self.depth = depth

    def vertex(self, face, object, nth_vert):
        self.object = object
        vertex = embed_vert(object.verts[face][nth_vert])
        vertex = self.viewport @ self.projection @ self.model_view @ vertex
        self.varying_vertex[nth_vert] = (vertex / vertex[3])[:3]
        return vertex

    def fragment(self, bar):
        p = bar @ self.varying_vertex
        color = tuple((255, 255, 255))
        # print(p[2])
        color = tuple(int(c * p[2] / self.depth) for c in color)
        return color 


def main():

    width, height = 800, 800

    depth = 255

    black = (0, 0, 0)
    white = (255, 255, 255)

    light_dir = norm(np.array((-1,1,1)))
    eye = np.array((-3, 4, 30))
    center = np.array((1,  1, 25))
    up = np.array((0,1,0))

    view = norm(center - eye)
    f = np.linalg.norm(eye-center)
    
    # model_view = lookat(eye, center, up)
    viewport = create_viewport(width/8, height/8, width*3/4, height*3/4, depth)
    projection = create_projection(0)

    image = Image.new("RGB", (width, height), black)

    obj_path = "/root/autodl-tmp/tiny-renderer/assets/tanks/OBJ_7050.obj"
    model = Model(obj_path)
    mtl_path = "/root/autodl-tmp/tiny-renderer/assets/tanks/OBJ_7050.mtl"
    model.load_mtl(mtl_path)

    # bake depth
    model_view = lookat(light_dir, center, up)
    shadow_buffer = np.full((width, height), -np.inf)
    depth_shader = Depth_Shader(model, light_dir, viewport, projection, model_view, view, depth)
    for name, object in model.objects.items():
        for face in tqdm(object.faces, desc='baking depth'):
            screen_coords = np.zeros((3, 4))
            for i in range(3):
                screen_coords[i] = depth_shader.vertex(face, object, i)
            triangle(screen_coords, depth_shader, image, shadow_buffer, depth)
       
        real_time_depth_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        real_time_depth_image.save('real-time depth image.png')


    # ipdb.set_trace()

    image = Image.new("RGB", (width, height), black)
    model_view = lookat(eye, center, up)
    projection = create_projection(-1./f)
    trans_shadow = viewport @ projection @ model_view
    shader = Gouraud_Shader(model, light_dir, viewport, projection, model_view, view, shadow_buffer, trans_shadow)
    z_buffer = np.full((width, height), -np.inf)
    for name, object in model.objects.items():
        # ipdb.set_trace()
        for face in tqdm(object.faces, desc='rendering current object'):
            screen_coords = np.zeros((3, 4))
            for i in range(3):
                screen_coords[i] = shader.vertex(face, object,  i)
            # ipdb.set_trace()
            triangle(screen_coords, shader, image, z_buffer, depth)
        
        real_time_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        real_time_image.save("real-time.png")

    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save("with_normal_mapping.png")


if __name__ == '__main__':
    main()