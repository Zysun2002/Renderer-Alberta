import numpy as np
import sys
sys.path .append("/root/autodl-tmp/tiny-renderer")
from groups import norm, Model, embed_vert, embed_vec
from PIL import Image
from tqdm import tqdm
import ipdb

def rand_point_on_unit_sphere():
    u = np.random.rand()
    v = np.random.rand()
    theta = 2. * np.pi * u
    phi = np.arccos(2. * v - 1)
    return np.array(
        [np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])


def lookat(eye, center, up):
    z = norm(eye-center)
    x = norm(np.cross(up, z))
    y = norm(np.cross(z, x))    
    # ipdb.set_trace()
        # build sreen coordinates
    return np.array(((x[0], x[1], x[2], -center[0]),
                     (y[0], y[1], y[2], -center[1]),
                     (z[0], z[1], z[2], -center[2]),
                     (0.,    0.,    0.,    1.)))

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

def triangle(projected_verts, viewport, shader, image, z_buffer, depth):
    def barycentric(pts, p):
        u = np.cross(np.array((pts[2, 0]-pts[0, 0], pts[1, 0]-pts[0, 0], pts[0, 0]-p[0])),
                    np.array((pts[2, 1]-pts[0, 1], pts[1, 1]-pts[0, 1], pts[0, 1]-p[1])))
        if(abs(u[2]) < 1): return np.array((-1, 1, 1))
        return np.array((float(1) - float(u[0]+u[1])/u[2], float(u[1])/u[2], float(u[0])/u[2]))

    # ipdb.set_trace()
    verts = (viewport @ projected_verts.T).T
    verts2 = (verts / (verts[:, -1])[:,np.newaxis]).astype(float)
    
    bbox_min = np.full((2,), np.inf)
    bbox_max = np.full((2, ), -np.inf)
    # ipdb.set_trace()
    for i in range(3):
        bbox_min[0] = np.minimum(bbox_min[0], verts2[i][0])
        bbox_min[1] = np.minimum(bbox_min[1], verts2[i][1])
        bbox_max[0] = np.maximum(bbox_max[0], verts2[i][0])
        bbox_max[1] = np.maximum(bbox_max[1], verts2[i][1])
    
    for x in range(int(bbox_min[0]), int(bbox_max[0]) + 1):
        for y in range(int(bbox_min[1]), int(bbox_max[1]) + 1):
            P = np.array((x, y)).astype(float)
            bc_screen = barycentric(verts2, P)
            if bc_screen[0] < 0 or bc_screen[1] < 0 or bc_screen[2] < 0: continue
            
            bc_clip = np.array([bc_screen[i] / verts[i, 3] for i in range(3)])
            bc_clip = bc_clip / (bc_clip[0]+bc_clip[1]+bc_clip[2])
            frag_depth = np.dot(projected_verts[:, 2], bc_clip)
            
            if (z_buffer[x, y] <= frag_depth):
                z_buffer[x, y] = frag_depth
                color = shader.fragment(np.array([P[0], P[1], frag_depth]), bc_screen)
                image.putpixel((x, y), color)

class Z_Shader():
    def __init__(self, model, light_dir, viewport, projection, model_view):
        self.model = model
        self.light_dir = light_dir
        self.viewport = viewport
        self.projection = projection
        self.model_view = model_view
        
        self.varying_vertex = np.zeros((3, 4)).astype(float)

    def vertex(self, face, nth_vert):

        vertex = embed_vert(self.model.verts[face][nth_vert])
        vertex = self.projection @ self.model_view @ vertex  
        self.varying_vertex[nth_vert] = vertex
        return vertex

    def fragment(self, frag_coords, bar):

        color = tuple(int(255*(frag_coords[2]+1.)/2.) for i in range(3))
        
        return color

class Shader():
    def __init__(self, model, light_dir, viewport, projection, model_view, shadow_buffer, occl):
        self.model = model
        self.light_dir = light_dir
        self.viewport = viewport
        self.projection = projection
        self.model_view = model_view
        
        self.varying_uv = np.zeros((3, 2)).astype(float)
        self.varying_vertex = np.zeros((3, 4)).astype(float)

        self.shadow_buffer = shadow_buffer
        self.occl = occl

    def vertex(self, face, nth_vert):
        self.varying_uv[nth_vert] = self.model.texture_coords[int(face[nth_vert])]

        vertex = embed_vert(self.model.verts[face][nth_vert])
        vertex = self.projection @ self.model_view @ vertex
        self.varying_vertex[nth_vert] = vertex
        return vertex

    def fragment(self, frag_coords, bar):
        # intensity = np.dot(self.varying_intensity, bar)
        uv = bar @ self.varying_uv
        
        idx, idy = int(frag_coords[0]), int(frag_coords[1])
        if np.abs(self.shadow_buffer[idx, idy] - frag_coords[2]) < 1e-2:
            self.occl.putpixel((int(uv[0]*1024),int(uv[1]*1024)), 255)

        color = tuple((255, 0, 0))
        
        return color




class Gouraud_Shader():
    def __init__(self, model, light_dir, viewport, projection, model_view, color):
        self.model = model
        self.light_dir = light_dir
        self.viewport = viewport
        self.projection = projection
        self.model_view = model_view
        self.varying_intensity = np.zeros((3)).astype(float)
        
        self.ambient = 20
        self.coefficient_diff = 0
        self.coefficient_spec = 0
        
        self.varying_uv = np.zeros((3, 2)).astype(float)
        self.varying_normal = np.zeros((3, 3)).astype(float)

    def vertex(self, face, nth_vert):
        self.varying_uv[nth_vert] = self.model.texture_coords[int(face[nth_vert])]
        self.varying_normal[nth_vert] = (norm((np.linalg.inv(self.projection @ self.model_view).T @ \
            embed_vert(self.model.v_normals[int(face[nth_vert])]))))[:3]
        
        vertex = embed_vert(self.model.verts[face][nth_vert])
        vertex = (self.viewport @ self.projection @ self.model_view @ vertex)

        

        return vertex

    def fragment(self, bar):
        # intensity = np.dot(self.varying_intensity, bar)
        uv = bar @ self.varying_uv
        


        normal = norm((np.linalg.inv(self.projection @ self.model_view).T @ \
            embed_vert(self.model.retrieve_normal(uv[0], uv[1])))[:3])

        # ipdb.set_trace()  
        light = norm((self.projection @ self.model_view @ embed_vert(self.light_dir))[:3])
        
        reflect = norm(2 * np.dot(light, normal) * normal - light)
        spec = pow(max(reflect[2], 0.), self.model.retrieve_specular(uv[0], uv[1]))
        diff = max(0., np.dot(normal, light))
        # intensity = max(0., np.dot(normal, light))

        color = self.model.retrieve_diffuse(uv[0], uv[1])

        # ipdb.set_trace()

        color = tuple(int(min(255, \
            self.ambient + c * (self.coefficient_diff * diff + self.coefficient_spec * spec))) \
            for c in color)
        
        return color


def main():
    n_renders = 100

    width, height = 800, 800

    depth = 255

    black = (0, 0, 0)
    white = (255, 255, 255)
    # orange = (255, 155, 0)

    light_dir = norm(np.array((1,1,1)))
    eye = np.array((1, 1, 3))
    center = np.array((0,0,0))
    up = np.array((0,1,0))

    f = np.linalg.norm(eye-center)

    image = Image.new("RGB", (width, height), black)

    obj_path = "/root/autodl-tmp/tiny-renderer/assets/obj_II/diablo3_pose.obj"
    model = Model(obj_path)
    texture_path = "/root/autodl-tmp/tiny-renderer/assets/obj_II/diablo3_pose_diffuse.tga"
    model.load_texture(texture_path)
    normal_path = "/root/autodl-tmp/tiny-renderer/assets/obj_II/diablo3_pose_nm.tga"
    model.load_normal_mapping(normal_path)
    specular_path = "/root/autodl-tmp/tiny-renderer/assets/obj_II/diablo3_pose_spec.tga"
    model.load_specular_mapping(specular_path)
    total = Image.new("L", (1024, 1024), 0)
    
    for iter in tqdm(range(n_renders)):
        up = np.random.rand(3)
        eye = rand_point_on_unit_sphere()
        eye[1] = np.abs(eye[1]) 

        model_view = lookat(eye, center, up)
        viewport = create_viewport(width/8, height/8, width*3/4, height*3/4, depth)
        projection = create_projection(0)

        # render shadow buffer
        shadow_buffer = np.full((width, height), -np.inf)
        z_shader = Z_Shader(model, light_dir, viewport, projection, model_view)
        
        for face in model.faces:
            projected_verts = np.zeros((3,4))
            for i in range(3):
                projected_verts[i] = z_shader.vertex(face, i)
            triangle(projected_verts, viewport, z_shader, image, shadow_buffer, depth)
        
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.save("frame_buffer.png")

        # render occl
        occl = Image.new("L", (1024, 1024), 0)

        z_buffer = np.full((width, height), -np.inf)
        shader = Shader(model, light_dir, viewport, projection, model_view, shadow_buffer, occl)
        for face in model.faces:
            projected_verts = np.zeros((3, 4))
            for i in range(3):
                projected_verts[i] = shader.vertex(face, i)
            triangle(projected_verts, viewport, shader, image, z_buffer, depth)
        
        for i in range(1024):
            for j in range(1024):
                tmp = total.getpixel((i, j))
                color = int((tmp * iter + occl.getpixel((i, j))) / float(iter + 1) + .5)
                total.putpixel((i, j), color)

    total = total.transpose(Image.FLIP_TOP_BOTTOM)
    total.save("occlusion.png")
    occl = occl.transpose(Image.FLIP_TOP_BOTTOM)
    occl.save("occl.png")


if __name__ == '__main__':
    main()