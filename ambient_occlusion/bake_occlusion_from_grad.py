import numpy as np
import sys
sys.path .append("/root/autodl-tmp/tiny-renderer")
from groups import norm, Model, embed_vert, embed_vec
from PIL import Image
from tqdm import tqdm
import ipdb

def max_elevation_angle(z_buffer, p, dir, width, height):
    max_angle = 0.
    for t in range(1000):
        cur = p + dir * t
        if cur[0] >= width or cur[1] >= height or cur[0] < 0 or cur[1] < 0:
            return max_angle
        
        distance = np.linalg.norm(p - cur)
        if distance < 1.: continue
        elevation = z_buffer[int(cur[0]), int(cur[1])] \
                    - z_buffer[int(p[0]), int(p[1])]
        # ipdb.set_trace()
        max_angle = np.max([max_angle, np.arctan(elevation/distance)])
    return max_angle


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
    verts2 = (verts / (verts[:, -1])[:,np.newaxis]).astype(int)
    
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
                color = shader.fragment(np.array([P[0], P[1], frag_depth]), bc_clip)
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



def main():

    width, height = 800, 800

    depth = 255

    black = (0, 0, 0)
    white = (255, 255, 255)
    # orange = (255, 155, 0)

    eye = np.array((1.2, -0.8, 3))
    center = np.array((0,0,0))
    up = np.array((0,1,0))

    f = np.linalg.norm(eye-center)

    model_view = lookat(eye, center, up)
    viewport = create_viewport(width/8, height/8, width*3/4, height*3/4, depth)
    projection = create_projection(-1./f)

    image = Image.new("RGB", (width, height), black)

    obj_path = "/root/autodl-tmp/tiny-renderer/assets/african_head.obj"
    model = Model(obj_path)
    # texture_path = "/root/autodl-tmp/tiny-renderer/assets/obj_II/diablo3_pose_diffuse.tga"
    # model.load_texture(texture_path)
    # normal_path = "/root/autodl-tmp/tiny-renderer/assets/obj_II/diablo3_pose_nm.tga"
    # model.load_normal_mapping(normal_path)
    # specular_path = "/root/autodl-tmp/tiny-renderer/assets/obj_II/diablo3_pose_spec.tga"
    # model.load_specular_mapping(specular_path)

    z_buffer = np.full((width, height), -np.inf)
    z_shader = Z_Shader(model, None, viewport, projection, model_view)
    
    for face in tqdm(model.faces):
        projected_verts = np.zeros((3,4))
        for i in range(3):
            projected_verts[i] = z_shader.vertex(face, i)
        triangle(projected_verts, viewport, z_shader, image, z_buffer, depth)
    
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save("depth.png")

    image = Image.new("RGB", (width, height), black)

    for x in tqdm(range(width)):
        for y in range(height):
            if (z_buffer[x, y] < -1e-5): continue
            total = 0.
            for a in np.arange(0., np.pi*2-1e-4, np.pi/4):
                total += np.pi / 2 - max_elevation_angle(z_buffer, np.array([x, y]), 
                                np.array([np.cos(a), np.sin(a)]), height, width)
            total /= (np.pi / 2 * 8)  
            total = np.pow(total, 100.)
            color = tuple([int(255 * total) for _ in range(3)])
            image.putpixel((x, y), color)

    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save("occlusion_from_grad_1.png")


if __name__ == '__main__':
    main()