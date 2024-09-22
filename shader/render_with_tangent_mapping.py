import numpy as np
import sys
sys.path .append("/root/autodl-tmp/tiny-renderer")
from groups import norm, Model, embed_vert, embed_vec
from PIL import Image
from tqdm import tqdm
import ipdb


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

def create_projection(f):
    # ipdb.set_trace()
    return np.array(((1, 0, 0, 0), 
                     (0, 1, 0, 0), 
                     (0, 0, 1, 0), 
                     (0, 0, -1./f, 1)))

def triangle(projected_verts, shader, image, z_buffer, viewport):
    def barycentric(pts, p):
        u = np.cross(np.array((pts[2, 0]-pts[0, 0], pts[1, 0]-pts[0, 0], pts[0, 0]-p[0])),
                    np.array((pts[2, 1]-pts[0, 1], pts[1, 1]-pts[0, 1], pts[0, 1]-p[1])))
        if(abs(u[2]) < 1): return np.array((-1, 1, 1))
        return np.array((float(1) - float(u[0]+u[1])/u[2], float(u[1])/u[2], float(u[0])/u[2]))

    verts = (viewport @ projected_verts.T).T
    verts2 = (verts / (verts[:, -1])[:,np.newaxis])[:, :2].astype(float)
    
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
                color = shader.fragment(bc_clip)
                image.putpixel((x, y), color)

class Gouraud_Shader():
    def __init__(self, model, light_dir, viewport, projection, model_view, color):
        self.model = model
        self.light_dir = light_dir
        self.viewport = viewport
        self.projection = projection
        self.model_view = model_view
        self.varying_intensity = np.zeros((3)).astype(float)
        
        self.varying_uv = np.zeros((3, 2)).astype(float)
        self.varying_normal = np.zeros((3, 3)).astype(float)
        self.ndc_tri = np.zeros((3, 3)).astype(float)

    def vertex(self, face, nth_vert):
        self.varying_uv[nth_vert] = self.model.texture_coords[int(face[nth_vert])]
        self.varying_normal[nth_vert] = norm((np.linalg.inv(self.projection @ self.model_view).T @ \
            embed_vec(self.model.v_normals[int(face[nth_vert])]))[:3])
        
        vertex = embed_vert(self.model.verts[face][nth_vert])
        vertex = self.projection @ self.model_view @ vertex

        self.ndc_tri[nth_vert] = (vertex / vertex[3])[:3]
        return vertex

    def fragment(self, bar):
        # intensity = np.dot(self.varying_intensity, bar)
        # ipdb.set_trace()
        uv = bar @ self.varying_uv
        face_normal = norm(bar @ self.varying_normal)

        A = np.zeros((3, 3)).astype(float)
        A[0] = self.ndc_tri[1] - self.ndc_tri[0]
        A[1] = self.ndc_tri[2] - self.ndc_tri[0]
        A[2] = face_normal
        # import ipdb; ipdb.set_trace()
        A_inverse = np.linalg.inv(A)
        i = A_inverse @ np.array((self.varying_uv[1, 0] - self.varying_uv[0, 0],
                                 self.varying_uv[2, 0] - self.varying_uv[0, 0],
                                 0)).T
        j = A_inverse @ np.array((self.varying_uv[1, 1] - self.varying_uv[0, 1],
                                 self.varying_uv[2, 1] - self.varying_uv[0, 1],
                                 0)).T

        # import ipdb; ipdb.set_trace()
        B = np.zeros((3, 3)).astype(float)
        B[0] = norm(i); B[1] = norm(j); B[2] = face_normal

        normal = norm(B.T @ self.model.retrieve_normal(uv[0], uv[1]))
        
        # ipdb.set_trace()
        light_dir = norm((self.projection @ self.model_view @ embed_vec(self.light_dir))[:3])
        diff = max(0., np.dot(normal, light_dir))
        # intensity = max(0., np.dot(normal, light))
        
        color = self.model.retrieve_diffuse(uv[0], uv[1])

        # ipdb.set_trace()

        color = tuple(int(c * diff) for c in color)
        
        return color

def main():

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
    
    model_view = lookat(eye, center, up)
    viewport = create_viewport(width/8, height/8, width*3/4, height*3/4, depth)
    projection = create_projection(f)

    image = Image.new("RGB", (width, height), black)
    z_buffer = np.full((width, height), -np.inf)

    obj_path = "/root/autodl-tmp/tiny-renderer/assets/african_head.obj"
    model = Model(obj_path)
    texture_path = "/root/autodl-tmp/tiny-renderer/assets/african_head_diffuse.tga"
    model.load_texture(texture_path)
    normal_tangent_path = "/root/autodl-tmp/tiny-renderer/assets/obj/african_head_nm_tangent.tga"
    model.load_normal_mapping(normal_tangent_path)

    shader = Gouraud_Shader(model, light_dir, viewport, projection, model_view, white)
    for face in tqdm(model.faces):
        projected_vertex = np.zeros((3, 4))
        for i in range(3):
            projected_vertex[i] = shader.vertex(face, i)

        triangle(projected_vertex, shader, image, z_buffer, viewport)

    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save("output4.1.png")
 

if __name__ == '__main__':
    main()