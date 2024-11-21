import numpy as np
import sys
sys.path.append("/root/autodl-tmp/tiny-renderer")
from groups import norm, Model, embed_v
from PIL import Image
from tqdm import tqdm
import ipdb

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

def create_projection(f):

    # return np.identity(4)

    # ipdb.set_trace()
    return np.array(((1, 0, 0, 0), 
                     (0, 1, 0, 0), 
                     (0, 0, 1, 0), 
                     (0, 0, -1./f, 1)))

def triangle(verts, intensity, image, z_buffer):
    vert0, vert1, vert2 = verts[0], verts[1], verts[2]
    ity0, ity1, ity2 = intensity[0], intensity[1], intensity[2]

    if vert0[1] == vert1[1] and vert1[1] == vert2[1]: return
    
    if vert0[1] > vert1[1]: vert0, vert1 = vert1, vert0; ity0, ity1 = ity1, ity0
    if vert0[1] > vert2[1]: vert0, vert2 = vert2, vert0; ity0, ity2 = ity2, ity0
    if vert1[1] > vert2[1]: vert1, vert2 = vert2, vert1; ity1, ity2 = ity2, ity1

    total_height = vert2[1] - vert0[1]
    for i in range(total_height):
        second_half = i > vert1[1] - vert0[1] or vert1[1] == vert0[1]
        segment_height = vert2[1] - vert1[1] if second_half else vert1[1] - vert0[1]
        alpha = float(i) / total_height
        beta = float(i - vert1[1] + vert0[1]) / segment_height if second_half else \
            float(i) / segment_height
        
        A = (vert0 + (vert2 - vert0) * alpha).astype(int)
        B = (vert1 + (vert2 - vert1) * beta).astype(int) if second_half else \
            (vert0 + (vert1 - vert0) * beta).astype(int)
        
        ityA = ity0 + (ity2 - ity0) * alpha
        ityB = ity1 + (ity2 - ity1) * beta if second_half else\
               ity0 + (ity1 - ity0) * beta
        
        if A[0] > B[0]: 
            A, B = B, A
            ityA, ityB = ityB, ityA

        for x in range(A[0], B[0]+1):
            phi = 1. if B[0] == A[0] else float(x - A[0]) / (B[0] - A[0])
            P = (A + (B - A) * phi).astype(int)
            ityP = ityA + (ityB - ityA) * phi
            if P[0] >= image.size[0] or P[1] >= image.size[1] or P[0] < 0 or P[1] < 0: continue

            if z_buffer[P[0], P[1]] < P[2]:
                z_buffer[P[0], P[1]] = P[2]
                color = tuple(int(x * ityP) for x in (255, 255, 255))
                image.putpixel((P[0], P[1]), color)

        
def main():

    width, height = 800, 800

    depth = 255

    black = (0, 0, 0)

    light_dir = norm(np.array((1,-1,1)))
    eye = np.array((1, 1, 3))
    center = np.array((0,0,0))
    up = np.array((0,1,0))

    f = np.linalg.norm(eye - center)
    
    model_view = lookat(eye, center, up)
    viewport = create_viewport(width/8, height/8, width*3/4, height*3/4, depth)
    # viewport = create_viewport(width/8, height/8, width*3/4, height*3/4, depth)
    projection = create_projection(f)
    # projection = np.identity(4)
    # ipdb.set_trace()
    image = Image.new("RGB", (width, height), black)
    z_buffer = np.full((width, height), -np.inf)

    obj_path = "/root/autodl-tmp/tiny-renderer/assets/african_head.obj"
    model = Model(obj_path)

    for face in tqdm(model.faces):
        verts = model.verts[face]
        v_normals = model.v_normals[face]
        screen_coords = np.zeros((3, 3)).astype(int)
        # ipdb.set_trace()
        intensity = np.zeros((3,)).astype(float)
        # ipdb.set_trace()
        for i in range(3):
            coords = viewport @ projection @ model_view @ embed_v(verts[i])
            screen_coords[i] = (coords[:3] / coords[-1]).astype(int)   
            ipdb.set_trace()                                                 
            intensity[i] = np.dot(v_normals[i], light_dir) 
        # ipdb.set_trace()
        triangle(screen_coords, intensity, image, z_buffer)

    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save("output1.png")


if __name__ == '__main__':
    main()