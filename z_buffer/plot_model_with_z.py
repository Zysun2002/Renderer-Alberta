import numpy as np
import sys
sys.path .append("/root/autodl-tmp/tiny-renderer")
from PIL import Image
from tqdm import tqdm
from groups import Model
import random

def barycentric(pts, p):
    u = np.cross(np.array((pts[2, 0]-pts[0, 0], pts[1, 0]-pts[0, 0], pts[0, 0]-p[0])),
                np.array((pts[2, 1]-pts[0, 1], pts[1, 1]-pts[0, 1], pts[0, 1]-p[1])))
    if(abs(u[2]) < 1): return np.array((-1, 1, 1))
    return np.array((float(1) - float(u[0]+u[1])/u[2], float(u[1])/u[2], float(u[0])/u[2]))

def triangle(verts, image, color, z_buffer):
    bbox_min = np.array((image.width -1, image.height - 1))
    bbox_max = np.array((0, 0))
    clamp = np.array((image.width - 1, image.height - 1))

    for i in range(3):
        bbox_min[0] = np.maximum(0, np.minimum(bbox_min[0], verts[i][0]))
        bbox_min[1] = np.maximum(0, np.minimum(bbox_min[1], verts[i][1]))
        bbox_max[0] = np.minimum(clamp[0], np.maximum(bbox_max[0], verts[i][0]))
        bbox_max[1] = np.minimum(clamp[1] - 1, np.maximum(bbox_max[1], verts[i][1]))

    for x in range(bbox_min[0], bbox_max[0] + 1):
        for y in range(bbox_min[1], bbox_max[1] + 1):
            P = np.array((x, y, 0)).astype(float)
            bc_screen = barycentric(verts, P)
            if bc_screen[0] < 0 or bc_screen[1] < 0 or bc_screen[2] < 0: continue
            for i in range(3): P[2] += verts[i][2] * bc_screen[i]
            if (z_buffer[x, y] < P[2]):
                z_buffer[x, y] = P[2]
                image.putpixel((x, y), color)


def main():
    red = (255, 0, 0)
    
    black = (0, 0, 0)
    white = (255, 255, 255)
    green = (0, 255, 0)

    height, width = 200, 200
    image = Image.new("RGB", (width, height), black)

    obj_path = "/root/autodl-tmp/tiny-renderer/assets/african_head.obj"
    model = Model(obj_path)

    light_dir = np.array((0, 0, -1))

    z_buffer = np.full((width, height), -np.inf)

    for face in tqdm(model.faces):
        
        # ipdb.set_trace()
        verts = model.verts[face]
        screen_coords = np.zeros((3, 3))
        world_coords = np.zeros((3, 3))

        for i in range(3):
            screen_coords[i] = np.array((float(verts[i, 0]+1)*width/2, float(verts[i, 1]+1)*height/2, verts[i, 2])).astype(int)

        
        n = np.cross(verts[2] - verts[0], verts[1] - verts[0])
        n = n / np.linalg.norm(n)

        intensity = np.dot(n, light_dir)
        
        color = tuple(int(x * float(intensity)) for x in white)

        if intensity > 0:
            triangle(screen_coords, image, color, z_buffer)

    
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save("output3.png")

if __name__ == '__main__':
    main()