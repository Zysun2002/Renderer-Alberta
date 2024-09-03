from PIL import Image
import numpy as np
import time

def barycentric(pts, p):
    u = np.cross(np.array((pts[2, 0]-pts[0, 0], pts[1, 0]-pts[0, 0], pts[0, 0]-p[0])),
                np.array((pts[2, 1]-pts[0, 1], pts[1, 1]-pts[0, 1], pts[0, 1]-p[1])))
    # print(u)
    if(abs(u[2]) < 1): return np.array((-1, 1, 1))
    return np.array((float(1) - float(u[0]+u[1])/u[2], float(u[0])/u[2], float(u[1])/u[2]))


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

    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)

    width, height = 200, 200

    render = Image.new("RGB", (width, height), black)
    # line(13, 20, 80, 40, image, white)
    z_buffer = np.full((width, height), -np.inf)

    t0 = np.array([[10, 70, 0], [50, 160, 0], [70, 80, 0]])
    t1 = np.array([[180, 50, 1], [150, 1, 1], [70, 180, 1]])
    t2 = np.array([[180, 150, 2], [150, 10, 2], [60, 130, 2]])

    triangle(t0, render, red,  z_buffer)
    triangle(t1, render, green, z_buffer)
    triangle(t2, render, blue, z_buffer)

    render = render.transpose(Image.FLIP_TOP_BOTTOM)
    render.save("output2.png")




if __name__ == '__main__':
    main()