from PIL import Image
import numpy as np
import time

def line(vert0, vert1, image, color):
    x0, y0, x1, y1 = vert0[0], vert0[1], vert1[0], vert1[1]
    
    steep = False

    if abs(x0-x1) < abs(y0-y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True
    
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = int(x1-x0)
    dy = int(y1-y0)
    derror2 = abs(dy)*2
    error2 = 0
    y = y0

    for x in range(x0, x1, 1):
        if steep:
            image.putpixel((y, x), color)
        else:
            image.putpixel((x, y), color)
        error2 += derror2
        if (error2 > dx):
            y += (1 if y1 > y0 else -1)
            error2 -= dx*2

    # optimize by removing any multiplication and division operations / reduce float operators
    # cost 1.625s / 1000times

def rasterize(vert0, vert1, image, color, y_buffer):
    if vert0[0] > vert1[0]: vert0, vert1 = vert1, vert0

    for x in range(vert0[0], vert1[0] + 1):
        t = float(x - vert0[0]) / (vert1[0] - vert0[0])
        y = vert0[1] * (1. - t) + vert1[1] * t
        if y_buffer[x] < y:
            y_buffer[x] = y
            image.putpixel((x, 0), color)

def main(): 

    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)

    width, height = 800, 1

    y_buffer = np.full((width), -np.inf)

    render = Image.new("RGB", (width, height), black)
    # line(13, 20, 80, 40, image, white)

    rasterize(np.array((20, 34)), np.array((744, 400)), render, red,  y_buffer)
    rasterize(np.array((120, 434)), np.array((444, 400)), render, green, y_buffer)
    rasterize(np.array((330, 463)), np.array((594, 200)), render, blue, y_buffer)

    render = render.transpose(Image.FLIP_TOP_BOTTOM)
    render.save("output.png")




if __name__ == '__main__':
    main()