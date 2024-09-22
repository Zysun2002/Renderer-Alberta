import numpy as np
import sys
sys.path .append("/root/autodl-tmp/tiny-renderer")
from PIL import Image

def line(vert0, vert1, image, color):
    x0, y0= vert0[0], vert0[1]
    x1, y1 = vert1[0], vert1[1]
    
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
    derror2 = int(abs(dy)*2)
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


def triangle(verts, image, color):
    vert0, vert1, vert2 = verts[0], verts[1], verts[2]

    if vert0[1] == vert1[1] and vert1[1] == vert2[1]: return
    
    if vert0[1] > vert1[1]: vert0, vert1 = vert1, vert0
    if vert0[1] > vert2[1]: vert0, vert2 = vert2, vert0
    if vert1[1] > vert2[1]: vert1, vert2 = vert2, vert1

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
    
        if A[0] > B[0]: A[0], B[0] = B[0], A[0]
        for x in range(A[0], B[0]+1):
            image.putpixel((x, i + vert0[1]), color)

    # for y in range(vert0[1], vert1[1]+1):
    #     segment_height = vert1[1] - vert0[1] + 1
    #     alpha = float(y-vert0[1])/total_height
    #     beta = float(y-vert0[1])/segment_height
    #     A = (vert0 + (vert2 - vert0) * alpha).astype(int)
    #     B = (vert0 + (vert1 - vert0) * beta).astype(int)
            
    #     if A[0] > B[0]: A[0], B[0] = B[0], A[0]
    #     for x in range(A[0], B[0]+1):
    #         image.putpixel((x, y), color)
    #     # line(A, B, image, color)

    # for y in range(vert1[1], vert2[1]):
    #     segement_height = vert2[1] - vert1[1] + 1
    #     alpha = float(y - vert0[1]) / total_height
    #     beta = float(y - vert1[1]) / segement_height
    #     A = (vert0 + (vert2 - vert0) * alpha).astype(int)
    #     B = (vert1 + (vert2 - vert1) * beta).astype(int)
    #     # line(A, B, image, color)
    #     if A[0] > B[0]: A[0], B[0] = B[0], A[0]
    #     for x in range(A[0], B[0]):
    #         image.putpixel((x, y), color)


def main():
    red = (255, 0, 0)
    
    black = (0, 0, 0)
    white = (255, 255, 255)
    green = (0, 255, 0)

    height, width = 200, 200
    image = Image.new("RGB", (width, height), black)


    # t0 = Triangle2(Vert2((10, 70)), Vert2((50, 160)), Vert2((70, 80)))
    # t1 = Triangle2(Vert2((180, 50)), Vert2((150, 1)), Vert2((70, 180)))
    # t2 = Triangle2(Vert2((180, 150)), Vert2((120, 160)), Vert2((130, 180)))

    t0 = np.array([[10, 70], [50, 160], [70, 80]])
    t1 = np.array([[180, 50], [150, 1], [70, 180]])
    t2 = np.array([[180, 150], [120, 160], [130, 180]])

    triangle(t0, image, green)
    triangle(t1, image, red)
    triangle(t2, image, red)

    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save("output.png")

if __name__ == '__main__':
    main()