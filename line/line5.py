from PIL import Image
import time

def line(x0, y0, x1, y1, image, color):
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


def main(): 

    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (255, 0, 0)
    width, height = 100, 100

    image = Image.new("RGB", (width, height), black)
    # line(13, 20, 80, 40, image, white)

    start = time.time()
    for _ in range(1000):
        line(20, 13, 40, 80, image, red)
        line(80, 40, 13, 20, image, red)
    end = time.time()
    print("plotting time:", -start + end)
    
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save("output.png")




if __name__ == '__main__':
    main()