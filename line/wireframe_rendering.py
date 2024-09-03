from PIL import Image
import sys
sys.path.append("/root/autodl-tmp/tiny-renderer")
from groups import Model
from tqdm import tqdm

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



def main(): 

    black = (0, 0, 0)
    white = (255, 255, 255)
    width, height = 800, 800
    obj_path = "/root/autodl-tmp/tiny-renderer/assets/african_head.obj"

    image = Image.new("RGB", (width, height), black)

    model = Model(obj_path)

    for face in tqdm(model.faces):
        for i in range(3):
            v0 = model.verts[face[i]]
            v1 = model.verts[face[(i+1)%3]]
            # import ipdb; ipdb.set_trace()
            x0 = int((v0[0]+1.-1e-5)*width/2.)
            y0 = int((v0[1]+1.-1e-5)*height/2.)
            x1 = int((v1[0]+1.-1e-5)*width/2.)
            y1 = int((v1[1]+1.-1e-5)*height/2.)
            line(x0, y0, x1, y1, image, white)

    image = image.transpose(Image.FLIP_TOP_BOTTOM)

    image.save("output.png")




if __name__ == '__main__':
    main()