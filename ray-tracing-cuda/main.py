import gpu_library

import numpy as np
import ipdb
from tqdm import tqdm
from PIL import Image
import argparse


def main(args):

    config = {}
    

    config["width"] = args.width
    config["height"] = args.height
    config["samples_per_pixel"] =args.samples_per_pixel
    config["max_depth"] = args.max_depth

    
    config["lookfrom"] = args.lookfrom
    config["lookat"] = args.lookat
    config["up"] = args.up_direction

    config["aperture"] = 0.1

    image_array = np.zeros((config["height"], config["width"], 3) , dtype=np.float32)

    duration = gpu_library.render(image_array, config)
    
    image_array = (image_array*255).astype(np.uint8)

    image = Image.fromarray(image_array).transpose(Image.FLIP_TOP_BOTTOM)
    image.save("output.png")   

    print("duration:", duration)

     

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./output.png")

    parser.add_argument("--lookfrom", type=float, nargs="+", required=True)
    parser.add_argument("--lookat", type=float, nargs="+", required=True)
    parser.add_argument("--up_direction", type=float, nargs="+", required=True)

    parser.add_argument("--samples_per_pixel", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)


    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)

    args = parser.parse_args()

    main(args)