# Alberta-Renderer
This project is part of my 3-month research internship program in the University of Alberta. 


## Introduction

This renderer is for learning purpose. You can use this project to learn how to build a Python renderer from scratch.

The renderer is an expension of several fantastic tutorials on [rasterization](https://github.com/ssloy/tinyrenderer) and [ray tracing](https://raytracing.github.io/). I chose Python for this renderer because I plan to integrate it with other rendering techniques, like neural rendering, in the future. Plus, it's easier for me to work with non-graphics features in Python. This renderer can also support multiple objects, MTL format and some othter features not included in the above-mentioned tutorials. 

Though implemented using python, the renderer does not rely on high-level graphics library during rendering process. I gain fundamental knowledge about rendering and computer graphics during building my renderer step by step.

Rasterization:

<img src="https://github.com/Zysun2002/Renderer-Alberta/blob/main/demo_assets/rasterization.png" alt="pipeline" width="600" height="300"/>

Ray tracing:

<img src="https://github.com/Zysun2002/Renderer-Alberta/blob/main/demo_assets/ray_tracing.png" alt="pipeline" width="600" height="300"/>

## Instruction guide

### Rasterization

1. clone the repo and build the environment. Thanks to Python's straightforward packaging, using the renderer is easy and efficient.
```
git clone https://github.com/Zysun2002/Renderer-Alberta.git
cd Renderer-Alberta
pip install -r requirements.txt
```

2. download the 3D assets from [google drive](https://drive.google.com/drive/folders/16McfVknOWoH1g2q_mMy05sRhQjE2ylHD?usp=drive_link) or prepare your own data.
```
mkdir assets/
```
and put 3D assets in this folder.

3. change the path to texture mapping in MTL file if applicable.
   
4. assign rendering parameter and do the rasterization.
```
cd integration
python render.py  --obj_path OBJ_PATH --mtl_path MTL_PATH --camera_position CAMERA_COORDINATES --lookat CENTER_COORDINATES --up_direction UP_DIRECTION --light_direction LIGHT_DIRECTION
```

| parameter | explanation | if required |
|:-------------|:--------------:|--------------:|
| --obj_path     | path to the .obj file | required   |
| --mtl_path     | path to the .mtl file  | required  |
| --output_path     | path to the rendering (one image only)  | optional  |
| --camera_position  | coordinates of the camera | required   |
| --lookat     | coordinates of the screen center  | required  |
| --up_direction     | up direction of the camera | required  |
| --width     | width of the canvas | optional, default 800  |
| --height     | height of the canvas | optional, default 800  |
| --background     | background color (rgb) | optional, default(0, 0, 0)  |
| --Ca     | intensity of ambient light | optional, default 0.2  |
| --Cd     | intensity of diffuse light | optional, default 0.9  |
| --Cs     | intensity of specular light | optional, default 0.8  |
| --allow_vis     | to allow real-time rendering visualization | optional, default False |
| --vis_iter     | frequenct of real-time visualization | optional, default 100  |
| --vis_path     | path to save real-time visualization results | optional, default "output"  |

Some examples can be found in Real-time visualization section.

You can also test the simplified version on [MecSimCalc](https://mecsimcalc.com/app/4328169). However, due to uploaded file size limitations, only basic rendering is avalable.

### Ray tracing

The ray tracing pipeline is still under renconstruction. However, you can still try the current version.
```
python ray_tracing.py
```


## Selected Features

### Ambient and specular light
I really love how ambient and specular lighting make materials look more realistic. By calculating the angle between the reflection and the viewer's direction, we can create the specular highlight.

You can adjust the intensity ratio of the three lights by setting the following parameters:
```
cd mtl_parse
python main.py --ratio_Ka YOUR_RATIO_KA --ratio_KD YOUR_RATIO_KD --ratio_Ks YOUR_RATIO_KS
```
<img src="https://github.com/Zysun2002/Renderer-Alberta/blob/main/demo_assets/specular_light.png" alt="ablation" width="800"/>


### Shadow mapping technique
A great way to boost the photo-realism of rendering is by getting more accurate shadows. By using a two-pass rendering process, we can pre-bake shadow information with shadow mapping before the actual rendering happens.

<img src="https://github.com/Zysun2002/Renderer-Alberta/blob/main/demo_assets/shadow_mapping.png" alt="ablation" width="800"/>

If you want to use shadow mapping, run the following commands:
```
cd shadow_mapping
python main.py
```
### Handle OBJ+MTL format
3D assets are everywhere online, but when I tried out Dmitry V. Sokolov's renderer, I had some trouble rendering a lot of the common formats available in online libraries. To fix this, I updated my renderer to support the OBJ + MTL format, which is super common on the Internet. This enhancement lets the renderer handle most of the 3D assets you can find online.

### Real-time visualization

Even using Python, building a renderer still feels harder than writing simpler code. I always try to make it easier and more intuitive to code and debug. That’s why I visualize the rendering process in real-time—it helps me understand the rendering sequence better.

pumpkin     | pan     |  tanks      | pikachu | fish
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/Zysun2002/Renderer-Alberta/blob/main/demo_assets/pumpkin.gif" width="300"> |<img src="https://github.com/Zysun2002/Renderer-Alberta/blob/main/demo_assets/pan.gif" width="300">|  <img src="https://github.com/Zysun2002/Renderer-Alberta/blob/main/demo_assets/tanks.gif" width="300"> | <img src="https://github.com/Zysun2002/Renderer-Alberta/blob/main/demo_assets/pikachu.gif" width="300"> | <img src="https://github.com/Zysun2002/Renderer-Alberta/blob/main/demo_assets/fish.gif" width="300">

To get the above results, the corresponding commands are:
```
python  render.py --obj_path ../assets/pumpkin/pumpkin.obj --mtl_path ../assets/pumpkin/pumpkin.mtl --camera_position 0 -65 -200 --lookat 0 -67 -198 --up_direction 0 -1 -1 --light_direction -1 -1 1 --width 800 --height 800 --allow_vis --vis_path pumpkin
```
```
python render.py --obj_path ../assets/tanks/tanks.obj --mtl_path ../assets/tanks/tanks.mtl --camera_position -3 4 30 --lookat 1 1 25 --up_direction 0 1 0 --light_di
rection 3 -4 -30 --width 800 --height 800 --allow_vis --vis_path tanks
```
```
python render.py --obj_path ../assets/pikachu/pikachu.obj --mtl_path ../assets/pikachu/pikachu.mtl --camera_position -0.3 0.5 0.6 --lookat 0 0.5 -0.4 --up_direction 0 1 0  --light_direction  0.1 -0.4 -0.5 --width 800 --height 800 --allow_vis --vis_path pikachu
```
```
python render.py  --obj_path ../assets/pan/pan.obj --mtl_path ../assets/pan/pan.mtl --camera_position 0 0 -60 --lookat -1.2 0 -58.5 --up_direction 0 1 -1 --light_direction 0 1 1 -width 800 --height 800 --allow_vis --vis_path pan
```
and 
```
python render.py --obj_path ../assets/fish/fish.obj --mtl_path ../assets/fish/fish.mtl --camera_position 0 0 20 --lookat 0 1.5  19 --up_direction 0 1 0 --light_direction 1 1 1 --allow_vis --vis_path fish
```
### Ray tracing pipeline
I am currently adding a pipeline based on ray tracing. Here are the results.
distant view  |close-up view(focus)     | close-up view(defocus)   
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/Zysun2002/Renderer-Alberta/blob/main/demo_assets/ray-tracing-demo.png" width="400">|<img src="https://github.com/Zysun2002/Renderer-Alberta/blob/main/demo_assets/focus_near.png" width="400"> |<img src="https://github.com/Zysun2002/Renderer-Alberta/blob/main/demo_assets/defocus_near.png" width="400">



## Todo 

1. Some additional features have not been integrated into the mainstream program, including some optimization for rasterization and the ray tracing function.
 
2. Python makes non-graphics tasks easier—like handling command interfaces and using simple data structures. However, the core functions should still run like C for better speed. My solution is to write the core rendering part in CUDA for speed, while using Python to manage the peripheral parts.

3. While working with MTL file, I found there could be over 10 rendering options which consider more aspects and are thus more complex. To achieve more photo-realistic results, to implement these features is always meaningful and more importantly, interesting!


## Appendix

This is my second visit to the University of Alberta. I really enjoy my time here!

<img src="https://github.com/user-attachments/assets/ff284f99-b100-4782-b07e-e0cec28ad572" alt="poster fair" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/39145eb4-3d7e-4bbd-b364-68452094053f" width="400" height="300"/>

A presentation in the poster fair and the most inspiring sentence I met in UofA.

**You can review my poster [here](https://github.com/Zysun2002/Renderer-Alberta/blob/main/poster_ZiyuSun.pdf).**

