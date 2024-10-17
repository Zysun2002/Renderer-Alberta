# Alberta-Renderer
This project is part of my 3-month research internship program in the University of Alberta. 


## Introduction

This renderer is for learning purpose. You can use this project to learn how to build a Python renderer from scratch.

The renderer is a expension of [Dmitry V. Sokolov](https://github.com/ssloy/tinyrenderer)'s tutorial on computer graphics and rendering. Besides including all the features introduced in his tutorial, the **advance** made in this version is:
Along with all the features from his tutorial, this version has made some advancements:

1. Implemented in Python: I chose Python for this renderer because I plan to integrate it with other rendering techniques, like neural rendering, in the future. Plus, it's easier for me to work with non-graphics features in Python since I'm more familiar with it.
2. Multiple Object Support: This renderer can now handle scenes with multiple objects.
3. Support for OBJ and MTL Formats: I've added support for OBJ and MTL file formats since these are widely available online, allowing the renderer to work with more 3D assets."

Though implemented using python, the renderer does not rely on high-level graphics library during rendering process. I gain fundamental knowledge about rendering and computer graphics during building my renderer step by step.

<img src="https://github.com/Zysun2002/Renderer-Alberta/blob/main/demo_assets/Pipeline.png" alt="pipeline" width="800" height="400"/>

## Instruction guide


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

You only have to assign the mtl file path to allow OBJ + MTL rendering.
```
mtl_path = PATH_TO_YOUR_MTL_FILE
model.load_mtl(mtl_path)
```

### Real-time visualization

Even using Python, building a renderer still feels harder than writing simpler code. I always try to make it easier and more intuitive to code and debug. That’s why I visualize the rendering process in real-time—it helps me understand the rendering sequence better.

pumpkin          |  tanks
:-------------------------:|:-------------------------:
<img src="https://github.com/Zysun2002/Renderer-Alberta/blob/main/demo_assets/pumpkin.gif" width="300"> |  <img src="https://github.com/Zysun2002/Renderer-Alberta/blob/main/demo_assets/tanks.gif" width="300">



<img src="https://github.com/Zysun2002/Renderer-Alberta/blob/main/demo_assets/pikachu.gif" width="300">

To get the above results, simply use:
```
python -m ipdb render_texture.py --obj_path ../assets/pumpkin/pumpkin_.obj --mtl_path ../assets/pumpkin/pumpkin_.mtl --camera_position 0 -65 -200 --lookat 0 -67 -198.3 --up_direction 1 -1 -1 --light_direction -2 2 1 --width 800 --height 800
```
and
```
python -m ipdb render_texture.py --obj_path ../assets/tanks/OBJ_7050.obj --mtl_path ../assets/tanks/OBJ_7050.mtl --camera_position -3 4 30 --lookat 1 1 25 --up_direction 0 1 0 --light_direction 1 -1 1 --width 800 --height 800
```

## Todo 

1. Some additional features have not been integrated into the mainstream program.
 
2. Python makes non-graphics tasks easier—like handling command interfaces and using simple data structures. However, the core functions should still run like C for better speed. My solution is to write the core rendering part in CUDA for speed, while using Python to manage the peripheral parts.

3. While working with MTL file, I found there could be over 10 rendering options which consider more aspects and are thus more complex. To achieve more photo-realistic results, to implement these features is always meaningful and more importantly, interesting!


## Appendix

This is my second visit to the University of Alberta. I really enjoy my time here!

<img src="https://github.com/user-attachments/assets/ff284f99-b100-4782-b07e-e0cec28ad572" alt="poster fair" width="400" height="300"/>
<img src="https://github.com/user-attachments/assets/39145eb4-3d7e-4bbd-b364-68452094053f" width="400" height="300"/>

A presentation in the poster fair and the most inspiring sentence I met in UofA.

**You can review my poster [here](https://github.com/Zysun2002/Renderer-Alberta/blob/main/poster_ZiyuSun.pdf).**

