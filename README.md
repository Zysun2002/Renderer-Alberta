# Alberta-Renderer
This project is part of my 3-month research internship program in the University of Alberta. 


## Introduction

This renderer is for learning purpose. You can use this project to learn how to build a Python renderer from scratch.

The renderer is a expension of [Dmitry V. Sokolov](https://github.com/ssloy/tinyrenderer)'s tutorial on computer graphics and rendering. Besides including all the features introduced in his tutorial, the advance made in this version is:
Along with all the features from his tutorial, this version has made some advancements:

1. Implemented in Python: I chose Python for this renderer because I plan to integrate it with other rendering techniques, like neural rendering, in the future. Plus, it's easier for me to work with non-graphics features in Python since I'm more familiar with it.
2. Multiple Object Support: This renderer can now handle scenes with multiple objects.
3. Support for OBJ and MTL Formats: I've added support for OBJ and MTL file formats since these are widely available online, allowing the renderer to work with more 3D assets."

Though implemented using python, the renderer does not rely on high-level graphics library during rendering process. I gain fundamental knowledge about rendering and computer graphics during building my renderer step by step.

## Selected Features：

### Ambient and specular light
I really love how ambient and specular lighting make materials look more realistic. By calculating the angle between the reflection and the viewer's direction, we can create the specular highlight.

You can adjust the intensity ratio of the three lights by setting the following parameters:
```
cd mtl_parse
python main.py --ratio_Ka YOUR_RATIO_KA --ratio_KD YOUR_RATIO_KD --ratio_Ks YOUR_RATIO_KS
```

### Shadow mapping technique
A great way to boost the photo-realism of rendering is by getting more accurate shadows. By using a two-pass rendering process, we can pre-bake shadow information with shadow mapping before the actual rendering happens.

If you want to use shadow mapping, run the following commands:
```
cd shadow_mapping
python main.py
```
### Handle MTL format
3D assets are everywhere online, but when I tried out Dmitry V. Sokolov's renderer, I had some trouble rendering a lot of the common formats available in online libraries. To fix this, I updated my renderer to support the OBJ + MTL format, which is super common on the Internet. This enhancement lets the renderer handle most of the 3D assets you can find online.

### Appendix


<img src="https://github.com/user-attachments/assets/ff284f99-b100-4782-b07e-e0cec28ad572" alt="poster fair" width="500"/>

A presentation in the poster fair organized by UofA. **You can review my poster [here](https://github.com/Zysun2002/Renderer-Alberta/blob/main/poster_ZiyuSun.pdf).**

![weareworldshapers](https://github.com/user-attachments/assets/39145eb4-3d7e-4bbd-b364-68452094053f)
The sentence that inspired me most.
