# import mecsimcalc as mesc
import base64
import io
import trimesh
import numpy as np
import ipdb

def interface(inputs):
    def parse_mesh(raw_obj):
        mesh = trimesh.load("/root/autodl-tmp/tiny-renderer/assets/african_head.obj") 
        return mesh

    mesh = parse_mesh(inputs['obj'])

    return {"mesh":mesh}

def main(inputs):
    
    raw_material = interface(inputs)

    
    mesh = raw_material['mesh']



    
    debug_info = mesh.vertices

    return {"debug_info": debug_info}

if __name__ == '__main__':
    inputs = {'obj': None}
    main(inputs)