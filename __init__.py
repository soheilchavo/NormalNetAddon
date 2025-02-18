bl_info = {
    "name": "PBR Generator Node",
    "author": "soheilchavo",
    "version": (1, 0, 3, 8),
    "blender": (3, 0, 0),
    "location": "Shader Editor > Add > PBR Generator",
    "description": "Generates PBR textures from a diffuse image using a generative model.",
    "category": "Node",
}

import bpy
from .import pbr_node

def register():
    pbr_node.register()

def unregister():
    pbr_node.unregister()

if __name__ == "__main__":
    register()
