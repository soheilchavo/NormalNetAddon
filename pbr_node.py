import bpy
from bpy.types import Node, NodeTree
from bpy.props import StringProperty, PointerProperty
from bpy.types import Operator

import torch
import numpy as np
import os

from TensorFunctions import transform_single_png, scale_transform_sample
from Models import single_pass, UNet

ADDON_DIR = os.path.dirname(os.path.abspath(__file__))

def create_image_from_ndarray(name, arr):
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()

    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
        arr /= 255.0

    if arr.shape[0] == 4 or arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))

    height = arr.shape[0]
    width = arr.shape[1]

    if arr.shape[2] == 3:
        arr = np.concatenate(
            [arr, np.ones((height, width, 1), dtype=arr.dtype)],
            axis=-1
        )

    # Create the new image data block
    image = bpy.data.images.new(name, width=width, height=height)

    flat_data = arr.ravel()
    image.pixels = flat_data

    return image

def create_image_texture_node(node_tree, image, node_name="Generated Texture"):
    tex_node = node_tree.nodes.new('ShaderNodeTexImage')
    tex_node.name = node_name
    tex_node.label = node_name
    tex_node.image = image
    return tex_node


# Optional: custom socket types, but you can also use built-in ones.
class PBRGeneratorSocket(bpy.types.NodeSocket):
    bl_idname = "PBRGeneratorSocket"
    bl_label = "PBR Generator Socket"

    # For demonstration, let's just store a color
    # You can store typical data like float, vector, color, etc.
    default_value: bpy.props.FloatVectorProperty(
        name="Color",
        subtype='COLOR',
        default=(0.0, 0.0, 0.0),
        min=0.0, max=1.0
    )

    def draw(self, context, layout, node, text):
        layout.prop(self, "default_value", text=text)

    def draw_color(self, context, node):
        return (0.2, 0.1, 0.9, 1.0)  # RGBA color for the socket icon


class PBRGeneratorNode(Node):
    """A custom node to generate PBR maps from a diffuse image."""
    bl_idname = "PBRGeneratorNodeType"
    bl_label = "PBR Generator"
    bl_icon = 'NODE_TEXTURE'

    image_path: StringProperty(
        name="Image Path",
        subtype='FILE_PATH',
        default="",
        description="File path to the input image"
    )

    def init(self, context):
        # OUTPUTS:
        # Albedo
        self.outputs.new("NodeSocketColor", "Albedo")

    def draw_buttons(self, context, layout):
        layout.prop(self, "image_path", text="Image")
        layout.operator("node.generate_pbr_maps", text="Generate")

    def draw_label(self):
        return "PBR Generator"

    def process_image(self, diffuse_image):
        albedo = transform_single_png(diffuse_image)
        down_sample = scale_transform_sample(albedo, standalone=True)

        AOGenerator = UNet(3)
        AOGenerator.load_state_dict(
            torch.load(os.path.join(ADDON_DIR, "ModelsStateDict/AmbientOcclusion_SD.pt")))
        DisplacementGenerator = UNet(3)
        DisplacementGenerator.load_state_dict(
            torch.load(os.path.join(ADDON_DIR, "ModelsStateDict/Displacement_SD.pt")))
        MetalnessGenerator = UNet(3)
        MetalnessGenerator.load_state_dict(
            torch.load(os.path.join(ADDON_DIR, "ModelsStateDict/Metalness_SD.pt")))
        NormalGLGenerator = UNet(3)
        NormalGLGenerator.load_state_dict(torch.load(os.path.join(ADDON_DIR, "ModelsStateDict/NormalGL_SD.pt")))
        RoughnessGenerator = UNet(3)
        RoughnessGenerator.load_state_dict(
            torch.load(os.path.join(ADDON_DIR, "ModelsStateDict/Roughness_SD.pt")))

        AODatasetInfo = ADDON_DIR + "/ModelInfo/AOTrainingDatasetInfo"
        DisplacementDatasetInfo = ADDON_DIR + "/ModelInfo/DisplacementTrainingDatasetInfo"
        NormalDatasetInfo = ADDON_DIR + "/ModelInfo/NormalGLTrainingDatasetInfo"
        RoughnessDatasetInfo = ADDON_DIR + "/ModelInfo/RoughnessTrainingDatasetInfo"
        MetalnessDatasetInfo = ADDON_DIR + "/ModelInfo/MetalnessTrainingDatasetInfo"

        device = "gpu" if torch.cuda.is_available() else "cpu"

        albedo_data = albedo
        roughness_data = single_pass(RoughnessGenerator, down_sample, albedo, device, RoughnessDatasetInfo,
                                     output_path="Roughness.png")
        normal_data = single_pass(NormalGLGenerator, down_sample, albedo, device, NormalDatasetInfo,
                                  output_path="Normal.png")
        metallic_data = single_pass(MetalnessGenerator, down_sample, albedo, device, MetalnessDatasetInfo,
                                    output_path="Metallic.png")
        ambient_occlusion_data = single_pass(AOGenerator, down_sample, albedo, device, AODatasetInfo,
                                             output_path="AO.png")
        displacement_data = single_pass(DisplacementGenerator, down_sample, albedo, device, DisplacementDatasetInfo,
                                        output_path="Displacement.png")

        return albedo_data, roughness_data, normal_data, metallic_data, ambient_occlusion_data, displacement_data


class PBRGeneratorOperator(Operator):
    """Operator to run the generative model and update node outputs"""
    bl_idname = "node.generate_pbr_maps"
    bl_label = "Generate PBR Maps"

    def execute(self, context):
        node = context.space_data.edit_tree.nodes.active
        if not isinstance(node, PBRGeneratorNode):
            self.report({'ERROR'}, "Active node is not a PBR Generator node.")
            return {'CANCELLED'}

        image_path = node.image_path
        if not image_path:
            self.report({'ERROR'}, "No image path specified.")
            return {'CANCELLED'}

        active_obj = context.view_layer.objects.active

        if active_obj and active_obj.active_material:
            material_name = active_obj.active_material.name
        else:
            material_name = "Null"

        # RUN THE MODEL:
        albedo_data, roughness_data, normal_data, metallic_data, ao_data, displacement_data = node.process_image(
            image_path)

        node_tree = context.space_data.edit_tree
        albedo_node = create_image_texture_node(node_tree,
                                                create_image_from_ndarray(f"{material_name} Albedo", albedo_data),
                                                "Albedo Map")
        ao_node = create_image_texture_node(node_tree, create_image_from_ndarray(f"{material_name} AO", ao_data),
                                            "Generated Ambient Occlusion Map")
        normal_node = create_image_texture_node(node_tree,
                                                create_image_from_ndarray(f"{material_name} NormalGL", normal_data),
                                                "Generated Normal Map")
        metallic_node = create_image_texture_node(node_tree,
                                                  create_image_from_ndarray(f"{material_name} Metallic", metallic_data),
                                                  "Generated Metallic Map")
        roughness_node = create_image_texture_node(node_tree, create_image_from_ndarray(f"{material_name} Roughness",
                                                                                        roughness_data),
                                                   "Generated Roughness Map")
        displacement_node = create_image_texture_node(node_tree,
                                                      create_image_from_ndarray(f"{material_name} Displacement",
                                                                                displacement_data),
                                                      "Generated Displacement Map")
        # 1) Find (or create) a Principled BSDF node in the current node tree
        bsdf_node = None
        for n in node_tree.nodes:
            if n.type == 'BSDF_PRINCIPLED':
                bsdf_node = n
                break
        if not bsdf_node:
            # If no Principled BSDF node found, create one
            bsdf_node = node_tree.nodes.new('ShaderNodeBsdfPrincipled')
            bsdf_node.location = (400, 0)

        # 2) Create a Normal Map node for the normal texture
        normal_map_node = node_tree.nodes.new('ShaderNodeBump')
        normal_map_node.location = (normal_node.location.x + 200, normal_node.location.y)

        # 3) Set color space for non-color textures:
        normal_node.image.colorspace_settings.is_data = True
        metallic_node.image.colorspace_settings.is_data = True
        roughness_node.image.colorspace_settings.is_data = True
        ao_node.image.colorspace_settings.is_data = True
        displacement_node.image.colorspace_settings.is_data = True

        # 4) Connect the newly created Image Texture nodes to the BSDF:
        links = node_tree.links

        # Albedo -> Base Color
        links.new(albedo_node.outputs["Color"], bsdf_node.inputs["Base Color"])

        # Metallic -> Metallic
        links.new(metallic_node.outputs["Color"], bsdf_node.inputs["Metallic"])

        # Roughness -> Roughness
        roughness_math_node = node_tree.nodes.new("ShaderNodeMath")
        roughness_math_node.label = "Roughness Scale"
        roughness_math_node.operation = 'MULTIPLY'
        roughness_math_node.inputs[1].default_value = 2.0  # factor to multiply roughness by
        roughness_math_node.location = (roughness_node.location.x + 200, roughness_node.location.y)


        # Connect the texture to the math node
        links.new(roughness_node.outputs["Color"], roughness_math_node.inputs[0])
        # Connect the math node to the Principled BSDF
        links.new(roughness_math_node.outputs["Value"], bsdf_node.inputs["Roughness"])

        # Normal -> Normal Map node -> BSDF Normal
        links.new(normal_node.outputs["Color"], normal_map_node.inputs["Height"])
        links.new(normal_map_node.outputs["Normal"], bsdf_node.inputs["Normal"])
        normal_map_node.inputs["Strength"].default_value = 0.2
        normal_map_node.inputs["Distance"].default_value = 1.0

        # AO multiply with Albedo
        mix_node = node_tree.nodes.new('ShaderNodeMixRGB')
        mix_node.blend_type = 'MULTIPLY'
        mix_node.inputs["Fac"].default_value = 1.0  # full effect
        mix_node.location = (albedo_node.location.x + 200, albedo_node.location.y - 200)

        links.new(albedo_node.outputs["Color"], mix_node.inputs["Color1"])
        links.new(ao_node.outputs["Color"], mix_node.inputs["Color2"])
        # Then connect that multiplied result to Base Color
        links.new(mix_node.outputs["Color"], bsdf_node.inputs["Base Color"])

        # Create the Shader Displacement node
        disp_shader_node = node_tree.nodes.new('ShaderNodeDisplacement')

        # Connect the displacement texture to the Displacement node
        links.new(displacement_node.outputs["Color"], disp_shader_node.inputs["Height"])
        # Optionally, you can set a midlevel or scale here:
        disp_shader_node.inputs["Scale"].default_value = 0.02
        disp_shader_node.inputs["Midlevel"].default_value = 0.50

        # Optionally rename Displacement node label
        disp_shader_node.label = "Displacement Node"

        # 1) Create a Color Ramp (ValToRGB) node
        color_ramp_node = node_tree.nodes.new("ShaderNodeValToRGB")
        color_ramp_node.label = "Displacement Color Ramp"
        color_ramp_node.location = (
            displacement_node.location.x + 300,
            displacement_node.location.y
        )

        # (Optional) Set up default ramp stops (black to white from 0 to 1)
        cr_elements = color_ramp_node.color_ramp.elements
        cr_elements[0].position = 0.0
        cr_elements[0].color = (0.0, 0.0, 0.0, 1.0)  # black
        cr_elements[1].position = 1.0
        cr_elements[1].color = (1.0, 1.0, 1.0, 1.0)  # white

        # 2) Connect the displacement texture's "Color" output to the ColorRamp's "Fac" input
        links.new(displacement_node.outputs["Color"], color_ramp_node.inputs["Fac"])

        # 3) Connect the ColorRamp output to the Displacement node's "Height"
        links.new(color_ramp_node.outputs["Color"], disp_shader_node.inputs["Height"])

        # 5) Arrange node locations so they don't overlap
        albedo_node.location = (-800, 200)
        ao_node.location = (-800, 0)
        metallic_node.location = (-800, -200)
        roughness_node.location = (-800, -400)
        roughness_math_node.location = (-500, -400)
        normal_node.location = (-800, -600)
        normal_map_node.location = (-500, -600)
        displacement_node.location = (-800, -800)
        disp_shader_node.location = (-500, -800)
        mix_node.location = (-400, 80)
        bsdf_node.location = (0, 0)

        # 6) Connect the BSDF to the Material Output node:
        output_node = None
        for n in node_tree.nodes:
            if n.type == 'OUTPUT_MATERIAL':
                output_node = n
                break
        if not output_node:
            output_node = node_tree.nodes.new('ShaderNodeOutputMaterial')
            output_node.location = (200, 0)

        links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])
        # -- Connect Displacement to the Material Output
        links.new(disp_shader_node.outputs["Displacement"], output_node.inputs["Displacement"])

        mat = None
        obj = context.view_layer.objects.active
        if obj and obj.active_material:
            mat = obj.active_material
            mat.cycles.displacement_method = 'DISPLACEMENT_AND_BUMP'

        # Create the Texture Coordinate node
        tex_coord_node = node_tree.nodes.new("ShaderNodeTexCoord")
        tex_coord_node.label = "Texture Coordinates"
        tex_coord_node.location = (-1200, 300)

        # Create the Mapping node
        mapping_node = node_tree.nodes.new("ShaderNodeMapping")
        mapping_node.label = "Mapping"
        mapping_node.location = (-1000, 300)

        # Hook up the TexCoord node to the Mapping node (using the "UV" output as an example)
        links.new(tex_coord_node.outputs["UV"], mapping_node.inputs["Vector"])

        # Now connect the Mapping output to each image texture node's 'Vector' input
        texture_nodes = [
            albedo_node,
            roughness_node,
            normal_node,
            metallic_node,
            ao_node,
            displacement_node
        ]
        for t_node in texture_nodes:
            links.new(mapping_node.outputs["Vector"], t_node.inputs["Vector"])

        self.report({'INFO'}, "PBR Maps generated successfully.")
        return {'FINISHED'}


classes = (
    PBRGeneratorSocket,
    PBRGeneratorNode,
    PBRGeneratorOperator,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # Make the node available in the Shader Editor 'Add' menu
    bpy.types.NODE_MT_add.append(menu_func)


def unregister():
    bpy.types.NODE_MT_add.remove(menu_func)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


def menu_func(self, context):
    self.layout.operator("node.add_node", text="PBR Generator", icon="NODE_TEXTURE").type = "PBRGeneratorNodeType"