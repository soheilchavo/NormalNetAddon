import bpy
from bpy.types import Node, NodeTree
from bpy.props import StringProperty, PointerProperty
from bpy.types import Operator

import torch

import cv2
import cv2.ximgproc as xip
import numpy as np

from torchvision import transforms
from PIL import Image

import pickle
import os

from torch import nn

img_transform = transforms.Compose([transforms.PILToTensor()])
ADDON_DIR = os.path.dirname(os.path.abspath(__file__))


def single_pass(model, input_tensor, guide_tensor, device, dataset_info, output_path):
    with open(dataset_info, 'rb') as f:
        values = pickle.load(f)

    dataset_mean = values[0]
    dataset_std = values[1]

    model = model.to(device)
    input_tensor = input_tensor.to(device)

    result = model(input_tensor)

    result = result.detach()
    result = result.to(torch.device("cpu"))
    result = unnormalize_tensor(result, dataset_mean, dataset_std)

    result = result.detach().numpy()
    result = result.squeeze(0)

    guide_tensor = guide_tensor.detach().numpy()

    if guide_tensor.shape[0] == 4:
        guide_tensor = guide_tensor[:3, :, :]

    if result[0].mean() > 1:
        result /= 255
        
    result = joint_bilateral_up_sample(result, guide_tensor, save_img=True, output_path=output_path)

    return result


# Returns an image's corresponding tensor
def transform_single_png(sample):
    img = img_transform(Image.open(sample))
    dim = min(img.shape[1], img.shape[2])
    square_transform = transforms.Compose([transforms.Resize((dim, dim))])
    return square_transform(img)


def scale_transform_sample(datapoint, standalone=False):
    # Standalone is true if the function is called on a single image rather than as a part of a dataset
    if standalone:
        datapoint = datapoint.float() / 255

    # Intial Size transform (256, 256) in order to save on processing, images get upscaled later
    transform1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Removes alpha channel if image has one
    if datapoint.shape[0] == 4:
        datapoint = datapoint[:3, :, :]

    # Size transform
    datapoint = transform1(datapoint)

    # Add batch dimension if single sample doesen't have it
    if standalone and len(datapoint.shape) == 3:
        datapoint = datapoint.unsqueeze(0)

    return datapoint


def normalize_sample(datapoint, mean, std):
    normal_transform = transforms.Compose([
        transforms.Normalize(mean=mean, std=std)
    ])

    return normal_transform(datapoint)


def normalize_data(dataset):
    mean = 0.0
    std = 0.0

    for datapoint in dataset:
        datapoint[0] = datapoint[0].float() / 255
        datapoint[1] = datapoint[1].float() / 255
        mean += datapoint[0].mean() + datapoint[1].mean()
        std += datapoint[0].std() + datapoint[1].std()

    mean /= len(dataset) * 2
    std /= len(dataset) * 2

    normalized_dataset = []

    for datapoint in dataset:
        scaled_datapoints = [scale_transform_sample(datapoint[0]), scale_transform_sample(datapoint[1])]
        normalized_dataset.append(
            [normalize_sample(scaled_datapoints[0], mean, std), normalize_sample(scaled_datapoints[1], mean, std)])

    return normalized_dataset, mean, std


def unnormalize_tensor(sample, mean, std):
    return sample * std + mean


def joint_bilateral_up_sample(low_res, guide, d=5, sigma_color=0.1, sigma_space=2.0, save_img=False, output_path=""):
    low_res = np.transpose(low_res, (1, 2, 0))
    guide = np.transpose(guide, (1, 2, 0))
    guide = np.float32(guide)

    new_width = guide.shape[0]
    new_height = guide.shape[1]

    up_scaled_f = cv2.resize(low_res, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    filtered_f = xip.jointBilateralFilter(guide, up_scaled_f, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    out = np.clip(filtered_f * 255.0, 0, 255).astype(np.uint8)

    if save_img:
        cv2.imwrite(output_path, out)

    return out


def create_image_from_ndarray(name, arr):
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()

    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
        arr /= 255.0

    if arr.shape[0] == 4 or arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))

    with open("log.txt", "a") as f:
        string_tensor = ','.join(str(val) for val in arr.shape)
        f.write(string_tensor)
        f.write('\n')

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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x2 = self.conv(x)
        return x2


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.conv = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        x2 = self.conv(x)
        return x2


class UpConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.transpose_conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, y):
        x2 = self.transpose_conv(x)
        x3 = torch.cat([x2, y], dim=1)
        out = self.double_conv(x3)
        return out


# Simplified U-Net structure (in order to save on training time)
class UNet(torch.nn.Module):
    def __init__(self, n_channels):
        super(UNet, self).__init__()
        self.n_channels = n_channels

        self.initial_conv = DoubleConv(n_channels, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)

        self.up1 = UpConv(256, 128)
        self.up2 = UpConv(128, 64)
        self.final_conv = nn.Conv2d(64, n_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.initial_conv(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x4 = self.up1(x3, x2)
        x6 = self.up2(x4, x1)

        x7 = self.final_conv(x6)
        return x7


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