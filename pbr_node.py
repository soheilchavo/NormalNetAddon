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

    result = joint_bilateral_up_sample(result, guide_tensor, save_img=True, output_path=output_path)

    return result

#Returns an image's corresponding tensor
def transform_single_png(sample):
    img = img_transform(Image.open(sample))
    dim = min(img.shape[1], img.shape[2])
    square_transform = transforms.Compose([transforms.Resize((dim, dim))])
    return square_transform(img)

def scale_transform_sample(datapoint, standalone=False):

    #Standalone is true if the function is called on a single image rather than as a part of a dataset
    if standalone:
        datapoint = datapoint.float() / 255

    #Intial Size transform (256, 256) in order to save on processing, images get upscaled later
    transform1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    #Removes alpha channel if image has one
    if datapoint.shape[0] == 4:
        datapoint = datapoint[:3, :, :]

    #Size transform
    datapoint = transform1(datapoint)

    #Add batch dimension if single sample doesen't have it
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

    mean /= len(dataset)*2
    std /= len(dataset)*2

    normalized_dataset = []

    for datapoint in dataset:
        scaled_datapoints = [scale_transform_sample(datapoint[0]), scale_transform_sample(datapoint[1])]
        normalized_dataset.append([normalize_sample(scaled_datapoints[0], mean, std), normalize_sample(scaled_datapoints[1], mean, std)])

    return normalized_dataset, mean, std

def unnormalize_tensor(sample, mean, std):
    return sample * std + mean

def joint_bilateral_up_sample(low_res, guide, d=5, sigma_color=0.1, sigma_space=2.0, save_img=False, output_path=""):

    low_res = np.transpose(low_res, (1,2,0))
    guide = np.transpose(guide, (1,2,0))
    guide = np.float32(guide)

    new_width = guide.shape[0]
    new_height = guide.shape[1]

    up_scaled_f = cv2.resize(low_res, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    filtered_f = xip.jointBilateralFilter(guide, up_scaled_f, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    out = np.clip(filtered_f * 255.0, 0, 255).astype(np.uint8)

    if save_img:
        cv2.imwrite(output_path, out)

    return out

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
        subtype='FILE_PATH',  # So Blender shows a file browser
        default="",
        description="File path to the input image"
    )

    def init(self, context):

        # OUTPUTS:
        # Albedo
        self.outputs.new("NodeSocketColor", "Albedo")
        # Roughness
        self.outputs.new("NodeSocketFloat", "Roughness")
        # Normal
        self.outputs.new("NodeSocketColor", "Normal")
        # Metallic
        self.outputs.new("NodeSocketFloat", "Metallic")
        # AO
        self.outputs.new("NodeSocketFloat", "AmbientOcclusion")
        # Displacement
        self.outputs.new("NodeSocketFloat", "Displacement")

    def draw_buttons(self, context, layout):
        layout.prop(self, "image_path", text="Image")
        layout.operator("node.generate_pbr_maps", text="Generate")

    def draw_label(self):
        return "PBR Generator"

    def process_image(self, diffuse_image):

        albedo = transform_single_png(diffuse_image)
        down_sample = scale_transform_sample(albedo, standalone=True)
        print(ADDON_DIR + "/Models/AOGenerator.pt")
        AOGenerator = torch.load(ADDON_DIR + "/Models/AOGenerator.pt")
        DisplacementGenerator = torch.load(ADDON_DIR + "/Models/DisplacementGenerator.pt")
        MetalnessGenerator = torch.load(ADDON_DIR + "/Models/MetalnessGenerator.pt")
        NormalGenerator = torch.load(ADDON_DIR + "/Models/NormalGLGenerator.pt")
        RoughnessGenerator = torch.load(ADDON_DIR + "/Models/RoughnessGenerator.pt")

        AODatasetInfo = ADDON_DIR + "/ModelInfo/AODatasetInfo.pt"
        DisplacementDatasetInfo = ADDON_DIR + "/ModelInfo/DisplacementDatasetInfo.pt"
        NormalDatasetInfo = ADDON_DIR + "/ModelInfo/NormalDatasetInfo.pt"
        RoughnessDatasetInfo = ADDON_DIR + "/ModelInfo/RoughnessDatasetInfo.pt"
        MetalnessDatasetInfo = ADDON_DIR + "/ModelInfo/MetalnessDatasetInfo.pt"

        device = "gpu" if torch.cuda.is_available() else "cpu"

        # For now, we'll just return placeholder data
        albedo_data = albedo
        roughness_data = single_pass(RoughnessGenerator, down_sample, albedo, device, RoughnessDatasetInfo, output_path="Roughness.png")
        normal_data = single_pass(NormalGenerator, down_sample, albedo, device, RoughnessDatasetInfo, output_path="Normal.png")
        metallic_data = single_pass(MetalnessGenerator, down_sample, albedo, device, RoughnessDatasetInfo, output_path="Metallic.png")
        ambient_occlusion_data = single_pass(AOGenerator, down_sample, albedo, device, RoughnessDatasetInfo, output_path="AO.png")

        return albedo_data, roughness_data, normal_data, metallic_data, ambient_occlusion_data


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

        # RUN THE MODEL:
        albedo_data, roughness_data, normal_data, metallic_data, ao_data = node.process_image(image_path)

        # For demonstration, let's just send placeholders to the sockets
        # (In practice, you'd need a more robust approach, either linking them to new Image Nodes or using custom properties.)
        node.outputs["Albedo"].default_value = albedo_data.cpu().numpy()
        node.outputs["Roughness"].default_value = roughness_data.cpu().numpy()
        node.outputs["Normal"].default_value = normal_data.cpu().numpy()
        node.outputs["Metallic"].default_value = metallic_data.cpu().numpy()
        node.outputs["Ambient Occlusion"].default_value = ao_data.cpu().numpy()

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