import bpy
from bpy.types import Node, NodeTree
from bpy.props import StringProperty, PointerProperty
from bpy.types import Operator

from transforms import transform_single_png, scale_transform_sample
from feedforward import single_pass
import torch

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
        return (0.8, 0.4, 0.4, 1.0)  # RGBA color for the socket icon


class PBRGeneratorNode(Node):
    """A custom node to generate PBR maps from a diffuse image."""
    bl_idname = "PBRGeneratorNodeType"
    bl_label = "PBR Generator"
    bl_icon = 'NODE_TEXTURE'

    # Properties for your node, e.g. a path to the model, or a "Generate" button
    model_path: StringProperty(
        name="Model Path",
        default="//pbr_model.pt",
        description="Path to your generative model"
    )

    def init(self, context):
        # INPUTS:
        # For the diffuse image, we can use Blender's built-in image socket
        self.inputs.new("NodeSocketColor", "Diffuse Image")

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
        layout.prop(self, "model_path", text="Model")
        layout.operator("node.generate_pbr_maps", text="Generate")

    def draw_label(self):
        return "PBR Generator"

    def process_image(self, diffuse_image):

        albedo = transform_single_png(diffuse_image)
        down_sample = scale_transform_sample(albedo, standalone=True)

        AOGenerator = torch.load("Models/AOGenerator.pt")
        DisplacementGenerator = torch.load("Models/DisplacementGenerator.pt")
        MetalnessGenerator = torch.load("Models/MetalnessGenerator.pt")
        NormalGenerator = torch.load("Models/NormalGLGenerator.pt")
        RoughnessGenerator = torch.load("Models/RoughnessGenerator.pt.pt")

        AODatasetInfo = "ModelInfo/AODatasetInfo.pt"
        DisplacementDatasetInfo = "ModelInfo/DisplacementDatasetInfo.pt"
        NormalDatasetInfo = "ModelInfo/NormalDatasetInfo.pt"
        RoughnessDatasetInfo = "ModelInfo/RoughnessDatasetInfo.pt"
        MetalnessDatasetInfo = "ModelInfo/MetalnessDatasetInfo.pt"

        device = "gpu" if torch.cuda.is_available() else "cpu"

        # For now, we'll just return placeholder data
        albedo_data = albedo
        roughness_data = single_pass(RoughnessGenerator, down_sample, albedo, device, RoughnessDatasetInfo)
        normal_data = single_pass(NormalGenerator, down_sample, albedo, device, RoughnessDatasetInfo)
        metallic_data = single_pass(MetalnessGenerator, down_sample, albedo, device, RoughnessDatasetInfo)
        ambient_occlusion_data = single_pass(AOGenerator, down_sample, albedo, device, RoughnessDatasetInfo)

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

        # Access the input image
        diffuse_input = node.inputs.get("Diffuse Image")
        if not diffuse_input or not diffuse_input.is_linked:
            self.report({'ERROR'}, "No diffuse image linked to the node.")
            return {'CANCELLED'}

        # The actual image node
        from_node = diffuse_input.links[0].from_node
        # If from_node is an Image Texture node, we can grab its image
        if from_node and from_node.type == 'TEX_IMAGE':
            diffuse_image = from_node.image
        else:
            self.report({'ERROR'}, "Input is not an image.")
            return {'CANCELLED'}

        # RUN THE MODEL:
        albedo_data, roughness_data, normal_data, metallic_data, ao_data = node.process_image(diffuse_image)

        # For demonstration, let's just send placeholders to the sockets
        # (In practice, you'd need a more robust approach, either linking them to new Image Nodes or using custom properties.)
        node.outputs["Albedo"].default_value = albedo_data
        node.outputs["Roughness"].default_value = roughness_data
        node.outputs["Normal"].default_value = normal_data
        node.outputs["Metallic"].default_value = metallic_data
        node.outputs["Ambient Occlusion"].default_value = ao_data

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
