import torch.nn as nn
import torch

from TensorFunctions import unnormalize_tensor, joint_bilateral_up_sample

class UpConv(nn.Module):
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
