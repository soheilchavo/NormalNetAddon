import numpy as np

import cv2
import cv2.ximgproc as xip

from PIL import Image
from torchvision import transforms

img_transform = transforms.Compose([transforms.PILToTensor()])

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