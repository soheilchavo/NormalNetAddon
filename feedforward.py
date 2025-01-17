import torch
from transforms import unnormalize_tensor
from upsample import joint_bilateral_up_sample
import pickle

def single_pass(model, input_tensor, guide_tensor, device, dataset_info):

    with open('Data/RoughnessTrainingDatasetInfo', 'rb') as f:
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

    result = joint_bilateral_up_sample(result, guide_tensor)

    return result