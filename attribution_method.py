import os
#import shap
import json

import matplotlib.pyplot
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from captum.attr import visualization as viz
from captum.attr._core.lime import get_exp_kernel_similarity_function
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
from torchvision.datasets import VOCSegmentation
from matplotlib.colors import LinearSegmentedColormap


import pylab as pl

import integratedGradients as IG

from captum.attr import (
    GradientShap,
    Lime,
    LimeBase,
    IntegratedGradients,
)
def batch_attribution():
    img_dirPath = r"C:\Users\krb8\NIST_work\dataset\c75pr2-resnet101\clean_example_data"

    fns = [(img_dirPath, fn) for fn in os.listdir(img_dirPath) if
           fn.endswith(".png")]

    modelLoc = r"C:\Users\krb8\PycharmProjects\captum\trojAImodel.pt"

    modelPath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), modelLoc)
    )

    model = torch.load(modelPath)
    model = model.eval()

    labels_path = r'C:\Users\krb8\PycharmProjects\captum\imagenet_class_index.json'
    # labels_path = r'C:\Users\krb8\PycharmProjects\captum\troj_class_index.json'
    with open(labels_path) as json_data:
        idx_to_labels = json.load(json_data)

    # transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor()
    # ])

    # transform_normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    # )

    img_dirPath = r"C:\Users\krb8\NIST_work\dataset\c75pr2-resnet101\clean_example_data"

    fns = [os.path.join(img_dirPath, fn) for fn in os.listdir(img_dirPath) if
           fn.endswith(".png")]

    for f in fns:
        gradientShap(f, model, idx_to_labels)


def gradientShap(inputFilename, model, idx_to_labels):
    # ###################### GradientShap
    # ###################### https://captum.ai/tutorials/Resnet_TorchVision_Interpret
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    #inputFilename = r'C:\Users\krb8\PycharmProjects\captum\class_0_example_0.png'
    img = Image.open(inputFilename)
    baseName = os.path.basename(inputFilename)
    print(baseName)

    transformed_img = transform(img)

    input = transform_normalize(transformed_img)
    input = input.unsqueeze(0)

    ##predict the class of the input model

    output = model(input)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
    print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

    gradient_shap = GradientShap(model)

    # Defining baseline distribution of images
    rand_img_dist = torch.cat([input * 0, input * 1])

    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#ffffff'),
                                                      (0.25, '#000000'),
                                                      (1, '#000000')], N=256)

    attributions_gs = gradient_shap.attribute(input,
                                              n_samples=50,
                                              stdevs=0.0001,
                                              baselines=rand_img_dist,
                                              target=pred_label_idx)
    _ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          ["original_image", "heat_map"],
                                          ["all", "absolute_value"],
                                          cmap=default_cmap,
                                          show_colorbar=True,
                                          titles=["Original", "GradientShap"])

    output_attribution = np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0))
    output_attribution = output_attribution[ :, :, 2]
    output_attribution = output_attribution * 255
    #PIL_image = Image.fromarray(np.uint8(output_attribution)).convert('RGB')
    PIL_image = Image.fromarray(output_attribution.astype('uint8'), 'L')
    output_filepath = os.path.join('C:\\Users\\krb8\\PycharmProjects\\captum\\attributionOutput\\', baseName)
    im1 = PIL_image.save(output_filepath)


def main():
    batch_attribution()

if __name__ == '__main__':
    main()

