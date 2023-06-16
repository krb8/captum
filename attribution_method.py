import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

import os, sys
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)

# class ToyModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lin1 = nn.Linear(3, 3)
#         self.relu = nn.ReLU()
#         self.lin2 = nn.Linear(3, 2)
#
#         # initialize weights and biases
#         self.lin1.weight = nn.Parameter(torch.arange(-4.0, 5.0).view(3, 3))
#         self.lin1.bias = nn.Parameter(torch.zeros(1,3))
#         self.lin2.weight = nn.Parameter(torch.arange(-3.0, 3.0).view(2, 3))
#         self.lin2.bias = nn.Parameter(torch.ones(1,2))
#
#     def forward(self, input):
#         return self.lin2(self.relu(self.lin1(input)))

def main():
    #model = ToyModel()
    #model.eval()
    #torch.manual_seed(123)
    #np.random.seed(123)
    ## replace input with images, save the output

    model = models.resnet18(weights='IMAGENET1K_V1')
    model = model.eval()

    test_img = Image.open("/home/krb8/NISTworkTest/captum/class_0_example_0.png")

    ## display image
    test_img_data = np.asarray(test_img)
    plt.imshow(test_img_data)
    plt.show()

    # model expects 224x224 3-color image
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # standard ImageNet normalization
    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transformed_img = transform(test_img)
    input_img = transform_normalize(transformed_img)
    input_img = input_img.unsqueeze(0)  # the model requires a dummy batch dimension

    labels_path = '/home/krb8/NISTworkTest/imagenet_class_index.json'
    with open(labels_path) as json_data:
        idx_to_labels = json.load(json_data)

    output = model(input_img)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()
    predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
    print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

    ##########################################################

    # Initialize the attribution algorithm with the model
    integrated_gradients = IntegratedGradients(model)

    # Ask the algorithm to attribute our output target to
    attributions_ig = integrated_gradients.attribute(input_img, target=pred_label_idx, n_steps=200)

    # Show the original image for comparison
    _ = viz.visualize_image_attr(None, np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                 method="original_image", title="Original Image")

    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#ffffff'),
                                                      (0.25, '#0000ff'),
                                                      (1, '#0000ff')], N=256)

    _ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                 np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                 method='heat_map',
                                 cmap=default_cmap,
                                 show_colorbar=True,
                                 sign='positive',
                                 title='Integrated Gradients')


    ###################### GradientShap

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    img = Image.open('/home/krb8/NISTworkTest/captum/class_0_example_0.png')

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
                                          show_colorbar=True)

    # input = torch.rand(2, 3)
    # baseline = torch.zeros(2, 3)
    # ig = IntegratedGradients(model)
    # attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
    # print('IG Attributions:', attributions)
    # print('Convergence Delta:', delta)



if __name__ == '__main__':
    main()