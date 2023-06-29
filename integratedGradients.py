import os
import shap
import json
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


from captum.attr import (
    GradientShap,
    Lime,
    LimeBase,
    IntegratedGradients,
)

dev = torch.device("cpu")
def integratedGradients():
    modelLoc = r"C:\Users\krb8\PycharmProjects\captum\trojAImodel.pt"
    modelPath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), modelLoc)
    )

    model = torch.load(modelPath)
    model = model.eval()

    test_img = Image.open(r"C:\Users\krb8\PycharmProjects\captum\class_0_example_0.png")

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

    labels_path = r'C:\Users\krb8\PycharmProjects\captum\imagenet_class_index.json'

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
