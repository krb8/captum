import os
import gc
#import shap
import json
import skimage
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
from matplotlib.colors import LinearSegmentedColormap


from captum.attr import (
    GradientShap,
    Lime,
    LimeBase,
    IntegratedGradients,
    Deconvolution,
    DeepLift,
    GuidedBackprop
)

dev = torch.device("cpu")
def batch_attribution(model_filepath, img_dirPath, output_dirpath):

    # create the output folder if it does not exist
    if os.path.exists(output_dirpath):
        print('INFO: output_dirpath:', output_dirpath, ' already exists')
    else:
        os.makedirs(output_dirpath)

    # Set the device to run the model on (gpu if available, cpu otherwise)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dev = "cpu"

    if dev != 'cpu':
        print('current device index:', str(torch.cuda.current_device()))

    print('loading model: ', model_filepath)
    torch.manual_seed(0)
    model = torch.load(model_filepath, map_location=dev)
    model.to(dev)
    model = model.eval()

    #labels_path = r'C:\Users\krb8\PycharmProjects\captum\imagenet_class_index.json'
    #labels_path = r'C:\Users\krb8\PycharmProjects\captum\troj_class_index_new.json'
    # with open(labels_path) as json_data:
    #     idx_to_labels = json.load(json_data)

    #img_dirPath = r"C:\Users\krb8\NIST_work\dataset\c75pr2-resnet101\clean_example_data"

    fns = [os.path.join(img_dirPath, fn) for fn in os.listdir(img_dirPath) if
           fn.endswith(".png")]

    #why does this jump from class 0 to class 10 (ultimately gets all classes)
    for f in fns:
        ''' Captum Methods '''
        #gradientShap(f, model, idx_to_labels)
        multiple_attribution(f, model, output_dirpath)
        #captumLime(f, model, idx_to_labels)

def captumLime(inputFilename, model, idx_to_labels):
    # ######################  CAPTUM Lime

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

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

    lime_attr = LimeBase(model)

    # Defining baseline distribution of images
    rand_img_dist = torch.cat([input * 0, input * 1])

    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#ffffff'),
                                                      (0.25, '#000000'),
                                                      (1, '#000000')], N=256)

    attributions_gs = lime_attr.attribute(input,
                                              n_samples=50,
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

    PIL_image = Image.fromarray(output_attribution.astype('uint8'), 'L')
    output_filepath = os.path.join('C:\\Users\\krb8\\PycharmProjects\\captum\\attributionOutput\\captum\\lime', baseName)
    im1 = PIL_image.save(output_filepath)


def multiple_attribution(inputFilename, model, output_dirpath):
    ### CAPTUM integrated gradients

    # model expects 224x224 3-color image
    transform = transforms.Compose([
        #transforms.Resize(224),
        #transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # standard ImageNet normalization
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
    input = input.unsqueeze(0) ## model requires dummy batch dimension

    ##predict the class of the input model

    with torch.no_grad():
        output = model(input)

    ## softmax: It is applied to all slices along dim,
    # and will re-scale them so that the elements lie in the range [0, 1] and sum to 1
    output = F.softmax(output, dim=1)

    # Returns the k largest elements of the given input tensor along a given dimension.
    prediction_score, pred_label_idx = torch.topk(output, 1)

    # pred_label_idx.squeeze_()
    # predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
    # print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

    basename = os.path.basename(inputFilename)
    split_str = basename.split("_")
    # [ "class", "1" , "example", "9" ] length = 4
    length = len(split_str)
    # example in example
    if length >= 4 and split_str[length - 2] in "example":
        # label = int(split_str[length - 3])
        # grab number from filename
        label = int(split_str[1])
        print(label)
    else:
        print("could not parse label")


    ##########################################################

    # Initialize the attribution algorithm with the model

    method = 1
    if(method == 0):
        integrated_gradients = IntegratedGradients(model)

        attribution_map = integrated_gradients.attribute(input, target=pred_label_idx, n_steps=200)

        # default_cmap and visualization stuff here

    if (method == 1):
        gradient_shap = GradientShap(model)

        rand_img_dist = torch.cat([input * 0, input * 1])
        attribution_map = gradient_shap.attribute(input,
                                                  n_samples=50,
                                                  stdevs=0.0001,
                                                  baselines=rand_img_dist,
                                                  target=pred_label_idx)
        # default_cmap and visualization stuff here
    if (method == 2):
        ## captum LIME
        #rand_img_dist = torch.cat([input * 0, input * 1])
        exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)

        lr_lime = Lime(
            model,
            interpretable_model=SkLearnLinearRegression(),  # build-in wrapped sklearn Linear Regression
            similarity_func=exp_eucl_distance
        )

        attribution_map = lr_lime.attribute(
            input,
            target=pred_label_idx,
            # feature_mask=feature_mask.unsqueeze(0),
            n_samples=40,
            perturbations_per_eval=16,
            show_progress=True,
            #baselines=rand_img_dist
        )
    if (method == 3):
        integrated_gradients = Deconvolution(model)

        attribution_map = integrated_gradients.attribute(input, target=pred_label_idx)
    if (method == 4):
        deep_lift = DeepLift(model)
        attribution_map = deep_lift.attribute(input,target=pred_label_idx)
    if (method == 5):
        guided_backprop = GuidedBackprop(model)
        attribution_map = guided_backprop.attribute(input,target=pred_label_idx)


    output_attribution = np.transpose(attribution_map.squeeze().cpu().detach().numpy(), (1, 2, 0))
    output_attribution = output_attribution[ :, :, 2]
    minvalue = output_attribution.min()
    maxvalue = output_attribution.max()
    #output_attribution = output_attribution * 255
    output_attribution = 255 * (output_attribution - minvalue) / (maxvalue - minvalue)
    #PIL_image = Image.fromarray(np.uint8(output_attribution)).convert('RGB')
    PIL_image = Image.fromarray(output_attribution.astype('uint8'), 'L')
    output_filepath = os.path.join(output_dirpath, baseName)
    im1 = PIL_image.save(output_filepath)

def main(model_f_path, img_dirPath, result_dirpath):
    batch_attribution(model_f_path, img_dirPath, result_dirpath)




