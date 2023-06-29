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
from torchvision.datasets import VOCSegmentation
from matplotlib.colors import LinearSegmentedColormap
from attribution_comparison import preprocess_round3, preprocess_round4
import pylab as pl
import integratedGradients as IG
from my_dataset import my_dataset

from captum.attr import (
    GradientShap,
    Lime,
    LimeBase,
    IntegratedGradients,
)

dev = torch.device("cpu")
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

    #labels_path = r'C:\Users\krb8\PycharmProjects\captum\imagenet_class_index.json'
    labels_path = r'C:\Users\krb8\PycharmProjects\captum\troj_class_index_new.json'
    # TODO: see if imagenet_class_index structure can be modified for TrojAI
    with open(labels_path) as json_data:
        idx_to_labels = json.load(json_data)

    img_dirPath = r"C:\Users\krb8\NIST_work\dataset\c75pr2-resnet101\clean_example_data"

    fns = [os.path.join(img_dirPath, fn) for fn in os.listdir(img_dirPath) if
           fn.endswith(".png")]

    # 6-28 below ####################################
    num_images_used = len(fns)
    print("NUM IMAGES USED: ", num_images_used)
    percent = 1.0
    num_images_used = int(percent * num_images_used)
    mydata = {}
    mydata['test'] = my_dataset(fns)
    valid_dl = torch.utils.data.DataLoader(mydata['test'], batch_size=num_images_used, shuffle=True,
                                           pin_memory=True, num_workers=1)

    ## 6-28 above  ########################################

    #why does this jump from class 0 to class 10 (ultimately gets all classes)
    for f in fns:
        ''' Captum Methods '''
        #gradientShap(f, model, idx_to_labels)
        integratedGradients(f, model, idx_to_labels)
        #captumLime(f, model, idx_to_labels)


def gradientShap(inputFilename, model, idx_to_labels):
    # ######################  CAPTUM GradientShap
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

    with torch.no_grad():
        output = model(input)

    ## softmax: It is applied to all slices along dim,
    # and will re-scale them so that the elements lie in the range [0, 1] and sum to 1
    output = F.softmax(output, dim=1)

    # Returns the k largest elements of the given input tensor along a given dimension.
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
    print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

    ###me

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
    output_filepath = os.path.join('C:\\Users\\krb8\\PycharmProjects\\captum\\attributionOutput\\captum\\gradientShap', baseName)
    im1 = PIL_image.save(output_filepath)

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

def integratedGradients(inputFilename, model, idx_to_labels):
    ### CAPTUM integrated gradients

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

    pred_label_idx.squeeze_()
    predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
    print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

    ##########################################################

    # Initialize the attribution algorithm with the model
    integrated_gradients = IntegratedGradients(model)

    # Ask the algorithm to attribute our output target to
    attributions_ig = integrated_gradients.attribute(input, target=pred_label_idx, n_steps=200)

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


    output_attribution = np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    output_attribution = output_attribution[ :, :, 2]
    output_attribution = output_attribution * 255
    #PIL_image = Image.fromarray(np.uint8(output_attribution)).convert('RGB')
    PIL_image = Image.fromarray(output_attribution.astype('uint8'), 'L')
    output_filepath = os.path.join('C:\\Users\\krb8\\PycharmProjects\\captum\\attributionOutput\\captum\\integratedGradients', baseName)
    im1 = PIL_image.save(output_filepath)

def main():
    batch_attribution()

if __name__ == '__main__':
    main()

## 6_28

'''
This method evaluates an ai model using images loaded from disk
'''
which_round = 4
def eval_model(model, test_loader, which_round):
    model.to("cpu")
    model.eval()
    # pre-allocate memmory
    #losses = np.empty(len(test_loader.dataset.labels), dtype=float)
    #accuracies = np.empty(len(test_loader.dataset.labels), dtype=float)
    #nums = np.empty(len(test_loader.dataset.labels), dtype=int)
    final_accuracy = 0
    final_num = 0
    with torch.no_grad():
        for i in range(0, len(test_loader.dataset.labels)):
            #label = test_loader.dataset.__getitem__(i)
            labels = []
            labels.append(test_loader.dataset.labels[i])
            #print('label:', labels)
            # prepare input batch
            # read the image from disk (using skimage)
            img = skimage.io.imread(test_loader.dataset.list_IDs[i])
            if which_round == 3:
                img = preprocess_round3(img)
            else:
                img = preprocess_round4(img)

            if dev == 'cpu':
                inputs = torch.from_numpy(img).float()
                targets = torch.FloatTensor(labels)
            else:
                # convert image and label to a GPU PyTorch tensor
                inputs = torch.cuda.FloatTensor(img)
                inputs = inputs.to(dev)
                #print('inputs are on the GPU:', inputs.is_cuda)
                targets = torch.cuda.FloatTensor(labels)
                targets = targets.to(dev)
                # print('targets are on the GPU:', targets.is_cuda)

            #loss, accuracy, num = epoch_func(model, inputs, targets, False)
            predictions = model(inputs)
            num = len(inputs)
            accuracy = (torch.argmax(predictions, axis=1) == targets).float().sum() / num

            del predictions
            # if i > 5:
            #     print('DEBUG:', i, ' num:', num, ' accuracy:', accuracy)
                # reset the TensorState counts
                #ts.reset_efficiency_model(model.efficiency_model)
                #ts.reset_efficiency_model(model.efficiency_layers)

            # print('DEBUG:', i)
            if dev == 'cpu':
                #losses.append(loss)
                # accuracies.append(accuracy)
                # nums.append(num)
                final_accuracy += (accuracy * num)
                final_num += num
                # accuracies[i] = accuracy
                # nums[i] = num
            else:
                # retrieve results from a GPU
                # losses.append(loss.cpu().detach().item())
                # accuracies.append(accuracy.cpu().detach().item())
                # nums.append(num)
                #losses[i] = loss.cpu().detach().item()
                # accuracies[i] = accuracy.cpu().detach().item()
                # nums[i] = num
                final_accuracy += (accuracy.cpu().detach().item() * num)
                final_num += num
                # clean up the GPU RAM
                del inputs
                del targets
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()  # https://forums.fast.ai/t/clearing-gpu-memory-pytorch/14637
                gc.collect()

        print('DEBUG: eval_model: finished model predictions for a batch of images')
        #accuracy = np.sum(np.multiply(accuracies, nums)) / np.sum(nums)
        final_accuracy = final_accuracy/final_num
        print('Network accuracy: {:.1f}%'.format(100 * accuracy))
        #return losses, accuracies, nums
        return final_accuracy



