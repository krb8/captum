import math
import os
import gc
import skimage
import torch
import numpy as np
from PIL import Image, ImageStat, ImageChops
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
from my_dataset import my_dataset

from captum.attr import (
    GradientShap,
    Lime,
    LimeBase,
    IntegratedGradients,
    Deconvolution,
    DeepLift,
    GuidedBackprop,
    FeatureAblation,
    FeaturePermutation,
    Saliency,
    InputXGradient,
    Occlusion,
    KernelShap,
    ShapleyValueSampling,
    LRP,
    GuidedGradCam
)

dev = torch.device("cpu")

def eval_model(model, test_loader, dev, which_round):
    # model.to(dev)
    # model.eval()

    final_accuracy = 0
    final_num = 0
    with torch.no_grad():
        for i in range(0, len(test_loader.dataset.labels)):
            #label = test_loader.dataset.__getitem__(i)
            labels = []
            labels.append(test_loader.dataset.labels[i])
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

            print("Predicted class: ", torch.argmax(predictions, axis=1) , "Actual label: ", targets)

            del predictions

            if dev == 'cpu':

                final_accuracy += (accuracy * num)
                final_num += num

            else:
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

def batch_attribution(model_filepath, img_dirPath, output_dirpath, method):

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


    fns = [os.path.join(img_dirPath, fn) for fn in os.listdir(img_dirPath) if
           fn.endswith(".png")]


    ################### 7-10
    num_images_used = len(fns)
    mydata = {}
    mydata['test'] = my_dataset(fns)
    valid_dl = torch.utils.data.DataLoader(mydata['test'], batch_size=num_images_used, shuffle=True,
                                           pin_memory=True, num_workers=1)

    which_round = 4

    eval_model(model, valid_dl, dev, which_round)

    ################################
    start = time.time()


    for f in fns:
        multiple_attribution(f, model, output_dirpath, method)

    end = time.time()
    print("Total time: ", end - start)

def multiple_attribution(inputFilename, model, output_dirpath, method):
    transform = transforms.Compose([
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
    input = input.unsqueeze(0)

    ##predict the class of the input model
    with torch.no_grad():
        output = model(input)


    ## softmax: It is applied to all slices along dim,
    # and will re-scale them so that the elements lie in the range [0, 1] and sum to 1
    output = F.softmax(output, dim=1)

    predMax= np.argmax(output[0])
    # Returns the k largest elements of the given input tensor along a given dimension.
    prediction_score, pred_label_idx = torch.topk(output, 1)


    pred_label_idx = pred_label_idx[0]
    print("predicted label index: ", pred_label_idx)

    basename = os.path.basename(inputFilename)
    split_str = basename.split("_")
    # [ "class", "1" , "example", "9" ] length = 4
    length = len(split_str)
    # example in example
    if length >= 4 and split_str[length - 2] in "example":
        # label = int(split_str[length - 3])
        # grab number from filename
        label = int(split_str[1])
        print("Actual class label: ", label)
    else:
        print("could not parse label")


    ##########################################################

    # Initialize the attribution algorithm with the model

    # method = 3
    if(method == 0):
        integrated_gradients = IntegratedGradients(model)

        attribution_map = integrated_gradients.attribute(input, target=pred_label_idx, n_steps=200)

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
        lr_lime = Lime(model)

        attribution_map = lr_lime.attribute(input, target=pred_label_idx, n_samples=1)
    if (method == 3):
        integrated_gradients = Deconvolution(model)

        attribution_map = integrated_gradients.attribute(input, target=pred_label_idx)
    if (method == 4):
        deep_lift = DeepLift(model)
        attribution_map = deep_lift.attribute(input,target=pred_label_idx)
    if (method == 5):
        guided_backprop = GuidedBackprop(model)
        attribution_map = guided_backprop.attribute(input,target=pred_label_idx)
    if (method == 6):
        feature_ablation = FeatureAblation(model)
        attribution_map = feature_ablation.attribute(input, target=pred_label_idx)
    if (method == 7):
        saliency = Saliency(model)
        # attribute returns (Tensor or tuple[Tensor, …]):
        attribution_map = saliency.attribute(input, target=pred_label_idx)
    if (method == 8):
        inputxgrad = InputXGradient(model)
        attribution_map = inputxgrad.attribute(input, target=pred_label_idx)
    if (method == 9):
        ## sliding_window_shapes (tuple or tuple[tuple]) – Shape of patch (hyperrectangle) to occlude each input.
        occulsion = Occlusion(model)
        attribution_map = occulsion.attribute(input, target=pred_label_idx, sliding_window_shapes=(3,3,3))
    if (method == 10):
        ## sliding_window_shapes (tuple or tuple[tuple]) – Shape of patch (hyperrectangle) to occlude each input.
        kernelShap = KernelShap(model)
        attribution_map = kernelShap.attribute(input, target=pred_label_idx)
    if (method == 11):
        shapValSample = ShapleyValueSampling(model)
        attribution_map = shapValSample.attribute(input, target=pred_label_idx)
    if (method == 12):
        featurePermute = FeaturePermutation(model)
        attribution_map = featurePermute.attribute(input, target=pred_label_idx)
    if (method == 13):
        lrp = LRP(model)
        attribution_map = lrp.attribute(input, target=pred_label_idx)
    if (method == 14):
        guidedGradCam = GuidedGradCam(model) #GuidedGradCam(model, layer, device_ids=None)
        attribution_map = guidedGradCam.attribute(input, target=pred_label_idx)

    #  Attributions will always be the same size as the provided inputs, with each value providing
    #  the attribution of the corresponding input index.

    ## convert tensor to numpy array
    output_attribution = np.transpose(attribution_map.squeeze().cpu().detach().numpy(), (1, 2, 0))

    output_attribution = output_attribution[ :, :, 2]
    minvalue = output_attribution.min()
    maxvalue = output_attribution.max()

    if(maxvalue == minvalue):
        output_attribution = np.zeros(output_attribution.shape)
    else:
        ## min max normalization ( [0,1] ) * 255
        output_attribution = 255 * (output_attribution - minvalue) / (maxvalue - minvalue)

    PIL_image = Image.fromarray(output_attribution.astype('uint8'), 'L')

    output_filepath = os.path.join(output_dirpath, baseName)
    im1 = PIL_image.save(output_filepath)

    ##



    ## generate binarized mask
    mean = ImageStat.Stat(PIL_image).mean[0]
    stdev = ImageStat.Stat(PIL_image).stddev[0]
    threshold = math.ceil(mean) + math.ceil(stdev)

    # threshold = 127
    bin_map = PIL_image.point(lambda p: 255 if p > threshold else 0)

    out_filepath2 = os.path.join(output_dirpath + "/bin_map/", baseName)
    im2 = bin_map.save(out_filepath2)


def main(model_f_path, img_dirPath, result_dirpath, method):
    batch_attribution(model_f_path, img_dirPath, result_dirpath, method)

def preprocess_round3(img):
    # code from trojai example
    # perform center crop to what the CNN is expecting 224x224
    h, w, c = img.shape
    dx = int((w - 224) / 2)
    dy = int((w - 224) / 2)
    img = img[dy:dy + 224, dx:dx + 224, :]

     # perform tensor formatting and normalization explicitly
    # convert to CHW dimension ordering
    img = np.transpose(img, (2, 0, 1))
    # convert to NCHW dimension ordering
    img = np.expand_dims(img, 0)
    # normalize the image
    img = img - np.min(img)
    img = img / np.max(img)
    # convert image to a gpu tensor
    # batch_data = torch.FloatTensor(img)
    return img

def preprocess_round4(img):
    # NOTE: this is taked from https://raw.githubusercontent.com/usnistgov/trojai-example/round4/fake_trojan_detector.py
    # TODO verify that this line can be skipped for PIL loader!!!!
    img = img.astype(dtype=np.float32)
    # perform center crop to what the CNN is expecting 224x224
    # TODO verify that this line can be replaced for PIL loader !!!!
    h, w, c = img.shape
    dx = int((w - 224) / 2)
    dy = int((w - 224) / 2)
    img = img[dy:dy + 224, dx:dx + 224, :]

    # convert to CHW dimension ordering
    img = np.transpose(img, (2, 0, 1))
    # convert to NCHW dimension ordering
    img = np.expand_dims(img, 0)
    # normalize the image matching pytorch.transforms.ToTensor()
    img = img / 255.0

    # convert image to a gpu tensor
    # batch_data = torch.from_numpy(img).cuda()
    return img

def compareMaps(img_dir, img_dir2, result_dirpath):

    # create the output folder if it does not exist
    if os.path.exists(result_dirpath):
        print('INFO: output_dirpath:', result_dirpath, ' already exists')
    else:
        os.makedirs(result_dirpath)


    fns = [os.path.join(img_dir, fn) for fn in os.listdir(img_dir) if
           fn.endswith(".png")]

    ## img 2 is binary map
    for f in fns:
        baseName = os.path.basename(f)
        comp_image = os.path.join(img_dir2, baseName)

        img1 = Image.open(f)
        img2 = Image.open(comp_image)

        diff = ImageChops.difference(img1, img2)

        if diff.getbbox():
            diff.show()
