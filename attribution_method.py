import os
import gc
import csv
import time
import torch
import skimage
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageStat
from my_dataset import my_dataset
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt

from captum.attr import (
    GradientShap,
    IntegratedGradients,
    Deconvolution,
    DeepLift,
    GuidedBackprop,
    Saliency,
    InputXGradient,
    DeepLiftShap,
    LRP,
    GuidedGradCam,
)

dev = torch.device("cpu")

def eval_model(model, test_loader, dev, which_round):

    final_accuracy = 0
    final_num = 0
    with torch.no_grad():
        for i in range(0, len(test_loader.dataset.labels)):
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

    num_images_used = len(fns)
    mydata = {}
    mydata['test'] = my_dataset(fns)
    valid_dl = torch.utils.data.DataLoader(mydata['test'], batch_size=num_images_used, shuffle=True,
                                           pin_memory=True, num_workers=1)
    which_round = 4
    eval_model(model, valid_dl, dev, which_round)


    start = time.time()


    for f in fns:
        multiple_attribution(f, model, output_dirpath, method)
        #multiple_attribution_evaluation(f, output_dirpath)

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
        #print("class label: ", label)
    else:
        print("could not parse label")


    ##########################################################

    # Initialize the attribution algorithm with the model
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
    if (method == 2):
        # A Module ReLU(inplace=True) was detected that does not contain some of the input/output attributes that are required for DeepLift computations
        rand_img_dist = torch.cat([input * 0, input * 1])
        deep_lift_shap = DeepLiftShap(model)
        attribution_map = deep_lift_shap.attribute(input,target=pred_label_idx, baselines=rand_img_dist)
    if (method == 3):
        integrated_gradients = Deconvolution(model)
        attribution_map = integrated_gradients.attribute(input, target=pred_label_idx)
    if (method == 4):  ######
        deep_lift = DeepLift(model)
        attribution_map = deep_lift.attribute(input,target=pred_label_idx)
    if (method == 5):
        guided_backprop = GuidedBackprop(model)
        attribution_map = guided_backprop.attribute(input,target=pred_label_idx)
    if (method == 7):
        saliency = Saliency(model)
        # attribute returns (Tensor or tuple[Tensor, …]):
        attribution_map = saliency.attribute(input, target=pred_label_idx)
    if (method == 8):
        inputxgrad = InputXGradient(model)
        attribution_map = inputxgrad.attribute(input, target=pred_label_idx)
    if (method == 13):
        lrp = LRP(model)
        attribution_map = lrp.attribute(input, target=pred_label_idx)
    if (method == 14):
        guidedGradCam = GuidedGradCam(model) #GuidedGradCam(model, layer, device_ids=None)
        attribution_map = guidedGradCam.attribute(input, target=pred_label_idx)

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

    attribution_map = Image.fromarray(output_attribution.astype('uint8'), 'L')

    output_filepath = os.path.join(output_dirpath, baseName)
    attribution_map.save(output_filepath)


    ######## GENERATE BINARIZED ATTRIBUTION MAP

    ##read in corresponding ground truth mask ###########
    ## eg. attribution map = "class_0_example_0" , mask_name = "class_0_example_0_mask"

    #ground_truth_path = os.path.join(output_dirpath, "ground_truth")
    ground_truth_path = os.path.join(output_dirpath, "ground_truth_polygon")

    fnsGroundTruth = [os.path.join(ground_truth_path, fn) for fn in os.listdir(ground_truth_path) if
                      fn[:-4] in basename]

    ground_truth_map = Image.open(fnsGroundTruth[0])
    #
    mean = ImageStat.Stat(attribution_map).mean[0]
    stdev = ImageStat.Stat(attribution_map).stddev[0]
    #
    #ground_truth_array = np.array(ground_truth_map, dtype=np.int_)
    ground_truth_array_orig = np.array(ground_truth_map, dtype=np.int_)


    fin_map = []

    flatAttrib = np.array(attribution_map)

    for p, pixI in enumerate(flatAttrib.flatten()):
        ## mask from ROI generates b/w image using [0,255], not [0,1]
        val = 0
        if mean - stdev > pixI or pixI > mean + stdev:
            val = 255
        fin_map.append(val)

    bin_map_attrib_array = np.array(fin_map).reshape((256,256))
    bin_image = Image.fromarray(bin_map_attrib_array.astype('uint8'), 'L')

    out_filepath2 = os.path.join(output_dirpath + "/bin_map/", baseName)
    bin_image.save(out_filepath2)

    rounding_precision = 4

    ## confusion_matrix(y_true, y_pred, *, labels=None, ...)
    #tn, fp, fn, tp = confusion_matrix(bin_map_attrib_array.ravel(), bin_map_attrib_array.ravel(), labels=[0,255]).ravel()
    cnf_matrix = confusion_matrix(ground_truth_array_orig.ravel(), bin_map_attrib_array.ravel(), labels=[0,255])
    tn, fp, fn, tp = cnf_matrix.ravel()
    print("tn, fp, fn, tp: ", tn, fp, fn, tp)
    # derive metrics from confusion matrix
    if (2 * tp + fp + fn) <= 0:
        print('ERROR: sum of ((2*tp + fp + fn) is zero ')
        dice_index = -1.0
        print("dice_index: ", dice_index)
    else:
        dice_index = round((2 * tp / (2 * tp + fp + fn)), rounding_precision)
        print("dice_index: ", dice_index)
    if (fp + fn + tp) <= 0:
        print('ERROR: sum of (fp + fn + tp) is zero ')
        jaccard_index = -1.0
        print("jaccard_index: ", jaccard_index)
    else:
        jaccard_index = round((tp / (fp + fn + tp)), rounding_precision)
        print("jaccard_index: ", jaccard_index)
    if (fp + fn) <= 0:
        print('ERROR: sum of ((fp + fn) is zero ')
        cosine_index = -1.0
        print("cosine_index: ", cosine_index)
    else:
        cosine_index = round((tp / (np.sqrt(fp + fn))), rounding_precision)
        print("cosine_index: ", cosine_index)

    # name of csv file

    filename = "result.csv"
    csv_dirpath = os.path.join(output_dirpath, "metrics")
    csv_filepath = os.path.join(csv_dirpath, filename)

    if not (os.path.exists(csv_dirpath)):
        os.makedirs(csv_dirpath)
        with open(csv_filepath, 'a') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Filename", "Dice", "Jaccard", "Cosine"])


    # writing to csv file
    with open(csv_filepath, 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the data rows
        csvwriter.writerow([baseName , str(dice_index), str(jaccard_index), str(cosine_index)])

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cnf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cnf_matrix.shape[0]):
        for j in range(cnf_matrix.shape[1]):
            ax.text(x=j, y=i, s=cnf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(os.path.join(csv_dirpath, "matrix_{}.png".format( baseName )), dpi='figure', format="png")


def batch_multiple_attribution_evaluation(inputAttributionDir, output_dirpath):
    attribution_files = [os.path.join(inputAttributionDir, fn) for fn in os.listdir(inputAttributionDir)]

    for f in attribution_files:
        if os.path.isfile(f):
            multiple_attribution_evaluation(f, output_dirpath)


def multiple_attribution_evaluation(inputAttributionFilename, output_dirpath):
    baseName = os.path.basename(inputAttributionFilename)
    attribution_map = Image.open(inputAttributionFilename)
    mean = ImageStat.Stat(attribution_map).mean[0]
    stdev = ImageStat.Stat(attribution_map).stddev[0]

    attribution_map = np.asarray(attribution_map)


    ######## GENERATE BINARIZED ATTRIBUTION MAP
    ground_truth_path = os.path.join(output_dirpath, "ground_truth")

    fnsGroundTruth = [os.path.join(ground_truth_path, fn) for fn in os.listdir(ground_truth_path) if
                      fn[:-4] in baseName]

    ground_truth_map = Image.open(fnsGroundTruth[0])

    #ground_truth_array_orig = np.array(ground_truth_map, dtype=np.int_)
    ground_truth_array_orig = np.asarray(ground_truth_map, dtype=np.int_)

    fin_map = []

    flatAttrib = np.array(attribution_map)

    for p, pixI in enumerate(flatAttrib.flatten()):
        ## mask from ROI generates b/w image using [0,255], not [0,1]
        val = 0
        if mean - stdev > pixI or pixI > mean + stdev:
            val = 255
        fin_map.append(val)

    bin_map_attrib_array = np.array(fin_map).reshape((256,256))
    bin_image = Image.fromarray(bin_map_attrib_array.astype('uint8'), 'L')

    out_filepath2 = os.path.join(output_dirpath + "/bin_map/", baseName)
    bin_image.save(out_filepath2)

    rounding_precision = 4

    ## TPR for polygon ground truth

    if ground_truth_array_orig.shape != bin_map_attrib_array.shape:
        ground_truth_array_orig =Image.open(fnsGroundTruth[0]).convert("L")
        ground_truth_array_orig = np.array(ground_truth_array_orig)
        #ground_truth_array_orig = ground_truth_array_orig.reshape((256,256))

    ## confusion_matrix(y_true, y_pred, *, labels=None, ...)
    cnf_matrix = confusion_matrix(ground_truth_array_orig.ravel(), bin_map_attrib_array.ravel(), labels=[0,255])
    tn, fp, fn, tp = cnf_matrix.ravel()
    print("tn, fp, fn, tp: ", tn, fp, fn, tp)
    # derive metrics from confusion matrix
    if (2 * tp + fp + fn) <= 0:
        print('ERROR: sum of ((2*tp + fp + fn) is zero ')
        dice_index = -1.0
        print("dice_index: ", dice_index)
    else:
        dice_index = round((2 * tp / (2 * tp + fp + fn)), rounding_precision)
        print("dice_index: ", dice_index)
    if (fp + fn + tp) <= 0:
        print('ERROR: sum of (fp + fn + tp) is zero ')
        jaccard_index = -1.0
        print("jaccard_index: ", jaccard_index)
    else:
        jaccard_index = round((tp / (fp + fn + tp)), rounding_precision)
        print("jaccard_index: ", jaccard_index)
    if (fp + fn) <= 0:
        print('ERROR: sum of ((fp + fn) is zero ')
        cosine_index = -1.0
        print("cosine_index: ", cosine_index)
    else:
        cosine_index = round((tp / (np.sqrt(fp + fn))), rounding_precision)
        print("cosine_index: ", cosine_index)
    if tp / ( tp + fn ) <= 0:

        print("ERROR: TPR is ",  tp / ( tp + fn ))
        tpr = -1.0
    else:

        tpr = tp/(tp + fn)
        print("TPR: ", tpr)



    # name of csv file

    filename = "result.csv"
    csv_dirpath = os.path.join(output_dirpath, "metrics_polygon")
    csv_filepath = os.path.join(csv_dirpath, filename)

    if not (os.path.exists(csv_dirpath)):
        os.makedirs(csv_dirpath)
        with open(csv_filepath, 'a') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            csvwriter.writerow(["Filename", "Dice", "Jaccard", "Cosine", "TPR"])



    # writing to csv file
    with open(csv_filepath, 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the data rows

        csvwriter.writerow([baseName, str(dice_index), str(jaccard_index), str(cosine_index), str(tpr)])



    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cnf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cnf_matrix.shape[0]):
        for j in range(cnf_matrix.shape[1]):
            ax.text(x=j, y=i, s=cnf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(os.path.join(csv_dirpath, "matrix_{}.png".format( baseName )), dpi='figure', format="png")


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

if __name__ == "__main__":
    import argparse

    print('torch version: %s \n' % (torch.__version__))

    parser = argparse.ArgumentParser(
        description='Efficiency estimator of AI Models to Demonstrate dependency on the number of predicted classes.')
    parser.add_argument('--output_dirpath', type=str,
                        help='Directory path  where output result should be written.', required=True)
    parser.add_argument('--image_dir', type=str,
                        help='image_dir is the name of a subdirectory in attribution maps with a batch of images to run',
                        required=True)

    args = parser.parse_args()
    print('args %s \n % s \n' % (
         args.output_dirpath,  args.image_dir))

    batch_multiple_attribution_evaluation(args.image_dir, args.output_dirpath)
    #multiple_attribution_evaluation(args.image_dir, args.output_dirpath )


