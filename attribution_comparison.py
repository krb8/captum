import gc
import json
import sys
import time

import skimage
import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from random import randint
from PIL import Image

import os
from my_dataset import my_dataset
import skimage.io

import TensorState as ts
import reference_probes as refprobes

import matplotlib.pyplot as plt

from PIL import Image


'''
save header and metadata information to all output files
'''
def prepare_output_files(is_per_label, output_filepath, common_header, model_filepath, model_architecture, model_basename,number_predicted_classes,
                         num_images_used, num_training_samples, final_combined_val_acc, gt_model_label,number_triggers,
                         triggered_classes, trigger_target_class, triggered_fraction, trigger_type, probe_name):

    # the metrics.csv is written only for the evaluations over all images per AI model and per layer (not per label)
    # this was moved to KLdivergenceModel.csv (per ai model) and KLdivergenceLabel.csv (per label)

    #######################################
    states_layers_labels_filepath = output_filepath[0:-4]
    if is_per_label:
        states_layers_labels_filepath += '_StatesLayerLabel.csv'
    else:
        states_layers_labels_filepath += '_StatesLayer.csv'
    if not os.path.isfile(states_layers_labels_filepath):
        # write header
        with open(states_layers_labels_filepath, 'w') as fh:
            if is_per_label:
                specific_header = "predicted class label, layer name, " \
                              "node utilization per layer per label, \n"
            else:
                specific_header = "layer name, " \
                              "node utilization per layer, \n"

            final_header = common_header + specific_header
            fh.write(final_header)

    max_number_triggers = len(triggered_classes)
    with open(states_layers_labels_filepath, 'a') as fh:
        fh.write("{}, ".format(model_filepath))
        fh.write("{}, ".format(model_architecture))
        fh.write("{}, ".format(model_basename))
        fh.write('{},'.format(number_predicted_classes))
        fh.write('{},'.format(num_images_used))
        fh.write("{}, ".format(num_training_samples))
        fh.write("{:.4f}, ".format(final_combined_val_acc))
        fh.write("{}, ".format(gt_model_label))
        fh.write('{}, '.format(number_triggers))
        for idx_triggger in range(0, max_number_triggers):
            fh.write("{}, ".format(triggered_classes[idx_triggger]))
            fh.write("{}, ".format(trigger_target_class[idx_triggger]))
            fh.write("{:.2f}, ".format(triggered_fraction[idx_triggger]))
            fh.write("{}, ".format(trigger_type[idx_triggger]))
        fh.write('{},'.format(probe_name))
        # for idx in range(0, len(config['EFFICIENCY_ATTACH_TO'])):
        #     fh.write('{},'.format(config['EFFICIENCY_ATTACH_TO'][idx]))

    KLefficiency_layers_labels_filepath = output_filepath[0:-4]
    if is_per_label:
        KLefficiency_layers_labels_filepath += '_KLdivergenceLayerLabel.csv'
    else:
        KLefficiency_layers_labels_filepath += '_KLdivergenceLayer.csv'
    if not os.path.isfile(KLefficiency_layers_labels_filepath):
        # write header
        with open(KLefficiency_layers_labels_filepath, 'w') as fh:
            if is_per_label:
                specific_header = "predicted class label, layer name, " \
                                           "KL divergence inefficiency per class label per layer, \n"
            else:
                specific_header = "layer name, " \
                                           "KL divergence inefficiency per layer, \n"

            final_header = common_header + specific_header
            fh.write(final_header)

    with open(KLefficiency_layers_labels_filepath, 'a') as fh:
        fh.write("{}, ".format(model_filepath))
        fh.write("{}, ".format(model_architecture))
        fh.write("{}, ".format(model_basename))
        fh.write('{},'.format(number_predicted_classes))
        fh.write('{},'.format(num_images_used))
        fh.write("{}, ".format(num_training_samples))
        fh.write("{:.4f}, ".format(final_combined_val_acc))
        fh.write("{}, ".format(gt_model_label))
        fh.write('{}, '.format(number_triggers))
        for idx_triggger in range(0, max_number_triggers):
            fh.write("{}, ".format(triggered_classes[idx_triggger]))
            fh.write("{}, ".format(trigger_target_class[idx_triggger]))
            fh.write("{:.2f}, ".format(triggered_fraction[idx_triggger]))
            fh.write("{}, ".format(trigger_type[idx_triggger]))
        fh.write('{},'.format(probe_name))
        # for idx in range(0, len(config['EFFICIENCY_ATTACH_TO'])):
        #     fh.write('{},'.format(config['EFFICIENCY_ATTACH_TO'][idx]))

    entropy_layers_labels_filepath = output_filepath[0:-4]
    if is_per_label:
        entropy_layers_labels_filepath += '_EntropyLayerLabel.csv'
    else:
        entropy_layers_labels_filepath += '_EntropyLayer.csv'
    if not os.path.isfile(entropy_layers_labels_filepath):
        # write header
        with open(entropy_layers_labels_filepath, 'w') as fh:
            if is_per_label:
                specific_header = "predicted class label, layer name, " \
                                           "Normalized Entropy efficiency per class label per layer [%], " \
                                           " \n"
            else:
                specific_header = "layer name, " \
                                           "Normalized Entropy efficiency per layer [%], " \
                                           " \n"
            final_header = common_header + specific_header
            fh.write(final_header)

    with open(entropy_layers_labels_filepath, 'a') as fh:
        fh.write("{}, ".format(model_filepath))
        fh.write("{}, ".format(model_architecture))
        fh.write("{}, ".format(model_basename))
        fh.write('{},'.format(number_predicted_classes))
        fh.write('{},'.format(num_images_used))
        fh.write("{}, ".format(num_training_samples))
        fh.write("{:.4f}, ".format(final_combined_val_acc))
        fh.write("{}, ".format(gt_model_label))
        fh.write('{}, '.format(number_triggers))
        for idx_triggger in range(0, max_number_triggers):
            fh.write("{}, ".format(triggered_classes[idx_triggger]))
            fh.write("{}, ".format(trigger_target_class[idx_triggger]))
            fh.write("{:.2f}, ".format(triggered_fraction[idx_triggger]))
            fh.write("{}, ".format(trigger_type[idx_triggger]))
        fh.write('{},'.format(probe_name))
        # for idx in range(0, len(config['EFFICIENCY_ATTACH_TO'])):
        #     fh.write('{},'.format(config['EFFICIENCY_ATTACH_TO'][idx]))

    summary_layers_filepath = output_filepath[0:-4]
    if is_per_label:
        summary_layers_filepath += '_KLdivergenceLabel.csv'
    else:
        summary_layers_filepath += '_KLdivergenceModel.csv'
    if not os.path.isfile(summary_layers_filepath):
        if is_per_label:
            specific_header = "predicted class label, min Entropy util per label over all layers, max Entropy util per label over all layers, avg Entropy util per label over all layers, " \
                              "min KL divergence per label over all layers, max KL divergence per label over all layers, avg KL divergence inefficiency per label over all layers," \
                              "Network accuracy per label [%], Entropy-based network efficiency per label [%], " \
                              "aIQ per label [%], Eval model time per label [s], " \
                              "\n "
        else:
            specific_header = "min Entropy util over all layers, max Entropy util over all layers, avg Entropy util over all layers, " \
                              "min KL divergence over all layers, max KL divergence over all layers, avg KL divergence inefficiency over all layers," \
                              "Network accuracy  [%], Entropy-based network efficiency  [%], " \
                              "aIQ [%], Eval model time [s], Eval model efficiency time [s]" \
                              "\n "
            # specific_header = "Network accuracy [%], Global Entropy-based network efficiency [%], " \
            #                   "aIQ [%], Global KL divergence network inefficiency, Avg node utilization [%]," \
            #                   "Eval model time [s], Eval model efficiency time [s], " \
            #                   "\n "
        # write header
        with open(summary_layers_filepath, 'w') as fh:
            final_header = common_header + specific_header
            fh.write(final_header)

    with open(summary_layers_filepath, 'a') as fh:
        fh.write("{}, ".format(model_filepath))
        fh.write("{}, ".format(model_architecture))
        fh.write("{}, ".format(model_basename))
        fh.write('{},'.format(number_predicted_classes))
        fh.write('{},'.format(num_images_used))
        fh.write("{}, ".format(num_training_samples))
        fh.write("{:.4f}, ".format(final_combined_val_acc))
        fh.write("{}, ".format(gt_model_label))
        fh.write('{}, '.format(number_triggers))
        for idx_triggger in range(0, max_number_triggers):
            fh.write("{}, ".format(triggered_classes[idx_triggger]))
            fh.write("{}, ".format(trigger_target_class[idx_triggger]))
            fh.write("{:.2f}, ".format(triggered_fraction[idx_triggger]))
            fh.write("{}, ".format(trigger_type[idx_triggger]))
        fh.write('{},'.format(probe_name))
        # for idx in range(0, len(config['EFFICIENCY_ATTACH_TO'])):
        #     fh.write('{},'.format(config['EFFICIENCY_ATTACH_TO'][idx]))

def infer_func(model, x, y):
    loss_func = nn.CrossEntropyLoss()
    predictions = model(x)
    num = len(x)
    accuracy = (torch.argmax(predictions, axis=1) == y).float().sum() / num
    loss = loss_func(predictions, y.long())
    #print('predictions:', predictions, ' label:', y)
    # print('num:', num)
    # print('x:', x, '\n y:', y)
    # print('accuracy:', accuracy)
    # print('loss:', loss)

    return loss, accuracy, num

def epoch_func(model, x, y, train=False):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,
                          weight_decay=0.0005, nesterov=True)
    predictions = model(x)
    num = len(x)
    accuracy = (torch.argmax(predictions, axis=1) == y).float().sum() / num
    loss = loss_func(predictions, y.long())
    #print('predictions:', predictions, ' label:', y)
    # print('num:', num)
    # print('x:', x, '\n y:', y)
    # print('accuracy:', accuracy)
    # print('loss:', loss)

    if train:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss, accuracy, num

'''
This is the image pre-processing for TrojAI Round3 AI models
'''
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

'''
This is the image pre-processing for TrojAI Round4 AI models
'''
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

'''
This method evaluates an ai model using images loaded from disk
'''
def eval_model(model, test_loader, dev, which_round):
    model.to(dev)
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

'''
This method evaluates an ai model using images loaded from a serialized npy file (data_array)
with a specific structure (array, file name encoding class labels) 
'''
def eval_model_from_npy(model, im_class_label, npy_image_indices, npy_data_array, dev, which_round):
    model.to(dev)
    model.eval()
    # losses = []
    # accuracies = []
    # nums = []

    ####
    # DEBUG:
    total_items = npy_data_array.shape[0]
    print(f"> Total: {total_items} items")
    print(f"> Item size: {npy_data_array[0][0].shape}")

    # TODO - add support
    if im_class_label == 'all':
        print('ERROR: analysis from .npy file for all class labels is not supported yet ')
        return

    final_accuracy = 0
    final_num = 0

    with torch.no_grad():
        for i in range(len(npy_image_indices)):
            # label = test_loader.dataset.__getitem__(i)
            labels = []
            labels.append(im_class_label)
            # print('label:', labels)
            # prepare input batch

            # TODO .npy support is not completed
            # read the image from npy file
            item_index = npy_image_indices[i]
            img = Image.fromarray(npy_data_array[item_index][0], 'RGB')
            test_out_filename = '/mnt/raid1/pnb/trojai/datasets/round4/scratch_r4/temp2.png'
            img.save(test_out_filename)

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
                # print('inputs are on the GPU:', inputs.is_cuda)
                targets = torch.cuda.FloatTensor(labels)
                targets = targets.to(dev)
                # print('targets are on the GPU:', targets.is_cuda)

            loss, accuracy, num = epoch_func(model, inputs, targets, False)
            if dev == 'cpu':
                # losses.append(loss)
                # accuracies.append(accuracy)
                # nums.append(num)
                final_accuracy += accuracy
                final_num += num
            else:
                # retrieve results from a GPU
                # losses.append(loss.cpu().detach().item())
                # accuracies.append(accuracy.cpu().detach().item())
                # nums.append(num)
                final_accuracy += (accuracy.cpu().detach().item() * num)
                final_num += num
                # clean up the GPU RAM
                del inputs
                del targets
                torch.cuda.empty_cache()
                gc.collect()

        print('DEBUG: eval_model_from_npy: finished model predictions for a batch of images')
        #return losses, accuracies, nums
        final_accuracy = final_accuracy/final_num
        return final_accuracy


'''
evaluate AI model per image class label 
the input is provided in valid_dl with images of a selected class label or 
in npy_data_array with class label specified in ima_class_label
'''
def evaluate_ai_model(efficiency_model, im_class_label, valid_dl, npy_data_array, dev, which_round):
    """ Evaluate model efficiency """
    # # Attach StateCapture layers to the model
    # efficiency_model = ts.build_efficiency_model(model,attach_to=['Conv2d'],method='after')

    # Collect the states for each layer
    print()
    print('evaluate_ai_model_efficiency: Running model predictions to capture states...')
    start = time.time()
    ########################
    # return values computed on gpu back to cpu
    if npy_data_array is None:
        # images are loaded from disk
        #losses, accuracies, nums = eval_model(efficiency_model, valid_dl, dev, which_round)
        accuracy = eval_model(efficiency_model, valid_dl, dev, which_round)
    else:
        # images are loaded from npy file
        accuracy = eval_model_from_npy(efficiency_model, im_class_label, valid_dl, npy_data_array, dev, which_round)

    # print('losses: ', losses)
    # print('accuracies: ', accuracies)
    # print('nums: ', nums)

    eval_model_time = time.time() - start
    print('evaluate_ai_model_efficiency: Finished in {:.2f}s!'.format(eval_model_time))
    # accuracy = np.sum(np.multiply(accuracies, nums)) / np.sum(nums)
    # print('Network accuracy: {:.1f}%'.format(100 * accuracy))

    #return losses, accuracies, nums, eval_model_time
    return accuracy, eval_model_time


'''
This method is for reading the serialized images in .npy file
'''
def read_sample_list(data_array):
    total_items = data_array.shape[0]
    print(f"> Total: {total_items} items")

    item_index = randint(0, total_items - 1)

    print(f"> Item size: {data_array[item_index][0].shape}")
    print(f"> Sample name: {data_array[item_index][1]}")

    file_name = []
    for i in range(0, total_items ):
        # add the missing image suffix for the parser
        file_name.append(str(data_array[i][1]) + '.png')

    return file_name


def main(model_dirpath,model_filepath, example_images_dirpath, poisoned_example_images_dirpath, output_dirpath, example_img_format ):

    # create the output folder if it does not exist
    if os.path.exists(output_dirpath):
        # TODO decide whether we want to remove previously created output folder
        #shutil.rmtree(output_dirpath)
        print('INFO: output_dirpath:', output_dirpath, ' already exists')
    else:
        os.makedirs(output_dirpath)

    # Set the device to run the model on (gpu if available, cpu otherwise)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dev = "cpu"

    if dev != 'cpu':
        print('current device index:', str(torch.cuda.current_device()) )
        # print('current device name:', str(torch.cuda.get_device_name(device=﻿0﻿)) )
        # print('number of cuda devices:', str(torch.cuda.device_count(﻿)) )

    # this flag is to accommodate folder structure changes between Round 3 and Round 4
    which_round = 4

    # based on the round 4 description at https://pages.nist.gov/trojai/docs/data.html#round-4
    # "Within each trained AI model there can be {0, 1, or 2} one-to-one triggers. "
    # TODO this max value might change for experimental design datasets!!
    max_number_triggers = 2

    ##### create model_dirpath and model_basename
    if which_round == 3:
        model_basename = os.path.split(model_filepath)[0]
        model_basename = os.path.split(model_basename)[0]
        model_dirpath = os.path.split(model_basename)[0]
        model_basename = os.path.split(model_basename)[1]

        print("Round ", which_round, ": bName: ", ": dirPath: ", model_dirpath )

        # read the training and testing model accuracy from the file in the model folder: DataParallel_resnet18.pt.1.stats.json
        stats_fn = [fn for fn in os.listdir(os.path.join(model_dirpath, model_basename, 'model')) if fn.endswith('json')][0]

        stats_fp = os.path.join(model_dirpath, model_basename, 'model', stats_fn)
        config_fp = os.path.join(model_dirpath, model_basename, 'config.json')  # round3
    else:
        model_basename = os.path.split(model_dirpath)[1]
        print("Round ", which_round, ": bName: ", model_basename, ": dirPath: ", model_dirpath)
        # Todo: what is the file structure supposed to be
        # originally had model_stats.json and config.json in dataset/c75/, moved it to dataset/c75/model
        stats_fp = os.path.join(model_dirpath, 'model_stats.json')
        config_fp = os.path.join(model_dirpath, 'config.json')  # round4

    with open(stats_fp, 'r') as fp:
        stats = json.load(fp)
    final_combined_val_acc = float(stats['final_combined_val_acc'])

    # read the model configuration from the model configuration file : config.json
    with open(config_fp, 'r') as fp:
        config_json = json.load(fp)

    # change of upper case to lower case config fields between round 3 and round 4!!!!
    if which_round == 3:
        number_predicted_classes = int(config_json['NUMBER_CLASSES'])
        num_training_samples = int(config_json['NUMBER_TRAINING_SAMPLES'])
        #TODO: change back num_images_per_class to the line below
        num_example_images_per_class = int(config_json['NUMBER_EXAMPLE_IMAGES'])
        model_architecture = config_json['MODEL_ARCHITECTURE']
        model_poisoned = bool(config_json['POISONED'])
    else:
        number_predicted_classes = int(config_json['number_classes'])
        num_training_samples = int(config_json['number_training_samples'])
        num_example_images_per_class = int(config_json['number_example_images'])
        model_architecture = config_json['model_architecture']
        model_poisoned = bool(config_json['poisoned'])

    triggered_classes = []
    trigger_target_class = []
    triggered_fraction = []
    trigger_type= []

    if not model_poisoned:
        gt_model_label = 0
        number_triggers = 0
        for idx_triggger in range(number_triggers, max_number_triggers):
            triggered_classes.append(-1)
            trigger_target_class.append(-1)
            triggered_fraction.append(-1.0)
            trigger_type.append('none')
    else:
        gt_model_label = 1
        if which_round == 3:
            number_triggers = int(config_json['NUMBER_TRIGGERED_CLASSES']) # TODO check if this number could be > 1 in Round3
            triggered_classes.append( int(config_json['TRIGGERED_CLASSES']) )# TODO this is an array !!
            trigger_target_class.append( int(config_json['TRIGGER_TARGET_CLASS']) ) # TODO this might be an array in future !!
            triggered_fraction.append( float(config_json['TRIGGERED_FRACTION']) )
            trigger_type.append( config_json['TRIGGER_TYPE'] )
            for idx_triggger in range(1,max_number_triggers):
                triggered_classes.append( -1 )
                trigger_target_class.append( -1 )
                triggered_fraction.append( -1.0 )
                trigger_type.append('none')
        else:
            number_triggers = int(config_json['number_triggers'])
            for idx_triggger in range(0,number_triggers):
                triggered_classes.append( int(config_json['triggers'][idx_triggger]['source_class']) )
                trigger_target_class.append(  int(config_json['triggers'][idx_triggger]['target_class']) )
                triggered_fraction.append( float(config_json['triggers'][idx_triggger]['fraction']) )
                trigger_type.append( config_json['triggers'][idx_triggger]['type'] )
            for idx_triggger in range(number_triggers,max_number_triggers):
                triggered_classes.append( -1 )
                trigger_target_class.append( -1 )
                triggered_fraction.append( -1.0 )
                trigger_type.append('none')

    #####################################################
    # setup the configuration of the efficiency run
    config = dict()
    config['MODEL_FILEPATH'] = model_filepath
    config['SAMPLE_IMAGE_DIRPATH'] = example_images_dirpath
    config['POISONED_SAMPLE_IMAGE_DIRPATH'] = poisoned_example_images_dirpath
    config['OUTPUT_DIRPATH'] = output_dirpath
    config['OUTPUT_EFFICIENCY_FILENAME'] = 'metric.csv'

    # TODO modify which module and which method to use for attaching the hooks
    # TODO load the hooks/probes from a file in /mnt/raid1/pnb/trojai/datasets/round4/arch_probes_r4/classification
    task = 'classification'
    probe_vector = refprobes.select_probe(model_architecture,task)
    probe_name = 'all_probes'
    print('INFO: probe vector:', probe_vector)
    if probe_vector == None:
        print('ERROR: probe vector does not exist for:', model_architecture)
        output_filepath = os.path.join(config['OUTPUT_DIRPATH'], '{}'.format('failed_models.csv'))
        common_header = "model_filepath, model_architecture, model_basename"
        common_header += ", efficiency_attached_to_"
        # for idx in range(0, len(config['EFFICIENCY_ATTACH_TO'])):
        #     common_header += ", efficiency_attached_to " + str(idx)
        common_header += "\n"
        if not os.path.isfile(output_filepath):
            # write header
            with open(output_filepath, 'w') as fh:
                fh.write(common_header)

        with open(output_filepath, 'a') as fh:
            fh.write("{}, ".format(model_filepath))
            fh.write("{}, ".format(model_architecture))
            fh.write("{}, ".format(model_basename))
            fh.write("{} ".format(probe_name))
            fh.write("\n")
        return

    config['EFFICIENCY_ATTACH_TO'] = ['layer1.2.conv2', 'layer1.2.bn2'] #,'layer1.0.downsample.0'] #['layer1.0.conv1', 'layer2.0.conv1', 'layer3.0.conv1', 'layer4.0.conv1'] #probe_vector # ['Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'Sequential', 'BasicBlock', 'AdaptiveAvgPool2d', 'Linear']
    #config['EFFICIENCY_ATTACH_TO'] = ['conv', 'bn', 'relu', 'maxpool', 'layer', 'downsample', 'avgppool', 'flatten', 'fc'] # 'BatchNorm2d' # 'ReLU' # 'Conv2d'
    config['EFFICIENCY_ATTACH_TO_METHOD'] = 'after' #'after'

    print('configuration of hooks attached to: ', config['EFFICIENCY_ATTACH_TO'])
    ############################## load from the model_filepath
    print('loading model: ', model_filepath)
    torch.manual_seed(0)
    model = torch.load(model_filepath, map_location=dev)


    # assume initial that the number of predicted classes is the same as the number of evaluated classes
    number_evaluated_classes = number_predicted_classes

    # load the validation image examples assuming that they are saved as individial image files
    # example image file name class_0_example_0.png
    fns = [os.path.join(config['SAMPLE_IMAGE_DIRPATH'], fn) for fn in os.listdir(config['SAMPLE_IMAGE_DIRPATH']) if
           fn.endswith(example_img_format)]

    print("DEBUG:" , fns)

    # class ID encoding: # class_1_example_9.png
    # estimate the number of predicted classes from the example images
    num_images_used = len(fns)
    print("NUM IMAGES USED: ", num_images_used)
    # TODO specify as input parameter
    #  use only 25 % of all images for extracting the utilization metrics
    # TODO: figure out how to split the clean and poisoned images when evaluating subsets of training data
    # use triggered_fraction.append( float(config_json['triggers'][idx_triggger]['fraction']) and
    # the order of poisoned fraction follwoed by clean fraction

    #TODO : percent originally 1.0, and temp flag was false
    percent = 1.0

    num_images_used = int(percent*num_images_used)


    # determine how the images are stored
    npy_data_array = None
    #temp_flag = False
    temp_flag = True

    # if temp_flag and num_images_used < 1:
    #     example_img_format = '.npy'
    #     # this is the case of serialized images into .npy file
    #     file_npy = [os.path.join(config['SAMPLE_IMAGE_DIRPATH'], fn) for fn in os.listdir(config['SAMPLE_IMAGE_DIRPATH']) if
    #            fn.endswith(example_img_format)]
    #     if len(file_npy) > 0:
    #         print(f"Loading {file_npy[0]}...")
    #         #npy_data_array = np.load(file_npy[0], allow_pickle=True)
    #
    #
    #
    #         # print("====  Clean set   ====")
    #         # fns = read_sample_list(npy_data_array[0])
    #         # num_images_used = len(fns)
    #         # print('INFO: clean num_images_used derived from .npy file:', num_images_used)
    #         #
    #         # if npy_data_array.shape[0] > 1:
    #         #     print("==== Poisoned set ====")
    #         #     fns1 = read_sample_list(npy_data_array[1])
    #         #     # TODO we could analyze poisoned image separately or together with clean images
    #
    #     else:
    #         print('ERROR: did not find any individual image files nor .npy file in the clean_example_data folder')
    #         exit(1)

    print("NPY array data: ", npy_data_array)
    estimated_number_evaluated_classes = 0
    unique_labels = []
    count_unique_labels = []
    # for i in range(number_predicted_classes):
    #     count_unique_labels.append(0)


    for i in range(num_images_used):
        # class_1_example_9.png or class_37_trigger_0_example_0.png
        basename = os.path.basename(fns[i])
        split_str = basename.split("_")
        # [ "class", "1" , "example", "9" ] length = 4
        length = len(split_str)
        # example in example
        if length >= 4 and split_str[length - 2] in "example":
            #label = int(split_str[length - 3])
            #grab number from filename
            label = int(split_str[1])
            found_match = False
            for idx in range(0, len(unique_labels)):
                if unique_labels[idx] == label:
                    count_unique_labels[idx] += 1
                    found_match = True
            if not found_match:
                unique_labels.append(label)
                count_unique_labels.append(1)
                estimated_number_evaluated_classes += 1

            # NOTE: this solution works only for consecutive class labels
            # if label in unique_labels:
            #     count_unique_labels[label] +=1
            # else:
            #     if label < 0 or label >= number_predicted_classes:
            #         #count_unique_labels.append(1)
            #         print('INFO: skipping unexpected label out of range:', label)
            #         continue
            #     else:
            #         count_unique_labels[label] = 1
            #
            #     unique_labels.append(label)
            #     estimated_number_predicted_classes += 1
        else:
            print("ERROR: image file names do not follow expected convention: class_1_example_9.png: ", fns[i])
            continue

    # sanity check
    if estimated_number_evaluated_classes != number_evaluated_classes:
        print('WARNING: mismatch between the number_evaluated_classes:', number_evaluated_classes,
              ' and estimated estimated_number_predicted_classes:', estimated_number_evaluated_classes)
        print('INFO: proceed with number_evaluated_classes and assume that examples from some classes are missing')

    if (num_example_images_per_class*number_evaluated_classes) != num_images_used:
        print('WARNING: mismatch between num_example_images_per_class in config.json:', num_example_images_per_class,
              ' and num_images_used in example folder:', num_images_used)
        print('INFO: proceed with num_images_used in example folder')

    # TODO: unmute later

    # for i in range(0, len(unique_labels)):
    #     if count_unique_labels[i] != num_example_images_per_class:
    #         print('WARNING: unique_label:', unique_labels[i], ' mismatched expected num_example_image_per_class:', num_example_images_per_class,
    #               ' and counted num_example_images:', count_unique_labels[i])

    # for i in unique_labels:
    #     if count_unique_labels[i] != num_example_images_per_class:
    #         print('WARNING: unique_label:', i, ' mismatched expected num_example_images:', num_example_images_per_class,
    #               ' and counted num_example_images:', count_unique_labels[i])

    print('number_predicted_classes:', number_predicted_classes)
    print('number_evaluated_classes:', number_evaluated_classes)
    print('total num_images_used:', num_images_used)
    print('unique_labels:', unique_labels)
    if len(count_unique_labels) > 0:
        print('number of example images per class:', count_unique_labels[0])
    else:
        print('INFO: exiting because it could not find any images with matching class numbers \n')
        return


    ####### DEBUG
    # total_items = npy_data_array.shape[0]
    # print(f"> after init: Total: {total_items} items")

    ######################################################################################
    # prepare headers of all output files per model
    output_filepath = os.path.join(config['OUTPUT_DIRPATH'], '{}'.format('metric.csv'))
    common_header = "model_filepath, model_architecture, model_basename, num_predicted_classes, " \
                    "num_images_used, num_training_samples, " \
                    "final_combined_val_acc, poisoned_model_label, number_triggers, "

    for idx_triggger in range(0, max_number_triggers):
        common_header += "source_trigger_class " + str(idx_triggger) + ','
        common_header += "trigger_target_class " + str(idx_triggger) + ','
        common_header += "triggered_fraction " + str(idx_triggger) + ','
        common_header += "trigger_type " + str(idx_triggger) + ','

    common_header += "efficiency_attached_to " + ","
    # for idx in range(0, len(config['EFFICIENCY_ATTACH_TO'])):
    #     common_header += "efficiency_attached_to " + str(idx) + ","

    ##########################################################################################################
    ### Evaluate an AI model by using all images
    #####################################################################
    evaluation_over_classes = True

    if evaluation_over_classes:
        # prepare the output files
        is_per_label = False
        prepare_output_files(is_per_label, output_filepath, common_header, model_filepath, model_architecture,
                             model_basename, number_predicted_classes,
                             num_images_used, num_training_samples, final_combined_val_acc, gt_model_label,
                             number_triggers,
                             triggered_classes, trigger_target_class, triggered_fraction, trigger_type, probe_name)

        ################ Prepare images for inference
        if npy_data_array is None:
            # valid_dl = fns
            counter = 0
            num_images_per_batch = 10
            batch_counter = 1
            outbatch = []
            for f in fns:
                basename = os.path.basename(f)
                split_str = basename.split("_")
                # [ "class", "1" , "example", "9" ] length = 4
                length = len(split_str)
                # example in example
                if length >= 4 and split_str[length - 2] in "example":
                    # label = int(split_str[length - 3])
                    # grab number from filename
                    extracted_label = int(split_str[1])
                else:
                    print("error grabbing label")

                im = Image.open(f)
                im = (np.array(im))

                r = im[:, :, 0].flatten()
                g = im[:, :, 1].flatten()
                b = im[:, :, 2].flatten()
                label = [extracted_label]

                outfilepath = os.path.join(config['OUTPUT_DIRPATH'], '{}'.format('data_batch_'))
                outfilepath = outfilepath + str(batch_counter)
                out = np.array(list(label) + list(r) + list(g) + list(b), np.uint8)
                outbatch.append(out)

                if(counter > num_images_per_batch):
                    #outbatch.tofile(outfilepath)
                    np.save(outfilepath, outbatch)


                    ##TODO: ^^^ pickle this file

                    outbatch.clear()
                    counter = 0
                    batch_counter = batch_counter + 1

                counter = counter +1

        else:
            # just pass the list of npy array indices for loading image data
            valid_dl = fns
            example_img_format = '.npy'
            # this is the case of serialized images into .npy file
            file_npy = [os.path.join(config['SAMPLE_IMAGE_DIRPATH'], fn) for fn in
                        os.listdir(config['SAMPLE_IMAGE_DIRPATH']) if
                        fn.endswith(example_img_format)]
            if len(file_npy) > 0:
                print(f"Loading {file_npy[0]}...")
                npy_data_array = np.load(file_npy[0], allow_pickle=True)
                ############### DEBUG
                total_items = npy_data_array.shape[0]
                print(f"> Before Total: {total_items} items")
                print(f"> beginning Item size: {npy_data_array[0][0].shape}")

        print('INFO: evaluate model efficiency and save to directory ', config['OUTPUT_DIRPATH'])


######################################################################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Load AI model and training data')
    parser.add_argument('--model_dirpath', type=str, required=True)
    parser.add_argument('--model_filepath', type=str, required=True)
    parser.add_argument('--example_images_dirpath', type=str, required=True)
    parser.add_argument('--poisoned_example_images_dirpath', type=str, required=True)
    parser.add_argument('--output_dirpath', type=str, required=True)
    parser.add_argument('--example_img_format', type=str,
                        help='Exampple image file format (suffix)  which might be useful for filtering a folder containing axample files.',
                        required=False)
    args = parser.parse_args()
    if args.model_filepath is None:
        print('ERROR: missing model_filepath ')
        exit(1)

    if args.output_dirpath is None:
        print('ERROR: missing output dir path ')
        exit(1)

    image_format = args.example_img_format
    if not image_format:
        image_format = 'png'

    # execute only if run as the entry point into the program
    #main(args.model_dirpath,args.model_filepath, args.example_images_dirpath, args.poisoned_example_images_dirpath, args.output_dirpath, image_format)
    main(args.model_dirpath, args.model_filepath, args.example_images_dirpath, args.poisoned_example_images_dirpath,
         args.output_dirpath, args.example_image_format)
