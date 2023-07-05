"""
Disclaimer
This software was developed by employees of the National Institute of Standards and Technology (NIST), an agency of the Federal Government and is being made available as a public service. Pursuant to title 17 United States Code Section 105, works of NIST employees are not subject to copyright protection in the United States.  This software may be subject to foreign copyright.  Permission in the United States and in foreign countries, to the extent that NIST may hold copyright, to use, copy, modify, create derivative works, and distribute this software and its documentation without fee is hereby granted on a non-exclusive basis, provided that this notice and disclaimer of warranty appears in all copies.
THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE.  IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.
"""
__author__ = "Peter Bajcsy"
__copyright__ = "Copyright 2020, The IARPA funded TrojAI project"
__credits__ = ["Michael Majurski", "Tim Blattner", "Derek Juba", "Walid Keyrouz"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Peter Bajcsy"
__email__ = "peter.bajcsy@nist.gov"
__status__ = "Research"

import json
import os
import time
import torch
import numpy as np
#import class_efficiency
import attribution_method
import csv


"""
This class is designed to loop over a folders containing AI models and training images provided
for the TrojAI Round 1-3 Challenge - see https://pages.nist.gov/trojai/docs/data.html#round-2
"""


def batch_efficiency(model_dirpath, result_dirpath, model_format='.pt', example_img_format='png'):
    print('model_dirpath = {}'.format(model_dirpath))
    print('result_filepath = {}'.format(result_dirpath))
    print('model_format = {}'.format(model_format))
    print('example_img_format = {}'.format(example_img_format ))

    # Identify all models in directories--up
    model_dir_names = os.listdir(model_dirpath)
    print('os.listdir(model_dirpath):',os.listdir(model_dirpath))
    model_filepath = []
    array_model_dir_names = []
    idx = 0
    ## model_dir_names only has c75pr2-resnet dir in it, loop thru files in first level of c76pr2-res
    for fn in model_dir_names:
        model_dirpath1 = os.path.join(model_dirpath, fn)
        model_dirpath1 = os.path.join(model_dirpath1, 'model')

        # looking for dataset/c75pr2/model directory
        isdir = os.path.isdir(model_dirpath1)
        if isdir:
            # get all models in dataset/c75pr2/model ending in .pt
            for fn1 in os.listdir(model_dirpath1):
                if fn1.endswith(model_format):
                    model_filepath.append(os.path.join(model_dirpath1, fn1))
                    array_model_dir_names.append(model_dirpath1)
                    idx = idx + 1

    number_of_models = idx
    print('number of models:', number_of_models, '\n model_filepath array:', model_filepath)
    #array_model_dir_names = np.asarray(model_dir_names)

    examples_dirpath = []
    idx = 0
    for fn in model_dir_names:
        model_dirpath1 = os.path.join(model_dirpath, fn)
        if not os.path.isdir(model_dirpath1):
            continue
        examples_dirpath1 = os.path.join(model_dirpath, fn, 'clean_example_data/')
        # check if examples_dirpath1 exists
        isdir = os.path.isdir(examples_dirpath1 )
        if not isdir:
            continue

        examples_dirpath.append(examples_dirpath1)
        idx = idx + 1

    # TODO : added poisoned dir

    poison_dirpath = []

    ind = 0
    for fn in model_dir_names:
        model_dirpath1 = os.path.join(model_dirpath, fn)
        if not os.path.isdir(model_dirpath1):
            continue
        poison_dirpath1 = os.path.join(model_dirpath, fn, 'poisoned_example_data/')
        # check if examples_dirpath1 exists
        isdir = os.path.isdir(poison_dirpath1)
        if not isdir:
            continue

        print("poisioned directory: ", poison_dirpath1)
        poison_dirpath.append(poison_dirpath1)
        ind = ind + 1

    number_of_poisondir = ind
    print('number of poisoned dirs:', number_of_poisondir, '\n')

    ####################

    number_of_exampledir = idx
    print('number of example dirs:', number_of_exampledir, '\n')

    # sanity check
    if number_of_models != number_of_exampledir:
        print('ERROR: mismatch between number of example dirs:', number_of_exampledir, ' and number of models: ', number_of_models, '\n')
        return False

    ################## loop over all models ##############################
    start = time.time()

    for idx in range(0, number_of_models):
        print('processing model_filepath:', model_filepath[idx])
        print('model dir:', array_model_dir_names[idx])
        basename = os.path.split(array_model_dir_names[idx])[0]
        basename = os.path.split(basename)[1]
        print('basename:', basename)

        start1 = time.time()
        ##################
        # run the efficiency measurement

        # attribution_comparison.main(array_model_dir_names[idx], model_filepath[idx], examples_dirpath[idx], poison_dirpath[idx], result_dirpath,
        #                        example_img_format) #kb

        attribution_method.main(model_filepath[idx], examples_dirpath[idx], result_dirpath)

        end1 = time.time()
        print('model: ', array_model_dir_names[idx])
        print('total processing time for this model:', (end1-start1))

    end = time.time()
    print('total processing time for all models:', (end-start))


    return True


    



if __name__ == "__main__":
    import argparse

    print('torch version: %s \n' % (torch.__version__))

    parser = argparse.ArgumentParser(
        description='Efficiency estimator of AI Models to Demonstrate dependency on the number of predicted classes.')
    parser.add_argument('--model_dirpath', type=str, help='Directory path to all pytorch generated data and models to be evaluated.',
                        required=True)
    parser.add_argument('--output_dirpath', type=str,
                        help='Directory path  where output result should be written.', required=True)
    parser.add_argument('--model_format', type=str,
                        help='Model file format (suffix)  which might be useful for filtering a folder containing a model .',
                        required=False)
    parser.add_argument('--image_format', type=str,
                        help='Exampple image file format (suffix)  which might be useful for filtering a folder containing axample files.',
                        required=False)

    args = parser.parse_args()
    print('args %s \n % s \n %s \n %s \n' % (
        args.model_dirpath, args.output_dirpath, args.model_format, args.image_format))

    batch_efficiency(args.model_dirpath, args.output_dirpath)

    # example inputs:
    # --model_dirpath     C:\PeterB\Projects\TrojAI\nn-efficiency-metrics\data\models0to1_nisaba
    # --output_dirpath     C:\PeterB\Projects\TrojAI\nn-efficiency-metrics\scratch
    # --model_dirpath     /mnt/raid1/pnb/trojai/datasets/round4/models0to1_nisaba /mnt/raid1/pnb/trojai/datasets/round4/round4-train-dataset
    # --output_dirpath     /mnt/raid1/pnb/trojai/datasets/roun
