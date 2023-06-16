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

class RefProbes():
    def __init__(self):
        super().__init__()

    ###########################
    ## image classification architectures
    VGG13_PROBES = ['Sequential', 'Conv2d', 'ReLU', 'MaxPool2d', 'AdaptiveAvgPool2d', 'Linear', 'Dropout']
    MNASNET1_0_PROBES = ['Sequential', 'Conv2d', 'BatchNorm2d', 'ReLU', '_InvertedResidual', 'Dropout', 'Linear']
    RESNEXT50_32X4D_PROBES = ['Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'Sequential', 'Bottleneck', 'AdaptiveAvgPool2d', 'Linear']
    VGG16_PROBES = ['Sequential', 'Conv2d', 'ReLU', 'MaxPool2d', 'AdaptiveAvgPool2d', 'Linear', 'Dropout']
    DENSENET161_PROBES = ['Sequential', 'Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', '_DenseBlock', '_DenseLayer', '_Transition', 'AvgPool2d', 'Linear']
    SHUFFLENET1_5_PROBES = ['Sequential', 'Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'InvertedResidual', 'Linear']
    SHUFFLENET1_0_PROBES = ['Sequential', 'Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'InvertedResidual', 'Linear']
    RESNET18_PROBES = ['Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'Sequential', 'BasicBlock', 'AdaptiveAvgPool2d', 'Linear']
    WIDE_RESNET50_PROBES = ['Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'Sequential', 'Bottleneck', 'AdaptiveAvgPool2d', 'Linear']
    MNASNET0_75_PROBES = ['Sequential', 'Conv2d', 'BatchNorm2d', 'ReLU', '_InvertedResidual', 'Dropout', 'Linear']
    VGG19_BN_PROBES = ['Sequential', 'Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'AdaptiveAvgPool2d', 'Linear', 'Dropout']
    DENSENET201_PROBES = ['Sequential', 'Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', '_DenseBlock', '_DenseLayer', '_Transition', 'AvgPool2d', 'Linear']
    SQUEEZENETV1_0_PROBES = ['Sequential', 'Conv2d', 'ReLU', 'MaxPool2d', 'Fire', 'Dropout', 'AdaptiveAvgPool2d']
    RESNET50_PROBES = ['Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'Sequential', 'Bottleneck', 'AdaptiveAvgPool2d', 'Linear']
    SQUEEZENETV1_1_PROBES = ['Sequential', 'Conv2d', 'ReLU', 'MaxPool2d', 'Fire', 'Dropout', 'AdaptiveAvgPool2d']
    VGG16_BN_PROBES = ['Sequential', 'Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'AdaptiveAvgPool2d', 'Linear', 'Dropout']
    DENSENET121_PROBES = ['Sequential', 'Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', '_DenseBlock', '_DenseLayer', '_Transition', 'AvgPool2d', 'Linear']
    INCEPTIONV3_PROBES = ['BasicConv2d', 'Conv2d', 'BatchNorm2d', 'MaxPool2d', 'InceptionA', 'InceptionB', 'InceptionC', 'InceptionAux', 'Linear', 'InceptionD', 'InceptionE', 'AdaptiveAvgPool2d', 'Dropout']
    VGG11_BN_PROBES = ['Sequential', 'Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'AdaptiveAvgPool2d', 'Linear', 'Dropout']
    RESNET34_PROBES = ['Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'Sequential', 'BasicBlock', 'AdaptiveAvgPool2d', 'Linear']
    VGG13_BN_PROBES = ['Sequential', 'Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'AdaptiveAvgPool2d', 'Linear', 'Dropout']
    SHUFFLENET0_5_PROBES = ['Sequential', 'Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'InvertedResidual', 'Linear']
    RESNET101_PROBES = ['Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'Sequential', 'Bottleneck', 'AdaptiveAvgPool2d', 'Linear']
    MNASNET1_3_PROBES = ['Sequential', 'Conv2d', 'BatchNorm2d', 'ReLU', '_InvertedResidual', 'Dropout', 'Linear']
    RESNEXT101_32X8D_PROBES = ['Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'Sequential', 'Bottleneck', 'AdaptiveAvgPool2d', 'Linear']
    ALEXNET_PROBES = ['Sequential', 'Conv2d', 'ReLU', 'MaxPool2d', 'AdaptiveAvgPool2d', 'Dropout', 'Linear']
    DENSENET169_PROBES = ['Sequential', 'Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', '_DenseBlock', '_DenseLayer', '_Transition', 'AvgPool2d', 'Linear']
    SHUFFLENET2_0_PROBES = ['Sequential', 'Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'InvertedResidual', 'Linear']
    MNASNET0_5_PROBES = ['Sequential', 'Conv2d', 'BatchNorm2d', 'ReLU', '_InvertedResidual', 'Dropout', 'Linear']
    VGG11_PROBES = ['Sequential', 'Conv2d', 'ReLU', 'MaxPool2d', 'AdaptiveAvgPool2d', 'Linear', 'Dropout']
    RESNET152_PROBES = ['Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'Sequential', 'Bottleneck', 'AdaptiveAvgPool2d', 'Linear']
    WIDE_RESNET101_PROBES = ['Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'Sequential', 'Bottleneck', 'AdaptiveAvgPool2d', 'Linear']
    INCEPTIONV1_PROBES = ['BasicConv2d', 'Conv2d', 'BatchNorm2d', 'MaxPool2d', 'Inception', 'Sequential', 'InceptionAux', 'Linear', 'AdaptiveAvgPool2d', 'Dropout']
    MOBILENETV2_PROBES = ['Sequential', 'ConvNormActivation', 'Conv2d', 'BatchNorm2d', 'ReLU6', 'InvertedResidual', 'Dropout', 'Linear']
    VGG19_PROBES = ['Sequential', 'Conv2d', 'ReLU', 'MaxPool2d', 'AdaptiveAvgPool2d', 'Linear', 'Dropout']

    ###########################
    ## object detection architectures
    KEYPOINTRCNN_RESNET50_FPN_PROBES = ['GeneralizedRCNNTransform', 'BackboneWithFPN', 'IntermediateLayerGetter', 'Conv2d', 'FrozenBatchNorm2d', 'ReLU', 'MaxPool2d', 'Sequential', 'Bottleneck', 'FeaturePyramidNetwork', 'ModuleList', 'LastLevelMaxPool', 'RegionProposalNetwork', 'AnchorGenerator', 'RPNHead', 'RoIHeads', 'MultiScaleRoIAlign', 'TwoMLPHead', 'Linear', 'FastRCNNPredictor', 'KeypointRCNNHeads', 'KeypointRCNNPredictor', 'ConvTranspose2d']
    SSDLITE320_MOBILENET_V3_LARGE_PROBES = ['SSDLiteFeatureExtractorMobileNet', 'Sequential', 'ConvNormActivation', 'Conv2d', 'BatchNorm2d', 'Hardswish', 'InvertedResidual', 'ReLU', 'SqueezeExcitation', 'AdaptiveAvgPool2d', 'Hardsigmoid', 'ModuleList', 'ReLU6', 'DefaultBoxGenerator', 'SSDLiteHead', 'SSDLiteClassificationHead', 'SSDLiteRegressionHead', 'GeneralizedRCNNTransform']
    FASTERRCNN_MOBILENET_V3_LARGE_FPN_PROBES = ['GeneralizedRCNNTransform', 'BackboneWithFPN', 'IntermediateLayerGetter', 'ConvNormActivation', 'Conv2d', 'FrozenBatchNorm2d', 'Hardswish', 'InvertedResidual', 'Sequential', 'ReLU', 'SqueezeExcitation', 'AdaptiveAvgPool2d', 'Hardsigmoid', 'FeaturePyramidNetwork', 'ModuleList', 'LastLevelMaxPool', 'RegionProposalNetwork', 'AnchorGenerator', 'RPNHead', 'RoIHeads', 'MultiScaleRoIAlign', 'TwoMLPHead', 'Linear', 'FastRCNNPredictor']
    MASKRCNN_RESNET50_FPN_PROBES = ['GeneralizedRCNNTransform', 'BackboneWithFPN', 'IntermediateLayerGetter', 'Conv2d', 'FrozenBatchNorm2d', 'ReLU', 'MaxPool2d', 'Sequential', 'Bottleneck', 'FeaturePyramidNetwork', 'ModuleList', 'LastLevelMaxPool', 'RegionProposalNetwork', 'AnchorGenerator', 'RPNHead', 'RoIHeads', 'MultiScaleRoIAlign', 'TwoMLPHead', 'Linear', 'FastRCNNPredictor', 'MaskRCNNHeads', 'MaskRCNNPredictor', 'ConvTranspose2d']
    SSD300_VGG16_PROBES = ['SSDFeatureExtractorVGG', 'Sequential', 'Conv2d', 'ReLU', 'MaxPool2d', 'ModuleList', 'DefaultBoxGenerator', 'SSDHead', 'SSDClassificationHead', 'SSDRegressionHead', 'GeneralizedRCNNTransform']
    FASTERRCNN_MOBILENET_V3_LARGE_320_FPN_PROBES = ['GeneralizedRCNNTransform', 'BackboneWithFPN', 'IntermediateLayerGetter', 'ConvNormActivation', 'Conv2d', 'FrozenBatchNorm2d', 'Hardswish', 'InvertedResidual', 'Sequential', 'ReLU', 'SqueezeExcitation', 'AdaptiveAvgPool2d', 'Hardsigmoid', 'FeaturePyramidNetwork', 'ModuleList', 'LastLevelMaxPool', 'RegionProposalNetwork', 'AnchorGenerator', 'RPNHead', 'RoIHeads', 'MultiScaleRoIAlign', 'TwoMLPHead', 'Linear', 'FastRCNNPredictor']
    FASTERRCNN_RESNET50_FPN_PROBES = ['GeneralizedRCNNTransform', 'BackboneWithFPN', 'IntermediateLayerGetter', 'Conv2d', 'FrozenBatchNorm2d', 'ReLU', 'MaxPool2d', 'Sequential', 'Bottleneck', 'FeaturePyramidNetwork', 'ModuleList', 'LastLevelMaxPool', 'RegionProposalNetwork', 'AnchorGenerator', 'RPNHead', 'RoIHeads', 'MultiScaleRoIAlign', 'TwoMLPHead', 'Linear', 'FastRCNNPredictor']
    RETINANET_RESNET50_FPN_PROBES = ['BackboneWithFPN', 'IntermediateLayerGetter', 'Conv2d', 'FrozenBatchNorm2d', 'ReLU', 'MaxPool2d', 'Sequential', 'Bottleneck', 'FeaturePyramidNetwork', 'ModuleList', 'LastLevelP6P7', 'AnchorGenerator', 'RetinaNetHead', 'RetinaNetClassificationHead', 'RetinaNetRegressionHead', 'GeneralizedRCNNTransform']

    ###########################
    ## segmentation architectures
    SEG_RESNET101_PROBES = ['IntermediateLayerGetter', 'Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'Sequential', 'Bottleneck', 'FCNHead', 'Dropout']
    SEG_RESNET50_PROBES = ['IntermediateLayerGetter', 'Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'Sequential', 'Bottleneck', 'FCNHead', 'Dropout']
    MOBILENETV3_LARGE_PROBES = ['IntermediateLayerGetter', 'ConvNormActivation', 'Conv2d', 'BatchNorm2d', 'Hardswish', 'InvertedResidual', 'Sequential', 'ReLU', 'SqueezeExcitation', 'AdaptiveAvgPool2d', 'Hardsigmoid', 'DeepLabHead', 'ASPP', 'ModuleList', 'ASPPConv', 'ASPPPooling', 'Dropout']
    DEEPLAB101_PROBES = ['IntermediateLayerGetter', 'Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'Sequential', 'Bottleneck', 'DeepLabHead', 'ASPP', 'ModuleList', 'ASPPConv', 'ASPPPooling', 'AdaptiveAvgPool2d', 'Dropout']
    DEEPLAB50_PROBES = ['IntermediateLayerGetter', 'Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'Sequential', 'Bottleneck', 'DeepLabHead', 'ASPP', 'ModuleList', 'ASPPConv', 'ASPPPooling', 'AdaptiveAvgPool2d', 'Dropout']
    LR_ASPP_MOBILENETV3_LARGE_PROBES = ['IntermediateLayerGetter', 'ConvNormActivation', 'Conv2d', 'BatchNorm2d', 'Hardswish', 'InvertedResidual', 'Sequential', 'ReLU', 'SqueezeExcitation', 'AdaptiveAvgPool2d', 'Hardsigmoid', 'LRASPPHead', 'Sigmoid']


###################################################
CLASSIFICATION_MODEL_NAMES = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "wideresnet50", "wideresnet101",
                  "resnext50_32x4d","resnext101_32x8d",
               "densenet121", "densenet161", "densenet169", "densenet201",
                  "mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3",
               "inceptionv1(googlenet)", "inceptionv3","alexnet",
               "squeezenetv1_0", "squeezenetv1_1", "mobilenetv2",
               "shufflenet0_5", "shufflenet1_0", "shufflenet1_5", "shufflenet2_0",
               "vgg11", "vgg13", "vgg16", "vgg19",
               "vgg11bn", "vgg13bn", "vgg16bn", "vgg19bn"]

DETECT_MODEL_NAMES = ['fasterrcnn_resnet50_fpn','fasterrcnn_mobilenet_v3_large_fpn','fasterrcnn_mobilenet_v3_large_320_fpn',
                      'retinanet_resnet50_fpn','ssd300_vgg16','ssdlite320_mobilenet_v3_large',
                      'maskrcnn_resnet50_fpn', 'keypointrcnn_resnet50_fpn']

SEGMENT_MODEL_NAMES = ['Deeplab50', 'Deeplab101','MobileNetV3-Large','LR-ASPP-MobileNetV3-Large', 'Resnet50', 'Resnet101']


def select_probe(name, task):
    # this name had to be modified since parenthesis are not allowed in variable names
    if name == 'inceptionv1(googlenet)' or name == 'googlenet':
        # modify the name
        name_probe = 'INCEPTIONV1_PROBES'
    elif name == 'wideresnet50':
        name_probe = 'WIDE_RESNET50_PROBES'
    elif name == 'wideresnet101':
        name_probe = 'WIDE_RESNET101_PROBES'
    elif str(name).startswith('vgg') and str(name).endswith('bn'):
        index = str(name).rfind('bn')
        temp_name = name[0:index] + "_" + name[index: len(name)]
        name_probe = str(temp_name).upper() + '_PROBES'
    elif str(name).startswith('resnet') and task == 'segment':
        temp_name = 'SEG_' + name
        name_probe = str(temp_name).upper() + '_PROBES'
    else:
        # this is needed since - is not allowed in  the variable name
        temp_name = str(name).replace('-', '_')
        name_probe = str(temp_name).upper() + '_PROBES'

    print('name_probe:', name_probe)
    myprobe = RefProbes()
    try:
        method_to_call = getattr(myprobe, name_probe)
    except :
        print('ERROR: did not find a matching probe to the input architecture ', name)
        return None

    # print('info: method_to_call:', method_to_call)
    # for i in range(0, len(method_to_call)):
    #     print('elem:', i, ',', method_to_call[i])

    return method_to_call


######################################################################
if __name__ == '__main__':

    for arch_name in CLASSIFICATION_MODEL_NAMES:
        print('INFO: architecture name:', arch_name)
        task = 'classification'
        select_probe(arch_name, task)
    # for arch_name in DETECT_MODEL_NAMES:
    #     print('INFO: architecture name:', arch_name)
    #     select_probe(arch_name)

    # for arch_name in SEGMENT_MODEL_NAMES:
    #     print('INFO: architecture name:', arch_name)
    #     select_probe(arch_name)

