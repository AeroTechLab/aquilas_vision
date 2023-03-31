#################################################################################
# Copyright (c) 2018-2022, Texas Instruments Incorporated - http://www.ti.com
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################


from . import constants
from ... import utils
from . import datasets
from . import training
from . import compilation
from .params import init_params
from ... import version


def _get_paretto_front_best(xy_list, x_index=0, y_index=1, inverse_relaionship=False):
    xy_list = sorted(xy_list, key=lambda x:x[x_index], reverse=inverse_relaionship)
    paretto_front = [xy_list[0]]
    for xy in xy_list[1:]:
        if xy[y_index] >= paretto_front[-1][y_index]:
            paretto_front.append(xy)
        #
    #
    # sort based on first index - reverse order in inference time is ascending order in FPS (faster performance)
    paretto_front = sorted(paretto_front, key=lambda x:x[x_index], reverse=True)
    return paretto_front


def _get_paretto_front_approx(xy_list, x_index=0, y_index=1, inverse_relaionship=False):
    # normalize the values
    min_x = min(xy[0] for xy in xy_list)
    max_x = max(xy[0] for xy in xy_list)
    min_y = min(xy[1] for xy in xy_list)
    max_y = max(xy[1] for xy in xy_list)
    norm_xy_list = [[(xy[0]-min_x+1)/(max_x-min_x+1), (xy[1]-min_y+1)/(max_y-min_y+1), xy[2]] for xy in xy_list]
    if inverse_relaionship:
        efficiency_list = [list(xy)+[xy[y_index]*xy[x_index]] for xy in norm_xy_list]
    else:
        efficiency_list = [list(xy)+[xy[y_index]/xy[x_index]] for xy in norm_xy_list]
    #
    efficiency_list = sorted(efficiency_list, key=lambda x:x[-1], reverse=True)
    # take the good models
    num_models_selected = max(len(efficiency_list)*2//3, 1)
    efficiency_list = efficiency_list[:num_models_selected]
    selected_indices = [xy[2] for xy in efficiency_list]
    selected_entries = [xy for xy in xy_list if xy[2] in selected_indices]
    # sort based on first index - reverse order in inference time is ascending order in FPS (faster performance)
    paretto_front = sorted(selected_entries, key=lambda x:x[x_index], reverse=True)
    return paretto_front


def get_paretto_front_combined(xy_list, x_index=0, y_index=1, inverse_relaionship=False):
    paretto_front_best = _get_paretto_front_best(xy_list, x_index=x_index, y_index=y_index, inverse_relaionship=inverse_relaionship)
    paretto_front_approx = _get_paretto_front_approx(xy_list, x_index=x_index, y_index=y_index, inverse_relaionship=inverse_relaionship)
    paretto_front_combined = paretto_front_best + paretto_front_approx
    # de-duplicate
    selected_indices = [xy[2] for xy in paretto_front_combined]
    selected_indices = set(selected_indices)
    paretto_front = [xy for xy in xy_list if xy[2] in selected_indices]
    # sort based on first index - reverse order in inference time is ascending order in FPS (faster performance)
    paretto_front = sorted(paretto_front, key=lambda x:x[x_index], reverse=True)
    return paretto_front


def set_model_selection_factor(model_descriptions):
    for m in model_descriptions.values():
        for target_device in m.training.target_devices.keys():
            m.training.target_devices[target_device].model_selection_factor = None
        #
    #

    task_types = set([m.common.task_type for m in model_descriptions.values()])
    target_devices = [list(m.training.target_devices.keys()) for m in model_descriptions.values()]
    target_devices = set([t for t_list in target_devices for t in t_list])
    for target_device in target_devices:
        for task_type in task_types:
            model_desc_list = [m for m in model_descriptions.values() if m.common.task_type == task_type]
            model_desc_list = [m for m in model_desc_list if target_device in list(m.training.target_devices.keys())]
            model_desc_list = [m for m in model_desc_list \
                                   if m.training.target_devices[target_device].performance_infer_time_ms is not None and
                                      m.training.target_devices[target_device].accuracy_factor is not None]
            performance_infer_time_ms = [m.training.target_devices[target_device].performance_infer_time_ms for m in model_desc_list]
            accuracy_factor = [m.training.target_devices[target_device].accuracy_factor for m in model_desc_list]
            xy_list = [(performance_infer_time_ms[i], accuracy_factor[i], i) for i in range(len(performance_infer_time_ms))]
            xy_list = get_paretto_front_combined(xy_list)
            for paretto_id, xy in enumerate(xy_list):
                xy_id = xy[2]
                m = model_desc_list[xy_id]
                m.training.target_devices[target_device].model_selection_factor = paretto_id
            #
        #
    #


def get_training_module_descriptions(params):
    # populate a good pretrained model for the given task
    training_module_descriptions = training.get_training_module_descriptions(target_device=params.common.target_device,
                                                         training_device=params.training.training_device)
    #
    training_module_descriptions = utils.ConfigDict(training_module_descriptions)
    return training_module_descriptions


def get_model_descriptions(params):
    # populate a good pretrained model for the given task
    model_descriptions = training.get_model_descriptions(task_type=params.common.task_type,
                                                         target_device=params.common.target_device,
                                                         training_device=params.training.training_device)

    #
    model_descriptions = utils.ConfigDict(model_descriptions)
    set_model_selection_factor(model_descriptions)
    return model_descriptions


def get_model_description(model_name):
    assert model_name, 'model_name must be specified for get_model_description().' \
        'if model_name is not known, use the method get_model_descriptions() that returns supported models.'
    model_description = training.get_model_description(model_name)
    return model_description


def set_model_description(params, model_description):
    assert model_description is not None, f'could not find pretrained model for {params.training.model_name}'
    assert params.common.task_type == model_description['common']['task_type'], \
        f'task_type: {params.common.task_type} does not match the pretrained model'
    # get pretrained model checkpoint and other details
    params.update(model_description)
    return params


def get_preset_descriptions(params):
    return constants.PRESET_DESCRIPTIONS


def get_target_device_descriptions(params):
    return constants.TARGET_DEVICE_DESCRIPTIONS


def get_sample_dataset_descriptions(params):
    return constants.SAMPLE_DATASET_DESCRIPTIONS


def get_task_descriptions(params):
    return constants.TASK_DESCRIPTIONS


def get_version_descriptions(params):
    version_descriptions = {
        'version': version.get_version(),
        'sdk_version': constants.TARGET_SDK_VERSION,
        'sdk_release': constants.TARGET_SDK_RELEASE,
    }
    return version_descriptions


def get_tooltip_descriptions(params):
    return {
        'common': {
        },
        'dataset': {
        },
        'training': {
            'training_epochs': {
                'name': 'Epochs',
                'description': 'Epoch is a term that is used to indicate a pass over the entire training dataset. '
                           'It is a hyper parameter that can be tuned to get best accuracy. '
                           'Eg. A model trained for 30 Epochs may give better accuracy than a model trained for 15 Epochs.'
            },
            'learning_rate': {
                'name': 'Learning rate',
                'description': 'Learning Rate determines the step size used by the optimization algorithm '
                               'at each iteration while moving towards the optimal solution. '
                               'It is a hyper parameter that can be tuned to get best accuracy. '
                               'Eg. A small Learning Rate typically gives good accuracy while fine tuning a model for a different task.'
            },
            'batch_size': {
                'name': 'Batch size',
                'description': 'Batch size specifies the number of inputs that are propagated through the '
                               'neural network in one iteration. Several such iterations make up one Epoch.'
                               'Higher batch size require higher memory and too low batch size can '
                               'typically impact the accuracy.'
            },
            'weight_decay': {
                'name': 'Weight decay',
                'description': 'Weight decay is a regularization technique that can improve '
                               'stability and generalization of a machine learning algorithm. '
                               'It is typically done using L2 regularization that penalizes parameters '
                               '(weights, biases) according to their L2 norm.'
            },
        },
        'compilation': {
            'calibration_frames': {
                'name': 'Calibration frames',
                'description': 'Calibration is a process of improving the accuracy during fixed point quantization. '
                               'Typically, higher number of Calibration Frames give higher accuracy, but it can also be time consuming.'
            },
            'calibration_iterations': {
                'name': 'Calibration iterations',
                'description': 'Calibration is a process of improving the accuracy during fixed point quantization. Calibration happens in iterations. '
                               'Typically, higher number of Calibration Iterations give higher accuracy, but it can also be time consuming.'
            },
            'tensor_bits': {
                'name': 'Tensor bits',
                'description': 'Bitdepth used to quantize the weights and activations in the neural network. '
                               'The neural network inference happens at this bit precision. '
            },
            'detection_threshold': {
                'name': 'Detection threshold',
                'description': 'Also called Confidence Threshold. A threshold used to select good detection boxes. '
                               'This is typically applied before a before the Non Max Suppression. '
                               'Higher Detection Threshold means less false detections (False Positives), '
                               'but may also result in misses (False Negatives). '
            },
            'detection_top_k': {
                'name': 'Detection topK',
                'description': 'Number of detection boxes to be selected during the initial shortlisting before the Non Max Suppression.'
                               'A higher number is typically used while measuring accuracy, but may impact the performance. '
            }
        }
    }


def get_help_descriptions(params):
    tooltip_descriptions = get_tooltip_descriptions(params)

    tooltip_string = ''
    for tooltip_section_key, tooltip_section_dict in tooltip_descriptions.items():
        if tooltip_section_dict:
            tooltip_string += f'\n### {tooltip_section_key.upper()}'
            for tooltip_key, tooltip_dict in tooltip_section_dict.items():
                tooltip_string += f'\n#### {tooltip_dict["name"]}'
                tooltip_string += f'\n{tooltip_dict["description"]}'
            #
        #
    #

    help_string = f'''
## Overview
This is a tool for collecting data, training and compiling AI models for use in TI embedded processors. The compiled models can be deployed on a local development board. A live preview/demo is also provided to inspect the quality of the developed model while it runs on the development board.

## Development flow
Bring your own data (BYOD): Retrain TI models from TI Model Zoo to fine-tune performance for your unique application requirements.

## Tasks supported
* {constants.TASK_DESCRIPTIONS[constants.TASK_TYPE_CLASSIFICATION]['task_name']}
* {constants.TASK_DESCRIPTIONS[constants.TASK_TYPE_DETECTION]['task_name']}

## Target device setup overview
Data capture from a development board is supported over ethernet connection. The live stream appears in the browser window and user can capture frames as needed. Similarly live preview/inference/demo can also be streamed into the browser window. 

To establish a connection with a physical development board over ethernet, please follow the steps below. Also use the supported SDK version for that device - given in the details below.
* Step 1: Make sure that you have physical development board of specific device with you procured, refer below to find how to procure for each specific device.
* Step 2: Download the image to be flashed in SD card (refer steps 3)
* Step 3: Make sure that the development board is setup and also put in the same local area network as the computer where you are using this service. Also connect a USB camera to the development board.
* Step 4: Get the IP address of the development board using serial port connection
* Step 5: Connect to the development board using ssh and run device agent service as mentioned below. 
```
ssh root@<ip_address_of_dev_board> 
cd /opt/edgeai-studio-agent/src 
python3 ./device_agent.py
```
* Step 6: Now you can connect to development board from model composer by providing the IP address of development board.

## Supported target devices
These are the devices that are supported currently. As additional devices are supported, this section will be updated.

### {constants.TARGET_DEVICE_TDA4VM}
{constants.TARGET_DEVICE_SETUP_INSTRUCTIONS_TDA4VM}

### {constants.TARGET_DEVICE_AM62A}
{constants.TARGET_DEVICE_SETUP_INSTRUCTIONS_AM62A}

### {constants.TARGET_DEVICE_AM68A}
{constants.TARGET_DEVICE_SETUP_INSTRUCTIONS_AM68A}

## Additional information
{constants.EDGEAI_TARGET_DEVICE_ADDITIONAL_INFORMATION}

## Dataset format
- The dataset format is similar to that of the [COCO](https://cocodataset.org/) dataset, but there are some changes as explained below.
- The annotated json file and images must be under a suitable folder with the dataset name. 
- Under the folder with dataset name, the following folders must exist: 
- (1) there must be an "images" folder containing the images
- (2) there must be an annotations folder containing the annotation json file with the name given below.

#### Object Detection dataset format
An object detection dataset should have the following structure. 

<pre>
data/datasets/dataset_name
                             |
                             |--images
                             |     |-- the image files should be here
                             |
                             |--annotations
                                   |--instances.json
</pre>

- Use a suitable dataset name instead of dataset_name
- The default annotation file name for object detection is instances.json
- The format of the annotation file is similar to that of the [COCO dataset 2017 Train/Val annotations](https://cocodataset.org/#download) - a json file containing 'info', 'images', 'categories' and 'annotations'.
- Look at the example dataset [animal_classification](https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/datasets/animal_classification.zip) to understand further.

#### Image Classification dataset format
An image classification dataset should have the following structure. (Use a suitable dataset name instead of dataset_name).

<pre>
data/datasets/dataset_name
                             |
                             |--images
                             |     |-- the image files should be here
                             |
                             |--annotations
                                   |--instances.json
</pre>

- Use a suitable dataset name instead of dataset_name
- The default annotation file name for image classification is instances.json
- The format of the annotation file is similar to that of the COCO dataset - a json file containing 'info', 'images', 'categories' and 'annotations'. However, one difference is that the bounding box information is not used for classification task and need not be present. The category information in each annotation (called the 'id' field) is needed.
- Look at the example dataset [animal_detection](https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/datasets/animal_detection.zip) to understand further.

## Model deployment
- The deploy page provides a button to download the compiled model artifacts to the target device / EVM. 
- The downloaded model artifacts are located in a folder inside /opt/projects.
- These model artifacts can be used with edgeai-gst-apps included in the SDK to run inference. 
- Please see the section "Edge AI sample apps" in the SDK documentation for more information.

## Glossary of terms
{tooltip_string}
'''
    return help_string
