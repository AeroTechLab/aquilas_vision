#################################################################################
# Copyright (c) 2018-2021, Texas Instruments Incorporated - http://www.ti.com
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

import copy
import types
import typing
import torch
import torch.ao.quantization.quantize_fx as quantize_fx
import edgeai_torchtoolkit.v1.tools.xnn as xnn
from .import qconfig

__all__ = ['prepare', 'load_weights', 'freeze', 'train', 'eval', 'convert']


def _set_quant_backend(backend=None):
    if backend:
        if backend not in torch.backends.quantized.supported_engines:
            raise RuntimeError("Quantized backend not supported: " + str(backend))
        #
        torch.backends.quantized.engine = backend
    #


def load_weights(model, pretrained=None, change_names_dict=None):
    # Load weights for accuracy evaluation of a QAT model
    if pretrained is not None and pretrained is not False:
        print("=> using pre-trained model from {}".format(pretrained))
        if hasattr(model, 'load_weights'):
            model.load_weights(pretrained, download_root='./data/downloads', change_names_dict=change_names_dict)
        else:
            xnn.utils.load_weights(model, pretrained, download_root='./data/downloads', change_names_dict=change_names_dict)
        #
    #


def prepare(model, qconfig_dict=None, pretrained=None, pretrained_after_prepare=False, backend=None,
            num_batch_norm_update_epochs=None, num_observer_update_epochs=None,
            is_qat=True, **kwargs):
    model = xnn.model_surgery.replace_modules(model, replacements_dict={torch.nn.ReLU6: [torch.nn.ReLU, 'inplace']})
    # if qat fx, fusion is needed/supported only for eval
    # if hasattr(model, 'fuse_model'):
    #     model.fuse_model(is_qat=is_qat)
    # else:
    #     model = quantize_fx.fuse_fx(model)
    # #
    _set_quant_backend(backend=backend)
    if qconfig_dict is None:
        qconfig_dict = {"": qconfig.get_qat_qconfig_for_target_device(backend, target_device=None)}
    #
    model.train()
    if not pretrained_after_prepare:
        load_weights(model, pretrained=pretrained)
    #
    # insert quant, dequant stubs - if it is not present
    has_quant_stub = [isinstance(m, torch.quantization.QuantStub) for m in model.modules()]
    has_dequant_stub = [isinstance(m, torch.quantization.DeQuantStub) for m in model.modules()]
    has_quant_stub = any(has_quant_stub)
    has_dequant_stub = any(has_dequant_stub)
    if (not has_quant_stub) or (not has_dequant_stub):
        if not has_quant_stub:
            model.quant = torch.ao.quantization.QuantStub()
        #
        if not has_dequant_stub:
            model.dequant = torch.ao.quantization.DeQuantStub() #QuantStub()
        #
        def _new_forward(model, *input: typing.Any):
            x = model.quant(*input) if not has_quant_stub else input
            x = model.forward(x)
            x = model.dequant(x) if not has_dequant_stub else x
            return x
        #
        model.forward = types.MethodType(_new_forward, model)
    #
    # prepare for quantization
    model = quantize_fx.prepare_qat_fx(model, qconfig_dict)
    if pretrained_after_prepare:
        load_weights(model, pretrained=pretrained)
    #
    # fake quantization for qat
    model.apply(torch.ao.quantization.enable_fake_quant)
    # observes for range estimation
    model.apply(torch.ao.quantization.enable_observer)
    # store additional information
    model.__quant_info__ = dict(num_batch_norm_update_epochs=num_batch_norm_update_epochs,
                                 num_observer_update_epochs=num_observer_update_epochs,
                                 num_epochs_tracked=0)
    return model


def freeze(model, freeze_bn=True, freeze_observers=True):
    if freeze_observers is True:
        model.apply(torch.ao.quantization.disable_observer)
    elif freeze_observers is False:
        model.apply(torch.ao.quantization.enable_observer)
    #
    if freeze_bn is True:
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    elif freeze_bn is False:
        model.apply(torch.nn.intrinsic.qat.update_bn_stats)
    #
    return model


def unfreeze(model, unfreeze_bn=True, unfreeze_observers=True):
    freeze(model, not unfreeze_bn, not unfreeze_observers)
    return model


def _get_quant_info(model):
    for m in model.modules():
        if hasattr(m, '__quant_info__'):
            return m.__quant_info__
        #
    #
    return None


def train(model):
    # put the model in train mode
    model.train()
    # freezing ranges after a few epochs improve accuracy
    __quant_info__ = _get_quant_info(model)
    if __quant_info__ is not None:
        num_batch_norm_update_epochs = __quant_info__['num_batch_norm_update_epochs']
        num_observer_update_epochs = __quant_info__['num_observer_update_epochs']
        num_epochs_tracked = __quant_info__['num_epochs_tracked']
        __quant_info__['num_epochs_tracked'] += 1
    else:
        num_batch_norm_update_epochs = None
        num_observer_update_epochs = None
        num_epochs_tracked = 0
    #
    num_batch_norm_update_epochs = num_batch_norm_update_epochs or 4
    num_observer_update_epochs = num_observer_update_epochs or 6
    freeze(model, freeze_bn=(num_epochs_tracked>=num_batch_norm_update_epochs),
           freeze_observers=(num_epochs_tracked>=num_observer_update_epochs))
    return model


def eval(model):
    model.eval()
    freeze(model, freeze_bn=True, freeze_observers=True)
    return model


def convert(model, inplace=False):
    # make a copy inorder not to alter the original
    model = copy.deepcopy(model)
    # convert requires cpu model
    model = model.to(torch.device('cpu'))
    # now do the actual conversion
    model = quantize_fx.convert_fx(model)
    return model


##################################################################
# this is a convenient Module form of the above APIs
class QuantTorchFxModule(torch.nn.Module):
    def __int__(self, module, qconfig_dict=None, pretrained=None,
            pretrained_after_prepare=False, backend=None,
            num_batch_norm_update_epochs=None, num_observer_update_epochs=None):
        super().__init__()
        self.module = prepare(module, qconfig_dict=qconfig_dict, pretrained=pretrained,
            pretrained_after_prepare=pretrained_after_prepare, backend=backend,
            num_batch_norm_update_epochs=num_batch_norm_update_epochs,
            num_observer_update_epochs=num_observer_update_epochs)

    def load_weights(self, pretrained):
        load_weights(self.module, pretrained=pretrained)

    def train(self):
        self.module = train(self.module)

    def eval(self):
        self.module = eval(self.module)

    def freeze(self):
        self.module = freeze(self.module)

    def unfreeze(self):
        self.module = unfreeze(self.module)

    def convert(self):
        return convert(self.module)
