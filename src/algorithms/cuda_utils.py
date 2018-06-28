import tensorflow as tf
import torch
from tensorflow.python.client import device_lib
from torch.autograd import Variable


class GPUWrapper:
    def __init__(self, gpu):
        self.gpu = gpu

    @property
    def tf_device(self):
        local_device_protos = device_lib.list_local_devices()
        gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
        return tf.device(gpus[self.gpu] if gpus else '/cpu:0')

    @property
    def torch_device(self):
        return torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() else 'cpu')

    def to_var(self, x, **kwargs):
        """PyTorch only: send Var to proper device."""
        x = x.to(self.torch_device)
        return Variable(x, **kwargs)

    def to_device(self, model):
        """PyTorch only: send Model to proper device."""
        model.to(self.torch_device)
