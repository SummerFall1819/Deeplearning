import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def gelu(x):
    """
    gelu
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        LayerNorm
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class FilterLayer(nn.Module):
    """
    滤波器层
    """
    def __init__(self, args):
        super(FilterLayer, self).__init__()
        # todo

    def forward(self, input_tensor):
        # todo
        # input_tensor.shape = [batch, seq_len, hidden]
        # 滤波器层可以参考文献[4] 的做法，仓库地址是:https://github.com/raoyongming/GFNet

        return hidden_states

class Intermediate(nn.Module):
    """
    中间层
    """
    def __init__(self, args):
        super(Intermediate, self).__init__()
        # todo
        

    def forward(self, input_tensor):
        # todo

        return hidden_states

class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.filterlayer = FilterLayer(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states):

        hidden_states = self.filterlayer(hidden_states)

        intermediate_output = self.intermediate(hidden_states)
        return intermediate_output

class Encoder(nn.Module):
    """
    根据自己搭建的Layer组成Transformer only-MLP 的Encoder
    """
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=True):
        # todo
        all_encoder_layers = []
        
        return all_encoder_layers
