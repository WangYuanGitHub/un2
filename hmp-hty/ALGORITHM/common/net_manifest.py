import math
import torch.nn as nn
def weights_init(m):
    def init_Linear(m, final_layer=False):
        nn.init.orthogonal_(m.weight.data)
        if final_layer:nn.init.orthogonal_(m.weight.data, gain=0.01)
        if m.bias is not None: nn.init.uniform_(m.bias.data, a=-0.02, b=0.02)
    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    initial_fn_dict = {
        'Net': None,
        'NetCentralCritic': None,
        'DataParallel':None,
        'BatchNorm1d':None,
        'Concentration':None,
        'ConcentrationHete':None,
        'Pnet':None,
        'Sequential':None,
        'DataParallel':None,
        'Tanh':None,
        'ModuleList':None,
        'ModuleDict':None,
        'MultiHeadAttention':None,
        'SimpleMLP':None,
        'SimpleAttention':None,
        'SelfAttention_Module':None,
        'ReLU':None,
        'Softmax':None,
        'DynamicNorm':None,
        'DynamicNormFix':None,
        'EXTRACT':None,
        'LinearFinal':lambda m:init_Linear(m, final_layer=True),
        'Linear':init_Linear,
        'ResLinear':None,
        'LeakyReLU':None,
        'HyperNet':None,
        'MyHyperNet':None,
        'DivTree':None,
        'E_GAT':None,
        'I_GAT':None,
        'GroupGAT':None,
        'GRUCell':None,
        'ParameterList':None,
        'HetGAT':None,
        'SingleHeadSelfAttention':None,
        'GradientReversalLayer':None,

    }

    classname = m.__class__.__name__
    assert classname in initial_fn_dict.keys(), ('how to handle the initialization of this class? ', classname)
    init_fn = initial_fn_dict[classname]
    if init_fn is None: return
    init_fn(m)

