import torch.nn as nn
import torch.nn.init as init


def xavier_init(model):
    if isinstance(model, nn.Conv2d):
        init.xavier_uniform_(model.weight.data)
        model.bias.data.zero_()


def kaiming_init(model):
    for key in model.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.kaiming_normal_(model.state_dict()[key], mode='fan_out')
            if 'bn' in key:
                model.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            model.state_dict()[key][...] = 0
