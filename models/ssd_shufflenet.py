import os
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from layers.l2_norm import L2Norm
from utils.init import xavier_init
from layers.detection import Detect
from backbone.backbones import ShuffleNetV2Backbone

# helper that ignores the marker string 'S'
def get_extras(config, in_channels, batch_norm=False):
    layers = []
    flag = False  # alternates between (1x1) and (3x3) kernels

    for i, out_channels in enumerate(config):
        if out_channels == 'S' or isinstance(out_channels, tuple):
            stride = 2  # Default stride
            if isinstance(out_channels, tuple):
                stride = out_channels[1]  # use custom stride if provided
            
            if i + 1 < len(config):  # check if next channel exists
                layers.append(nn.Conv2d(in_channels=in_channels,
                                        out_channels=config[i + 1],
                                        kernel_size=(1, 3)[flag],
                                        stride=stride,
                                        padding=1))
            flag = not flag
        else:
            # regular (1x1) or (3x3) conv layer
            layers.append(nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=(1, 3)[flag]))
            flag = not flag
            in_channels = out_channels  # update input channels for next layer

    return layers

def multibox(mbox_config, backbone_channels, extra_channels, class_count):
    class_layers = []
    loc_layers = []

    # process backbone feature maps
    for k, in_channels in enumerate(backbone_channels):
        class_layers.append(nn.Conv2d(in_channels, mbox_config[k] * class_count,
                                      kernel_size=3, padding=1))
        loc_layers.append(nn.Conv2d(in_channels, mbox_config[k] * 4,
                                    kernel_size=3, padding=1))

    # process extra feature maps
    for k, in_channels in enumerate(extra_channels):
        class_layers.append(nn.Conv2d(in_channels, mbox_config[len(backbone_channels) + k] * class_count,
                                      kernel_size=3, padding=1))
        loc_layers.append(nn.Conv2d(in_channels, mbox_config[len(backbone_channels) + k] * 4,
                                    kernel_size=3, padding=1))

    return [class_layers, loc_layers]

class SSDShuffleNet(nn.Module):
    """
    SSD architecture with ShuffleNetV2 backbone.
    Assumes the backbone returns 6 feature maps with channels with channels 116, 116, 232, 232, 464, 1024.
    """
    def __init__(self, 
                 mode, 
                 backbone, 
                 extras, 
                 head, 
                 anchors, 
                 class_count):
        super(SSDShuffleNet, self).__init__()

        self.mode = mode
        self.backbone = backbone 
        self.L2Norm = L2Norm(116, 20)
        self.extras = nn.ModuleList(extras)
        self.class_head = nn.ModuleList(head[0])
        self.loc_head = nn.ModuleList(head[1])
        self.anchors = anchors
        self.class_count = class_count

        if mode in ['test', 'val']:
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect.apply

    def forward(self, x):
        sources = []
        class_preds = []
        loc_preds = []

        # get backbone features
        feats = self.backbone(x)  # [feat1, feat2, feat3, feat4, feat5, feat6]

        #  DEBUG: Print feature map shapes from EfficientNet backbone
        # for i, feature in enumerate(feats):
        #     print(f"Feature Map {i} Shape: {feature.shape}")  # <-- Debug print

        # check if all backbone features are used as sources
        sources.append(self.L2Norm(feats[0]))  # apply L2Norm to feat1
        # print("added l2norm to first feature map!")
        sources.extend(feats[1:])              # append the rest as-is

        # apply extra layers
        for i, layer in enumerate(self.extras):
            x_extra = F.relu(layer(x_extra), inplace=True)
            if i % 2 == 1:
                sources.append(x_extra)
        
        #  Debug: Print final feature map shapes
        # for i, src in enumerate(sources):
        #     print(f"Final Feature Map {i} Shape: {src.shape}")  # Debug print

        # apply multibox head on each source
        for (src, c, l) in zip(sources, self.class_head, self.loc_head):
            class_preds.append(c(src).permute(0, 2, 3, 1).contiguous())
            loc_preds.append(l(src).permute(0, 2, 3, 1).contiguous())

        batch_size = x.shape[0]
        class_preds = torch.cat([pred.view(batch_size, -1) for pred in class_preds], 1)
        loc_preds = torch.cat([pred.view(batch_size, -1) for pred in loc_preds], 1)

        class_preds = class_preds.view(batch_size, -1, self.class_count)
        loc_preds = loc_preds.view(batch_size, -1, 4)

        if self.mode in ['test', 'val']:
            output = self.detect(self.class_count,
                                 self.softmax(class_preds),
                                 loc_preds,
                                 self.anchors)
        else:
            output = (class_preds, loc_preds)
        return output

    def init_weights(self, model_save_path, basenet):
        if basenet:
            weights_path = osp.join(model_save_path, basenet)
            print(f"Using ShuffleNetV2 weights! ({weights_path})")

            shufflenet_weights = torch.load(weights_path)  # load pre-trained weights
            self.backbone.backbone.load_state_dict(shufflenet_weights, strict=False)

        else:
            self.backbone.apply(fn=xavier_init)
        self.extras.apply(fn=xavier_init)
        self.class_head.apply(fn=xavier_init)
        self.loc_head.apply(fn=xavier_init)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext in ['.pkl', '.pth']:
            print('Loading weights...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry, only .pth and .pkl files supported.')

def build_SSDShuffleNet(mode, new_size, anchors, class_count):
    backbone = ShuffleNetV2Backbone()


    # extras_config = [256, 'S', 256, 'S', 128]
    # ^ downsamples last input channel from feature map (320) to 256 -> 'S' downsamples to 5x5 -> another 1x1 to match 128 -> downsample again -> 128 final
    # extras = get_extras(extras_config, in_channels=320)  


    backbone_channels = [116, 116, 232, 232, 464, 1024]


    # extra_channels = [256, 128]
    mbox_config = [6, 6, 6, 6, 4, 4]  # One per feature map; should match anchor config last line e.g [2,3] = 6 ; [2] = 4
    head = multibox(mbox_config=mbox_config, backbone_channels=backbone_channels, extra_channels=[], class_count=class_count)
    return SSDShuffleNet(mode, backbone, [], head, anchors, class_count)


# removed extras and just used all 6 feature maps solely from efficientnet
