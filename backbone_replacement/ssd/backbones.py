import torch.nn as nn
from torchvision.models import efficientnet_b0, shufflenet_v2_x1_0, mobilenet_v2
from torchvision.models.feature_extraction import create_feature_extractor

class SSDExtraLayers(nn.Module):
    def __init__(self, in_channels=1280):
        super().__init__()
        self.extras = nn.ModuleList([
            # Layer 0: 10x10 → 5x5
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=2, padding=1),
            # Layer 1: 5x5 → 3x3
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            # Layer 2: 3x3 → 1x1
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
        ])

    def forward(self, x):
        features = []
        for layer in self.extras:
            x = F.relu(layer(x), inplace=True)
            features.append(x)
        return features

class EfficientNetB0Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = efficientnet_b0(weights="IMAGENET1K_V1")

        # debug statement to check if imagenet weights are being loaded in 
        # print("Before modification, first conv layer weights:")
        # print(model.features[0][0].weight[0, :, :, :])  # Print first layer weights

        # EfficientNet feature layers for SSD
        return_nodes = {
        'features.3.0': 'feat1',  # 38×38 (small objects)
        'features.4.0': 'feat2',  # 19×19 (medium-small objects)
        'features.4.2': 'feat3',  # 19×19 (medium objects)
        'features.5.0': 'feat4',  # 10×10 (large objects)
        'features.6.2': 'feat5',  # 10×10 (very large objects)
        'features.7.0': 'feat6',  # 10×10 (largest receptive field)
        }

        self.backbone = create_feature_extractor(model, return_nodes=return_nodes)
    
    def forward(self, x):
        feats = self.backbone(x)
        return [
            feats['feat1'], feats['feat2'], feats['feat3'],
            feats['feat4'], feats['feat5'], feats['feat6']
        ]


class MobileNetV2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = mobilenet_v2(weights="IMAGENET1K_V1")
        return_nodes = {
            'features.4.conv': 'feat1',   # 38×38 (small objects)
            'features.7.conv': 'feat2',   # 19×19 (medium-small objects)
            'features.10.conv': 'feat3',  # 19×19 (medium objects)
            'features.14.conv': 'feat4',  # 10×10 (large objects)
            'features.17.conv': 'feat5',  # 10×10 (very large objects)
            'features.18': 'feat6',       # 10×10 (largest receptive field)
        }
        self.backbone = create_feature_extractor(model, return_nodes)
    
    def forward(self, x):
        feats = self.backbone(x)
        return [
            feats['feat1'], feats['feat2'], feats['feat3'],
            feats['feat4'], feats['feat5'], feats['feat6']
        ]


class ShuffleNetV2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = shufflenet_v2_x1_0(weights="IMAGENET1K_V1")
        
        return_nodes = {
            'stage3.0.branch2.0': 'feat1',   # 38×38 (small objects)
            'stage2.3': 'feat2',   # 38×38 (small objects)
            'stage3.6': 'feat3',   # 19×19 (medium-small objects)
            'stage3.7': 'feat4',   # 19×19 (medium objects)
            'stage4.3': 'feat5',   # 10×10 (large objects)
            'conv5': 'feat6',      # 10×10 (very large objects)
        }

        self.backbone = create_feature_extractor(model, return_nodes)
    
    def forward(self, x):
        feats = self.backbone(x)
        return [
            feats['feat1'], feats['feat2'], feats['feat3'],
            feats['feat4'], feats['feat5'], feats['feat6']
        ]


