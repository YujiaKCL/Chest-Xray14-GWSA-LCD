import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model, is_model

from .attentions import GroupSpatialAttention, LabelCorrelationHead
from .attention_insertion import insert_attn_forward_fn
from timm.models.densenet import DenseLayer


def build_model(backbone, num_classes, pretrained=False, add_attention=False, add_correlation=False):
    model = ChestClassifier(backbone, num_classes, pretrained, add_attention, add_correlation)
    return model


class ChestClassifier(nn.Module):
    def __init__(self, 
                backbone, 
                num_classes, 
                pretrained=False,
                add_attention=False,
                add_correlation=False):
        super().__init__()
        self.num_classes = num_classes
        self.add_attention = add_attention
        self.add_correlation = add_correlation

        if is_model(backbone):
            if backbone.startswith('densenet'):
                model = create_model(backbone, pretrained=pretrained, memory_efficient=True)
            else:
                model = create_model(backbone, pretrained=pretrained)
            
            self.num_features = model.num_features
            if list(model.children())[:-2].__len__() > 1:
                self.features = nn.Sequential(*list(model.children())[:-2])
            else:
                self.features = list(model.children())[0]

            self.global_pool = model.global_pool
            if add_attention:
                blocks = [3, 4]
                self.insert_attention_module(blocks=blocks)
            
            if add_correlation:
                print('Add label correlation module')
                self.fc = LabelCorrelationHead(model.num_features, num_classes, projection_depth=1)
            else:
                self.fc = nn.Linear(model.num_features, num_classes)
                
            del model            
        else:
            raise NotImplementedError(f'Backbone {backbone} is not available.') 


    def insert_attention_module(self, blocks):
        if isinstance(blocks, int):
            blocks = [blocks]

        target_blocks = [f'denseblock{b}'for b in blocks]
        
        inserted = False
        for name, module in self.features.named_modules():
            if isinstance(module, DenseLayer) and name.split('.')[0] in target_blocks:
                attn = GroupSpatialAttention(module.norm1.num_features, groups=module.norm1.num_features//32)
                insert_attn_forward_fn(module, attn, insert_after='bn1')
                inserted = True

        if inserted: print('Add attention to blocks: ', blocks)


    def forward(self, x):
        features = self.features(x)
        features = self.global_pool(features)
        logits = self.fc(features)
        return logits