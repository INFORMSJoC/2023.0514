from typing import List
import torch.nn as nn
import timm
import torch


class ViTBackbone(nn.Module):
    def __init__(self, vit_name, pretrained=True, **kwargs):
        super(ViTBackbone, self).__init__()
        model = timm.create_model(vit_name, pretrained=pretrained)
        self.patch_embed = model.patch_embed
        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.pos_drop = model.pos_drop
        self.blocks = model.blocks
        self.norm = model.norm
        self.pre_logits = model.pre_logits
        self.head = model.head  # no need train as features_params because not using
        self.in_features = model.num_features

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return self.pre_logits(x[:, 0])
    
    def output_num(self):
        return 768