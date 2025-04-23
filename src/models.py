import torch
import torch.nn as nn
import backbone
from coral import CORAL
from mmd import MMDLoss
from lmmd import LMMDLoss
from adv import AdversarialLoss
from daan_loss import *
import torch.nn.init as init
from vit import ViTBackbone


class ExpertNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(ExpertNetwork, self).__init__()
        layers = []
        for i in range(len(hidden_sizes)):
            layers.append(nn.Linear(input_size if i == 0 else hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_sizes, expert_count):
        super(GatingNetwork, self).__init__()
        total_input_size = sum(input_sizes)
        self.gate = nn.Sequential(
            nn.Linear(total_input_size, expert_count),
            nn.Softmax(dim=1)
        )

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        return self.gate(x)

class MMoE(nn.Module):
    def __init__(self, input_sizes, hidden_sizes, output_size, num_experts):
        super(MMoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([ExpertNetwork(input_size, hidden_sizes, output_size) for input_size in input_sizes])
        self.gating_network = GatingNetwork(input_sizes, num_experts)

    def forward(self, image_input, text_input, user_info_input):
        expert_outputs = [expert(image_input if i==0 else (text_input if i==1 else user_info_input)) 
                          for i, expert in enumerate(self.experts)]
        expert_outputs = torch.stack(expert_outputs, dim=1) 
        gating_weights = self.gating_network(image_input, text_input, user_info_input)  
        weighted_expert_outputs = expert_outputs * gating_weights.unsqueeze(-1)  
        mixed_output = weighted_expert_outputs.sum(1)  
        return gating_weights,mixed_output

class TransferNet(nn.Module):
    def __init__(self, num_class,trans_loss_type='dann',use_bottleneck=True, bottleneck_width=256, width=1024,user_len=7):
        super(TransferNet, self).__init__()
        self.user_info_lenth = user_len
        self.user_hidden_length = 64
        self.n_class = num_class
        self.trans_loss_type = trans_loss_type.lower()
        self.base_network = ViTBackbone(vit_name='vit_base_patch16_224',pretrained=True, freeze_weight=True)
        # self.base_network = backbone.ResNet101Fc()
        for param in self.base_network.parameters():
            param.requires_grad = False
        self.text_feature_extractor = backbone.TextCLS()
        self.user_info_extractor = nn.Sequential(nn.Linear(self.user_info_lenth, self.user_hidden_length),
                                                 nn.BatchNorm1d(64),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.5))
        self.use_bottleneck = use_bottleneck
        ##定义MMoE模型需要的模块##
        self.expert_input_sizes = [self.base_network.output_num(),self.text_feature_extractor.output_num(),self.user_hidden_length]
        self.expert_hidden_sizes = [256,128,128]
        self.expert_output_size = 256
        self.num_experts = 3
        self.mmoe = MMoE(self.expert_input_sizes,self.expert_hidden_sizes,self.expert_output_size,self.num_experts)
        self.img_adv_layer = nn.Sequential(nn.Linear(self.base_network.output_num(),bottleneck_width),
                                           nn.BatchNorm1d(bottleneck_width), 
                                           nn.ReLU(),
                                           nn.Dropout(0.5))
        self.text_adv_layer = nn.Sequential(nn.Linear(self.text_feature_extractor.output_num(),bottleneck_width),
                                           nn.BatchNorm1d(bottleneck_width), 
                                           nn.ReLU(),
                                           nn.Dropout(0.5))
        
        self.user_adv_layer = nn.Sequential(nn.Linear(self.user_hidden_length,bottleneck_width),
                                            nn.BatchNorm1d(bottleneck_width), 
                                            nn.ReLU(),
                                            nn.Dropout(0.5))
        concat_dim = self.base_network.output_num() + self.text_feature_extractor.output_num() + self.user_hidden_length +  self.expert_output_size
        bottleneck_list = [nn.Linear(concat_dim, bottleneck_width), 
                           nn.BatchNorm1d(bottleneck_width), 
                           nn.ReLU(),
                           nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        
        classifier_layer_list = [nn.Linear(concat_dim, width), nn.ReLU(), nn.Dropout(0.5),nn.Linear(width, num_class)]
        self.classifier_layer = nn.Sequential(*classifier_layer_list)
        
        self.img_adv_layer[0].weight.data.normal_(0, 0.005)
        self.img_adv_layer[0].bias.data.fill_(0.1)
        self.text_adv_layer[0].weight.data.normal_(0, 0.005)
        self.text_adv_layer[0].bias.data.fill_(0.1)
        self.user_adv_layer[0].weight.data.normal_(0, 0.005)
        self.user_adv_layer[0].bias.data.fill_(0.1)
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for i in range(2):
            self.classifier_layer[i * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[i * 3].bias.data.fill_(0.0)
        if self.trans_loss_type == 'dann':
            self.adv = AdversarialLoss()
        self.omega = 1

    def forward(self, source_img,source_ids,source_mask,source_user_info,target_img,target_ids,target_mask,target_user_info,source_label):
        source_img = self.base_network(source_img)
        target_img = self.base_network(target_img)
        source_text = self.text_feature_extractor(source_ids,source_mask)
        target_text = self.text_feature_extractor(target_ids,target_mask)
        source_user = self.user_info_extractor(source_user_info)
        target_user = self.user_info_extractor(target_user_info)
        # 定义混合特征
        _, mixed_source = self.mmoe(source_img,source_text,source_user)
        _, mixed_target = self.mmoe(target_img,target_text,target_user)
        
        
        source_concat = torch.cat([source_img,source_text,source_user,mixed_source],dim=1)
        target_concat = torch.cat([target_img,target_text,target_user,mixed_target],dim=1)
        
        
        # classification
        source_clf = self.classifier_layer(source_concat)
        target_clf = self.classifier_layer(target_concat)
        
        
        if self.use_bottleneck:
            source_concat = self.bottleneck_layer(source_concat)
            target_concat = self.bottleneck_layer(target_concat)
        
        transfer_loss = self.cal_transferloss(source_label,
                                         target_clf,
                                         source_img,
                                         target_img,
                                         source_text,
                                         target_text,
                                         source_user,
                                         target_user,
                                         source_concat,
                                         target_concat)
        return source_clf, transfer_loss

    def predict(self,img_data,ids,mask,user_info):
        img_features = self.base_network(img_data)
        text_features = self.text_feature_extractor(ids,mask)
        user_features = self.user_info_extractor(user_info)
        gating_weights, mixed_features = self.mmoe(img_features,text_features,user_features)
        # Concatenate original and attention-weighted features
        full_features = torch.cat([img_features, text_features,user_features,mixed_features], dim=1)
        clf = self.classifier_layer(full_features)
        return gating_weights, clf
    
    def cal_transferloss(self,source_label,
                         target_clf,
                         source_img,
                         target_img,
                         source_text,
                         target_text,
                         source_user,
                         target_user,
                         source_concat,
                         target_concat):
        kwargs = {}
        kwargs['source_label'] = source_label
        kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        source_img_adv = self.img_adv_layer(source_img)
        target_img_adv = self.img_adv_layer(target_img)
        source_text_adv = self.text_adv_layer(source_text)
        target_text_adv = self.text_adv_layer(target_text)
        source_user_adv = self.user_adv_layer(source_user)
        target_user_adv = self.user_adv_layer(target_user)
        
        img_transfer_loss = self.adapt_loss(source_img_adv,target_img_adv,**kwargs)
        text_transfer_loss = self.adapt_loss(source_text_adv,target_text_adv,**kwargs)
        user_transfer_loss = self.adapt_loss(source_user_adv,target_user_adv,**kwargs)
        full_transfer_loss = self.adapt_loss(source_concat,target_concat,**kwargs)
        
        DA_full_feature = 2 * (1-2 * full_transfer_loss)
        DA_img  = 2 * (1-2 * img_transfer_loss)
        DA_text = 2 * (1-2 * text_transfer_loss)
        DA_user = 2 * (1-2 * user_transfer_loss)
        self.omega = DA_full_feature/(DA_full_feature+(DA_img + DA_text + DA_user)/3)
        transfer_loss = (1 - self.omega) * full_transfer_loss + self.omega * (img_transfer_loss + 
                                                                              text_transfer_loss+
                                                                              user_transfer_loss)
        return transfer_loss
        
    def adapt_loss(self, X, Y, **kwargs):
        if self.trans_loss_type == 'mmd':
            loss = MMDLoss()(X, Y)
        elif self.trans_loss_type == 'coral':
            loss = CORAL(X, Y)
        elif self.trans_loss_type == 'lmmd':
            loss = LMMDLoss(self.n_class)(X, Y, kwargs['source_label'], kwargs['target_logits'])
        elif self.trans_loss_type == 'dann':
            loss = self.adv(X, Y)
        else:
            loss = 0
        return loss

    def get_optimizer(self, args):
        params = [
            {'params': self.base_network.parameters()},
            {'params': self.text_feature_extractor.parameters()},
            {'params': self.bottleneck_layer.parameters(), 'lr': 10 * args.lr},
            {'params': self.img_adv_layer.parameters(), 'lr': 10 * args.lr},
            {'params': self.text_adv_layer.parameters(), 'lr': 10 * args.lr},
            {'params': self.user_adv_layer.parameters(), 'lr': 10 * args.lr},
            {'params': self.classifier_layer.parameters(), 'lr': 10 * args.lr},
            {'params': self.user_info_extractor.parameters(), 'lr': 10 * args.lr},
            {'params': self.mmoe.parameters(), 'lr': 10 * args.lr}
        ]
        if self.trans_loss_type == 'dann':
            params.append({'params': self.adv.domain_classifier.parameters(), 'lr': 10 * args.lr})
        optimizer = torch.optim.SGD(params, lr=args.lr,weight_decay=args.decay)
        return optimizer



