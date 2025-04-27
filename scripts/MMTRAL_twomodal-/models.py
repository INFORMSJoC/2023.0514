import torch
import torch.nn as nn
import backbone
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

    def forward(self, image_input, text_input):
        # 计算每个专家的输出
        expert_outputs = [expert(image_input if i==0 else (text_input)) 
                          for i, expert in enumerate(self.experts)]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, output_size]

        # 为每个样本计算门控网络的权重
        gating_weights = self.gating_network(image_input, text_input)  # 门控网络使用所有输入

        # 加权求和专家的输出
        weighted_expert_outputs = expert_outputs * gating_weights.unsqueeze(-1)  # 广播门控权重
        mixed_output = weighted_expert_outputs.sum(1)  # 对所有专家的输出求和

        return mixed_output
    
##############################修改后的TransferNet##############################
class TransferNet(nn.Module):
    def __init__(self, num_class, use_bottleneck=True, bottleneck_width=256, width=1024):
        super(TransferNet, self).__init__()
        self.n_class = num_class
        self.base_network = ViTBackbone(vit_name='vit_base_patch16_224',pretrained=True, freeze_weight=True)
        self.text_feature_extractor = backbone.TextCLS()
        self.use_bottleneck = use_bottleneck

        ##定义MMoE模型需要的模块##
        self.expert_input_sizes = [self.base_network.output_num(), self.text_feature_extractor.output_num()]
        self.expert_hidden_sizes = [256, 128]
        self.expert_output_size = 256
        self.num_experts = 2
        self.mmoe = MMoE(self.expert_input_sizes, self.expert_hidden_sizes, self.expert_output_size, self.num_experts)
        concat_dim = self.base_network.output_num() + self.text_feature_extractor.output_num() + self.expert_output_size
        classifier_layer_list = [nn.Linear(concat_dim, width), nn.ReLU(), nn.Dropout(0.5), nn.Linear(width, num_class)]
        self.classifier_layer = nn.Sequential(*classifier_layer_list)

    def forward(self, img_data, ids, mask):
        img_features = self.base_network(img_data)
        text_features = self.text_feature_extractor(ids, mask)
        mixed_features = self.mmoe(img_features, text_features)
        full_features = torch.cat([img_features, text_features, mixed_features], dim=1)
        clf = self.classifier_layer(full_features)
        return clf
    
    def predict(self,img_data,ids,mask):
        img_features = self.base_network(img_data)
        text_features = self.text_feature_extractor(ids,mask)
        mixed_features = self.mmoe(img_features,text_features)
        full_features = torch.cat([img_features, text_features,mixed_features], dim=1)
        clf = self.classifier_layer(full_features)
        
        return clf
    def get_optimizer(self, args):
        params = [
            {'params': self.base_network.parameters()},
            {'params': self.text_feature_extractor.parameters()},
            {'params': self.classifier_layer.parameters(), 'lr': 10 * args.lr},
            {'params': self.mmoe.parameters(), 'lr': 10 * args.lr}
        ]
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.decay)
        return optimizer
