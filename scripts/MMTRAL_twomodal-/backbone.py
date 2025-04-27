import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torch.autograd import Variable
from transformers import RobertaModel,BertModel
from torchvision.models import resnet50,resnet101
EMBEDDING_DIM = 768
NUM_FILTERS = 256
#NUM_CLASSES = 100
FILTER_SIZES = [2, 3, 4]


class ResNet50Fc(nn.Module):
    def __init__(self):
        super(ResNet50Fc, self).__init__()
        model_resnet50 = resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__in_features = model_resnet50.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("/home/pkuzhangcc/workspace_zcc/DANN/DANN_2/bert-base-uncased")
        # 冻结BERT的参数
        for param in self.bert.parameters():
            param.requires_grad = False
        self.conv1 = nn.Conv2d(1, NUM_FILTERS, (2, EMBEDDING_DIM))
        self.conv2 = nn.Conv2d(1, NUM_FILTERS, (3, EMBEDDING_DIM))
        self.conv3 = nn.Conv2d(1, NUM_FILTERS, (4, EMBEDDING_DIM))
        self.linear = nn.Linear(NUM_FILTERS * 3, NUM_FILTERS * 3)

    def conv_and_pool(self, conv, input):
        out = conv(input)
        out = F.relu(out)
        return F.max_pool2d(out, (out.shape[2], out.shape[3])).squeeze()

    def forward(self, input, mask):
        input = input.squeeze(1)
        mask = mask.squeeze(1)
        out = self.bert(input, mask)[0].unsqueeze(1)
        #print(out.shape)
        #import pdb;pdb.set_trace()
        out1 = self.conv_and_pool(self.conv1, out)
        out2 = self.conv_and_pool(self.conv2, out)
        out3 = self.conv_and_pool(self.conv3, out)
        out = torch.cat([out1, out2, out3], dim=1)
        return self.linear(out)
    
    def output_num(self):
        return NUM_FILTERS*3
    

class TextCLS(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained RoBERTa model
        self.roberta = RobertaModel.from_pretrained("/home/pkuzhangcc/workspace_zcc/Weibo/advanced_MMDANN/roberta-base")
        # for param in self.roberta.parameters():
        #     param.requires_grad = False

    def forward(self, input, mask):
        # Pass the input through the RoBERTa model
        out = self.roberta(input, attention_mask=mask).last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]
        cls_embedding = out[:, 0, :]  # Shape: [batch_size, hidden_size]
        return cls_embedding 

    def output_num(self):
        return self.roberta.config.hidden_size
