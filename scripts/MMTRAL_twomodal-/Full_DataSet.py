from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pandas as pd
from PIL import Image
import os
import torch
from transformers import BertModel

class MultiModalDataset(Dataset):
    def __init__(self,dataset_path,tokenizer, transform=None):
        self.data = pd.read_csv(dataset_path)
        self.image_paths = [os.path.join('../../data/images', image_id)+'.jpg' for image_id in self.data['imageId(s)']]
        self.texts = self.data['tweetText']
        self.labels = self.data['label']
        self.tokenizer = tokenizer
        self.transform = transform
        #self.bert = BertModel.from_pretrained("bert-base-uncased")
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 图片处理
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 文本处理
        text = self.texts[idx]
        text_encodings = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        input_ids = text_encodings["input_ids"].squeeze(0)  # 去除第0维度
        attention_mask = text_encodings["attention_mask"].squeeze(0)  # 去除第0维度
        label = int(self.labels[idx])
        return image, input_ids.clone().detach(), attention_mask.clone().detach(), torch.tensor(label)
