import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Full_DataSet import MultiModalDataset
import models
import utils
import argparse
import numpy as np
import pandas as pd
from transformers import BertTokenizer,RobertaTokenizer
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support,classification_report
import csv
import os 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("/home/pkuzhangcc/workspace_zcc/Weibo/advanced_MMDANN/roberta-base")

def test(model, test_loader):
    model.eval()
    test_loss = utils.AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(test_loader.dataset)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for img_data, input_ids, mask,target in test_loader:
            img_data, input_ids, mask,target = img_data.to(DEVICE), input_ids.to(DEVICE), mask.to(DEVICE),target.to(DEVICE)
            s_output = model.predict(img_data, input_ids, mask)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            
            # Store all predictions and targets to calculate metrics later
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    acc = 100. * sum([1 if i==j else 0 for i, j in zip(all_preds, all_targets)]) / len_target_dataset
    
    # precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average=None)
    # weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')

        # 生成分类报告并提取指标
    report = classification_report(all_targets, all_preds, output_dict=True)
    accuracy = report['accuracy']
    weighted_avg_precision = report['weighted avg']['precision']
    weighted_avg_recall = report['weighted avg']['recall']
    weighted_avg_f1 = report['weighted avg']['f1-score']

    # 检查 CSV 文件是否存在，如果不存在则写入表头
    csv_file = 'evaluation_results.csv'
    if not os.path.isfile(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Accuracy', 'Weighted Avg Precision', 'Weighted Avg Recall', 'Weighted Avg F1-Score'])

    # 将结果写入 CSV 文件
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([accuracy, weighted_avg_precision, weighted_avg_recall, weighted_avg_f1])
    # return acc, precision, recall, f1, weighted_precision, weighted_recall, weighted_f1


if __name__ == "__main__":
    hours_lt = [0, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    model = models.TransferNet(2).to(DEVICE)
    model.load_state_dict(torch.load('./saved_models/best_model.pth'))
    for hour in hours_lt:
        data_path = f'/home/pkuzhangcc/workspace_zcc/Weibo/split_data/split1/df2_hours_{hour}.csv'
        print(data_path)
        test_set = MultiModalDataset(data_path,tokenizer, transform)
        batch_size = 60
        test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False)
        test(model,test_loader)





