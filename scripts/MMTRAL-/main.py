import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Full_DataSet import MultiModalDataset
import models
import utils
import argparse
import numpy as np
import pandas as pd
from transformers import RobertaTokenizer,BertTokenizer
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support
import os
log = []

parser = argparse.ArgumentParser(description='DDC_DCORAL')
parser.add_argument('--n_class', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_epoch', type=int, default=50)
parser.add_argument('--trans_loss_type', type=str, default='dann')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--decay', type=float, default=0.2)
parser.add_argument('--lamb', type=float, default=0.4)
parser.add_argument('--datafolder',type=str,default='')
parser.add_argument('--savepath',type=str,default='train_log.csv')
parser.add_argument('--savemodelname',type=str,default='best_model.pth')
args = parser.parse_args()



# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
tokenizer = RobertaTokenizer.from_pretrained("/roberta-base")


source_data_path = '../data/source.txt'
target_train_path = '../data/target.txt'
target_test_path = '../data/target.txt'
source_dataset = MultiModalDataset(source_data_path,tokenizer, transform)
target_train_dataset = MultiModalDataset(target_train_path,tokenizer, transform)
target_test_dataset = MultiModalDataset(target_test_path,tokenizer, transform)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def test(model, target_test_loader):
    model.eval()
    test_loss = utils.AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    all_preds = []
    all_targets = []
    all_feature_weights_mean = []
    feature_weights_sum = 0
    with torch.no_grad():
        for img_data, input_ids, mask, user_info, target in target_test_loader:
            img_data, input_ids, mask, user_info, target = img_data.to(DEVICE), input_ids.to(DEVICE), mask.to(DEVICE), user_info.to(DEVICE),target.to(DEVICE)
            feature_weights, s_output = model.predict(img_data, input_ids, mask,user_info)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            # Store all predictions and targets to calculate metrics later
            feature_weights_sum += feature_weights.sum(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    mean_weights = feature_weights_sum / len_target_dataset
    acc = sum([1 if i==j else 0 for i, j in zip(all_preds, all_targets)]) / len_target_dataset
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    weighted_precision = round(weighted_precision, 3)
    weighted_recall = round(weighted_recall, 3)
    weighted_f1 = round(weighted_f1, 3)

    return round(acc, 3), weighted_precision, weighted_recall, weighted_f1, mean_weights


def load_data(args):
    batch_size = 96
    if args.trans_loss_type == 'lmmd':
        source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
        target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
        target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    else:
        source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
        target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size,shuffle=True,num_workers=4)
        target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size,num_workers=4)
    return source_loader, target_train_loader, target_test_loader


def train(source_loader, target_train_loader, target_test_loader, model, optimizer,scheduler):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    best_acc = 0
    stop = 0
    for e in tqdm(range(args.n_epoch)):
        stop += 1
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.train()
        iter_source, iter_target = iter(source_loader), iter(target_train_loader)
        n_batch = min(len_source_loader, len_target_loader)
        criterion = torch.nn.CrossEntropyLoss()
        for _ in tqdm(range(n_batch)):
            source_img,source_input_ids,source_mask,source_user_info,label_source = next(iter_source)
            target_img,target_input_ids,target_mask,target_user_info,_ = next(iter_target)
            source_img,source_input_ids,source_mask,source_user_info,label_source = source_img.to(DEVICE),source_input_ids.to(DEVICE),source_mask.to(DEVICE),source_user_info.to(DEVICE),label_source.to(DEVICE)
            target_img,target_input_ids,target_mask,target_user_info = target_img.to(DEVICE),target_input_ids.to(DEVICE),target_mask.to(DEVICE),target_user_info.to(DEVICE)
            optimizer.zero_grad()
            label_source_pred = model(source_img,source_input_ids,source_mask,source_user_info,target_img,target_input_ids,target_mask,target_user_info,label_source)
            clf_loss = criterion(label_source_pred, label_source)
            loss = clf_loss
            loss.backward()
            optimizer.step()
            train_loss_clf.update(clf_loss.item())
            train_loss_total.update(loss.item())
        scheduler.step(train_loss_total.avg)
        
        acc, weighted_precision, weighted_recall, weighted_f1, mean_weights = test(model, target_test_loader)
        source_acc, source_precision, source_recall, source_f1, _ = test(model, source_loader)
        image_weight, text_weight, user_weight = mean_weights
        image_weight = image_weight.item()
        text_weight = text_weight.item()
        user_weight = user_weight.item()
        omega = model.omega.item()
        log.append([
            e,
            round(train_loss_clf.avg, 4),
            round(train_loss_transfer.avg, 4),
            round(train_loss_total.avg, 4),
            round(acc, 4),
            round(weighted_precision, 4),
            round(weighted_recall, 4),
            round(weighted_f1, 4),
            round(source_acc, 4),          
            round(source_precision, 4),     
            round(source_recall, 4),        
            round(source_f1, 4),            
            round(omega, 4),
            round(image_weight, 4),
            round(text_weight, 4),
            round(user_weight, 4)
        ])

        pd.DataFrame.from_dict(log).to_csv('./saved_results/' + args.savepath, header=[
            'Epoch', 
            'Cls_loss', 
            'Transfer_loss', 
            'Total_loss', 
            'Tar_acc', 
            'Weighted Precision', 
            'Weighted Recall', 
            'Weighted F1', 
            'Source Acc',          
            'Source Precision',   
            'Source Recall',       
            'Source F1',          
            'omega', 
            'image_weight', 
            'text_weight', 
            'user_weight'
        ])

        print(f'Epoch: [{e:2d}/{args.n_epoch}], cls_loss: {train_loss_clf.avg:.4f}, transfer_loss: {train_loss_transfer.avg:.4f}, total_Loss: {train_loss_total.avg:.4f}, Weighted Precision: {weighted_precision:.4f}, Weighted Recall: {weighted_recall:.4f}, Weighted F1: {weighted_f1:.4f}, Source Acc: {source_acc:.4f}, Source Precision: {source_precision:.4f}, Source Recall: {source_recall:.4f}, Source F1: {source_f1:.4f}, omega: {omega:.4f}, image_weight: {image_weight:.4f}, text_weight: {text_weight:.4f}, user_weight: {user_weight:.4f}')
            
if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    source_loader, target_train_loader, target_test_loader = load_data(args)
    model = models.TransferNet(args.n_class,args.trans_loss_type).to(DEVICE)
    optimizer = model.get_optimizer(args)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-7)
    train(source_loader, target_train_loader,target_test_loader, model, optimizer,scheduler)