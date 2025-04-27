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
from sklearn.metrics import precision_recall_fscore_support
import os
log = []

parser = argparse.ArgumentParser(description='DDC_DCORAL')
parser.add_argument('--n_class', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--decay', type=float, default=0.2)
parser.add_argument('--lamb', type=float, default=0.4)
parser.add_argument('--datafolder',type=str,default='')
parser.add_argument('--savepath',type=str,default='train_log2.csv')
parser.add_argument('--savemodelname',type=str,default='best_model.pth')
args = parser.parse_args()


# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
tokenizer = RobertaTokenizer.from_pretrained("/roberta-base")
##########################数据定义###########################
source_data_path = '../data/source.txt'
# target_train_path = '../data/target.txt'
target_test_path = '../data/target.txt'

train_dataset = MultiModalDataset(source_data_path,tokenizer, transform)
test_dataset  = MultiModalDataset(target_test_path,tokenizer, transform)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average=None)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    return acc, precision, recall, f1, weighted_precision, weighted_recall, weighted_f1

def load_data(args):
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader , test_loader

def train(train_loader, test_loader, model, optimizer, scheduler, args=args, DEVICE=DEVICE):
    best_acc = 0
    log = []  # 用于记录训练日志
    for e in range(args.n_epoch):
        train_loss_clf = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.train()
        iter_train = iter(train_loader)
        len_train_loader = len(train_loader)
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(len_train_loader):
            img, input_ids, mask, label = next(iter_train)
            img, input_ids, mask, label = img.to(DEVICE), input_ids.to(DEVICE), mask.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            label_pred = model(img, input_ids, mask)
            clf_loss = criterion(label_pred, label)
            clf_loss.backward()
            optimizer.step()
            train_loss_clf.update(clf_loss.item())
            train_loss_total.update(clf_loss.item())
        scheduler.step(train_loss_total.avg)
        # 测试模型性能
        acc, precision, recall, f1, weighted_precision, weighted_recall, weighted_f1 = test(model, test_loader)
        log.append([e, train_loss_clf.avg, train_loss_total.avg, acc, precision, recall, f1, weighted_precision, weighted_recall, weighted_f1])
        pd.DataFrame(log).to_csv(args.savepath, header=['Epoch', 'Cls_loss', 'Total_loss', 'Accuracy', 'Precision', 'Recall', 'F1', 'Weighted Precision', 'Weighted Recall', 'Weighted F1'])
        print(f'Epoch: [{e:2d}/{args.n_epoch}], Cls_loss: {train_loss_clf.avg:.4f}, Total_loss: {train_loss_total.avg:.4f}, Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}, Weighted Precision: {weighted_precision:.4f}, Weighted Recall: {weighted_recall:.4f}, Weighted F1: {weighted_f1:.4f}')
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join('./saved_models', args.savemodelname))
            print('Best model saved')
        print('Best result so far: {:.4f}'.format(best_acc))
        
if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train_loader,test_loader = load_data(args)
    model = models.TransferNet(args.n_class).to(DEVICE)
    optimizer = model.get_optimizer(args)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-7)
    train(train_loader, test_loader, model, optimizer,scheduler)