# -*- coding: UTF-8 -*-
"""
@file:main.py
@author: Wei Jie
@date: 2022/9/22
@description:
"""
import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from model import MHGNN
from single_model import MHGNN_wav,MHGNN_text,MHGNN_wotrans
from dataloader import get_loaders
from transformers import AdamW
import copy
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_everything(2293)
def eval_model(model, dataloader, args):

    losses, preds, labels = [], [], []

    model.eval()

    for data in dataloader:

        wav_fea,text_fea, label,adj= data

        wav_fea = wav_fea.cuda()
        text_fea=text_fea.cuda()
        label = label.cuda()
        for i in range(args.graph_num):
            adj[i]=adj[i].cuda()

        log1,log2,log_prob = model(wav_fea,text_fea, adj)

        label = label.cpu().numpy().tolist()
        pred = torch.argmax(log_prob, dim = 2).cpu().numpy().tolist()
        preds += pred
        labels += label


    if preds != []:
        new_preds = []
        new_labels = []
        for i,label in enumerate(labels):
            for j,l in enumerate(label):
                if l != -1:
                    new_labels.append(l)
                    new_preds.append(preds[i][j])
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []


    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)

    avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)
    avg_micro_fscore = round(f1_score(new_labels, new_preds, average='micro') * 100, 2)
    avg_macro_fscore = round(f1_score(new_labels, new_preds, average='macro') * 100, 2)

    return avg_accuracy, avg_fscore, avg_micro_fscore, avg_macro_fscore


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='MELD', type= str, help='dataset name, IEMOCAP or MELD')

    parser.add_argument('--gnn_layers', type=int, default=1, help='Number of gnn layers.')

    parser.add_argument('--graph_num', type=int, default=3, help='Number of sub-graphs.')

    parser.add_argument('--subGraph', type=list, default=[1,2,3], help='fileds of graph.')

    parser.add_argument('--emb_dim', type=int, default=1024, help='Feature size.')

    parser.add_argument('--hidden_dim', type = int, default=300)

    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate')

    parser.add_argument('--dropout', type=float, default=0.2, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch_size', type=int, default=64, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=80, metavar='E', help='number of epochs')

    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

    args = parser.parse_args()
    print(args)

    _, _, test_loader = get_loaders(dataset_name=args.dataset_name, batch_size=args.batch_size,subGraph=args.subGraph)
    if(args.dataset_name=='IEMOCAP'):
        n_classes=6
    elif(args.dataset_name=='MELD'):
        n_classes=7

    model = MHGNN(args, n_classes)
    model.cuda()
    checkpoints=torch.load('checkpoints/MELD.pth')
    model.load_state_dict(checkpoints)


    test_acc, test_fscore,test_micro_fscore, test_macro_fscore= eval_model(model, test_loader, args)

    print('Best F-Score based on test:{}'.format(test_fscore))
    print('Best micro-F-Score based on test:{}'.format(test_micro_fscore))
    print('Best macro-F-Score based on test:{}'.format(test_macro_fscore))

if __name__ == '__main__':
    main()


