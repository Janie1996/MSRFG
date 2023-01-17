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

def train_or_eval_model(model, loss_function, dataloader, args, optimizer=None, train=False):

    losses, preds, labels = [], [], []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        wav_fea,text_fea, label,adj= data

        wav_fea = wav_fea.cuda()
        text_fea=text_fea.cuda()
        label = label.cuda()
        for i in range(args.graph_num):
            adj[i]=adj[i].cuda()

        log1,log2,log_prob = model(wav_fea,text_fea, adj)

        loss = loss_function(log_prob.permute(0,2,1), label)+loss_function(log2.permute(0,2,1), label)+loss_function(log1.permute(0,2,1), label)


        label = label.cpu().numpy().tolist()
        pred = torch.argmax(log_prob, dim = 2).cpu().numpy().tolist()
        preds += pred
        labels += label
        losses.append(loss.item())

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

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

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)

    avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)
    avg_micro_fscore = round(f1_score(new_labels, new_preds, average='micro') * 100, 2)
    avg_macro_fscore = round(f1_score(new_labels, new_preds, average='macro') * 100, 2)
    # if(train):
    #     print('Train acc:',avg_accuracy)
    # else:
    #     print('Test acc:',avg_accuracy)
    return avg_loss, avg_accuracy, avg_fscore, avg_micro_fscore, avg_macro_fscore


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='IEMOCAP', type= str, help='dataset name, IEMOCAP or MELD')

    parser.add_argument('--gnn_layers', type=int, default=1, help='Number of gnn layers.')

    parser.add_argument('--graph_num', type=int, default=4, help='Number of sub-graphs.')

    parser.add_argument('--subGraph', type=list, default=[1,2,3,4], help='fileds of graph.')

    parser.add_argument('--emb_dim', type=int, default=1024, help='Feature size.')

    parser.add_argument('--hidden_dim', type = int, default=300)

    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate')

    parser.add_argument('--dropout', type=float, default=0.2, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch_size', type=int, default=2, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=80, metavar='E', help='number of epochs')

    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

    args = parser.parse_args()
    print(args)

    seed_everything(9413)

    train_loader, valid_loader, test_loader = get_loaders(dataset_name=args.dataset_name, batch_size=args.batch_size,subGraph=args.subGraph)
    if(args.dataset_name=='IEMOCAP'):
        n_classes=6
    elif(args.dataset_name=='MELD'):
        n_classes=7

    model = MHGNN(args, n_classes)
    model.cuda()

    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = AdamW(model.parameters() , lr=args.lr)


    all_fscore, all_acc, all_loss,all_macro_fscore,all_micro_fscore = [], [], [],[],[]


    for e in range(args.epochs):

        train_loss, train_acc,train_fscore,train_micro_fscore, train_macro_fscore = train_or_eval_model(model, loss_function, train_loader, args, optimizer, True)

        valid_loss, valid_acc, valid_fscore,valid_micro_fscore, valid_macro_fscore= train_or_eval_model(model, loss_function,valid_loader, args)

        test_loss, test_acc, test_fscore,test_micro_fscore, test_macro_fscore= train_or_eval_model(model,loss_function, test_loader, args)

        all_fscore.append([valid_fscore, test_fscore])
        all_micro_fscore.append([valid_micro_fscore, test_micro_fscore])
        all_macro_fscore.append([valid_macro_fscore, test_macro_fscore])

    #print('Test performance..')
    all_fscore = sorted(all_fscore, key=lambda x: (x[0],x[1]), reverse=True)
    all_micro_fscore = sorted(all_micro_fscore, key=lambda x: (x[0],x[1]), reverse=True)
    all_macro_fscore = sorted(all_macro_fscore, key=lambda x: (x[0],x[1]), reverse=True)


    print('Best F-Score based on test:{}'.format(max([f[1] for f in all_fscore])))
    print('Best micro-F-Score based on test:{}'.format(max([f[1] for f in all_micro_fscore])))
    print('Best macro-F-Score based on test:{}'.format(max([f[1] for f in all_macro_fscore])))


if __name__ == '__main__':
    main()

