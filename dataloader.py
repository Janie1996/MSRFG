# -*- coding: UTF-8 -*-
"""
@file:dataloader.py
@author: Wei Jie
@date: 2022/9/21
@description:
"""
import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import warnings
warnings.filterwarnings('ignore')

import pickle,json
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random

def get_sampler(trainset):
    size = len(trainset)
    idx = list(range(size))
    return SubsetRandomSampler(idx)

emotion_labels={
    'IEMOCAP':['exc','neu','fru','sad','hap','ang'],
    'MELD':['neutral','surprise','fear','sadness','joy','disgust','anger']
}
def get_loaders(dataset_name = 'IEMOCAP', batch_size=32,subGraph=[1]):

    dict=pickle.load(open('data/%s_dict.pkl' % (dataset_name), 'rb'))
    speakers=dict['speaker']
    emotions=dict['emotion']

    trainset = MyDataset(dataset_name, 'train',subGraph, speakers, emotions)
    train_sampler = get_sampler(trainset)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,)

    devset = MyDataset(dataset_name, 'dev',subGraph, speakers, emotions)
    testset = MyDataset(dataset_name, 'test',subGraph, speakers, emotions)
    valid_loader = DataLoader(devset,
                              batch_size=batch_size,
                              collate_fn=devset.collate_fn,
                              )
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,)

    return train_loader, valid_loader, test_loader

class MyDataset(Dataset):

    def __init__(self, dataset_name = 'IEMOCAP', split = 'train',subGraph=[1], speakers=None, emotions=None):
        self.speakers = speakers
        self.emotions = emotions
        self.subGraph = subGraph
        self.data = self.loadConversations(dataset_name, split)


    def loadConversations(self, dataset_name, split):
        with open('data/wj_%s_%s'%(dataset_name,split), encoding='utf-8') as f:
            raw_data = json.load(f)

        #emotion_label=emotion_labels[dataset_name]
        dialogs = []
        sum=0

        for d in raw_data:

            sum+=len(d)
            text = []
            labels = []
            speakers = []
            wav = []
            for i,u in enumerate(d):
                text.append(u['cls'])

                labels.append(self.emotions[u['label']] if 'label' in u.keys() else -1)
                speakers.append(self.speakers[u['speaker']])
                wav.append(u['wav'])

            dialogs.append({
                'text': text,
                'labels': labels,
                'speakers':speakers,
                'wav': wav
            })
        random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):

        return torch.FloatTensor(self.data[index]['text']), \
               torch.LongTensor(self.data[index]['labels']), \
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               torch.FloatTensor(self.data[index]['wav'])

    def __len__(self):
        return len(self.data)


    def graph_construct(self,spk_list,max_dialog_len,max_hip=1):
        edge_index = []
        for conv in spk_list:
            a=torch.zeros(max_dialog_len,max_dialog_len)
            speakers=[]
            sp=np.unique(conv)
            for i in range(len(conv)):
                react_spk=[]
                sp_dict={sp[i]:0 for i in range(len(sp))}#{sp[0]:0,sp[1]:0}
                s_i = conv[i]
                j = i - 1   # conv[j]表示s_i前面的speaker
                while j >= 0:
                    a[i,j] = 1
                    #edge_idx.append([j, i])  # i,j是下标   i>j
                    if(conv[j] not in react_spk):
                        now=conv[j]
                        sp_dict[now]+=1
                        if(sp_dict[now]==max_hip):
                            react_spk.append(conv[j])
                    if(len(react_spk)==len(speakers)):
                        break
                    j -= 1
                if(s_i not in speakers):
                    speakers.append(s_i)
            edge_index.append(a)
        return torch.stack(edge_index)

    def collate_fn(self, data):
        max_dialog_len = max([d[3] for d in data])
        text_fea = pad_sequence([d[0] for d in data], batch_first = True)
        wav_fea =  pad_sequence([d[4] for d in data], batch_first = True)
        labels = pad_sequence([d[1] for d in data], batch_first = True, padding_value = -1) # (B, N )
        adj=[]
        for i in self.subGraph:
            adj.append(self.graph_construct([d[2] for d in data], max_dialog_len,i))
        return wav_fea,text_fea, labels,adj


if __name__=='__main__':
    train,dev,test=get_loaders('IEMOCAP',batch_size=1,subGraph=[1,2])
    for i in train:
        print(i)

