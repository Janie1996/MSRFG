# -*- coding: UTF-8 -*-
"""
@file:model.py
@author: Wei Jie
@date: 2022/9/22
@description:
"""
import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import DenseGraphConv


class MHGNN(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args

        self.dropout = nn.Dropout(args.dropout)
        self.gnn_layers = args.gnn_layers
        self.graphs = args.graph_num

        self.wav_fc = nn.Linear(args.emb_dim, args.hidden_dim)

        rnn=[]
        for i in range(args.gnn_layers):
            rnn+=[nn.TransformerEncoderLayer(d_model=args.hidden_dim,nhead=4,batch_first=True)]
        self.wav_rnn = nn.ModuleList(rnn)

        wav_graph=[]
        for j in range(self.graphs):
            graph=[]
            for i in range(args.gnn_layers):
                graph+=[DenseGraphConv(args.hidden_dim,args.hidden_dim)]
            wav_graph+=nn.ModuleList(graph)
        self.wav_graph=nn.ModuleList(wav_graph)
        #
        # graph=[]
        # for i in range(args.gnn_layers):
        #     graph+=[DenseGraphConv(args.hidden_dim,args.hidden_dim)]
        # self.wav_graph1 = nn.ModuleList(graph)


        self.text_fc = nn.Linear(args.emb_dim, args.hidden_dim)

        rnn=[]
        for i in range(args.gnn_layers):
            rnn+=[nn.TransformerEncoderLayer(d_model=args.hidden_dim,nhead=4,batch_first=True)]
        self.text_rnn = nn.ModuleList(rnn)

        text_graph=[]
        for j in range(self.graphs):
            graph=[]
            for i in range(args.gnn_layers):
                graph+=[DenseGraphConv(args.hidden_dim,args.hidden_dim)]
            text_graph+=nn.ModuleList(graph)
        self.text_graph=nn.ModuleList(text_graph)

        graph=[]
        for i in range(args.gnn_layers):
            graph+=[DenseGraphConv(args.hidden_dim,args.hidden_dim)]
        self.text_graph1 = nn.ModuleList(graph)


        transf_layer=nn.TransformerEncoderLayer(d_model=args.hidden_dim*2*self.graphs,nhead=4,batch_first=True)
        self.fusion_trans=nn.TransformerEncoder(transf_layer,2)

        self.outFC1= nn.Linear(args.hidden_dim*2,num_class)
        self.outFC2= nn.Linear(args.hidden_dim,num_class)
        self.fusion_fc=nn.Linear(args.hidden_dim*2*self.graphs,num_class)

    def forward(self, wav_fea,text_fea, adj):

        wav_fea = F.relu(self.wav_fc(wav_fea))   # 1024-->300
        text_fea = F.relu(self.text_fc(text_fea))

        for i in range(self.args.gnn_layers):

            wav_fea=self.wav_rnn[i](wav_fea)
            text_fea=self.text_rnn[i](text_fea)

            wav={}
            wav_subs=torch.zeros((wav_fea.shape)).cuda()
            for j in range(self.graphs):
                wav[j]=self.wav_graph[i*self.graphs+j](wav_fea,adj[j])
                wav_subs+=wav[j]
            wav_fea=wav_subs

            text={}
            text_subs=torch.zeros((text_fea.shape)).cuda()
            for j in range(self.graphs):
                text[j]=self.text_graph[i*self.graphs+j](text_fea,adj[j])
                text_subs+=text[j]
            text_fea=text_subs

        for i in range(self.graphs):
            if(i==0):
                fuse_fea=torch.cat((text[i],wav[i]),dim=2)
            else:
                fuse_fea=torch.cat((fuse_fea,text[i]),dim=2)
                fuse_fea=torch.cat((fuse_fea,wav[i]),dim=2)
        text=self.outFC2(text_fea)
        log=self.fusion_trans(fuse_fea)
        log=self.fusion_fc(log)

        return log,text,text+log


if __name__ == "__main__":
    print()

