import json
import os
from text2vec import SentenceModel, EncoderType, Similarity, cos_sim
import numpy as np
import torch
import torch.utils.data as Data
from transformers import *

class Scorer(torch.nn.Module):
    """Classifier"""
    def __init__(self, tokenizer):
        super(Scorer, self).__init__()
        self.tokenizer = tokenizer
        self.model = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.fc1 = torch.nn.Linear(768, 768)
        self.fc2 = torch.nn.Linear(768, 2)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, embed):
        
        x = self.fc1(embed)
        logit = self.fc2(x)

        return logit


class Dataset(Data.Dataset):
    """Util Dataset"""
    def __init__(self, data_list, tokenizer): 
        src_list = []; labels = []
        for d in data_list:
            src = d['src_txt']
            label = d['class']
            src_list.append(src)
            labels.append(label)
        
        self.data = tokenizer(
            src_list,
            return_tensors="pt",
            max_length=320,
            truncation=True,
            padding=True
        )

        self.label = np.array(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data["input_ids"][idx]
        label = self.label[idx]
        return data, label

        
    