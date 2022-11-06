#!/usr/bin/env python
# coding: utf-8

# # Bert test by latest torchtext

# # Preparation

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import urllib.request
from urllib.parse import urlparse
from glob import glob
import os
import re
from itertools import chain
import sys

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from janome.tokenizer import Tokenizer
from transformers import BertJapaneseTokenizer, pipeline, BertForSequenceClassification, AdamW
from torchtext.vocab import vocab, build_vocab_from_iterator
from torch.utils.data import DataLoader
import torch
from torch.nn import functional as F
import torchtext.transforms as T
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, plot_confusion_matrix
) 


# In[3]:


bert_tokenizer_path = 'cl-tohoku/bert-base-japanese-whole-word-masking'
bert_model_path = 'cl-tohoku/bert-base-japanese-whole-word-masking'


# # Data settings

# In[4]:


def _read_data(fil: str, columns: list) -> pd.DataFrame:
    with open(fil, 'r') as fr:
        lines = fr.readlines()
        label = fil.split('/')[-2]
        id = label2id[label]
        title = lines[2].strip("\n")

        text = "".join(lines[3:])
        text = "".join(text.split())

        record = pd.Series([id, label, title, text], index=columns)
    return record.to_frame().T
    pass


# In[5]:


if set(os.listdir("title_dataset")) >= {"train.tsv", "dev.tsv", "test.tsv"}:
    df_dataset_train = pd.read_csv('title_dataset/train.tsv',  names=['id', 'title'], sep='\t',
                            header=None)
    df_dataset_dev = pd.read_csv('title_dataset/dev.tsv', names=['id', 'title'], sep='\t',
                          header=None)
    df_dataset_test = pd.read_csv('title_dataset/test.tsv',  names=['id', 'title'], sep='\t',
                           header=None)
    pass
else:
    columns = ["id", "label", "title", "text"]
    df_dataset = pd.concat((
        _read_data(fil, columns)
        for fil in glob.glob("raw_data/text/**/*", recursive=False) if "LICENCE" not in fil
    ), ignore_index=True)


    df_dataset = df_dataset.sample(frac=1, random_state=123)

    df_dataset['all'] = df_dataset['title'] + ':' + df_dataset['text']

    df_dataset_train = df_dataset[:-1000]
    df_dataset_dev = df_dataset[-1000:-500]
    df_dataset_test = df_dataset[-500:]    
    pass


# # Dataloader

# ## tokenize

# In[9]:


# cl-tohokuのモデルは512トークンまで対応しているが、512トークンではモデルがcolabのGPUメモリに乗らないため簡略化
n_tokens = 256
batch_size=16
num_labels = len(set(
    df_dataset_train.id.unique().tolist() +
    df_dataset_test.id.unique().tolist() +
    df_dataset_dev.id.unique().tolist()
))


# In[11]:


tokenizer = BertJapaneseTokenizer.from_pretrained(bert_tokenizer_path)
tokenizer.tokenize('今日はいい天気ですね。')


# In[12]:


df_dataset_train["text"] = df_dataset_train.title.apply(tokenizer.tokenize)
df_dataset_dev["text"] = df_dataset_dev.title.apply(tokenizer.tokenize)
df_dataset_test["text"] = df_dataset_test.title.apply(tokenizer.tokenize)


# # Transform obj

# In[13]:


text_vocab = vocab(tokenizer.vocab, specials=tokenizer.all_special_tokens)
text_vocab.set_default_index(text_vocab['[UNK]'])


# In[14]:


label_vocab = build_vocab_from_iterator(
    pd.concat([
        df_dataset_train.id,
        df_dataset_dev.id,
        df_dataset_test.id,
    ]).astype(str),
)


# In[15]:


text_transform = T.Sequential(
    T.VocabTransform(text_vocab),
    T.ToTensor(padding_value=text_vocab['[UNK]'])
)


# In[22]:


label_transform = T.Sequential(
    T.VocabTransform(label_vocab),
    T.ToTensor()
)


# In[23]:


def collate_batch(batch):
    texts = text_transform([text for (label, _, text) in batch])
    labels = label_transform([label for (label, _, text) in batch])
    return texts, labels
    pass


# In[24]:


dl_train = DataLoader(
    df_dataset_train\
    .astype({"id": str})\
    .values, 
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch
)

dl_dev = DataLoader(
    df_dataset_dev\
    .astype({"id": str})\
    .values, 
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch
)

dl_test = DataLoader(
    df_dataset_test\
    .astype({"id": str})\
    .values, 
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch
)


# # Model train

# In[25]:


model = BertForSequenceClassification.from_pretrained(
    bert_model_path, 
    num_labels=num_labels,
    output_attentions = False,
    output_hidden_states = False)


# In[26]:


optimizer = AdamW(model.parameters(), lr=1e-5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 使用デバイスにGPUを設定
# 以下のような出力が出ていれば正常に設定ができている
# device(type='cuda', index=0)
device


# In[27]:


def train_classification(dataloader_train, dataloader_dev, n_epoch=1):
    for e in range(n_epoch):
        model.train()
        train_loss = 0
        for input_ids, labels in dataloader_train:
            # b_input_ids = batch.Text[0].to(device)
            # b_labels = batch.Label.to(device)
            b_input_ids = input_ids.to(device)
            b_labels = labels.to(device)            
            optimizer.zero_grad()
            outputs = model(b_input_ids)
            loss = F.cross_entropy(outputs.logits, b_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        trues, preds = [], []
        for input_ids, labels in dataloader_dev:
            b_input_ids = input_ids.to(device)
            trues = np.concatenate([trues, labels.numpy()])

            outputs = model(b_input_ids)
            outputs = np.argmax(outputs[0].cpu().detach().numpy(), axis=1)
            preds = np.concatenate([preds, outputs])

        print('精度　 :{:.3f}'.format(accuracy_score(trues, preds)))
        print('適合率 :{:.3f}'.format(precision_score(trues, preds, average='macro')))
        print('再現率 :{:.3f}'.format(recall_score(trues, preds, average='macro')))
        print('f-1値  :{:.3f}'.format(f1_score(trues, preds, average='macro')))


# In[28]:


# epoch数3程度が最もdevデータで高精度となるが、1でも大差はない
n_epoch = 3

train_classification(dl_train, dl_dev, n_epoch=n_epoch)


# In[29]:


model.eval()

trues, preds = [], []
for input_ids, labels in dl_test:
    b_input_ids = input_ids.to(device)
    trues = np.concatenate([trues, labels.numpy()])

    outputs = model(b_input_ids)
    outputs = np.argmax(outputs[0].cpu().detach().numpy(), axis=1)
    preds = np.concatenate([preds, outputs])


# In[30]:


print('精度　 :{:.3f}'.format(accuracy_score(trues, preds)))
print('適合率 :{:.3f}'.format(precision_score(trues, preds, average='macro')))
print('再現率 :{:.3f}'.format(recall_score(trues, preds, average='macro')))
print('f-1値  :{:.3f}'.format(f1_score(trues, preds, average='macro')))


# In[31]:


labels = [v for v in label2id.values()]
cm = confusion_matrix(trues, preds, labels=labels)
cm


# In[34]:


import seaborn as sn
import matplotlib.pyplot as plt

for k,v in label2id.items(): print(v, k)
sn.heatmap(cm, annot=True)


# In[52]:


os.makedirs("model", exist_ok=True)


# In[32]:


model.cpu().save_pretrained('model/')


# In[33]:


model_l = BertForSequenceClassification.from_pretrained('./model/')


# In[55]:


model_l.to(device)

