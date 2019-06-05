#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from copy import deepcopy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from skimage import io
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score
import tensorflow as tf
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
from torchvision import models
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter



# data_path = "~/Datasets/iMet_Colelction_2019"
load_path = "~/Datasets/iMet_Colelction_2019/input/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)


# 複数GPU使用宣言
# if device == 'cuda:1':
#     net = torch.nn.DataParallel(net) # make parallel
#     cudnn.benchmark = True


torch.manual_seed(823)
# torch.manual_seed(823)


# In[2]:


class iMetsDataset(data.Dataset):
    """Face Landmarks dataset."""
 
    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            df (dataframe): ファイル名がindex、Nhot_LabelsカラムにNhot化したラベルを格納したDataframe
            root_dir (string): 対象の画像ファイルが入っているフォルダ
            transform (callable, optional): 施す変換
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
#         print(self.root_dir)
#         print(self.df.index)
#         if type(idx) == torch.Tensor:
#             idx = idx.item()
        img_name = os.path.join(self.root_dir, self.df.index[idx])
        image = Image.fromarray(io.imread(img_name))
        label = self.df.iloc[idx].Nhot_Labels.astype('float32')
        if self.transform:
            image = self.transform(image)
        return image, label


# In[3]:


def Nhot_encoding(arr, l):
    """
    Nhotエンコーディングを行う

    Parameters
    ----------
    arr : ndarray
        ラベル
    l : int
        総ラベル数
    """
    if arr.ndim == 1:
        ret = np.zeros(l,dtype='int')
        ret[arr] = 1
        return ret
    else:
#         A = deepcopy(arr)
#         A = A.reshape((-1,A.size//A.ndim))
        lst = list()
        for i,_ in enumerate(arr):
            lst.extend([i] * arr.shape[1])
            
        ret = np.zeros((arr.shape[0],l),dtype='int')
        ret[lst,arr.flatten()] = 1
        return ret


# In[4]:


batch_size = 100
num_classes = 1103
epochs = 50
extract_attribute = 10


# データ呼び出し

# In[5]:


label_name = pd.read_csv(load_path + 'labels.csv')
label_name = label_name.set_index("attribute_id")
sample = pd.read_csv(load_path + 'sample_submission.csv')
train_df = pd.read_csv(load_path + 'train.csv')
train_size = len(train_df)

train_df["attribute_ids"] = train_df["attribute_ids"].apply(lambda x: np.array([int(s) for s in x.split(" ")]))
train_df["Nhot_Labels"] = train_df["attribute_ids"].apply(lambda x: Nhot_encoding(x,1103))
train_df["id"] = train_df["id"].apply(lambda x: x + ".png")
train_df = train_df.set_index('id')


# In[6]:


torch.manual_seed(823)
np.random.seed(823)

ds = iMetsDataset(train_df,load_path+'train',
                            transform = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            ])
                        )


train_ds, valid_ds, test_ds = data.random_split(ds, [75000,20000,14237])
# # print(train_ds.indices)
# train_ds.indices = train_ds.indices.numpy()
# valid_ds.indices = valid_ds.indices.numpy()
# test_ds.indices = test_ds.indices.numpy()


dataloader_train = data.DataLoader(dataset=train_ds,batch_size=batch_size,shuffle=True)
dataloader_valid = data.DataLoader(dataset=valid_ds,batch_size=batch_size,shuffle=False)
dataloader_test = data.DataLoader(dataset=test_ds,batch_size=batch_size,shuffle=False)


# モデルを定義

# In[7]:


# %%time

# Load the pretrained model from pytorch
vgg16 = models.vgg16_bn(pretrained=True)
# vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
# print(vgg16.classifier[6].out_features) # 1000 

# Freeze training for all layers
# for param in vgg16.features.parameters():
#     param.requires_grad = False

# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, num_classes)]) # Add our layer
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
# load weight
# vgg16.load_state_dict(torch.load('model_epoch2.pkl'))
model = vgg16.to(device)


# In[8]:


criterion = nn.BCELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# trainとevalの定義

# In[11]:


global_step = 0
def train(epoch, writer):
    start = time.time()
    model.train()
    steps = len(train_ds)//batch_size
    for step, (images, labels) in enumerate(dataloader_train, 1):
        global global_step
        global_step += 1
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = torch.sigmoid(model(images))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if step % 30 == 0:
            elapsed_time = time.time() - start
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, time: %d分%d秒' % (epoch, epochs, step, steps, loss.item(), elapsed_time//60, int(elapsed_time % 60)))
            writer.add_scalar('train/train_loss', loss.item() , global_step)
            
#         if step % 250 == 0:
#             eval(step,writer)

            
            
def eval(epoch, writer):
    start = time.time()
    model.eval()
    fbeta_lst = list()
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader_valid):
            images, labels = images.to(device), labels.to(device)
            labels = labels.cpu().detach().numpy()
            outputs = torch.sigmoid(model(images))
            outputs = outputs.cpu().detach().numpy()
            outputs_topN = np.argsort(outputs, axis=1)[:,-extract_attribute:]
            outputs_topN_Nhots = Nhot_encoding(outputs_topN, num_classes)
#             print(labels.shape)
#             print(fbeta_score(labels,outputs_topN_Nhots, beta=2 ,average='samples'))
            fbeta_lst.append(fbeta_score(labels,outputs_topN_Nhots, beta=2 ,average='samples'))
    
    elapsed_time = time.time() - start
    print("Val Acc : %.4f, time: %d分%d秒" % (sum(fbeta_lst)/len(fbeta_lst), elapsed_time//60, int(elapsed_time % 60)))
    writer.add_scalar('eval/val_acc', sum(fbeta_lst)*100/len(fbeta_lst), epoch)


# In[12]:


# %%time
torch.manual_seed(1)
writer = SummaryWriter()
 
for epoch in range(1, epochs+1):
    train(epoch, writer)
    torch.save(model.state_dict(),'model_epoch'+str(epoch)+'.pkl')
    eval(epoch, writer)

writer.close()


# In[ ]:




