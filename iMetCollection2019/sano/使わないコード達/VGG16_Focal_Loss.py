#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
import sys
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
# load_path = "../input/"
load_path = "~/Datasets/iMet_Colelction_2019/input/"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("device", device)


# 複数GPU使用宣言
# if device == 'cuda:1':
#     net = torch.nn.DataParallel(net) # make parallel
#     cudnn.benchmark = True


torch.manual_seed(823)


# In[3]:


batch_size = 30
num_classes = 1103
epochs = 20
extract_attribute = 5 # 予測した上位何個を属性として出力するか


# In[4]:


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


# In[5]:


class iMetsDataset(data.Dataset):
 
    def __init__(self, df, root_dir, transform=None, mode='train'):
        """
        Args:
            df (dataframe): ファイル名がindex、Nhot_LabelsカラムにNhot化したラベルを格納したDataframe
            root_dir (string): 対象の画像ファイルが入っているフォルダ
            transform (callable, optional): 施す変換
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
#         if type(idx) == torch.Tensor:
#             idx = idx.item()
        img_name = os.path.join(self.root_dir, self.df.index[idx])
        image = Image.fromarray(io.imread(img_name))
        if self.transform:
            image = self.transform(image)
            
        if self.mode == 'train':
            label = self.df.iloc[idx].Nhot_Labels.astype('float32')
            return image, label
        else:
            return image
    
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
        lst = list()
        for i,_ in enumerate(arr):
            lst.extend([i] * arr.shape[1])
            
        ret = np.zeros((arr.shape[0],l),dtype='int')
        ret[lst,arr.flatten()] = 1
        return ret


# データ呼び出し

# In[6]:


label_name = pd.read_csv(load_path + 'labels.csv')
label_name = label_name.set_index("attribute_id")
submit_df = pd.read_csv(load_path + 'sample_submission.csv')
submit_df["id"] = submit_df["id"].apply(lambda x: x + ".png")
submit_df = submit_df.set_index('id')
test_size = len(submit_df)

train_df = pd.read_csv(load_path + 'train.csv')
train_size = len(train_df)
train_df["attribute_ids"] = train_df["attribute_ids"].apply(lambda x: np.array([int(s) for s in x.split(" ")]))
train_df["Nhot_Labels"] = train_df["attribute_ids"].apply(lambda x: Nhot_encoding(x,1103))
train_df["id"] = train_df["id"].apply(lambda x: x + ".png")
train_df = train_df.set_index('id')


# In[7]:


torch.manual_seed(823)
np.random.seed(823)

ds_allTrain = iMetsDataset(train_df,load_path+'train',
                            transform = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            ]),
                        )

ds_train, ds_valid = data.random_split(ds_allTrain, [90000, 19237])

ds_test = iMetsDataset(submit_df,load_path+'test',
                            transform = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            ]),
                           mode='test'
                        )

if type(ds_train.indices) == torch.Tensor:
    ds_train.indices = ds_train.indices.numpy()
    ds_valid.indices = ds_valid.indices.numpy()


dataloader_train = data.DataLoader(dataset=ds_train,batch_size=batch_size,shuffle=True)
dataloader_valid = data.DataLoader(dataset=ds_valid,batch_size=batch_size,shuffle=False)
dataloader_test = data.DataLoader(dataset=ds_test,batch_size=batch_size,shuffle=False)


# モデルを定義

# In[7]:



torch.manual_seed(823)
np.random.seed(823)


# Load the pretrained model from pytorch
vgg16 = models.vgg16_bn(pretrained=False)
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
# vgg16.load_state_dict(torch.load('model_weight/vgg16/model_epoch6.pkl'))
model = vgg16.to(device)


# In[8]:


# criterion = nn.BCELoss()
criterion = FocalLoss(gamma=2, logits=False)
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# train,eval,predictの定義

# In[9]:


global_step = 0
def train(epoch, writer):
    start = time.time()
    model.train()
    steps = len(ds_train)//batch_size
    for step, (images, labels) in enumerate(dataloader_train, 1):
        global global_step
        global_step += 1
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = torch.sigmoid(model(images))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            elapsed_time = time.time() - start
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.10f, time: %d分%d秒' % (epoch, epochs, step, steps, loss.item(), elapsed_time//60, int(elapsed_time % 60)))
            writer.add_scalar('train/train_loss', loss.item() , global_step)

            
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
            fbeta_lst.append(fbeta_score(labels,outputs_topN_Nhots, beta=2 ,average='samples'))
    
    elapsed_time = time.time() - start
    print("Val Acc : %.10f, time: %d分%d秒" % (sum(fbeta_lst)/len(fbeta_lst), elapsed_time//60, int(elapsed_time % 60)))
    writer.add_scalar('eval/val_acc', sum(fbeta_lst)*100/len(fbeta_lst), epoch)


def predict():
    pred_attr = list()
    model.eval()
    with torch.no_grad():
        for i, images in enumerate(dataloader_test,1):
            images = images.to(device)
            outputs = torch.sigmoid(model(images))
            outputs = outputs.cpu().detach().numpy()
            outputs_topN = np.argsort(outputs, axis=1)[:,-extract_attribute:]
            for attr in outputs_topN:
                pred_attr.append(attr)
            if i % 10 == 0:
                sys.stdout.write('\r[%d/%d]' % (min((i * batch_size),test_size), test_size))
                sys.stdout.flush()
    return pred_attr          


# In[ ]:




torch.manual_seed(1)
writer = SummaryWriter()
 
for epoch in range(1, epochs+1):
    train(epoch, writer)
    eval(epoch, writer)
    torch.save(model.state_dict(),'./model_weight/vgg16/VGG_FocalLoss_epoch'+str(epoch)+'.pkl')


writer.close()


# In[ ]:


# pred = predict()
# pred_str = list()
# for lst in pred:
#     pred_str.append(" ".join(list(map(str, lst))))

# submit_df.index = submit_df.index.map(lambda x:x.rstrip(".png"))
# submit_df.attribute_ids = pred_str

# submit_df.to_csv("submission.csv", index=True)


# In[ ]:





# In[ ]:




