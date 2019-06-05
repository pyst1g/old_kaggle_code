#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
from collections import Counter
from copy import deepcopy
from functools import partial
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from skimage import io
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, precision_score, recall_score
import sys
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

ON_KAGGLE: bool = 'KAGGLE_WORKING_DIR' in os.environ


# argment

# In[2]:


parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('save_name')
arg('model', choices=['resnet34','resnet50', 'resnet101', 'resnet152', 'vgg16_bn'])
arg('cuda')
arg('mode', choices=['train', 'eval'])
arg('valid_fold', type=int, choices=[0,1,2,3,4])
arg('image_size', type=int)
arg('--loss', choices=['BCE', 'FL'], default='BCE')
arg('--pos_weight', type=int, default=1)
arg('--use_tuned_pos_weight', action='store_true')
arg('--no_pretrained', action='store_true')
arg('--batch-size', type=int, default=50)
arg('--epochs', type=int, default=50)
arg('--tta', type=int, default=1)
arg('--n_workers', type=int, default=3)
arg('--lr', type=float, default=0.0001)
arg('--verbose', action='store_true')
arg('--lr_tuned', action='store_true')
args = parser.parse_args()#args=['b','resnet34','cuda','train','0','288','--use_tuned_pos_weight','--batch-size', '10','--tta','3'])


# file, GPU setting

# In[3]:


load_path = "../input/kaggle-imet-2019/" if ON_KAGGLE else "/home/sano/Datasets/iMet_Colelction_2019/input/"

log_dir = 'runs/' + args.save_name
weight_path = './model_weight/' + args.save_name

device = torch.device(args.cuda)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
print('log_dir = ', log_dir)
print('weight save path = ', weight_path)

torch.manual_seed(823)


# const

# In[4]:


batch_size = args.batch_size
num_classes = 1103
extract_attribute = 5 # 予測した上位何個を属性として出力するか

models_dict = {'resnet34'  : models.resnet34,
               'resnet50'  : models.resnet50,
               'resnet101' : models.resnet101,
               'resnet152' : models.resnet152,
               'vgg16_bn'  : models.vgg16_bn}


# Focal Loss

# In[5]:


class FocalLoss(nn.Module):
    def __init__(self, pos_weight=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.pos_weight * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


# In[6]:


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
        image = Image.open(img_name)
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

# In[7]:


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

np.random.seed(823)
fold = pd.Series(np.arange(len(train_df)) % 5)
np.random.shuffle(fold)
train_df['fold'] = fold
del fold


train_df = train_df.set_index('id')


# In[8]:


torch.manual_seed(823)
np.random.seed(823)

ds_allTrain = iMetsDataset(train_df,load_path+'train',mode ='train')


# ds_train, ds_valid = data.random_split(ds_allTrain, [90000, 19237])
ds_train = data.Subset(dataset=ds_allTrain, indices=np.where(train_df['fold'] != args.valid_fold)[0]) 
ds_valid = data.Subset(dataset=ds_allTrain, indices=np.where(train_df['fold'] == args.valid_fold)[0]) 
ds_train.dataset = deepcopy(ds_allTrain)


ds_test = iMetsDataset(submit_df,load_path+'test', mode='test')



ds_train.dataset.transform = transforms.Compose([
                                transforms.RandomResizedCrop((args.image_size,args.image_size)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation((-20,20)),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    [0.485, 0.456, 0.406], 
                                    [0.229, 0.224, 0.225]
                                ),
                            ])


if args.tta == 1:
    ds_valid.dataset.transform = transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        [0.485, 0.456, 0.406], 
                                        [0.229, 0.224, 0.225]
                                    )
                                ])
else:
    ds_valid.dataset.transform = ds_train.dataset.transform


ds_test.transform = ds_valid.dataset.transform
#                           transforms.Compose([
#                                 transforms.Resize((args.image_size, args.image_size)),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(
#                                     [0.485, 0.456, 0.406], 
#                                     [0.229, 0.224, 0.225]
#                                 ),
#                             ])


if type(ds_train.indices) == torch.Tensor:
    ds_train.indices = ds_train.indices.numpy()
    ds_valid.indices = ds_valid.indices.numpy()


dataloader_train = data.DataLoader(dataset=ds_train,batch_size=batch_size,shuffle=True,num_workers=args.n_workers)
dataloader_valid = data.DataLoader(dataset=ds_valid,batch_size=batch_size,shuffle=False,num_workers=args.n_workers)
dataloader_test = data.DataLoader(dataset=ds_test,batch_size=batch_size,shuffle=False,num_workers=args.n_workers)


# calculate frequency of attributes and bias_pos_weight

# In[9]:


cnt_attribute = Counter()
for i in train_df.attribute_ids:
    cnt_attribute.update(i)

freq_attr = np.asarray(cnt_attribute.most_common())

bias_pos_weight = np.zeros(num_classes)
bias_pos_weight[freq_attr[:,0]] = (len(dataloader_train.dataset) / 2) / freq_attr[:,1]
bias_pos_weight[bias_pos_weight>100] = 100


# モデルを定義

# In[10]:


torch.manual_seed(823)
np.random.seed(823)

model = models_dict[args.model](pretrained=not args.no_pretrained)

if args.model.startswith('resnet'):
    num_features = model.fc.in_features
    features = list(model.fc.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, num_classes)]) # Add our layer
    model.fc = nn.Sequential(*features) # Replace the model classifier

elif args.model.startswith('vgg'):
    num_features = model.classifier[-1].in_features
    features = list(model.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, num_classes)]) # Add our layer
    model.classifier = nn.Sequential(*features) # Replace the model classifier

    
# model.load_state_dict(torch.load('model_weight/resnet152/resnet152_FocalLoss_epoch7.pkl'))
model = model.to(device)


# In[25]:


if args.loss == 'BCE':
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(np.ones(num_classes) * args.pos_weight))
elif args.loss == 'FL':
    criterion = FocalLoss(gamma=2, logits=True, pos_weight=torch.from_numpy(np.ones(num_classes) * args.pos_weight))

if args.use_tuned_pos_weight:
    criterion.pos_weight = torch.from_numpy(bias_pos_weight)

criterion.pos_weight = criterion.pos_weight.float().to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25, 30, 35, 40, 45], gamma=0.5)


# train,eval,predictの定義

# In[12]:


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
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if step % (len(dataloader_train.dataset) // (60 * args.batch_size)) == 0:
            elapsed_time = time.time() - start
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.10f, time: %d分%d秒' % (epoch, args.epochs, step, steps, loss.item(), elapsed_time//60, int(elapsed_time % 60)))
            writer.add_scalar('train/train_loss', loss.item() , global_step)
            
def eval(epoch, writer):
    start = time.time()
    model.eval()
    
    propotion_arr, labels_arr = pred_prop()
    thr, max_f2 = make_only_threthold(np.mean(propotion_arr,axis=0),labels_arr)
    elapsed_time = time.time() - start
    print("elapsed_time: %d分%d秒" % (elapsed_time//60, int(elapsed_time % 60)))
    
    if args.verbose:
        for t in range(args.tta):
            pred_labels = np.mean(propotion_arr[:t+1],axis=0) > thr
            f2 = fbeta_score(labels_arr,pred_labels, beta=2 ,average='samples')
            precision = precision_score(labels_arr,pred_labels,average='samples')
            recall = recall_score(labels_arr,pred_labels,average='samples')
            print('tta ', t)
            print("Val Acc   : %.10f" % f2)
            print("precision : %.10f" % precision)
            print("recall    : %.10f" % recall)
    else:
        pred_labels = np.mean(propotion_arr,axis=0) > thr
        f2 = fbeta_score(labels_arr,pred_labels, beta=2 ,average='samples')
        precision = precision_score(labels_arr,pred_labels,average='samples')
        recall = recall_score(labels_arr,pred_labels,average='samples')
        print("Val Acc   : %.10f" % f2)
        print("precision : %.10f" % precision)
        print("recall    : %.10f" % recall)
        
    writer.add_scalar('eval/val_acc', f2*100, epoch)
    writer.add_scalar('eval/precision', precision*100, epoch)
    writer.add_scalar('eval/recall', recall*100, epoch)
    

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


def pred_prop(dataloader = dataloader_valid):
    start = time.time()
    model.eval()
    steps = len(dataloader.dataset)
    propotion_arr_TTA = np.zeros((args.tta,steps, num_classes))
    
    for t in range(args.tta):
        print(t, end=' ')
        propotion_arr = list()
        labels_arr = list()
        # ラベル確率を推論
        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader,1):
                images = images.to(device)
                labels = labels.cpu().detach().numpy()
                labels_arr.extend(labels)
                outputs = torch.sigmoid(model(images))
                outputs = outputs.cpu().detach().numpy()
                propotion_arr.extend(outputs)
            #         outputs_topN = np.argsort(outputs, axis=1)[:,-extract_attribute:]
            #         for attr in outputs_topN:
            #             pred_attr.append(attr)
#                 if i % 10 == 0:
#                     elapsed_time = time.time() - start
#                     print('\r[%d/%d], TTA %d time: %d分%d秒' % (min((i * batch_size),steps), steps, t, elapsed_time//60, int(elapsed_time % 60)))
#                     clear_output(wait=True)
        propotion_arr = np.asarray(propotion_arr)
        labels_arr = np.asarray(labels_arr)
        propotion_arr_TTA[t] += propotion_arr
    print()
#     propotion_arr_TTA /= args.tta
    
    return propotion_arr_TTA, labels_arr


# In[13]:


def make_only_threthold(propotion_arr, labels_arr, sample_num = 10000, tta_num=None):
    start = time.time()
    
#     model.eval()
#     steps = len(ds_valid)
#     propotion_arr = list()
#     labels_arr = list()

#     # ラベル確率を推論
#     with torch.no_grad():
#         for i, (images, labels) in enumerate(dataloader_valid,1):
#             images = images.to(device)
#             labels = labels.cpu().detach().numpy()
#             labels_arr.extend(labels)
#             outputs = torch.sigmoid(model(images))
#             outputs = outputs.cpu().detach().numpy()
#             propotion_arr.extend(outputs)
#         #         outputs_topN = np.argsort(outputs, axis=1)[:,-extract_attribute:]
#         #         for attr in outputs_topN:
#         #             pred_attr.append(attr)
#             if i % 10 == 0:
#                 elapsed_time = time.time() - start
#                 print('\r[%d/%d], time: %d分%d秒' % (min((i * batch_size),steps), steps, elapsed_time//60, int(elapsed_time % 60)))
#                 clear_output(wait=True)


#     propotion_arr = np.asarray(propotion_arr)
#     labels_arr = np.asarray(labels_arr)

    pc = deepcopy(propotion_arr)
    lc = deepcopy(labels_arr)
    pc = np.reshape(pc,-1)
    lc = np.reshape(lc,-1)
    idx = np.argsort(pc)
    pc = pc[idx]
    lc = lc[idx]

    TP = np.sum(labels_arr==1, axis=1)
    FN = np.zeros_like(TP)
    FP = np.sum(labels_arr==0, axis=1)
    TN = np.zeros_like(TP)

    f2 = np.zeros_like(TP)

    tmp_max = 0
    max_thr = 0
    pos = 0
    for i, thr in enumerate(np.linspace(10**-3,1,sample_num)):
        if i % 10 == 0:
            elapsed_time = time.time() - start
#             print('\r[%d/%d], time: %d分%d秒' % (i, sample_num, elapsed_time//60, int(elapsed_time % 60)))
        while pos < len(pc) and pc[pos] < thr:
            if lc[pos] == 0:
                FP[idx[pos] // num_classes] -= 1
                TN[idx[pos] // num_classes] += 1
            else:
                TP[idx[pos] // num_classes] -= 1
                FN[idx[pos] // num_classes] += 1
#             if pos % 100000 == 0: 
#                 elapsed_time = time.time() - start
#                 if tta_num: print(tta_num)
#                 print('\r[%d/%d], time: %d分%d秒' % (i, sample_num, elapsed_time//60, int(elapsed_time % 60)))
#                 print('\r[%d/%d]' % (pos//1000, len(pc)//1000))
#                 clear_output(wait=True)
            pos += 1

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f2_arr = 5*(precision * recall) / (4*precision + recall)
        f2_arr[np.isnan(f2_arr)] = 0
        f2 = np.mean(f2_arr)
        if f2 > tmp_max:
            tmp_max = f2
            max_thr = thr
    return max_thr, tmp_max


# In[17]:


torch.manual_seed(1)
writer = SummaryWriter(log_dir)
 
for epoch in range(1, args.epochs+1):
    train(epoch, writer)
    eval(epoch, writer)
    torch.save(model.state_dict(), weight_path + '_epoch' + str(epoch)+'.pkl')
    if args.lr_tuned:
        scheduler.step()

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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




