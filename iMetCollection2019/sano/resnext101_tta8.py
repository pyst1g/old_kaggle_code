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
arg('model', choices=['resnet34','resnet50', 'resnet101', 'resnet152', 'vgg16_bn', 'resnext101'])
arg('cuda')
arg('mode', choices=['train', 'test'])
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
arg('--lr_dec_rate', type=int, default=2)
args = parser.parse_args(args=['b','resnext101','cuda','test','0','328','--use_tuned_pos_weight','--batch-size', '10','--tta','8','--n_workers','5','--no_pretrained'])


# weight_list is trained weight data with iMets train images <br>
# these files are created with <br>
# ・resnext101_tta8.py ResNext101_lrt_p2t10BCE328_rrcflrono_f0 resnext101 cuda train 0 328 --pos_weight 2  --lr 0.00006666 --tta 10 --lr_tuned<br>
# ・resnext101_tta8.py ResNext101_lrt_p2t4BCE328_rrcflno_f1 resnext101 cuda train 1 328 --pos_weight 2  --lr 0.00006666 --tta 4 --lr_tuned<br>
# ・resnext101_tta8.py ResNext101_lrt_p2t4BCE328_rrcflno_f2 resnext101 cuda train 2 328 --pos_weight 2  --lr 0.00006666 --tta 4 --lr_tuned<br>
# ・resnext101_tta8.py ResNext101_lrt_p2t4BCE328_rrcflno_f3 resnext101 cuda train 3 328 --pos_weight 2  --lr 0.00006666 --tta 4 --lr_tuned<br>
# ・resnext101_tta8.py ResNext101_lrt_p2t10BCE328_rrcflno_f4 resnext101 cuda train 4 328 --pos_weight 2  --lr 0.00006666 --tta 10 --lr_tuned<br>
# 
# respectively.

# In[22]:


weight_list = ['ResNext101_lrt_p2t10BCE328_rrcflrono_f0_epoch26.pkl', 'ResNext101_lrt_p2t4BCE328_rrcflno_f1_epoch16.pkl',
               'ResNext101_lrt_p2t4BCE328_rrcflno_f2_epoch21.pkl', 'ResNext101_lrt_p2t4BCE328_rrcflno_f3_epoch17.pkl',
               'ResNext101_lrt_p2t10BCE328_rrcflno_f4_epoch17.pkl']


# file, GPU setting

# In[5]:


load_path = "../input/imet-2019-fgvc6/" if ON_KAGGLE else "/home/sano/Datasets/iMet_Colelction_2019/input/"

log_dir = 'runs/' + args.save_name
weight_path = './model_weight/' + args.save_name

device = torch.device(args.cuda)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
print('log_dir = ', log_dir)
print('weight save path = ', weight_path)

torch.manual_seed(823)


# define resnext101

# In[6]:


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(arch, inplanes, planes, pretrained, progress, **kwargs):
    model = ResNet(inplanes, planes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnext101_32x8d(pretrained,**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained=pretrained, progress=True, **kwargs)


# const

# In[7]:


batch_size = args.batch_size
num_classes = 1103
extract_attribute = 5 # 予測した上位何個を属性として出力するか

models_dict = {'resnet34'  : models.resnet34,
               'resnet50'  : models.resnet50,
               'resnet101' : models.resnet101,
               'resnet152' : models.resnet152,
               'vgg16_bn'  : models.vgg16_bn,
               'resnext101': resnext101_32x8d
              }


# Focal Loss

# In[8]:


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


# In[9]:


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


# read data

# In[10]:


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


# In[11]:


torch.manual_seed(823)
np.random.seed(823)

ds_allTrain = iMetsDataset(train_df,load_path+'train',mode ='train')


# ds_train, ds_valid = data.random_split(ds_allTrain, [90000, 19237])
ds_train = data.Subset(dataset=ds_allTrain, indices=np.where(train_df['fold'] != args.valid_fold)[0]) 
ds_valid = data.Subset(dataset=ds_allTrain, indices=np.where(train_df['fold'] == args.valid_fold)[0]) 
ds_train.dataset = deepcopy(ds_allTrain)


ds_test = iMetsDataset(submit_df,load_path+'test', mode='test')



ds_train.dataset.transform = transforms.Compose([
                                transforms.Resize((args.image_size*2,args.image_size*2)),
                                transforms.RandomResizedCrop((args.image_size,args.image_size)),
                                transforms.RandomHorizontalFlip(p=1),
#                                 transforms.RandomRotation((-20,20)),
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

# In[12]:


cnt_attribute = Counter()
for i in train_df.attribute_ids:
    cnt_attribute.update(i)

freq_attr = np.asarray(cnt_attribute.most_common())

bias_pos_weight = np.zeros(num_classes)
bias_pos_weight[freq_attr[:,0]] = (len(dataloader_train.dataset) / 2) / freq_attr[:,1]
bias_pos_weight[bias_pos_weight>100] = 100


# モデルを定義

# In[23]:


torch.manual_seed(823)
np.random.seed(823)

model = models_dict[args.model](pretrained = not args.no_pretrained)

if args.model.startswith('resne'):
    num_features = model.fc.in_features
    features = list(model.fc.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, num_classes)]) # Add our layer
    model.fc = nn.Sequential(*features) # Replace the model classifier

elif args.model.startswith('vgg'):
    num_features = model.classifier[-1].in_features
    features = list(model.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, num_classes)]) # Add our layer
    model.classifier = nn.Sequential(*features) # Replace the model classifier

# model.load_state_dict(torch.load('../input/imets-resnext101-fold5-tta8/ResNext101_lrt_p2t10BCE328_rrcflrono_f0_epoch26.pkl',map_location=device))
# model.load_state_dict(torch.load('model_weight/'+weight_list[4],map_location=device))
model = model.to(device)


# In[ ]:


if args.loss == 'BCE':
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(np.ones(num_classes) * args.pos_weight))
elif args.loss == 'FL':
    criterion = FocalLoss(gamma=2, logits=True, pos_weight=torch.from_numpy(np.ones(num_classes) * args.pos_weight))

if args.use_tuned_pos_weight:
    criterion.pos_weight = torch.from_numpy(bias_pos_weight)

criterion.pos_weight = criterion.pos_weight.float().to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25, 30, 35, 40, 45], gamma=1/args.lr_dec_rate)


# train,eval,predictの定義

# In[ ]:


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
    
    # TTA topN
    propotion_arr, labels_arr = pred_prop()
#     pred_labels_topN = Nhot_encoding(np.argsort(np.mean(propotion_arr,axis=0), axis=1)[:,-extract_attribute:], num_classes)
#     f2 = fbeta_score(labels_arr, pred_labels_topN, beta=2 ,average='samples')
#     precision = precision_score(labels_arr,pred_labels_topN,average='samples')
#     recall = recall_score(labels_arr,pred_labels_topN,average='samples')
#     print("Val Acc(topN)   : %.10f" % f2)
#     print("precision(topN) : %.10f" % precision)
#     print("recall(topN)    : %.10f" % recall)

    
    thr, max_f2 = make_only_threthold(np.mean(propotion_arr,axis=0),labels_arr)
    elapsed_time = time.time() - start
    print("elapsed_time: %d分%d秒" % (elapsed_time//60, int(elapsed_time % 60)))
    
    pred_top12 = np.mean(propotion_arr,axis=0) * Nhot_encoding(np.argsort(np.mean(propotion_arr,axis=0), axis=1)[:,-12:], num_classes)
#     pred_labels = np.mean(propotion_arr,axis=0) > thr
    pred_labels = pred_top12 > thr
    f2 = fbeta_score(labels_arr,pred_labels, beta=2 ,average='samples')
    precision = precision_score(labels_arr,pred_labels,average='samples')
    recall = recall_score(labels_arr,pred_labels,average='samples')
    print("Val Acc   : %.10f" % f2)
    print("precision : %.10f" % precision)
    print("recall    : %.10f" % recall)
        
#     writer.add_scalar('eval/val_acc', f2*100, epoch)
#     writer.add_scalar('eval/precision', precision*100, epoch)
#     writer.add_scalar('eval/recall', recall*100, epoch)
    
    return thr

def predict(thr, dataloader=dataloader_test):
    steps = len(dataloader.dataset)
    propotion_arr_TTA = np.zeros((steps, num_classes))
    for i in range(5):
        model.load_state_dict(torch.load('../input/imets-resnext101-fold5-tta8/'+model_weight[i],map_location=device))
        model = model.to(device)
        propotion_arr_TTA += pred_test()
        
    propotion_arr_TTA /= 5
    
    

    pred_top12 = propotion_arr_TTA * Nhot_encoding(np.argsort(propotion_arr_TTA, axis=1)[:,-12:], num_classes)
#     pred_labels = propotion_arr_TTA > thr
    pred_labels = pred_top12 > thr
    
    pos, label = np.where(pred_labels==1)
    
    pred_attr = list()
    for i in range(len(dataloader_test.dataset)):
        pred_attr.append(label[pos==i])
    return pred_attr


# In[ ]:


def pred_prop(dataloader = dataloader_valid):
    start = time.time()
    model.eval()
    steps = len(dataloader.dataset)
    propotion_arr_TTA = np.zeros((steps, num_classes))
    
    for t in range(args.tta):
        if t==0:
            ds_valid.dataset.transform = transforms.Compose([
                                transforms.Resize((args.image_size,args.image_size)),
#                                 transforms.RandomResizedCrop((args.image_size,args.image_size)),
                                transforms.RandomHorizontalFlip(p=1),
#                                 transforms.RandomRotation((-20,20)),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    [0.485, 0.456, 0.406], 
                                    [0.229, 0.224, 0.225]
                                ),
                            ])
        if t==1:
            ds_valid.dataset.transform = transforms.Compose([
                                transforms.Resize((args.image_size,args.image_size)),
#                                 transforms.RandomResizedCrop((args.image_size,args.image_size)),
#                                 transforms.RandomHorizontalFlip(p=1),
#                                 transforms.RandomRotation((-20,20)),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    [0.485, 0.456, 0.406], 
                                    [0.229, 0.224, 0.225]
                                ),
                            ])
        else:
            ds_valid.dataset.transform = transforms.Compose([
                                transforms.RandomResizedCrop((args.image_size,args.image_size)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    [0.485, 0.456, 0.406], 
                                    [0.229, 0.224, 0.225]
                                ),
                            ])

            
        print(dataloader.dataset.dataset.transform)
    
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
                if i % 10 == 0:
                    elapsed_time = time.time() - start
                    print('\r[%d/%d], TTA %d time: %d分%d秒' % (min((i * batch_size),steps), steps, t, elapsed_time//60, int(elapsed_time % 60)))
                    clear_output(wait=True)
        propotion_arr = np.asarray(propotion_arr)
        labels_arr = np.asarray(labels_arr)
        propotion_arr_TTA += propotion_arr
        elapsed_time = time.time() - start
        print('time: %d分%d秒' % (elapsed_time//60, int(elapsed_time % 60)))
    print()
    propotion_arr_TTA /= args.tta
    
    return propotion_arr_TTA, labels_arr


def pred_test(dataloader=dataloader_test):
    for t in range(args.tta):
        if t==0:
            ds_test.transform = transforms.Compose([
                                transforms.Resize((args.image_size,args.image_size)),
#                                 transforms.RandomResizedCrop((args.image_size,args.image_size)),
                                transforms.RandomHorizontalFlip(p=1),
#                                 transforms.RandomRotation((-20,20)),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    [0.485, 0.456, 0.406], 
                                    [0.229, 0.224, 0.225]
                                ),
                            ])
        if t==1:
            ds_test.transform = transforms.Compose([
                                transforms.Resize((args.image_size,args.image_size)),
#                                 transforms.RandomResizedCrop((args.image_size,args.image_size)),
#                                 transforms.RandomHorizontalFlip(p=1),
#                                 transforms.RandomRotation((-20,20)),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    [0.485, 0.456, 0.406], 
                                    [0.229, 0.224, 0.225]
                                ),
                            ])
        else:
            ds_test.transform = transforms.Compose([
                                transforms.RandomResizedCrop((args.image_size,args.image_size)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    [0.485, 0.456, 0.406], 
                                    [0.229, 0.224, 0.225]
                                ),
                            ])


        
        
        print(t, end=' ')
        propotion_arr = list()
        # ラベル確率を推論
        with torch.no_grad():
            for i, images in enumerate(dataloader,1):
                images = images.to(device)
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
#                 if i % 20 == 0:
#                     elapsed_time = time.time() - start
#                     sys.stdout.write('\r%d [%d/%d] time: %d分%d秒' % (t,min((i * args.batch_size),test_size), test_size, elapsed_time//60, int(elapsed_time % 60)))
#                     sys.stdout.flush()
        propotion_arr = np.asarray(propotion_arr)
        propotion_arr_TTA += propotion_arr
    print()
    propotion_arr_TTA /= args.tta
    return propotion_arr_TTA


# In[ ]:


def make_only_threthold(propotion_arr, labels_arr, sample_num = 10000, tta_num=None):
    start = time.time()
    

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


# In[14]:


if args.mode == 'train':
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


if args.mode == 'test':
    pred = predict(0.17074707470747075)
    pred_str = list()
    for lst in pred:
        pred_str.append(" ".join(list(map(str, lst))))

    submit_df.index = submit_df.index.map(lambda x:x.rstrip(".png"))
    submit_df.attribute_ids = pred_str

    submit_df.to_csv("submission.csv", index=True)

