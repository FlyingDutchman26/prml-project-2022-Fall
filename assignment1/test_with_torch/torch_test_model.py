# 利用MNIST 手写数字识别数据集 实现一个完善的 多分类任务的 全连接神经网络
# 不需要划分 validation_set ，直接有一个test_dataset 可以测试准确率
# 我的实验结果准确率达到了了97%，与老师的实验结果相符合

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
torch.set_default_tensor_type(torch.FloatTensor)
# 初始化并固定随机种子
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1012)

# prepare dataset
import numpy as np
dataset_path = './dataset'

X_train = np.load(dataset_path + '/X_train.npy')
y_train = np.load(dataset_path + '/y_train.npy')
X_test = np.load(dataset_path + '/X_test.npy')
y_test = np.load(dataset_path + '/y_test.npy')

from imblearn.over_sampling import SMOTE
from collections import Counter
np.random.seed(1012)
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
print('当前各类别数量',sorted(Counter(y_train).items()))



is_shuffle = True
if is_shuffle:
    idx = torch.randperm(X_train.shape[0])
    X_train = X_train[idx]
    y_train = y_train[idx]
    
batch_size = 64

import numpy as np
mean_train = np.mean(X_train)
std_train = np.std(X_train)
mean_test = np.mean(X_test)
std_test = np.std(X_test)
X_train = (X_train-mean_train)/std_train
X_test = (X_test - mean_test)/std_test
X_train = torch.tensor(X_train) 
y_train = torch.tensor(y_train,dtype=int)   
X_test= torch.tensor(X_test)
y_test = torch.tensor(y_test,dtype = int)
X_train = X_train.to(torch.float32)
X_test = X_test.to(torch.float32)

class Dataset(Dataset):
    def __init__(self,X,y):
        super(Dataset,self).__init__()
        self.X = X
        self.y = y
        self.len = self.y.shape[0]
    def __getitem__(self, idx):
        return self.X[idx],self.y[idx]
    def __len__(self):
        return self.len

train_dataset = Dataset(X_train,y_train)
test_dataset = Dataset(X_test,y_test)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_dataset,batch_size=batch_size)


# design model

class Net_L2(nn.Module):
    def __init__(self) -> None:
        super(Net_L2, self).__init__()
        self.l1 = nn.Linear(784, )
        self.l2 = nn.Linear(256, 10)
        self.layers = nn.Sequential(
            self.l1,
            nn.ReLU(),
            self.l2,
            # 最后一层softmax之后会自动接在损失函数之上
        )

    def forward(self, x):
        x = x.view(-1, 784)  # 把minibatch reshape to [N, 784], 784是每个图像的特征维度
        return self.layers(x)

class Net_L5(nn.Module):
    def __init__(self) -> None:
        super(Net_L5, self).__init__()
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, 10)
        self.layers = nn.Sequential(
            self.l1,
            nn.ReLU(),
            self.l2,
            nn.ReLU(),
            self.l3,
            nn.ReLU(),
            self.l4,
            nn.ReLU(),
            self.l5
            # 最后一层softmax之后会自动接在损失函数之上
        )

    def forward(self, x):
        x = x.view(-1, 784)  # 把minibatch reshape to [N, 784], 784是每个图像的特征维度
        return self.layers(x)

model = Net_L5()

# construct loss and optimiter

# 包含了softmax层，并且会根据标签类别（即使是多类）,自动构建one-hot计算交叉熵，需要LongTensor类标签
criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.9)

# training and test

loss_list = []
accuracy_list = []


def train(epoch):
    '''某一轮epoch上的训练'''
    epoch_loss = []  # 记录该轮epoch上每个batch的loss
    for batch_idx, batch_data in enumerate(train_loader, 1):
        X, y_label = batch_data
        y_pred = model(X)
        loss = criterion(y_pred, y_label)

        epoch_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = sum(epoch_loss)/len(epoch_loss)
    loss_list.append(average_loss)
    print(f'[epoch]:{epoch},  [average_loss]: {average_loss}')


def test():
    '''在全集合上测试一次准确率'''
    correct_num = 0
    num = len(test_dataset)
    with torch.no_grad():
        for batch_data in test_loader:
            X, y = batch_data
            y_pred = model(X)
            y_pred = torch.argmax(y_pred, dim=1)
            correct_num += torch.sum(y_pred == y).item()
    accuracy = correct_num/num
    accuracy_list.append(accuracy)
    print(f'Current accuracy on the test set is {accuracy}')

# start training now!


num_epochs = 100


for epoch in range(1, num_epochs+1):
    train(epoch)
    test()
