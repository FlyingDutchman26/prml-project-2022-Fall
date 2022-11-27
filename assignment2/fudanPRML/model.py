import torch
import torch.nn as nn 
import torch.nn.functional as F
from fudanPRML.layers import *

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

class CNNModel(nn.Module):
    def __init__(self,args,init_embedding):
        super(CNNModel,self).__init__()
        self.dropout = nn.Dropout(args['dropout'])
        # self.embedding = nn.Embedding(num_embeddings=len(init_embedding.embedding_matrix),embedding_dim=args['len_feature'],weight = torch.tensor(init_embedding.embedding_matrix)).cuda()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(init_embedding.embedding_matrix),padding_idx=0,freeze=False).to(device)
        # 注意： the entries at :attr:`padding_idx` do not contribute to the gradient
        self.filter_num = args['filter_num']
        self.len_feature = args['len_feature']
        embedding_dim = self.len_feature
        self.branch2xd = nn.Sequential(
            nn.Conv2d(1,self.filter_num,kernel_size=(2,embedding_dim),padding = (1,0)),# 这里pad是为了为了防止某一批次单词数量过少无法做卷积，Padding added to all four sides
            nn.Tanh()
        )
        self.branch3xd = nn.Sequential(
            nn.Conv2d(1,self.filter_num,kernel_size=(3,embedding_dim),padding=(1,0)),
            nn.Tanh()
        )
        self.branch4xd = nn.Sequential(
            nn.Conv2d(1,self.filter_num,kernel_size=(4,embedding_dim),padding = (2,0)),
            nn.Tanh()
        )
        self.branch5xd = nn.Sequential(
            nn.Conv2d(1,self.filter_num,kernel_size=(5,embedding_dim),padding=(2,0)),
            nn.Tanh()
        )
        self.fc = nn.Linear(4*self.filter_num,args['num_class'])
        
    def forward(self,X):
        # 接收某一batch的输入 [b,l] b:batch_size, l:seq_length (after padding in this batch)
        X = self.embedding(X).view(X.shape[0], 1, X.shape[1], self.len_feature) #[b,1,h,w] h:即为seq_length, w:len_feature:300
        X = self.dropout(X)
        conv1 = self.branch2xd(X).squeeze(3) #[b,filter_num,h,1] 然后丢弃最后一个维度(3) 之后再进行最大池化(池化会针对维度h)
        conv2 = self.branch3xd(X).squeeze(3)
        conv3 = self.branch4xd(X).squeeze(3)
        conv4 = self.branch5xd(X).squeeze(3)
        pool1 = F.max_pool1d(conv1, conv1.shape[2])
        pool2 = F.max_pool1d(conv2, conv2.shape[2])
        pool3 = F.max_pool1d(conv3, conv3.shape[2])
        pool4 = F.max_pool1d(conv4, conv4.shape[2])
        # 经过poolinig, 现在是 [b,filter_num,1]
        pool = torch.cat([pool1, pool2, pool3, pool4], 1).squeeze(2) 
        out_put = self.fc(pool)
        return out_put
        
        
class BiLSTM_Model_V1(nn.Module):
    def __init__(self,args,init_embedding,num_layers=1,bidirectional = True):
        super(BiLSTM_Model_V1,self).__init__()
        self.hidden_size = args['len_feature']
        self.num_layers = num_layers
        self.n_direction = 2 if bidirectional else 1
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(init_embedding.embedding_matrix),padding_idx=0,freeze=False).to(device)
        self.dropout = nn.Dropout(args['dropout'])
        self.lstm = nn.LSTM(input_size = self.hidden_size,hidden_size = self.hidden_size,num_layers=self.num_layers,bidirectional=bidirectional,batch_first = True,dropout = 0.1)
        # RNN层的输入格式 [input,h_0] 隐藏层初始状态h0可以自己给定(default zeros) 然而这个不能指定batch_first !
        # 已经指定 batch_first = True, 因此input维度为[B,L,M] = [batch_size,seq_length,input_dim(len_feature)]
        # 一般的 RNN 的输出有两项[ouput,h_n],  LSTM比较特殊，是 [output, (h_n, c_n)],output即为输出结果，h_n指最终全部隐藏层状态
        # 指定batch_first后，output是所有每次输出[B,L,hidden * bidirectional]
        # 最终全部隐藏层状态 h_n 维度为[bidirectional*num_layers,batch_size,hidden_size] 不能指定batch_first
        self.fc = nn.Linear(self.n_direction*self.hidden_size,args['num_class']) # 和layer无关了，就是最后一层的最终隐藏层
        
    def forward(self,input):
        # input : [batch_size, seq_length] : 尚未embedding
        batch_size = input.shape[0]
        h_0 = torch.zeros(self.n_direction*self.num_layers,batch_size,self.hidden_size).to(device) # 这个不能batch_first 有点坑
        c_0 = torch.zeros(self.n_direction*self.num_layers,batch_size,self.hidden_size).to(device)
        lstm_input = self.embedding(input)
        _,(h_n,_) = self.lstm(lstm_input,(h_0,c_0))
        # 注意！！！ batch_first : does not apply to hidden or cell states! 坑比！
        # h_n: [bidirectional*num_layers,batch_size,hidden_size]
        if self.n_direction == 2:
            final_hidden = torch.cat([h_n[-1],h_n[-2]],dim = 1) # 双向，则取最后一层layer的最后一个bi-lstm隐藏层输出拼接(沿着hidden_size方向拼)
        else:
            final_hidden = h_n[-1] # 非双向，就要最后一个
        
        output = self.fc(final_hidden)
        return output

class Model_LSTMSelftAttention(nn.Module):
    def __init__(self,args,init_embedding,num_layers=1,bidirectional = True,attention = None):
        super(Model_LSTMSelftAttention,self).__init__()
        self.hidden_size = args['len_feature']
        self.num_layers = num_layers
        self.n_direction = 2 if bidirectional else 1
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(init_embedding.embedding_matrix),padding_idx=0,freeze=False).to(device)
        self.dropout = nn.Dropout(args['dropout'])
        self.lstm = nn.LSTM(input_size = self.hidden_size,hidden_size = self.hidden_size,num_layers=self.num_layers,bidirectional=bidirectional,batch_first = True,dropout = 0.1)
        # RNN层的输入格式 [input,h_0] 隐藏层初始状态h0可以自己给定(default zeros) 然而这个不能指定batch_first !
        # 已经指定 batch_first = True, 因此input维度为[B,L,M] = [batch_size,seq_length,input_dim(len_feature)]
        # 一般的 RNN 的输出有两项[ouput,h_n],  LSTM比较特殊，是 [output, (h_n, c_n)],output即为输出结果，h_n指最终全部隐藏层状态
        # 指定batch_first后，output是所有每次输出[B,L,hidden * bidirectional]
        # 最终全部隐藏层状态 h_n 维度为[bidirectional*num_layers,batch_size,hidden_size] 不能指定batch_first
        self.attention = attention
        self.average_layer = AveragePooling()
        self.fc = nn.Linear(self.n_direction*self.hidden_size,args['num_class']) # 和layer无关了，就是最后一层的最终隐藏层
        
    def forward(self,input,valid_lens):
        # input : [batch_size, seq_length] : 尚未embedding
        batch_size = input.shape[0]
        valid_lens = valid_lens
        h_0 = torch.zeros(self.n_direction*self.num_layers,batch_size,self.hidden_size).to(device) # 这个不能batch_first 有点坑
        c_0 = torch.zeros(self.n_direction*self.num_layers,batch_size,self.hidden_size).to(device)
        lstm_input = self.embedding(input)
        output,(h_n,c_n) = self.lstm(lstm_input,(h_0,c_0))
        # 注意！！！ batch_first : does not apply to hidden or cell states! 坑比！
        # h_n: [bidirectional*num_layers,batch_size,hidden_size]
        output = self.attention(output,valid_lens)
        output = self.average_layer(output, valid_lens)
        output = self.fc(output)
        return output   