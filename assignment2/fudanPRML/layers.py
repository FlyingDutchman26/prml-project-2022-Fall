import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

class QKVAttention(nn.Module):
    def __init__(self, size):
        super(QKVAttention, self).__init__()
        size = torch.tensor([size], dtype=float)
        self.sqrt_size = torch.sqrt(size).to(device)

    def forward(self, Q, K, V, valid_lens) :
        """
        输入：
            - Q：查询向量，shape = [batch_size, seq_len, hidden_size]
            - K：键向量，shape = [batch_size, seq_len, hidden_size]
            - V：值向量，shape = [batch_size, seq_len, hidden_size]
            - valid_lens：序列长度，shape =[batch_size]
        输出：
            - context ：输出矩阵，表示的是注意力的加权平均的结果
        """
        batch_size, seq_len, hidden_size = Q.shape
        # score: [batch_size, seq_len, seq_len]
        score = torch.matmul(Q.to(device), torch.transpose(K,2,1).to(device)) / self.sqrt_size
        # arrange: [1,seq_len],比如seq_len=2, arrange变为 [0, 1]
        arrange = torch.arange((seq_len), dtype=torch.float32).unsqueeze(0).to(device)
        # valid_lens : [batch_size*seq_len, 1]
        valid_lens = valid_lens.unsqueeze(1).to(device)
        # mask [batch_size*seq_len, seq_len]
        mask = arrange < valid_lens
        # mask : [batch_size, seq_len, seq_len]
        mask = torch.reshape(mask, [batch_size, seq_len, seq_len]).to(device)
        # 给mask为False的区域填充-1e9
        # y: [batch_size, seq_len, seq_len]
        y = torch.full(score.shape, -1e9).to(device)
        # score: [batch_size, seq_len,seq_len]
        score = torch.where(mask, score, y).float()
        # attention_weights: [batch_size, seq_len, seq_len]
        attention_weights = F.softmax(score, -1)
        self._attention_weights = attention_weights
        # 加权平均
        # context: [batch_size, seq_len, hidden_size]
        context = torch.matmul(attention_weights, V)
        return context

    @property
    def attention_weights(self):
        return self._attention_weights


 
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, inputs_size, heads_num, dropout=0.0):
        super(MultiHeadSelfAttention, self).__init__()
        # 输入的embedding维度
        self.inputs_size = inputs_size
        self.qsize, self.ksize, self.vsize = inputs_size, inputs_size, inputs_size
        # head的数目
        self.heads_num = heads_num
        # 每个head的输入向量的维度
        self.head_size = inputs_size // heads_num
        # 输入的维度inputs_size需要整除head数目heads_num
        assert (
            self.head_size * heads_num == self.inputs_size
        ), "embed_size must be divisible by heads_num"
        # 查询
        self.Q_proj = nn.Linear(self.qsize, inputs_size, bias=False)
        # 键
        self.K_proj = nn.Linear(self.ksize, inputs_size, bias=False)
        # 值
        self.V_proj = nn.Linear(self.vsize, inputs_size, bias=False)
        # 输出映射
        self.out_proj = nn.Linear(inputs_size, inputs_size, bias=False)
        # QKV注意力
        self.attention = QKVAttention(self.head_size)

    def forward(self, X, valid_lens):
        """
        输入：
            - X：输入矩阵，shape=[batch_size,seq_len,hidden_size]
            - valid_lens： 长度矩阵，shape=[batch_size]
        输出：
            - output：输出矩阵，表示的是多头注意力的结果
        """
        self.batch_size, self.seq_len, self.hidden_size = X.shape
        # Q,K,V: [batch_size, seq_len, hidden_size]
        Q, K, V = self.Q_proj(X), self.K_proj(X), self.V_proj(X)
        # Q,K,V: [batch_size*heads_num, seq_len, head_size]
        Q, K, V = [
            self.split_head_reshape(item, self.heads_num, self.head_size)
            for item in [Q, K, V]
        ]

        # 把valid_lens复制 heads_num * seq_len次
        # 比如valid_lens_np=[1,2],num_head*seq_len=2 则变为 [1,1,2,2] # 是为了attention层的输入
        valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.heads_num * self.seq_len, dim=0)
        # out: [batch_size*heads_num, seq_len, head_size]
        out = self.attention(Q, K, V, valid_lens)
        # out: [batch_size, heads_num, seq_len, head_size]
        out = torch.reshape(
            out, [self.batch_size, self.heads_num, self.seq_len, self.head_size]
        )
        # out: [batch_size, seq_len, heads_num, head_size]
        out = out.transpose(1,2)
        # 多头注意力输出拼接
        # out: [batch_size, seq_len, heads_num * head_size]
        out = torch.reshape(x=out, shape=[out.shape[0],out.shape[1], out.shape[2] * out.shape[3]])
        # 输出映射
        out = self.out_proj(out)
        return out

    def split_head_reshape(self, X, heads_num, head_size):
        """
        输入：
            - X：输入矩阵，shape=[batch_size, seq_len, hidden_size]
        输出：
            - output：输出多头的矩阵，shape= [batch_size * heads_num, seq_len, head_size]
        """
        batch_size, seq_len, hidden_size = X.shape
        # X: [batch_size, seq_len, heads_num, head_size]
        # 多头分组
        X = torch.reshape(x=X, shape=[batch_size, seq_len, heads_num, head_size])
        # X: [batch_size, heads_num, seq_len, head_size]
        # 形状重组
        X = X.transpose(1,2)
        # X: [batch_size*heads_num, seq_len, head_size]
        X = torch.reshape(X, [batch_size * heads_num, seq_len, head_size])
        return X
    
class AveragePooling(nn.Module):
    def __init__(self):
        super(AveragePooling, self).__init__()
    
    def forward(self, sequence_output, sequence_length):
        sequence_length = sequence_length.unsqueeze(-1).float().to(device)
        # 根据sequence_length生成mask矩阵，用于对Padding位置的信息进行mask
        max_len = sequence_output.shape[1]
        mask = torch.arange(max_len).to(device) < sequence_length
        mask = mask.unsqueeze(-1).float().to(device)
        # 对序列中paddling部分进行mask
        sequence_output = torch.multiply(sequence_output, mask).to(device)
        # 对序列中的向量取均值
        batch_mean_hidden = torch.divide(torch.sum(sequence_output, dim=1), sequence_length).to(device)
        return batch_mean_hidden