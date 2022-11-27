import torch
def split_head_reshape(X, heads_num, head_size):
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