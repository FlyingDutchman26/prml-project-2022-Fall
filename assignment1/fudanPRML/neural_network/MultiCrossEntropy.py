import paddle


class Op(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, inputs):
        raise NotImplementedError


def softmax(inputs):
    x_max = paddle.max(inputs, axis=1, keepdim=True) #N,1
    x_exp = paddle.exp(inputs - x_max)
    partition = paddle.sum(x_exp, axis=1, keepdim=True) #N,1
    outputs =  x_exp / partition
    return outputs
    

class MultiCrossEntropyLoss(Op):
    def __init__(self):
        self.input = None
        self.predicts = None
        self.log_predicts = None
        self.labels = None
        self.num = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, X, labels):
        """
        输入：
            - X : input：softmax前的值，shape=[N, D]，N为样本数量, D 为 类别数量 
            - labels：真实标签，shape=[N, 1]
        输出：
            - 损失值：shape=[1]
        """
        self.input = X
        self.predicts = softmax(self.input)
        
        x_max= paddle.max(X,axis=1,keepdim=True)
        x_exp = paddle.exp(self.input - x_max)
        self.log_predicts = X - x_max - paddle.log(paddle.sum(x_exp,axis=1, keepdim=True))
        # 使用上述方法可以避免溢出 https://blog.csdn.net/muyuu/article/details/122757470
        # self.log_predicts = paddle.log(self.predicts) # 这是常规方法
        self.labels = labels
        self.num = self.predicts.shape[0]
        
        loss = 0
        for i in range(0, self.num):
            index = self.labels[i]
            loss -= self.log_predicts[i][index]
        return loss / self.num
    
    def backward(self, inputs):
        pass
    
X = paddle.to_tensor([[1,4,3],[1,9,3],[1,10,3]], dtype = 'float32')
y = paddle.to_tensor([1,1,1])

Loss = MultiCrossEntropyLoss()

print(Loss(X,y))
      