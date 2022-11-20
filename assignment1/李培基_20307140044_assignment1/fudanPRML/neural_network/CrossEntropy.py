import paddle
from paddle.nn.functional import one_hot


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


class BinaryCrossEntropyLoss(Op):
    def __init__(self, model):
        self.predicts = None
        self.labels = None
        self.num = None

        self.model = model

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        输入：
            - predicts：预测值，shape=[N, 1]，N为样本数量
            - labels：真实标签，shape=[N, 1]
        输出：
            - 损失值：shape=[1]
        """
        self.predicts = predicts
        self.labels = labels
        self.num = self.predicts.shape[0]
        loss = -1. / self.num * (paddle.matmul(self.labels.t(), paddle.log(self.predicts)) 
                + paddle.matmul((1-self.labels.t()), paddle.log(1-self.predicts)))

        loss = paddle.squeeze(loss, axis=1)
        return loss

    def backward(self):
        # 计算损失函数对模型预测的导数
        loss_grad_predicts = -1.0 * (self.labels / self.predicts - 
                       (1 - self.labels) / (1 - self.predicts)) / self.num
        # 这里我搞懂啦！
        
        # 梯度反向传播
        self.model.backward(loss_grad_predicts)

class MultiCrossEntropyLoss(Op):
    def __init__(self,model = None):
        self.input = None
        self.predicts = None
        self.log_predicts = None
        self.labels = None
        self.num = None
        self.model = model

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward_init(self, X, labels):
        """
        输入：
            - X : input：softmax前的值，shape=[N, D]，N为样本数量, D 为 类别数量 
            - labels：真实标签，shape=[N, 1]
        输出：
            - 损失值：shape=[1]
        """
        # 这是paddle书的版本，居然loss是用循环计算的，特别慢
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
    
    def forward(self, X, labels):
        """
        输入：
            - X : input：softmax前的值，shape=[N, D]，N为样本数量, D 为 类别数量 
            - labels：真实标签，shape=[N, 1]
        输出：
            - 损失值：shape=[1]
        """
        # 这是我自己优化后的用矩阵的版本，非常快
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
        one_hot_labels = one_hot(self.labels,num_classes=10).t()
        loss =  -1/self.num * paddle.sum( paddle.diag(paddle.matmul(self.log_predicts,one_hot_labels)) )
        return loss
    
    def backward_init(self):
        # 自己推导的版本 是对的但是慢
        loss_grads_inputs = paddle.zeros(shape = self.input.shape,dtype = 'float32')
        for n in range(0,self.num):
            # print('debug 1:',self.predicts[n])
            grad_predicts_inputs_n = paddle.diag(self.predicts[n]) - paddle.matmul(paddle.unsqueeze(self.predicts[n],axis=1),paddle.unsqueeze(self.predicts[n],axis=0))
            # print('debug 1.5:',paddle.matmul(paddle.unsqueeze(self.predicts[n],axis=1),paddle.unsqueeze(self.predicts[n],axis=0)))
            # print('debug:2',grad_predicts_inputs_n)
            index = self.labels[n]
            loss_grads_inputs[n] = -1/(self.num*self.predicts[n][index]) * grad_predicts_inputs_n[index]
        self.model.backward(loss_grads_inputs)
        #return loss_grads_inputs
    
    def backward(self):
        # 网上推导的快速版本，神奇
        one_hot_labels = one_hot(self.labels,num_classes=10)
        loss_grads_inputs = 1/(self.num)*(self.predicts - one_hot_labels)
        self.model.backward(loss_grads_inputs)
        #return loss_grads_inputs





# 测试部分(需要去掉model输入，然后让函数返回值)

# X = paddle.to_tensor([[0,1,2,3,4,5,6,7,8,15],[0,1,2,3,4,5,6,7,8,9]], dtype = 'float32')
# y = paddle.to_tensor([0,3])

# Loss = MultiCrossEntropyLoss()

# print(Loss(X,y))
# print(Loss.predicts)
# print(Loss.backward_init())
# print(Loss.backward())

