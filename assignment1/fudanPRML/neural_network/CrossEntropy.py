from re import L
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
    def __init__(self,model):
        self.input = None
        self.predicts = None
        self.log_predicts = None
        self.labels = None
        self.num = None
        self.model = model

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
    
    def backward(self):
        loss_grads_inputs = paddle.zeros(shape = self.input.shape,dtype = 'float32')
        for n in range(0,self.num):
            # print('debug 1:',self.predicts[n])
            grad_predicts_inputs_n = paddle.diag(self.predicts[n]) - paddle.matmul(paddle.unsqueeze(self.predicts[n],axis=1),paddle.unsqueeze(self.predicts[n],axis=0))
            # print('debug 1.5:',paddle.matmul(paddle.unsqueeze(self.predicts[n],axis=1),paddle.unsqueeze(self.predicts[n],axis=0)))
            # print('debug:2',grad_predicts_inputs_n)
            index = self.labels[n]
            loss_grads_inputs[n] = -1/(self.num*self.predicts[n][index]) * grad_predicts_inputs_n[index]
        self.model.backward(loss_grads_inputs)
        # return loss_grads_inputs


# 测试部分(无model)

X = paddle.to_tensor([[1,10,3],[1,2,9]], dtype = 'float32')
y = paddle.to_tensor([1,2])

Loss = MultiCrossEntropyLoss()

print(Loss(X,y))
print(Loss.predicts)
print(Loss.backward())

