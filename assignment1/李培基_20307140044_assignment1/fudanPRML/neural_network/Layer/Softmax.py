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
    

class Softmax(Op):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.params = None

    def forward(self, inputs):
        x_max = paddle.max(inputs, axis=1, keepdim=True) #N,1
        x_exp = paddle.exp(inputs - x_max)
        partition = paddle.sum(x_exp, axis=1, keepdim=True) #N,1
        outputs =  x_exp / partition
        self.outputs = outputs
        return outputs

    def backward(self, grads):
        '''
        输入grads, 为最终的loss对本层输出的梯度
        
        此函数输出为最终loss对本层输入的梯度
        '''
        # 计算Softmax对输入的导数
        outputs_grad_inputs = paddle.multiply(self.outputs, (1.0 - self.outputs))
        # 注意，此处是multiply，即为逐元素相乘，不是matmul
        return paddle.multiply(grads,outputs_grad_inputs)