import paddle
import numpy as np
import time

class Op(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, inputs):
        raise NotImplementedError

class Dropout(Op):
    def __init__(self,p = 0.5):
        self.mask = None
        self.unmask = None
        self.params = None
        self.p = p
            
    def forward(self, X):
        self.mask = (paddle.rand(X.shape) > self.p)
        self.mask = self.mask.cast('float32')
        return self.mask * X / (1.0 - self.p)

    def backward(self, grads):
        '''
        输入grads, 为最终的loss对本层输出的梯度
        
        此函数输出为最终loss对本层输入的梯度
        '''

        return paddle.multiply(grads,self.mask) /(1.0 - self.p)