import paddle
import numpy as np


class Op(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, inputs):
        raise NotImplementedError
    
    
class ReLU(Op):
    def __init__(self):
        self.mask = None
        self.params = None
            
    def forward(self, X):
        # self.mask = (X > 0)
        # return paddle.multiply(X,self.mask)
        return paddle.maximum(X, paddle.to_tensor(0.))

    def backward(self, grads):
        '''
        输入grads, 为最终的loss对本层输出的梯度
        
        此函数输出为最终loss对本层输入的梯度
        '''

        return paddle.maximum(grads,paddle.to_tensor(0.))

class Leaky_ReLU(Op):
    def __init__(self,negative_slope = 0.1):
        self.negative_slope = negative_slope
        self.params = None

    def forward(self, X):
        # 当前版本paddle暂不支持直接将bool类型转成int类型，因此调用了paddle的cast函数来进行显式转换
        a1 = (paddle.cast((X > 0), dtype='float32') * X) 
        a2 = (paddle.cast((X <= 0), dtype='float32') * (self.negative_slope * X))
        return a1 + a2

    def backward(self, grads):
        '''
        输入grads, 为最终的loss对本层输出的梯度
        
        此函数输出为最终loss对本层输入的梯度
        '''

        return paddle.maximum(grads,paddle.to_tensor(0.)) + self.negative_slope*paddle.minimum(grads,paddle.to_tensor(0.))