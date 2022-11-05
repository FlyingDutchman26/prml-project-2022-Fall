import paddle
import numpy as np
from fudanPRML.neural_network.Layer.ReLU import ReLU
from fudanPRML.neural_network.Layer.Logistic import Logistic
class Op(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, inputs):
        raise NotImplementedError
    
    
class Linear(Op):
    def __init__(self, input_size, output_size, name, act_fn = Logistic, bias_init=paddle.zeros):
        self.params = {}
        if act_fn == Logistic:
            # 即为Logistic
            self.params['W'] = paddle.normal(mean=0, std= np.sqrt(1/input_size), shape=[input_size, output_size])
            self.params['b'] = bias_init(shape=[1, output_size])
        else:
            # 即为ReLU
            self.params['W'] = paddle.normal(mean=0, std= np.sqrt(2/input_size), shape=[input_size, output_size])
            self.params['b'] = bias_init(shape=[1, output_size]) + 0.01

        self.inputs = None
        self.grads = {}

        self.name = name

    def forward(self, inputs):
        self.inputs = inputs
        outputs = paddle.matmul(self.inputs, self.params['W']) + self.params['b']
        return outputs

    def backward(self, grads):
        """
        输入：
            - grads：损失函数对当前层输出的导数
        输出：
            - 损失函数对当前层输入的导数
        """
        self.grads['W'] = paddle.matmul(self.inputs.T, grads)
        self.grads['b'] = paddle.sum(grads, axis=0)

        # 线性层输入的梯度
        return paddle.matmul(grads, self.params['W'].T)