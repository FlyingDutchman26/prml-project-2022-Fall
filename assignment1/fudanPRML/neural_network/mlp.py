import paddle
from fudanPRML.neural_network.Layer.Linear import Linear
from fudanPRML.neural_network.Layer.Logistic import Logistic
from fudanPRML.neural_network.CrossEntropy import MultiCrossEntropyLoss
class Op(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, inputs):
        raise NotImplementedError
    
class Model_MLP_L2(Op):
    def __init__(self, input_size, hidden_size, output_size):
        # 线性层
        self.fc1 = Linear(input_size, hidden_size, name="fc1")
        # Logistic激活函数层
        self.act_fn1 = Logistic()
        self.fc2 = Linear(hidden_size, output_size, name="fc2")

        self.layers = [self.fc1, self.act_fn1, self.fc2]

    def __call__(self, X):
        return self.forward(X)

    # 前向计算
    def forward(self, X):
        z1 = self.fc1(X)
        a1 = self.act_fn1(z1)
        z2 = self.fc2(a1)
        return z2
        
    # 反向计算
    def backward(self, loss_grad_z2):
        '''
        输入：loss对本模块输出的导数
        
        在本模块中，逐层反向传播，同时计算并更新loss对本模块中参数的梯度
        '''

        loss_grad_a1 = self.fc2.backward(loss_grad_z2)
        loss_grad_z1 = self.act_fn1.backward(loss_grad_a1)
        loss_grad_inputs = self.fc1.backward(loss_grad_z1)