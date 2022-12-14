import paddle
from fudanPRML.neural_network.Layer.Linear import Linear
from fudanPRML.neural_network.Layer.Logistic import Logistic
from fudanPRML.neural_network.CrossEntropy import MultiCrossEntropyLoss
from fudanPRML.neural_network.Layer.Dropout import Dropout
class Op(object):
    def __init__(self):
        pass

    def __call__(self, inputs, **kwargs):
        return self.forward(inputs)

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, inputs):
        raise NotImplementedError
    
class Model_MLP_L2(Op):
    def __init__(self, input_size, hidden_size, output_size,act_fn = Logistic):
        # 线性层
        self.fc1 = Linear(input_size, hidden_size, name="fc1",act_fn = act_fn)
        # Logistic激活函数层
        self.act_fn1 = act_fn()
        self.fc2 = Linear(hidden_size, output_size, name="fc2",act_fn = act_fn)

        self.layers = [self.fc1, self.act_fn1, self.fc2]

    def __call__(self, X,**kwargs):
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
        
class Model_MLP_L5(Op):
    def __init__(self, input_size, output_size = 10,act_fn = Logistic):
        # Logistic激活函数层
        self.act_fn1 = act_fn()
        self.act_fn2 = act_fn()
        self.act_fn3 = act_fn()
        self.act_fn4 = act_fn()
        # 线性层
        self.fc1 = Linear(input_size, 512, name="fc1",act_fn=act_fn)
        self.fc2 = Linear(512, 256, name="fc2",act_fn = act_fn)
        self.fc3 = Linear(256,128,name= 'fc3',act_fn = act_fn)
        self.fc4 = Linear(128,64,name='fc4',act_fn = act_fn)
        self.fc5 = Linear(64,output_size,name='fc5',act_fn = act_fn)

        self.layers = [self.fc1,self.fc2,self.fc3,self.fc4,self.fc5,self.fc5]

    def __call__(self, X,**kwargs):
        return self.forward(X)

    # 前向计算
    def forward(self, X):
        z1 = self.fc1(X)
        a1 = self.act_fn1(z1)
        z2 = self.fc2(a1)
        a2 = self.act_fn2(z2)
        z3 = self.fc3(a2)
        a3 = self.act_fn3(z3)
        z4 = self.fc4(a3)
        a4 = self.act_fn4(z4)
        z5 = self.fc5(a4)
        
        return z5
        
    # 反向计算
    def backward(self, loss_grad_z5):
        '''
        输入：loss对本模块输出的导数
        
        在本模块中，逐层反向传播，同时计算并更新loss对本模块中参数的梯度
        '''

        loss_grad_a4 = self.fc5.backward(loss_grad_z5)
        loss_grad_z4 = self.act_fn4.backward(loss_grad_a4)
        loss_grad_a3 = self.fc4.backward(loss_grad_z4)
        loss_grad_z3 = self.act_fn3.backward(loss_grad_a3)
        loss_grad_a2 = self.fc3.backward(loss_grad_z3)
        loss_grad_z2 = self.act_fn2.backward(loss_grad_a2)
        loss_grad_a1 = self.fc2.backward(loss_grad_z2)
        loss_grad_z1 = self.act_fn1.backward(loss_grad_a1)
        loss_grad_inputs = self.fc1.backward(loss_grad_z1)
        
        
class Model_MLP_Dropout_L5(Op):
    def __init__(self, input_size, output_size = 10,act_fn = Logistic):
        # Logistic激活函数层
        self.act_fn1 = act_fn()
        self.act_fn2 = act_fn()
        self.act_fn3 = act_fn()
        self.act_fn4 = act_fn()
        # 线性层
        self.fc1 = Linear(input_size, 512, name="fc1",act_fn=act_fn)
        self.fc2 = Linear(512, 256, name="fc2",act_fn = act_fn)
        self.fc3 = Linear(256,128,name= 'fc3',act_fn = act_fn)
        self.fc4 = Linear(128,64,name='fc4',act_fn = act_fn)
        self.fc5 = Linear(64,output_size,name='fc5',act_fn = act_fn)
        
        # Dropout 层
        self.dropout1 = Dropout(p = 0.2)
        self.dropout2 = Dropout(p = 0.5)

        self.layers = [self.fc1,self.fc2,self.fc3,self.fc4,self.fc5,self.fc5]


    def __call__(self, X, **kwargs):
        mode = kwargs.get("mode",'evaluate')
        return self.forward(X, mode)

    # 前向计算
    def forward(self, X, mode):
        if mode == 'train':
            z1 = self.fc1(X)
            a1 = self.act_fn1(z1)
            d1 = self.dropout1(a1)
            z2 = self.fc2(d1)
            a2 = self.act_fn2(z2)
            z3 = self.fc3(a2)
            a3 = self.act_fn3(z3)
            z4 = self.fc4(a3)
            a4 = self.act_fn4(z4)
            d2 = self.dropout2(a4)
            z5 = self.fc5(d2)
        else:
            z1 = self.fc1(X)
            a1 = self.act_fn1(z1)
            z2 = self.fc2(a1)
            a2 = self.act_fn2(z2)
            z3 = self.fc3(a2)
            a3 = self.act_fn3(z3)
            z4 = self.fc4(a3)
            a4 = self.act_fn4(z4)
            z5 = self.fc5(a4)
            
        return z5
        
    # 反向计算
    def backward(self, loss_grad_z5):
        '''
        输入：loss对本模块输出的导数
        
        在本模块中，逐层反向传播，同时计算并更新loss对本模块中参数的梯度
        '''

        loss_grad_d2 = self.fc5.backward(loss_grad_z5)
        loss_grad_a4 = self.dropout2.backward(loss_grad_d2)
        loss_grad_z4 = self.act_fn4.backward(loss_grad_a4)
        loss_grad_a3 = self.fc4.backward(loss_grad_z4)
        loss_grad_z3 = self.act_fn3.backward(loss_grad_a3)
        loss_grad_a2 = self.fc3.backward(loss_grad_z3)
        loss_grad_z2 = self.act_fn2.backward(loss_grad_a2)
        loss_grad_d1 = self.fc2.backward(loss_grad_z2)
        loss_grad_a1 = self.dropout1.backward(loss_grad_d1)
        loss_grad_z1 = self.act_fn1.backward(loss_grad_a1)
        loss_grad_inputs = self.fc1.backward(loss_grad_z1)