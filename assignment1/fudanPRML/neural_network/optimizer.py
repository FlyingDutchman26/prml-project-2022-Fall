import paddle

class Optimizer(object):
    def __init__(self, init_lr, model):
        """
        优化器类初始化
        """
        #初始化学习率，用于参数更新的计算
        self.init_lr = init_lr
        #指定优化器需要优化的模型
        self.model = model

    def step(self):
        """
        定义每次迭代如何更新参数
        """
        pass
    
    
class BatchGD(Optimizer):
    def __init__(self, init_lr, model):
        super(BatchGD, self).__init__(init_lr=init_lr, model=model)

    def step(self):
        # 参数更新
        for layer in self.model.layers: # 遍历所有层
            if isinstance(layer.params, dict):
                for key in layer.params.keys():
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]