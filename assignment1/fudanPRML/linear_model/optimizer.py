import paddle

# 优化器:简单的批梯度下降
class SimpleBatchGD:
    def __init__(self, init_lr, model):
        self.init_lr = init_lr
        self.model = model

    def step(self):
        #参数更新
        #遍历所有参数，按照公式(3.8)和(3.9)更新参数
        if isinstance(self.model.params, dict):
            for key in self.model.params.keys():
                self.model.params[key] = self.model.params[key] - self.init_lr * self.model.grads[key]