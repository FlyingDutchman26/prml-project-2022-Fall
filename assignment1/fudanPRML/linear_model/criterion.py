import paddle
from paddle.nn.functional import one_hot

# 多类别交叉熵损失函数
class MultiCrossEntropyLoss:
    def __init__(self):
        self.predicts = None
        self.log_predicts = None
        self.labels = None
        self.num = None

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
        self.log_predicts = paddle.log(self.predicts)
        self.labels = labels
        self.num = self.predicts.shape[0]
        loss = 0
        one_hot_labels = one_hot(self.labels,num_classes=10).t()
        loss =  -1/self.num * paddle.sum( paddle.diag(paddle.matmul(self.log_predicts,one_hot_labels)) )
        return loss
    
        # 这是原始版本，没有并行性
        # for i in range(0, self.num):
        #     index = self.labels[i]
        #     loss -= paddle.log(self.predicts[i][index])
        # return loss / self.num