import paddle
class SoftmaxClassifier():
    def __init__(self, input_dim, output_dim):
        self.params = {}
        # 将线性层的权重参数全部初始化为0
        self.params['W'] = paddle.zeros(shape=[input_dim, output_dim])
        #self.params['W'] = paddle.normal(mean=0, std=0.01, shape=[input_dim, output_dim])
        # 将线性层的偏置参数初始化为0
        self.params['b'] = paddle.zeros(shape=[output_dim])
        # 存放参数的梯度
        self.grads = {}
        self.X = None
        self.outputs = None
        self.output_dim = output_dim

    def __call__(self, inputs):
        return self.forward(inputs)

    
    def softmax(self,X):
        """
        输入：
            - X：shape=[N, C]，N为向量数量，C为向量维度
        
        此函数进行了防止溢出的操作
        """
        x_max = paddle.max(X, axis=1, keepdim=True)#N,1
        x_exp = paddle.exp(X - x_max)
        partition = paddle.sum(x_exp, axis=1, keepdim=True)#N,1
        return x_exp / partition
    
    def forward(self, inputs):
        self.X = inputs
        # 线性计算
        score = paddle.matmul(self.X, self.params['W']) + self.params['b']
        # Softmax 函数
        self.outputs = self.softmax(score)
        return self.outputs

    def backward(self, labels):
        """
        输入：
            - labels：真实标签，shape=[N, 1]，其中N为样本数量
        """
        # 计算偏导数
        N = labels.shape[0]
        labels = paddle.nn.functional.one_hot(labels, self.output_dim)
        self.grads['W'] = -1 / N * \
            paddle.matmul(self.X.t(), (labels-self.outputs))
        self.grads['b'] = -1 / N * \
            paddle.matmul(paddle.ones(shape=[N]), (labels-self.outputs))
