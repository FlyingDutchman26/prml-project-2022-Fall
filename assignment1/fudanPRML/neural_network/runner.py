import paddle
import time
import os

class RunnerV2_1(object):
    def __init__(self, model, optimizer, metric, loss_fn, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric

        # 记录训练过程中的评估指标变化情况
        self.train_scores = []
        self.dev_scores = []

        # 记录训练过程中的评价指标变化情况
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):
        # 传入训练轮数，如果没有传入值则默认为0
        num_epochs = kwargs.get("num_epochs", 0)
        # 传入log打印频率，如果没有传入值则默认为100
        log_epochs = kwargs.get("log_epochs", 100)

        # 传入模型保存路径
        save_dir = kwargs.get("save_dir", None)
        
        # 记录全局最优指标
        best_score = 0
        # 进行num_epochs轮训练
        for epoch in range(1,num_epochs+1):
            start_epoch = time.perf_counter()
            
            X, y = train_set
            # 获取模型预测
            start = time.perf_counter()
            logits = self.model(X)
            end = time.perf_counter()
            print('前向计算时间',end-start,'second')
            
            # 计算交叉熵损失
            start = time.perf_counter()
            trn_loss = self.loss_fn(logits, y) # return a tensor
            self.train_loss.append(trn_loss.item())
            end = time.perf_counter()
            print('计算交叉熵损失时间',end-start,'second')
            
            # 计算评估指标
            trn_score = self.metric(logits, y).item()
            self.train_scores.append(trn_score)

            start = time.perf_counter()
            self.loss_fn.backward()
            end = time.perf_counter()
            print('反向传播时间',end-start,'second')
            
            # 参数更新
            self.optimizer.step()
           
            dev_score, dev_loss = self.evaluate(dev_set)
            # 如果当前指标为最优指标，保存该模型
            if dev_score > best_score:
                print(f"[Evaluate] best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
                if save_dir:
                    self.save_model(save_dir)

            if log_epochs and epoch % log_epochs == 0:
                print(f"[Train] epoch: {epoch}/{num_epochs}, loss: {trn_loss.item()}")
            
            end_epoch = time.perf_counter()
            print('本轮epoch训练时间:',end_epoch-start_epoch,'second')    
                
    def evaluate(self, data_set):
        X, y = data_set
        # 计算模型输出
        logits = self.model(X)
        # 计算损失函数
        loss = self.loss_fn(logits, y).item()
        self.dev_loss.append(loss)
        # 计算评估指标
        score = self.metric(logits, y).item()
        self.dev_scores.append(score)
        return score, loss
    
    def predict(self, X):
        return self.model(X)

    def save_model(self, save_dir):
        # 对模型每层参数分别进行保存，保存文件名称与该层名称相同
        for layer in self.model.layers: # 遍历所有层
            if isinstance(layer.params, dict):
                paddle.save(layer.params, os.path.join(save_dir, layer.name+".pdparams"))

    def load_model(self, model_dir):
        # 获取所有层参数名称和保存路径之间的对应关系
        model_file_names = os.listdir(model_dir)
        name_file_dict = {}
        for file_name in model_file_names:
            name = file_name.replace(".pdparams","")
            name_file_dict[name] = os.path.join(model_dir, file_name)

        # 加载每层参数
        for layer in self.model.layers: # 遍历所有层
            if isinstance(layer.params, dict):
                name = layer.name
                file_path = name_file_dict[name]
                layer.params = paddle.load(file_path)