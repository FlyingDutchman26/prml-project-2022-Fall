import torch

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

class Runner_V1():
    def __init__(self,model,optimizer,loss_fn):
        self.model =model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        # 记录训练过程中的评估指标变化情况
        self.train_scores = []
        self.dev_scores = []

        # 记录训练过程中的评价指标变化情况
        self.train_loss = []
        self.dev_loss = []
        
    def train(self,train_loader,valid_loader,num_epoch=3):
        self.model.train()
        step = 0
        best_accuracy = 0
        for epoch in range(1,num_epoch+1):
            for batch_id, (X,labels,valid_lens) in enumerate(train_loader):
                self.model.train()
                out = self.model(X)
                loss = self.loss_fn(out,labels)
                self.train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                with torch.no_grad():
                    out = torch.argmax(out,dim=1)
                    score = (out == labels).sum()/len(labels)
                    self.train_scores.append(score.item())
                valid_accuracy = self.evaluate(valid_loader)
                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    self.save_model()
                    print(f'Best performance on valid set upgraded: accuracy: {best_accuracy}')
                step += 1
                if step%10 == 0:
                    print(f'[epoch]:{epoch},[step]:{step},[loss]:{loss},[score]:{score}')
                    

                


    @torch.no_grad()
    def evaluate(self,valid_loader):
        self.model.eval()
        correct = 0
        total = 0
        loss = 0
        cnt = 0
        for batch_id, (X,labels,valid_lens) in enumerate(valid_loader):
            out = self.model(X)
            loss += self.loss_fn(out,labels).item()
            out = torch.argmax(out,dim=1)
            correct += (out == labels).sum().item()
            total += len(labels)
            cnt += 1
        score = correct/total
        loss = loss/cnt
        self.dev_scores.append(score)
        self.dev_loss.append(loss)
        return score
        
    @torch.no_grad()
    def predict(self,test_loader):
        self.load_model()
        self.model.eval()
        correct = 0
        total = 0
        for batch_id, (X,labels,valid_lens) in enumerate(test_loader):
            out = self.model(X)
            out = torch.argmax(out,dim=1)
            correct += (out == labels).sum().item()
            total += len(labels)
        score = correct/total
        # print(total)
        print(f'Score on test set:{score}')
        return score
    
    def save_model(self, save_path = './modelparams/bestmodel_parms.pth'):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, model_path='./modelparams/bestmodel_parms.pth'):
        self.model.load_state_dict(torch.load(model_path))
        

class Runner_V2():
    def __init__(self,model,optimizer,loss_fn):
        self.model =model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        # 记录训练过程中的评估指标变化情况
        self.train_scores = []
        self.dev_scores = []

        # 记录训练过程中的评价指标变化情况
        self.train_loss = []
        self.dev_loss = []
        
    def train(self,train_loader,valid_loader,num_epoch=3):
        self.model.train()
        step = 0
        best_accuracy = 0
        for epoch in range(1,num_epoch+1):
            for batch_id, (X,labels,valid_lens) in enumerate(train_loader):
                self.model.train()
                out = self.model(X,valid_lens)
                loss = self.loss_fn(out,labels)
                self.train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                with torch.no_grad():
                    out = torch.argmax(out,dim=1)
                    score = (out == labels).sum()/len(labels)
                    self.train_scores.append(score.item())
                valid_accuracy = self.evaluate(valid_loader)
                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    self.save_model()
                    print(f'Best performance on valid set upgraded: accuracy: {best_accuracy}')
                step += 1
                if step%10 == 0:
                    print(f'[epoch]:{epoch},[step]:{step},[loss]:{loss},[score]:{score}')
                
                


    @torch.no_grad()
    def evaluate(self,valid_loader):
        self.model.eval()
        correct = 0
        total = 0
        loss = 0
        cnt = 0
        for batch_id, (X,labels,valid_lens) in enumerate(valid_loader):
            out = self.model(X,valid_lens)
            loss += self.loss_fn(out,labels).item()
            out = torch.argmax(out,dim=1)
            correct += (out == labels).sum().item()
            total += len(labels)
            cnt += 1
        score = correct/total
        loss = loss/cnt
        self.dev_scores.append(score)
        self.dev_loss.append(loss)
        return score
        
    @torch.no_grad()
    def predict(self,test_loader):
        self.load_model()
        self.model.eval()
        correct = 0
        total = 0
        for batch_id, (X,labels,valid_lens) in enumerate(test_loader):
            out = self.model(X,valid_lens)
            out = torch.argmax(out,dim=1)
            correct += (out == labels).sum().item()
            total += len(labels)
        score = correct/total
        # print(total)
        print(f'Score on test set:{score}')
        return score
    
    def save_model(self, save_path = './modelparams/bestmodel_parms.pth'):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, model_path='./modelparams/bestmodel_parms.pth'):
        self.model.load_state_dict(torch.load(model_path))