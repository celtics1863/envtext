from .model_base import ModelBase
import torch
from torch.utils.data import TensorDataset,DataLoader
from tqdm import tqdm
from collections import defaultdict
import numpy as np

class RNNBase(ModelBase):
    def __init__(self):
        super().__init__()
        self.train_reports = defaultdict(list)
        self.valid_reports = defaultdict(list)
        self.key_metric = 'validation loss'
        self.optim = None
        self.schedule = None
        
    
    def get_train_reports(self):
        '''
        获得训练集metrics报告
        '''
        return self._report(self.train_reports)
    
    def get_valid_reports(self):
        '''
        获得验证集metrics报告
        '''
        return self._report(self.valid_reports)
    
    def set_optim(self,optim, learning_rate = 1e-3):
        '''
        设置优化器，
        Args:
            optim `str` or `torch.optim`:
                默认是Adam优化器，learning rate在训练时传入
                用此接口设置优化器后，不必传入learning rate
                optim == 'SDG': 
                    使用SDG with momentum优化器
                optim == 'Adam':
                    使用Adam优化器
        '''
        self.optim = optim
        
    def set_learing_rate_schedule(self,schedule):
        '''
        设置学习率迭代方法，默认是torch.optim.lr_scheduler.CosineAnnealingLR
        '''
        self.schedule = schedule
    
    def load_checkpoint(self,path):
        '''
        从checkpoint导入模型，但是必须要重新设置
        '''
        checkpoint = torch.load(path)
        self.model = checkpoint['best_model'].to(self.device)
        print('epoch number and learning rate of checkpoint is {} and {:.4e}'.format(checkpoint['epoch'],checkpoint['learning rate']))
        

    def load(self,path):
        '''
        Args:
            path `str`:
                模型pytorch_model.bin所在文件夹
        '''
        self.model = torch.load(os.path.join(path,'pytorch_model.bin'))
        self.model = self.model.to(self.device)
              
    def _update_train_reports(self,report):
        for k,v in report.items():
            self.train_reports[k].append(v)

    def _update_valid_reports(self,report):
        for k,v in report.items():
            self.valid_reports[k].append(v)
            
    def _tokenizer_for_training(self,texts, des = "Tokenizing进度"):
        bar = tqdm(texts)
        bar.set_description(des)
        tokens = []
        for text in bar:
            tokens.append(self.tokenizer(text)[0])
        tokens = torch.tensor(tokens)
        return tokens
    
    def _train_per_step(self,train_dataloader):
        self.model.train()
        bar = tqdm(train_dataloader)
        preds,labels,losses = [],[],[]
        for X,label in bar:
            #move to device
            X = X.to(self.device)
            label = label.to(self.device)
            #更新模型参数
            self.optim.zero_grad()
            loss,predict = self.model(X,label)
            loss.backward()
            self.optim.step()
            self.schedule.step()
            #打印报告
            preds.append(predict.clone().detach().cpu().numpy())
            labels.append(label.clone().detach().cpu().numpy())
            losses.append(loss.clone().detach().cpu().item())
            bar.set_description("Train Loss is {:.4f}".format(losses[-1]))
        preds = np.concatenate(preds,axis = 0)
        labels = np.concatenate(labels,axis = 0)
        loss = sum(losses)/len(bar)
        
        report = self.compute_metrics((preds,labels))
        if isinstance(report,dict):
            report['training loss'] = loss
        elif isinstance(report,float):
            report = {'metric':report,'training loss':loss}
        else:
            report = {'training loss':loss}
        return report

    def _valid_per_step(self,train_dataloader):
        self.model.eval()
        bar = tqdm(train_dataloader)
        preds,labels,losses = [],[],[]
        for X,label in bar:
            X = X.to(self.device)
            label = label.to(self.device)
            loss,predict = self.model(X,label)
            preds.append(predict.clone().detach().cpu().numpy())
            labels.append(label.clone().detach().cpu().numpy())
            losses.append(loss.clone().detach().cpu().item())
            bar.set_description("Valid Loss is {:.4f}".format(losses[-1]))
        
        preds = np.concatenate(preds,axis = 0)
        labels = np.concatenate(labels,axis = 0)
        loss = sum(losses)/len(bar)
        
        report = self.compute_metrics((preds,labels))
        if isinstance(report,dict):
            report['validation loss'] = loss
        elif isinstance(report,float):
            report = {'metric':report,'validation loss':loss}
        else:
            report = {'validation loss':loss}
        return report

    def _raw_report(self,dic):
        for k,v in dic.items():
            print('{}: {:.4f} \t'.format(k.capitalize(),v))
                                 
    def _ipython_report(self,dic):
        markdown = ''
        markdown += '|'
        for k,v in dic.items():
            markdown += f'{k.capitalize()}|'
        markdown += '  \n'
        markdown += '|'
        for k,v in dic.items():
            markdown += '---|'
        markdown += '  \n'

        markdown += '|'
        for k,v in dic.items():
            markdown += '{:.4f}|'.format(v)

        markdown += '  \n'
        from IPython.display import display_markdown
        display_markdown(markdown,raw = True)

    def _report(self,dic):
        try:
            self._ipython_report(dic)
        except:
            self._raw_report(dic)
    
    
    def test(self,my_datasets = None):
        '''
        my_datasets (`dict`): 数据集
            格式为: {'test':{'text':[]}}
        '''
        #模型
        self.model = self.model.to(self.device)
        
        #准备数据集
        if my_datasets is None:
            my_datasets = self.datasets
        
        #清空results
        self.results = defaultdict(list)
        self.predict(my_datasets['test']['text'],save_result = True)
        
        
    def train(self,my_datasets = None, epoch = 3 ,batch_size = 2 , learning_rate = 1e-3 ,save_path = None ,checkpoint_path = None,**kwargs):
        '''
        Args:
             my_datasets (`dict`): 数据集
                 格式为: {'train':{'text':[],'label':[]},'valid':{'text':[],'label':[]}}
                 
             epoch (`int`): epoch数量
                 遍历数据集数量，一般是3-10遍
                 
             batch_size (`int`): 批大小
                 一般batch_size越大越好，但是如果太大会超出内存/显存容量，常用的batch_size为4-32
                 
             save_path (`str`): 模型保存位置
                 会写入save_path指定的文件夹，模型名称为pytorch_model.bin
                 
             checkpoint_path (`str`): 检查点保存的位置
                 会写入checkpoint_path指定的文件夹
        '''   
        #模型
        self.model = self.model.to(self.device)
        
        #准备数据集
        if my_datasets is None:
            my_datasets = self.datasets
            
        train_dataset = TensorDataset(
            self._tokenizer_for_training(my_datasets['train']['text'],des = "训练集 Tokenizing 进度"),
            torch.tensor(my_datasets['train']['label'])
            )
        valid_dataset = TensorDataset(
            self._tokenizer_for_training(my_datasets['valid']['text'],des = "验证集 Tokenizing 进度"),
            torch.tensor(my_datasets['valid']['label'])
            )   
        
        train_dataloader = DataLoader(train_dataset,batch_size = batch_size, shuffle = True,drop_last = False)
        valid_dataloader = DataLoader(valid_dataset,batch_size = batch_size, shuffle = False,drop_last = False)
        
        #优化器
        if self.optim is None:
            self.optim = torch.optim.Adam(self.model.parameters() ,lr = learning_rate)
        if self.schedule is None:
            self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim,T_max= epoch * len(train_dataloader),eta_min = 1e-7)    
        
        #训练
        print("*"*7,"  开始训练  ","*"*7)
        for i in range(epoch):
            train_report = self._train_per_step(train_dataloader)
            self._update_train_reports(train_report)
                
            valid_report = self._valid_per_step(valid_dataloader)
            self._update_valid_reports(valid_report)

            self.report(self.valid_report)
            if i == 0:
                self._best_metric = valid_report[self.key_metric]
                self._best_model = self.model
            else:
                if self.key_metric.find('loss') != -1 and valid_report[self.key_metric] < self._best_metric:
                    self._best_model = self.model
                elif valid_report[self.key_metric] > self._best_metric:
                    self._best_model = self.model
                else:
                    pass
            if checkpoint_path:
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                checkpoint = {'epoch':i,
                          'learning rate': self.optim.state_dict()['param_groups'][0]['lr'],
                           'best_model':self._best_model}
                torch.save(checkpoint,os.path.join(checkpoint_path,f'checkpoint_e{i}.pt'))
        
        self.model = self._best_model
        
        del self._best_model
        
        if save_path:
            self.save_model(save_path)
