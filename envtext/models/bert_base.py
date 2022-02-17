from .model_base import ModelBase
from transformers import TrainingArguments, Trainer ,BertTokenizerFast, BertConfig , BertForMaskedLM
from datasets import Dataset
import torch
import os #for os.

class BertBase(ModelBase):
    def __init__(self,path,config = None,**kwargs):
        super().__init__()
        self.initialize_tokenizer(path)
        self.initialize_config(path)
        self.update_config(kwargs)
        self.update_config(config)
        self.initialize_bert(path)
        
    
    def initialize_tokenizer(self,path):
        '''
        初始化tokenizer
        '''
        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(path)  
        except:
            print("导入Vocab.txt失败，从celtics1863/env-bert-chinese导入")
            self.tokenizer = BertTokenizerFast.from_pretrained('celtics1863/env-bert-chinese')

   
    def initialize_config(self,path):
        '''
        初始化config
        '''
        if os.path.exists(os.path.join(path,'config.json')):
            self.config = BertConfig.from_pretrained(path)
            self.config ._name_or_path = path
        else:
            self.config = BertConfig()
            self.config._name_or_path
    
    def initialize_bert(self,path = None,config = None,**kwargs):
        '''
        初始化bert,需要继承后实现
        Args:
            BertPretrainedModel `transformers.models.bert.modeling_bert.BertPreTrainedModel`:
                Hugging face transformer版本的 Bert模型
                默认为 BertForMaskedLM
                目前只支持pytorch版本
           path `str`:
               模型的路径，默认为None。如果不是None，优先从导入
           config `dict`:
               模型的配置
        '''
        if path is not None:
            self.update_model_path(path)
        
        self.model = BertPretrainedModel.from_pretrained(self.model_path)
 
        
    def add_spetial_tokens(self,tokens):
        self.tokenizer.add_special_tokens({'additional_special_tokens':tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def update_config(self,config):
        '''
        更新config
        '''
        if self.config is None:
            pass
        else:
            if isinstance(config,dict):
                self.config.update(config)
            else:
                assert 'config参数必须是字典'
    
    def update_model_path(self,path):
        '''
        更新模型路径
       Args:
           path `str`:
               模型新的路径
        '''
        if path is not None:
            self.config._name_or_path = path
            
    @property
    def model_path(self):
        '''
        获得模型路径
        '''
        return self.config._name_or_path
    
    @property
    def label2id(self):
        '''
        返回一个dict,标签转id
        '''
        return self.config.label2id

    @property
    def id2label(self):
        '''
        返回一个dict,id转标签
        '''
        return self.config.id2label
    
    @property
    def labels(self):
        '''
        返回一个list,所有标签
        '''
        return self.config.labels
    
    @property
    def entities(self):
        '''
        返回一个list,所有实体
        '''
        return self.config.entities   
    
    def get_train_reports(self):
        '''
        获得训练集metrics报告
        '''
        raise NotImplemented
    
    def get_valid_reports(self):
        '''
        获得验证集metrics报告
        '''
        raise NotImplemented
    
    def set_optim(self,optim):
        '''
        设置优化器，需要是torch.optim，默认是Adam优化器，learning rate在训练时传入
        '''
        raise NotImplemented
        
    def set_learing_rate_schedule(self,schedule):
        '''
        设置学习率迭代方法，默认是torch.optim.lr_scheduler.CosineAnnealingLR
        '''
        raise NotImplemented
    
    def load_checkpoint(self,path):
        '''
        从checkpoint导入模型，但是必须要重新设置
        '''
        raise NotImplemented
        
        
    def _tokenizer_for_training(self,dataset):
        res = self.tokenizer(dataset['text'],max_length = self.max_length,padding='max_length',truncation=True)
        return res
    
    def train(self,my_datasets = None,epoch = 1,batch_size = 4, learning_rate = 2e-5, max_length = 512,save_path = None,checkpoint_path='checkpoint',**kwargs):
        '''
        训练模型，只保留了最关键几个参数的接口。
        Args:
            my_datasets `Dict[Dict[List]]`:
                格式为 {'train':{'text':[],'label':[]} ,'valid':{'text':[],'label':[]}}
                如果用self.load_dataset导入模型，则可以为None
           
           epoch `int`:
               默认：1
               模型完整经历一遍训练数据集，并更新若干遍参数，称为一个Epoch。一般Bert模型为2-5遍，RNN为3-10遍。
               
           batch_size `int`:
               默认：4
               一般来说越大模型效果越好，但是太大了内存/显存会溢出。
               
           learning_rate `float`:
               默认：2e-5
               最关键的参数，对不同数据集，需要的初始学习率不同。
               一般使用预训练bert的初始学习率为1e-4到5e-6之间，使用RNN的初始学习率为1e-3左右。
               
           max_length `int`:
               默认：512
               模型处理序列时，处理后的序列长度，如果序列长度不足，则pad，如果序列长度过长，则truncate
               序列长度越长，模型推理的时间也越长。
               
           save_path `str`:
               模型保存的文件夹
               
           checkpoint_path `str`:
               模型保存训练中途数据的文件夹
        '''
        if my_datasets is None:
            my_datasets = self.datasets
        dataset = {
            "train":Dataset.from_dict(my_datasets['train']),
            "valid":Dataset.from_dict(my_datasets['valid'])}
        print('train dataset: \n',dataset['train'])
        print('valid dataset: \n',dataset['valid'])
        self.max_length = max_length
        g = lambda data:data.map(self._tokenizer_for_training,remove_columns=["text"]) #batched=True, num_proc=4,
        tokenized_datasets = {}
        for K,V in dataset.items():
            tokenized_datasets[K] = g(V)
#         print(tokenized_datasets)
    
        self.args = TrainingArguments(
            checkpoint_path,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epoch,
            weight_decay=0.01,
            load_best_model_at_end=True,
            save_total_limit = 2,
            metric_for_best_model=self.key_metric,
        )


        self.trainer = Trainer(
            self.model,
            self.args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["valid"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        self.trainer.train()
        if save_path:
            self.trainer.save_model(save_path)
            
    
    def load_dataset(path,task,format = None,split=0.5,label_as_key = False,
                       sep = ' ',dataset = 'dataset',train = 'train',valid = 'valid' ,test = 'test', text = 'text', label = 'label'):
        '''
        读取数据集。
          参见 envText.data.utils.load_dataset
          
        '''
        try:
            self.datasets,self.data_config = load_dataset(path,format,task,split,label_as_key,sep,dataset,train,valid,test,text,label)
            print("*"*7,"读取数据集成功","*"*7)
        except:
            print("*"*7,"读取数据集失败","*"*7)
            
        self.initialize_bert(None,self.data_config)