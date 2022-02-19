from .model_base import ModelBase
from transformers import TrainingArguments, Trainer ,BertTokenizerFast, BertConfig
from datasets import Dataset
import torch
import os 
from tqdm import tqdm
import numpy as np #for np.concatnate

class BertBase(ModelBase):
    def __init__(self,path,config = None,**kwargs):
        super().__init__()
        self.initialize_tokenizer(path)
        self.initialize_config(path)
        self.update_config(kwargs)
        self.update_config(config)
        self.initialize_bert(path)
        self.set_attribute(key_metric = 'loss',max_length = 512)
        
        
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
            config = BertConfig.from_pretrained(path)
        else:
            config = BertConfig()
        config.update(self.config.to_diff_dict())
        self.config = config
    
    def align_config(self):
        '''
        对齐config，在initialize_bert的时候调用，如有必要则进行重写。
        '''
        pass
    
    def initialize_bert(self,path = None,config = None,**kwargs):
        '''
        初始化bert,需要继承之后重新实现
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
        
        if config is not None:
            self.update_config(config)
        
        self.update_config(kwargs)
        
        self.align_config()
            
        #例如：
        #self.model = BertPretrainedModel.from_pretrained(self.model_path)
 
        
    def add_spetial_tokens(self,tokens):
        self.tokenizer.add_special_tokens({'additional_special_tokens':tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))

    
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
        获得模型路径，没有路径则返回None
        '''
        if hasattr(self.config,'_name_or_path'):
            return self.config._name_or_path
        else:
            return None
    
    @property
    def label2id(self):
        '''
        返回一个dict,标签转id
        '''
        if hasattr(self.config,'label2id'):
            return self.config.label2id
        else:
            return None
        
    @property
    def id2label(self):
        '''
        返回一个dict,id转标签
        '''
        if hasattr(self.config,'id2label'):
            return self.config.id2label
        else:
            return None
        
    @property
    def labels(self):
        '''
        返回一个list,所有标签
        '''
        if hasattr(self.config,'labels'):
            return self.config.labels
        else:
            return None
        
    @property
    def entities(self):
        '''
        返回一个list,所有实体
        '''
        if hasattr(self.config,'entities'):
            return self.config.entities   
        else:
            return None
        
    @property
    def num_labels(self):
        '''
        返回一个list,所有标签
        '''
        if hasattr(self.config,'num_labels'):
            return self.config.num_labels
        else:
            return None

    @property
    def num_entities(self):
        '''
        返回一个list,所有标签
        '''
        if hasattr(self.config,'num_entities'):
            return self.config.num_entities
        else:
            return None

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
        
        
    def _tokenizer_for_inference(self,texts,des):
        bar = tqdm(texts)
        bar.set_description(des)
        tokens = []
        for text in bar:
            tokens.append(self.tokenizer.encode(text,max_length = self.max_length,padding='max_length',truncation=True))
        tokens = torch.tensor(tokens)
        return tokens

    def _inference_per_step(self,dataloader):
        self.model.eval()
        bar = tqdm(dataloader)
        bar.set_description("正在 Inference ...")
        preds = []
        for X in bar:
            X = X[0].to(self.device)
            with torch.no_grad():
                predict = self.model(X)[0]
            preds.append(predict.clone().detach().cpu().numpy())
            
        preds = np.concatenate(preds,axis = 0)
        return preds
    
    def inference(self, texts = None, batch_size = 2):
        '''
        推理数据集，更快的速度，更小的cpu依赖，建议大规模文本推理时使用。
        与self.predict() 的区别是会将数据打包为batch，并使用gpu(如有)进行预测，最后再使用self.postprocess()进行后处理，保存结果至self.result
        
        texts (`List[str]`): 数据集
            格式为列表
        '''        
        texts = self._align_input_texts(texts)
        
        #模型
        self.model = self.model.to(self.device)
        
        #准备数据集
        from torch.utils.data import TensorDataset,DataLoader
        tokens = self._tokenizer_for_inference(texts,des = "正在Tokenizing...")
        dataset = TensorDataset(tokens)
        dataloader = DataLoader(dataset,batch_size = batch_size, shuffle = False,drop_last = False)
        
        #推理
        preds = self._inference_per_step(dataloader)
        
        #保存results获得
        bar = tqdm(zip(texts,preds))
        bar.set_description('正在后处理...')
        for text,pred in bar:
            self.postprocess(text,pred,print_result = False,save_result = True)
            
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
            
        if my_datasets is None:
            assert 0,"请输入有效数据集，或者使用load_dataset()导入数据集"
            
        dataset = {
            "train":Dataset.from_dict(my_datasets['train']),
            "valid":Dataset.from_dict(my_datasets['valid'])}
        print('train dataset: \n',dataset['train'])
        print('valid dataset: \n',dataset['valid'])
        
        self.set_attribute(max_length = max_length)
        
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
            
    
    def load_dataset(self,*args,**kwargs):
        '''
        读取数据集。
          参见 envText.data.utils.load_dataset
        
        Args:
            path `str`:
                数据集的路径
                
            task `str`:
                任务名称：
                分类任务：'cls','classification','CLS','class'
                回归任务：'reg'，'regression','REG'
                情感分析：'sa','SA','Sentimental Analysis'
                命名实体识别：'ner','NER','namely entity recognition'
                多选：'MC','mc','multi-class','multi-choice','mcls'
                关键词识别：'key word','kw','key_word'
                
           format `str`:
               格式：详细见envText.data.utils.load_dataset的注释
               - json: json的格式
                   {'train':{'text':[],'label':[]},'valid':{'text':[],'label':[]}}
                   或 {'text':[],'label':[]}
               - json2:json的格式，但是label作为key
                   {'train':{'label1':[],'label2':{},...},'valid':{'label1':[],'label2':{},...}}
                   或 {'label1':[],'label2':{},...}
               - text: 纯文本格式，一行中同时有label和text
                       text label datasets
                       text1 label1 train
                       text2 label2 valid
                       ...
                   或
                       text label
                       text1 label1
                       text2 label2
                       ...
                   或
                       train
                       text1 label1
                       text2 label2
                       ...
                       valid
                       text1 label1
                       text2 label2
                       ...
    
               - text2:纯文本格式，一行text，一行label
                       train
                       text1
                       label1
                       ...
                       valid
                       text2
                       label2
                       ...
                    或：
                       text1
                       label1
                       text2
                       label2
               - excel: excel,csv等格式
                  |text | label | dataset |
                  | --- | ---  | ------ |
                  |text1| label1| train |
                  |text2| label2| valid |
                  |text3| label3| test |
                  或
                  |text | label | 
                  | --- | ---  | 
                  |text1| label1|
                  |text2| label2|
                  |text3| label3|
       Kwargs:   
         
         split [Optional] `float`: 默认：0.5
               训练集占比。
               当数据集没有标明训练/验证集的划分时，安装split:(1-split)的比例划分训练集:验证集。
               
          sep [Optional] `str`: 默认：' '
               分隔符：
               text文件读取时的分隔符。
               如果keyword、ner任务中，实体标注没有用list分开，而是用空格或逗号等相连，则sep作为实体之间的分隔符。
               例如：有一条标注为
                   "气候变化,碳中和"，设置sep=','，可以将实体分开
                   一般建议数据集格式为["气候变化","碳中和"]，避免不必要的混淆
                   
          label_as_key `bool`: 默认：False
              如果格式为json且设置label_as_key，等效于json2格式
          
          dataset `str`: 默认：'dataset'
              标示数据集一列的列头。
              例如csv文件中：
                  |text | label | **dataset **|
                  | --- | ---  | ------ |
                  |text1| label1| train |
                  |text2| label2| valid |
                  |text3| label3| test |
                  
          
          train `str`: 默认：'train'
              标示数据是训练/验证集/测试集
            例如csv文件中：
                  |text | label | dataset|
                  | --- | ---  | ------ |
                  |text1| label1| **train** |
                  |text2| label2| valid |
                  |text3| label3| test |
         
         valid `str`: 默认：'valid'
              标示数据是训练/验证集/测试集
            例如csv文件中：
                  |text | label | dataset|
                  | --- | ---  | ------ |
                  |text1| label1| train |
                  |text2| label2| **valid** |
                  |text3| label3| test |
         
         
         test `str: 默认：'test'
           标示数据是训练/验证集/测试集
            例如csv文件中：
                  |text | label | dataset|
                  | --- | ---  | ------ |
                  |text1| label1| train |
                  |text2| label2| valid |
                  |text3| label3| **test** |
          
         text `str`: 默认：'text'
            标示文本列的列头
            例如csv文件中：
                  |**text** | label | dataset|
                  | --- | ---  | ------ |
                  |text1| label1| train |
                  |text2| label2| valid |
                  |text3| label3| test  |
                  
         label `str`: 默认：'label'
            标示标签列的列头
            例如csv文件中：
                  |text | **label** | dataset|
                  | --- | ---  | ------ |
                  |text1| label1| train |
                  |text2| label2| valid |
                  |text3| label3| test  |
        '''
        super().load_dataset(*args,**kwargs)
        self.initialize_bert( None, self.data_config)