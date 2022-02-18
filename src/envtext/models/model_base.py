from threading import Thread
from collections import defaultdict
from transformers.configuration_utils import PretrainedConfig
import torch #for torch.cuda.is_available
from ..data.utils import load_dataset
import os

class ModelBase:
    def __init__(self,*args,**kwargs):
        self.model = None
        self.config = PretrainedConfig(
            package = 'envtext',
            liscence = 'Apache Lisence',
            contact = 'bi.huaibin@foxmail.com',
            max_length = 512,
        )
        self.tokenizer = None
        self.datasets = None
        self.data_config = None
        self.args = None
        self.trainer = None
        self.result = defaultdict(dict)
        
        self.training_results = defaultdict(list)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        
    def set_attribute(self,**kwargs):
        '''
        设置参数
        '''
        self.config.update(kwargs)
        
    @property
    def max_length(self):
        '''
        获得最大长度
        '''
        if hasattr(self.config,'max_length'):
            return self.config.max_length
        else:
            return None
    
    @property
    def key_metric(self):
        if hasattr(self.config,'key_metric'):
            return self.config.key_metric
        else:
            self.set_attribute(key_metric = 'validation loss')
            return 'validation loss'
    
    def get_device(self):
        return self.device

    def set_device(self,device):
        '''
        设置模型运行的设备，cpu或cuda等
        '''
        self.device = device
        self.model.to(device)
        
    def save_result(self,save_path,sep = ' '):
        '''
        保存结果
        Args:
            save_path `str`: 模型保存的文件名
                支持 csv, excel, txt, json等多种格式
           sep [Optional] `str`:
               分隔符，只用于保存为txt文件时
        '''
        import os
        dir_name = os.path.dirname(os.path.realpath(save_path))
        
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        if save_path.split('.')[-1] == 'csv':
            self._save_result2csv(save_path)
        elif save_path.split('.')[-1] in ['xlsx','xls']:
            self._save_result2excel(save_path)
        elif save_path.split('.')[-1] == 'txt':
            self._save_result2txt(save_path,sep)
        elif save_path.split('.')[-1] == 'json':
            self._save_result2json(save_path)
        else:
            raise NotImplemented 
    
    def _save_result2json(self,path):
        from ..utils.json_ops import write_json
        write_json(path,self.result)
    
    def _save_result2csv(self,path):
        '''
        保存到csv
        '''
        import pandas as pd
        df = pd.DataFrame(self.result).transpose()
        df.to_csv(path,index = True)

    def _save_result2excel(self,path):
        '''
        保存到csv
        '''
        import pandas as pd
        df = pd.DataFrame(self.result).transpose()
        df.to_excel(path,index = True)
        
    def _save_result2txt(self,path,sep = ' '):
        '''
        保存到txt
        '''
        f = open(path,'w',encoding='utf-8')
        columns = []
        for k,v in self.result.items():
            if len(v) > len(columns):
                columns = list(v.keys())
        columns = ['text'] + columns
        f.write(sep.join(columns) + '\n')
        
        for k,v in self.result.items():
            result = [k]
            for kk,vv in v.items():
                result.append(str(vv))
            f.write(sep.join(result) + '\n')
        f.close()
        
    def save_model(self,save_path):
        '''
        保存模型
        Args:
            save_path `str`: 模型保存的文件夹
        '''
        if not os.path.exists(os.path.realpath(save_path)):
            os.makedirs(os.path.realpath(save_path))
        
        if hasattr(self.model,'save_pretrained'):
            self.model.save_pretrained(path)
        else:
            torch.save(self._best_model,os.path.join(save_path,'pytorch_model.bin'))
            if self.config:
                self.config.save_pretrain(save_path)


    def update_config(self,config = None,**kwargs):
        '''
       更新模型的配置参数config

       Args:
           config [Optional] `dict` : 默认None
               用config的内容更新配置参数
               如果是None则忽略


       Kwargs:
           可以任何参数，会更新进入配置参数self.config中
       '''
        if self.config is None:
            self.config = PretrainedConfig()
            
        if config is None:
            pass
        else:
            if isinstance(config,dict):
                self.config.update(config)
            else:
                assert 'config参数必须是字典'
        
        self.config.update(kwargs)
        
    def predict_sentences(self,text_list,**kwargs):
        '''
        预测多个句子
        Args:
         text_list `List[str]`:
             文本的列表
        '''
        thread_list = []
        for text in text_list:
            thread_list.append(Thread(target = self.predict_per_sentence,args=(text,),kwargs=kwargs))

        for thread in thread_list:
            thread.start()

        for thread in thread_list:
            thread.join()
    
    def predict(self,texts,**kwargs):
        '''
        使用模型预测文本。
        Args：
            texts `List[str] or str`
                文本，或者 list(文本)
                模型的输入，输入后，自动会进行tokenize处理（包括预处理，翻译为token），再送入bert模型，进行预测。
                
        Kwargs:
           topk `int`： 
               默认为5,报告预测概率前topk的结果。
           
           print_result `bool`: 
               默认为True
               是否打印结果，对于大批量的文本，建议设置为False
               
           save_result: 
               默认为True
               是否保存结果
       '''
        if isinstance(texts,str):
            texts = [texts]
        elif isinstance(texts,list):
            pass
        elif isinstance(texts,tuple):
            texts = [t for t in texts]  
        else:
            raise NotImplemented
        
        thread_list = []
        for text in texts:
            thread_list.append(Thread(target = self.predict_per_sentence,args=(text,),kwargs=kwargs ))

        for thread in thread_list:
            thread.start()

        for thread in thread_list:
            thread.join()

    def __call__(self,*args,**kwargs):
        self.predict(*args,**kwargs)
        
    def compute_metrics(self,eval_pred):
        '''
        计算预测结果评价指标，需要继承之后实现，默认返回为空的dict
        
        Args:
            eval_pred `Tuple(numpy.ndarray)`:
                eval_pred = (预测结果的概率分布,真实结果)
                
       
       return: `dict` 
        '''
        self.key_metric = 'validation loss'
        return {}
    
    def predcit_per_sentence(self,text, **kwargs):
        '''
        需要继承之后重新实现
        如果选择保存结果，结果保存在self.result里：
        eg: 
            self.result[text] = {
                'label':预测的结果,
                }
                
        Args:
           text `str`
        
        Kwargs:
           topk [Optional] `int`： 默认为5
               报告预测概率前topk的结果。
           
           print_result [Optional] `bool`: 默认为True
               是否打印结果，对于大批量的文本，建议设置为False
               
           save_result: [Optional] `bool`: 默认为True
               是否保存结果
        '''
        pass
  
    def train(self,my_datasets,epoch ,batch_size , learning_rate ,save_path ,checkpoint_path,**kwargs):
        '''
        需要继承之后重新实现
        '''
        pass
    
            
#     def load_dataset(self,path,task,format ,split=0.5,label_as_key = False,
#                        sep = ' ',dataset = 'dataset',train = 'train',valid = 'valid' ,test = 'test', text = 'text', label = 'label'):
    def load_dataset(self,*args,**kwargs):
        '''
        读取数据集。
          参见 envText.data.utils.load_dataset
        
        Args:
            path `str`:
                数据集的路径
                
            task `str`:
                通用任务名称：
                    分类任务：'cls','classification','CLS','class'
                    回归任务：'reg'，'regression','REG'
                    情感分析：'sa','SA','Sentimental Analysis'
                    命名实体识别：'ner','NER','namely entity recognition'
                    多选：'MC','mc','multi-class','multi-choice','mcls'
                    关键词识别：'key','kw','key word','keyword','keywords','key words'
                
               专用任务名称：
                   2020CLUENER: 'clue_ner','cluener','CLUENER' 
                
           format [Optional] `str`:
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
               
         sampler [Optional] `float`: 默认：1
               当sampler在0-1是，对数据集进行随机采样，以概率p = sampler随机保留数据。
               
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
        try:
#             self.datasets,self.data_config = load_dataset(path,task,format,split,label_as_key,sep,dataset,train,valid,test,text,label)
            self.datasets,self.data_config = load_dataset(*args,**kwargs)

            print("*"*7,"读取数据集成功","*"*7)
        except Exception as e:
            print("*"*7,"读取数据集失败","*"*7)
            
#         return self.datasets,self.data_config