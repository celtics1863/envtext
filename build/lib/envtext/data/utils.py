from .load_dataset import *
from collections import defaultdict
from ..files import FileConfig


def sampler_dataset(dataset, p = 0.5):
    import random
    sampled_dataset = {'train':defaultdict(list),'valid':defaultdict(list),'test':defaultdict(list)}
    for k,v in dataset.items():
        keys = list(v.keys())
        for values in zip(*(v[kk] for kk in keys)):
            if random.random() < p:
                for kk,vv in zip(keys,values):
                    sampled_dataset[k][kk].append(vv)
                
    return sampled_dataset


#确认task
def _unify_task(task):
    if task.lower() in [0,'cls','classification','classify']:
        return 'CLS'
    elif task.lower() in [1,'reg','regression','regressor','sa','sentitive analysis']:
        return 'REG'
    elif task.lower() in [2,'ner','namely entity recognition']:
        return 'NER'
    elif task.lower() in [2,'key','kw','key word','keyword','keywords','key words']:
        return 'KW'
    elif task.lower() in [3,'mcls','multi-class','multiclass','multiclasses','mc','multi-choice','multichoice']:
        return 'MCLS'
    elif task.lower() in [4,'cluener','clue_ner','clue ner']:
        return 'CLUENER'


def load_dataset(path,task = None,format = None , sampler = 1 ,split=0.5,label_as_key = False,
                       sep = ' ',dataset = 'dataset',train = 'train',valid = 'valid' ,test = 'test', text = 'text', label = 'label'):
    '''
    读取训练数据集的通用接口，用来处理各种输入。
    format = 'json'
       适用于如下类型的数据格式：
            {'train':{'text':[],  'label':[] },...}
               或 
            {'text':[],  'label':[] } 以split为比例随机划分训练集和验证集
    format = 'json2'
        适用于如下类型的数据格式：
            {'train':{'label_1':[],  'label_2':[], ... },...}
               或 
            {'label_1':[],  'label_2':[], ... } 以split为比例随机划分训练集和验证集
   format = 'jsonL'
        适用于如下类型的json line数据格式：
            {'text': text_1,  'label':label_1, 'dataset':'train'}
            {'text': text_2,  'label':label_2, 'dataset':'train'}
            {'text': text_3,  'label':label_3, 'dataset':'valid'}
            {'text': text_4,  'label':label_4, 'dataset':'valid'}
            ...
            或
            {'text': text_1,  'label':label_1}
            {'text': text_2,  'label':label_2}
            {'text': text_3,  'label':label_3}
            ... 以split为比例随机划分训练集和验证集
   format = 'text'
        适用于如下类型的数据格式：
            train
            text_1 label_1
            text_2 label_2
            valid
            text_1 label_1
            text_2 label_2
            ...
            或者：
            text_1 label_1 train
            text_2 label_2 train
            text_3 label_3 valid
            text_4 label_4 valid
            ...
            或者：
            text_1 label_1
            text_2 label_2 
            ... 以split为比例随机划分训练集和验证集
            三类数据格式

    format = 'text2'
        适用于如下类型的数据格式：
           train
           text_1
           label_1
           text_2
           label_2
           valid
           text_1
           label_1
           text_2
           label_2
           ...
           或
           text_1
           label_1
           text_2
           label_2
            ... 以split为比例随机划分训练集和验证集
           两类数据格式
           
   format = 'excel'
       适用于如下类型的数据格式（.csv 或.xls 或.xlsx 或pd.DataFrame()）：
            | text | label  | dataset|
            |text_1| label_1|  train |
            |text_2| label_2|  train |
            |text_3| label_3|  valid |
            |text_4| label_4|  valid |
            ...
            或者：
            | text | label  |
            |text_1| label_1|
            |text_2| label_2|
            |text_3| label_3|
            |text_4| label_4|
            ...
            两类数据格式
       
    '''
#     kwargs = {'task':task,'split':split,'sep':sep,'dataset':dataset,'train':train,'valid':valid,'test':test,'label':label}
    config = FileConfig()

    if path.lower() in config.datasets_names:
        info = config.datasets_info[config.datasets_names[path.lower()]]
        task = info['task']
        format = info['format']
        path = info['path']
        
    task = _unify_task(task)
    
    if task == 'CLUENER':
        format = 'jsonL'
        
    if format is None:
        if path.split('.')[-1] == 'json':
            try:
                datasets,config = LoadJson.load_dataset(path,task,split,sep,dataset,train,valid,test,text,label)
            except:
                datasets,config = LoadJson2.load_dataset(path,task,split,sep,dataset,train,valid,test,text,label)
        elif path.split('.')[-1] == 'csv':
            datasets,config = LoadExcel.load_dataset(path,task,split,sep,dataset,train,valid,test,text,label)
        else:
            try:
                datasets,config = LoadText.load_dataset(path,task,split,sep,dataset,train,valid,test,text,label)
            except:      
                datasets,config = LoadText2.load_dataset(path,task,split,sep,dataset,train,valid,test,text,label)
            
    elif format == 'json' and not label_as_key:
        datasets,config = LoadJson.load_dataset(path,task,split,sep,dataset,train,valid,test,text,label)
    elif format == 'json2' or (format == 'json' and label_as_key):
        datasets,config = LoadJson2.load_dataset(path,task,split,sep,dataset,train,valid,test,text,label)
    elif format == 'jsonL':
        datasets,config = LoadJsonL.load_dataset(path,task,split,sep,dataset,train,valid,test,text,label)
    elif format == 'text':
        datasets,config = LoadText.load_dataset(path,task,split,sep,dataset,train,valid,test,text,label)
    elif format == 'text2':
        datasets,config = LoadText2.load_dataset(path,task,split,sep,dataset,train,valid,test,text,label)
    elif format == 'excel':
        datasets,config = LoadExcel.load_dataset(path,task,split,sep,dataset,train,valid,test,text,label)
    else:
        raise NotImplemented
    
    if sampler and 0 < sampler < 1:
        datasets = sampler_dataset(datasets,sampler)
    
    return datasets,config