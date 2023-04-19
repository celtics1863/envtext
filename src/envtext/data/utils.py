from .load_files import *
from collections import defaultdict
from ..files import Config
from .dataset_utils import _unify_task

def sampler_dataset(dataset, p = 0.5):
    import random
    sampled_dataset = {'train':defaultdict(list),'valid':dataset["valid"],'test':dataset["test"]}

    k,v = "train",dataset["train"]
    keys = list(v.keys())

    if isinstance(p, float) or p == 1.0:
        for values in zip(*(v[kk] for kk in keys)):
            if random.random() < p:
                for kk,vv in zip(keys,values):
                    sampled_dataset[k][kk].append(vv)
    elif isinstance(p, int):
        ids = set(random.choices(range(len(v["text"])), k=p))
        for idx,values in enumerate(zip(*(v[kk] for kk in keys))):
            if idx in ids:
                for kk,vv in zip(keys,values):
                    sampled_dataset[k][kk].append(vv)
    else:
        return dataset

    # sampled_dataset = {'train':defaultdict(list),'valid':defaultdict(list),'test':defaultdict(list)}

    # k,v = "train",dataset["train"]
    # keys = list(v.keys())

    # for k,v in dataset.items():
    #     keys = list(v.keys())
    #     for values in zip(*(v[kk] for kk in keys)):
    #         if random.random() < p:
    #             for kk,vv in zip(keys,values):
    #                 sampled_dataset[k][kk].append(vv)
                
    return sampled_dataset


def load_dataset(path,task = None,format = None , sampler = 1 ,split=0.5,label_as_key = False, label_inline = False,ner_encoding = 'BIO',
                       sep = ' ',dataset = 'dataset',train = 'train',valid = 'valid' ,test = 'test', text = 'text', label = 'label',
                        entity_label = 'label',loc = 'loc',**kwargs):
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
    if path.lower() in Config.datasets_names:
        info = Config.datasets_info[Config.datasets_names[path.lower()]]
        task = info['task']
        format = info['format']
        path = info['path']
        
    task = _unify_task(task)
    
    if task == 'CLUENER':
        format = 'jsonL'
 
    kwargs.update({
        'path':path,
        'task':task,
        'format':format,
        'sampler':sampler,
        'split':split,
        'label_as_key':label_as_key,
        'label_inline':label_inline,
        'ner_encoding':ner_encoding,
        'sep':sep,
        'dataset': dataset,
        'train': train,
        'valid':valid,
        'test': test,
        'text':text, 
        'label':label,
        'entity_label':entity_label,
        'loc':loc
        })

    
    if format is None:
        if path.split('.')[-1] == 'json':
            try:
                datasets,config = LoadJson.load_dataset(**kwargs)
            except:
                datasets,config = LoadJson2.load_dataset(**kwargs)
        elif path.split('.')[-1] == 'csv':
            datasets,config = LoadExcel.load_dataset(**kwargs)
        else:
            try:
                datasets,config = LoadText.load_dataset(**kwargs)
            except:      
                datasets,config = LoadText2.load_dataset(**kwargs)
            
    elif format == 'json' and not label_as_key:
        datasets,config = LoadJson.load_dataset(**kwargs)
    elif format == 'json2' or (format == 'json' and label_as_key):
        datasets,config = LoadJson2.load_dataset(**kwargs)
    elif format == 'jsonL':
        datasets,config = LoadJsonL.load_dataset(**kwargs)
    elif format.startswith('text') and label_inline:
        if kwargs["task"] in ["DP","Triple","Relation"]:
            kwargs["sep"] = "\n\n"
            datasets,config = LoadRawText.load_dataset(**kwargs)
        else:
            datasets,config = LoadRawText.load_dataset(**kwargs)
    elif format == 'text':
        datasets,config = LoadText.load_dataset(**kwargs)

    elif format in ['text2','textline']:
        datasets,config = LoadText2.load_dataset(**kwargs)
    elif format in ['excel','csv','xlsx','xls']:
        datasets,config = LoadExcel.load_dataset(**kwargs)
    else:
        raise NotImplementedError()
    
    if sampler:
        datasets = sampler_dataset(datasets,sampler)
    
    for k in datasets:
        config[f"num_{k}_texts"] = len(datasets[k]["text"])
        
    return datasets,config