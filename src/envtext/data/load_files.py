from collections import defaultdict,Counter
from ..utils.json_ops import read_json, read_jsonL #for read_json() read_jsonL()
from ..utils.txt_ops import txt_generator
import random #for random.random()
import pandas as pd #for pd.read_csv() pd.read_excel()
import numpy as np #for np.isnan()
from .process_train_dataset import LoadDataset

class LoadJson(LoadDataset):
    @classmethod
    def load_dataset(cls,**kwargs):
        '''
            适用于如下类型的数据格式：
            {'train':{'text':[],  'label':[] },...}
               或 
            {'text':[],  'label':[] } 以split为比例随机划分训练集和验证集
        '''
        js = read_json(kwargs['path'])

        #划分数据集
        if not cls().has_split(js):
            new_js = {
                kwargs['train']:defaultdict(list),
                kwargs['valid']:defaultdict(list)
            }
            keys = list(js.keys())
            for values in zip(*(js[kk] for kk in keys)):
                if random.random() < kwargs['split']:
                    for k,v in zip(keys,values):
                        new_js[kwargs['train']][k].append(v)
                else:
                    for k,v in zip(keys,values):
                        new_js[kwargs['valid']][k].append(v)
            js = new_js
         
        kwargs['label_as_key'] = False
        return cls().generate_datasets(js,**kwargs)
        
class LoadJson2(LoadDataset):
    @classmethod
    def load_dataset(cls,**kwargs):
        '''
            适用于如下类型的数据格式：
            {'train':{'label_1':[],  'label_2':[], ... },...}
               或 
            {'label_1':[],  'label_2':[], ... } 以split为比例随机划分训练集和验证集
        '''
        js = read_json(kwargs['path'])

        #划分数据集
        if not cls().has_split(js):
            new_js = {
                kwargs['train']:defaultdict(list),
                kwargs['valid']:defaultdict(list)
            }
            for k,v in js.items():
                for vv in v:
                    if random.random() < kwargs['split'] :
                        new_js[kwargs['train']][k].append(vv)
                    else:
                        new_js[kwargs['valid']][k].append(vv)
            js = new_js
        
        kwargs['label_as_key'] = True
        return cls().generate_datasets(js,**kwargs)
        
class LoadJsonL(LoadDataset):
    @classmethod
    def load_dataset(cls,**kwargs):
        '''
            适用于如下类型的json line数据格式：
            {'text': text_1,  'label':label_1, 'dataset':'train'}
            {'text': text_2,  'label':label_2, 'dataset':'train'}
            {'text': text_3,  'label':label_3, 'dataset':'valid'}
            {'text': text_4,  'label':label_4, 'dataset':'valid'}
            或
            {'text': text_1,  'label':label_1}
            {'text': text_2,  'label':label_2}
            {'text': text_3,  'label':label_3}
            以split为比例随机划分训练集和验证集
        '''
        flag = {'dataset':None}
        
        for idx,line in enumerate(read_jsonL(kwargs['path'])):
            if idx == 0:# 通过第一行判断
                if kwargs["dataset"] in line.keys():
                    flag['dataset'] = 1
                    js = {
                            kwargs['train']:defaultdict(list),
                            kwargs['valid']:defaultdict(list),
                            kwargs['test']:defaultdict(list)
                        }
                else:
                    flag['dataset'] = 0
                    js = {
                            kwargs['train']:defaultdict(list),
                            kwargs['valid']:defaultdict(list)
                        }
            
            if flag['dataset']:
                js[line[kwargs['dataset']]][kwargs['text']].append(line[kwargs['text']])
                js[line[kwargs['dataset']]][kwargs['label']].append(line[kwargs['label']])
            else:
                if random.random() < kwargs['split'] :
                    js[kwargs['train']][kwargs['text']].append(line[kwargs['text']])
                    js[kwargs['train']][kwargs['label']].append(line[kwargs['label']])
                else:
                    js[kwargs['valid']][kwargs['text']].append(line[kwargs['text']])
                    js[kwargs['valid']][kwargs['label']].append(line[kwargs['label']])

        kwargs['label_as_key'] = False
        return cls().generate_datasets(js,**kwargs)
        
class LoadText(LoadDataset):
    @classmethod
    def load_dataset(cls,**kwargs):
        '''
            适用于如下类型的数据格式：
            train:
            text_1 label_1
            text_2 label_2
            valid:
            text_1 label_1
            text_2 label_2
            或者：
            text_1 label_1 train
            text_2 label_2 train
            text_3 label_3 valid
            text_4 label_4 valid
            或者：
            text_1 label_1
            text_2 label_2
            三类数据格式
        '''
        #数据集
        js = {kwargs['train']:defaultdict(list),kwargs['valid']:defaultdict(list),kwargs['test']:defaultdict(list)}
        flag = {'format':0,'dataset':None}
        
        for idx,line in enumerate(txt_generator(kwargs['path'])):
            #通过第一行判断数据集的格式
            words = line.split(kwargs['sep'])
            if idx == 0: 
                if line in [kwargs['train'],kwargs['valid'],kwargs['test']]:
                    flag['format'] = 1
                elif len(words) == 3:
                    flag['format'] = 2
                elif len(words) == 2:
                    flag['format'] = 3
                else:
                    assert 0,'数据集一行中只允许2-3个片段，但是现在有{}个'.format(len(words))
            
            if flag['format'] == 1:
                if line in [kwargs['train'],kwargs['valid'],kwargs['test']]:
                    flag['dataset'] = line
                    continue
                else:
                    assert len(words) == 2,'第一类数据集一行中只允许2个片段，但是第{}行有{}个'.format(idx,len(words))
                    js[flag['dataset']][kwargs['text']].append(words[0])
                    js[flag['dataset']][kwargs['label']].append(words[1])
            elif flag['format'] == 2:
                assert len(words) == 3,'第二类数据集一行中只允许3个片段，但是第{}行有{}个'.format(idx,len(words))
                js[words[2]][kwargs['text']].append(words[0])
                js[words[2]][kwargs['label']].append(words[1])
            elif flag['format'] == 3:
                assert len(words) == 2,'第三类数据集一行中只允许2个片段，但是第{}行有{}个'.format(idx,len(words))
                if random.random() < kwargs['split']:
                    js[kwargs['train']][kwargs['text']].append(words[0])
                    js[kwargs['train']][kwargs['label']].append(words[1])
                else:
                    js[kwargs['valid']][kwargs['text']].append(words[0])
                    js[kwargs['valid']][kwargs['label']].append(words[1])
        
        kwargs['label_as_key'] = False
        return cls().generate_datasets(js,**kwargs)


class LoadRawText(LoadDataset):
    @classmethod
    def load_dataset(cls,**kwargs):
        '''
        适用于如下类型的数据格式：
           train
           text_1
           text_2
           valid
           text_1
           text_2
           ...
           或
           text_1
           text_2
            ... 以split为比例随机划分训练集和验证集
           两类数据格式
        '''
        #数据集
        js = {kwargs['train']:defaultdict(list),kwargs['valid']:defaultdict(list),kwargs['test']:defaultdict(list)}
        flag = {'format':0,'dataset':None}
        
        import re

        def data_generator():
            data = []
            for line in open(kwargs["path"],"r",encoding="utf-8"):
                line = line.strip()
                if kwargs["sep"] == "\n\n":
                    if not re.sub("\s", "", line): #有一行为空
                        if data:
                            yield data
                            data = []
                    else: #否则append这一行
                        data.append(line) 
                else:
                    yield line

        for idx,line in enumerate(data_generator()):
            #通过第一行判断数据集的格式
            if idx == 0: 
                first_line = line if isinstance(line, str) else line[0]
                if first_line in [kwargs['train'],kwargs['valid'],kwargs['test']]:
                    flag['format'] = 1

            if flag['format'] == 1:
                js[flag['dataset']]['raw_text'].append(line)
            else:
                if random.random() < kwargs['split'] :
                    js[kwargs['train']]['raw_text'].append(line)
                else:
                    js[kwargs['valid']]['raw_text'].append(line)
        
        kwargs['label_inline'] = True
        return cls().generate_datasets(js,**kwargs)
    



class LoadText2(LoadDataset):
    @classmethod
    def load_dataset(cls,**kwargs):
        '''
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
        '''
        #数据集
        js = {kwargs['train']:defaultdict(list),kwargs['valid']:defaultdict(list),kwargs['test']:defaultdict(list)}
        datasets = {kwargs['train']:defaultdict(list),kwargs['valid']:defaultdict(list),kwargs['test']:defaultdict(list)}
        flag = {'format':0,'dataset':None,'is_text':False,'text':''}
        for idx,line in enumerate(txt_generator(kwargs['path'])):
            line = line.strip()
            #通过第一行判断数据集的格式
            if idx == 0: 
                if line in [kwargs['train'],kwargs['valid'],kwargs['test']]:
                    flag['format'] = 1
                    flag['is_text'] = False
                else:
                    flag['is_text'] = True
            
            if flag['format'] == 1:
                if line in [kwargs['train'],kwargs['valid'],kwargs['test']]:
                    flag['dataset'] = line
                    flag['is_text'] = False
                    continue
                elif flag['is_text']:
                    js[flag['dataset']][kwargs['text']].append(line)
                else:
                    if kwargs["task"] != "MCLS":
                        js[flag['dataset']][kwargs['label']].append(line)
                    else:
                        js[flag['dataset']][kwargs['label']].append(line.split(",")) #多选使用,分割不同标签 
            else:
                if flag['is_text']:
                    flag['text'] = line
                else:
                    if random.random() < kwargs['split']:
                        js['train'][kwargs['text']].append(flag['text'])
                        if kwargs["task"] != "MCLS":
                            js['train'][kwargs['label']].append(line)
                        else:
                            js['train'][kwargs['label']].append(line.split(",")) #多选使用,分割不同标签 
                    else:
                        js['valid'][kwargs['text']].append(flag['text'])
                        if kwargs["task"] != "MCLS":
                            js['valid'][kwargs['label']].append(line)
                        else:
                            js['valid'][kwargs['label']].append(line.split(",")) #多选使用,分割不同标签 
            
            flag['is_text'] = False if flag['is_text'] else True
    
        kwargs['label_as_key'] = False
        return cls().generate_datasets(js,**kwargs)

class LoadExcel(LoadDataset):
    @classmethod
    def load_dataset(cls,**kwargs):
        '''
            适用于如下类型的数据格式：
            | text | label  | dataset|
            |text_1| label_1|  train |
            |text_2| label_2|  train |
            |text_3| label_3|  valid |
            |text_4|      |  valid |
            ...
            或者：
            | text | label  |
            |text_1| label_1|
            |text_2| label_2|
            |text_3|      |
            |text_4| label_4|
            ...
            两类数据格式
        '''
        #读数据
        path = kwargs['path']
        if isinstance(path,str):
            suffix = path.split('.')[-1]
            if suffix in ['xlsx','xls']:
                df = pd.read_excel(path)
            elif path.split('.')[-1] == 'csv':
                df = pd.read_csv(path)
            else:
                assert 0,'数据集后缀应该是.csv或.xlsx或.xls，但是是{}'.format(suffix)
        elif isinstance(path,pandas.core.frame.DataFrame):
            df = path
        else:
            assert 0,'path输入应该是字符串str或者pandas.core.frame.DataFrame，但是是{}'.format(type(path))
            

            
        #数据集
        js = {kwargs['train']:defaultdict(list),kwargs['valid']:defaultdict(list),kwargs['test']:defaultdict(list)}
        datasets = {kwargs['train']:defaultdict(list),kwargs['valid']:defaultdict(list),kwargs['test']:defaultdict(list)}
        mapping = {kwargs['train']:'train', kwargs['valid']:'valid', kwargs['test']:'test'}
        
        #判断数据格式
        if kwargs['dataset'] in df.columns :
            for index in df.index:
                #label 为空自动被移入 testset
                if np.isnan(df.loc[index,kwargs['label']]):
                    js[kwargs['test']][kwargs['text']].append(df.loc[index,kwargs['text']])
                #判断是训练集
                elif df.loc[index,kwargs['dataset']] == kwargs['train']:
                    js[train][kwargs['text']].append(df.loc[index,kwargs['text']])
                    js[train][kwargs['label']].append(df.loc[index,kwargs['label']])
                #判断是验证集
                elif df.loc[index,kwargs['dataset']] == kwargs['valid']:
                    js[valid][kwargs['text']].append(df.loc[index,kwargs['text']])
                    js[valid][kwargs['label']].append(df.loc[index,kwargs['label']])
                #判断是测试集
                elif df.loc[index,kwargs['dataset']] == kwargs['test']:
                    js[kwargs['test']][kwargs['text']].append(df.loc[index,kwargs['text']])
                    js[kwargs['test']][kwargs['label']].append(df.loc[index,kwargs['label']])
                else:
                    assert 0,'dataset一列下的元素只能是{},{},{}中的一个，但是现在是{}'.format(kwargs['train'],kwargs['valid'],kwargs['test'],df.loc[index,kwargs['dataset']])
        else:
            for index in df.index:
                #label 为空自动被移入 testset
                if np.isnan(df.loc[index,kwargs['label']]):
                    js[kwargs['test']][kwargs['text']].append(df.loc[index,kwargs['text']])
                elif random.random() < kwargs['split']:
                    js[kwargs['train']][kwargs['text']].append(df.loc[index,kwargs['text']])
                    js[kwargs['train']][kwargs['label']].append(df.loc[index,kwargs['label']])
                else:
                    js[kwargs['valid']][kwargs['text']].append(df.loc[index,kwargs['text']])
                    js[kwargs['valid']][kwargs['label']].append(df.loc[index,kwargs['label']])
        
        kwargs['label_as_key'] = False
        return cls().generate_datasets(js,**kwargs)