from multiprocessing import Pool

from collections import defaultdict,OrderedDict
from transformers.configuration_utils import PretrainedConfig
import torch #for torch.cuda.is_available
from ..data.utils import load_dataset
import os
from typing import List, Optional, Union , Tuple, Collection, Dict
from tqdm import tqdm
import numpy as np


class ModelBase:
    def __init__(self,*args,**kwargs):
        self.model = None
        self.config = PretrainedConfig(
            package = 'envtext',
            liscence = 'Apache Lisence',
            contact = 'bi.huaibin@foxmail.com',
            mirror='https://mirror.nju.edu.cn/hugging-face-models',
            max_length = 510,
        )
        self.sp_tokenizer = None
        self.tokenizer = None
        self.visualizer = None
        self.datasets = None
        self.data_config = None
        self.args = None
        self.trainer = None
        self.result = OrderedDict()
        self.scorer = None
        self.training_results = defaultdict(list)
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'


        
    def set_attribute(self,**kwargs):
        '''
        设置参数
        '''
        self.config.update(kwargs)
    
    def clear_result(self):
        '''
        清除结果
        '''
        self.result = OrderedDict()
        
    @property
    def device(self):
        if self.model and hasattr(self.model, "device"):
            return self.model.device
        else:
            return self._device

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
    
    def set_device(self,device):
        '''
        设置模型运行的设备，cpu或cuda等
        '''
        self._device = device
        self.model = self.model.to(device)

        if self.scorer:
            self.scorer = self.scorer.to(device)



    def align_result_with_dataframe(self,df, column_name = None, prefix = '',inplace = False):
        '''
        对齐结果和pandas.DataFrame格式的数据。
        
        Args:
            df `pandas.DataFrame`:
                pandas.DataFrame格式的文件
           
            column_name [Optional] `str`: 默认为None
                文本列的列名
                如果值为None，则默认index是文本
           
            prefix [Optional] `str`: 默认为''
                结果的前缀
                例如：
                    prefix = "气候变化"
                    结果的列名为："气候变化label","气候变化p"...
                    
            inplace [Optional] ``: 默认为False
                是否原地改变值，不反回新的pandas.DataFrame
       
       Returns:
           df `pandas.DataFrame`:
               对齐后的pandas.DataFrame格式的文件
       '''
        import pandas as pd
        
        if inplace :
            df = df.copy()
            
        result_texts = list(self.result.keys())
        if len(result_texts) == 0:
            print('self.result的结果为空')
            return df
        
        sample_dict = {k:None for k,v in self.result[result_texts[0]].items() }

        key_mapping = {k:prefix+str(k) for k,v in sample_dict.items()}

        def return_result_columns(text):
            if text in self.result.keys():
                return self.result[text]
            else:
                return sample_dict.copy()

        if column_name is not None:
            if column_name not in df.columns:
                assert 0,f"输入的column_name必须是df的列名，但现在是{column_name}"
            else:
                result_columns = df[column_name].apply(return_result_columns)
                result_columns = pd.DataFrame(result_columns.values.tolist())        
        else:
            result_columns = list(map(return_result_columns,df.index))
            result_columns = pd.DataFrame(result_columns)    

        for col in result_columns.columns:
            df[key_mapping[col]] = result_columns[col].values
        
        if not inplace:
            return df
        
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
        
        if dir_name and not os.path.exists(dir_name):
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
            raise NotImplementedError() 
    
    def _save_result2json(self,path):
        from ..utils.json_ops import write_json
        write_json(path,self.result)
    
    def _save_result2csv(self,path):
        '''
        保存到csv
        '''
        import pandas as pd
        df = pd.DataFrame(self.result).transpose()
        df.to_csv(path,index = True,encoding="utf_8_sig")

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
            self.model.save_pretrained(save_path)
        else:
            torch.save(self.model,os.path.join(save_path,'pytorch_model.bin'))
        
        if hasattr(self.tokenizer,'save_pretrained'):
            self.tokenizer.save_pretrained(save_path)

        if self.config and hasattr(self.config,'save_pretrained'):
            self.config.save_pretrained(save_path)

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
        


    def update_data_config(self,dataset,**kwargs):
        '''
        更新 self.data_config，可以继承后实现
        '''
        pass

    def _calc_resample_prob(self,text,label,**kwargs):
        '''
        计算重采样概率，如果需要在训练时对样本进行重采样的话，可以继承后实现
        '''
        return 0.3

    def _resampling_dataset(self,dataset):
        #计算重采样概率
        if "resample_prob" not in dataset:
            if not hasattr(self, "_calc_resample_prob"):
                from warnings import warn
                warn("没有_calc_resample_prob函数，需要实现")
                return dataset
            
            #更新data_config
            if not hasattr(self, "data_config") or self.data_config is None:
                self.update_data_config(dataset)
                
            #为每一条样本计算概率
            probs = []
            from tqdm import tqdm
            for idx in tqdm(range(len(dataset["text"])),desc="计算重采样概率"):
                data = {k:dataset[k][idx] for k in dataset}
                probs.append(self._calc_resample_prob(**data))

            dataset["resample_prob"] = probs

        #设置重采样流程中，过滤数据集的遍数
        if not hasattr(self.config, "resampling_ratio"):
            self.set_attribute(resampling_ratio=3)
        
        import random
        from copy import deepcopy
        new_dataset = deepcopy(dataset)
        #进行采样，如果随机种子小于采样概率，则进行采样
        for i in range(self.config.resampling_ratio):
            for idx in tqdm(range(len(dataset["text"])),leave=False,desc=f"进行第{i}遍重采样"):
                if random.random() < dataset["resample_prob"][idx]:
                    for k in dataset:
                        new_dataset[k].append(dataset[k][idx])
        return new_dataset
    
    def _align_input_texts(self,texts):
        '''
        对齐输入文本的格式
        '''
        if isinstance(texts,str):
            if os.path.exists (texts):#如果是地址
                import warnings
                warnings.warn("正在从文件里读取数据。。。")
                texts = list(filter(lambda x: len(x.strip()) > 0, open(texts,"r",encoding="utf-8"))) 
            else:
                texts = [texts]
        elif isinstance(texts,list):
            pass
        elif isinstance(texts,tuple):
            texts = [t for t in texts]  
        else:
            import pandas as pd
            if isinstance(texts,pd.Series):
                texts = texts.apply(str).values.tolist()
            elif isinstance(texts,pd.DataFrame):
                texts = list(map(str,texts.index.to_list()))
            else:
                assert 0,'文本格式不支持，请输入 str, List[str], Tuple(str), pandas.Series ,pandas.DataFrame等格式的文本'
        
        return texts
    
    def predict(self,texts : Union[List[str],Tuple[str],str], 
                    batch_size : Union[str,int] = 8,
                    max_length : Union[int,None] = 510,
                    auto_group : bool = True,
                    min_group_size : Union[int,str] = "auto",
                    save_result : bool = True,
                    print_result : Union[bool,str] = "auto",
                    return_result : bool = True,
                    multiprocess : bool = False , 
                    return_logits : bool = False,
                    **kwargs):
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
               是否可视化结果，对于大批量的文本，建议设置为False
            
           return_result `bool`:
                默认为True
            
           save_result: 
               默认为True
               是否保存结果

            batch_size `int` or `str`: auto
                默认为auto，自动设置batchsize

            auto_group `bool`:
                默认为True，自动选择是否按照文本长度进行group

           multiprocess `bool`: = False
                多进程进行预处理和后处理 
            
            max_length `int`= 510
                文本最长的长度，默认是510
       '''
        texts = self._align_input_texts(texts)

        if multiprocess: #多进程处理，对大批量数据可以加速
            with Pool() as p:
                inputs = []
                for text in texts:
                    inputs.append(p.apply_async(self.preprocess,(text,),kwargs).get())
        else:
            input_list = list(map(lambda x:self.preprocess(x,**kwargs),texts))

            #整理不同格式的输入
            # List[文本1，文本2，文本3]
            if isinstance(input_list[0], str):
                inputs = input_list

            # dict{"text":[文本1，文本2，文本3],"label":[标签1，标签2，标签3] ...}
            elif isinstance(input_list[0], dict):
                inputs = {
                 k : [v[k] for v in input_list]  for k in input_list[0]
                }
    
            # tuple([文本1，文本2，文本3],[标签1，标签2，标签3] ...}
            elif isinstance(input_list[0], tuple):
                inputs = (
                   [v[0] for v in input_list] for i in range(len(input_list[0]))
                )
            
            else:
                assert 0,'预处理输入格式不支持'


        logits = self.get_logits(inputs, batch_size = batch_size, max_length = max_length,
                                auto_group = auto_group , min_group_size = min_group_size,
                                **kwargs)

        #返回logits
        if return_logits:
            return logits

        #将logits拆分，整理输出
        if isinstance(logits, (tuple,list)):
            outputs = zip(*logits)
        elif isinstance(logits, dict):
            outputs = map(lambda x: dict(zip(logits.keys(),x)), zip(*logits.values()))
        else:
            outputs = logits

        if multiprocess: #多进程处理，对大批量数据可以加速
            with Pool() as p:
                results = []
                for args in zip(texts,outputs):
                    results.append(p.apply_async(self.postprocess,args,kwargs).get())
        else:
            results = list(map(lambda input,output : self.postprocess(input, output, **kwargs),input_list, outputs))


        if  ((print_result == "auto" and len(texts) < batch_size) or (print_result is True)) and self.visualizer :
            for text,result in zip(texts, results):
                if isinstance(result,dict):
                    self.visualizer(text = text,**result)
                elif isinstance(result,(tuple,list)):
                    self.visualizer.render(text, *result)
                else:
                    self.visualizer.render(text, result)

        #保存结果至 self.result
        if save_result:
            if isinstance(inputs, list):
                texts = inputs
            elif isinstance(inputs, dict):
                texts = inputs["text"]
            elif isinstance(inputs, tuple):
                texts = inputs[0]
            else:
                assert 0,'预处理输入格式不支持'

            for text,result in zip(texts,results):
                self.result[text] = result

        #返回结果
        if return_result:
            return results
        

    def __call__(self,*args,**kwargs):
        return self.predict(*args,**kwargs)
        
    def compute_metrics(self,eval_pred):
        '''
        计算预测结果评价指标，需要继承之后实现，默认返回为空的dict
        
        Args:
            eval_pred `Tuple(numpy.ndarray)`:
                eval_pred = (预测结果的概率分布,真实结果)
                
       
       return: `dict` 
        '''
        return {}
    
    def preprocess(self,text,**kwargs):
        '''
        需要继承后重新实现，对文本进行必要的预处理（清洗）。不是tokenize。
        
        '''
        return text
    

    def preprocess_for_scorer(self,text,pred,**kwargs):
        '''
        评价模型的预处理步骤
        '''
        return text_or_tokens


    def train_scorer(self,text,pred,**kwargs):
        '''
        训练模型评价函数，需要继承后实现
        '''
        pass


    def postprocess(self, text ,logits, **kwargs):
        '''
        需要继承之后重新实现
        如果选择保存结果，结果保存在self.result里：
        eg: 
            self.result[text] = {
                'label':预测的结果,
                }
                
        Args:
           text `str` :
                预测的文本
           logits `numpy.ndarray`
                模型输出的结果 logits
                
        Kwargs:
           topk [Optional] `int`： 默认为5
               报告预测概率前topk的结果。
        '''
        results = None
        return results
    

    @torch.no_grad()
    def get_logits(self,inputs : Union[List,Tuple,Dict], 
                    batch_size = 8 ,
                    max_length = 510,
                    auto_group = True, 
                    min_group_size = 24,
                    **kwargs):

        '''
        获得模型预测的logits
        
        Args:
            inputs: list of texts or dict of lists
                [文本1，文本2，文本3] `str`: 文本
                or 
                {
                    "text": [文本1,文本2,文本3],
                    "label": [标签1,标签2,标签3],
                    ...
                }
        Kwargs:

        '''
        # tokens=self.tokenizer.encode(text, max_length = self.max_length, return_tensors='pt',add_special_tokens=True,truncation = True)

        #对文本进行重排，相似长度的文本放在一起
        #一致性指标 std() ?< 1 

        if max_length is None:
            max_length = self.max_length

        if max_length > 510:
            from warnings import warn
            warn(f"max_length为{max_length}，但是长度最长为510，将进行截断")
            
        if isinstance(inputs,list):
            texts = inputs
        elif isinstance(inputs,dict):
            texts = inputs["text"].copy()
        elif isinstance(inputs,tuple):
            texts = inputs[0].copy()

        if auto_group and len(texts) > batch_size:
            print("正在对长度相似的文本进行聚合，将会加速推理...")
            if isinstance(min_group_size, int) and min_group_size > 0:
                min_group_size = int(min_group_size)
            else:
                if min_group_size != "auto":
                    Warning("min group size 设置有问题，自动转为10*batchsuze")
                min_group_size = 10 * batch_size

            import numpy as np
            length_of_text = np.array(list(map(len,texts)))
            index = length_of_text.argsort()
            sorted_texts = [texts[i] for i in index]

            #初始化group
            if isinstance(inputs,list):
                groups = [sorted_texts[:batch_size]]
            elif isinstance(inputs, dict):
                groups = [
                    {k:[v[j] for j in index[:batch_size] ]  for k,v in inputs.items()}
                ]
            elif isinstance(inputs, tuple):
                groups = [
                    ([v[j] for j in index[:batch_size]] for v in inputs)
                ]

            # import pdb;pdb.set_trace();
            #方差函数
            import numpy as np
            std = lambda list_of_texts : np.std(list(map(len,list_of_texts))) if len(list_of_texts) > 5 else 0
            
            for t in range(batch_size,len(index)):
                if std(sorted_texts[t - batch_size + 1: t + 1]) > 1 and (
                    isinstance(inputs, list) and len(groups[-1]) > min_group_size
                    or isinstance(inputs, dict) and len(groups[-1]["text"]) > min_group_size
                    or isinstance(inputs, tuple) and len(groups[-1][0]) > min_group_size
                    ):
                    
                    if isinstance(inputs,list):
                        groups.append([sorted_texts[t]])
                    elif isinstance(inputs,dict):
                        groups.append(
                            {k: [v[index[t]]] for k,v in inputs.items()}
                        )
                    elif isinstance(inputs, tuple):
                        groups.append(
                            ([v[index[t]]] for v in inputs)
                        )
                else:
                    if isinstance(inputs,list):
                        groups[-1].append(sorted_texts[t])
                    elif isinstance(inputs,dict):
                        for k in inputs:
                            groups[-1][k].append(inputs[k][index[t]])
                    elif isinstance(inputs, tuple):
                        for idx in range(len(inputs)):
                            groups[-1][idx].append(inputs[idx][index[t]])
                    
                    
            logits = []
            for group_id,inputs in enumerate(groups):
                #模型推理
                #整理每个group中数据
                if isinstance(inputs, list):
                    texts = inputs
                elif isinstance(inputs, dict):
                    texts = inputs["text"]
                elif isinstance(inputs, tuple):
                    texts = inputs[0]

                #计算平均长度
                avg_length = np.mean(list(map(len,texts)))

                #判断文本长度，进行截断
                if max(map(len,texts)) > 510:
                    from warnings import warn
                    warn(f"存在长度大于{max_length}的文本，将会进行截断，也可能会出现潜在的错误！")
                    texts = [t[:max_length] for t in texts]

                #编码文本
                tokens=self.tokenizer(texts, return_tensors='pt',add_special_tokens=True,padding = "longest").to(self.device)
                bar = tqdm(range(0,len(texts),batch_size),desc=f"正在推理group {group_id},平均文本长度{avg_length:.2f}...") if len(texts) > batch_size else range(0,len(texts),batch_size)
                from transformers.tokenization_utils_base import BatchEncoding
                for idx in bar:
                    if isinstance(tokens, (dict,BatchEncoding)):
                        #整理数据，
                        data = {k:v[idx:idx+batch_size] for k,v in tokens.items()}
                        #整合文本的编码和预处理部分的信息
                        if isinstance(inputs,dict):
                            data.update({
                            k: torch.tensor(v[idx:idx+batch_size],device = self.device)  for k,v in inputs.items() if k != "text"
                            })
                        logit = self.model(**data)
                    else:
                        logit = self.model(tokens[idx:idx+batch_size])
                    logit = self._detach(logit)
                    logits.append(logit)
        

            #reindex
            def fun(list_of_array,index):
                # tensor = torch.cat(list(map(torch.tensor,list_of_tensors)))
                lists = []
                for array in list_of_array:
                    lists += list(array)

                results = [None for idx in range(len(index))]
                for idx in range(len(index)) :
                    results[index[idx]] = lists[idx]
                return np.array(results)

            #聚合各个batch的结果
            if isinstance(logits[0],dict):
                logits = {k:fun([logit[k] for logit in logits],index)  for k in logits[0]}
            elif isinstance(logits[0],tuple):
                logits = tuple(fun([logit[i] for logit in logits],index) for i in range(len(logits[0])))
            else:
                logits = fun(logits,index)

            return logits
        else:
            #模型推理
            logits = []
            index = None

            #判断文本长度，进行截断
            if max(map(len,texts)) > 510:
                from warnings import warn
                warn("存在长度大于510的文本，将会进行截断，也可能会出现潜在的错误！")
                texts = [t[:510] for t in texts]

            #编码文本
            tokens=self.tokenizer(texts, return_tensors='pt',add_special_tokens=True,padding = "longest").to(self.device)
            bar = tqdm(range(0,len(texts),batch_size),desc="正在推理...") if len(texts) > batch_size else range(0,len(texts),batch_size)
            from transformers.tokenization_utils_base import BatchEncoding
            for idx in bar:
                if isinstance(tokens, (dict,BatchEncoding)):
                    #整理数据，
                    data = {k:v[idx:idx+batch_size] for k,v in tokens.items()}
                    #整合文本的编码和预处理部分的信息
                    if isinstance(inputs,dict):
                        data.update({
                        k: torch.tensor(v[idx:idx+batch_size],device = self.device)  for k,v in inputs.items() if k != "text"
                        })
                    logit = self.model(**data)
                else:
                    logit = self.model(tokens[idx:idx+batch_size])
                
                logit = self._detach(logit)

                logits.append(logit)
            
            import numpy as np
            #聚合各个batch的结果
            if isinstance(logits[0],tuple):
                logits = tuple(np.concatenate([logit[i] for t in logits] ,axis=0) for i in range(len(logits[0])))
            elif isinstance(logits[0],dict):
                logits = {k: np.concatenate([logit[k] for t in logits],axis=0 ) for k in logits[0]}
            else:
                logits = np.concatenate(logits, axis=0)
            
            return logits


    def _detach(self,logit):
        '''
        logit:
            tuple(tensor)
            dict(key:tensor)
            tensor
        
        return:
            tuple(numpy.array)
            dict(key: numpy.array)
            numpy.array
        '''
        if isinstance(logit,tuple):
            return tuple(t.cpu().detach().numpy() for t in logit)
        elif isinstance(logit,dict):
            return {k:t.cpu().detach().numpy() for k,t in logit.items()}
        else:
            return logit.cpu().detach().numpy()


    def _raw_report(self,dic):
        keys = list(dic.keys())
        if isinstance(dic[keys[0]],list):
            report = '\t  \t'
            for i in range(1,len(dic[keys[0]])+1):
                report += f'Epoch{i} \t'
            report += '\n'

            for k,values in dic.items():
                report += f'{k}: \t'
                for v in values:
                    try:
                        report += '{:.4f} \t'.format(v)
                    except:
                        report += ' --- \t'
                report += '\n'
        else:
            report = ''
            for k,v in dic.items():
                try:
                    report += '{} \t : {:.4f} \t \n'.format(k,v)
                except:
                    report += '{} \t : --- \t \n'.format(k)
        print(report)
                                 
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

        keys = list(dic.keys())
        if isinstance(dic[keys[0]],list):
            for values in zip(*dic.values()):
                markdown += '|'
                for v in values:
                    try:
                        markdown += '{:.4f}|'.format(v)
                    except:
                        markdown += ' --- |'
                markdown += '|  \n'
        else:
            markdown += '|'
            for k,v in dic.items():
                try:
                    markdown += '{:.4f}|'.format(v)
                except:
                    markdown += ' --- |'
            markdown += '  \n'
            
        from IPython.display import display_markdown
        display_markdown(markdown,raw = True)

    def _report(self,dic):
        from ..utils.notebook_ops import in_notebook
        if in_notebook():
            self._ipython_report(dic)
        else:
            self._raw_report(dic)

    def train(self,my_datasets,epoch ,batch_size , learning_rate ,save_path ,checkpoint_path,**kwargs):
        '''
        需要继承之后重新实现
        '''
        pass

    @torch.no_grad()
    def eval(self,my_datasets,epoch ,batch_size , learning_rate ,save_path ,checkpoint_path,**kwargs):
        '''
        需要继承之后重新实现
        '''
        pass

    def __getitem__(self,text):
        '''
        返回结果
        '''
        if text in self.result:
            return self.result[text]            
        else:
            return None

    def load_dataset(self,*args,**kwargs):
        '''
        读取训练数据集。
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
            new_kwargs = self.config.to_dict()
            new_kwargs.update(kwargs)
        except:
            new_kwargs = kwargs
        self.datasets,self.data_config = load_dataset(*args,**new_kwargs)

        print("*"*7,"读取数据集成功","*"*7)