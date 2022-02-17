from threading import Thread
from collections import defaultdict
import torch #for torch.cuda.is_available
from ..data.utils import load_dataset

class ModelBase:
    def __init__(self,*args,**kwargs):
        self.model = None
        self.config = None
        self.tokenizer = None
        self.datasets = None
        self.args = None
        self.trainer = None
        self.max_length = 512
        self.result = defaultdict(dict)
        self.key_metric = 'loss'
        self.training_results = defaultdict(list)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    def get_key_metric(self):
        return self.key_metric
    
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
        from .utils.json_ops import write_json
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
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self._best_model,os.path.join(save_path,'pytorch_model.bin'))


    def predict_sentences(self,text_list,**kwargs):
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

    def __call__(self,texts,**kwargs):
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
        
    def compute_metrics(self,eval_pred):
        self.key_metric = 'validation loss'
        '''
        需要继承之后实现
        '''
        return {}
    
    def predcit_per_sentence(self,text, **kwargs):
        '''
        需要继承之后实现
        如果要保存结果，需要保存在self.result里：
            self.result[text] = {
                'label':预测的结果,
                }
                
       一些常用的参数在本项目中实现，在继承的时候未必要实现：
           topk `int`： 
               默认为5,报告预测概率前topk的结果。
           
           print_result `bool`: 
               默认为True
               是否打印结果，对于大批量的文本，建议设置为False
               
           save_result: 
               默认为True
               是否保存结果
        '''
        pass
  
    def train(self,my_datasets,epoch ,batch_size , learning_rate ,save_path ,checkpoint_path,**kwargs):
        '''
        需要继承之后实现
        '''
        pass
    
            
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
            
#         return self.datasets,self.data_config