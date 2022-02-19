from .bert_base import BertBase
import torch # for torch.topk
import torch.nn.functional as F
from transformers import BertForSequenceClassification,BertTokenizerFast,BertConfig
import numpy as np #for np.argmax
from ..utils.metrics import metrics_for_cls

class BertCLS(BertBase): 
    '''
    Bert分类模型
    
    Args:
        path `str`: 默认：None
            预训练模型保存路径，如果为None，则从celtics1863进行导入预训练模型
            
        config [Optional] `dict` :
            配置参数
            
   Kwargs:
        labels [Optional] `List[int]` or `List[str]`: 默认None
            分类问题中标签的种类。
            分类问题中和num_labels必须填一个，代表所有的标签。
            默认为['LABEL_0','LABEL_0']

        num_labels [Optional] `int`: 默认None
            分类问题中标签的数量。
            分类问题中和num_labels必须填一个，代表所有的标签。
            默认为2
       
        max_length [Optional] `int`: 默认：512
           支持的最大文本长度。
           如果长度超过这个文本，则截断，如果不够，则填充默认值。
    '''
    def initialize_bert(self,path = None,config = None,**kwargs):
        super().initialize_bert(path,config,**kwargs)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path,config = self.config)
        if self.key_metric == 'loss':
            if self.num_labels == 2:
                self.set_attribute(key_metric = 'f1')
            else:
                self.set_attribute(key_metric = 'macro_f1')
        
    def align_config(self):
        super().align_config()
        if self.labels:
            if self.label2id:
                label2id = self.label2id
                id2label = {v:k for k,v in self.label2id.items()}
            elif self.id2label:
                label2id = {v:k for k,v in self.id2label.items()}
                id2label = self.id2label
            
            self.update_config(num_labels = self.num_labels,
                         label2id = label2id,
                         id2label = id2label)   
        elif self.num_labels:
            if self.label2id:
                label2id = self.label2id
                labels = self.label2id.keys()
                id2label = {v:k for k,v in self.label2id.items()}
                
                self.update_config(labels = labels,
                         label2id = label2id,
                         id2label = id2label)   
            elif self.id2labels:
                id2label = self.id2label
                labels = [v for k,v in self.id2labels.items()]
                label2id = {v:k for k,v in self.id2label.items()}
                self.update_config(labels = labels,
                         label2id = label2id,
                         id2label = id2label)  
            else:
                labels = [f'LABEL_{i}' for i in range(self.num_labels)]
                self.update_config(num_labels = self.num_labels,
                            labels = labels)
        
        else:
            self.update_config(num_labels = 2,
                         labels = ['LABEL_0','LABEL_1'],
                         id2label = {0:'LABEL_0',1:'LABEL_1'},
                         label2id = {'LABEL_0':0,'LABEL_1':1},
                         )

    def postprocess(self,text,logits,topk=5,print_result = True,save_result = True):
        logits = F.softmax(torch.tensor(logits),dim=-1)
        topk = topk if logits.shape[-1] > topk else logits.shape[-1]
        p,pred = torch.topk(logits,topk)
        if print_result:
            self._report_per_sentence(text,pred,p)
        
        if save_result:
            self._save_per_sentence_result(text,pred,p)

    def _report_per_sentence(self,text,pred,p):
        log = f'text:{text} \n'
        for i,j in  zip(pred,p):
            log += '\t pred_classes:{}, \t probability:{:.4f} \n'.format(self.id2label[i.item()],j)
        print(log)
    
    def _save_per_sentence_result(self,text,pred,p):
        result = {}
        for topk,(i,j) in enumerate(zip(pred,p)):
            if topk == 0:
                result['label'] = self.id2label[i.item()]
                result['p'] = '{:.4f}'.format(j.item())
            else:
                result[f'top{topk+1} label'] = self.id2label[i.item()]
                result[f'top{topk+1} p'] = '{:.4f}'.format(j.item())
        
        self.result[text] = result
        
    def compute_metrics(self,eval_pred):
        dic = metrics_for_cls(eval_pred)
        return dic