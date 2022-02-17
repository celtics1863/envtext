from .bert_base import BertBase
import torch # for torch.topk
import torch.nn.functional as F
from transformers import BertForSequenceClassification,BertTokenizerFast,BertConfig
# from ..tokenizers import WoBertTokenizer#,WoBertTokenizerFast
import numpy as np #for np.argmax
from ..utils.metrics import metrics_for_cls

class BertCLS(BertBase): 
    def initialize_bert(self,path = None,config = None,**kwargs):
        self.update_model_path(path)
        self.update_config(kwargs)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path,config = self.config)
    
    def predict_per_sentence(self,text,topk=5,print_result = True,save_result = True):
        tokens=self.tokenizer.encode(text, return_tensors='pt',add_special_tokens=True).to(self.model.device)
        logits = F.softmax(self.model(tokens)['logits'],dim=-1)
        topk = topk if logits.shape[-1] > topk else logits.shape[-1]
        p,pred = torch.topk(logits,topk)
        
        if print_result:
            self._report_per_sentence(text,pred[0].clone().detach().cpu(),p[0].clone().detach().cpu())
        
        if save_result:
            self._save_per_sentence_result(text,pred[0].clone().detach().cpu(),p[0].clone().detach().cpu())
            
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
                result[f'top{topk} label'] = self.id2label[i.item()]
                result[f'top{topk} p'] = '{:.4f}'.format(j.item())
        
        self.result[text] = result
        
    def compute_metrics(self,eval_pred):
        dic = metrics_for_cls(eval_pred)
        if 'f1' in dic.keys():
            self.key_metric = 'f1'
        else:
            self.key_metric = 'micro_f1'
#         self.training_results['confusion matrix'].append(confusion_matrix(eval_pred))
#         self.training_results['report'].append(dic)
        return dic