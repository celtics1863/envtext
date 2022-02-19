from .bert_base import BertBase
import torch # for torch.topk
import torch.nn.functional as F
from transformers import BertForMaskedLM
import numpy as np #for np.argmax
from queue import PriorityQueue as PQ
import math #for math.log()

class BertMLM(BertBase):
    '''
    Bert完型填空模型，支持多个[MASK]
    
    Args:
        path `str`: 默认：None
            预训练模型保存路径，如果为None，则从celtics1863进行导入预训练模型
            
       config [Optional] `dict` :
            配置参数
   Kwargs:
       
        max_length [Optional] `int`: 默认：512
           支持的最大文本长度。
           如果长度超过这个文本，则截断，如果不够，则填充默认值。
    '''
    def initialize_bert(self,path = None,config = None,**kwargs):
        super().initialize_bert(path,config,**kwargs)
        self.model = BertForMaskedLM.from_pretrained(self.model_path,config = self.config)

    def predict_per_sentence(self,text,topk = 5,print_result = True,save_result = True):
        tokens=self.tokenizer.encode(text, max_length = self.max_length, return_tensors='pt',add_special_tokens=True,truncation = True)[0].to(self.model.device)
        # print(mask_input_ids)
        mask_inds = self._get_mask_id(tokens)
        if len(mask_inds) == 1:
            probs,preds = self.predict_per_mask(tokens,mask_inds[0],topk)
            preds = self.tokenizer.convert_ids_to_tokens(preds)
            probs = [p.cpu().item() for p in probs]
        elif len(mask_inds) == 0:
            preds = []
            probs = []
        else:
            topk_preds = self.beam_search(tokens,mask_inds,topk)
            preds,probs = [],[]
            for prob,pred,token in topk_preds:
                preds.append(self.tokenizer.convert_ids_to_tokens(pred))
                probs.append(math.exp(-prob))
        
        if print_result:
            self._report_per_sentence(text,preds,probs)
        
        
        if save_result:
            self._save_per_sentence_result(text,preds,probs)
            
    def predict_per_mask(self,tokens,mask_ind,topk):
        logits = self.model(tokens.unsqueeze(0))[0]
        probs = F.softmax(logits, -1)
        per_probs = probs[0, mask_ind, :]
        probs,preds = torch.topk(per_probs,k=topk)
        return (probs,preds)

    def beam_search(self,tokens,mask_inds,B):
        B_preds = []
        for mask_ind in mask_inds:
            if len(B_preds) == 0:
                probs,preds = self.predict_per_mask(tokens,mask_ind,B)
                for prob,pred in zip(probs,preds):
                    new_token = tokens.clone()
                    new_token[mask_ind] = pred
                    B_preds.append((-math.log(prob.item()),[pred],new_token))
            else:
                q = PQ(maxsize=B*B)
                for p,before_preds,old_tokens in B_preds:
                    probs,preds = self.predict_per_mask(old_tokens,mask_ind,B)
                    for prob,pred in zip(probs,preds):
                        new_tokens = old_tokens.clone()
                        new_tokens[mask_ind] = pred
                        new_preds = before_preds.copy()
                        new_preds.append(pred)
                        q.put((p,new_preds,new_tokens))
                B_preds = [q.get() for i in range(B)]
        return B_preds
                
    def _report_per_sentence(self,text,preds,probs):
        s = f"text:{text} \n"
        for pred,prob in zip(preds,probs):
            s += "  predict: {} ; probability: {:.4f} \n".format(pred,prob) 
        print(s)

    
    def _save_per_sentence_result(self,text,preds,probs):
        result = {}
        for topk,(pred,prob) in enumerate(zip(preds,probs)):
            if topk == 0 :
                result['label'] = ''.join(pred)
                result['p'] = prob
            else:
                result[f'top{topk+1} label'] = ''.join(pred)
                result[f'top{topk+1} p'] = prob
        
        self.result[text] = result
        
    def _get_mask_id(self,tokens):
        mask_ids = []
        for idx,token in enumerate(tokens):
            if token == self.tokenizer.mask_token_id:
                mask_ids.append(idx)
        return mask_ids
