from .bert_base import BertBase

import torch # for torch.nonzero
from torch import nn #for nn.linear
import torch.nn.functional as F
from transformers import BertPreTrainedModel,BertModel
# from ..tokenizers import WoBertTokenizer
from ..utils.metrics import metrics_for_cls_with_binary_logits


class BertMultiCLS(BertPreTrainedModel):
    def __init__(self, config):
        super(BertMultiCLS, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size,self.num_labels)
        self.loss = nn.BCEWithLogitsLoss()
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
              position_ids=None, inputs_embeds=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        #使用[CLS]
        cls_output = outputs[0][:,0,:] 

        # 得到判别值
        logits = self.classifier(cls_output)

        outputs = (logits,)
        if labels is not None:
            loss = self.loss(logits,labels.float())
            outputs = (loss,) + outputs
        return outputs


class BertMultiChoice(BertBase):
    '''
    Bert多项选择模型
    
    Args:
        path `str`: 默认：None
            预训练模型保存路径，如果为None，则从celtics1863进行导入预训练模型
            
        config [Optional] `dict` :
            配置参数
   Kwargs:
        labels [Optional] `List[int]` or `List[str]`: 默认None
            分类问题中标签的种类。
            分类问题中和num_labels必须填一个，代表所有的标签。
            默认为['LABEL_0']

        num_labels [Optional] `int`: 默认None
            分类问题中标签的数量。
            分类问题中和num_labels必须填一个，代表所有的标签。
            默认为1
       
        max_length [Optional] `int`: 默认：512
           支持的最大文本长度。
           如果长度超过这个文本，则截断，如果不够，则填充默认值。
    '''
    def initialize_bert(self,path = None,config = None,**kwargs):
        super().initialize_bert(path,config,**kwargs)
        self.model = BertMultiCLS.from_pretrained(self.model_path,config = self.config)
        if self.key_metric == 'validation loss':
            if self.num_labels == 1:
                self.set_attribute(key_metric = 'f1')
            else:
                self.set_attribute(key_metric = 'macro_f1')
    
    def align_config(self):
        super().align_config()
        if self.labels:
            self.update_config(num_labels = len(self.labels))   
        elif self.num_labels:
            self.update_config(labels = list(range(self.num_labels)))
        else:
            self.update_config(num_labels = 1,
                         labels = ['LABEL_0'],
                         id2label = {0:'LABEL_0'},
                         label2id = {'LABEL_0':0},
                         )
            
    def postprocess(self,text, logits ,print_result = True, save_result = True):
        preds = np.nonzero(logits > 0.5)
        if print_result:
            self._report_per_sentence(text,preds,logits)
        
        if save_result:
            self._save_per_sentence_result(text,preds.clone().detach().cpu(),logits[0][preds].clone().detach().cpu())
            
    def _report_per_sentence(self,text,preds,probs):
        log = f'text: {text}\n'
        for pred,prob in zip(preds,probs) :
            log += '\t prediction: {} \t ; probability : {:.4f}\n'.format(self.id2label[pred],prob)
            self.result[text].append((self.id2label[pred],prob))
        print(log)
 
    def _save_per_sentence_result(self,text,preds,probs):
        result = {}
        for idx,(pred,prob) in enumerate(zip(preds,probs)) :
            if idx == 0:
                result['label'] = self.id2label[pred]
                result['p'] = prob
            else:
                result[f'label_{idx+1}'] = self.id2label[pred]
                result[f'p_{idx+1}'] = prob
        
        self.result[text] = result
        
    def compute_metrics(self,eval_pred):
        dic = metrics_for_cls_with_binary_logits(eval_pred)
        return dic