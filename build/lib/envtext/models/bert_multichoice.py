from .bert_base import BertBase

import torch # for torch.nonzero
from torch import nn #for nn.linear
import torch.nn.functional as F
from transformers import BertPreTrainedModel,BertModel
# from ..tokenizers import WoBertTokenizer
import numpy as np #for np.nonzero
from .mc_base import MCBase


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
        else:
            return logits


class BertMultiChoice(MCBase,BertBase):
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
    

            
    def postprocess(self,text, logits ,print_result = True, save_result = True, save_vis = None):
        logits = logits[0] # logits = (logits, ) ,fetch first of tuple
        
        def sigmoid(z):
            return 1/(1 + np.exp(-z))

        logits = sigmoid(logits)
        preds = np.nonzero(logits > 0.5)
        labels = [self.id2label[pred[0]] for pred in preds]

        # if print_result:
        #     labels = [self.id2label[pred[0]] for pred in preds] #pred[0] from convert numpy to int
        #     probs = [logits[pred][0] for pred in preds]
        #     self._visualize(text,labels,probs,save_vis)
        
        # if save_result:
        #     self._save_per_sentence_result(text,preds,logits)
        return labels

