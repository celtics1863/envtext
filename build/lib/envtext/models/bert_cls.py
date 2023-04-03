from .bert_base import BertBase
import torch # for torch.topk
from torch import nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel,BertForSequenceClassification,BertTokenizerFast,BertConfig,BertModel
import numpy as np #for np.argmax
from ..utils.loss import FocalLoss
from .cls_base import CLSBase

class BertCLSModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCLSModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size,self.num_labels)

        if hasattr(config, "focal"):
            if hasattr(config, "alpha"):
                alpha = config.alpha
            else:
                alpha = 0.25
            
            if hasattr(config, "gamma"):
                gamma = config.gamma
            else:
                gamma = 2

            self.loss = FocalLoss(alpha, gamma, self.num_labels)
        else:
            self.loss = nn.CrossEntropyLoss()

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
            print(labels.shape,logits.shape)
            labels = labels.nonzero()[:,1]
            print(labels.shape,logits.shape)
            print(labels)
            loss = self.loss(logits,labels.long())
            outputs = (loss,) + outputs
        return outputs

class BertCLS(CLSBase,BertBase): 
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
        # self.model = BertForSequenceClassification.from_pretrained(self.model_path,config = self.config)
        self.model = BertCLSModel.from_pretrained(self.model_path,config = self.config)

        if self.key_metric == 'loss':
            if self.num_labels == 2:
                self.set_attribute(key_metric = 'f1')
            else:
                self.set_attribute(key_metric = 'macro_f1')
        