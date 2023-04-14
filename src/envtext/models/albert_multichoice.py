from .bert_base import BertBase
import torch # for torch.nonzero
from torch import nn #for nn.linear
import torch.nn.functional as F
from transformers import AlbertModel,AlbertPreTrainedModel
import numpy as np #for np.nonzero
from .mc_base import MCBase


class AlbertMultiCLS(AlbertPreTrainedModel):
    def __init__(self, config):
        super(AlbertMultiCLS, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = AlbertModel(config)
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


class AlbertMultiChoice(MCBase,BertBase):
    def initialize_bert(self,path = None,config = None,**kwargs):
        super().initialize_bert(path,config,**kwargs)
        self.model = AlbertMultiCLS.from_pretrained(self.model_path,config = self.config)
        if self.key_metric == 'validation loss':
            if self.num_labels == 1:
                self.set_attribute(key_metric = 'f1')
            else:
                self.set_attribute(key_metric = 'macro_f1')
    

            
