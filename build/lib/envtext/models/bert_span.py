# -*- coding: utf-8 -*-
"""
Reference code url: https://github.com/gaohongkui/GlobalPointer_pytorch/tree/main/models
"""

import torch
import numpy as np
from .bert_base import BertBase
import torch #for torch.argmax torch.max
import torch.nn.functional as F
from torch import nn #for nn.Dropout nn.LSTM nn.Linear
from transformers import BertPreTrainedModel,BertModel
# from ..tokenizers import WoBertTokenizer
from ..utils.metrics import metrics_for_ner
from einops import rearrange
from .span_base import SpanBase
import jieba


class BertSpanModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSpanModel, self).__init__(config)
        self.bert = BertModel(config)

        if hasattr(config, "num_entities"):
            self.num_entities = config.num_entities
        else:
            self.num_entities = 1 #默认1类实体
            
    
        if hasattr(config, "max_entity_num"):
            self.max_entity_num = config.max_entity_num
        else:
            self.max_entity_num = 16 #默认一个句子最多16份实体


        self.lstm = nn.LSTM(
            input_size= config.hidden_size,  # 768 + vector
            hidden_size=config.hidden_size,  # 768
            batch_first=True,
            num_layers=1,
            dropout=0,  # 0.5
            bidirectional=False
            )


        #预测头实体
        self.start = nn.Linear(config.hidden_size, self.num_entities) #start,end

        #预测尾实体
        self.end = nn.Sequential(
            nn.Linear(config.hidden_size + self.num_entities, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, self.num_entities),
        )


    def forward(self, input_ids, vectors = None , token_type_ids=None, attention_mask=None, labels=None,
              position_ids=None, inputs_embeds=None, head_mask=None):

        outputs = self.bert(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
        
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        seq_outputs = outputs[0]

        if vectors is not None:
            vectors = vectors.clone().permute(0,2,1)        

        lstm_output,_ = self.lstm(seq_outputs)

        # logits = self.classifier(lstm_output)

        start = self.start(lstm_output)
        end = self.end(torch.cat([lstm_output,start],dim=-1))

        logits = rearrange(torch.stack([start,end]), "e b l n -> b n l e")


        #对logits进行重整
        # logits = rearrange(logits,"b l (n m) ->b n l m",m = 2)

        # 预测的实体
        logits_for_pred = logits.clone().detach()
        #词汇增强
        if vectors is not None:
            logits_for_pred = logits_for_pred - (1 - vectors > 0).unsqueeze(1).expand_as(logits_for_pred) * 1e12

        preds = self._decode(logits_for_pred, attention_mask.clone().detach())

        outputs = (preds, logits)
        

        # print(labels.nonzero())

        if labels is not None:
            # weight = labels != -100
            #[B, N ,L,2]
            weight = attention_mask.unsqueeze(1).unsqueeze(-1).expand_as(labels)

            #词汇增强
            if vectors is not None:
                word_mask = vectors.unsqueeze(1).expand_as(labels) > 0
                # weight = weight + word_mask # 提高关键字的权重

            loss = F.binary_cross_entropy_with_logits(logits, labels.float(),weight = weight)

            # print("loss is", loss.item())

            # yes = torch.logical_and(labels == 1, loss > 0)
            # yes = torch.logical_and(yes, weight > 0)
            # print( "yes is ",yes.sum())

            # word_yes = torch.logical_and(labels == 1, word_mask > 0)
            # print( "word yes is ",word_yes.sum())

            outputs = (loss,) + outputs


        return outputs


    def _decode(self,logits,attention_mask, T = 0):
        '''
        logits: [B,N,L,2]
        '''

        B,N,L,_ = logits.shape
        entities = torch.ones(B,self.max_entity_num,3,dtype=int) * (-100)
        for b in range(B):
            offset = 0
            length = attention_mask[b].sum()
            for n in range(N):
                start = 0
                end = 0
                for l in range(L):
                    if logits[b,n,l,0] > T:
                        start = l

                    if logits[b,n,l,1] > T:
                        end = l
                        if end > start and start > 0 and end < length:

                            entities[b,offset,0] = n
                            entities[b,offset,1] = start
                            entities[b,offset,2] = end

        return entities                



class BertSpan(SpanBase,BertBase):
    def initialize_bert(self,path = None,config = None,**kwargs):
        super().initialize_bert(path,config,**kwargs)
        self.model = BertSpanModel.from_pretrained(self.model_path,config = self.config)
        # if self.key_metric == 'validation loss':
        # if self.num_entities == 1:
        #     self.set_attribute(key_metric = 'f1')
        # else:
        #     self.set_attribute(key_metric = 'macro_f1')
        
        self.set_attribute(key_metric = 'f1')


        if  self.datasets:
            print("正在jieba编码")
            for k,v in self.datasets.items():
                v["vectors"] = list(map(self._jieba_tokenizer,v["text"]))

    def _jieba_tokenizer(self,text):
        import re
        start_vector = [-100]
        end_vector = [-100]
        for word in jieba.cut(re.sub("\s","",text),HMM=False):
            for i in range(len(word)):
                if i == 0:
                    start_vector.append(1)
                else:
                    start_vector.append(0)

                if i == len(word) - 1:
                    end_vector.append(1)
                else:
                    end_vector.append(0)
        
        start_vector.append(-100)
        end_vector.append(-100)

        # cut
        if self.max_length > len(start_vector):
            start_vector += [-100] * (self.max_length - len(start_vector))
            end_vector += [-100] * (self.max_length - len(end_vector))

        elif self.max_length < len(start_vector):
            start_vector = start_vector[:self.max_length]
            end_vector = end_vector[:self.max_length]
        
        return [start_vector,end_vector]

    def preprocess(self, text, *args,**kwargs):
        text  = super().preprocess(text, *args,**kwargs)

        return {
            "text": text,
            "vectors": self._jieba_tokenizer(text)
        }
        