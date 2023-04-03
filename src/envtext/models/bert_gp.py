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
from .gp_base import GPBase
import jieba

class GlobalPointer(BertPreTrainedModel):
    def __init__(self, config):
        super(GlobalPointer, self).__init__(config)
        self.bert = BertModel(config)

        if hasattr(config, "num_entities"):
            self.num_entities = config.num_entities
        else:
            self.num_entities = 1 #默认1类实体
            
    
        if hasattr(config, "max_entity_num"):
            self.max_entity_num = config.max_entity_num
        else:
            self.max_entity_num = 32 #默认一个句子最多16份实体


        if hasattr(config, "inner_size"):
            self.inner_size = config.inner_size
        else:
            self.inner_size = config.hidden_size #默认一个句子最多16份实体


        self.qk = nn.Linear(config.hidden_size, 
                                self.num_entities * self.inner_size * 2)

        if hasattr(config, "treshold"):
            self.treshold = config.treshold
        else:
            self.treshold = 0

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings
        
    def forward(self, input_ids, input_text = None, vectors = None , token_type_ids=None, attention_mask=None, labels=None,
              position_ids=None, inputs_embeds=None, head_mask=None):

        outputs = self.bert(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
        
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        seq_outputs = outputs[0]

        B,L,_ = seq_outputs.shape

        qk = self.qk(seq_outputs)
        
        #调整句子shape， 
        qk = rearrange(qk,"b l (n f) -> b l n f",n = self.num_entities,f = self.inner_size * 2)
        #q: [B, L, F] k:[B ,L, F]
        q,k = torch.split(qk,self.inner_size,-1)

        # # import pdb;pdb.set_trace();
        # #相对位置编码
        # # pos_emb:(batch_size, seq_len, inner_dim)
        pos_emb = self.sinusoidal_position_embedding(B, L, self.inner_size)
        # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None,::2].repeat_interleave(2, dim=-1)
        q2 = torch.stack([-q[..., 1::2], q[...,::2]], -1)
        q2 = q2.reshape(q.shape)
        q2 = q2 * cos_pos + q2 * sin_pos

        k2 = torch.stack([-k[..., 1::2], k[...,::2]], -1)
        k2 = k2.reshape(k.shape)
        k2 = k2 * cos_pos + k2 * sin_pos

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', q2, k2)
        # logits = torch.einsum('bmhd,bnhd->bhmn', q, k)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(B, self.num_entities, L, L)
        logits = logits*pad_mask - (1-pad_mask)*1e12



        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12
        logits = logits/(self.inner_size**0.5)
        

        # 预测的实体

        # #vectors mask
        # #vectors: [B, N , L, L]
        logits_for_pred = logits.clone().detach()
        if vectors is not None:
            logits_for_pred = logits_for_pred - (1-vectors[:,:L,:L].unsqueeze(1).expand_as(logits))*1e12
    
        preds = self._decode(logits_for_pred, T = self.treshold)

        outputs = (preds, logits)
        
        if labels is not None:
            # loss = self.multilabel_categorical_crossentropy(logits, labels)
            loss = self._get_training_loss(logits, labels)

            # labels[mask] = -100
            
            # w = mask == 0
            # loss = F.binary_cross_entropy_with_logits(logits, labels.float(),weight=w)
            # print(loss)
            # tr = torch.logical_and(logits > 0,w).sum()
            # yes = (torch.logical_and(logits > 0,labels > 0)).sum().item()
            # print(
            #     f"当前batch中预测正确{yes}个，预测了{tr}个"
            # )

            outputs = (loss,) + outputs
        return outputs
        
    def multilabel_categorical_crossentropy(self, y_pred, y_true):
        """
        y_true:(batch_size, ent_type_size, seq_len, seq_len)
        y_pred:(batch_size, ent_type_size, seq_len, seq_len)
        """
        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
        y_pred_pos = y_pred - (1 - y_true) * 1e12 # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return (neg_loss + pos_loss).mean()

    def _get_training_loss(self,y_true, y_pred):
        """
        y_true:(batch_size, ent_type_size, seq_len, seq_len)
        y_pred:(batch_size, ent_type_size, seq_len, seq_len)
        """
        batch_size, ent_type_size = y_pred.shape[:2]
        y_true = y_true.reshape(batch_size * ent_type_size, -1)
        y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
        loss = self.multilabel_categorical_crossentropy(y_true, y_pred)
        return loss


    def _decode(self,logits,T=0):
        '''
        logits: [B,N,L,L]
        '''
        logits[:, :, [0, -1]] -= torch.inf

        B,N,L,_ = logits.shape
        entities = torch.ones(B,self.max_entity_num,3,dtype=int) * (-100)
        for b in range(B):
            offset = 0
            
            for l, start, end in zip(*torch.where(logits[b] > T)):
                if offset >= self.max_entity_num:
                    break
                if end >= start:
                    # entities[b].append([l,start,end])
                    entities[b,offset,0] = l
                    entities[b,offset,1] = start
                    entities[b,offset,2] = end
                    offset += 1

        return entities                



class BertGP(GPBase,BertBase):
    def initialize_bert(self,path = None,config = None,**kwargs):
        super().initialize_bert(path,config,**kwargs)
        self.model = GlobalPointer.from_pretrained(self.model_path,config = self.config)
        # if self.key_metric == 'validation loss':
        # if self.num_entities == 1:
        #     self.set_attribute(key_metric = 'f1')
        # else:
        #     self.set_attribute(key_metric = 'macro_f1')
        
        self.set_attribute(key_metric = 'f1')

        if self.datasets and hasattr(self.config,"jieba"):
            print("正在jieba编码")
            for k,v in self.datasets.items():
                v["vectors"] = list(map(self._jieba_tokenizer,v["text"]))


    def _jieba_tokenizer(self,text):
        import re
        vector = np.zeros((self.max_length,self.max_length),dtype=int)

        starts = []
        ends = []

        ith = 1
        for word in jieba.lcut(re.sub("\s","",text),HMM=False):
            for i in range(len(word)):
                if i == 0:
                    starts.append(ith)

                if i == len(word) - 1:
                    ends.append(ith+i)
            ith += len(word)

        from itertools import product
        for start, end in product(starts,ends):
            if start >= self.max_length - 1:
                break

            if end > start and end < self.max_length - 1:
                vector[start,end] = 1
            
        return vector
