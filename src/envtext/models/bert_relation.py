from .bert_base import BertBase
import torch # for torch.topk
from torch import nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel,BertForSequenceClassification,BertTokenizerFast,BertConfig,BertModel
import numpy as np #for np.argmax
from ..utils.loss import FocalLoss
from .relation_base import RelationBase
from itertools import product
from collections import defaultdict


class BertRelationModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertRelationModel, self).__init__(config)
        self.bert = BertModel(config)

        if hasattr(config, "num_rels"):
            self.num_rels = config.num_rels + 1
        else:
            self.num_rels = 2 #默认 yes or not
        

        self.rel_classifier = nn.Linear(config.hidden_size * 2,self.num_rels)

        if hasattr(config, "spo_counter"):
            self.potential_label_pair = defaultdict(set) # (s,o),r
            for k in self.config.spo_counter:
                s,p,o = k.split('\t')
                s = self.config.entity2id[s]
                p = self.config.relation2id[p]
                o = self.config.entity2id[o]
                self.potential_label_pair[s,o].add(p)

        if hasattr(config, "ner_encoding"):
            self.ner_encoding = config.ner_encoding #ner 编码方式，GP,BIO,BIOE,BIOES
        else:
            self.ner_encoding = 'Pointer' #默认 Pointer

        if hasattr(config, "max_triple_num"):
                self.max_triple_num = config.max_triple_num
        else:
            self.max_triple_num = 32

        if hasattr(config, "max_cands_triple_num"):
                self.max_cands_triple_num = config.max_cands_triple_num
        else:
            self.max_cands_triple_num = 256 # 大约10 x 10

        self.loss = nn.CrossEntropyLoss()

    def get_training_loss(self,seq_output,
                                ner_label,
                                rel_label):
        '''
        获得训练时损失
        input:
            seq_output：
                序列的输出 [B,L,H]
                    B: 批大小
                    L: 句子长度
                    H: 隐藏层大小

            ner_label：
                实体标签 [B,N,3]
                    在数据中，预先给定的被识别出的实体。
                    B: 批大小
                    N: 读取数据时，每个句子最多所含有的实体数量
                    3: 定位一个实体所以具有3个信息
                        [B,N,0] : 实体类别
                        [B,N,1] : 实体的起始位置
                        [B,N,2] : 实体的结束位置

            rel_label：
                关系标签 [B,N,5]
                    B: 批大小
                    N: 读取数据时，每个句子最多所含有的关系数量
                    5: 定位一个关系所以具有5个信息
                        [B,N,0] : 关系类别
                        [B,N,1] : src start position 头实体的起始位置
                        [B,N,2] : src end position 头实体的结束位置
                        [B,N,3] ：tgt start position 尾实体的起始位置
                        [B,N,4] ：tgt end position 尾实体的结束位置
        output:
            loss : torch.tensor
                损失
            preds : torch.tensor
                对于标注数据的预测结果
        
        '''
        rel_label = rel_label.clone() # 拷贝一份，因为要修改值

        B,N,_ = rel_label.shape
        ents_mask = ner_label[:,:,0] != -100

        H = self.bert.config.hidden_size

        logits_tensor = torch.zeros(B,N,H * 2,device = seq_output.device)


        for b in range(B):
            pos_hash = set()
            ents = ner_label[b][ents_mask[b]]
            #打散实体
            idx = torch.randperm(len(ents))
            ents = ents[idx]

            #正采样
            for n in range(N):
                if rel_label[b,n,0] == -100:
                    break
                src_start,src_end,tgt_start,tgt_end = rel_label[b,n,1:]
                # logits_tensor[b,n,:H] = seq_output[b,n,src_start:src_end+1].max(dim =-1)[0] #pooling
                # logits_tensor[b,n,H:] = seq_output[b,n,tgt_start:tgt_end+1].max(dim =-1)[0] #pooling
                logits_tensor[b,n,:H] = seq_output[b,src_start:src_end+1].mean(dim =0) #pooling
                logits_tensor[b,n,H:] = seq_output[b,tgt_start:tgt_end+1].mean(dim =0) #pooling     
                pos_hash.add((src_start,src_end))
                pos_hash.add((tgt_start,tgt_end))

            #负采样
            n = len(pos_hash)
            for (src_id,src_start,src_end) , (tgt_id,tgt_start,tgt_end) in product(ents,ents):
                if src_start == tgt_start and src_end == tgt_end:
                    continue

                n += 1
                #关系正例较多时，负采样数量最多是正采样的两倍
                #关系负例较多时，负采样数量最多是正采样的三倍
                gamma = 4 if len(pos_hash) < len(ents) else 3  
                if n >= N or  n >= gamma * len(pos_hash): 
                    break

                if not hasattr(self,"potential_label_pair") or \
                    (self.potential_label_pair and (src_id.item(),tgt_id.item()) in self.potential_label_pair):
                    # logits_tensor[b,n,:H] = seq_output[b,n,src_start:src_end+1].max(dim =-1)[0] #pooling
                    # logits_tensor[b,n,H:] = seq_output[b,n,tgt_start:tgt_end+1].max(dim =-1)[0] #pooling
                    logits_tensor[b,n,:H] = seq_output[b,src_start:src_end+1].mean(dim =0) #pooling
                    logits_tensor[b,n,H:] = seq_output[b,tgt_start:tgt_end+1].mean(dim =0) #pooling     

                    rel_label[b,n,0] = self.num_rels - 1 # 新增的一类样本label 


        # 得到判别值
        logits = self.rel_classifier(logits_tensor)
        preds = logits.argmax(dim=-1)

        #计算损失
        rel_label = rel_label[:,:,0].reshape(-1).long()
        logits = logits.reshape(-1,self.num_rels)
        loss = self.loss(logits,rel_label)
        return loss,preds

    @torch.no_grad()
    def get_validation_predict(self,seq_output ,ner_label,rel_label = None):
        '''
        获得推理时的预测结果
        input:
            seq_output：
                序列的输出 [B,L,H]
                    B: 批大小
                    L: 句子长度
                    H: 隐藏层大小

            ner_label：
                实体标签 [B,N,3]
                    在数据中，预先给定的被识别出的实体。
                    B: 批大小
                    N: 读取数据时，每个句子最多所含有的实体数量
                    3: 定位一个实体所以具有3个信息
                        [B,N,0] : 实体类别
                        [B,N,1] : 实体的起始位置
                        [B,N,2] : 实体的结束位置

            rel_label：
                关系标签 [B,N,5] ， 默认 None
                    如不为空，则返回cands_recall

                    B: 批大小
                    N: 读取数据时，每个句子最多所含有的关系数量
                    5: 定位一个关系所以具有5个信息
                        [B,N,0] : 关系类别
                        [B,N,1] : src start position 头实体的起始位置
                        [B,N,2] : src end position 头实体的结束位置
                        [B,N,3] ：tgt start position 尾实体的起始位置
                        [B,N,4] ：tgt end position 尾实体的结束位置

        output:
            triple_tensors:
                三元组的输出 [B, N, 5]
                    B : 批大小
                    N : 一个句子中最多允许的三元组数量 = self.max_triple_num。
                        三元组从候选三元组中识别获得，候选三元组数量： self.max_cands_triple_num
                    5 : 定位一个关系所需要的5个信息
                        [B,N,0] : 关系类别
                        [B,N,1] : src start position 头实体的起始位置
                        [B,N,2] : src end position 头实体的结束位置
                        [B,N,3] ：tgt start position 尾实体的起始位置
                        [B,N,4] ：tgt end position 尾实体的结束位置

            logits:
                预测概率的输出 (取softmax之前) [B, N, num_rels]
                    B : 批大小
                    N : 一个句子中最多允许的三元组数量 = self.max_triple_num。
                        三元组从候选三元组中识别获得，候选三元组数量： self.max_cands_triple_num
                    num_rels : 预测关系的数量，
                        等于给定关系数量 + 1
            
            cands_recall : (Optional)
                当属于rel_label时，会计算输入候选三元组中计算三元组的召回率。[num_rels, 2, 2]
                以混淆矩阵的形式返回
                [num_rels,0,0] : True Positive
                [num_rels,0,1] : False Positive
                [num_rels,1,0] : False Negtive
        '''
        B,N,_ = ner_label.shape
        ents_mask = ner_label[:,:,0] != -100

        H = self.bert.config.hidden_size

        logits_tensor = torch.zeros(B, self.max_cands_triple_num ,H * 2,device = seq_output.device)
        triples_tensor = torch.ones(B, self.max_triple_num, 5, device = seq_output.device, dtype=int) * (-100)

        if rel_label is not None:
            cands_recall = torch.zeros(B,2, dtype= int ,device= logits_tensor.device)

        loc_hash = dict()

        for b in range(B):
            ents = ner_label[b][ents_mask[b]]
            #打散实体
            # idx = torch.randperm(len(ents))
            # ents = ents[idx]
            if rel_label is not None:
                label_loc_hash = dict()
                for n in range(len((rel_label[b]))):
                    label,src_start,src_end,tgt_start,tgt_end = rel_label[b,n]
                    if label == -100:
                        break
                    label_loc_hash[(src_start.item(),src_end.item(),tgt_start.item(),tgt_end.item())] = label.item()

            n = 0
            for (src_id,src_start,src_end) , (tgt_id,tgt_start,tgt_end) in product(ents,ents):
                if src_start == tgt_start and src_end == tgt_end:
                    continue

                if not hasattr(self,"potential_label_pair") or \
                    self.potential_label_pair is None or \
                    (src_id.item(),tgt_id.item()) in self.potential_label_pair:
                    # logits_tensor[b,n,:H] = seq_output[b,n,src_start:src_end+1].max(dim =-1)[0] #pooling
                    # logits_tensor[b,n,H:] = seq_output[b,n,tgt_start:tgt_end+1].max(dim =-1)[0] #pooling
                    logits_tensor[b,n,:H] = seq_output[b,src_start:src_end+1].mean(dim =0) #pooling
                    logits_tensor[b,n,H:] = seq_output[b,tgt_start:tgt_end+1].mean(dim =0) #pooling                    
                    loc_hash[b,n] = [[src_id,src_start,src_end],[tgt_id,tgt_start,tgt_end]]

                    loc = (src_start.item(),src_end.item(),tgt_start.item(),tgt_end.item())
                    if rel_label is not None:
                        label = label_loc_hash.get(loc,None)
                        if label is not None:
                            cands_recall[b,0] += 1
                    n += 1
                    if n >= self.max_cands_triple_num:
                        break
            
            if rel_label is not None:
                cands_recall[b, 1] = len(label_loc_hash) - cands_recall[b, 0]


        logits = self.rel_classifier(logits_tensor)
        labels = logits.argmax(dim=-1)

        for b in range(B):
            offset = 0
            for n in range(self.max_cands_triple_num):
                if offset >= self.max_triple_num :
                    break

                if (b,n) in loc_hash:
                    label = labels[b,n].item()
                    src, tgt = loc_hash[b,n]
                    # if  label in self.potential_label_pair[src[0].item(),tgt[0].item()]:
                    if not hasattr(self,"potential_label_pair") or \
                        self.potential_label_pair is None or \
                        label in self.potential_label_pair[src[0].item(),tgt[0].item()]:

                        triples_tensor[b,offset,0] = label
                        triples_tensor[b,offset,1] = src[1]
                        triples_tensor[b,offset,2] = src[2]
                        triples_tensor[b,offset,3] = tgt[1]
                        triples_tensor[b,offset,4] = tgt[2]
                        offset += 1
                else:
                    break
            
        if rel_label is None:
            return triples_tensor,logits
        else:
            return triples_tensor,logits,cands_recall



    def forward(self, input_ids, ner_label = None  , token_type_ids=None, attention_mask=None, rel_label=None,
              position_ids=None, inputs_embeds=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        #bert 输出
        seq_output = outputs[0]
        seq_output = seq_output[:,1:-1,:] 

        outputs = ()
        #训练和验证时输出
        if rel_label is not None:
            loss,train_preds = self.get_training_loss(seq_output, ner_label, rel_label)
            preds,logits,cands_recall = self.get_validation_predict(seq_output.clone().detach() , ner_label, rel_label)
            outputs = outputs + (loss, train_preds, preds,logits,cands_recall) 
        
        #推理时输出
        else:
            preds,logits = self.get_validation_predict(seq_output.clone().detach() , ner_label)
            outputs = outputs + (preds,logits)
        
        return outputs

class BertRelation(RelationBase,BertBase): 
    '''
    Bert关系抽取模型
    
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

        num_rels [Optional] `int`: 默认2
            分类问题中标签的数量。
       
        max_length [Optional] `int`: 默认：512
           支持的最大文本长度。
           如果长度超过这个文本，则截断，如果不够，则填充默认值。
    '''
    def initialize_bert(self,path = None,config = None,**kwargs):
        super().initialize_bert(path,config,**kwargs)
        self.model = BertRelationModel.from_pretrained(self.model_path,config = self.config)

        if self.key_metric == 'loss':
            if self.num_labels == 2:
                self.set_attribute(key_metric = 'f1')
            else:
                self.set_attribute(key_metric = 'macro_f1')
        
