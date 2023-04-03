from .bert_base import BertBase
from torch.nn.utils.rnn import pad_sequence
import torch #for torch.argmax torch.max
import torch.nn.functional as F
from torch import nn #for nn.Dropout nn.LSTM nn.Linear
from torchcrf import CRF
from transformers import BertPreTrainedModel,BertModel
# from ..tokenizers import WoBertTokenizer
from ..utils.metrics import metrics_for_ner
from ..visualizers import EntityVisualizer
from .triple_base import TripleBase
from ..utils.algorithms import ner_decode
import numpy as np
from collections import defaultdict
import random #for random.shuffle
import itertools #for itertools.combinations

class BertTripleModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertTripleModel, self).__init__(config)
        self.bert = BertModel(config)
        if hasattr(config,'max_length'):
            self.max_length = config.max_length
        else:
            self.max_length = 510


        if hasattr(config, "num_rels"):
            self.num_rels = config.num_rels + 1 # 有一类无关
        else:
            self.num_rels = 2


        if hasattr(config, "max_triple_length"):
            self.max_triple_length = max_triple_length
        else:
            self.max_triple_length = 24

        if hasattr(config, "spo_counter"):
            self.potential_label_pair = defaultdict(set) # (s,o),r
            for k in self.config.spo_counter:
                s,p,o = k.split('\t')
                s = self.config.entity2id[s]
                p = self.config.relation2id[p]
                o = self.config.entity2id[o]
                self.potential_label_pair[s,o].add(p)

        self.ner_encoding = config.ner_encoding #ner 编码方式，BIO,BIOE,BIOES

        if hasattr(config, "max_triple_num"):
                self.max_triple_num = config.max_triple_num
        else:
            self.max_triple_num = 64

        if hasattr(config, "max_cands_triple_num"): #最多候选三元组个数
                self.max_cands_triple_num = config.max_cands_triple_num
        else:
            self.max_cands_triple_num = 400

        if hasattr(config, "max_entity_num"):
                self.max_entity_num = config.max_entity_num
        else:
            self.max_entity_num = 64

        if hasattr(config,'lstm'): #使用LSTM
            self.lstm = config.lstm
        else:
            self.lstm = True

        if self.lstm :
            self.bilstm = nn.LSTM(
                input_size= config.hidden_size,  # 768 + vector
                hidden_size=config.hidden_size,  # 768
                batch_first=True,
                num_layers=int(self.lstm),
                dropout=0,  # 0.5
                bidirectional=True
                )

            self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)


            self.rel_bilstm = nn.LSTM(
                input_size= config.hidden_size * 2,  # 768 + vector
                hidden_size=config.hidden_size,  # 768
                batch_first=True,
                num_layers=int(self.lstm),
                dropout=0,  # 0.5
                bidirectional=True
                )
            
            self.rel_classifier = nn.Linear(config.hidden_size*2 , self.num_rels)


        #ner loss 条件随机场
        self.crf = CRF(config.num_labels, batch_first=True)
        if hasattr(config, "transition"):
            self.crf.transitions = nn.Parameter(torch.tensor(config.transition))

        #rel loss
        # self.rel_loss_fn = nn.BCEWithLogitsLoss()
        self.rel_loss_fn = nn.CrossEntropyLoss()

    def _get_training_rel_losses(self,ner_outputs,ner_label,rel_label,debug=False):
        rel_label = rel_label.clone()
        B,N,_ = rel_label.shape
        triple_logits = torch.zeros(B,N,self.max_triple_length,self.rel_bilstm.input_size,device=ner_outputs.device)

        for b in range(B):
            # 正样本采样
            triple_hash = {}
            
            for n in range(N):
                if rel_label[b,n,0] == -100:
                    continue

                label, src_start,src_end,tgt_start,tgt_end = rel_label[b,n]

                src_len = src_end - src_start + 1
                tgt_len = tgt_end - tgt_start + 1

                #长度过长
                if src_len + tgt_len + 1> self.max_triple_length:
                    continue

                triple_logits[b,n,:src_len] = ner_outputs[b,src_start:src_end+1]
                triple_logits[b,n,src_len] = 0 #lstm中的分割符号
                triple_logits[b,n,src_len+1:src_len+1+tgt_len] = ner_outputs[b,tgt_start:tgt_end+1]

                #哈希
                triple_hash[src_start,src_end,tgt_start,tgt_end] = [b,n,label]

            #负样本采样
            num_pos = len(triple_hash)
            offset = len(triple_hash)

            #打乱顺序
            cands = list(itertools.combinations(ner_label[b],r=2))
            random.shuffle(cands)

            #进行采样
            for (src_id,src_start,src_end),(tgt_id,tgt_start,tgt_end) in cands:
                if offset >=  N or offset > num_pos * 2: #正负样本最多1:1
                    continue

                if not hasattr(self, "potential_label_pair") or (src_id,tgt_id) in self.potential_label_pair:
                    if (src_start,src_end,tgt_start,tgt_end) in triple_hash:
                        continue

                    src_len = src_end - src_start + 1
                    tgt_len = tgt_end - tgt_start + 1
                    #长度过长
                    if src_len + tgt_len + 1> self.max_triple_length:
                        continue

                    #搬迁bert logits
                    triple_logits[b,offset,:src_len] = ner_outputs[b,src_start:src_end+1]
                    triple_logits[b,offset,src_len] = 0 #lstm中的分割符号
                    triple_logits[b,offset,src_len+1:src_len+1+tgt_len] = ner_outputs[b,tgt_start:tgt_end+1]

                    #保存一下位置
                    rel_label[b,offset,0] = self.num_rels - 1
                    rel_label[b,offset,1] = src_start
                    rel_label[b,offset,2] = src_end
                    rel_label[b,offset,3] = tgt_start
                    rel_label[b,offset,4] = tgt_end

                    offset += 1

        rel_lstm_output,_ = self.rel_bilstm(triple_logits.reshape(-1, self.max_triple_length , self.rel_bilstm.input_size))
        rel_lstm_output = rel_lstm_output.max(dim=1)[0].reshape(-1 ,self.rel_bilstm.hidden_size * 2)

        rel_logits = self.rel_classifier(rel_lstm_output) # [2*hidden, 5]

        # preds = rel_logits.argmax(-1)
        # tp = sum(preds == rel_label[:,:,0].reshape(-1))
        # print(f"training acc is {tp / len(preds)}")

        loss = self.rel_loss_fn(rel_logits,rel_label[:,:,0].reshape(-1))

        if debug:
            return loss,rel_label
        else:
            return loss

    @torch.no_grad()
    def _get_validation_rel_label(self,ner_outputs,ner_preds, rel_label = None,debug=False):
        '''
        ner_preds : list : B,*
            label 以[label_id,start,end]的形式标出
        '''

        B,_,_ = ner_outputs.shape
        triple_logits = torch.zeros(B,
                                    self.max_cands_triple_num,
                                    self.max_triple_length,
                                    self.rel_bilstm.input_size,
                                    device=ner_outputs.device) # * 2 for bi lstm
        
        triple_hash = dict()


        rel_overlap = torch.zeros(B,2,2)
        rel_tp = set()

        #整理rel logits格式
        for b in range(B):
            offset = 0
            for src_id,src_start,src_end in ner_preds[b]:
                for tgt_id,tgt_start,tgt_end in ner_preds[b]:
                    if offset >= self.max_cands_triple_num:
                        continue
                    if not hasattr(self, "potential_label_pair") or (src_id,tgt_id) in self.potential_label_pair:
                        src_len = src_end - src_start + 1
                        tgt_len = tgt_end - tgt_start + 1
                        #长度过长
                        if src_len + tgt_len + 1> self.max_triple_length:
                            continue

                        #搬迁bert logits
                        triple_logits[b,offset,:src_len] = ner_outputs[b,src_start:src_end+1]
                        triple_logits[b,offset,src_len] = 0 #lstm中的分割符号
                        triple_logits[b,offset,src_len+1:src_len+1+tgt_len] = ner_outputs[b,tgt_start:tgt_end+1]

                        #保存一下位置
                        triple_hash[(b,offset)] = [ [src_id,src_start,src_end] , [tgt_id,tgt_start,tgt_end] ]
                        
                        offset += 1

            #计算 待计算实体对的 覆盖率
            if rel_label is not None:
                for (b,offset),(s,o) in triple_hash.items():
                    for i in range(len(rel_label[b])):
                        if rel_label[b][i][0] != -100:
                            if s[1] == rel_label[b][i][1] and s[2] == rel_label[b][i][2] and o[1] == rel_label[b][i][3] and o[2] == rel_label[b][i][4]:
                                rel_overlap[b,0,0] += 1
                                rel_tp.add((b,offset,i))
                                break
                        else:
                            break
                    else:
                        rel_overlap[b,0,1] += 1

                for i in range(len(rel_label[b])) :
                    if rel_label[b][i][0] != -100:
                        for s,o in triple_hash.values():
                                if s[1] == rel_label[b][i][1] and s[2] == rel_label[b][i][2] and o[1] == rel_label[b][i][3] and o[2] == rel_label[b][i][4]:
                                    break
                        else:
                            rel_overlap[b,1,0] += 1


        #获得rel预测结果
        rel_lstm_output,_ = self.rel_bilstm(triple_logits.reshape(-1, self.max_triple_length , self.rel_bilstm.input_size))
        rel_lstm_output = rel_lstm_output.max(dim=1)[0].reshape(B, -1 ,self.rel_bilstm.hidden_size * 2) # [BxN, hidden_size * 2]

        rel_logits = self.rel_classifier(rel_lstm_output) # [2*hidden, 5]

        rel_preds = rel_logits.argmax(dim = -1)

        #validation tp
        # yes = 0
        # for b,offset,i in rel_tp:
        #     if rel_preds[b,offset] == rel_label[b,i,0]:
        #         yes += 1
        # print(f"validation acc is {yes/(len(rel_tp) + 1e-5)}")


        #整理结果为三元组
        batch_triples = [] 
        for b in range(B):
            triples = []
            #找遍历每一个
            for offset,l in enumerate(rel_preds[b]):
                if (b,offset) in triple_hash:
                    ent_pair = triple_hash[(b,offset)] #去除 对应的实体对
                    #过滤结果
                    if hasattr(self, "potential_label_pair") and l.item() in self.potential_label_pair[ent_pair[0][0],ent_pair[1][0]]:
                        res = [
                            l.item(),
                            ent_pair[0][1],
                            ent_pair[0][2],
                            ent_pair[1][1],
                            ent_pair[1][2],
                        ]
                        triples.append(res)
            batch_triples.append(triples)

        if rel_label is None:
            return batch_triples
        else:
            return batch_triples,rel_overlap

    def ner_decode(self,ner_logits):
        if isinstance(ner_logits, list):
            ner_logits = np.array(ner_logits)

        if len(ner_logits.shape) == 3:
            preds = self.crf.decode(ner_logits) #[B ,L ,cls]
        elif len(ner_logits.shape) == 2:
            preds = ner_logits #[B, L]

        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
            preds = preds.tolist()

        # import pdb;pdb.set_trace();
        ner_preds = []
        for b in range(len(preds)):
            label = ner_decode(preds[b][1:-1],ner_encoding=self.ner_encoding,return_method="Pointer") #[id,start,end]
            ner_preds.append(label.tolist())
        
        return ner_preds


        
    def forward(self, input_ids, rel_label = None,per_text = None, vectors = None , token_type_ids=None, attention_mask=None, labels=None,
              position_ids=None, inputs_embeds=None, head_mask=None):

        outputs = self.bert(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
        
        #bert 输出
        sequence_output = outputs[0]
        B = sequence_output.shape[0]

        lstm_output, _ = self.bilstm(sequence_output)
        ## 得到NER判别值
        logits = self.classifier(lstm_output)
        ## 得到NER预测结果
        ner_preds = self.ner_decode(logits)
        ## 去掉[SEP]和[CLS]
        ner_outputs = lstm_output[:,1:-1,:]   
        # ner_outputs = sequence_output[:,1:-1,:]
      
        # import pdb; pdb.set_trace();
        outputs = ()
        if labels is not None and rel_label is not None:
            #处理label长度和logits长度不一致的问题
            labels = labels.clone() #make a copy
            B,L,C,S = logits.shape[0],logits.shape[1],logits.shape[2],labels.shape[1]
            ## print(B,L,C,S)
            if S > L :
                labels = labels[:,:L]
                loss_mask = labels.gt(-1)
            elif S < L:
                pad_values = torch.tensor([[-100]],device = logits.device).repeat(B,L-S)
                labels = torch.cat([labels,pad_values],dim=1)
                loss_mask = labels.gt(-1)
            else:
                loss_mask = labels.gt(-1)
            
            labels[~loss_mask] = 0
            loss = self.crf(logits, labels.long() , mask = loss_mask.bool()) * (-1)
            

            ner_label = self.ner_decode(labels)
            rel_loss =  self._get_training_rel_losses(ner_outputs, ner_label , rel_label)

            ner_confusion_matrix = self.ner_confusion_matrix(ner_preds, ner_label)

            outputs = (loss + rel_loss * 100 , 
                        torch.ones(labels.shape[0],device=logits.device) * loss ,
                        torch.ones(labels.shape[0],device=logits.device) * rel_loss,
                        torch.from_numpy(ner_confusion_matrix),
                        )

        ner_outputs = ner_outputs.clone().detach()

        if rel_label is not None:
            triples,rel_overlap =  self._get_validation_rel_label(ner_outputs, ner_preds, rel_label)
            outputs += (rel_overlap,) 
        else:
            triples =  self._get_validation_rel_label(ner_outputs, ner_preds)

        #后处理，整理为tensor，符合框架规范
        triples_tensor = torch.ones(B,self.max_triple_num,5,device = ner_outputs.device,dtype=int) * (-100)
        for b in range(B):
            for j,t in enumerate(triples[b]):
                if j < self.max_triple_num:
                    triples_tensor[b,j,0] = t[0]
                    triples_tensor[b,j,1] = t[1]
                    triples_tensor[b,j,2] = t[2]
                    triples_tensor[b,j,3] = t[3]
                    triples_tensor[b,j,4] = t[4]

        entities_tensor = torch.ones(B,self.max_entity_num,3,device = ner_outputs.device,dtype=int) * (-100)
        for b in range(B):
            for j,ent in enumerate(ner_preds[b]):
                if j < self.max_entity_num:
                    entities_tensor[b,j,0] = ent[0]
                    entities_tensor[b,j,1] = ent[1]
                    entities_tensor[b,j,2] = ent[2]

        return outputs + (triples_tensor,entities_tensor)    

    
    def ner_confusion_matrix(self,pred,gt):
        '''
        pred: N, List
        gt: N,list
        '''
        B = len(pred)

        matrix = np.zeros((B,len(self.bert.config.entities),2,2))

        for b in range(B):
            for p in pred[b]:
                for g in gt[b]:
                    if p[0] == g[0] and p[1] == g[1] and p[2] == g[2]:
                        matrix[b][p[0]][0][0] += 1
                        break
                else:
                    matrix[b][p[0]][0][1] += 1

            for g in gt[b]:
                for p in pred[b]:
                    if p[0] == g[0] and p[1] == g[1] and p[2] == g[2]:
                        break
                else:
                    matrix[b][g[0]][1][0] += 1

        return matrix


class BertTriple(TripleBase,BertBase):
    '''
    Args:
       path `str`: 
           模型保存的路径
           
       config [Optional] `dict` :
           配置参数
   
   Kwargs:
       entities [Optional] `List[int]` or `List[str]`: 默认为None
           命名实体识别问题中实体的种类。
           命名实体识别问题中与entities/num_entities必设置一个，若否，则默认只有1个实体。

       num_entities [Optional] `int`: 默认None
           命名实体识别问题中实体的数量。
           命名实体识别问题中与labels/num_labels/entities必设置一个，若否，则默认只有1个实体。
           实体使用BIO标注，如果n个实体，则有2*n+1个label。
       
       ner_encoding [Optional] `str`: 默认BIO
           目前支持三种编码：
               BI
               BIO
               BIOES
          
           如果实体使用BIO标注，如果n个实体，则有2*n+1个label。
           eg:
               O: 不是实体
               B-entity1：entity1的实体开头
               I-entity1：entity1的实体中间
               B-entity2：entity2的实体开头
               I-entity2：entity2的实体中间

       max_length [Optional] `int`: 默认：512
           支持的最大文本长度。
           如果长度超过这个文本，则截断，如果不够，则填充默认值。
    '''

    def initialize_bert(self,path = None,config = None,**kwargs):
        super().initialize_bert(path,config,**kwargs)
        self.model = BertTripleModel.from_pretrained(self.model_path,config = self.config)
        # if self.key_metric == 'validation loss':
        # if self.num_entities == 1:
        #     self.set_attribute(key_metric = 'f1')
        # else:
        self.set_attribute(key_metric = 'macro_f1')
        
        from transformers import DataCollatorForTokenClassification
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)
        self.visualizer = EntityVisualizer()



    
    
# if __name__ == "__main__":
