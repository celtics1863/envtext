from ..visualizers import EntityVisualizer,SpanVisualizer
import torch
import numpy as np
from ..utils.metrics import (
        f1_for_ner,
        accuracy,
        macro_f1_for_ner,
        precision_for_ner,
        recall_for_ner
        )

from collections import defaultdict

class GPBase:
    def align_config(self):
        super().align_config()

        if not hasattr(self.config, "visualizer"):
            self.set_attribute(visualizer= "span")

        if not hasattr(self.threshold, "threshold"):
            self.set_attribute(threshold= 0.)
        

    def preprocess(self,text,**kwargs):
        import re
        text = text.strip().replace("【", "").replace("】", "")
        text = re.sub("\s","",text)

        #加入词汇信息
        if hasattr(self.config, "jieba"):
            return {
                "text": text,
                "vectors": self._jieba_tokenizer(text)
            }
        
        return text

    def postprocess(self,text, outputs ,**kwargs):
        preds,logits = outputs

        entities = []
        for ent in preds:
            if ent[0] == -100:
                break
            entities.append([ent[0],ent[1]-1,ent[2]-1])

        return entities


    def _calc_resample_prob(self,raw_label,**kwargs):
        # entity_labels = set([k["label"] for k in raw_label])

        if "entity" in raw_label:
            entity_labels = set([k["label"] for k in raw_label["entity"]])
        else:
            entity_labels = set([k["label"] for k in raw_label])

        if not entity_labels:
            return 0
            
        if hasattr(self, "data_config"):
            p = torch.tensor([self.data_config["counter"][e] for e in self.entities])
            inv_p = 1/ (p/p.sum()+1e-5) #逆频率
            p = 1 - p/(p.sum()+1) # 1防止除0
            p = torch.sigmoid((inv_p - inv_p.mean())*3/inv_p.std())
            prob =  max([p[self.data_config["entities"].index(e)] for e in entity_labels])
            return prob.item()
        else:
            from warnings import warn
            warn("缺少self.data_config，可能是函数self.update_data_config()没有重写的问题")
            return 0

    
    def _visualize(self,text,classes,locs,path = None,**kwargs):
        #准备可视化
        tokens = []
        spans = []

        #准备实体id
        ents2id = {}
        for label,start,end in entities:
            ents2id[(start,end)] = len(ents2id)
        id2ents = {v:k for k,v in ents2id}

        #准备实体label
        ents2label = {}
        for label,start,end in entities:
            ents2label[ents2id[(start,end)]] = self.id2entity[label]

        #准备实体开始的位置
        start2ents = defaultdict(list)
        for k,v in ents2id.items():
            start2ents[k[0]].append(v)
        
        #准备实体结束的位置
        end2ents = defaultdict(list)
        for k,v in ents2id.items():
            end2ents[k[1]].append(v)

        #构造新的句子
        line = ""
        offset = 0
        while offset < len(raw_line):
            if offset in start2ents:
                #首先处理开始的地方
                max_len = 1
                for v in start2ents[offset]:
                    max_len = max(max_len, id2ents[v][1] - id2ents[v][0] + 1)
                    spans.append({
                        "token_start":id2ents[v][0],
                        "token_end":id2ents[v][1],
                        "label":ents2label[v],
                    })
                    tokens.append(text[id2ents[v][0]:id2ents[v][1]+1])

                offset += max_len

            else:
                spans.append({
                    "token_start":offset,
                    "token_end":offset,
                    "label":"",
                })
                offset += 1
                tokens.append(text[offset])

        self.visualizer.render(tokens,spans)


    def save_results_inline(self,path = None, nested= False,return_lines = False):
        lines = []
        for raw_line,entities in self.result.items():

            #准备实体id
            ents2id = {}
            for label,start,end in entities:
                ents2id[(start,end)] = len(ents2id)
            id2ents = {v:k for k,v in ents2id}

            #准备实体label
            ents2label = {}
            for label,start,end in entities:
                ents2label[ents2id[(start,end)]] = self.id2entity[label]

            #准备实体开始的位置
            start2ents = defaultdict(list)
            for k,v in ents2id.items():
                start2ents[k[0]].append(v)
            
            #准备实体结束的位置
            end2ents = defaultdict(list)
            for k,v in ents2id.items():
                end2ents[k[1]].append(v)

            #构造新的句子
            line = ""
            offset = 0
            while offset < len(raw_line):
                if offset in start2ents:
                    #首先处理开始的地方
                    for v in start2ents[offset]:
                        label = ents2label[v]
                        if nested:
                            line += f"【{label}_{v}|{v} " #留下空白
                        else:
                            line += f"【{label}|{v} " #留下空白
                
                #再添加字
                line += raw_line[offset]

                if offset in end2ents:
                    #再处理结束的地方
                    for v in end2ents[offset]:
                        if nested:
                            line += f" {v}_】"
                        else:
                            line += f" {v}】"
                
                #最后 +1
                offset += 1

            lines.append(line)

        if path is not None:
            f = open(path,'w',encoding='utf-8')
            for line in lines:
                f.write(line.strip() + '\n')
            f.close()

    def _report_per_sentence(self,text,labels,classes,probs):
        log = f'text:{text}\n'
        
        for l,c,p in zip(labels,classes,probs):
            log += f'\t pred: {l} \t entity:{c} \t prob: {p} '
            
        print(log)
     
    
    @property
    def threshold(self):
        if hasattr(self.config, "threshold"):
            return self.config.threshold
        else:
            return None

    @property
    def ner_encoding(self):
        if hasattr(self.config, "ner_encoding"):
            return self.config.ner_encoding
        else:
            from warnings import warn
            warn("缺少config.ner_encoding，可能正在初始化，或者数据集/模型导入错误")
            return None

    @property
    def max_entity_num(self):
        if hasattr(self.config, "max_entity_num"):
            return self.config.max_entity_num
        else:
            from warnings import warn
            warn("缺少config.max_entity_num，可能正在初始化，或者数据集/模型导入错误")
            return None

    @property
    def entity2id(self):
        if hasattr(self.config, "entity2id"):
            return self.config.entity2id
        else:
            from warnings import warn
            warn("缺少config.entity2id，可能正在初始化，或者数据集/模型导入错误")
            return None

    @property
    def id2entity(self):
        if hasattr(self.config, "id2entity"):
            id2entity = {
                int(k):v  for k,v in  self.config.id2entity.items()
            }
            return id2entity
        else:
            from warnings import warn
            warn("缺少config.id2entity，可能正在初始化，或者数据集/模型导入错误")
            return None

    def compute_metrics(self,eval_pred):
        (preds,logits),labels = eval_pred

        B,N,L,_ = logits.shape

        matrix = np.zeros((N,2,2))

        for n in range(N):
            pred = []
            true = []
            for b, start, end in zip(*np.where(logits[:,n] > 0)):
                pred.append((b, n, start, end))
            for b, start, end in zip(*np.where(labels[:,n] > 0)):
                true.append((b, n, start, end))

            R = set(pred)
            T = set(true)
            X = len(R & T)
            Y = len(R)
            Z = len(T)
            
            matrix[n,0,0] = X
            matrix[n,0,1] = Y
            matrix[n,1,0] = Z



        dic = {
            "macro_f1": macro_f1_for_ner(None,confusion_matrix=matrix),
            "f1":f1_for_ner(None,confusion_matrix=matrix),
            "precision": precision_for_ner(None,confusion_matrix=matrix),
            "recall": recall_for_ner(None,confusion_matrix=matrix),
            "confusion_matrix": matrix.tolist(),
        }

        print(dic)
        return dic

    
