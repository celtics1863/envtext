from ..visualizers import CLSVisualizer
from ..utils.metrics import (
        f1_for_ner,
        accuracy,
        macro_f1_for_ner,
        precision_for_ner,
        recall_for_ner
        )

from collections import defaultdict


import numpy as np

class RelationBase:
    def align_config(self):
        super().align_config()
        if self.max_triple_num is None:
            self.set_attribute(max_triple_num = 32)
        
        if self.max_cands_triple_num is None:
            self.set_attribute(max_cands_triple_num = 256 )#16 * 16

        if self.ner_encoding is None:
            self.set_attribute(ner_encoding = "Pointer")
        
        if self.rel_encoding is None:
            self.set_attribute(rel_encoding = "REL")
        
        if self.max_entity_num is None:
            self.set_attribute(max_entity_num = 16)


    def preprocess(self,text,**kwargs):
        from ..data.dataset_utils import extract_inline_nested_entities
        try:
            text,ents = extract_inline_nested_entities(text)
        except:
            from warnings import warn
            warn(f"查看数据集格式是否正确{text}")
            assert 0

        import re

        inputs = {
            "text":re.sub("\s","",text),
            "ner_label": [[self.entity2id[e["label"]],e["loc"][0],e["loc"][1]] for e in ents]
        }

        if len(inputs["ner_label"]) < self.max_entity_num:
            inputs["ner_label"] = inputs["ner_label"] + [[-100,-100, -100]] * (self.max_entity_num - len(inputs["ner_label"]))
        else:
            inputs["ner_label"] = inputs["ner_label"][:self.max_entity_num]
        
        return inputs


    def postprocess(self,inputs,outputs, **kwargs):
        triples, logits = outputs
        
        ents2labelid = {}
        for t in inputs["ner_label"]:
            if t[0] == -100:
                break
            ents2labelid[(t[1],t[2])] = t[0]

        results = []
        for t in triples: 
            if t[0] == -100:
                break
            try:
                rel = self.id2relation[t[0]]
                source = inputs["text"][t[1]:t[2]+1]
                target = inputs["text"][t[3]:t[4]+1]
                source_label = self.id2entity[ents2labelid[t[1],t[2]]] #source entity
                target_label = self.id2entity[ents2labelid[t[3],t[4]]] #target entity
                results.append(
                    [rel,source,target,source_label,target_label,t[0],t[1],t[2],t[3],t[4]]
                )
            except Exception as e:
                from warnings import warn
                warn(f"{inputs['text']}在后处理的时候有错误：{e}")

        entities = []
        for ent in inputs["ner_label"]:
            if ent[0] == -100:
                break
            entities.append(ent)

        return results,entities

    def _visualize(self,text,labels,probs,save_vis,**kwargs):
        if not hasattr(self, "visualizer"):
            self.visualizer = CLSVisualizer()
        
        self.visualizer.render(text,labels,probs,save_vis)

    def _calc_resample_prob(self,raw_label,**kwargs):
        return 0
        # labels = set([k["label"] for k in raw_label])

        # if hasattr(self, "data_config"):
        #     p = torch.tensor([self.data_config["counter"][e] for e in self.labels])
        #     p = 1/ (p/p.sum()+1e-5) #逆频率
        #     p = p/(p.sum()) #归一化
        #     prob = max([p[self.data_config["labels"].index(label)] for label in labels])
        #     return prob.item()
        # else:
        #     from warnings import warn
        #     warn("缺少self.data_config，可能是函数self.update_data_config()没有重写的问题")
        #     return 0

    def save_results_inline(self,path = None, nested= False,return_lines = False):
        lines = []
        for raw_line,(triples,entities) in self.result.items():

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
                            line += f" 】"
                
                #最后 +1
                offset += 1

            lines.append(line)

            #构造关系标注
            for triple in triples:
                label,src_start,src_end,tgt_start,tgt_end = triple[-5:]

                src_id = ents2id[src_start,src_end]
                tgt_id = ents2id[tgt_start,tgt_end]

                line = f"{src_id} {self.id2relation[label]} {tgt_id}"
                lines.append(line)

            lines.append("\n") #分割符


        if path is not None:
            f = open(path,'w',encoding='utf-8')
            for line in lines:
                f.write(line.strip() + '\n')
            f.close()

    def _report_per_sentence(self,text,pred,p):
        log = f'text:{text} \n'
        for i,j in  zip(pred,p):
            log += '\t pred_classes:{}, \t probability:{:.4f} \n'.format(self.id2label[i.item()],j)
        print(log)

    @property
    def ner_encoding(self):
        if hasattr(self.config, "ner_encoding"):
            return self.config.ner_encoding
        else:
            from warnings import warn
            warn("缺少config.ner_encoding，可能正在初始化，或者数据集/模型导入错误")
            return None

    @property
    def rel_encoding(self):
        if hasattr(self.config, "rel_encoding"):
            return self.config.rel_encoding
        else:
            from warnings import warn
            warn("缺少config.rel_encoding，可能正在初始化，或者数据集/模型导入错误")
            return None

    @property
    def max_triple_num(self):
        if hasattr(self.config, "max_triple_num"):
            return self.config.max_triple_num
        else:
            from warnings import warn
            warn("缺少config.max_triple_num，可能正在初始化，或者数据集/模型导入错误")
            return None
    @property
    def max_cands_triple_num(self):
        if hasattr(self.config, "max_cands_triple_num"):
            return self.config.max_cands_triple_num
        else:
            from warnings import warn
            warn("缺少config.max_cands_triple_num，可能正在初始化，或者数据集/模型导入错误")
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
    def num_rels(self):
        if hasattr(self.config, "num_rels"):
            return self.config.num_rels
        else:
            from warnings import warn
            warn("缺少config.num_rels，可能正在初始化，或者数据集/模型导入错误")
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

    @property
    def relation2id(self):
        if hasattr(self.config, "relation2id"):
            return self.config.relation2id
        else:
            from warnings import warn
            warn("缺少config.relation2id，可能正在初始化，或者数据集/模型导入错误")
            return None

    @property
    def id2relation(self):
        if hasattr(self.config, "id2relation"):
            id2relation = {
                int(k):v  for k,v in  self.config.id2relation.items()
            }
            return id2relation
        else:
            from warnings import warn
            warn("缺少config.id2relation，可能正在初始化，或者数据集/模型导入错误")
            return None

    @property
    def num_labels(self):
        return self.num_rels + 1

    @property
    def num_relations(self):
        return self.num_rels

    @property
    def potential_label_pair(self):
        if hasattr(self.config, "potential_label_pair"):
            return self.config.potential_label_pair
        else:
            from warnings import warn
            warn("缺少config.potential_label_pair，可能正在初始化，或者数据集/模型导入错误")
            return None
    
    def set_potential_label_pair(self):
        pass

    def compute_metrics(self,eval_pred):
        # (preds,valid_preds,logits),(ner_label,rel_label) =  eval_pred
        (train_preds, preds,logits,cands_recall),(ner_label,rel_label) =  eval_pred


        B,N,_ = rel_label.shape

        train_preds_matrix = np.zeros((self.num_rels,2,2),dtype=int)
        valid_preds_matrix = np.zeros((self.num_rels,2,2),dtype=int)



        for b in range(B):
            for n in range(len(rel_label[b])):
                if rel_label[b,n,0] == -100:
                    break

                #计算training时的macro-f1
                if rel_label[b,n,0] == train_preds[b,n]:
                    train_preds_matrix[rel_label[b,n,0],0,0] += 1
                else:
                    train_preds_matrix[rel_label[b,n,0],1,0] += 1
                    train_preds_matrix[rel_label[b,n,0],0,1] += 1
                
                for j in range(len(preds[b])):
                    if preds[b,j,0] == -100:
                        #计算 False Negative
                        valid_preds_matrix[rel_label[b,n,0],1,0] += 1

                    #计算 True Positive
                    if rel_label[b,n,0] == preds[b,j,0]  and \
                        rel_label[b,n,1] == preds[b,j,1] and \
                        rel_label[b,n,2] == preds[b,j,2] and \
                        rel_label[b,n,3] == preds[b,j,3] and \
                        rel_label[b,n,4] == preds[b,j,4]:
                        valid_preds_matrix[rel_label[b,n,0],0,0] += 1
                
                    

            #计算 False positive
            for j in range(len(preds[b])):
                if preds[b,j,0] == -100:
                    break
                
                for n in range(len(rel_label)): 
                    if rel_label[b,n,0] == -100:
                        valid_preds_matrix[preds[b,j,0],0,1] += 1  
                        break

                    if rel_label[b,n,0] == preds[b,j,0]  and \
                        rel_label[b,n,1] == preds[b,j,1] and \
                        rel_label[b,n,2] == preds[b,j,2] and \
                        rel_label[b,n,3] == preds[b,j,3] and \
                        rel_label[b,n,4] == preds[b,j,4]:
                        break


        cands_recall = cands_recall.sum(axis = 0)
        cands_recall = cands_recall[0] / (cands_recall[0] + cands_recall[1] + 1e-5)


        dic = {
            "training_macro_f1": macro_f1_for_ner(None,confusion_matrix=train_preds_matrix),
            "macro_f1": macro_f1_for_ner(None,confusion_matrix=valid_preds_matrix),
            "f1":f1_for_ner(None,confusion_matrix=valid_preds_matrix),
            "precision": precision_for_ner(None,confusion_matrix=valid_preds_matrix),
            "recall": recall_for_ner(None,confusion_matrix=valid_preds_matrix),
            "confusion_matrix": valid_preds_matrix.tolist(),
            "cands_recall" : cands_recall
        }

        return dic