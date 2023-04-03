from ..visualizers import EntityVisualizer,SpanVisualizer
from ..utils.metrics import metrics_for_ner
import torch
from collections import defaultdict

class TripleBase:
    def align_config(self):
        super().align_config()

        if not self.ner_encoding:
            self.set_attribute( ner_encoding = 'BIO')

        if self.entities:
            if not self.num_entities:
                self.set_attribute( num_entities = len(self.entities))

            if not self.num_labels:
                if self.ner_encoding == 'IO':
                    num_labels = len(self.entities) +1
                elif self.ner_encoding == 'BIO':
                    num_labels = len(self.entities) * 2 +1
                elif self.ner_encoding == 'BIOES':
                    num_labels = len(self.entities) * 4 +1
                
                self.set_attribute( num_labels = num_labels )
            
            if not self.labels or len(self.labels) != self.num_labels:
                if self.ner_encoding == 'IO':
                    labels = ['O']
                    for e in self.entities:
                        labels.append(f'I-{e}')
                elif self.ner_encoding == 'BIO':
                    labels = ['O']
                    for e in self.entities:
                        labels.append(f'B-{e}')
                        labels.append(f'I-{e}')
                elif self.ner_encoding == 'BIOES':
                    labels = ['O']
                    for e in self.entities:
                        labels.append(f'B-{e}')
                        labels.append(f'I-{e}')
                        labels.append(f'O-{e}')
                        labels.append(f'E-{e}')
                        labels.append(f'S-{e}')
                self.set_attribute( labels = labels )
            
        elif self.num_entities:
            entities = [f'entity-{i}' for i in range(self.num_entities)]
            self.set_attribute( entities = entities )
            if not self.num_labels:
                if self.ner_encoding == 'IO':
                    num_labels = len(self.entities) +1
                elif self.ner_encoding == 'BIO':
                    num_labels = len(self.entities) * 2 +1
                elif self.ner_encoding == 'BIOES':
                    num_labels = len(self.entities) * 4 +1
                
                self.set_attribute( num_labels = num_labels )
            
            if not self.labels or len(self.labels) != self.num_labels:
                if self.ner_encoding == 'IO':
                    labels = ['O']
                    for e in self.entities:
                        labels.append(f'I-{e}')
                elif self.ner_encoding == 'BIO':
                    labels = ['O']
                    for e in self.entities:
                        labels.append(f'B-{e}')
                        labels.append(f'I-{e}')
                elif self.ner_encoding == 'BIOES':
                    labels = ['O']
                    for e in self.entities:
                        labels.append(f'B-{e}')
                        labels.append(f'I-{e}')
                        labels.append(f'O-{e}')
                        labels.append(f'E-{e}')
                        labels.append(f'S-{e}')
                        
                self.set_attribute( labels = labels )

        else:
            entities = ['entity']
            num_entities = 1
            num_labels = 3
            labels = ['O','B','I']
            
            self.update_config(
                 entities = entities,
                num_entities = num_entities,
                num_labels = num_labels,
                labels = labels
            )


        if self.datasets is not None and self.transition is None \
            and self.num_labels is not None and self.num_labels > 1:
            #更新 transition matrix
            transition = torch.zeros(self.num_labels,self.num_labels)
            for label in self.datasets["train"]["label"]:
                for i,j in zip(label[:-1],label[1:]):
                    if i!=-100 and j != -100:
                        transition[i,j] += 1
            
            transition = torch.log1p(torch.log1p(transition))
            transition = transition / (transition.sum(dim=-1)[:,None] + 1e-5)
            self.set_attribute(transition=transition.numpy().tolist())

        # if not hasattr(self.config, "visualizer"):
        #     self.set_attribute(visualizer= "entity")
        
        # if self.config.visualizer == "entity":
        #     self.visualizer = EntityVisualizer()
        # else:
        #     self.visualizer = SpanVisualizer()

        if self.max_triple_num is None:
            self.set_attribute(max_triple_num = 64)
        
        if self.max_cands_triple_num is None:
            self.set_attribute(max_cands_triple_num = 256 )#16 * 16

        if self.ner_encoding is None:
            self.set_attribute(ner_encoding = "Pointer")
        
        if self.rel_encoding is None:
            self.set_attribute(rel_encoding = "REL")
        
        if self.max_entity_num is None:
            self.set_attribute(max_entity_num = 16)

    @property
    def transition(self):
        if hasattr(self.config, "transition"):
            return self.config.transition
        else:
            return None    

    def preprocess(self,text,**kwargs):
        import re
        text = re.sub("\s","",text)
        text = ' '.join(text) #white-space tokenizer
        return text


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


    def compare(self,text,label):
        pred_result = self.predict(text,print_result = False)
        gt_result = self._convert_rawlabel2result(text, label)


    def _convert_rawlabel2result(self,text,raw_label):
        raw_label = sorted(raw_label,key=lambda x:x["loc"][0])
        words,poses = [],[]
        start = 0
        for label in raw_label:
            new_start,new_end = label["loc"]
            if new_start > start:
                words += list(text[start:new_start])
                poses += [""] * (new_start-start)
            words.append(text[new_start:new_end+1])
            poses.append(label["label"])
            start = new_end +1
        
        if start < len(text)-1:
            words += list(text[start:])
            poses += [""] * (len(text) - 1 - start)
            
        return words,poses


    def _convert_label2result(self,text,classes,locs):
        # words = []
        # poses = []
        # start = 0
        # for c,loc in zip(classes,locs):
        #     words += list(text[start:loc[0]])
        #     poses += ['']*(loc[0]-start)
            
        #     words.append(text[loc[0]:loc[1]+1])
        #     poses.append(c)
            
        #     start = loc[1] + 1
        
        # words += list(text[start:])
        # poses += [''] * (len(text)-start)
        # return words,poses
        return classes,locs
    
    # def _visualize(self,text,classes,locs,path = None,**kwargs):
    #     #entity visualizer
    #     if self.config.visualizer == "entity":
    #         words = []
    #         poses = []
    #         start = 0
    #         for c,loc in zip(classes,locs):
    #             words += list(text[start:loc[0]])
    #             poses += ['']*(loc[0]-start)
                
    #             words.append(text[loc[0]:loc[1]+1])
    #             poses.append(c)
                
    #             start = loc[1] + 1
            
    #         words += list(text[start:])
    #         poses += [''] * (len(text)-start)
            
    #         self.visualizer.render(words,poses,path)
    #     #span visualizer
    #     else:
    #         tokens = list(text)
    #         spans = [{"token_start":start,"token_end":end,"label":c} for c,(start,end) in  zip(classes,locs)]
    #         self.visualizer.render(tokens,spans)



    def mapping_entities(self,config={},**kwargs):
        '''
        config: {old_entity:new_entity}

        kwargs: old_entity = new_entity
        '''
        config.update(kwargs)

        entities = [config[e] if e in config else e for e in self.entities ]

        self.set_attribute(entities = entities)

    def preprocess(self,text):
        import re
        text = re.sub("\s","",text)
        # text = ' '.join(text) #white-space tokenizer
        return text
    
    def postprocess(self,text,logits, print_result = True, save_result = True,return_result = False,save_vis = None):
        triples_tensor,entities_tensor = logits

        triples = [t for t in triples_tensor if t[0] != -100]
        entities = [e for e in entities_tensor if e[0] != -100]

        return triples,entities

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
                ents2label[ents2id[(start,end)]] = label

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
                            line += f"【{self.id2entity[label]}_{v}|{v} " #留下空白
                        else:
                            line += f"【{self.id2entity[label]}|{v} " #留下空白
                
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
            for label,src_start,src_end,tgt_start,tgt_end in triples:
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



    def _report_per_sentence(self,text,labels,classes,probs):
        log = f'text:{text}\n'
        
        for l,c,p in zip(labels,classes,probs):
            log += f'\t pred: {l} \t entity:{c} \t prob: {p} '
            
        print(log)
     
    @property
    def ner_encoding(self):
        if hasattr(self.config, "ner_encoding"):
            return self.config.ner_encoding
        else:
            from warnings import warn
            warn("缺少config.ner_encoding，可能正在初始化，或者数据集/模型导入错误")
            return None



    def _to_scalar(self,value):
        if isinstance(value, (int,str,float)):
            return value 
        elif isinstance(value, torch.Tensor):
           return value.clone().cpu().item()
        else:
            return value[0]
    

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
            warn("缺少config.num_rels，可能是数据集导入错误的问题")
            return None

    @property
    def entity2id(self):
        if hasattr(self.config, "entity2id"):
            return self.config.entity2id
        else:
            from warnings import warn
            warn("缺少config.entity2id，可能是数据集导入错误的问题")
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
            warn("缺少config.id2entity，可能是数据集导入错误的问题")
            return None

    @property
    def relation2id(self):
        if hasattr(self.config, "relation2id"):
            return self.config.relation2id
        else:
            from warnings import warn
            warn("缺少config.relation2id，可能是数据集导入错误的问题")
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
            warn("缺少config.id2relation，可能是数据集导入错误的问题")
            return None

    @property
    def num_labels(self):
        return self.num_rels

    @property
    def num_relations(self):
        return self.num_rels


    def compute_metrics(self,eval_pred):
        (ner_loss,rel_loss,ner_confusion_matrix,rel_overlap,triples,entities),(rel_labels,ner_labels) = eval_pred

        # print(ner_labels.shape,rel_labels.shape,rel_labels)
        import numpy as np
        B = len(triples)

        rel_labels = rel_labels.tolist()

        matrix = np.zeros((self.num_rels,2,2)) #True Positive
        for b in range(B):
            for a in triples[b]:
                if a[0] == -100:
                    break
                
                for c in rel_labels[b]:
                    if c[0] == -100:
                        break

                    if a[0] == c[0] and a[1]==c[1] and a[2]==c[2] and a[3] == c[3] and a[4] == c[4]:
                        matrix[a[0],0,0] += 1
                        break
                else:
                    matrix[a[0],0,1] += 1 #False Positive

            for c in rel_labels[b]:
                if c[0] == -100:
                    break

                for a in triples[b]:
                    if a[0] == -100:
                        break

                    if a[0] == c[0] and a[1]==c[1] and a[2]==c[2] and a[3] == c[3] and a[4] == c[4]:
                        break
                else:
                    matrix[c[0],1,0] += 1 #False Negtive

        precision = matrix[:,0,0] / (matrix[:,0,0] + matrix[:,0,1] + 1e-5)
        recall = matrix[:,0,0] / (matrix[:,0,0] + matrix[:,1,0]  + 1e-5)
        f1 = 2 * precision * recall /(precision + recall + 1e-5)


        ner_matrix = ner_confusion_matrix.sum(0)
        ner_precision = ner_matrix[:,0,0] / (ner_matrix[:,0,0] + ner_matrix[:,0,1] + 1e-5)
        ner_recall = ner_matrix[:,0,0] / (ner_matrix[:,0,0] + ner_matrix[:,1,0]  + 1e-5)
        ner_f1 = 2 * ner_precision * ner_recall /(ner_precision + ner_recall + 1e-5)


        rel_overlap = rel_overlap.sum(0)
        overlap_precision = rel_overlap[0,0] / (rel_overlap[0,0] + rel_overlap[0,1] + 1e-5)
        overlap_recall = rel_overlap[0,0] / (rel_overlap[0,0] + rel_overlap[1,0]  + 1e-5)
        overlap_f1 = 2 * overlap_precision * overlap_recall /(overlap_recall + overlap_precision + 1e-5)


        dic = {
            "macro_f1": f1.mean(),
            "ner_loss_value":ner_loss[0],
            "rel_loss_value":rel_loss[0],
            "ner_macro_f1":ner_f1.mean(),
            "confusion_matrix": matrix.tolist(),
            "ner_confusion_matrix": ner_matrix.tolist(),
            "overlap_precision":overlap_precision,
            "overlap_recall":overlap_recall,
        }


            
        return dic
    
