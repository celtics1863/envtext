from ..visualizers import EntityVisualizer,SpanVisualizer,POSVisualizer
from ..utils.metrics import metrics_for_ner
import torch

class NERBase:
    def align_config(self):
        super().align_config()
        if hasattr(self.config,"word2vec"):
            self.set_attribute(input_text=True)

        if not hasattr(self.config, "visualizer"):
            self.set_attribute(visualizer= "entity")
        
        if self.config.visualizer == "entity":
            self.visualizer = EntityVisualizer()
        elif self.config.visualizer == "pos":
            self.visualizer = POSVisualizer()
        else:
            self.visualizer = SpanVisualizer()

    @property
    def transition(self):
        if hasattr(self.config, "transition"):
            return self.config.transition
        else:
            return None    

    @property
    def viterbi(self):
        if hasattr(self.config, "viterbi"):
            return self.config.viterbi
        else:
            return False    


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
    def ner_encoding(self):
        if hasattr(self.config,'ner_encoding'):
            return self.config.ner_encoding
        else:
            return None


    def preprocess(self,text,**kwargs):
        '''
        预处理
        '''
        import re
        text = text.replace("【", "").replace("】", "")
        text = re.sub("\s","",text)

        if hasattr(self.config, "jieba") and self.config.jieba:
            #如果存在词典，加入词汇特征
            pass

        return " ".join(text)

    
    def postprocess(self,text,output, **kwargs):
        '''
        后处理
        '''
        import re
        text = re.sub("\s","",text)

        if self.model.crf:
            preds = torch.tensor(output[1][1:-1])
            logits = torch.tensor(output[0][1:-1])
            entities,locs,labels = self._decode(text,preds)
        else:
            raise NotImplementedError()
        return entities,labels


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
    
    def _visualize(self,text,classes,locs,path = None,**kwargs):
        #entity visualizer
        if self.config.visualizer == "entity":
            words = []
            poses = []
            start = 0
            for c,loc in zip(classes,locs):
                words += list(text[start:loc[0]])
                poses += ['']*(loc[0]-start)
                
                words.append(text[loc[0]:loc[1]+1])
                poses.append(c)
                
                start = loc[1] + 1
            
            words += list(text[start:])
            poses += [''] * (len(text)-start)
            
            self.visualizer.render(words,poses,path)
        #span visualizer
        else:
            tokens = list(text)
            spans = [{"token_start":start,"token_end":end,"label":c} for c,(start,end) in  zip(classes,locs)]
            self.visualizer.render(tokens,spans)


    def mapping_entities(self,config={},**kwargs):
        '''
        config: {old_entity:new_entity}

        kwargs: old_entity = new_entity
        '''
        config.update(kwargs)

        entities = [config[e] if e in config else e for e in self.entities ]

        self.set_attribute(entities = entities)


    def save_results_to_baidu_platform(self,path):
        import pandas as pd
        from collections import defaultdict
        data = defaultdict(dict)
        
        for k,(entities,labels) in self.result.items():
            offset = 0
            idx = 1
            for ent,label in zip(entities,labels):
                if label:
                    start = offset
                    end = offset + len(ent) - 1
                    k = k.replace(" ","")
                    data[k][f"实体标注{idx}"] = f"[{start},{end}],{label}"
                    idx += 1
                offset += len(ent)

        df = pd.DataFrame(data)

        df = df.transpose()
        
        df.to_excel(path)
        

    def save_results_inline(self,path = None,return_lines = False, add_id = False):
        lines = []
        for k,(entities,labels) in self.result.items():
            offset = 0
            line = ""
            idx = 0
            for ent,label in zip(entities,labels):
                if label:
                    if add_id:
                        line += f"【{label}|{idx} {ent} 】"
                    else:
                        line += f"【{label} {ent} 】"
                        
                    idx += 1
                else:
                    line += ent
            lines.append(line)
        
        if path is not None:
            f = open(path,'w',encoding='utf-8')
            for line in lines:
                f.write(line.strip() + '\n')
            f.close()
        
        if return_lines:
            return lines

    def _report_per_sentence(self,text,labels,classes,probs):
        log = f'text:{text}\n'
        
        for l,c,p in zip(labels,classes,probs):
            log += f'\t pred: {l} \t entity:{c} \t prob: {p} '
            
        print(log)
     


    def _viterbi_decode(self,text,observation):
        '''
        viterbi解码
        '''
        if hasattr(self.config,"transition"):
            transition = torch.tensor(self.transition)
        else:
            transition = torch.ones(self.config.num_labels,self.config.num_labels)/ self.config.num_labels
            pass

        #取对数
        observation = observation.log()
        transition = transition.log()

        dp = torch.zeros_like(observation)
        prev_states = torch.zeros_like(dp)
        for i in range(len(observation)):
            if i == 0:
                dp[i] = observation[i]
            else:
                for t in range(len(transition)):
                    arr = torch.tensor([dp[i-1,c] + transition[c,t] + observation[i,t] for c in range(len(transition))])
                    dp[i,t] = torch.max(arr)
                    prev_states[i,t] = torch.argmax(arr)

        path = [torch.argmax(dp[-1])] 
        for i in range(len(observation)-2,-1,-1):
            prev = prev_states[i+1,int(path[-1])]
            path.append(prev)
            # path.append(prev_states[i,int(prev)])
        
        path.reverse()
        return self._decode(text, torch.tensor(path).long() ,observation)

    def _to_scalar(self,value):
        if isinstance(value, (int,str,float)):
            return value 
        elif isinstance(value, torch.Tensor):
           return value.clone().cpu().item()
        else:
            return value[0]

    def _decode(self,text,pred):
        '''
        有限状态机解码
        '''
        entities = []
        locs = []
        labels = []
        end = 0
        
        #最大长度
        max_len = min(510,len(text),len(pred))
        for start in range(max_len):
            c = pred[start]

            if start < end:
                    continue
            #BIO
            elif (self.ner_encoding == 'BIO' and c > 0) or (self.ner_encoding is None): #start B
                class_id = (self._to_scalar(c)-1) // 2
                
                end = start
                while end < max_len and (pred[end] - 1) // 2 == class_id:
                    if end == max_len - 1 or pred[end + 1] == 0 or (pred[end+1] - 1) // 2 != class_id : #stop I
                        entities.append(text[start:end+1])
                        locs.append([start,end])
                        labels.append(self.entities[class_id])
                        end += 1
                        break
                    else:
                        end += 1
            #IO
            elif self.ner_encoding == 'IO' and c > 0:
                class_id = (self._to_scalar(c)-1)-1

                end = start + 1
                while end < max_len and pred[end] != c:
                    end += 1
                
                labels.append(self.entities[class_id])
                entities.append(text[start:end])
                locs.append([start,end-1])

            #BIOE
            elif self.ner_encoding == 'BIOE' and (c % 3 == 1 or (c % 3 == 0 and c > 0)):
                class_id = (self._to_scalar(c)-1) //3
                
                end = start + 1                    
                while end < max_len:
                    if (pred[end] > 0 and pred[end] % 3 == 0) or end == max_len - 1: # E : break
                        labels.append(self.entities[class_id])
                        entities.append(text[start:end+1])
                        locs.append([start,end])
                        end += 1
                        break
                    elif pred[end] % 3 == 1 or pred[end] == 0: # O or B : break
                        labels.append(self.entities[class_id])
                        entities.append(text[start:end])
                        locs.append([start,end-1])
                        break
                    else: # I : continue
                        end += 1

            #BIOES
            elif self.ner_encoding == 'BIOES' and (c % 4 == 1 or (c % 4 == 0 and c > 0)):
                class_id = (self._to_scalar(c)-1) //4
                
                end = start + 1
                if c % 4 == 0 and c > 0: # S: break
                    entities.append(text[start])
                    labels.append(self.entities[class_id])
                    locs.append([start,start])
                    continue
                    
                while end < max_len:
                    if pred[end] % 4 == 3 or end == max_len - 1: # E : break
                        labels.append(self.entities[class_id])
                        entities.append(text[start:end+1])
                        locs.append([start,end])
                        end += 1
                        break
                    elif pred[end] % 4 == 0 or pred[end] % 4 == 1: # O or S or B : break
                        labels.append(self.entities[class_id])
                        entities.append(text[start:end])
                        locs.append([start,end-1])
                        break
                    else: # I : continue
                        end += 1
            else:
                entities.append(text[start])
                labels.append("")
                locs.append([start,start])

        return entities,locs,labels
    
    def _save_per_sentence_result(self,text,entities,locs,labels):
        result = {}
        for idx,(l,loc,c,p) in enumerate(zip(entities,locs,labels)):
            if idx == 0:
                result[f'entity'] = l
                result[f'loc'] = loc
                result[f'label'] = c
            else:
                result[f'entity_{idx+1}'] = l
                result[f'loc_{idx+1}'] = loc
                result[f'label_{idx+1}'] = c
                
        self.result["".join(text)] = result
        
    def compute_metrics(self,eval_pred):
        preds,labels = eval_pred

        pred_labels = torch.tensor(self.model.crf.decode(torch.tensor(preds,device=self.device))).cpu().numpy()
        eval_pred = (preds,pred_labels,labels)
        dic = metrics_for_ner(eval_pred,self.ner_encoding)
        
        #如果有弱监督学习器，则收集推理的数据
        if self.scorer and hasattr(self, "trainer"):
            dic["pred_labels"] = pred_labels.tolist()
            

        # else:
            # dic = metrics_for_ner(eval_pred,self.ner_encoding)
        # print(dic)
        return dic
    
