from ..visualizers import CLSVisualizer
from ..utils.metrics import metrics_for_cls


class CLSBase:
    def align_config(self):
        super().align_config()
        self.visualizer = CLSVisualizer()

    def _visualize(self,text,labels,probs,save_vis,**kwargs):
        if not hasattr(self, "visualizer"):
            self.visualizer = CLSVisualizer()
        
        self.visualizer.render(text,labels,probs,save_vis)

    def _calc_resample_prob(self,raw_label,**kwargs):
        '''
        计算每条文本的重采样概率
        '''
        labels = set()
        
        if raw_label in self.labels:
            labels.add(raw_label)
        elif raw_label in self.id2label:
            labels.add(self.id2label[raw_label])
        else:       
            labels = set([k["label"] for k in raw_label])

        if hasattr(self, "data_config"):
            import torch
            p = torch.tensor([self.data_config["counter"][e] for e in self.labels])
            p = 1/ (p/p.sum()+1e-5) #逆频率
            p = p/(p.sum()) #归一化
            prob = max([p[self.data_config["labels"].index(label)] for label in labels])
            return prob.item()
        else:
            from warnings import warn
            warn("缺少self.data_config，可能是函数self.update_data_config()没有重写的问题")
            return 0


    def _report_per_sentence(self,text,pred,p):
        log = f'text:{text} \n'
        for i,j in  zip(pred,p):
            log += '\t pred_classes:{}, \t probability:{:.4f} \n'.format(self.id2label[i.item()],j)
        print(log)
    
    def _save_per_sentence_result(self,text,pred,p):
        result = {}
        for topk,(i,j) in enumerate(zip(pred,p)):
            if topk == 0:
                result['label'] = self.id2label[i.item()]
                result['p'] = '{:.4f}'.format(j.item())
            else:
                result[f'top{topk+1} label'] = self.id2label[i.item()]
                result[f'top{topk+1} p'] = '{:.4f}'.format(j.item())
        
        self.result[text] = result
        
    def compute_metrics(self,eval_pred):
        dic = metrics_for_cls(eval_pred)

        print(dic)
        return dic

        
    def postprocess(self,text,logits,topk=5,**kwargs):
        '''
        for cls tasks:
            logits: (N, ) N is the number of the labels
        
        return: 
            [label_index_list, prob_list]
        '''
        logits = logits[0]
        topk = topk if len(logits) > topk else logits.shape[-1]
        index = sorted(range(len(logits)), key = lambda x: logits[x], reverse=True)
        
        ind_list = []
        p_list = []
        for ind in index[:topk]:
            p_list.append(logits[ind])
            ind_list.append(self.id2label[ind])
        
        return [ind_list,p_list]