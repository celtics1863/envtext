from ..visualizers import CLSVisualizer
from ..utils.metrics import metrics_for_cls_with_binary_logits
import numpy as np

class MCBase:
    def align_config(self):
        super().align_config()
        if self.labels:
            self.update_config(num_labels = len(self.labels))   
        elif self.num_labels:
            self.update_config(labels = list(range(self.num_labels)))
        else:
            self.update_config(num_labels = 1,
                         labels = ['LABEL_0'],
                         id2label = {0:'LABEL_0'},
                         label2id = {'LABEL_0':0},
                         )

        self.visualizer = CLSVisualizer()
        
    def _calc_resample_prob(self,raw_label,**kwargs):
        labels = set()
        for label in raw_label:
            if label in self.labels:
                labels.add(label)
            elif label in self.id2label:
                labels.add(self.id2label[label])
            else:       
                labels.update(set([k["label"] for k in label]))

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


    def postprocess(self,text, logits, **kwargs):
        logits = logits[0] # logits = (logits, ) ,fetch first of tuple
        
        def sigmoid(z):
            import numpy as np
            return 1/(1 + np.exp(-z))

        logits = sigmoid(logits)
        preds = np.nonzero(logits > 0.5)[0]
        labels = [self.id2label[pred.item()] for pred in preds]
        ps = [logits[pred.item()] for pred in preds]
        return [labels,ps]


        
    def compute_metrics(self,eval_pred):
        dic = metrics_for_cls_with_binary_logits(eval_pred)
        return dic