from ..visualizers import CLSVisualizer
from ..utils.metrics import metrics_for_cls_with_binary_logits

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


    def _visualize(self,text,labels,probs,save_vis,**kwargs):
        if not hasattr(self, "visualizer") or self.visualizer is None:
            self.visualizer = CLSVisualizer()
        
        self.visualizer.render(text,labels,probs,save_vis)
        
    def _calc_resample_prob(self,raw_label,**kwargs):
        labels = set([k["label"] for k in raw_label])

        if hasattr(self, "data_config"):
            p = torch.tensor([self.data_config["counter"][e] for e in self.labels])
            p = 1/ (p/p.sum()+1e-5) #逆频率
            p = p/(p.sum()) #归一化
            prob = max([p[self.data_config["labels"].index(label)] for label in labels])
            return prob.item()
        else:
            from warnings import warn
            warn("缺少self.data_config，可能是函数self.update_data_config()没有重写的问题")
            return 0

    def _report_per_sentence(self,text,preds,probs):
        log = f'text: {text}\n'
        for idx,prob in enumerate(probs) :
            log += '\t label: {} \t ; probability : {:.4f}\n'.format(self.id2label[idx],prob)
        print(log)
 
    def _save_per_sentence_result(self,text,preds,probs):
        result = {}
        for idx,prob in enumerate(probs) :
            result[f'label_{idx}'] = self.id2label[idx]
            result[f'p_{idx}'] = prob
        
        self.result[text] = result
        
    def compute_metrics(self,eval_pred):
        dic = metrics_for_cls_with_binary_logits(eval_pred)
        return dic