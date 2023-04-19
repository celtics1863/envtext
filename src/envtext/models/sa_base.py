from ..utils.metrics import metrics_for_reg
from ..visualizers import SAVisualizer


class SABase:
    '''
    Bert情感分析/回归模型
    
    Args:
        path `str`: 默认：None
            预训练模型保存路径，如果为None，则从celtics1863进行导入预训练模型
        
        config [Optional] `dict` :
            配置参数
            
    Kwargs:
       max_length [Optional] `int`: 默认：128
           支持的最大文本长度。
           如果长度超过这个文本，则截断，如果不够，则填充默认值。
   '''

    def align_config(self,*args,**kwargs):
        super().align_config(*args,**kwargs)

        if self.range is None:
            self.set_attribute(range = [[0,1]]) # list[[start, end]]

        if not hasattr(self.config, "visualizer"):
            self.set_attribute(visualizer = "sa")
        
        if self.config.visualizer == "sa":
            self.visualizer = SAVisualizer(range= self.range)

    @property
    def range(self):
        if hasattr(self.config, "range"):
            return self.config.range
        else:
            return None

    def postprocess(self,text, logits, **kwargs):
        logits = logits[0]
        return logits

        
    def compute_metrics(self,eval_pred):
        dic = metrics_for_reg(eval_pred)
        return dic