
# 模板采用 spacy 的模板： https://github.com/explosion/spaCy/blob/master/spacy/displacy/render.py
import random
from .visualizer_base import VisualizerBase

class SAVisualizer(VisualizerBase):    
    TMP = """ 
        <progress value='{value}' max='{max}'></progress>{text}{text}
        """
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
    
    def generate_html(self,text,labels,ranges = None,*args,**kwargs):
        '''
        text:文本
        labels：值
        ranges：范围
        '''
        html = '<div style="line-height:2.5;"/>'
        params = {
          "max": '1',
          "text": '',
          "value": "0",
            }

        if ranges is None:
            ranges = [(0,1) for i in range(len(labels))]

        for idx,(label,_range) in enumerate(zip(labels,ranges)):
            param = params.copy()
            param['value'] = label - _range[0]
            param['max'] = _range[1] - _range[0]
            html += self.TMP.format(**param)
        
        html += f"{text}"
        return html
    
    def generate_text(self,text,labels,probs,*args,**kwargs):
        log = f'text:{text} \n'
        for i,j in  zip(labels,probs):
            log += '\t pred_classes:{}, \t probability:{:.4f} \n'.format(i,j)
        return log


    def export_html(self,html,path):
        f = open(path,'w',encoding='utf-8')
        f.write(html)
        f.close()
        
    def __call__(self,**kwargs):
        self.render(**kwargs)
            