
# 模板采用 spacy 的模板： https://github.com/explosion/spaCy/blob/master/spacy/displacy/render.py
import random
from .visualizer_base import VisualizerBase

class CLSVisualizer(VisualizerBase):    
    TMP = """ 
        <mark class="cls" style="background: {bg}; text-indent:10em; ; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em"> 
            {label} 
            <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-right: 0.5rem">
            {prob}%</span>
        </mark>{text}
        """
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
    
    def generate_html(self,text,labels,probs,*args,**kwargs):
        html = '<div style="line-height:2.5;"/>'
        params = {
          "label": '标签',
          "text": '',
          "bg": 'light red',
          "prob": ''
            }

        for idx,(label,prob) in enumerate(zip(labels,probs)):
            param = params.copy()
            param['bg'] = self._get_color(label)
            param['label'] = label
            param['prob'] = "{:.1f}".format(float(prob)*100)
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
            