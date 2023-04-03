
# 模板采用 spacy 的模板： https://github.com/explosion/spaCy/blob/master/spacy/displacy/render.py

from palettable.colorbrewer.qualitative import Set3_12,Set3_12_r
from palettable.tableau import  Tableau_20_r,TableauLight_10,TableauMedium_10,Tableau_20
from ..utils.html_ops import *
import random


class VisualizerBase:
    DEFAULT_LABEL_COLORS = {
    }
    
    
    TMP = """ 
        """
    
    def __init__(self,*args,**kwargs):
        try:
            from IPython.display import display_html
            self.env = 'jupyter'
        except:
            self.env = 'cmd'
        
        self.COLORS = Tableau_20_r.hex_colors
    
    def generate_html(self,*args,**kwargs):
        html = '<div style="line-height:2.5;"/>'
        return html
    
    def _get_color(self,label):
        if label in self.DEFAULT_LABEL_COLORS:
            return self.DEFAULT_LABEL_COLORS[label]
        else:
            color = self.COLORS[len(self.DEFAULT_LABEL_COLORS) % len(self.COLORS)]
            self.DEFAULT_LABEL_COLORS[label] = color
        return color

    def render(self,*args,**kwargs):
        if self.env == 'jupyter':
            html = self.generate_html(*args,**kwargs)
            from IPython.display import display_html
            display_html(html,raw = True)
        else:
            text = self.generate_text(*args,**kwargs)
            print(text)
        
        if "save_path" in kwargs:
            save_path = kwargs["save_path"]
            import os
            dir_name = os.path.dirname(os.path.realpath(save_path))

            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
                
            self.export_html(html,save_path)
    
    def export_html(self,html,path):
        f = open(path,'w',encoding='utf-8')
        f.write(html)
        f.close()
    
    def export_svg(self,html,path):
        pass


    def export_png(self,html,path):
        pass


    def __call__(self,*args,**kwargs):
        self.render(*args,**kwargs)
            