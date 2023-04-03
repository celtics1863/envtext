from .visualizer_base import VisualizerBase
class POSVisualizer(VisualizerBase):
    DEFAULT_LABEL_COLORS = {
        "n":"lightblue",
        "vn":"lightblue",
        "v":"lightblue",
        "d":"lightblue",
        "a":"lightblue",
        "p":"lightblue",
        "f":"lightblue",
        "q":"lightblue",
        "m":"lightblue",
        "conj":"lightblue",
        "u":"lightblue",
        "xc":"lightblue",
        "w":"lightblue",
        "med":"lightgreen",
        "phe":"lightgreen",
        "microbe":"lightgreen",
        "plant":"lightgreen",
        "animal":"lightgreen",
        "desease":"lightgreen",
        "hy":"lightgreen",
        "group":"lightgreen",
        "act":"lightgreen",
        "policy":"lightgreen",
        "b":"lightgreen",
        "env":"lightgreen",
        "time":"lightcoral",
        "loc":"lightcoral",
        "com":"lightcoral",
        "org":"lightcoral",
        "gov":"lightcoral",
        "doc":"lightcoral",
        "event":"lightcoral",
        "pro":"lightcoral",
        "ins":"lightcoral",
        "means":"lightcoral",
        "meet":"lightcoral",
        "code":"lightcoral",
        "c":"lightcoral"
    }

    DEFAULT_LABEL_MAPPING = {
        "n":"n/名词",
        "vn":"vn/动名词",
        "v":"v/动词",
        "d":"d/副词",
        "a":"a/形容词",
        "p":"p/介词",
        "f":"f/方位词",
        "q":"q/量词",
        "m":"m/数词",
        "conj":"conj/连词",
        "u":"u/助词",
        "xc":"xc/虚词",
        "w":"w/标点",
        "med":"环境介质",
        "phe":"环境现象",
        "microbe":"微生物",
        "plant":"植物",
        "animal":"动物",
        "desease":"疾病",
        "hy":"行业",
        "group":"群体",
        "act":"政策行动",
        "policy":"政策工具",
        "b":"属性",
        "env":"环境术语",
        "time":"时间",
        "loc":"地点",
        "com":"公司名",
        "org":"组织",
        "gov":"政府",
        "doc":"文件名",
        "event":"事件名",
        "pro":"工程/项目/设施",
        "ins":"设备/工具",
        "means":"方法/技术",
        "meet":"会议",
        "code":"编码",
        "c":"专名"
    }

    TMP = """ 
        {text} 
        <span style="font-size: 0.8em; background: {bg};  font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-right: 0.5rem">{label}{kb_link}</span>
    """
    
    def generate_html(self,text,words,poses,*args,**kwargs):
        '''
        生成html格式的可视化
        '''
        html = '''
        <div style="line-height:2.5;">
        '''
        params = {
          "label": '标签',
          "text": '实体',
          "bg": 'red',
          "kb_link": ''
            }
        
        for word,pos in zip(words,poses):
            pos = pos.lower()
            if pos in self.DEFAULT_LABEL_COLORS:
                param = params.copy()
                param['bg'] = self.DEFAULT_LABEL_COLORS[pos]
                param['text'] = "".join(word)
                param['label'] = self.DEFAULT_LABEL_MAPPING.get(pos,pos)
                html += self.TMP.format(**param)
            elif pos:
                #随机分配颜色
                param = params.copy()
                param['bg'] = "lightblue"
                param['text'] = "".join(word)
                param['label'] = self.DEFAULT_LABEL_MAPPING.get(pos,pos)
                html += self.TMP.format(**param)
            else:
                html += "".join(word)
        html += "</div>"
        return html


    def generate_text(self,text,words,poses,*args,**kwargs):
        '''
        生成纯文本的输出
        '''
        text = "".join(list(text))
        for w,p in zip(words,poses):
            text += f"\t{w}\t{p}\n"
        return text
