env_vocab = './vocab/env_vocab.jieba.txt'
onehot_vocab = './vocab/onehot_vocab.txt'
bert_vocab = './vocab/bert_vocab.txt'
stop_words = './vocab/stop_words.txt'

font_dir = './fonts'

word2vec64 = './pretrained_models/word2vec64'
word2vec256 = './pretrained_models/word2vec256'
datasets_dir = './datasets'
sa_intensity = './datasets/SA_Intensity.json'
cls_isclimate = './datasets/CLS_IsClimate.json'
cluener = './datasets/CLUENER.json'

templates_dir = './templates'
js_dir = './js'

import os

basedir = os.path.dirname(__file__)


def load_env_vocab():
    words = []
    with open(Config.env_vocab, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            words.append(line.strip().split(" ")[0])    
    return words



def generate_properties(names):
    """用于批量生成 property 的函数"""
    def getter(name):
        return lambda self: self._data.get(name)

    def setter(name):
        return lambda self, value: self._data.update({name: value})

    properties = {}
    for name in names:
        properties[name] = property(getter(name), setter(name))
    return properties

class ModelConfig:
    def __init__(self, data):
        self._data = data
        for key, value in generate_properties(data.keys()).items():
            setattr(self.__class__, key, value)


class FileConfig:
    def __init__(self):
        self.bert = ModelConfig(
            {
                "bert_mlm":"celtics1863/env-bert-chinese",
                "topic_cls":"celtics1863/env-bert-topic",
                "paper_cls":"celtics1863/env-paper-cls-bert",
                "policy_cls":"celtics1863/env-policy-cls-bert",
                "news_cls":"celtics1863/env-news-cls-bert",
                "pos_ner":"celtics1863/pos-bert",
            }
        )        


        self.albert = ModelConfig(
            {
                "albert_mlm":"celtics1863/env-albert-chinese",
                "topic_cls":"celtics1863/env-topic-albert",
                "paper_cls":"celtics1863/env-paper-cls-albert",
                "news_cls":"celtics1863/env-news-cls-albert",
                "policy_cls":"celtics1863/env-policy-cls-albert",
                "pos_ner":"celtics1863/env-pos-ner-albert"
            }
        )
    
    @property
    def pretrained_cls_models(self):
        cls_models = []
        for k,v in self.albert._data.items():
            if k.endswith("cls"):
                cls_models.append(v)

        for k,v in self.bert._data.items():
            if k.endswith("cls"):
                cls_models.append(v)
        
        return cls_models

    @property
    def pretrained_ner_models(self):
        ner_models = []
        for k,v in self.albert._data.items():
            if k.endswith("ner"):
                ner_models.append(v)

        for k,v in self.bert._data.items():
            if k.endswith("ner"):
                ner_models.append(v)
        
        return ner_models

    @property
    def pretrained_mlm_models(self):
        mlm_models = []
        for k,v in self.albert._data.items():
            if k.endswith("mlm"):
                mlm_models.append(v)

        for k,v in self.bert._data.items():
            if k.endswith("mlm"):
                mlm_models.append(v)
        
        return mlm_models

    @property
    def pretrained_models(self):
        return list(self.albert._data.values()) + list(self.bert._data.values())

    @property
    def env_vocab(self):
        return self.get_abs_path(env_vocab)
        
    @property
    def onehot_vocab(self):
        return self.get_abs_path(onehot_vocab)
    
    @property
    def bert_vocab(self):
        return self.get_abs_path(bert_vocab)
    
    @property
    def word2vec64(self):
        return self.get_abs_path(word2vec64)
    
    @property
    def word2vec256(self):
        return self.get_abs_path(word2vec256)
    
    @property
    def datasets(self):
        datasets = []
        for file in os.listdir(os.path.join(basedir,datasets_dir)):
            datasets.append(os.path.normpath(os.path.join(basedir,datasets_dir,file)))
        return datasets
    
    @property
    def SA_Intensity(self):
        return {
            'path':self.get_abs_path(sa_intensity),
            'name':['SA','sa','sa_intensity','reg','regression'],
            'task':'sentitive analysis',
            'format':'json2'
        }

    @property
    def CLS_IsClimate(self):
        return {
            'path':self.get_abs_path(cls_isclimate),
            'name':['cls','classification','isclimate','cls_isclimate'],
            'task':'classification',
            'format':'json2'
        }

    
    @property
    def CLUENER(self):
        return {
            'path':self.get_abs_path(cluener),
            'name':['ner','namely entity recognition','clue ner','cluener'],
            'task':'cluener',
            'format':'jsonL'
        }
    
    @property
    def datasets_info(self):
        info = {
            'sa_intensity':self.SA_Intensity,
            'cls_isclimate':self.CLS_IsClimate,
            'cluener':self.CLUENER
        }
        return info
    
    @property
    def datasets_names(self):
        info = self.datasets_info
        names = {}
        for k,v in info.items():
            for name in v['name']:
                names[name] = k
        return names
    
    @property
    def fonts(self):
        _font_dir = self.get_abs_path(font_dir)
        fonts = {font.split('.')[0]: os.path.normpath(os.path.join(_font_dir,font))  for font in os.listdir(_font_dir)}
        return fonts
    
    @property
    def stop_words(self):
        stop_words_path = self.get_abs_path(stop_words)
        
        _stop_words = {}
        
        with open(stop_words_path,'r',encoding='utf-8') as f:
            for word in f:
                _stop_words[word.strip()] = True
            f.close()
        
        return _stop_words
    

    @property
    def templates_dir(self):
        return self.get_abs_path(templates_dir)

    @property
    def js_dir(self):
        return self.get_abs_path(js_dir)

    @property
    def js(self):
        _js_dir = self.get_abs_path(js_dir)
        js = [os.path.normpath(os.path.join(_js_dir,j))  for j in os.listdir(_js_dir)]
        return js

    def get_word2vec_path(self,vector_size = 64):
        if vector_size == 256:
            return self.word2vec256
        else:
            return self.word2vec64

    def get_abs_path(self,relative_path):
        return os.path.normpath(os.path.join(basedir,relative_path))



Config = FileConfig()