env_vocab = './env_vocab.jieba.txt'
onehot_vocab = './onehot_vocab.txt'
bert_vocab = './bert_vocab.txt'
word2vec64 = './word2vec64'
word2vec256 = './word2vec256'
datasets_dir = './datasets'
sa_intensity = './datasets/SA_Intensity.json'
cls_isclimate = './datasets/CLS_IsClimate.json'
cluener = './datasets/CLUENER.json'

import os

basedir = os.path.dirname(__file__)

class FileConfig:
    def __init__(self):
#         print(os.path.normpath(env_vocab))
        pass
        
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
            datasets.append(os.path.join(basedir,datasets_dir,file))
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
    
    def get_word2vec_path(self,vector_size = 64):
        if vector_size == 256:
            return self.word2vec256
        else:
            return self.word2vec64

    def get_abs_path(self,relative_path):
        return os.path.normpath(os.path.join(basedir,relative_path))