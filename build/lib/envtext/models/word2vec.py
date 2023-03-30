from gensim.models import Word2Vec
from ..files import FileConfig

config = FileConfig()

def load_word2vec(word2vec_path = None):
    '''
    导入word2vec模型
    Args:
       word2vec_path [Optional] `str`:
           word2vec路径，如果设置，则从此路径导入word2vec
    '''
    if word2vec_path is None:
        model = Word2Vec.load(config.word2vec64)
    
    else:
        model = Word2Vec.load(word2vec_path)
        
    return model.wv