from gensim.models import KeyedVectors
from ..files import Config
from ..utils.vocab_ops import load_jieba_vocab
import jieba
import numpy as np
import os
from typing import *

class Sim2Vec:
    '''
    一种基于词表构造句子相似度的方法

    假设词表中一共有三个词 [A, B ,C]
    词向量大小为3：

    对于每个句子，首先进行识别，例如以下句子：
    （1）AABBBC  对应的向量为：[2,3,1]
    （2）BAAAC   对应的向量为：[3,1,1]
    
    如果出现了不在词表里的词，例如D，计算其和A，B，C的相似度，
    例如 sim(D,A) = 0.1, sim(D,B) = 0.2, sim(C,B) = 0.5
    D 的向量为 [0.1, 0.2 ,0.5]
    '''
    def __init__(self, word2vec = None,
                    vocab : Union[List[str],Tuple[str]] = None,
                    word2vec_size: Union[os.PathLike, None] = None,
                    word2vec_path : Union[os.PathLike,None] = None,
                    vocab_path : Union[os.PathLike,None] = None,
                    default_freq: int = 1000
                    ):

        if word2vec is not None:
            if word2vec_path is not None:
                self.word2vec = load_word2vec(word2vec_path)
            else:
                assert word2vec_size is not None, "word2vec_size is not specified"
                self.word2vec = KeyedVectors(word2vec_size)
        else:
            self.word2vec = word2vec

        if vocab is not None:
            self.vocab = list(vocab)
            for v in self.vocab:
                jieba.add_word(v, default_freq)
        else:
            if vocab_path is None:
                jieba.load_userdict(Config.env_vocab)
                self.vocab = load_jieba_vocab(Config.env_vocab)
            elif  os.path.exists(vocab_path):
                jieba.load_userdict(Config.env_vocab)
                self.vocab = load_jieba_vocab(Config.env_vocab)
            else:
                raise NotImplemented


        self._mapping = {word:idx for idx,word in enumerate(self.vocab.keys())}

        self.tokenizer = lambda x: [word for word in jieba.lcut(x,cut_all = True) if word in self.vocab \
                                         and self.word2vec.wv.has_index_for(word)]
    
        self.vector_size = len(self.vocab)
        self.wv = KeyedVectors(len(self.vocab))

        
    def add_text(self,text):
        vec = self.get_vector(text)

        self.wv.add_vector(text, vec)

    def add_texts(self,texts):
        if len(texts):
            vecs = []
            keys = []
            for t in texts:
                # self.add_text(t)
                vecs.append(self.get_vector(t))
                keys.append(t)

            self.wv.add_vectors(keys, vecs)

    def get_vector(self,text):    
        if isinstance(text,str):
            words = self.tokenizer(text)
        elif isinstance(text, (list,tuple)):
            words = list(text)
        else:
            raise NotImplemented

        vec = np.zeros(self.vector_size)
        for word in words:
            p = self.word2vec.wv.most_similar(word,topn=1)[0][1]
            vec[self._mapping[word]] = p
        
        return vec

    def most_similar(self,text,topn = 5):
        vec = self.get_vector(text)

        return self.wv.most_similar(vec,topn=topn)

    def inference(self):
        pass

    def clear(self):
        self.wv = KeyedVectors(len(self.vocab))