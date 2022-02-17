env_vocab = 'envText/files/env_vocab.jieba.txt'
onehot_vocab = 'envText/files/onehot_vocab.txt'
bert_vocab = 'envText/files/bert_vocab.txt'
word2vec64 = 'envText/files/word2vec64'
word2vec256 = 'envText/files/word2vec256'


import os

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
    
    
    def get_word2vec_path(self,vector_size = 64):
        if vector_size == 256:
            return self.word2vec256
        else:
            return self.word2vec64

    def get_abs_path(self,relative_path):
        return os.path.realpath(relative_path)