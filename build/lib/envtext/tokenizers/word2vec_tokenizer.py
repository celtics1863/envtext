VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt",
               "jieba_vocab_file":"/root/data/new_pretrain/vocab.final",
               "word2vec_file":"/root/NLP/models/word2vec/word2vec64"}

from gensim.models import Word2Vec
from threading import Thread #多线程加速
from ..files import Config
import numpy as np


class Word2VecTokenizer:
    def __init__(self,truncation = True,padding = True,max_length = 128,word2vec_path = None,encode_character=True):
        '''
        Args:
            truncate (`bool`): 截断至最大长度 
                默认：TRUE
            padding (`bool`): 填充至最大长度 
                默认：TRUE
            max_length (`int`): 最大长度
                默认: 128
            word2vec_path (`str`): 导入本地的gensim训练后的word2vec模型
               默认：None
            encode_character (`str`) :
                如果word2vec中没有词的时候，按照单字进行解码，向量被padding到每一个字上

            add_bert_tokens (`bool`)：在首尾添加[CLS]和[SEP]
        '''
        import jieba
        jieba.load_userdict(Config.env_vocab)
        self._jieba_tokenizer = lambda text: jieba.cut(text)
        self.word2vec = Word2Vec.load(Config.word2vec64) if word2vec_path is None \
                        else Word2Vec.load(word2vec_path)
        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length
        self.vector_size = self.word2vec.wv.vector_size
        self.encode_character = encode_character
        
        # self.unk_values = self.word2vec.wv.vectors.mean(axis = 0).tolist() #空值，word2vec中没有的词
        self.unk_values = [0.]*self.word2vec.wv.vector_size #空值，word2vec中没有的词
        self.padding_values = [0.]*self.word2vec.wv.vector_size #填充值
        self.eps = 1e-5
    
    def _wash_text(self,text):
        import re
        return re.sub("\s","",text)

    def _encode_per_sentence(self,text, encode_for_bert = False):
        words = self._jieba_tokenizer(self._wash_text(text))
        
        if encode_for_bert:
            vectors = [self.padding_values]
        else:
            vectors = []

        for word in words:
            if self.word2vec.wv.has_index_for(word):
                vectors.append(self.word2vec.wv.get_vector(word).tolist())
            else:
                if not self.encode_character:
                    vectors.append(self.unk_values)
                else:
                    for t in word:
                        if self.word2vec.wv.has_index_for(t):
                            vectors.append(self.word2vec.wv.get_vector(t).tolist())
                        else:
                            vectors.append(self.unk_values)
        
        if encode_for_bert:
            vectors.append(self.padding_values)

        if self.truncation:
            vectors = vectors[:self.max_length]
        
        if self.padding:
            vectors += [self.padding_values] * (self.max_length-len(vectors))

        
        return vectors
    
    
    def encode(self,texts, return_tensors = None, encode_for_bert = False,**kwargs):
        if isinstance(texts,str):
            tokens = [self._encode_per_sentence(texts, encode_for_bert = encode_for_bert)]
        elif isinstance(texts,(list,set)):
            tokens = []
            for text in texts:
                vectors = self._encode_per_sentence(text, encode_for_bert = encode_for_bert)
                tokens.append(vectors)
        else:
            raise NotImplemented
        
        if return_tensors == 'pt':
            import torch
            return torch.tensor(tokens)
        else:
            return tokens
    
    def _decode_per_vector(self,vector):
        return self.word2vec.wv.most_similar(np.array(vector),topn=1)[0][0]
    
    def _distance_for_vectors(self,vA,vB):
        return max(abs(np.array(vA)-np.array(vB)))
    
    def get_vector(self,word):
        if self.word2vec.wv.has_index_for(word):
            return self.word2vec.wv.get_vector(word).tolist()
        else:
            return self.unk_values
        
    def _decode_per_sentence(self,vectors):
        words = []
        for v in vectors:
            
            if self._distance_for_vectors(v,self.unk_values) < self.eps:
                words.append('X')
            elif self._distance_for_vectors(v,self.padding_values) < self.eps:
                continue
            else:
                word = self._decode_per_vector(v)
                words.append(word)
        return words
                                            
    def decode(self,tokens):
        texts = []
        import torch
        #处理输入
        if isinstance(tokens,torch.Tensor):
            tokens = tokens.clone().detach().cpu().numpy().tolist()
        elif isinstance(tokens,np.ndarray):
            tokens = tokens.tolist()
        if not isinstance(tokens[0],list):
            tokens = [tokens]
        
        #decode
        for vectors in tokens:
            texts.append(self._decode_per_sentence(vectors))

        return texts

 
    def distance(self,wordA,wordB):
        return self.word2vec.wv.distance(wordA,wordB)
                
    def __call__(self,texts,return_tensors = None,encode_for_bert = False,**kwargs):
        return self.encode(texts,return_tensors,encode_for_bert = encode_for_bert,**kwargs)
            
            