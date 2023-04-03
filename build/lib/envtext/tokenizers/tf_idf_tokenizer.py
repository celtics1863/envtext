from ..files import Config
import numpy as np #for np.eye()
import math
import jieba
from collections import Iterable

class TFIDFTokenizer:
    def __init__(self,truncation = True,padding = True,max_length = 128,vocab_path = None,use_jieba = True):
        '''
        Args:
            truncate (`bool`): 截断至最大长度 
                默认：TRUE
            padding (`bool`): 填充至最大长度 
                默认：TRUE
            max_length (`int`): 最大长度
                默认: 128
            use_jieba (`bool`): 默认：True
                使用jieba
        '''

        self.vocab = {}

        if use_jieba:
            jieba.load_userdict(Config.env_vocab)
            self._jieba_tokenizer = lambda text: jieba.cut(text)
        else:
            self._jieba_tokenizer = lambda text: jieba.cut(" ".join(text))

        if vocab_path is not None:
            self.make_vocab(vocab_path)
        elif vocab_path == "default":
            self.make_vocab(Config.env_vocab)


        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length
        
        self.vector_size = len(self.vocab) + 1

        self.eps = 1e-5

        self.unk_values = [0.,0.,0.,0.]
        self.padding_values = [0.,0.,0.,0.]

        self._total_doc_nums = 0

    
    def make_vocab(self,path_or_lines):
        if isinstance(path_or_lines, str):
            f = open(path_or_lines,'r',encoding = 'utf-8')
            self.vocab = {line.strip().split(" ")[0]:[0,1,idx + 1] for idx,line in enumerate(f.readlines())}
        elif isinstance(path_or_lines, Iterable):
            for line in path_or_lines:
                is_exist = set()
                for word in self._jieba_tokenizer(line):
                    if word not in self.vocab:
                        self.vocab[word] = [1,2,len(self.vocab)]
                    else:
                        self.vocab[word][0] += 1
                        if word not in is_exist:
                            self.vocab[word][1] += 1

                self._total_doc_nums += 1
        
        self.vector_size = len(self.vocab) + 1
        self._bak_values = np.zeros(self.vector_size,dtype=np.float32).tolist()
        self.padding_values = self._bak_values.copy()
        self.unk_values = self._bak_values.copy()


    def _decode_per_word(self,vector):
        assert NotImplementedError()
        
    def _encode_per_sentence(self,text):

        def get_tf_idf(x):
            vec = self._bak_values.copy()
            vec[x[2]] = - x[0] * math.log(self.eps + x[1]/(self._total_doc_nums + self.eps))
            return vec

        vectors = [get_tf_idf(self.vocab[word]) if word in self.vocab else self.unk_values for word in text]
        
        if self.truncation:
            vectors = vectors[:self.max_length]
        
        if self.padding:
            vectors += [self.padding_values] * (self.max_length-len(vectors))
        
        return vectors
    
    def _distance_for_vectors(self,vA,vB):
        return max(abs(np.array(vA)-np.array(vB)))
    
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
        import numpy as np
        #处理输入
        if isinstance(tokens,torch.Tensor):
            tokens = tokens.clone().detach().cpu().numpy().tolist()
        elif isinstance(tokens,numpy.ndarray):
            tokens = tokens.tolist()
        if not isinstance(tokens[0],list):
            tokens = [tokens]
        
        #decode
        for vectors in tokens:
            texts.append(self._decode_per_sentence(vectors))

        return texts
                                            
    def encode(self,texts, return_tensors = None,**kwargs):
        if isinstance(texts,str):
            tokens = [self._encode_per_sentence(texts)]
        elif isinstance(texts,(list,set)):
            tokens = [self._encode_per_sentence(text) for text in texts]
        else:
            raise NotImplemented
        
        if return_tensors == 'pt':
            import torch
            return torch.tensor(tokens)
        else:
            return tokens
        
    def __call__(self,texts,return_tensors = None,**kwargs):
        return self.encode(texts,return_tensors,**kwargs)
        
        
        