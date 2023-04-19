from ..files import Config
import numpy as np #for np.eye()
from collections import Iterable,OrderedDict
import jieba

class OnehotTokenizer:
    def __init__(self,truncation = True,padding = True,max_length = 128,vocab_path = None,use_jieba = True):
        '''
        Args:
            truncate (`bool`): 截断至最大长度 
                默认：TRUE
            padding (`bool`): 填充至最大长度 
                默认：TRUE
            max_length (`int`): 最大长度
                默认: 128
            vocab_path (`str`): 本地词表
               默认：None
            use_jieba (`bool`): 是否使用jieba分词
                默认False
        '''

        self.unk_values = 0
        self.padding_values = 1
        
        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length
        
        self.eps = 1e-5

        self.vocab = OrderedDict() 
        self.vocab.update({"[UNK]":self.unk_values,"[PAD]":self.padding_values})
        
        self.reverse_vocab = OrderedDict() 
        self.reverse_vocab.update({self.unk_values:"[UNK]",self.padding:"[PAD]"})

        if use_jieba:
            # jieba.load_userdict(Config.env_vocab)
            self._jieba_tokenizer = lambda text: jieba.cut(text)
        else:
            self._jieba_tokenizer = lambda text: jieba.cut(" ".join(text))

        if vocab_path is not None:
            self.make_vocab(vocab_path)
        elif vocab_path == "default":
            self.make_vocab(Config.env_vocab)

        self.vector_size = len(self.vocab) + 1
        

    def save_pretrained(self,path):
        import os
        file_path = os.path.join(path,"vocab.txt")
        f = open(file_path,"w",encoding="utf-8")
        for k in self.vocab:
            f.write(k+"\n")
            
    def from_pretrained(self,path):
        import os
        self.vocab = OrderedDict()
        self.reverse_vocab = OrderedDict()
        file_path = os.path.join(path,"vocab.txt")
        if os.path.exists(file_path):
            for line in open(file_path,"r",encoding="utf-8"):
                self.vocab[line.strip()] = len(self.vocab)
                self.reverse_vocab[self.vocab[line.strip()]] = line.strip()
                
        self.vector_size = len(self.vocab) + 1
    
    def make_vocab(self,path_or_lines):
        if isinstance(path_or_lines, str):
            f = open(path_or_lines,'r',encoding = 'utf-8')
            self.vocab = {line.strip().split(" ")[0]:idx+1 for idx,line in enumerate(f.readlines())}
            self.reverse_vocab = {v: key for key,v in self.vocab.items()}
        elif isinstance(path_or_lines, Iterable):
            for line in path_or_lines:
                for word in self._jieba_tokenizer(line):
                    if word not in self.vocab:
                        self.vocab[word] = len(self.vocab)
                        self.reverse_vocab[self.vocab[word]] = word
        
        self.vector_size = len(self.vocab) + 1


    def _decode_per_word(self,vector):
        return self.reverse_vocab[vector]
        
    def _encode_per_sentence(self,text):
        vectors = [self.vocab[word] if word in self.vocab else self.unk_values 
                                    for word in self._jieba_tokenizer(text)]
        
        if self.truncation:
            vectors = vectors[:self.max_length]
        
        if self.padding:
            vectors += [self.padding_values] * (self.max_length-len(vectors))
        
        return vectors
    
    def _distance_for_vectors(self,vA,vB):
        return max(abs(vA-vB))
    
    def _decode_per_sentence(self,vector):
        words = []
        for v in vector:
            if v not in self.reverse_vocab:
                words.append('X')
            else:
                word = self._decode_per_word(v)
                words.append(word)
        return words

    def decode(self,tokens):
        texts = []
        import torch
        import numpy as np
        #处理输入
        if isinstance(tokens,torch.Tensor):
            tokens = tokens.clone().detach().cpu().numpy().tolist()
        elif isinstance(tokens,np.ndarray):
            tokens = tokens.tolist()
        if not isinstance(tokens[0],list):
            tokens = [tokens]
        
        #decode
        for vector in tokens:
            texts.append(self._decode_per_sentence(vector))

        return texts
                                            
    def encode(self,texts, return_tensors = None,**kwargs):
        if isinstance(texts,str):
            tokens = [self._encode_per_sentence(texts)]
        elif isinstance(texts,(list,set)):
            tokens = [self._encode_per_sentence(text) for text in texts]
        else:
            raise NotImplementedError()
        
        if return_tensors == 'pt':
            import torch
            return torch.tensor(np.array(tokens))
        else:
            return tokens
        
    def __call__(self,texts,return_tensors = None,**kwargs):
        return self.encode(texts,return_tensors)
        
        
        