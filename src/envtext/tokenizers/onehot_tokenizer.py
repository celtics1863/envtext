from ..files import FileConfig
import numpy as np #for np.eye()

config = FileConfig()
class OnehotTokenizer:
    def __init__(self,truncation = True,padding = True,max_length = 128,vocab_path = None):
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
        '''
        if vocab_path is None:
            self.vocab = self._make_vocab(config.onehot_vocab)
        else:
            self.vocab = self._make_vocab(vocab_path)
        self.vector_size = len(vocab)
        self.unk_values = (np.ones(vocab)/self.vector_size).tolist()
        self.padding_values = np.zeros(vocab).tolist()
        
        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length
        
        self.eps = 1e-5
    
    def _make_vocab(self,path):
        f = open(VOCAB_FILES_NAMES["onehot_vocab"],'r',encoding = 'utf-8')
        words = [line.strip() for line in f.readlines()]
        vocab = np.eye(len(words)).tolist()
        vocab = {word:onehot for word,onehot in zip(words,vocab)}
        reverse_vocab = {word: idx for idx,word in enumerate(words)}
        return vocab
        
    def _decode_per_word(self,vector):
        return np.nonzero(vector)[0][0]
        
    def _encode_per_sentence(self,text):
        vectors = [self.vocab[word] if word in vocab.keys() else self.unk_values for word in text]
        
        if self.truncation:
            vectors = vectors[:self.max_length]
        
        if self.padding:
            vectors += [self.padding_values] * (self.max_length-len(vectors))
        
        return vectors
    
    def _distance_for_vectors(self,vA,vB):
        return max(abs(vA-vB))
    
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
                                            
    def encode(self,texts, return_tensor = None):
        if isinstance(texts,str):
            tokens = [self._encode_per_sentence(texts)]
        elif isinstance(texts,list):
            tokens = [self._encode_per_sentence(text) for text in texts]
        else:
            raise NotImplemented
        
        if return_tensor == 'pt':
            import torch
            return torch.tensor(tokens)
        else:
            return tokens
        
    def __call__(self,texts,return_tensor = None):
        return self.encode(texts,return_tensor)
        
        
        