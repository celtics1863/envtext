from gensim.models import KeyedVectors
import numpy as np
from typing import *
import os
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel, utils
import torch

class Bert2Vec:
    def __init__(self, model_path : Union[os.PathLike,None] = None, 
                    word2vec_path : Union[os.PathLike,None] = None,
                    layer :  Union[int,Tuple,List] = (0,4),
                    batch_size : int = 128,
                    weights : str = "length",
                    half: bool = True
                    ):
        if word2vec_path is not None and word2vec_path:
            self.wv = KeyedVectors.load(word2vec_path)
        else:
            self.wv = KeyedVectors(768)

        self.model_path = model_path
        self.bert = None
        self.tokenizer = None    
        self.layer = layer
        self.batch_size = batch_size
        self.weights = weights
        self.half = half


    def load_weights(self):
        if self.weights == "length":
            # self.weights = 
            pass
        
    def add_word(self,text):
        if text in self.wv.key_to_index:
            print("词已经在词向量中")
            return

        self.load_bert()
        vec = self.get_vec(text)
        self.wv.add_vector(text, vec)
        self.wv.resize_vectors()

    def load_bert(self):
        if self.bert is None:
            import torch
            if torch.cuda.is_available():
                self.bert = BertModel.from_pretrained(self.model_path, output_hidden_states=True).cuda()
            else:
                self.bert = BertModel.from_pretrained(self.model_path, output_hidden_states=True)

            self.tokenizer = BertTokenizerFast.from_pretrained(self.model_path)

    @torch.no_grad()
    def add_words(self,texts):
        self.load_bert()

        texts = list(filter(lambda x:  x not in self.wv.key_to_index , texts))
        if len(texts) == 0:
            print("全部词已经在词向量中")
            return

        vector_list = []
        for i in tqdm(range(0,len(texts),self.batch_size)):
            words = texts[i:i+self.batch_size]
            inputs = self.tokenizer(words, return_tensors='pt',padding="longest")

            inputs = {k:v.to(self.bert.device) for k,v in inputs.items()}

            outputs = self.bert(**inputs)  # Run model

            vector = []

            for idx,w in enumerate(words):
                vec = self._hidden_states2vec(outputs["hidden_states"], 0 , len(w) ,idx)
                vector.append(vec)
            
            vector = np.stack(vector)
            self.wv.add_vectors(words, vector)

            vector_list.append(vector)

        self.wv.resize_vectors()

    def _hidden_states2vec(self,hidden_states, start, end ,idx=0):
        '''
        args:
            hidden_states: list of hidden states
            start,end: 左闭右开区间
        '''
        if isinstance(self.layer,int):
            tensor = hidden_states[self.layer][idx][1+start:1+end].mean(dim=0).detach().cpu()
        
        elif isinstance(self.layer,(tuple,list)):
            start_layer, end_layer = self.layer[0],self.layer[1]
            tensor = torch.stack(hidden_states[start_layer:end_layer])[:,idx][:,1+start:1+end].mean(dim=(0,1)).detach().cpu()
        else:
            raise NotImplemented("layer must be int or tuple or list")

        if self.half is True:
            tensor = tensor.half()
        
        return tensor.numpy()


    @torch.no_grad()
    def get_vector(self,word):    
        self.load_bert()
        inputs = self.tokenizer(word,return_tensors='pt')
        inputs = {k:v.to(self.bert.device) for k,v in inputs.items()}
        outputs = self.bert(**inputs)
        vec = self._hidden_states2vec(outputs["hidden_states"],0,len(word))
        return vec

    @torch.no_grad()
    def get_sent_vector(self,words):
        '''
        words:
            list of words
            已经被分好词的句子
        '''
        sent = ""
        loc = []
        for word in words:
            loc.append((len(sent),len(sent)+len(word)))
            sent += word
        
        #white space tokenization
        sent = " ".join(sent) 

        self.load_bert()
        inputs = self.tokenizer(sent,return_tensors='pt')
        inputs = {k:v.to(self.bert.device) for k,v in inputs.items()}
        outputs = self.bert(**inputs)

        vectors = []
        for start,end in loc:
            vec = self._hidden_states2vec(outputs["hidden_states"],start,end)
            vectors.append(vec)

        return vectors


    @torch.no_grad()
    def get_tokens_vector(self,text : str):
        '''
        text:一段文本，返回每一个字的向量
        '''
        self.load_bert()
        inputs = self.tokenizer(text,return_tensors='pt')
        inputs = {k:v.to(self.bert.device) for k,v in inputs.items()}
        outputs = self.bert(**inputs)

        if isinstance(self.layer,int):
            tensor = outputs["hidden_states"][self.layer][0][1:-1].detach().cpu().numpy()

        elif isinstance(self.layer, (tuple,list)):
            start_layer, end_layer = self.layer[0],self.layer[1]

            tensor = torch.stack(outputs["hidden_states"][start_layer:end_layer])[:,0][:,1:-1].mean(dim=0).detach().cpu().numpy()

        return tensor

    def save_model(self, path : os.PathLike = "bert2vec.npy"):
        self.wv.save(path)


    def most_similar(self,text,topn = 5):
        if text not in self.wv:
            vec = self.get_vector(text)
        else:
            vec = self.wv[text]
        return self.wv.most_similar(vec,topn=topn)