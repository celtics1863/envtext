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
                    batch_size : int = 8,
                    weights : str = "length",
                    half: bool = True,
                    word_tokenizer = "default"
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
        self.word_tokenizer = word_tokenizer

    def load_word_tokenizer(self):
        #导入 word_tokenizer
        if not callable(self.word_tokenizer):
            if self.word_tokenizer == "default":
                try:
                    import envpos
                    self.word_tokenizer = lambda x: envpos.cut(x)[0]
                except:
                    try:
                        import jiepa
                        self.word_tokenizer = lambda x: jiepa.lcut(x)
                    except:
                        import jieba
                        self.word_tokenizer = lambda x: jieba.lcut(x)
            else:
                import jieba
                self.word_tokenizer = lambda x: jieba.lcut(x)

    def load_weights(self):
        if self.weights == "length":
            # self.weights = 
            pass
        
    def add_word(self,text : str):
        if text in self.wv.key_to_index:
            return

        self.load_bert()
        vector = self.get_vector(text)
        self.wv.add_vector(text, vector)
        self.wv.resize_vectors()

    def add_sentence(self, text : str):
        '''
        text: 文本，将会分词后，取动态词向量的平均值
        '''
        if text in self.wv.key_to_index:
            return

        #导入分词器
        self.load_word_tokenizer()
        
        #分词
        words = self.word_tokenizer(text)
        
        #获得每一个词的向量
        vectors = self.get_sent_vector(words)

        #对每一个词的向量取平均
        vector = np.mean(vectors,axis=0)

        #添加句子
        self.wv.add_vector(text, vector)
        self.wv.resize_vectors()

    def _get_loc_of_words(self, words):
        locs = []
        lens = 0
        for word in words:
            locs.append((lens,lens +  len(word))) #左闭右开
            lens += len(word)
        return locs

    def add_sentences(self, texts : List[str]):
        '''
        texts: 一些句子，list[str] 或者 tuple[str]
        '''
        #排序，用以加速
        texts.sort(key= lambda x:len(x))

        #载入分词器
        self.load_word_tokenizer()
        list_of_words = [self.word_tokenizer(t) for t in tqdm(texts,desc="正在分词....")]
        list_of_locs = [self._get_loc_of_words(words) for words in list_of_words]
        list_of_texts = [" ".join(t[:510]) for t in texts]

        #载入bert模型
        self.load_bert()

        #分批进行推理
        for i in tqdm(range(0,len(list_of_texts),self.batch_size)):
            texts_batch = list_of_texts[i:i+self.batch_size]
            locs_batch = list_of_locs[i: i+self.batch_size]

            #准备输入
            inputs = self.tokenizer(texts_batch, return_tensors='pt',padding="longest")
            inputs = {k:v.to(self.bert.device) for k,v in inputs.items()}

            #推理
            outputs = self.bert(**inputs)  # Run model

            #准备输出，保存向量的数组
            vectors = []
            for idx,(text,loc) in enumerate(zip(texts_batch,locs_batch)):
                _vectors = [] #临时存储句子中每一个词的向量
                for start,end in loc:
                    vec = self._hidden_states2vec(outputs["hidden_states"],start,end,idx) #获得每一个词的向量
                    _vectors.append(vec)
                _vector = np.mean(_vectors,axis=0) #对词的向量取平均
                vectors.append(_vector) 
            
            vectors = np.stack(vectors)
            self.wv.add_vectors(texts_batch, vectors)

        #重置词向量大小
        self.wv.resize_vectors()


    def load_bert(self):
        #载入bert模型
        if self.bert is None:
            import torch
            if torch.cuda.is_available():
                self.bert = BertModel.from_pretrained(self.model_path, output_hidden_states=True).cuda()
            else:
                self.bert = BertModel.from_pretrained(self.model_path, output_hidden_states=True)

            self.tokenizer = BertTokenizerFast.from_pretrained(self.model_path)

    @torch.no_grad()
    def add_words(self,words : Union[List[str],Tuple[str]]):
        '''
        words: 一些单词，list[str] 或者 tuple[str]
        '''

        #导入bert莫小仙
        self.load_bert()

        #过滤文本
        words = list(filter(lambda x:  x not in self.wv.key_to_index and x, words))

        if len(words) == 0:
            print("全部词已经在词向量中")
            return

        #分批进行推理
        for i in tqdm(range(0,len(words),self.batch_size)):
            words_batch = words[i:i+self.batch_size]

            #准备输入
            inputs = self.tokenizer(words_batch, return_tensors='pt',padding="longest")
            inputs = {k:v.to(self.bert.device) for k,v in inputs.items()}

            #推理
            outputs = self.bert(**inputs)  # Run model

            #存储向量列表
            vectors = []
            for idx,w in enumerate(words_batch):
                vec = self._hidden_states2vec(outputs["hidden_states"], 0 , len(w) ,idx)
                vectors.append(vec)
            vector = np.stack(vector)

            #添加向量
            self.wv.add_vectors(words_batch, vector)

        #重置词表大小
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
            raise NotImplementedError()("layer must be int or tuple or list")

        if self.half is True:
            tensor = tensor.half()
        
        return tensor.numpy()


    @torch.no_grad()
    def get_vector(self,word : str):    
        '''
        获得单词的向量
        word: str
        '''
        self.load_bert()
        inputs = self.tokenizer(word,return_tensors='pt')
        inputs = {k:v.to(self.bert.device) for k,v in inputs.items()}
        outputs = self.bert(**inputs)
        vec = self._hidden_states2vec(outputs["hidden_states"],0,len(word))
        return vec

    @torch.no_grad()
    def get_sent_vector(self,words : List[str]):
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
        '''
        返回topn最相似的词
        '''
        if text not in self.wv:
            vec = self.get_vector(text)
        else:
            vec = self.wv[text]
        return self.wv.most_similar(vec,topn=topn)