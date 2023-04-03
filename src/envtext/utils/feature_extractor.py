import jieba
from LAC import LAC
from tqdm import tqdm
from .txt_ops import read_txt
from .json_ops import write_json
import os
from collections import Counter,defaultdict
import math
from itertools import combinations,product

def Entropy(cnt):
    if not cnt:
        return 0
    
    s = sum(cnt.values())
    
    if s == 0:
        return 0
    
    for v in cnt:
        cnt[v] = cnt[v] / s
        
    return - sum([v * math.log(v) for k,v in cnt.items()])

class EntityFeature:
    def __init__(self,cands,tokenizer = "jieba"):
        
        if isinstance(cands,str) and os.path.exists(cands):
            self.cands = set(read_txt(cands))
        else:
            self.cands = set(cands)
        
        if tokenizer == "jieba":
            # for cand in cands:
            #     jieba.add_word(cand,freq = 5000)
            self.tokenizer = lambda x:jieba.lcut(x)
        
        elif tokenizer == "lac":
            self.lac = LAC()
            self.tokenizer = lambda x: self.lac.run(x)[0] #only use tokens
            
        
        self.tf = Counter() # term frequnce
        self.df = Counter() #document frequence
        self.bigram = Counter()
        
        self.TotalDocuments = 0
            
        
    def collect_stats(self,texts):
        if isinstance(texts,str) and os.path.exists(texts):
            texts = read_txt(texts)
        
        texts = list(filter(lambda x:isinstance(x,str) ,texts))
        for text in tqdm(texts, desc = "正在计算统计信息。。。"):
            words = self.tokenizer(text)
            
            local_word_set = set()

            for i,j in combinations(range(len(words)),r = 2):
                
                k = "".join(words[i:j])
                
                if k in self.cands:
                    local_word_set.add(k)
                    self.tf[k] += 1
                
                if  i > 0 and (words[i-1] in self.cands and k in self.cands) :
                    self.bigram[(words[i-1],k)] += 1
            
            #更新文档频率
            for w in local_word_set:
                self.df[w] += 1
                
        self.TotalDocuments += len(texts)
        
        write_json("词频统计.json",self.tf)
        
        num_all_words = sum(self.tf.values())
        for k in self.tf:
            self.tf[k] = self.tf[k]/len(texts)
        
    
    def _tf(self):
        tf = Counter()
        for e in tqdm(self.cands, desc = "正在计算tf。。。" ):
            tf[e] = self.tf[e]
        
        return tf
    
    def tfidf(self,idf):
        tfidf = Counter()
        for e in tqdm(self.cands, desc = "正在计算tfidf。。。" ):
            tfidf[e] = self.tf[e] * idf[e]
        
        return tfidf
    
    def idf(self):
        idf = Counter()
        
        for e in tqdm(self.cands, desc = "正在计算idf。。。" ):
            idf[e] = math.log( self.TotalDocuments / ( 1 + self.df.get(e,0)))
            
        return idf
        
    def pmi(self):
        PMI = Counter()
        
        for e in tqdm(self.cands,desc="正在计算pmi。。。"):
            words = jieba.lcut(e)
            idx = 0
            c_sum = 0
            f = self.tf.get(e,1)
            pmis = []
            for i in range(1,len(words)):
                left = "".join(words[:i])
                right = "".join(words[i:])
                left = self.tf.get(left,1)
                right = self.tf.get(right,1)
                if left <= 0 or right <= 0:
                    pmi = 0
                else:
                    pmi =  f/left/right
                pmis.append(pmi)
            
            if len(pmis) > 0:
                PMI[e] =  math.log(min(pmis))
            else:
                PMI[e] = 0
        return PMI
    
    def cvalue(self):
        cvalue = Counter()
        
        parents = defaultdict(Counter)
        for e in tqdm(self.cands, desc="正在准备计算cvalue。。。"):
            words = jieba.lcut(e)
            for i,j in combinations(range(len(words)),r=2):
                local_word = "".join(words[i:j])
                if local_word in self.cands: #判断是否存在父短语
                    cnt = self.tf.get(local_word,0)
                    parents[local_word][e] += 1
            

        for e in tqdm(self.cands, desc="正在计算cvalue。。。"):
            f = self.tf.get(e,0)
            
            c_parents = parents.get(e,Counter())
            
            if len(c_parents) != 0:
                c_sum = sum(c_parents.values())
                f -= (c_sum / len(c_parents))
            
            weight = math.log(len(e))
            cvalue[e] = weight * f
            
        return cvalue
                      
                      
    def ncvalue(self,cvalue):
        ncvalue = Counter()
                      
        P5_words = set([k for k,v in cvalue.most_common(int(len(cvalue)*0.05))])
        N = len(P5_words)
                      
        Cu = defaultdict(Counter)
        
        for (a,b),v in tqdm(self.bigram.items(), desc="正在准备ncvalue所需的上下文词语计算。。。"):
            if a in P5_words:
                Cu[a][b] += 1
            
            if b in P5_words:
                Cu[b][a] += 1
            
                      
        for e in tqdm(self.cands, desc="正在计算ncvalue。。。"):
            
            nc_sum = 0
            for b,fub in Cu[e].items():
                nc_sum += fub * self.tf.get(b,0) / N
                      
            ncvalue[e] = 0.8 * cvalue[e] + 0.2 * nc_sum
        return ncvalue 
  
    def LRE(self):
        LE = Counter()
        RE = Counter()
                      
        left_distribution = defaultdict(Counter)
        right_distribution = defaultdict(Counter)
                      
        for (a,b),v in tqdm(self.bigram.items(), desc="正在准备LRE所需的上下文词语计算。。。"):
            if b in self.cands:
                left_distribution[b][a] += v 
            
            if a in self.cands:
                right_distribution[a][b] += v 
    
        for e in tqdm(self.cands, desc="正在计算LRE。。。"):
            LE[e] = Entropy(left_distribution[e])
            RE[e] = Entropy(right_distribution[e])
        
        return LE,RE
    
    def make_dataframe(self,path="实体数据库.csv",return_info = False):
        tf = self._tf()
        idf = self.idf()
        tfidf = self.tfidf(idf)
        pmi = self.pmi()
        cvalue = self.cvalue()
        ncvalue = self.ncvalue(cvalue)
        # LE,RE = self.LRE()
        
        data = []
        for k in self.cands:
            data.append(
                {
                    "word":k,
                    "tf":tf.get(k,0),
                    "idf":idf.get(k,0),
                    "tfidf":tfidf.get(k,0),
                    "cvalue":cvalue.get(k,0),
                    "ncvalue":ncvalue.get(k,0),
                    "pmi":pmi.get(k,0),
                    # "LE":LE.get(k,0),
                    # "RE":RE.get(k,0),
                }
            )
        import pandas as pd
        df = pd.DataFrame(data)
        df.to_csv("实体数据库.csv")
        
        if return_info:
            return tf,idf,tfidf,cvalue,ncvalue,pmi#,LE,RE