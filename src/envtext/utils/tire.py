import jieba
import os
from typing import *
from .json_ops import *

class SimpleTier:
    def __init__(self, tier_path : Union[os.PathLike, None ]  = None):
        if os.path.exists(tier_path):
            self.dic = read_json(tier_path)
        else:
            self.dic = {}

        self.end = True
    
    def add_word(self,words):
        if isinstance(words,str):
            words = jieba.cut(words)
            
        dic = self.dic
        for c in words:
            if c not in dic:
                dic[c] = {}
            dic = dic[c]
        dic[self.end] = c
    
    def tokenize(self,sent , jieba = True):
        if isinstance(sent, str) and jieba:
            sent = jieba.lcut(sent)
            
        result = []
        start, end = 0,1
        for i,word in enumerate(sent):
            dic = self.dic
            if i == end:
                result.append(sent[start:end])
                start, end = i,i+1
            
            for j,c in enumerate(sent[i:]):
                if c in dic:
                    dic = dic[c]
                    if self.end in dic:
                        if i + j + 1 > end:
                            end = i + j + 1
                else:
                    break
        
        result.append(sent[start: end])
        return result
    
    def keyword(self,sent, jieba = True):
        if isinstance(sent, str) and jieba:
            sent = jieba.lcut(sent)

        result = []
        start, end = 0,1
        j = 0
        for i,word in enumerate(sent):
            dic = self.dic
            if i == end and j > 0:
                result.append(sent[start:end])
                start, end = i,i+1
            
            for j,c in enumerate(sent[i:]):
                if c in dic:
                    dic = dic[c]
                    if self.end in dic:
                        if i + j + 1 > end:
                            end = i + j + 1
                else:
                    break
        
        return result

