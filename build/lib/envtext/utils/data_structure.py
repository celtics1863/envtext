from collections import defaultdict
from .txt_ops import txt_generator
import random #for random.random()

class Tree:
    def __init__(self):
        self.root = 0
        self.nodes = []
        self.paths = defaultdict(set)
        self.node2index = {}
        self.index2node = {}
        pass

    def insert(self,A,B):
        self.add_node(A)
        self.add_node(B)

        
        if B not in self.node2index:
            self.node2index[B] = len(self.node2index)
        

    def add_node(self,A):
        if A not in self.node2index:
            self.index2node[len(self.node2index)] = A
            self.node2index[A] = len(self.node2index)
            self.nodes.append(A)

    def add_path(self,A,B):
        pass


    def istree(self):
        pass



def Ngrams(path,N=4,min_cnt = 16,sep='///'):
    ngrams = defaultdict(int)
    if isinstance(path, str):
        bar = tqdm(txt_generator(path),f'读取{N}grams')
    else:
        bar = tqdm(path,f'读取{N}grams')
    # for words in tqdm(txt_generator(path),f'读取{N}grams'):
    for words in bar:
        for idx in range(len(words)-N):
            ngrams[sep.join(words[idx:idx+N+1])] += 1
            
    ngrams_list = [defaultdict(int) for i in range(4)]
    min_count = 16
    total = 0
    for k,v in tqdm(ngrams.items(),'计算ngrams：'):
        if v < min_count:
            continue
        total += v
        words = k.split(sep)
        for i,j in product(range(len(words)),range(len(words))):
            if i <= j and j-i < 4:
                ngrams_list[j-i][sep.join(words[i:j+1])] += 1
    
    return ngrams,ngrams_list


class SimpleTrie:
    def __init__(self,tokenize=False):
        self.dic = {}
        self.end = True
        self.tokenize=tokenize
        if self.tokenize == "jieba":
            import jieba
            self.tokenizer = lambda x:jieba.lcut(x)
        
        elif self.tokenize == "jiepa":
            import jiepa
            self.tokenizer = lambda x:jiepa.lcut(x)
        
        elif self.tokenize == "lac":
            from LAC import LAC
            self.lac = LAC()
            self.tokenizer = lambda x: self.lac.run(x)[0]
        
        elif self.tokenize == "en":
            self.tokenizer = lambda x: x.split(" ") #whitespace
            
        else:
            self.tokenizer = lambda x:list(x)
            
    def add(self,words):
        '''
        words: List[str] or str
        '''
        words = self._preprocess(words)
        dic = self.dic
        for c in words:
            if c not in dic:
                dic[c] = {}
            dic = dic[c]
        dic[self.end] = c
    
    
    def _preprocess(self,words):
        
        '''
        预处理
        '''
        if isinstance(words,list):
            return words
        
        elif isinstance(words,str):
            return self.tokenizer(words)
        else:
            NotImplemented
    
    def __contains__(self,words):
        '''
        words: List[str] or str
        '''
        
        words = self._preprocess(words)
        
        dic = self.dic
        
        for c in words:
            if c not in dic:
                return False
            
            dic = dic[c]
        
        if self.end in dic and dic[self.end] == c:
            return True
        else:
            return False
    
    def tokenize(self,sent):
        result = []
        start, end =0,1
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
    

if __name__ == "__main__":
    tree = SimpleTrie(tokenize="jieba")

