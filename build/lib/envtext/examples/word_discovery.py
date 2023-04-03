import tqdm
from ..utils.data_structure import Ngrams,SimpleTrie
from ..utils.txt_ops import txt_generator,write_txts
from collections import defaultdict
from ..files import Config

import jieba

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path')
parser.add_argument('-N','--N',type=int,default=4)
parser.add_argument('-m','--min_count',type=int,default = '')


args = parser.parse_args()


class Discovery:
    def __init__(self,file_path,N = 4,min_count = 16,sep='///'):
        self.file_path = file_path
        self.sep = sep
        self.N = N
        self.min_count = min_count

        jieba.load_userdict(Config.env_vocab)
        self.tokenizer = lambda x: jieba.lcut(x)
        
    def run(self):
        if isinstance(self.file_path, str):
            lines = []
            for line in tqdm(txt_generator(path),"读取文件"):
                words = self.tokenizer(line.strip())
                lines.append(words)
        
        WD = WordDiscovery(lines,self.file_path,self.min_count,self.sep)

        cands = WD.run()
        print(cands)
        



    
class WordDiscovery:
    def __init__(self,file_path,N = 4,min_count = 16,sep='///'):
        self.file_path = file_path
        self.sep = sep
        self.N = N
        self.min_count = 16
    
    #点互信息过滤
    def pmi_filter(self,ngrams):
        output_ngrams = set()
        for i in range(3,0,-1):
            for w,v in ngrams_list[i].items():
                words = w.split(self.sep)
                pmi = min(
                    total * v / (ngrams_list[j][self.sep.join(words[:j+1])] * ngrams_list[i-j-1][self.sep.join(words[j+1:])])
                    for j in range(i)
                )
                if math.log(pmi) >= min_pmi[i]:
                        output_ngrams.add(w)
        return output_ngrams
    
    #互信息过滤
    def mi_filter(self,candidates, ngrams, order):
        """通过与ngrams对比，排除可能出来的不牢固的词汇(回溯)
        """
        result = {}
        for i, j in candidates.items():
            i_order = len(i.split(self.sep))
            if i_order == 1 and len(i) <= 5: # 5:
                continue
            if i_order < 3:
                result[i] = j
            elif i_order <= order and i in ngrams:
                result[i] = j
            elif i_order > order:
                flag = True
                words = i.split(self.sep)
                for k in range(len(words) + 1 - order):
                    if self.sep.join(words[k: k+order]) not in ngrams:
                        flag = False
                if flag:
                    result[i] = j
        return result
    
    # 频数过滤
    def freq_filter(self,vocab,min_count=16):
        candidates = {i: j for i, j in candidates.items() if j >= min_count}
        return candidates
    
    #规则过滤
    def reg_filter(self,vocab):
        def is_ok(words):
            for txt in words.split('///'):
                for p in [
                    r'[^\u4e00-\u9fa5]+',
                    r"[与是的了过]",
                    r'.*[和内副上从中下]',
                    r'[个在将和对等或].*'
                ]:
                    if re.search(p,txt) is not None:
                        return False
            return True
        candidates = {k:v for k,v in candidates.items() if is_ok(k)}
        return candidates
    
    def run():
        ngrams,ngrams_list = Ngrams(self.file_path,self.N,self.min_count)
        
        #pmi filter
        output_ngrams = self.pmi_filter(ngrams_list)
        
        trie = SimpleTrie()
        for w in tqdm(output_ngrams,'构建Trie树'):
            trie.add_word(w.split('///'))
        
        candidates = defaultdict(int)
        for words in tqdm(txt_generator(self.path),'使用最长片段再次分词'):
            for word in trie.tokenize(words):
                candidates['///'.join(word)] += 1
        
        #频数过滤
        candidates = self.freq_filter(candidates,self.min_count)
        #互信息过滤
        candidates = self.mi_filter(candidates,ngrams,self.N)
        #规则过滤
        candidates = self.reg_filter(candidates)
        return candidates
        
