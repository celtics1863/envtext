import os

def load_jieba_vocab(path = None):
    '''
    导入结巴词表
    '''
    if not path:
        import jieba
        path = jieba.get_dict_file()
    
    if os.path.exists(path):
        lines = []
        for idx, line in enumerate(open(path,"r",encoding = "utf-8")):
            lines.append(line.strip().split(" ")[:2])

    vocab = {word:freq for word,freq in lines}
    return vocab


def write_jieba_vocab(path,dic):
    '''
    path: 导出词表的路径
    dic ：dic 词典 
        {
            word1: count1,
            word2: count2
        }
    '''
    

    _dir = os.path.basedir(path)
    
    if _dir and not os.path.exists(_dir):
        os.makedirs(_dir)
        
    with open(path,"w",encoding="utf-8") as f:
        for k,v in dic.items():
            f.write(f"{k}\t{v}\n")
            
    