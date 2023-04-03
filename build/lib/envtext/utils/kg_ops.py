from collections import defaultdict
import os
from .txt_ops import txt_generator

def get_triples_with_single_relation_from_model_result(model,A,R,return_lines = False):
    pred_lines = []
    bad_preds = []

    triples = defaultdict(list)

    for k,v in model.result.items():
        line = k
        if len(v) > 0:
            for kk,vv in v.items():
                if kk.startswith('label'):
                    entity = kk.replace('label','entity')
                    line = line.replace(v[kk],f"【{v[entity]}{vv}】")
                    if v[entity] == A:
                        triples[k].append((vv,R,None))
                    elif k not in triples:
                        continue
                    else:
                        if len(triples[k]) and triples[k][-1][-1] is None:
                            triples[k][-1] = (triples[k][-1][0],R,vv)
                        else:
                            triples[k].append((triples[k][-1][0],R,vv))

            pred_lines.append(line)
        else:
            bad_preds.append(line)  
    
    if return_lines:
        return triples,(pred_lines,bad_lines)
    else:
        return triples


def align_triples_format(triples,callback = None):
    '''
    Args:
        triples:
            List[Dict[Dict[R,B]]]]
            List[Dict[List[R,B]]]]
            List[Dict[R,B]]]
            Dict[List[A,R,B]]
            Dict[List[R,B]]
            Dict[Dict[Set(B)]]
            Dict[Dict[List(B)]]
            Dict[Dict[B]]
            
        callback `Funtional(A,R,B)`: default None
            function on each triple
            default return [[A,R,B]]
    '''
    if len(triples) == 0:
        return []
    
    if callable(callback):
        g = callback
    else:
        g = lambda a,r,b : [[a,r,b]]
    
    texts = []
    
    if isinstance(triples,list):
        #List[Tuple] or List[List]
        if isinstance(triples[0],tuple) or isinstance(triples[0],list):
            for triple in triples:
                a,r,b = triple[0],triple[1],triple[2]
                texts += g(a,r,b)
        #List[Dict]
        elif isinstance(triples[0],dict):
            for triple in triples:
                for k,v in triple.items():
                    #List[Dict[Dict]]
                    if isinstance(v,dict):
                        for r,b in v.items():
                            if isinstance(b,str):
                                texts += g(k,r,b)
                            else:
                                assert 0,f"错误的格式：支持List[Dict[Dict[str]]]]，现在是List[Dict[Dict[{type(b)}]]]]"
                    elif isinstance(v,list):
                        if len(v) == 0:
                            continue
                        
                        if isinstance(v[0],list):
                            try:
                                for r,b in zip(*v):
                                    texts += g(k,r,b)
                            except:
                                assert 0,f"错误的格式：支持List[Dict[List[R,B]]]]，现在是List[Dict[List[{type(b)}]]]]"
                        else:
                            try:
                                r,b = v[0],v[1]
                                texts += g(k,r,b)
                            except:
                                assert 0,f"错误的格式：支持List[Dict[R,B]]]，现在是List[Dict[{type(b)}]]]"
                                
        elif isinstance(triples[0],str):
            a,r,b = triples[0],triples[1],triples[2]
            texts += g(a,r,b)
        else:
            raise NotImplemented
            
    elif isinstance(triples,dict):
        for k,v in triples.items():
            if isinstance(v,list) or isinstance(v,tuple):
                if len(v) == 0:
                    continue
                #Dict[List[R,B]] or #Dict[List[A,R,B]]
                if isinstance(v[0],tuple) or isinstance(v[0],list):
                    for triple in v:
                        if len(triple) == 3:
                            a,r,b = triple[0],triple[1],triple[2]
                            texts += g(a,r,b)
                        elif len(triple) == 2:
                            r,b = triple[1],triple[2]
                            texts += g(k,r,b)
                        else:
                            raise NotImplemented
                #Dict[List[R,B]]
                elif isinstance(v[0],str):
                    if len(v) == 3:
                        a,r,b = v[0],v[1],v[2]
                        texts += g(a,r,b)
                    elif len(v) == 2:
                        r,b = v[1],v[2]
                        texts += g(k,r,b)
                    else:
                        raise NotImplemented
            
            elif isinstance(v,dict):
                for kk,vv in v.items():
                    if isinstance(vv,set) or isinstance(vv,list):
                        for b in vv:
                            texts += g(k,kk,b)
                    else:
                        texts += g(k,kk,vv)
                        raise NotImplemented
            else:
                raise NotImplemented
            
    else:
        raise NotImplemented
    
    return texts
    

def read_triples(path):
    f = open(path,'r',encoding = 'utf-8')
    triples = [line.strip().split('\t') for line in f] 
    triples = [t for t in triples if len(t) == 3]
    return triples
    
def align_triples(triplesA,triplesB):
    '''
    Args:
        triplesA `triples` or `os.PathLike`: 
            List[Dict[Dict[R,B]]]]
            List[Dict[List[R,B]]]]
            List[Dict[R,B]]]
            Dict[List[A,R,B]]
            Dict[List[R,B]]
            Dict[Dict[Set(B)]]
            Dict[Dict[List(B)]]
            Dict[Dict[B]]
            
        triplesB `triples` or `os.PathLike`: 
            List[Dict[Dict[R,B]]]]
            List[Dict[List[R,B]]]]
            List[Dict[R,B]]]
            Dict[List[A,R,B]]
            Dict[List[R,B]]
            Dict[Dict[Set(B)]]
            Dict[Dict[List(B)]]
            Dict[Dict[B]]
            
    '''
    
    if isinstance(triplesA,str):
        triplesA = read_triples(triplesA)
    else:
        triplesA = align_triples_format(triplesA)
    
    if isinstance(triplesB,str):
        triplesB = read_triples(triplesB)
    else:
        triplesB = align_triples_format(triplesB)
        
    hashmap = defaultdict(dict)
    for a,r,b in triplesA:
        if a not in hashmap:
            hashmap[a] = defaultdict(set)
        hashmap[a][r].add(b)
    
    for a,r,b in triplesB:
        if a not in hashmap:
            hashmap[a] = defaultdict(set)
        hashmap[a][r].add(b)
    
    triples = align_triples_format(hashmap)
    
    return triples


def write_triples2txt(path,triples,mode = 'a'):
    '''
        path `str`:
            文件路径
            
        triples:
            支持的格式：
                List[Dict[Dict[R,B]]]]
                List[Dict[List[R,B]]]]
                List[Dict[R,B]]]
                Dict[List[A,R,B]]
                Dict[List[R,B]]
                Dict[List[R,B]]
                Dict[Dict[Set(B)]]
                Dict[Dict[List(B)]]
       mode `str`:
           文件的读入格式，默认'a'
       
    '''
    f = open(path,mode,encoding='utf-8')
    g = lambda a,r,b :f'{a}\t{r}\t{b}\n'
    
    if len(triples) == 0:
        return
    
    texts = align_triples_format(triples,g)
    
    for line in texts:
        f.write(line)
    
    f.close()
    
            
            
            
            
def read_triples(path,sep = "\t"):
    import re
    wash = lambda x : re.sub("\s","",x)

    triples = []
    for line in txt_generator(path):
        try:
            a,r,b = line.strip().split(sep)
            a = wash(a)
            r = wash(r)
            b = wash(b)
            if a and r and b:
                triples.append((a,r,b))
    
        except:
            continue

    return triples



            

    
            
            
            
            
            
            