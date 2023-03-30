from collections import defaultdict,Counter

class LoadDataset:
    @classmethod
    def load_dataset(cls,path,task,split =0.5,text = 'text',label='label'):
        pass
        
    @staticmethod
    def has_split(js):
        if isinstance(js,dict):
            return isinstance(js[list(js.keys())[0]],dict)
        elif isinstance(js,list):
            raise NotImplemented


    @staticmethod
    def generate_datasets(js,task,train,valid,test,text,label,sep = ' ' ,label_as_key = False):
#         task = _unify_task(task)
        if task == 'CLUENER':
            return generate_cluener_datasets(js,train,valid,test,text,label)
        
        if not label_as_key:
            if task == 'CLS':
                return generate_cls_datasets(js,train,valid,test,text,label)
            elif task == 'REG':
                return generate_reg_datasets(js,train,valid,test,text,label)
            elif task == 'MCLS':
                return generate_mcls_datasets(js,train,valid,test,text,label)
            elif task == 'KW':
                return generate_keyword_datasets(js,train,valid,test,text,label,sep)
            elif task == 'NER':
                return generate_keyword_datasets(js,train,valid,test,text,label)
            else:
                raise NotImplemented
        else:
            if task == 'CLS':
                return generate_cls_datasets_label_as_key(js,train,valid,test,text,label)
            elif task == 'REG':
                return generate_reg_datasets_label_as_key(js,train,valid,test,text,label)
            elif task == 'NER':
                return generate_ner_datasets_label_as_key(js,train,valid,test,text,label)
            else:
                raise NotImplemented
                

#将标签转换为数字label
def convert_label2onehot(ids,labels):
    onehot = [0.]*len(labels)
    if isinstance(ids,list):
        for id in ids:
            if isinstance(id,str):
                onehot[labels.index(id)] = 1.
            else:
                onehot[id] = 1.
    elif isinstance(ids,int):
        onehot[ids] = 1.
    elif isinstance(ids,str):
        onehot[labels.index(ids)] = 1.
    else:
        raise NotImplemented
    return onehot

def generate_reg_datasets(js,train,valid,test,text,label):
    #数据集
    datasets = { 'train':defaultdict(list),'valid':defaultdict(list),'test':defaultdict(list)}
    mapping = { train:'train', valid:'valid', test:'test'}

    #统计文本长度
    AvgL = []
    for idx,(k,v) in enumerate(js.items()):
        AvgL += list(map(len,v[text]))
    AvgL = sum(AvgL)/len(AvgL)

    #统计分位数
    import numpy as np
    all_labels = []
    for idx,(k,v) in enumerate(js.items()):
        all_labels += list(map(float,v[label]))
    
    min_value = min(all_labels)
    p25 = np.percentile(all_labels,25)
    p50 = np.percentile(all_labels,50)
    p75 = np.percentile(all_labels,75)
    max_value = max(all_labels)
    
    
    #将标签转换为数字label
    for k,v in js.items():
        if len(v) == 0 or (not isinstance(v,dict)) or (text not in v.keys() or label not in v.keys()):
            print(f'{k}没有导入，可能是因为数据集格式不是dict，或没有{text} 、{label}作为key')
            continue
        datasets[mapping[k]]['text'] = v[text]
        datasets[mapping[k]]['raw_label'] = v[label]
        datasets[mapping[k]]['label'] = [float(l) for l in v[label]]

    #数据集参数
    config = {
        'Avg. Length':AvgL,
        'min':min_value,
        'percentile 25':p25,
        'percentile 50':p50,
        'percentile 75':p75,
        'max':max_value,
    }

    return datasets,config

def generate_reg_datasets_label_as_key(js,train,valid,test,text,label):
    #数据集
    datasets = {'train':defaultdict(list),'valid':defaultdict(list),'test':defaultdict(list)}
    mapping = {train:'train', valid:'valid', test:'test'}

    #将标签转换为数字label
    for idx,(k,v) in enumerate(js.items()):
        for kk,vv in v.items():
            datasets[mapping[k]]['text'].append(vv)
            datasets[mapping[k]]['raw_label'].append([kk] * len(vv))
            datasets[mapping[k]]['label'].append([float(kk)] * len(vv))
    #数据集参数
    config = {}

    return datasets,config
  
def generate_cls_datasets(js,train,valid,test,text,label):
    #数据集
    datasets = { 'train':defaultdict(list),'valid':defaultdict(list),'test':defaultdict(list)}
    mapping = { train:'train', valid:'valid', test:'test'}

    #计数器，统计每类数量
    counter = Counter()
    for idx,(k,v) in enumerate(js.items()):
        counter.update(v[label])

    #统计文本长度
    AvgL = []
    for idx,(k,v) in enumerate(js.items()):
        AvgL += list(map(len,v[text]))
    AvgL = sum(AvgL)/len(AvgL)
        
    #整理标签种类
    labels = list(counter.keys())

    #将标签转换为数字label
    for k,v in js.items():
        if len(v) == 0 or (not isinstance(v,dict)) or (text not in v.keys() or label not in v.keys()):
            print(f'{k}没有导入，可能是因为数据集格式不是dict，或没有{text} 、{label}作为key')
            continue
        datasets[mapping[k]]['text'] = v[text]
        datasets[mapping[k]]['raw_abel'] = [labels.index(l) for l in v[label]]
        datasets[mapping[k]]['label'] = [convert_label2onehot(l,labels) for l in v[label]]

    #数据集参数
    config = {'labels':labels,
           'label2id':{l:i for i,l in enumerate(labels)},
           'id2label':{i:l for i,l in enumerate(labels)},
           'counter': counter,
           'Avg. Length': AvgL
           }

    return datasets,config

def generate_cls_datasets_label_as_key(js,train,valid,test,text,label):
    #数据集
    datasets = {'train':defaultdict(list),'valid':defaultdict(list),'test':defaultdict(list)}
    mapping = {train:'train', valid:'valid', test:'test'}

    #整理标签种类
    labels = set()
    for k,v in js.items():
        labels.update(set(v.keys()))
    labels = list(labels)

    #计数器，统计类别
    counter = Counter()

    #将标签转换为数字label
    for idx,(k,v) in enumerate(js.items()):
        for kk,vv in v.items():
            datasets[mapping[k]]['text'] += vv
            datasets[mapping[k]]['raw_label'] += [labels.index(kk)] * len(vv)
            datasets[mapping[k]]['label'] += [convert_label2onehot(kk,labels)] *len(vv)
            counter.update([kk]*len(vv))

    #数据集参数
    config = {'labels':labels,
           'label2id':{l:i for l,i in enumerate(labels)},
           'id2label':{i:l for l,i in enumerate(labels)},
           'counter': counter
           }

    return datasets,config


def generate_mcls_datasets(js,train,valid,test,text,label):
    #数据集
    datasets = { 'train':defaultdict(list),'valid':defaultdict(list),'test':defaultdict(list)}
    mapping = { train:'train', valid:'valid', test:'test'}

    #计数器，统计每类数量
    counter = Counter()
    for idx,(k,v) in enumerate(js.items()):
        for vv in v[label]:
            counter.update(vv)

    #统计文本长度
    AvgL = []
    for idx,(k,v) in enumerate(js.items()):
        AvgL += list(map(len,v[text]))
    AvgL = sum(AvgL)/len(AvgL)
    
    #计数器，统计类别数量分布
    hist = Counter()
    for idx,(k,v) in enumerate(js.items()):
        hist.update(map(len,v[label]))
            
    #整理标签种类
    labels = list(counter.keys())
    
    for k,v in js.items():
        if len(v) == 0 or (not isinstance(v,dict)) or (text not in v.keys() or label not in v.keys()):
            print(f'{k}没有导入，可能是因为数据集格式不是dict，或没有{text} 、{label}作为key')
            continue
        datasets[mapping[k]]['text'] = v[text]
        datasets[mapping[k]]['raw_label'] = [[labels.index(l)for l in vv] for vv in v[label]]
#         datasets[mapping[k]]['id'] = [convert_label2onehot([labels.index(l) for l in vv]) for vv in v[label]]
        datasets[mapping[k]]['label'] = [convert_label2onehot(vv,labels) for vv in v[label]]

    #数据集参数
    config = {'labels':labels,
           'label2id':{l:i for l,i in enumerate(labels)},
           'id2label':{i:l for l,i in enumerate(labels)},
           'counter': counter,
           'hist':hist,
           'Avg. Length':AvgL
           }

    return datasets,config

def generate_keyword_datasets(js,train,valid,test,text,label,sep):
    #数据集
    datasets = { 'train':defaultdict(list),'valid':defaultdict(list),'test':defaultdict(list)}
    mapping = { train:'train', valid:'valid', test:'test'}
    
    #统计文本长度
    AvgL = []
    for idx,(k,v) in enumerate(js.items()):
        AvgL += list(map(len,v[text]))
    AvgL = sum(AvgL)/len(AvgL)
    
    #计数器，统计每类数量
    counter = Counter()
    for idx,(k,v) in enumerate(js.items()):
        for vv in v[label]:
            if isinstance(vv,list):
                counter.update(vv)
            else:
                counter.update(vv.split(sep))
        
    hist = Counter()
    for idx,(k,v) in enumerate(js.items()):
        hist.update(map(lambda x: len(x) if isinstance(x,list) else len(x.split(sep)) ,v[label]))
    # B: 2
    # I: 1
    # O: 0
    def align_keyword_label(text,words):
        text = text.replace(' ','')
        label = [0]*len(text)
        #复杂度 len(words)*len(word)*len(text)
        for word in words.split(sep):
            p_word = 0
            p_text = 0
            while p_text < len(text):
                #已经标注的不要再标
                if label[p_text] >= 1:
                    p_text += 1
                    continue

                if text[p_text] == word[p_word]:
                    while p_text < len(text) and text[p_text] == word[p_word]:
                        p_word += 1
                        p_text += 1
                        if p_word == len(word):
                            label[p_text-p_word+1:p_text] = [1]*(p_word-1)
                            label[p_text-p_word] = 2
                            break
                    else:
                        p_text = p_text - p_word
                    p_word = 0
                p_text += 1
                            
        if len(label) > 510:
            label = label[:510]
        else:
            label = label + [-100] * (510-len(label))
        return label
                    
        
    
    #将标签转换为数字label
    for k,v in js.items():
        if len(v) == 0 or (not isinstance(v,dict)) or (text not in v.keys() or label not in v.keys()):
            print(f'{k}没有导入，可能是因为数据集格式不是dict，或没有{text} 、{label}作为key')
            continue

        for t,l in zip(v[text],v[label]):
            datasets[mapping[k]]['text'].append(' '.join(t)) #white space tokenize
            datasets[mapping[k]]['raw_label'].append(l)
            datasets[mapping[k]]['label'].append(align_keyword_label(t,l))

    #数据集参数
    config = {
           'entities':['keyword'],
           'labels':['O','B','I'],
           'label2id':{'O':0,'B':1,'I':2},
           'id2label':{0:'O',1:'B',2:'I'},
           'counter':counter,
           'hist': hist,
           'Avg. Length':AvgL
           }

    return datasets,config

def generate_cluener_datasets(js,train,valid,test,text,label):
    #数据集
    datasets = { 'train':defaultdict(list),'valid':defaultdict(list),'test':defaultdict(list)}
    mapping = { train:'train', valid:'valid', test:'test'}
    
    #统计文本长度
    AvgL = []
    for idx,(k,v) in enumerate(js.items()):
        AvgL += list(map(len,v[text]))
    AvgL = sum(AvgL)/len(AvgL)
    
    #计数器，统计每类数量
    counter = Counter()
    for idx,(k,v) in enumerate(js.items()):
        for vv in v[label]:
            counter.update(vv.keys())
        
    hist = Counter()
    for idx,(k,v) in enumerate(js.items()):
        hist.update(map(lambda x: sum(len(v) for k,v in x.items()),v[label]))

    
    labels = list(counter.keys())
    label2id = {'O':0}
    id2label = {0:'0'}
    for idx,l in enumerate(labels):
        label2id[f'B-{l}'] = idx*2 + 1
        label2id[f'I-{l}'] = idx*2 + 2
        id2label[idx*2 + 1] = f'B-{l}'
        id2label[idx*2 + 2] = f'I-{l}'

    # B: 2
    # I: 1
    # O: 0
    def align_keyword_label(text,annos):
        label = [0]*len(text)
        #复杂度 len(words)*len(word)*len(text)
        for k,v in annos.items():
            for kk,vv in v.items():
                for vvv in vv:
                    label[vvv[0]] = label2id[f'B-{k}']
                    label[vvv[0]+1:vvv[1]] = [label2id[f'I-{k}']] * (vvv[1] - vvv[0])
                            
        if len(label) > 510:
            label = label[:510]
        else:
            label = label + [-100] * (510-len(label))
        return label
                    
        
    
    #将标签转换为数字label
    for k,v in js.items():
        if len(v) == 0 or (not isinstance(v,dict)) or (text not in v.keys() or label not in v.keys()):
            print(f'{k}没有导入，可能是因为数据集格式不是dict，或没有{text} 、{label}作为key')
            continue

        for t,l in zip(v[text],v[label]):
            datasets[mapping[k]]['text'].append(' '.join(t)) #white space tokenize
            datasets[mapping[k]]['raw_label'].append(l)
            datasets[mapping[k]]['label'].append(align_keyword_label(t,l))

    #数据集参数
    config = {
           'labels':list(label2id.keys()),
           'entities':labels,
           'num_labels':len(label2id),
           'num_entities':len(labels),
           'label2id':label2id,
           'id2label':id2label,
           'counter':counter,
           'hist': hist,
           'Avg. Length':AvgL
           }

    return datasets,config

def generate_ner_datasets_label_as_key(js,train,valid,test,text,label):
    #数据集
    datasets = { 'train':defaultdict(list),'valid':defaultdict(list),'test':defaultdict(list)}
    mapping = { train:'train', valid:'valid', test:'test'}
    
    #统计文本长度
    AvgL = []
    for idx,(k,v) in enumerate(js.items()):
        AvgL += list(map(len,v[text]))
    AvgL = sum(AvgL)/len(AvgL)
    
    #计数器，统计每类数量
    counter = Counter()
    for idx,(k,v) in enumerate(js.items()):
        for vv in v[label]:
            counter.update(vv.keys())
        
    hist = Counter()
    for idx,(k,v) in enumerate(js.items()):
        hist.update(map(lambda x: sum(len(v) for k,v in x.items()),v[label]))

    
    labels = list(counter.keys())
    label2id = {'O':0}
    id2label = {0:'0'}
    for idx,l in enumerate(labels):
        label2id[f'B-{l}'] = idx*2 + 1
        label2id[f'I-{l}'] = idx*2 + 2
        id2label[idx*2 + 1] = f'B-{l}'
        id2label[idx*2 + 2] = f'I-{l}'

    # B: 2
    # I: 1
    # O: 0
    def align_keyword_label(text,annos):
        label = [0]*len(text)
        #复杂度 len(words)*len(word)*len(text)
        for k,v in annos.items():
            for vv in v:
                label[vv[0]] = label2id[f'B-{k}']
                label[vv[0]+1:vv[1]] = [label2id[f'I-{k}']] * (vv[1] - vv[0])
                            
        if len(label) > 510:
            label = label[:510]
        else:
            label = label + [-100] * (510-len(label))
        return label
                    

    
    #将标签转换为数字label
    for k,v in js.items():
        if len(v) == 0 or (not isinstance(v,dict)) or (text not in v.keys() or label not in v.keys()):
            print(f'{k}没有导入，可能是因为数据集格式不是dict，或没有{text} 、{label}作为key')
            continue

        for t,l in zip(v[text],v[label]):
            datasets[mapping[k]]['text'].append(' '.join(t)) #white space tokenize
            datasets[mapping[k]]['raw_label'].append(l)
            datasets[mapping[k]]['label'].append(align_keyword_label(t,l))

    #数据集参数
    config = {
           'labels':list(label2id.keys()),
           'entities':labels,
           'num_labels':len(label2id),
           'num_entities':len(labels),
           'label2id':label2id,
           'id2label':id2label,
           'counter':counter,
           'hist': hist,
           'Avg. Length':AvgL
           }

    return datasets,config


def generate_ner_datasets(js,train,valid,test,text,label):
    pass