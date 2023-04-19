from collections import defaultdict,Counter
from .dataset_utils import *

class LoadDataset:
    @classmethod
    def load_dataset(cls,path,task,split =0.5,text = 'text',label='label'):
        pass
        
    @staticmethod
    def has_split(js):
        if isinstance(js,dict):
            return isinstance(js[list(js.keys())[0]],dict)
        elif isinstance(js,list):
            raise NotImplementedError()


    @staticmethod
    def generate_datasets(js,**kwargs):
#         task = _unify_task(task)
        task = kwargs['task']
    
        if task == 'CLUENER':
            return generate_cluener_datasets(js,**kwargs)
        
        print(kwargs)

        if kwargs['label_inline']:
            if task == 'NER':
                return generate_ner_datasets_inline(js,**kwargs)
            elif task == "NestedNER":
                return generate_nested_ner_datasets_inline(js,**kwargs)
            elif task == "DP":
                return generate_dp_datasets_inline(js,**kwargs)
            elif task == "Triple":
                return generate_triple_datasets_inline(js,**kwargs)
            elif task == "Relation":
                return generate_relation_datasets_inline(js,**kwargs)


        elif not kwargs['label_as_key']:
            if task == 'CLS':
                return generate_cls_datasets(js,**kwargs)
            elif task == 'REG':
                return generate_reg_datasets(js,**kwargs)
            elif task == 'MCLS':
                return generate_mcls_datasets(js,**kwargs)
            elif task == 'KW':
                return generate_keyword_datasets(js,**kwargs)
            elif task == 'NER':
                return generate_ner_datasets(js,**kwargs)
            else:
                raise NotImplementedError()
        else:
            if task == 'CLS':
                return generate_cls_datasets_label_as_key(js,**kwargs)
            elif task == 'REG':
                return generate_reg_datasets_label_as_key(js,**kwargs)
            elif task == 'NER':
                return generate_ner_datasets_label_as_key(js,**kwargs)
            else:
                raise NotImplementedError()
                

def generate_reg_datasets(js,train = "train",valid = "valid",test = "test",text = "text",label = "label",**kwargs):
    #数据集,
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

def generate_reg_datasets_label_as_key(js,train="train",valid="valid",test="test",text="text",label="label",**kwargs):
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
  
def generate_cls_datasets(js,train="train",valid="valid",test="test",text="text",label="label",**kwargs):
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
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for i,l in enumerate(labels)}


    #将标签转换为数字label
    for k,v in js.items():
        if len(v) == 0 or (not isinstance(v,dict)) or (text not in v.keys() or label not in v.keys()):
            print(f'{k}没有导入，可能是因为数据集格式不是dict，或没有{text} 、{label}作为key')
            continue
        datasets[mapping[k]]['text'] = v[text]
        datasets[mapping[k]]['raw_label'] = [labels.index(l) for l in v[label]]
        datasets[mapping[k]]['label'] = [convert_label2onehot(l,labels) for l in v[label]]

    #数据集参数
    config = {'labels':labels,
           'label2id':label2id,
           'id2label':id2label,
           'counter': counter,
           'Avg. Length': AvgL
           }

    return datasets,config

def generate_cls_datasets_label_as_key(js,train="train",valid="valid",test="test",text="text",label="label",**kwargs):
    #数据集
    datasets = {'train':defaultdict(list),'valid':defaultdict(list),'test':defaultdict(list)}
    mapping = {train:'train', valid:'valid', test:'test'}

    #整理标签种类
    labels = set()
    for k,v in js.items():
        labels.update(set(v.keys()))
    labels = list(labels)
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for i,l in enumerate(labels)}


    #计数器，统计类别
    counter = Counter()
    AvgL,num_text = 0,0

    #将标签转换为数字label
    for idx,(k,v) in enumerate(js.items()):
        for kk,vv in v.items():
            datasets[mapping[k]]['text'] += vv
            AvgL += len(vv)
            num_text += 1

            datasets[mapping[k]]['raw_label'] += [kk] * len(vv)
            datasets[mapping[k]]['label'] += [label2id[kk]] *len(vv)
            counter.update([kk]*len(vv))
    
    AvgL /= num_text

    #数据集参数
    config = {'labels':labels,
           'label2id':label2id,
           'id2label':id2label,
           'counter': counter,
           'Avg. Length': AvgL
           }

    return datasets,config


def generate_mcls_datasets(js,train="train",valid="valid",test="test",text="text",label="label",**kwargs):
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
           'label2id':{l:i for i,l in enumerate(labels)},
           'id2label':{i:l for i,l in enumerate(labels)},
           'counter': counter,
           'hist':hist,
           'Avg. Length':AvgL
           }

    return datasets,config

def generate_keyword_datasets(js,train="train",valid="valid",test="test",text="text",label="label",sep=" ",**kwargs):
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



def generate_cluener_datasets(js,train="train",valid="valid",test="test",text="text",label="label",ner_encoding = 'BIO',**kwargs):
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

def generate_ner_datasets_label_as_key(js,train="train",valid="valid",test="test",text="text",label="label",ner_encoding = 'BIO',**kwargs):
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

    converter = NERLabelConverter(ner_encoding = ner_encoding,**kwargs)
    
    #将标签转换为数字label
    for k,v in js.items():
        if len(v) == 0 or (not isinstance(v,dict)) or (text not in v.keys() or label not in v.keys()):
            print(f'{k}没有导入，可能是因为数据集格式不是dict，或没有{text} 、{label}作为key')
            continue

        for t,l in zip(v[text],v[label]):
            import re
            t = re.sub(r"\s", "", t)
            datasets[mapping[k]]['text'].append(' '.join(t)) #white space tokenize
            datasets[mapping[k]]['raw_label'].append(l)
            datasets[mapping[k]]['label'].append(converter.encode_per_sentence(t,l))

    config = {
           'Avg. Length':AvgL,
           }
    
    config.update(converter.generate_config())

    return datasets,config


def generate_ner_datasets_inline(js,train="train",valid="valid",test="test",ner_encoding = 'BIO',**kwargs):
    #数据集
    datasets = { 'train':defaultdict(list),'valid':defaultdict(list),'test':defaultdict(list)}
    mapping = { train:'train', valid:'valid', test:'test'}
    
    
    #提取数据
    for idx,(k,v) in enumerate(js.items()):
        for vv in v['raw_text']:
            text,raw_label = extract_inline_nested_entities(vv)
            v['text'].append(text)
            v['raw_label'].append(raw_label)
        
    #统计文本长度
    AvgL = []
    for idx,(k,v) in enumerate(js.items()):
        AvgL += list(map(len,v['text']))
    AvgL = sum(AvgL)/len(AvgL)
    
        
    converter = NERLabelConverter(ner_encoding = ner_encoding)
    
    #将标签转换为数字label
    for k,v in js.items():
        if len(v) == 0 or (not isinstance(v,dict)):
            print(f'{k}没有导入，可能是因为数据集格式不是dict，或为空')
            continue

        for r_t,t,l in zip(v['raw_text'],v['text'],v['raw_label']):
            datasets[mapping[k]]['raw_text'].append(r_t)
            import re
            t = re.sub(r"\s", "", t)
            datasets[mapping[k]]['text'].append(' '.join(t)) #white space tokenize
            datasets[mapping[k]]['raw_label'].append(l)
            datasets[mapping[k]]['label'].append(converter.encode_per_sentence(t,l))


    #数据集参数
    config = {
           'Avg. Length':AvgL,
           }
    
    config.update(converter.generate_config())
    return datasets,config


def generate_dp_datasets_inline(js,train="train",valid="valid",test="test",ner_encoding = 'BIO',**kwargs):
    #数据集
    datasets = { 'train':defaultdict(list),'valid':defaultdict(list),'test':defaultdict(list)}
    mapping = { train:'train', valid:'valid', test:'test'}
    
    print("开始提取行内标注")


    #提取数据和标签
    entity_labels = set()
    relaton_labels = set()
    for idx,(k,v) in enumerate(js.items()):
        for vv in v['raw_text']:
            text,raw_label = extract_inline_dp_entities(vv)
            for anno in raw_label["entity"]:
                entity_labels.add(anno["label"])
            
            for anno in raw_label["relation"]:
                relaton_labels.add(anno)

            v['text'].append(text)
            v['raw_label'].append(raw_label)
        
    #统计文本长度
    AvgL = []
    for idx,(k,v) in enumerate(js.items()):
        AvgL += list(map(len,v['text']))
    AvgL = sum(AvgL)/len(AvgL)
    
        
    converter = DependecyParserLabelConverter(entity_labels=entity_labels,relation_labels=relaton_labels,ner_encoding = ner_encoding,**kwargs)
    
    #将标签转换为数字label
    for k,v in js.items():
        if len(v) == 0 or (not isinstance(v,dict)):
            print(f'{k}没有导入，可能是因为数据集格式不是dict，或为空')
            continue

        for r_t,t,l in zip(v['raw_text'],v['text'],v['raw_label']):
            datasets[mapping[k]]['raw_text'].append(r_t)
            import re
            t = re.sub(r"\s", "", t)
            datasets[mapping[k]]['text'].append(' '.join(t)) #white space tokenize
            datasets[mapping[k]]['raw_label'].append(l)
            ent_ids,rela_ids =  converter.encode_per_sentence(t,l)
            datasets[mapping[k]]['label'].append(ent_ids)
            datasets[mapping[k]]['rel_label'].append(rela_ids)

    #数据集参数
    config = {
           'Avg. Length':AvgL,
           }
    
    config.update(converter.generate_config())
    return datasets,config


def generate_triple_datasets_inline(js,train="train",valid="valid",test="test",ner_encoding = 'BIO',**kwargs):
    #数据集
    datasets = { 'train':defaultdict(list),'valid':defaultdict(list),'test':defaultdict(list)}
    mapping = { train:'train', valid:'valid', test:'test'}
    
    print("开始提取行内标注")


    #提取数据和标签
    entity_labels = set()
    relaton_labels = set()
    for idx,(k,v) in enumerate(js.items()):
        for vv in v['raw_text']:
            text,raw_label = extract_inline_triple_entities(vv)
            for anno in raw_label["entity"]:
                entity_labels.add(anno["label"])
            
            for anno in raw_label["relation"]:
                relaton_labels.add(anno)

            v['text'].append(text)
            v['raw_label'].append(raw_label)
        
    #统计文本长度
    AvgL = []
    for idx,(k,v) in enumerate(js.items()):
        AvgL += list(map(len,v['text']))
    AvgL = sum(AvgL)/len(AvgL)
    
        
    converter = DependecyParserLabelConverter(entity_labels=entity_labels,relation_labels=relaton_labels,ner_encoding = ner_encoding,**kwargs)
    
    #将标签转换为数字label
    for k,v in js.items():
        if len(v) == 0 or (not isinstance(v,dict)):
            print(f'{k}没有导入，可能是因为数据集格式不是dict，或为空')
            continue

        for r_t,t,l in zip(v['raw_text'],v['text'],v['raw_label']):
            datasets[mapping[k]]['raw_text'].append(r_t)
            import re
            t = re.sub(r"\s", "", t)
            datasets[mapping[k]]['text'].append(' '.join(t)) #white space tokenize
            datasets[mapping[k]]['raw_label'].append(l)
            ent_ids,rela_ids =  converter.encode_per_sentence(t,l)
            datasets[mapping[k]]['label'].append(ent_ids)
            datasets[mapping[k]]['rel_label'].append(rela_ids)

    #数据集参数
    config = {
           'Avg. Length':AvgL,
           }
    
    config.update(converter.generate_config())
    return datasets,config


def generate_relation_datasets(js,train="train",valid="valid",test="test",text = "text",ner_encoding = 'BIO',
                                ner_label = "entity", rel_label = "relation",annos = "annotations",
                                **kwargs):
    #数据集
    datasets = { 'train':defaultdict(list),'valid':defaultdict(list),'test':defaultdict(list)}
    mapping = { train:'train', valid:'valid', test:'test'}
    
    print("开始提取行内标注")


    #提取数据和标签
    entity_labels = set()
    relaton_labels = set()
    for idx,(k,v) in enumerate(js.items()):
        if rel_label in v:
            for text, ner_label,rel_label in zip(v[text],v[ner_label],v[rel_label]):
                raw_label = {
                    "entity":ner_label,
                    "relation":rel_label,
                }

                for anno in raw_label["entity"]:
                    entity_labels.add(anno["label"])
                
                for anno in raw_label["relation"]:
                    relaton_labels.add(anno)

                v['text'].append(text)
                v['raw_label'].append(raw_label)
            else:
                for text, raw_label in zip(v[text],v[annos]):
                    for anno in raw_label["entity"]:
                        entity_labels.add(anno["label"])
                    
                    for anno in raw_label["relation"]:
                        relaton_labels.add(anno)

                    v['text'].append(text)
                    v['raw_label'].append(raw_label)
        
    #统计文本长度
    AvgL = []
    for idx,(k,v) in enumerate(js.items()):
        AvgL += list(map(len,v['text']))
    AvgL = sum(AvgL)/len(AvgL)
    
    
    if "rel_encoding" in kwargs:
        del kwargs["rel_encoding"]
    
    if "ner_encoding" in kwargs:
        del kwargs["ner_encoding"]

    converter = DependecyParserLabelConverter(entity_labels=entity_labels,
                                            relation_labels=relaton_labels,
                                            ner_encoding = "Pointer", #使用指针的方式编码关系 [类别，start,end]
                                            rel_encoding= "REL", #使用指针的方式编码关系 [类别，src_start,src_end,tgt_start,tgt_end]
                                            **kwargs)
    
    #将标签转换为数字label
    for k,v in js.items():
        if len(v) == 0 or (not isinstance(v,dict)):
            print(f'{k}没有导入，可能是因为数据集格式不是dict，或为空')
            continue

        for r_t,t,l in zip(v['raw_text'],v['text'],v['raw_label']):
            datasets[mapping[k]]['raw_text'].append(r_t)
            import re
            t = re.sub(r"\s", "", t) #去除文本中空格
            datasets[mapping[k]]['text'].append(' '.join(t)) #white space tokenize
            datasets[mapping[k]]['raw_label'].append(l)
            ent_ids,rela_ids =  converter.encode_per_sentence(t,l)
            datasets[mapping[k]]['ner_label'].append(ent_ids) #不需要编码ner
            datasets[mapping[k]]['rel_label'].append(rela_ids) 

    #数据集参数
    config = {
           'Avg. Length':AvgL,
           }
    
    config.update(converter.generate_config())
    return datasets,config


def generate_relation_datasets_inline(js,train="train",valid="valid",test="test",ner_encoding = 'BIO',**kwargs):
    #数据集
    datasets = { 'train':defaultdict(list),'valid':defaultdict(list),'test':defaultdict(list)}
    mapping = { train:'train', valid:'valid', test:'test'}
    
    print("开始提取行内标注")


    #提取数据和标签
    entity_labels = set()
    relaton_labels = set()
    for idx,(k,v) in enumerate(js.items()):
        for vv in v['raw_text']:
            text,raw_label = extract_inline_triple_entities(vv)
            for anno in raw_label["entity"]:
                entity_labels.add(anno["label"])
            
            for anno in raw_label["relation"]:
                relaton_labels.add(anno)

            v['text'].append(text)
            v['raw_label'].append(raw_label)
        
    #统计文本长度
    AvgL = []
    for idx,(k,v) in enumerate(js.items()):
        AvgL += list(map(len,v['text']))
    AvgL = sum(AvgL)/len(AvgL)
    
    
    if "rel_encoding" in kwargs:
        del kwargs["rel_encoding"]
    
    if "ner_encoding" in kwargs:
        del kwargs["ner_encoding"]

    converter = DependecyParserLabelConverter(entity_labels=entity_labels,
                                            relation_labels=relaton_labels,
                                            ner_encoding = "Pointer", #使用指针的方式编码关系 [类别，start,end]
                                            rel_encoding= "REL", #使用指针的方式编码关系 [类别，src_start,src_end,tgt_start,tgt_end]
                                            **kwargs)
    
    #将标签转换为数字label
    for k,v in js.items():
        if len(v) == 0 or (not isinstance(v,dict)):
            print(f'{k}没有导入，可能是因为数据集格式不是dict，或为空')
            continue

        for r_t,t,l in zip(v['raw_text'],v['text'],v['raw_label']):
            datasets[mapping[k]]['raw_text'].append(r_t)
            import re
            t = re.sub(r"\s", "", t) #去除文本中空格
            datasets[mapping[k]]['text'].append(' '.join(t)) #white space tokenize
            datasets[mapping[k]]['raw_label'].append(l)
            ent_ids,rela_ids =  converter.encode_per_sentence(t,l)
            datasets[mapping[k]]['ner_label'].append(ent_ids) #不需要编码ner
            datasets[mapping[k]]['rel_label'].append(rela_ids) 

    #数据集参数
    config = {
           'Avg. Length':AvgL,
           }
    
    config.update(converter.generate_config())
    return datasets,config

def generate_nested_ner_datasets_inline(js,train="train",valid="valid",test="test",text="text",label="label",sep=" ",ner_encoding = 'GP',**kwargs):
    #数据集
    datasets = { 'train':defaultdict(list),'valid':defaultdict(list),'test':defaultdict(list)}
    mapping = { train:'train', valid:'valid', test:'test'}
    
    print("开始提取行内标注")
    #提取数据
    entity_labels = list()
    for idx,(k,v) in enumerate(js.items()):
        for vv in v['raw_text']:
            text,raw_label = extract_inline_nested_entities(vv)
            for anno in raw_label:
                entity_labels.append(anno["label"])
            v['text'].append(text)
            v['raw_label'].append(raw_label)
        
    #统计文本长度
    AvgL = []
    for idx,(k,v) in enumerate(js.items()):
        AvgL += list(map(len,v['text']))
    AvgL = sum(AvgL)/len(AvgL)
    
        
    converter = NERLabelConverter(entity_labels=entity_labels,ner_encoding = ner_encoding,**kwargs)
    # print(converter.entities_labels,converter.id2label,converter.label2id)
    
    #将标签转换为数字label
    for k,v in js.items():
        if len(v) == 0 or (not isinstance(v,dict)):
            print(f'{k}没有导入，可能是因为数据集格式不是dict，或为空')
            continue

        for r_t,t,l in zip(v['raw_text'],v['text'],v['raw_label']):
            datasets[mapping[k]]['raw_text'].append(r_t)
            import re
            t = re.sub(r"\s", "", t)
            datasets[mapping[k]]['text'].append(' '.join(t)) #white space tokenize
            datasets[mapping[k]]['raw_label'].append(l)
            datasets[mapping[k]]['label'].append(converter.encode_per_sentence(t,l))


    #数据集参数
    config = {
           'Avg. Length':AvgL,
           }
    
    config.update(converter.generate_config())
    return datasets,config


def generate_ner_datasets(js,train="train",valid="valid",test="test",text="text",label="label",sep=" ",entity_label = 'label',loc = 'loc',ner_encoding= 'BIO',**kwargs):
    #数据集
    datasets = { 'train':defaultdict(list),'valid':defaultdict(list),'test':defaultdict(list)}
    mapping = { train:'train', valid:'valid', test:'test'}
    
        
    #统计文本长度
    AvgL = []
    for idx,(k,v) in enumerate(js.items()):
        AvgL += list(map(len,v[text]))
    AvgL = sum(AvgL)/len(AvgL)
    
    
    converter = NERLabelConverter(loc=loc,ner_encoding = ner_encoding,**kwargs)
    
    #将标签转换为数字label
    for k,v in js.items():
        if len(v) == 0 or (not isinstance(v,dict)) or (text not in v.keys() or label not in v.keys()):
            print(f'{k}没有导入，可能是因为数据集格式不是dict，或没有{text} 、{label}作为key')
            continue

        for t,l in zip(v[text],v[label]):
            import re
            t = re.sub(r"\s", "", t)
            datasets[mapping[k]]['text'].append(' '.join(t)) #white space tokenize
            datasets[mapping[k]]['raw_label'].append(l)
            datasets[mapping[k]]['label'].append(converter.encode_per_sentence(t,l))

    #数据集参数
    config = {
           'Avg. Length':AvgL,
           }
    
    config.update(converter.generate_config())
    return datasets,config