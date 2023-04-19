from ..utils.english import _is_english_char_lower,_is_english_char_upper
from typing import *

from collections import Counter,defaultdict
import numpy as np

#确认task
#统一任务类型
def _unify_task(task):
    if task.lower() in [0,
                        'cls',
                        'classification',
                        'classify']:

        return 'CLS'
    elif task.lower() in [1,
                        'reg',
                        'regression',
                        'regressor',
                        'sa',
                        'sentitive analysis']:

        return 'REG'
    elif task.lower() in [2,
                        'ner',
                        'namely entity recognition']:

        return 'NER'
    elif task.lower() in [2,
                        'key',
                        'kw',
                        'key word',
                        'keyword',
                        'keywords',
                        'key words']:

        return 'KW'
    elif task.lower() in [3,
                        'mcls',
                        'multi-class',
                        'multiclass',
                        'multiclasses',
                        'mc',
                        'multi-choice',
                        'multichoice']:

        return 'MCLS'
    elif task.lower() in [4,
                        'cluener',
                        'clue_ner',
                        'clue ner']:

        return 'CLUENER'

    elif task.lower() in [5,
                        "nested",
                        "nest",
                        "nested ner",
                        "nestedner",
                        "nestner"]:

        return "NestedNER"
    
    elif task.lower() in [6,
                        "DP",
                        "dp",
                        "parser",
                        "parsing",
                        "dependency",
                        "dependency parsing",
                        "dependency-parser"]:
        
        return "DP"
    elif task.lower() in [7,
                        "triple",
                        "tri",
                        "tr"]:

        return "Triple"
    elif task.lower() in [8,
                        "rel",
                        "rela",
                        "relation",
                        "relas",
                        "r"]:

        return "Relation"


#将标签转换为数字onehot label
def convert_label2onehot(ids,labels):
    '''
    ids `List[str]` or `List[int]` or `str`:
        标签或者0,1,2,3...
        
    labels `List[str]`:
        所有的标签
    '''
    onehot = [0.]*len(labels)
    if isinstance(ids,list):
        for id in ids:
            if isinstance(id,str):
                onehot[labels.index(id)] = 1.
            else:
                onehot[id] = 1.
    elif isinstance(ids,(float,int,np.int_,np.int64)):
        onehot[ids] = 1.
    elif isinstance(ids,str):
        onehot[labels.index(ids)] = 1.
    else:
        raise NotImplementedError()
    return onehot



def _unify_ner_anno_format(ner_anno_format):
    '''
    统一ner标注的标签格式
    '''
    if ner_anno_format in ["eng","lower","l"]: # 英文
        return _is_english_char_lower
    elif ner_anno_format in ["ENG","capital","upper","u"]: #中文
        return _is_english_char_upper
    elif ner_anno_format in ["digital","id_end","1","d"]: #数字
        return lambda x: x.isdigit()
    elif isinstance(ner_anno_format, callable):
        return ner_anno_format


def _unify_ner_anno_bracket(ner_anno_bracket):
    '''
    统一ner标注的括号形式
    '''
    a,b = ner_anno_bracket

    if isinstance(a, str):
        # check_a = a.startswith
        check_a = lambda x: x == a
    elif isinstance(a, callable):
        check_a = a 
    else:
        assert 0,f"ner_anno_bracket[0]应该是一种字符，例如【，（，或使用一个函数确定左括号，但是现在是{a}"

    if isinstance(a, str):
        # check_b = b.startswith
        check_b = lambda x: x == b 
    elif isinstance(a, callable):
        check_b = b
    else:
        assert 0,f"ner_anno_bracket[1]应该是一种字符，例如】，），或使用一个函数确定右括号，但是现在是{a}"

    return check_a,check_b

def _unify_nested_ner_split(nested_ner_split):
    if isinstance(nested_ner_split, str):
        # return nested_ner_split.startswith
        return lambda x: x == nested_ner_split
    elif isinstance(nested_ner_split, callable):
        return nested_ner_split

def _unify_ner_id_format(ner_id_format):
    if ner_id_format in ["eng","lower","l"]: # 小写
        return _is_english_char_lower
    elif ner_id_format in ["ENG","capital","upper","u"]: #大写
        return _is_english_char_upper
    elif ner_id_format in ["digital","digit","1","d"]: #数字
        return lambda x: x.isdigit()
    elif isinstance(ner_id_format, callable):
        return ner_id_format

def _unify_ner_id_split(ner_id_split):
    if isinstance(ner_id_split, str):
        return lambda x: x == ner_id_split if ner_id_split else False
    elif isinstance(ner_id_split, callable):
        return ner_id_split


def _get_per_ner_anno(s,is_ner_bracket,is_ner_anno,is_nested_ner_split,is_ner_id,is_ner_id_split):
    '''
    s：
        例如：【标注|实体id....实体id】，在不混淆的情况下，标注实体和实体id可以为空
                
                可以使用空格分开标注和原文，以防混淆
    '''
    is_left_bracket,is_right_bracket = is_ner_bracket

    #寻找起点位置
    start = 0
    while not is_left_bracket(s[start]):
        start += 1
        if start > len(s):
            assert 0,f"请检查{s}的标注，没有标签"

    #寻找标注类别
    idx = start + 1
    while is_ner_anno(s[idx]):
        idx += 1

    class_anno = s[start+1:idx]
  
    #寻找嵌套实体标注位置
    jdx = idx
    if is_nested_ner_split(s[jdx]):
        jdx += 1
        while is_ner_anno(s[jdx]):
            jdx += 1
        nested_id = s[idx+1:jdx]
    else:
        nested_id = None
    
    #寻找id标注位置
    id_end = jdx
    if is_ner_id_split(s[id_end]):
        id_end += 1
        id_end = id_end
        while is_ner_id(s[id_end]):
            id_end += 1
        ent_id = s[jdx+1:id_end]
    else:
        ent_id = None
    
    right_bracket = None
    end = id_end
    ent_end = None
    entity = ""
    while ent_end is None:
        if is_left_bracket(s[end]): #如果是左括号，即存在同样起点的嵌套的实体
            end += 1
            while is_ner_anno(s[end]):
                end += 1

            if is_nested_ner_split(s[end]): #有nested标注
                if is_nested_ner_split(s[end]):
                    end += 1
                    while is_ner_anno(s[end]):
                        end += 1


            if is_ner_id_split(s[end]): # 有id标注
                if is_ner_id_split(s[end]):
                    end += 1
                    while is_ner_id(s[end]):
                        end += 1
            
        #找到一个右括号
        elif is_right_bracket(s[end]):
            #匹配 nested_id
            kdx = end - 1

            # while kdx > id_end and is_ner_anno(s[kdx]):
            #     kdx -= 1
            #     entity = entity[:-1] #更新entity
            
            # if (nested_id is None and kdx+1 == end) or  s[kdx+1:end] == nested_id:
            #     ent_end = kdx + 1
            #     break

            #寻找id标注
            while kdx > id_end and is_ner_id(s[kdx]):
                kdx -= 1
                entity = entity[:-1] #更新entity

            #如果id匹配或者没有id标注或者右括号没有id标注
            if ent_id is None or kdx+1 == end or  s[kdx+1:end] == ent_id:
                ent_end = kdx + 1
                break

            end += 1
        else:
            entity += s[end]
            end += 1
        if end > len(s):
            assert 0,f"请检查{s}的标注，没有找到右括号或者和{nested_id}对应的嵌套标注"

    label = {
        "entity":entity,
        "label":class_anno,
        "nested_id":nested_id,
        "ent_id":ent_id,
        "origin_loc":[start,end],
        "loc":[start,start + len(entity) - 1] #左闭右闭区间
    }

    return label




def _remove_ner_anno(s,origin_loc,is_ner_bracket,is_ner_anno,is_nested_ner_split,is_ner_id,is_ner_id_split):
    new_s = "" #make a copy
    start,end = origin_loc
    
    is_left_bracket,is_right_bracket = is_ner_bracket
    
    new_s += s[:start]
    #remove the left anno
    ent_left = start
    if is_left_bracket(s[ent_left]):
        ent_left +=1
        while is_ner_anno(s[ent_left]) or is_nested_ner_split(s[ent_left]):
            ent_left += 1
        
        if is_nested_ner_split(s[ent_left]): #有nested标注
            if is_nested_ner_split(s[ent_left]):
                ent_left += 1
                while is_ner_anno(s[ent_left]):
                    ent_left += 1


        if is_ner_id_split(s[ent_left]): # 有id标注
            if is_ner_id_split(s[ent_left]):
                ent_left += 1
                while is_ner_id(s[ent_left]):
                    ent_left += 1
        
        
    #remove the right anno
    ent_right = end
    if is_right_bracket(s[ent_right]):
        ent_right -= 1
        # while is_ner_anno(s[ent_right]) or is_nested_ner_split(s[ent_right]):
        #     ent_right -= 1
        while is_ner_anno(s[ent_right]):
            ent_right -= 1

        # 有id标注
        while is_ner_id(s[ent_right]):
            ent_right -= 1
    
    new_s += s[ent_left:ent_right+1]
    new_s += s[end+1:]

    return new_s

def extract_inline_dp_entities(raw_labels,ner_anno_bracket=("【","】"),ner_anno_format = "eng",nested_ner_split="_", \
               ner_id_format = "digit" ,ner_id_split="|",rel_label_split=" ",rel_ent_id_split=",", use_head = True , **kwargs):
    '''
    行内的嵌套实体抽取
    '''
    is_ner_anno = _unify_ner_anno_format(ner_anno_format)
    is_ner_bracket = _unify_ner_anno_bracket(ner_anno_bracket)
    is_nested_ner_split = _unify_nested_ner_split(nested_ner_split)
    is_ner_id = _unify_ner_id_format(ner_id_format)
    is_ner_id_split = _unify_ner_id_split(ner_id_split)

    #提取第一行的实体
    s = raw_labels[0]
    ents = []
    start = 0
    cadidates = {}
    while start < len(s):
        c = s[start]
        if is_ner_bracket[0](c): #如果找到一个左侧的括号
            label = _get_per_ner_anno(s, is_ner_bracket, is_ner_anno, is_nested_ner_split,is_ner_id,is_ner_id_split) #寻找第一个标签
            s = _remove_ner_anno(s,label['origin_loc'],is_ner_bracket, is_ner_anno, is_nested_ner_split, is_ner_id,is_ner_id_split ) #去除这个标签
            ents.append(label)
        else:
            start += 1



    ids2ent = {ent["ent_id"]:ent for ent in ents}
    ids2ent["-100"] = {"entity":None,"label":"HEAD","nested_id":None, "ent_id":None,"origin_loc":[-100,-100],"loc":[-100,-100]}  #左闭右闭区间

    #提取链接关系，一行一个关系
    rels = defaultdict(list)
    for line in raw_labels[1:]:
        if line:
            rel,ent_ids = line.split(rel_label_split)
            A,B = ent_ids.split(rel_ent_id_split)[:2]

            if A not in ids2ent or B not in ids2ent:
                print(f'''请检查标注数据{line}\n分隔符为"{rel_label_split}"和"{rel_ent_id_split}"''')
            
            A = ids2ent[A]
            B = ids2ent[B]
            
            rels[rel].append([A,B])


    label = {"entity":ents,"relation":rels}
    return s,label


def extract_inline_triple_entities(raw_labels,ner_anno_bracket=("【","】"),ner_anno_format = "eng",nested_ner_split="_", \
               ner_id_format = "digit" ,ner_id_split="|",rel_label_split=" ",rel_ent_id_split=",", use_head = True , **kwargs):
    '''
    行内的嵌套实体抽取
    '''
    # is_ner_anno = _unify_ner_anno_format(ner_anno_format)
    # is_ner_bracket = _unify_ner_anno_bracket(ner_anno_bracket)
    # is_nested_ner_split = _unify_nested_ner_split(nested_ner_split)
    # is_ner_id = _unify_ner_id_format(ner_id_format)
    # is_ner_id_split = _unify_ner_id_split(ner_id_split)

    #提取第一行的实体
    s = raw_labels[0]

    s,ents = extract_inline_nested_entities(s,
                    ner_anno_bracket= ner_anno_bracket,
                    ner_anno_format = ner_anno_format,
                    nested_ner_split= nested_ner_split, 
                    ner_id_format = ner_id_format ,
                    ner_id_split= ner_id_split,
                    **kwargs
                    )
                
    # texts = ''
    # ents = []
    # start = 0
    # cadidates = {}

    # try:
    #     while start < len(s):
    #         c = s[start]
    #         if is_ner_bracket[0](c): #如果找到一个左侧的括号
    #             label = _get_per_ner_anno(s, is_ner_bracket, is_ner_anno, is_nested_ner_split,is_ner_id,is_ner_id_split) #寻找第一个标签
    #             s = _remove_ner_anno(s,label['origin_loc'],is_ner_bracket, is_ner_anno, is_nested_ner_split, is_ner_id,is_ner_id_split ) #去除这个标签
    #             ents.append(label)
    #         else:
    #             start += 1
    # except:
    #     raise Exception(f"错误的标注文本为:\n {s} \n")

    # #移除空格带来的偏移
    # start ,space_num = 0,0
    # while start < len(s):
    #     if s[start].isspace():
    #         for j in range(len(ents)):
    #             if ents[j]["loc"][0] > start - space_num:
    #                 ents[j]["loc"][0] -= 1
                
    #             if ents[j]["loc"][1] >= start - space_num:
    #                 ents[j]["loc"][1] -= 1
            
    #         space_num += 1
    #     start += 1
    # s = s.replace(" ","=")
    
    # assert 0

    ids2ent = {ent["ent_id"]:ent for ent in ents}
    ids2ent["-100"] = {"entity":None,"label":"HEAD","nested_id":None, "ent_id":None,"origin_loc":[-100,-100],"loc":[-100,-100]}  #左闭右闭区间

    #提取链接关系，一行一个关系
    rels = defaultdict(list)
    for line in raw_labels[1:]:
        if line:
            try:
                A,rel,B = line.split(rel_label_split)
            except:
                raise Exception(f"标注数据错误，请检查标注数据，分隔符为{rel_label_split}，\n 文本为 \n {raw_labels[0]} \n，数据为\n{line}，")
            
            try:
                A = ids2ent[A]
                B = ids2ent[B]
            except:
                raise Exception(f"标注数据错误，请检查标注数据，A为{A}，B为{B}，\n 映射关系为{ids2ent}，\n 文本为 \n {raw_labels[0]} \n，数据为\n{line}，")

            rels[rel].append([A,B])


    label = {"entity":ents,"relation":rels}
    return s,label

def extract_inline_nested_entities(s,ner_anno_bracket=("【","】"),ner_anno_format = "eng",nested_ner_split="_", \
               ner_id_format = "digit" ,ner_id_split="|",**kwargs):
    '''
    行内的嵌套实体抽取
    输入：
        原始句子
    输出：
        去除标注后句子,entities

        entites: {
            "entity":entity,    #实体
            "label":class_anno, #实体的类别
            "nested_id":nested_id,  #嵌套标注
            "ent_id":ent_id, #实体 id
            "origin_loc":[start,end], #原始位置
            "loc":[start,start + len(entity) - 1] #去除标注后位置 左闭右闭区间
        }
    '''
    is_ner_anno = _unify_ner_anno_format(ner_anno_format)
    is_ner_bracket = _unify_ner_anno_bracket(ner_anno_bracket)
    is_nested_ner_split = _unify_nested_ner_split(nested_ner_split)
    is_ner_id = _unify_ner_id_format(ner_id_format)
    is_ner_id_split = _unify_ner_id_split(ner_id_split)


    texts = ''
    entities = []
    start = 0


    try:
        while start < len(s):
            c = s[start]
            if is_ner_bracket[0](c): #如果找到一个左侧的括号
                # import pdb;pdb.set_trace();
                label = _get_per_ner_anno(s, is_ner_bracket, is_ner_anno, is_nested_ner_split,is_ner_id,is_ner_id_split) #寻找第一个标签
                s = _remove_ner_anno(s,label['origin_loc'],is_ner_bracket, is_ner_anno, is_nested_ner_split, is_ner_id,is_ner_id_split ) #去除这个标签
                entities.append(label)
            else:
                start += 1
    except:
        raise Exception(f"错误的标注文本为:\n {s} \n")

    #移除空格带来的偏移
    start ,space_num = 0,0
    while start < len(s):
        if s[start].isspace():
            for j in range(len(entities)):
                if entities[j]["loc"][0] > start - space_num:
                    entities[j]["loc"][0] -= 1
                
                if entities[j]["loc"][1] >= start - space_num:
                    entities[j]["loc"][1] -= 1
            
            space_num += 1
        start += 1
    return s,entities


def extract_inline_entities(s,**kwargs):
    '''
    抽取在行内的实体
    '''
    texts = ''
    labels = []
    start = 0
    while start < len(s):
        c = s[start]
        if c == '【':
            end = start
            while s[end] != '】':
                end += 1
                if end >= len(s):
                    assert 0,f"文本\n \t{s}\n缺少】"
            label = ''
            entity = ''
            flag = True
            for label_dix in range(start+1,end):
                c = s[label_dix]
                if flag and _is_english_char_lower(c) :
                    label += c
                else:
                    flag = False
                    entity += c
            
            
            labels.append({'label':label,'entity':entity,'loc':[len(texts),len(texts)+len(entity)-1]})
            texts += entity
            start = end + 1
        else:
            texts += c
            start += 1
            
    return texts,labels

def _unify_nested_ner_encoding(ner_encoding):
    if ner_encoding in ["GlobalPointer","GP","gp"]:
        return "GP"
    elif ner_encoding in ["span","Span","SPAN","SP"]:
        return "SPAN"
    else:
        Warning("未识别出ner编码方式，默认使用GlobalPointer编码")
        return "GP"

def _unify_ner_encoding(ner_encoding):
    if ner_encoding in ["BIO","bio",1]:
        return "BIO"
    elif ner_encoding in ["io","IO",0]:
        return "IO"
    elif ner_encoding in ["bioes","BIOES",2]:
        return "BIOES"
    else:
        Warning("未识别出ner编码方式，默认使用BIO编码")
        return "BIO"


class DependecyParserLabelConverter:
    def __init__(self,entity_labels=[],relation_labels=[],max_length=510,max_relations=16,
                            ner_encoding="BIO",rel_encoding="REL",
                            loc_name = 'loc',
                            ent_id_name = "ent_id",
                            relation_head = "relation",
                            entity_head = "entity",
                            entity_label_name = "label", 
                            **kwargs):

        self.ner_label_converter = NERLabelConverter(entity_labels, entity_label_name=entity_label_name,loc_name = loc_name,
                                    max_length=max_length,ner_encoding=ner_encoding,**kwargs)

        self.rel_label_converter = NERLabelConverter(entity_labels, entity_label_name=entity_label_name,loc_name = loc_name,
                                    max_length=max_length,ner_encoding="SPAN",**kwargs)


        self.relation_labels = []
        self.relation_id2label = {}
        self.relation_label2id = {}

        self.relation_counter = Counter()
        self.relation_head = relation_head
        self.entity_head = entity_head

        self.max_length = max_length
        self.max_relations = max_relations
        self.encoding = rel_encoding
        self.loc_name = loc_name
        self.ent_id_name = ent_id_name,
        self.num_relations = len(self.relation_labels)
        self.spo = set()
        self.spo_counter = Counter()
        self.params = kwargs

        for relation_label in relation_labels:
            self._update_relation_label(relation_label)
        

    def _update_spo_label(self,s,p,o):
        self.spo_counter[s,p,o] += 1
        self.spo.add((s,p,o))


    def _update_relation_label(self,relation_label):
        self.relation_counter[relation_label] += 1

        if relation_label in self.relation_labels:
            return
        
        self.relation_labels.append(relation_label)
        self.num_relations += 1

        idx = len(self.relation_labels) - 1
        self.relation_id2label[idx] = relation_label
        self.relation_label2id[relation_label] = idx
            
    def generate_config(self):
        config = {
            'num_rels':self.num_relations,
            'rels':self.relation_labels,
            "relation2id":self.relation_label2id,
            "id2relation":self.relation_id2label,
            "rel_encoding":self.encoding,
            'rel_counter':self.relation_counter,
            "spo_counter":{"\t".join(k):v for k,v in self.spo_counter.items()}
        }

        config.update(self.ner_label_converter.generate_config())
        return config    

    def _wash_text(self,text):
        import re
        return re.sub("\s","",text)

    def encode_per_relation(self,rel,rel_label,ids = None,encoding = None):
        self._update_relation_label(rel_label)
        source,target = rel

        if encoding is None:
            encoding = self.encoding

        if self.loc_name in source:
            src_start,src_end = source[self.loc_name]

        if self.loc_name in target:
            tgt_start,tgt_end = target[self.loc_name]
        
        if "label" in source and "label" in target:
            self._update_spo_label(source["label"],rel_label, target["label"])
        
        if target in ["-100",-100]:
            tgt_start = tgt_end = -100

        if source in ["-100",-100]:
            src_start = src_end = -100


        if encoding == "Pointer":
            #找到非空的第一个relation，进行填充
            relation_idx = len(ids[self.relation_label2id[rel_label]].sum(axis=(1,2)).nonzero()[0])

            try:
                ids[self.relation_label2id[rel_label],relation_idx,0,0] = src_start
                ids[self.relation_label2id[rel_label],relation_idx,0,1] = src_end
                ids[self.relation_label2id[rel_label],relation_idx,1,0] = tgt_start
                ids[self.relation_label2id[rel_label],relation_idx,1,1] = tgt_end
            except Exception as e:
                print(e)
                print(f"请检查标注数据，source是{source}, target是{target}")

        elif encoding == "REL":
            #找到非空的第一个relation，进行填充
            relation_idx = len(ids.sum(axis=1).nonzero()[0])
            ids[relation_idx,0] = self.relation_label2id[rel_label]
            try:
                ids[relation_idx,1] = src_start
                ids[relation_idx,2] = src_end
                ids[relation_idx,3] = tgt_start
                ids[relation_idx,4] = tgt_end
            except Exception as e:
                print(e)
                print(f"请检查标注数据，source是{source}, target是{target}")

        elif encoding == "DP":
            try:
                ids[src_start:src_end,0] = source[self.ent_id_name]
                ids[src_start:src_end,1] = src_start
                ids[src_start:src_end,2] = src_end
                ids[src_start:src_end,3] = target[self.ent_id_name]
                ids[src_start:src_end,4] = tgt_start
                ids[src_start:src_end,5] = tgt_end
                ids[src_start:src_end,6] = self.relation_label2id[rel_label]
            except Exception as e:
                print(e)
                print(f"请检查标注数据，source是{source}, target是{target}")

    def encode_per_sentence(self,text,annos,encoding = None):
        text = self._wash_text(text)

        if encoding is None:
            encoding = self.encoding

        if encoding == "Pointer":
            relation_ids = np.zeros((self.num_relations,self.max_relations,2,2), dtype=np.int64) #[num_rels,max_entities,src_or_tgt,start_or_end]
        elif encoding == "REL":
            relation_ids = np.zeros((self.num_relations*self.max_relations,5), dtype=np.int64) #[rel_type, src_start,src_end,tgt_start,tgt_end]
            
        elif encoding == "DP":
            relation_ids = np.ones((len(text),7),dtype=np.int64) #[src_id,src_start,src_end,tgt_id,tgt_start,tgt_end,rela_label] 
            relation_ids = -100 * relation_ids


        if isinstance(annos, dict):
            #{rel_label_name:rels,ent_label_name:ents}
            if self.relation_head in annos and self.entity_head in annos:
                #需要对齐rela_label和ent_label，这里对齐loc
                ner_ids = self.ner_label_converter.encode_per_sentence(text, annos[self.entity_head])

                relation_annos = annos[self.relation_head]
                # [...]
                if isinstance(relation_annos, list):
                    pass
                elif isinstance(relation_annos, dict):
                    for rel_label,rels in relation_annos.items():
                        if isinstance(rels, (list,tuple)):
                            for rel in rels:
                                self.encode_per_relation(rel,rel_label,relation_ids,encoding)

                #填充没有值的地方
                relation_ids[relation_ids.sum(axis=1) == 0] = -100

        if encoding == "DP":
            for idx in range(len(text)):
                src_loc = (relation_ids[idx,1],relation_ids[idx,2])
                tgt_loc = (relation_ids[idx,4],relation_ids[idx,5])
                relation_ids[idx,0] = ent2id[src_loc]
                relation_ids[idx,3] = ent2id[tgt_loc]

            #向右填充
            if len(relation_ids) > self.max_length:
                relation_ids = relation_ids[:self.max_length]
            else:
                relation_ids = np.pad(relation_ids,((0,self.max_length-len(relation_ids)),(0,0)),constant_values = -100)
                
        return ner_ids,relation_ids


        
class NERLabelConverter:
    def __init__(self,entity_labels = [],
                        ner_encoding = 'BIO',
                        entity_label_name = 'label',
                        loc_name = 'loc',
                        max_length = 510, 
                        max_entities = 64,
                        padding_method = "right",
                        **kwargs):
        '''
        NER标签转换的方式
        '''


        self.entity_labels = []
        self.entity2id = {}

        self.id2label = {}
        self.label2id = {}
        self.counter = Counter()
                
        self.entity_label_name = entity_label_name
        self.loc_name = loc_name
        self.max_length = max_length
        self.max_entities = max_entities
        self.encoding = ner_encoding
        self.padding_method = padding_method
        
        for entity_label in entity_labels:
            self._update_entity_label(entity_label)
        
        self.num_labels = len(self.label2id)
        self.labels = list(self.label2id.keys())

        self.params = kwargs
        
    def _update_entity_label(self,entity_label):
        #BIO
        self.counter[entity_label] += 1

        if entity_label in self.entity2id:
            return

        self.entity_labels.append(entity_label)
        self.entity2id[entity_label] = len(self.entity2id)
        
        if self.encoding in ["IO","BIO","BIOE","BIOES"]:
            #初始化
            # if len(self.entity_labels) == 0:
            self.id2label[0] = 'O'
            self.label2id['O'] = 0
            
            
            idx = len(self.entity_labels) - 1
            
            #BIO
            if self.encoding == 'BIO':
                self.id2label[idx*2+1] = f'B-{entity_label}'
                self.id2label[idx*2+2] = f'I-{entity_label}'
                self.label2id[f'B-{entity_label}'] = idx*2+1
                self.label2id[f'I-{entity_label}'] = idx*2+2
            #IO
            elif self.encoding == 'IO':
                self.id2label[idx+1] = f'I-{entity_label}'
                self.label2id[f'I-{entity_label}'] = idx+1
            #BIOE
            elif self.encoding == "BIOE":
                self.id2label[idx*3+1] = f'B-{entity_label}'
                self.id2label[idx*3+2] = f'I-{entity_label}'
                self.id2label[idx*3+3] = f'E-{entity_label}'
                self.label2id[f'B-{entity_label}'] = idx*3+1
                self.label2id[f'I-{entity_label}'] = idx*3+2
                self.label2id[f'E-{entity_label}'] = idx*3+3
            #BIOES
            elif self.encoding == 'BIOES':
                self.id2label[idx*4+1] = f'B-{entity_label}'
                self.id2label[idx*4+2] = f'I-{entity_label}'
                self.id2label[idx*4+3] = f'E-{entity_label}'
                self.id2label[idx*4+4] = f'S-{entity_label}'
                self.label2id[f'B-{entity_label}'] = idx*4+1
                self.label2id[f'I-{entity_label}'] = idx*4+2
                self.label2id[f'E-{entity_label}'] = idx*4+3
                self.label2id[f'S-{entity_label}'] = idx*4+4
                
            self.num_labels = len(self.label2id)
            self.labels = list(self.label2id.keys())
        
        elif self.encoding == "GP":
            self.num_labels = len(self.entity_labels) * int(self.max_length * (self.max_length+1) // 2)

        elif self.encoding == "SPAN":
            self.num_labels = len(self.entity_labels) * self.max_length * 2

    def generate_config(self):
        if self.encoding in ["IO","BIO","BIOE","BIOES"]:
            config = {
                "num_entities":len(self.entity_labels),
                'num_labels':self.num_labels,
                'labels':self.labels,
                'id2label':self.id2label,
                'label2id':self.label2id,
                'entity2id':self.entity2id,
                'id2entity':{v:k for k,v in self.entity2id.items()},
                'ner_encoding':self.encoding,
                'entities':self.entity_labels,
                'counter':self.counter
            }
        else:
            config = {
                "num_entities":len(self.entity_labels),
                'num_labels':self.num_labels,
                'entity2id':self.entity2id,
                "id2entity":{v:k for k,v in self.entity2id.items()},
                'ner_encoding':self.encoding,
                'entities':self.entity_labels,
                'counter':self.counter
            }
        return config
        
    def convert(self,id_or_label):
        '''
        id 和 label之间进行转化
        '''
        if isinstance(id_or_label,int):
            return self.id2label[id_or_label]
        elif isinstance(id_or_label,str):
            if id_or_label.isnumeric():
                return self.id2label[int(id_or_label)]
            else:
                return self.label2id[int(id_or_label)]

    def _wash_text(self,text):
        import re
        return re.sub("\s","",text)
        
    def encode_per_entity(self,entity,entity_label,loc,ids = None,encoding = None):
        start,end = loc

        if encoding is None:
            encoding = self.encoding

        self._update_entity_label(entity_label)

        if encoding in ["BIO","IO","BIOE","BIOES"]:
            new_ids = [0]*len(entity)
            if len(entity) == 0:
                return []

            #BIO编码
            if encoding == 'BIO':
                new_ids[0] = self.label2id[f'B-{entity_label}']
                new_ids[1:] = [self.label2id[f'I-{entity_label}']]* (len(entity)-1)
            #IO编码
            elif encoding == 'IO':
                new_ids = [self.label2id[f'I-{entity_label}']]* len(entity)
            #BIOE编码
            elif encoding == "BIOE":
                new_ids[0] = self.label2id[f'B-{entity_label}']
                new_ids[1:-1] = [self.label2id[f'I-{entity_label}']]* (len(entity)-1)
                new_ids[-1] = self.label2id[f'E-{entity_label}']
            #BIOES编码
            elif encoding == 'BIOES':
                if len(entity) == 1:
                    new_ids[0] = self.label2id[f'S-{entity_label}']
                else:
                    new_ids[0] = self.label2id[f'B-{entity_label}']
                    new_ids[-1] = self.label2id[f'E-{entity_label}']
                    if len(entity) > 2:
                        new_ids[1:-1] = [self.label2id[f'I-{entity_label}']] * (len(entity) -2)
        
            ids[start:end] = new_ids

        #Global Pointer 编码
        elif encoding == "GP":
            if ids.shape[0] != len(self.entity_labels):
                ids.resize(len(self.entity_labels),self.max_length,self.max_length,refcheck=False)
            # ids[self.entity2id[entity_label],[start,end]]  = 1
            ids[self.entity2id[entity_label],start+1,end]  = 1 #原先是左闭右开，模型中改为左闭右闭


        #SPAN 编码
        elif encoding == "SPAN":
            if ids.shape[0] != len(self.entity_labels):
                ids.resize(len(self.entity_labels),self.max_length,2,refcheck=False)
            # ids[self.entity2id[entity_label],start,0]  = 1
            # ids[self.entity2id[entity_label],end,0]  = 1
            ids[self.entity2id[entity_label],start+1 ,0]  = 1
            ids[self.entity2id[entity_label],end,1]  = 1
        
        #原始编码，用于调试
        elif encoding == "RAW":
            ids.append({
                "loc":loc,
                "entity":entity,
                "label":entity_label
            })
        #Pointer编码
        elif encoding == "Pointer":
            num_existed = sum(ids[:,0] != -100)
            ids[num_existed,0] = self.entity2id[entity_label]
            ids[num_existed,1] = start
            ids[num_existed,2] = end - 1 #存储位置时，左闭右闭



    def encode_per_sentence(self,text,annos,encoding = None):
        text = self._wash_text(text)

        if encoding is None:
            encoding = self.encoding

        if encoding in ["IO","BIO","BIOE","BIOES"]:
            ids = [0]*len(text)
        elif encoding == "GP" :
            ids = np.zeros((len(self.entity_labels),self.max_length,self.max_length),dtype=np.int64)
            # ids = np.zeros((2,self.max_length,self.max_length),dtype=np.int64)
            # 对角线以下置为-100
            # ids[np.tril(np.ones(len(self.entity_labels),self.max_length))>0] = -100

        elif encoding == "SPAN":
            # ids = np.zeros((len(self.entity_labels),self.max_length,2),dtype=np.int64)
            ids = np.zeros((len(self.entity_labels),self.max_length,2),dtype=np.int64)

        elif encoding == "RAW":
            ids = []  #list of dict，返回原始标注
        elif encoding == "Pointer":
            ids = np.ones((self.max_entities,3),dtype=np.int64) * (-100)
        else:
            raise NotImplementedError()

        if isinstance(annos,(tuple,list)):
            for v in annos:
                #[[entity_label,(0,3)]]
                if isinstance(v,(tuple,list)):
                    entity_label = v[0]
                    start,end = v[1]
                #[{entity_label_name:'entity_label,'loc':(0,3)}]
                elif isinstance(v,dict):
                    entity_label = v[self.entity_label_name]
                    if isinstance(self.loc_name, (tuple,list)): 
                        start = v[self.loc_name[0]]
                        end = v[self.loc_name[1]]
                    else:
                        start,end = v[self.loc_name]
                else:
                    raise NotImplementedError()
                #[start,end]左闭右闭
                end += 1
                entity = text[start:end]

                self.encode_per_entity(entity,entity_label,(start,end),ids,encoding)

        #label as key {entity_label:...}
        elif isinstance(annos,dict):
            for k,v in annos.items():
                entity_label = k
                #{entity_label:[...]}
                if isinstance(v,(tuple,list)):
                    #{entity_label:[[...],[...]]} 
                    if isinstance(v[0],list):
                        for vv in v:
                            #{entity_label:[[0,3],[5,9]]}
                            if isinstance(vv[0],int):
                                start,end = vv
                                end += 1
                                entity = text[start:end]
                                self.encode_per_entity(entity,entity_label,(start,end),ids,encoding)
                            #{entity_label:[[entity,(0,3)],[entity,(5,9)]]}
                            elif isinstance(vv[0],str) and isinstance(vv[1][0],int):
                                start,end = vv[1]
                                end += 1
                                entity = text[start:end]
                                self.encode_per_entity(entity,entity_label,(start,end),ids,encoding)
                            else:
                                raise NotImplementedError()
                    
                    #{entity_label:[{'loc':[0,3]},{'loc':[5,9]}]}
                    #{entity_label:[{'start':0,'end':3},{'start':0,'end':3}]}
                    elif isinstance(v[0],dict):       
                         for vv in v:
                            if isinstance(self.loc_name, (tuple,list)): 
                                start = vv[self.loc_name[0]]
                                end = vv[self.loc_name[1]]
                            else:
                                start,end = vv[self.loc_name]
                            end += 1
                            entity = text[start:end]
                            self.encode_per_entity(entity,entity_label,(start,end),ids,encoding)
                                
                    
                    elif isinstance(v[0],int):
                        start,end = v
                        end += 1
                        entity = text[start:end]
                        self.encode_per_entity(entity,entity_label,(start,end),ids,encoding)
                        
                elif isinstance(v,dict):
                    for kk,vv in v.items():
                        #{entity_label:{entity:[...]}
                        if isinstance(vv,(tuple,list)):
                            #{entity_label:{entity:[[(0,3)],[(5,9)]]}
                            if isinstance(vv[0],(tuple,list)): 
                                for vvv in vv:
                                    start,end = vvv[1]
                                    end += 1
                                    entity = text[start:end]
                                    self.encode_per_entity(entity,entity_label,(start,end),ids,encoding)
                            
                            #{entity_label:{entity:[{'loc':(0,3)},{'loc':(5,9)}]}
                            #{entity_label:{entity:[{'start':0,'end':3},{'start':0,'end':3}]}
                            elif isinstance(vv[0],dict):
                                for vvv in vv:
                                    if isinstance(self.loc_name, (tuple,list)): 
                                        start = vvv[self.loc_name[0]]
                                        end = vvv[self.loc_name[1]]
                                    else:
                                        start,end = vvv[self.loc_name]
                                    end += 1
                                    entity = text[start:end]
                                    self.encode_per_entity(entity,entity_label,(start,end),ids,encoding)
                            
                            #{entity_label:{entity:(0,3)}
                            elif isinstance(vv[0],int):
                                start,end = vv
                                end += 1
                                entity = text[start:end]
                                self.encode_per_entity(entity,entity_label,(start,end),ids,encoding)
                            else:
                                raise NotImplementedError()
                        
                        #{entity_label:{entity:{'loc':...}}
                        elif isinstance(vv,dict): 
                            label = k
                            loc = vv[self.loc_name]
                            
                            #{entity_label:{entity:{'loc':[[0,3],[5,9]]}}
                            if isinstance(loc[0],(tuple,list)):
                                for l in loc:
                                    start,end = l[:2]
                                    end += 1
                                    entity = text[start:end]
                                    self.encode_per_entity(entity,entity_label,(start,end),ids,encoding)
                            
                            #{entity_label:{entity:{'loc':[3,0]}}
                            elif isinstance(loc[0],int):
                                start,end = loc
                                end += 1
                                entity = text[start:end]
                                self.encode_per_entity(entity,entity_label,(start,end),ids,encoding)
                            else:
                                raise NotImplementedError()
             
        #padding 
        if self.encoding in ["IO","BIO","BIOE","BIOES"]:
            #向右填充
            if len(ids) > self.max_length:
                ids = ids[:self.max_length]
                ids = [0] + ids + [0]
            else:
                ids = ids + [-100] * (self.max_length -len(ids)) # padding
                ids = [0] + ids + [0]

        return ids

    def encode(self,text):
        pass



if __name__ == "__main__":
    s = "事后【s 【e|0【e|1 环境 1】核查 0】】"
    res = extract_inline_entities(s)
    print(res)
