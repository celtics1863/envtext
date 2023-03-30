import json
from ntpath import join
import os
from sys import path
from typing import Match, Pattern
import re

def read_json(path):
    f = open(path, 'r', encoding='utf-8')
    d = json.load(f)
    return d

def read_jsonL(path):
    f = open(path, 'r', encoding='utf-8')
    for idx,line in enumerate(f):
        js = json.loads(line)
        yield js

def read_jsons(pattern,dir):
    files = [os.path.join(dir,file) for file in os.listdir(dir) if re.match(pattern,file) is not None]
    content = []
    for file in files:
        if file.find('.json') == -1:
            continue
        try:
            js = read_json(file)
            if isinstance(js,dict):
                for k,v in js.items(): 
                    content.append(v)
            elif isinstance(js,list):
                content += js
        except Exception as e:
            print(file)
            print(e)
    
    return content
    # files = []
def write_jsons(dir,list_of_dic,max_num = 1000):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for i in range(0,len(list_of_dic),max_num):
        content = {}
        for j in range(i,i+max_num):
            if j < len(list_of_dic):
                content[j] = list_of_dic[j]
            else:
                break
        write_json(os.path.join(dir,str(i)+'.json'),content)
        

def write_json(path,dic):
    if not os.path.exists(os.path.dirname(os.path.realpath(path))):
        os.makedirs(os.path.dirname(path))
        
    with open(path,'w+',encoding="utf-8") as f:
        json.dump(dic,f,ensure_ascii=False,indent=4)
        f.close()

def update_json(path,dic):
    if os.path.exists(path):
        with open(path,'r',encoding="utf-8") as f:
            js = json.load(f)
            js.update(dic)
            f.close()
        
        with open(path,'w',encoding="utf-8") as f:
            json.dump(js,f,ensure_ascii=False,indent=4)
            f.close()
    else:
        with open(path,'w',encoding="utf-8") as f:
            json.dump(dic,f,ensure_ascii=False,indent=4)
            f.close()


