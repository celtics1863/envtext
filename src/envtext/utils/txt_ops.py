def txt_generator(path):
    f = open(path,'r',encoding='utf-8')
    for idx,line in enumerate(f):
        yield line.strip()
        
def write_txts(path,txts):
    with open(path,"w",encoding='utf-8') as f:
        f.write('\n'.join(txts))
        f.close()
        
def read_txt(path):
    with open(path,"r",encoding = "utf-8") as f:
        lines = f.readlines()
        f.close()
    return lines

def read_txts(pattern,dir):
    files = [os.path.join(dir,file) for file in os.listdir(dir) if re.match(pattern,file) is not None]
    content = []
    for file in files:
        if file.find('.txt') == -1:
            continue
        try:
            txt = read_txt(file)
            content += txt
        except Exception as e:
            print(file)
            print(e)
    return content