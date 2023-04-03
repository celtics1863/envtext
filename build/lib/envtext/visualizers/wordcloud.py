from tqdm import tqdm
from ..files import Config
from collections import Counter,OrderedDict,defaultdict
import math
from queue import PriorityQueue
import wordcloud
from PIL import Image 
import os

eps = 1e-7


class WordCloud:
    '''
    生成词云
    '''
    def __init__(self,texts,textsB = None, method = 'tf-idf', font = '楷体',max_words = 1000,**kwargs):
        '''
        Args:
        texts:
           支持格式 文本或词频
                文本：
                    str
                    List[str]
                    Dict[str]
                    Dict[List[str]]

                词频：
                    Dict[int]
            
        
       textsB:
           只支持输入文本
           第二个文本，默认为空。
           如果传入则可以用于与第一个文本对比，只使用两个文本tf-idf差异最大的词，画出词语云。
       
       method `Choice`:
           默认：
               对文本格式：tf-idf
               对词频格式：tf
       
        '''
        self.method = method
        self.config = kwargs
        self.config['font_path'] = Config.fonts[font] if font in Config.fonts else font
        self.config['max_words'] = max_words
        
        self.stop_words = config.stop_words
        
        self._cache = {}
        
        if texts is not None and textsB is None:
            texts,format = self._align_text_format(texts)
            if format == 'text':
                self.method = 'tf-idf'
                self.tf,self.idf = self.get_tf_idf(texts)
                self.generate_wordcloud()
                #self.generate_wordcloud(font = font,max_words = max_words,**kwargs)
            elif format == 'freq':
                self.method = 'tf'
                self.tf = texts
                self.generate_wordcloud()
        
        elif texts is not None and textsB is not None:
            textsA,format = self._align_text_format(texts)
            textsB,format = self._align_text_format(textsB)
            self.tf_A,self.idf_A = self.get_tf_idf(texts)
            self.tf_B,self.idf_B = self.get_tf_idf(textsB)
            self.generate_compare_wordcloud()
            self.method = 'compare'
            
    def plot(self):
        if isinstance(self.wc,list):
            img_list = [Image.fromarray(wc.to_array()) for wc in self.wc]
            for img in img_list:
                try:
                    from IPython.display import display
                    display(img,raw = True)
                except:
                    img.show()
        else:
            img = Image.fromarray(self.wc.to_array())
            try:
                from IPython.display import display
                display(img,raw = True)
            except:
                img.show()

        
    def generate_wordcloud(self):
        method = self.method
        
        if method == 'compare':
            self.generate_compare_wordcloud()
            return
        elif method == 'tf-idf':
            counter = {}
            if not hasattr(self,'idf') or not hasattr(self,'tf'):
                assert 0,'没有self.idf,或self.tf,请初始化时输入文本'
                
            tf_idf = self.tf_idf
            tf = self.tf

            while not tf_idf.empty() and len(counter) < self.config['max_words']:
                v,k = tf_idf.get_nowait()
                counter[k] = tf[k]

        elif method == 'tf':
            if not hasattr(self,'tf'):
                assert 0,'没有self.tf,请初始化'
                
            counter = self.tf
            
        self.wc = wordcloud.WordCloud(**self.config)

        self.wc.generate_from_frequencies(counter)
        self.plot()
        
    def _align_method(self,method):
        if method in ['freq','frequence','tf']:
            return 'tf'
        elif method in ['tf-idf']:
            return 'tf-idf'
    
    def add_mask(self,mask_path):
        img = Image.open(mask_path)
        self.config ['mask'] = img
        self.generate_wordcloud()
        
    def add_stopwords(self,stop_words):
        self.config['stopwords'] = stop_words
        for word in stop_words:
            self.stop_words[word] = True
        self.generate_wordcloud()
        
    def set_background_color(self,background_color = 'black'):
        self.config ['background_color'] = background_color
        self.generate_wordcloud()
    
    def set_shape(self,shape):
        '''
        shape: (width,height)
        '''
        width,height = shape
        self.config['width'] = width
        self.config['height'] = height
        self.generate_wordcloud()
    
    def set_margin(self,margin):
        self.config['margin'] = margin
        self.generate_wordcloud()
    
    def set_is_repeat(self,is_repeat):
        if is_repeat:
            self.config['repeat'] = True
        else:
            self.config['repeat'] = False
        
        self.generate_wordcloud()
    
    @property
    def wordcloud(self):
        if isinstance(self.wc,list):
            array_list = []
            for wc in self.wc:
                array_list.append(wc.to_array())
            return array_list
        else:
            return self.wc.to_array()
    
    def save_wordcloud(self,save_file):
        if save_file is not None:
            base_dir = os.path.dirname(os.path.normpath(save_file))
            if base_dir and (not os.path.exists(base_dir)):
                os.makedirs(base_dir)
            
            if isinstance(self.wc,list):
                basename = os.path.basename(os.path.normpath(save_file))
                name,suffix = basename.split('.')
                for idx,wc in enumerate(self.wc):
                    wc.to_file(os.path.join(base_dir,f'{name}_{idx}.{suffix}'))
            else:
                self.wc.to_file(save_file)
        
    def _align_text_format(self,texts):
        format = 'text'
        if isinstance(texts,str):
            lines = texts.split('。')
        elif isinstance(texts,dict):
            lines = []
            for k,v in texts.items():
                if len(v) == 0:
                    continue
                if isinstance(v,str):
                    lines.append(v)
                elif isinstance(v,list) and isinstance(v[0],str):
                    lines += v
                elif isinstance(v,int):
                    format = 'freq'
                    
                else:
                    NotImplemented

        elif isinstance(texts,list):
            lines = texts
        else:
            import pandas as pd
            if isinstance(texts,pd.Series):
                lines = texts.values.tolist()
            else:
                raise NotImplemented
        
        return lines,format
    
    
    def generate_compare_wordcloud(self):
        tfA,idfA,tf_idfA = self.tf_A,self.idf_A,self.tf_idf_A
        tfB,idfB,tf_idfB = self.tf_B,self.idf_B,self.tf_idf_B
        
        compare_dict = OrderedDict()
        while not tf_idfA.empty():
            v,k = tf_idfA.get_nowait()
            if k not in compare_dict:
                compare_dict[k] = [0,0]
            
            compare_dict[k][0] = -v

        while not tf_idfB.empty():
            v,k = tf_idfB.get_nowait()
            if k not in compare_dict:
                compare_dict[k] = [0,0]
            
            compare_dict[k][1] = -v
        
        for k,v in compare_dict.items():
            if v[0] != 0:
                tf_idfA.put_nowait((-v[0],k))
            
            if v[1] != 0:
                tf_idfB.put_nowait((-v[1],k))
            
        delta_A = PriorityQueue()
        for k,v in compare_dict.items():
            if v[0]-v[1] > 0:
                delta_A.put_nowait((-(v[0]-v[1]),k))
        
        delta_B = PriorityQueue()
        for k,v in compare_dict.items():
            if v[1]-v[0] > 0:
                delta_B.put_nowait((-(v[1]-v[0]),k))

        counter_A = {}
        while not delta_A.empty():
            v,k = delta_A.get_nowait()
            counter_A[k] = tfA[k]
             
        counter_B,old_tf_idf = {},{}
        while not delta_B.empty():
            v,k = delta_B.get_nowait()
            counter_B[k] = tfB[k]
        
        wcA = wordcloud.WordCloud(**self.config)
        wcA.generate_from_frequencies(counter_A)
        
        wcB = wordcloud.WordCloud(**self.config)
        wcB.generate_from_frequencies(counter_B)
        
        
        self.wc = [wcA,wcB]
        
        self.plot()
       
    @property
    def tf_idf_A(self):
        if hasattr(self,'tf_A') and hasattr(self,'idf_A'):
            return self._to_tfidf(self.tf_A,self.idf_A)
        else:
            return None

    @property
    def tf_idf_B(self):
        if hasattr(self,'tf_B') and hasattr(self,'idf_B'):
            return self._to_tfidf(self.tf_B,self.idf_B)
        else:
            return None

    @property
    def tf_idf(self):
        if hasattr(self,'tf') and hasattr(self,'idf'):
            return self._to_tfidf(self.tf,self.idf)
        else:
            return None
    
    
    def save_tf_idf(self,path):
        import pandas as pd
        df = defaultdict(dict)
        if hasattr(self,'tf') and hasattr(self,'idf'):
            tf_idf = self.tf_idf
            while not tf_idf.empty():
                v,k = tf_idf.get_nowait()
                df[k]['tfidf'] = -v
                df[k]['tf'] = self.tf[k]
                df[k]['idf'] = self.idf[k]

        if hasattr(self,'tf_A') and hasattr(self,'idf_A'):
            tf_idf = self.tf_idf_A
            while not tf_idf.empty():
                v,k = tf_idf.get_nowait()
                df[k]['tfidf_A'] = -v
                df[k]['tf_A'] = self.tf_A[k]
                df[k]['idf_A'] = self.idf_A[k]
        
        if hasattr(self,'tf_B') and hasattr(self,'idf_B'):
            tf_idf = self.tf_idf_B
            while not tf_idf.empty():
                v,k = tf_idf.get_nowait()
                df[k]['tfidf_B'] = -v
                df[k]['tf_B'] = self.tf_B[k]
                df[k]['idf_B'] = self.idf_B[k]
                
        df = pd.DataFrame(df).transpose()
        df.to_csv(path,encoding = 'utf_8_sig')

    def _to_tfidf(self,tf,idf):
        tf_idf = PriorityQueue()

        for k,v in tf.items():
            if (k in idf and k in tf) and k not in self.stop_words and len(k) > 1:
                tf_idf.put_nowait((-idf[k]*tf[k],k))
        
        return tf_idf
    
    def get_tf_idf(self,texts):
        import jieba
        jieba.load_userdict(config.env_vocab)
        tf,idf = Counter(),Counter()

        bar = tqdm(texts)
        bar.set_description('正在分词：')
        for line in bar:
            tf.update(jieba.lcut(line))
            idf.update(set(jieba.lcut(line)))

        idf = {k:math.log(len(texts)/v + eps) for k,v in idf.items()}
        
        return tf,idf
