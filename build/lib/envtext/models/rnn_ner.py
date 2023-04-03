from torch import nn
import torch
import torch.nn.functional as F
from .rnn_base import RNNBase
from ..tokenizers import Word2VecTokenizer,OnehotTokenizer
from torchcrf import CRF
from collections import Counter
from ..utils.metrics import metrics_for_ner
from .ner_base import NERBase

class RNNNERModel(nn.Module):
    def __init__(self,length,token_size,hidden_size ,num_layers, num_classes, onehot_embed = False ,embed_size = None,  model_name ='lstm'):
        super().__init__()
        self.onehot_embed = onehot_embed

        if onehot_embed:
            self.embed_layer = nn.Embedding(token_size,embed_size)
        else:
            if embed_size:
                self.proj_layer = nn.Linear(token_size, embed_size)
            else:
                self.proj_layer = nn.Identity()
                embed_size = token_size

        if model_name.lower() == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size ,num_layers,bias = True,batch_first = True,dropout = 0.1,bidirectional = True) 
        elif model_name.lower() == 'gru':
            self.rnn = nn.GRU(embed_size, hidden_size ,num_layers,bias = True,batch_first = True,dropout = 0.1,bidirectional = True) 
        elif model_name.lower() == 'rnn':
            self.rnn = nn.RNN(embed_size, hidden_size ,num_layers,bias = True,batch_first = True,dropout = 0.1,bidirectional = True) 
        
        self.fc = nn.Linear(hidden_size*2,num_classes)
        
        self.crf = CRF(num_classes,batch_first=True)
        
    def forward(self,X,labels=None,**kwargs):
        if self.onehot_embed:
            X = self.embed_layer(X.long())
        else:
            X = self.proj_layer(X.float())

        X,hidden = self.rnn(X)

        logits = self.fc(X)

        outputs = (logits,)
        if labels is not None:
            #处理label长度和logits长度不一致的问题
            B,L,C,S = logits.shape[0],logits.shape[1],logits.shape[2],labels.shape[1]
            if S > L :
                labels = labels[:,:L]
                loss_mask = labels.gt(-1)
            elif S < L:
                pad_values = torch.tensor([[-100]],device = logits.device).repeat(B,L-S)
                labels = torch.cat([labels,pad_values],dim=1)
                loss_mask = labels.gt(-1)
            else:
                loss_mask = labels.gt(-1)
            
            labels[~loss_mask] = 0
            loss = self.crf(logits, labels.long(), mask = loss_mask.bool()) * (-1)

            outputs = (loss,) + outputs
        elif self.crf:
            labels = self.crf.decode(logits)      
            outputs = outputs + (labels,)
            
        return outputs
    
class RNNNER(NERBase,RNNBase):
    def initialize_rnn(self,path = None,config = None,**kwargs):
        super().initialize_rnn(path,config,**kwargs)


        self.model = RNNNERModel(self.config.max_length,
                         self.tokenizer.vector_size,
                         self.config.hidden_size,
                         self.config.num_layers,
                         self.config.num_labels,
                         self.config.onehot_embed,
                         self.config.embed_size,
                         self.config.model_name
                        )
        
        self.model = self.model.to(self.device)
        if self.key_metric == 'validation loss':
            if self.num_entities == 1:
                self.set_attribute(key_metric = 'f1')
            else:
                self.set_attribute(key_metric = 'macro_f1')
        

        if self.datasets and self.token_method in ["onehot","word2vec","tf-idf"]:
            #调整tokenizer：
            if self.token_method == "word2vec":
                self.tokenizer.encode_character = False

            #统一数据集格式
            import re
            for k,v in self.datasets.items():
                v["text"] =  [re.sub("\s","",vv) for vv in v["text"]]
                # texts = ["".join(self.tokenizer._jieba_tokenizer(re.sub("\s","",vv))) for vv in v["text"]] #去除white-space token

                labels = []
                for t,l in zip(v["text"],v["label"]):
                    start = 0
                    l = l[1:-1]
                    label = []
                    for word in self.tokenizer._jieba_tokenizer(t):
                        end = start + len(word)
                        word_label = l[start:end]
                        word_label_counter = Counter(word_label)

                        start = end
                        if self.ner_encoding == "BI":
                            new_label,_ = word_label_counter.most_common(1)[0]
                            label.append(new_label)
                        elif self.ner_encoding == "BIO":
                            for wl in word_label:
                                if wl % 2 == 1:
                                    label.append(wl)
                                    break
                            else:
                                for wl in word_label:
                                    if wl % 2 == 0:
                                        label.append(wl)
                                        break
                                else:
                                    label.append(0)
                        elif self.ner_encoding == "BIOES":
                            #S
                            if (word_label[0] % 4 == 1 and word_label[-1] % 4 == 3):
                                label.append(word_label[0]//4 * 4 + 3)
                            elif sum([True for wl in word_label if wl % 4 ==0]):
                                for wl in word_label:
                                    if wl % 4 == 0:
                                        label.append(wl)
                            #B
                            elif word_label[0] % 4 == 1:
                                label.append(word_label[0])
                            #E
                            elif word_label[-1] % 4 == 1:
                                label.append(word_label[-1])
                            #I
                            elif sum([True for wl in word_label if wl % 4 ==2]):
                                for wl in word_label:
                                    if wl % 4 == 2:
                                        label.append(wl)
                            #O
                            else:
                                label.append(0)
                    # print(t,label,l)
                    #归一化
                    if len(label) < self.max_length:
                        label += [-100] * (self.max_length - len(label))               
                    else:
                        label = label[:self.max_length]

                    labels.append(label)

                v["label"] = labels 
                
    def preprocess(self,text):
        import re
        text = re.sub("\s", "",text)
        return text
        
    def postprocess(self,text,logits, print_result = True, save_result = True,return_result = False,save_vis = None):
        logits = torch.tensor(logits)
        logits = F.softmax(logits,dim=-1)

        text = list(self.tokenizer._jieba_tokenizer(text))

        if self.viterbi:
            labels,locs,classes,probs = self._viterbi_decode(text,logits)
        else:
            pred = torch.argmax(logits,dim=-1)
            labels,locs,classes,probs = self._decode(text,pred,logits)
        
        if print_result:
            try:
                self._visualize(text,classes,locs,save_vis)
            except:
                self._report_per_sentence(text,labels,classes,probs)
        
        if save_result:
            self._save_per_sentence_result(text,labels,locs,classes,probs)
            
        if return_result:
            return self._convert_label2result(text,classes,locs)