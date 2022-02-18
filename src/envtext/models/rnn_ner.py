from torch import nn
import torch
import torch.nn.functional as F
from .rnn_base import RNNBase
from ..tokenizers import Word2VecTokenizer,OnehotTokenizer
from torchcrf import CRF

from ..utils.metrics import metrics_for_cls

class RNNNERModel(nn.Module):
    def __init__(self,length,token_size,hidden_size ,num_layers, num_classes, embed_size = None  , model_name ='lstm'):
        super().__init__()
        
        if embed_size and num_embeddings:
            self.embed_layer = nn.Embeddings(token_size,embed_size)
        else:
            self.embed_layer = nn.Identity()
            embed_size = token_size
            
        if model_name.lower() == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size ,num_layers,bias = True,batch_first = True,dropout = 0.1,bidirectional = True) 
        elif model_name.lower() == 'gru':
            self.rnn = nn.GRU(embed_size, hidden_size ,num_layers,bias = True,batch_first = True,dropout = 0.1,bidirectional = True) 
        elif model_name.lower() == 'rnn':
            self.rnn = nn.RNN(embed_size, hidden_size ,num_layers,bias = True,batch_first = True,dropout = 0.1,bidirectional = True) 
        
        self.fc = nn.Linear(hidden_size*2,num_classes)
        
        self.loss = CRF(num_classes,batch_first=True)
        
    def forward(self,X,labels=None):
        X = self.embed_layer(X)
        X,hidden = self.rnn(X)
        logits = self.fc(X)
        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fn(logits,labels)
            outputs = (loss,) + outputs
        return outputs
    
class RNNNER(RNNBase):
    def __init__(self,model_name = 'lstm',entities=None, max_length = 128 , hidden_size = 512 ,num_layers =3 ,embed_size = 512,token_method = 'word2vec'):
        '''
        Args:
            model_name [Optional] `str`:
                在'lstm','rnn','gru'三个中的一个
                
            entities [Optional] `int` or `List`:
                实体个数或实体名称
                默认为1个实体，标签为['B','I','O']
                
           max_length [Optional]`int`:
               文本序列的最大长度
               
           hidden_size [Optional] `int`:
               隐藏层大小
               
           num_layers [Optional] `int`:
               默认3，RNN的层数
               
           embed_size [Optional] `int`:
               向量嵌入维度，只有当token_method = 'onehot'时会使用
        '''
        super().__init__()
        if entities is None:
            self.num_labels = 3
            self.entities = ['B','I','O']
        elif isinstance(entities,int):
            self.num_labels = labels
            self.entities = list(range(entities))
        elif isinstance(entities,list):
            self.num_labels = len(labels)
            self.entities = entities
        
        
        if token_method == 'word2vec':
            self.tokenizer = Word2VecTokenizer(max_length = 128,padding=True,truncation=True) 
            self.model = RNNNERModel(max_length, self.tokenizer.vector_size, hidden_size ,num_layers, self.num_labels, None , model_name)
        else:
            self.tokenizer = OnehotTokenizer(max_length = 128,padding=True,truncation=True) 
            self.model = RNNNERModel(max_length, self.tokenizer.vector_size, hidden_size ,num_layers, self.num_labels, embed_size ,model_name)
            
        self.model = self.model.to(self.device)
        
    def predict_per_sentence(self,text, print_result = True ,save_result = True):
        tokens=torch.tensor(self.tokenizer.encode(text),device = self.device)
        with torch.no_grad():
            logits = F.softmax(self.model(tokens)[0],dim=-1)
        pred = torch.argmax(logits,dim=-1)
        
        if print_result:
            self._report_per_sentence(text,pred[0].clone().detach().cpu(),logits[0].clone().detach().cpu())
        
        if save_result:
            self._save_per_sentence_result(text,pred[0].clone().detach().cpu(),logits[0].clone().detach().cpu())
            
    def _report_per_sentence(self,text,pred,p):
        text = text.replace(' ','')
        log = f'text:{text}\n'
        for i,c in enumerate(pred):
            if c % 2 == 1: #start B
                class_id = (c.clone().cpu().item()-1) // 2
                idx = i+1
                while idx <= pred.shape[0]:
                    if idx == len(pred) or pred[idx].item() == 0: #stop I
                        local_p= torch.max(p[i:idx,:],dim=-1)[0]
                        s = f'\t pred_word:{text[i:idx]}, class: {self.entities[class_id]} ,probaility: '
                        for j in range(idx-i):
                            s += '{:.2f} '.format(local_p[j].item())
                        log += s
                        log += '\n'
                        break
                    elif pred[idx].item() == c + 1:
                        idx += 1
                    else:
                        break

        print(log)
     
    def _save_per_sentence_result(self,text,pred,p):
        text = text.replace(' ','')
        labels = []
        locs = []
        classes = []
        probs = []
        for i,c in enumerate(pred):
            if c % 2 == 1: #start B
                class_id = (c.clone().cpu().item()-1) // 2
                idx = i+1
                while idx <= pred.shape[0]:
                    if idx == len(pred) or pred[idx].item() == 0: #stop I
                        local_p= torch.max(p[i:idx,:],dim=-1)[0]
                        labels.append(text[i:idx])
                        locs.append([i,idx-1])
                        classes.append(self.entities[class_id])
                        s = ''
                        for j in range(idx-i):
                            s += '{:.2f} '.format(local_p[j].item())
                        probs.append(s)
                        break
                    elif pred[idx].item() == c + 1:
                        idx += 1
                    else:
                        break
        result = {}
        for idx,l,loc,c,p in enumerate(zip(labels,locs,classes,probs)):
            if idx == 0:
                result[f'label'] = l
                result[f'loc'] = loc
                result[f'class'] = c
                result[f'p'] = p
            else:
                result[f'label_{idx+1}'] = l
                result[f'loc_{idx+1}'] = loc
                result[f'class_{idx+1}'] = c
                result[f'p_{idx+1}'] = p

        self.result[text] = result
        
    def compute_metrics(self,eval_pred):
        self.key_metric='f1'
        return metrics_for_ner(eval_pred)