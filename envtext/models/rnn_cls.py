from torch import nn
import torch
import torch.nn.functional as F
from .rnn_base import RNNBase
from ..tokenizers import Word2VecTokenizer,OnehotTokenizer

from ..utils.metrics import metrics_for_cls

class RNNCLSModel(nn.Module):
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
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(length*hidden_size*2,num_classes)
            )
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self,X,labels=None):
        X = self.embed_layer(X)
        X,hidden = self.rnn(X)
        logits = self.fc(X)
        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fn(logits,labels)
            outputs = (loss,) + outputs
        return outputs
    
class RNNCLS(RNNBase):
    def __init__(self,model_name = 'lstm',labels=None, max_length = 128 , token_size = 512, hidden_size = 512 ,num_layers =3 ,embed_size = 512,token_method = 'word2vec'):
        super().__init__()
        if isinstance(labels,int):
            self.num_labels = labels
            self.labels = list(range(labels))
        elif isinstance(labels,list):
            self.num_labels = len(labels)
            self.labels = labels

        self.id2label = {idx:l for idx,l in enumerate(labels)}
            
        if token_method == 'word2vec':
            self.tokenizer = Word2VecTokenizer(max_length = 128,padding=True,truncation=True) 
            self.model = RNNCLSModel(max_length, self.tokenizer.vector_size, token_size , hidden_size ,num_layers, self.num_labels, None , model_name)
        else:
            self.tokenizer = OnehotTokenizer(max_length = 128,padding=True,truncation=True) 
            self.model = RNNCLSModel(max_length, self.tokenizer.vector_size, token_size , hidden_size ,num_layers, self.num_labels, embed_size ,model_name)
        
        self.model = self.model.to(self.device)
    
    def predict_per_sentence(self,text,topk = 3,save_result = True,print_result = True):
        tokens=torch.tensor(self.tokenizer.encode(text),device = self.device)
        with torch.no_grad():
            logits = F.softmax(self.model(tokens)[0],dim=-1)
        topk = topk if logits.shape[-1] > topk else logits.shape[-1]
        p,pred = torch.topk(logits,topk)
        if save_result:
            self._save_per_sentence_result(text,pred[0].clone().detach().cpu(),p[0].clone().detach().cpu())
            
        if print_result:
            self._report_per_sentence(text,pred[0].clone().detach().cpu(),p[0].clone().detach().cpu())
    
    def _report_per_sentence(self,text,pred,p):
        log = f'text:{text} \n'
        for i,j in zip(pred.cpu(),p.cpu()):
            self.result[text].append({'class':self.labels[i.item()],'p':j.item()})
            log += '\t pred_classes:{}, \t probability:{:.4f} \n'.format(self.labels[i.item()],j)
        print(log)
 
    def _save_per_sentence_result(self,text,pred,p):
        result = {}
        for topk,(i,j) in enumerate(zip(pred,p)):
            if topk == 0:
                result['label'] = self.id2label[i.item()]
                result['p'] = j.item()
            else:
                result[f'top{topk} label'] = self.id2label[i.item()]
                result[f'top{topk} p'] = j.item()
        
        self.result[text] = result
        
    def compute_metrics(self,eval_pred):
        dic = metrics_for_cls(eval_pred)
        if 'f1' in dic.keys():
            self.key_metric = 'f1'
        else:
            self.key_metric = 'macro_f1'
        return dic