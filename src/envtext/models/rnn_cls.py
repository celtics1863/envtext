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
    def align_config(self):
        super().align_config()
        if self.labels:
            self.update_config(num_labels = len(self.labels))   
        elif self.num_labels:
            self.update_config(labels = list(range(self.num_labels)))
        else:
            self.update_config(num_labels = 2,
                         labels = ['LABEL_0','LABEL_1'],
                         id2label = {0:'LABEL_0',1:'LABEL_1'},
                         label2id = {'LABEL_0':0,'LABEL_1':1},
                         )

    def initialize_rnn(self,path = None,config = None,**kwargs):
        super().initialize_rnn(path,config,**kwargs)
        self.model = RNNCLSModel(self.config.max_length,
                         self.tokenizer.vector_size,
                         self.config.hidden_size,
                         self.config.num_layers,
                         self.config.num_labels,
                         self.config.embed_size,
                         self.config.model_name
                        )
        self.model = self.model.to(self.device)
        
        if self.key_metric == 'validation loss' or self.key_metric == 'loss':
            if self.num_labels == 1:
                self.set_attribute(key_metric = 'f1')
            else:
                self.set_attribute(key_metric = 'macro_f1')
    
    
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
        return dic