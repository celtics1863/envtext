from torch import nn
import torch
import torch.nn.functional as F
from .rnn_base import RNNBase
from ..tokenizers import Word2VecTokenizer,OnehotTokenizer

from ..utils.metrics import metrics_for_cls_with_binary_logits

class RNNMultiCLSModel(nn.Module):
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
        
        self.loss = nn.BCEWithLogitsLoss()
        
    def forward(self,X,labels=None):
        X = self.embed_layer(X)
        X,hidden = self.rnn(X)
        logits = self.fc(X)
        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fn(logits,labels)
            outputs = (loss,) + outputs
        return outputs
    
class RNNMultiChoice(RNNBase):
    def align_config(self):
        super().align_config()
        if self.labels:
            self.update_config(num_labels = len(self.labels))   
        elif self.num_labels:
            self.update_config(labels = list(range(self.num_labels)))
        else:
            self.update_config(num_labels = 1,
                         labels = ['LABEL_0'],
                         id2label = {0:'LABEL_0'},
                         label2id = {'LABEL_0':0},
                         )
    
    def initialize_rnn(self,path = None,config = None,**Kwargs):
        super().initialize_rnn(path,config,**Kwargs)
        self.model = RNNMultiCLSModel(self.config.max_length,
                         self.tokenizer.vector_size,
                         self.config.hidden_size,
                         self.config.num_layers,
                         self.config.num_labels,
                         self.config.embed_size,
                         self.config.model_name
                        )
        self.model = self.model.to(self.device)
        if self.key_metric == 'validation loss':
            if self.num_labels == 1:
                self.set_attribute(key_metric = 'f1')
            else:
                self.set_attribute(key_metric = 'macro_f1')
    

    def postprocess(self,text, logits ,print_result = True, save_result = True):
        preds = np.nonzero(logits > 0.5)
        if print_result:
            self._report_per_sentence(text,preds,logits)
        
        if save_result:
            self._save_per_sentence_result(text,preds.clone().detach().cpu(),logits[0][preds].clone().detach().cpu())
            
    def _report_per_sentence(self,text,preds,probs):
        log = f'text: {text}\n'
        for pred,prob in zip(preds,probs) :
            log += '\t prediction: {} \t ; probability : {:.4f}\n'.format(self.id2label[pred],prob)
            self.result[text].append((self.id2label[pred],prob))
        print(log)
 
    def _save_per_sentence_result(self,text,preds,probs):
        result = {}
        for idx,(pred,prob) in enumerate(zip(preds,probs)) :
            if idx == 0:
                result['label'] = self.id2label[pred]
                result['p'] = prob
            else:
                result[f'label_{idx+1}'] = self.id2label[pred]
                result[f'p_{idx+1}'] = prob
        
        self.result[text] = result
        
    def compute_metrics(self,eval_pred):
        dic = metrics_for_cls_with_binary_logits(eval_pred)
        return dic