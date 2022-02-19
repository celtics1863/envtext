from torch import nn
import torch
import torch.nn.functional as F
from .rnn_base import RNNBase
from ..tokenizers import Word2VecTokenizer,OnehotTokenizer

from ..utils.metrics import metrics_for_reg

class RNNREGModel(nn.Module):
    def __init__(self,length,token_size,hidden_size ,num_layers, embed_size = None  , model_name ='lstm'):
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
            nn.Linear(length*hidden_size*2,1)
            )
        
        self.loss_fn = nn.MSELoss()
        
    def forward(self,X,labels=None):
        X = self.embed_layer(X)
        X,hidden = self.rnn(X)
        logits = self.fc(X)
        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fn(logits.squeeze(),labels)
            outputs = (loss,) + outputs
        return outputs
    
class RNNSA(RNNBase):
    def initialize_rnn(self,path = None,config = None,**Kwargs):
        super().initialize_rnn(path,config,**Kwargs)
        self.model = RNNREGModel(self.config.max_length,
                 self.tokenizer.vector_size,
                 self.config.hidden_size,
                 self.config.num_layers,
                 self.config.embed_size,
                 self.config.model_name
                )
        
        self.model = self.model.to(self.device)
        if self.key_metric == 'validation loss':
            self.set_attribute(key_metric = 'rmse')

    def postprocess(self,text, logits, print_result = True ,save_result = True):
        logits = logits.squeeze()
        if print_result:
            self._report_per_sentence(text,logits)
        
        if save_result:
            self._save_per_sentence_result(text,logits)
            
            
    def _report_per_sentence(self,text,score):
        log = 'text:{} score: {:.4f} \n '.format(text,score)
        print(log)
        self.result[text].append(score)
    
    def _save_per_sentence_result(self,text,score):
        result = {
            'label':':.4f'.format(score)
        }
        self.result[text] = result
        
    def compute_metrics(self,eval_pred):
        dic = metrics_for_reg(eval_pred)
        return dic