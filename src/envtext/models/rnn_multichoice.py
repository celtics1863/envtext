from torch import nn
import torch
import torch.nn.functional as F
from .rnn_base import RNNBase
from ..tokenizers import Word2VecTokenizer,OnehotTokenizer
import numpy as np
from ..utils.metrics import metrics_for_cls_with_binary_logits
from .mc_base import MCBase

class RNNMultiCLSModel(nn.Module):
    def __init__(self,length,token_size,hidden_size ,num_layers, num_classes, onehot_embed = False, embed_size = None  , model_name ='lstm'):
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
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(length*hidden_size*2,num_classes)
            )
        
        self.loss = nn.BCEWithLogitsLoss()
        
    def forward(self,X,labels=None,**kwargs):
        if self.onehot_embed:
            X = self.embed_layer(X.long())
        else:
            X = self.proj_layer(X.float())

        X,hidden = self.rnn(X)
        logits = self.fc(X)
        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fn(logits,labels)
            outputs = (loss,) + outputs
        return outputs
    
class RNNMultiChoice(MCBase,RNNBase):
    def initialize_rnn(self,path = None,config = None,**Kwargs):
        super().initialize_rnn(path,config,**Kwargs)
        self.model = RNNMultiCLSModel(self.config.max_length,
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
            if self.num_labels == 1:
                self.set_attribute(key_metric = 'f1')
            else:
                self.set_attribute(key_metric = 'macro_f1')
    
    def postprocess(self,text, logits ,print_result = True, save_result = True):
        preds, = np.nonzero(logits > 0.5)
        if print_result:
            self._report_per_sentence(text,preds,logits)
        
        if save_result:
            self._save_per_sentence_result(text,preds,logits)
            