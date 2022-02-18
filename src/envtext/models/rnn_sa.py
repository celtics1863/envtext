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
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_name = model_name
        
    def forward(self,X,labels=None):
        X = self.embed_layer(X)
        X,hidden = self.rnn(X)
        logits = self.fc(X)
        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fn(logits,labels)
            outputs = (loss,) + outputs
        return outputs
    
class RNNSA(RNNBase):
    def __init__(self,model_name = 'lstm', max_length = 128 , token_size = 512, hidden_size = 512 ,num_layers =3 ,embed_size = 512,token_method = 'word2vec'):
        super().__init__()
            
        if token_method == 'word2vec':
            self.tokenizer = Word2VecTokenizer(max_length = 128,padding=True,truncation=True) 
            self.model = RNNREGModel(max_length, self.tokenizer.vector_size, token_size , hidden_size ,num_layers, None , model_name)
        else:
            self.tokenizer = OnehotTokenizer(max_length = 128,padding=True,truncation=True) 
            self.model = RNNREGModel(max_length, self.tokenizer.vector_size, token_size , hidden_size ,num_layers, embed_size ,model_name)
        
        self.model = self.model.to(self.device)
            
    def predict_per_sentence(self,text, print_result = True ,save_result = True):
        tokens=torch.tensor(self.tokenizer.encode(text),device = self.device)
        with torch.no_grad():
            logits = self.model(tokens)[0]
        if print_result:
            self._report_per_sentence(text,logits[0])
        
        if save_result:
            self._save_per_sentence_result(text,logits[0])
            
            
    def _report_per_sentence(self,text,score):
        log = f'text:{text} score: {score.cpu().item()} \n '
        print(log)
        self.result[text].append(score.cpu().item())
    
    def _save_per_sentence_result(self,text,score):
        result = {
            'label':score
        }
        self.result[text] = result
        
    def compute_metrics(self,eval_pred):
        dic = metrics_for_reg(eval_pred)
        self.key_metric = 'rmse'
        return dic