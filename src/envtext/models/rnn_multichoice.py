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
    def __init__(self,model_name = 'lstm',labels=None, max_length = 128 , hidden_size = 512 ,num_layers =3 ,embed_size = 512,token_method = 'word2vec'):
        super().__init__()
        if labels is None:
            self.num_labels = 1
            self.labels = ['0']
        elif isinstance(labels,int):
            self.num_labels = labels
            self.labels = list(range(labels))
        elif isinstance(labels,list):
            self.num_labels = len(labels)
            self.labels = labels

        self.id2label = {idx:l for idx,l in enumerate(self.labels)}

        if token_method == 'word2vec':
            self.tokenizer = Word2VecTokenizer(max_length = 128,padding=True,truncation=True) 
            self.model = RNNMultiCLSModel(max_length, self.tokenizer.vector_size, hidden_size ,num_layers, self.num_labels, None , model_name)
        else:
            self.tokenizer = OnehotTokenizer(max_length = 128,padding=True,truncation=True) 
            self.model = RNNMultiCLSModel(max_length, self.tokenizer.vector_size, hidden_size ,num_layers, self.num_labels, embed_size ,model_name)
        
        self.model = self.model.to(self.device)
            
    def predict_per_sentence(self,text, print_result = True ,save_result = True):
        tokens=torch.tensor(self.tokenizer.encode(text),device = self.device)
        with torch.no_grad():
            logits = self.model(tokens)[0]
        logits = self.model(tokens)[0]
        preds = torch.nonzero(logits[0] > 0.5)
        if print_result:
            self._report_per_sentence(text,preds.clone().detach().cpu(),logits[0][preds].clone().detach().cpu())
        
        if save_result:
            self._save_per_sentence_result(text,preds.clone().detach().cpu(),logits[0][preds].clone().detach().cpu())
            
    def _report_per_sentence(self,text,preds,probs):
        log = f'text: {text}\n'
        for pred,prob in zip(preds,probs) :
            log += '\t prediction: {} \t ; probability : {:.4f}\n'.format(self.id2label[pred.item()],prob.item())
            self.result[text].append((self.id2label[pred.item()],prob.item()))
        print(log)
 
    def _save_per_sentence_result(self,text,preds,probs):
        result = {}
        for idx,(pred,prob) in enumerate(zip(preds,probs)) :
            if idx == 0:
                result['label'] = self.id2label[pred.item()]
                result['p'] = prob.item()
            else:
                result[f'label_{idx+1}'] = self.id2label[pred.item()]
                result[f'p_{idx+1}'] = prob.item()
        
        self.result[text] = result
        
    def compute_metrics(self,eval_pred):
        dic = metrics_for_cls_with_binary_logits(eval_pred)
        self.key_metric = 'macro_f1'
        return dic