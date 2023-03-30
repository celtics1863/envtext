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
        
        self.crf = CRF(num_classes,batch_first=True)
        
    def forward(self,X,labels=None):
        X = self.embed_layer(X)
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
            
        return outputs
    
class RNNNER(RNNBase):
    def initialize_rnn(self,path = None,config = None,**Kwargs):
        super().initialize_rnn(path,config,**Kwargs)
        self.model = RNNNERModel(self.config.max_length,
                         self.tokenizer.vector_size,
                         self.config.hidden_size,
                         self.config.num_layers,
                         self.config.num_labels,
                         self.config.embed_size,
                         self.config.model_name
                        )
        self.model = self.model.to(self.device)
        if self.key_metric == 'validation loss':
            if self.num_entities == 1:
                self.set_attribute(key_metric = 'f1')
            else:
                self.set_attribute(key_metric = 'macro_f1')
        
    def align_config(self):
        super().align_config()
        if self.entities:
            if not self.num_entities:
                num_entities = len(self.entities)
            else:
                num_entities = self.num_entities
            
            num_labels = len(self.entities) * 2 +1

            if not self.labels or len(self.labels) != num_labels:
                labels = ['O']
                for e in self.entities:
                    labels.append(f'B-{e}')
                    labels.append(f'I-{e}')
            else:
                labels = self.labels

            self.update_config(num_entities = num_entities,
                         num_labels = num_labels,
                         labels = labels)   
            
        elif self.num_entities:

            entities = [f'entity-{i}' for i in range(self.num_entities)]
            
            num_labels = self.num_entities * 2 +1
            
            if not self.labels or len(self.labels) != num_labels:
                labels = ['O']
                for e in entities:
                    labels.append(f'B-{e}')
                    labels.append(f'I-{e}')
            else:
                labels = self.labels
            
            self.update_config(
                         entities = entities,
                         num_labels = num_labels,
                         labels = labels
                         )
        
        elif self.labels:
            num_labels = len(self.labels)
            if num_labels % 2 == 0:
                assert 0,"在NER任务中，配置参数labels的长度必须是奇数，可以通过set_attribute()或者初始化传入entities,num_entities或正确的labels进行修改"
            num_entities = num_labels//2
            entities = [f'entity-{i}' for i in range(num_entities)]
            self.update_config(
                     entities = entities,
                     num_labels = num_labels,
                     num_entities = num_entities
                     )
            
        elif self.num_labels:
            if self.num_labels % 2 == 0:
                assert 0,"在NER任务中，配置参数num_labels必须是奇数，可以通过set_attribute()或者初始化传入entities,num_entities或正确的num_labels进行修改"
            num_entities = num_labels//2
            entities = [f'entity-{i}' for i in range(num_entities)]
            labels = ['O']
            for e in entities:
                labels.append(f'B-{e}')
                labels.append(f'I-{e}')
                
            self.update_config(
                         entities = entities,
                         num_entities = num_entities,
                         labels = labels
                         )

        else:
            entities = ['entity']
            num_entities = 1
            num_labels = 3
            labels = ['O','B','I']
            
            self.update_config(
                 entities = entities,
                num_entities = num_entities,
                num_labels = num_labels,
                labels = labels
            )
        
            
    def postprocess(self,text,logits,print_result = True, save_result = True):
        logits = torch.tensor(logits)
        logits = F.softmax(logits,dim=-1)
        pred = torch.argmax(logits,dim=-1)
        
        if print_result:
            self._report_per_sentence(text,pred,logits)
        
        if save_result:
            self._save_per_sentence_result(text,pred,logits)
            
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
        dic = metrics_for_ner(eval_pred)
        return dic