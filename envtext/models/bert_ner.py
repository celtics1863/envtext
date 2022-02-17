from .bert_base import BertBase
from torch.nn.utils.rnn import pad_sequence
import torch #for torch.argmax torch.max
import torch.nn.functional as F
from torch import nn #for nn.Dropout nn.LSTM nn.Linear
from torchcrf import CRF
from transformers import BertPreTrainedModel,BertModel
# from ..tokenizers import WoBertTokenizer
from ..utils.metrics import metrics_for_ner

class BertCRF(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCRF, self).__init__(config)
        self.bert = BertModel(config)
        
        if hasattr(config,'num_entities'):
            self.num_entities = config.num_entities
            self.num_labels = config.num_entities * 2 +1
        else:
            if hasattr(config,'num_labels'):
                self.num_labels = config.num_labels
                self.num_entities = config.num_labels // 2 -1
            else:
                self.num_entities = 1
                self.num_labels = 3
            
        if hasattr(config,'lstm'):
            self.lstm = config.lstm
        else:
            self.lstm = True
            
        if self.lstm :
            self.bilstm = nn.LSTM(
                input_size=config.hidden_size,  # 768
                hidden_size=config.hidden_size,  # 512
                batch_first=True,
                num_layers=self.lstm,
                dropout=0,  # 0.5
                bidirectional=True
                )

            self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        
        if hasattr(config,'crf'):
            self.crf = config.crf
        else:
            self.crf = True
            
        
        if self.crf:
            self.crf = CRF(config.num_labels, batch_first=True)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
              position_ids=None, inputs_embeds=None, head_mask=None):
        outputs = self.bert(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
        
        ##去掉 [CLS] 和 [SEP] token
        sequence_output = outputs[0][:,1:-1,:]
        
        if self.lstm:
            lstm_output, _ = self.bilstm(padded_sequence_output)
            ## 得到判别值
            logits = self.classifier(lstm_output)
        else:
            logits = self.classifier(sequence_output)
        
        outputs = (logits,)
        if labels is not None:
            #处理label长度和logits长度不一致的问题
            B,L,C,S = logits.shape[0],logits.shape[1],logits.shape[2],labels.shape[1]
            ## print(B,L,C,S)
            if S > L :
                labels = labels[:,:L]
                loss_mask = labels.gt(-1)
            elif S < L:
                pad_values = torch.tensor([[-100]],device = logits.device).repeat(B,L-S)
                labels = torch.cat([labels,pad_values],dim=1)
                loss_mask = labels.gt(-1)
            else:
                loss_mask = labels.gt(-1)
            
            if self.crf:
                labels[~loss_mask] = 0
                loss = self.crf(logits, labels.long(), mask = loss_mask.bool()) * (-1)
            else:
                loss = self.loss_fn(logits.reshape(-1,self.num_labels),labels.reshape(-1))
            
            outputs = (loss,) + outputs

        return outputs
    
class BertNER(BertBase):
    '''
    Args:
       path `str`: 
           模型保存的路径
           
       config [Optional] `dict` :
           配置参数
   
   Kwargs:
       num_entities [Optional] `int`:
           实体类别（注意是实体类别，不是转换为标签后（B、I、O）的类别数量
           num_entities 和 num_labels只需要设置一个即可
   
       num_labels [Optional]`int`:
           默认:3
           标签类别
       
       crf  [Optional] `bool`:
           默认:True
           是否使用条件随机场
           
       lstm [Optional] `int`:
           默认:1
           是否使用lstm
    '''
    def initialize_bert(self,path = None,**kwargs):
        self.update_model_path(path)
        self.update_config(kwargs)
        self.model = BertCRF.from_pretrained(self.model_path,config = self.config)

    def predict_per_sentence(self,text,print_result = True, save_result = True):
        tokens=self.tokenizer.encode(text, return_tensors='pt',add_special_tokens=True).to(self.model.device)
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
                        s = f'\t pred_word:{text[i:idx]}, entity: {self.entities[class_id]} ,probaility: '
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
        for idx,(l,loc,c,p) in enumerate(zip(labels,locs,classes,probs)):
            if idx == 0:
                result[f'label'] = l
                result[f'loc'] = loc
                result[f'entity'] = c
                result[f'p'] = p
            else:
                result[f'label_{idx+1}'] = l
                result[f'loc_{idx+1}'] = loc
                result[f'entity_{idx+1}'] = c
                result[f'p_{idx+1}'] = p
        

        self.result[text] = result
        
    def compute_metrics(self,eval_pred):
        self.key_metric='f1'
        return metrics_for_ner(eval_pred)