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
                num_layers=int(self.lstm),
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
            lstm_output, _ = self.bilstm(sequence_output)
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
       entities [Optional] `List[int]` or `List[str]`: 默认为None
           命名实体识别问题中实体的种类。
           命名实体识别问题中与labels/num_labels/num_entities必填一个。
           实体使用BIO标注，如果n个实体，则有2*n+1个label。
           eg:
               O: 不是实体
               B-entity1：entity1的实体开头
               I-entity1：entity1的实体中间
               B-entity2：entity2的实体开头
               I-entity2：entity2的实体中间

       num_entities [Optional] `int`: 默认None
           命名实体识别问题中实体的数量。
           命名实体识别问题中与labels/num_labels/entities必填一个。
           实体使用BIO标注，如果n个实体，则有2*n+1个label。
   
       num_labels [Optional]`int`:
           默认:3
           标签类别
       
       labels [Optional] `List[int]` or `List[str]`: 默认None
            NER问题中标签的种类。
            分类问题中和num_labels必须填一个，代表所有的标签。
            默认为['O','B','I']
       
       crf  [Optional] `bool`:
           默认:True
           是否使用条件随机场
           
       lstm [Optional] `int`:
           默认:1,代表LSTM的层数为1
           是否使用lstm，设置为0，或None，或False不使用LSTM
        
       max_length [Optional] `int`: 默认：512
           支持的最大文本长度。
           如果长度超过这个文本，则截断，如果不够，则填充默认值。
    '''
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
            
        elif self.num_labels > 2:
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


    def initialize_bert(self,path = None,config = None,**kwargs):
        super().initialize_bert(path,config,**kwargs)
        self.model = BertCRF.from_pretrained(self.model_path,config = self.config)
        if self.key_metric == 'validation loss':
            if self.num_entities == 1:
                self.set_attribute(key_metric = 'f1')
            else:
                self.set_attribute(key_metric = 'macro_f1')
            
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
        dic = metrics_for_ner(eval_pred)
        return dic