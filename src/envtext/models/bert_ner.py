from .bert_base import BertBase
from torch.nn.utils.rnn import pad_sequence
import torch #for torch.argmax torch.max
import torch.nn.functional as F
from torch import nn #for nn.Dropout nn.LSTM nn.Linear
from torchcrf import CRF
from transformers import BertPreTrainedModel,BertModel
# from ..tokenizers import WoBertTokenizer
from ..utils.metrics import metrics_for_ner
from .ner_base import NERBase
from multiprocessing import Pool


class BertCRF(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCRF, self).__init__(config)
        self.bert = BertModel(config)
        
        if hasattr(config,'max_length'):
            self.max_length = config.max_length


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

        if hasattr(config, "word2vec"): #使用word2vec增强bert输出得特征，解决边界问题
            self.word2vec = True
            from ..tokenizers import Word2VecTokenizer
            self.word2vec_tokenizer = Word2VecTokenizer(max_length=self.max_length,padding=True,encode_character=True)
            
        else:
            self.word2vec = False

        #半监督
        if hasattr(config,'semi_supervise'):
            self.semi_supervise = True
        else:
            self.semi_supervise = False
            
        if hasattr(config,'lstm'): #使用LSTM
            self.lstm = config.lstm
        else:
            self.lstm = True
            
        if self.lstm :    
            if self.word2vec:
                self.proj = nn.LSTM(
                    input_size= self.word2vec_tokenizer.vector_size,  # 768 + vector
                    hidden_size=config.hidden_size // 2,  # 768
                    batch_first=True,
                    num_layers=int(self.lstm),
                    dropout=0,  # 0.5
                    bidirectional=True
                    )
                # self.proj = nn.Linear(self.word2vec_tokenizer.vector_size, config.hidden_size)

            self.bilstm = nn.LSTM(
                input_size= config.hidden_size,  # 768 + vector
                hidden_size=config.hidden_size,  # 768
                batch_first=True,
                num_layers=int(self.lstm),
                dropout=0,  # 0.5
                bidirectional=True
                )

            self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        

        #条件随机场
        self.crf = CRF(config.num_labels, batch_first=True)
        if hasattr(config, "transition"):
            self.crf.transitions = nn.Parameter(torch.tensor(config.transition))

        self.init_weights()

    def forward(self, input_ids, input_text = None, vectors = None , token_type_ids=None, attention_mask=None, labels=None,
              position_ids=None, inputs_embeds=None, head_mask=None):
        outputs = self.bert(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
        
        #bert 输出
        sequence_output = outputs[0]
        B = sequence_output.shape[0]

        if self.word2vec and vectors is not None:
            pad_vectors = torch.tensor([self.word2vec_tokenizer.padding_values] * B).reshape(B,1,-1).to(self.device)
            vectors = torch.cat([pad_vectors,vectors[:,0:-2],pad_vectors],dim=1)
            # sequence_output = torch.cat([sequence_output,vectors],dim=-1)
            sequence_output = sequence_output + self.proj(vectors)[0]

        elif self.word2vec and input_text is not None:
            #B N
            vectors = self.word2vec_tokenizer(input_text,return_tensors="pt").to(self.device)
            pad_vectors = torch.tensor([self.word2vec_tokenizer.padding_values] * B).reshape(B,1,-1).to(self.device)
            vectors = torch.cat([pad_vectors,vectors[:,0:-2],pad_vectors],dim=1)
            vectors = vectors[:,:sequence_output.shape[1],:]
            # sequence_output = torch.cat([sequence_output,vectors],dim=-1)
            sequence_output = sequence_output + self.proj(vectors)[0]
                

        if self.lstm:
            lstm_output, _ = self.bilstm(sequence_output)
            ## 得到判别值
            logits = self.classifier(lstm_output)
        else:
            logits = self.classifier(sequence_output)
        
        ##去掉 [CLS] 和 [SEP] token
        # outputs = (logits[:,1:-1,:],)
        

        if labels is not None:
            #处理label长度和logits长度不一致的问题
            labels = labels.clone() #make a copy
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
            
            if self.semi_supervise:
                loss_mask = torch.logical_or(loss_mask,labels == 0)
                loss_mask[:,0] = True #<cls> token label. For the first time stamp of crf inputs must be true
                             
            labels[~loss_mask] = 0
            loss = self.crf(logits, labels.long() , mask = loss_mask.bool()) * (-1)
            outputs = (loss, logits)
        else:
            labels = torch.tensor(self.crf.decode(logits))
            outputs = (logits,labels)

        return outputs
    
class BertNER(NERBase,BertBase):
    '''
    Args:
       path `str`: 
           模型保存的路径
           
       config [Optional] `dict` :
           配置参数
   
   Kwargs:
       entities [Optional] `List[int]` or `List[str]`: 默认为None
           命名实体识别问题中实体的种类。
           命名实体识别问题中与entities/num_entities必设置一个，若否，则默认只有1个实体。

       num_entities [Optional] `int`: 默认None
           命名实体识别问题中实体的数量。
           命名实体识别问题中与labels/num_labels/entities必设置一个，若否，则默认只有1个实体。
           实体使用BIO标注，如果n个实体，则有2*n+1个label。
       
       ner_encoding [Optional] `str`: 默认BIO
           目前支持三种编码：
               BI
               BIO
               BIOES
          
           如果实体使用BIO标注，如果n个实体，则有2*n+1个label。
           eg:
               O: 不是实体
               B-entity1：entity1的实体开头
               I-entity1：entity1的实体中间
               B-entity2：entity2的实体开头
               I-entity2：entity2的实体中间
           
           
       crf  [Optional] `bool`:
           默认:True
           是否使用条件随机场
           
       lstm [Optional] `int`:
           默认:1,代表LSTM的层数为1
           是否使用lstm，设置为0，或None，或False不使用LSTM
        
       word2vec [Optional] `bool`:
            默认：False
            是否使用word2vec得结果增强bert的结果。
            这种方式会减慢速度，但是会增强模型对边界的识别。

       max_length [Optional] `int`: 默认：512
           支持的最大文本长度。
           如果长度超过这个文本，则截断，如果不够，则填充默认值。
    '''

    def initialize_bert(self,path = None,config = None,**kwargs):
        super().initialize_bert(path,config,**kwargs)
        self.model = BertCRF.from_pretrained(self.model_path,config = self.config)
        # if self.key_metric == 'validation loss':
        if self.num_entities == 1:
            self.set_attribute(key_metric = 'f1')
        else:
            self.set_attribute(key_metric = 'macro_f1')

        if self.model.word2vec and self.datasets:
            for k,v in self.datasets.items():
                v["vectors"] = self.model.word2vec_tokenizer(v["text"])
