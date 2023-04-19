from .bert_base import BertBase
import torch # for torch.topk torch.tanh
from torch import nn #for nn.linear
import torch.nn.functional as F
from transformers import BertPreTrainedModel,BertTokenizer,BertTokenizerFast,BertConfig,BertModel
# from ..tokenizers import WoBertTokenizer
from .sa_base import SABase

class BertREG(BertPreTrainedModel):
    def __init__(self, config):
        super(BertREG, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.regressor = nn.Linear(config.hidden_size, 1)
        self.loss = nn.MSELoss()
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
              position_ids=None, inputs_embeds=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        #使用[CLS]
        cls_output = outputs[0][:,0,:] 

        # 得到判别值
        logits = self.regressor(cls_output)
#         logits = (torch.tanh(logits)+1)/2

        if labels is not None:
            loss = self.loss(logits.squeeze(),labels)
            return (loss,logits)
        else:
            return (logits,)


class BertSA(SABase,BertBase):
    '''
    Bert情感分析/回归模型
    
    Args:
        path `str`: 默认：None
            预训练模型保存路径，如果为None，则从celtics1863进行导入预训练模型
        
        config [Optional] `dict` :
            配置参数
            
    Kwargs:
       max_length [Optional] `int`: 默认：128
           支持的最大文本长度。
           如果长度超过这个文本，则截断，如果不够，则填充默认值。
   '''
    def initialize_bert(self,path = None,config = None,**kwargs):
        super().initialize_bert(path,config,**kwargs)
        self.model = BertREG.from_pretrained(self.model_path)
        if self.key_metric == 'validation loss':
            self.set_attribute(key_metric = 'rmse')
        

    def initialize_config(self,*args,**kwargs):
        super().initialize_config(*args,**kwargs)

        self.set_attribute(model_name="bert_sa")


