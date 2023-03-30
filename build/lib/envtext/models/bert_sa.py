from .bert_base import BertBase
import torch # for torch.topk torch.tanh
from torch import nn #for nn.linear
import torch.nn.functional as F
from transformers import BertPreTrainedModel,BertTokenizer,BertTokenizerFast,BertConfig,BertModel
# from ..tokenizers import WoBertTokenizer
from ..utils.metrics import metrics_for_reg

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

        outputs = (logits,)
        if labels is not None:
            loss = self.loss(logits.squeeze(),labels)
            outputs = (loss,) + outputs
        return outputs


class BertSA(BertBase):
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