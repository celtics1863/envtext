from torch import nn
import torch
import torch.nn.functional as F
from .rnn_base import RNNBase
from ..tokenizers import Word2VecTokenizer,OnehotTokenizer
import math
from .cls_base import CLSBase


class CNNDiscourseCLSModel(nn.Module):
    def __init__(self,length,
                        token_size,
                        num_classes, 
                        onehot_embed = False, 
                        embed_size = None  , 
                        kernel_size = 3,
                        **kwargs
                        ):
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
            
        self.max_length = 2 ** (math.ceil(math.log2(length)))
        self.num_layers = math.ceil(math.log2(length))

        if kernel_size % 2 == 0:
            assert 0,"kernel_size 必须是奇数"

        #CNN层
        #这里使用了dilate卷积，可以参考相关论文
        #减少了参数量，根据试验结果，效果要比普通卷积好很多
        self.cnn_stack = nn.ModuleList()
        dilation = kernel_size // 2
        pad_size = kernel_size // 2 + 2 ** dilation - 2 * dilation 

        for i in range(self.num_layers):
            self.cnn_stack.append(nn.Conv1d(embed_size, embed_size, kernel_size = kernel_size, padding= pad_size, dilation=dilation))
            self.cnn_stack.append(nn.ReLU())

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_size,num_classes)
            )
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self,X,labels=None,**kwargs):
        if self.onehot_embed:
            X = self.embed_layer(X.long())
        else:
            X = self.proj_layer(X.float())

        length = X.shape[1]

        #padding
        pad_sum = self.max_length - length 
        X = F.pad(X,(0,0,pad_sum,0))

        #permute
        X = X.permute(0,2,1)

        #run network
        for module in self.cnn_stack:
            X = module(X)
        
        #pool
        X,_ = X.max(dim=-1)

        #get logits
        logits = self.fc(X)

        if labels is not None:
            loss = self.loss_fn(logits,labels)
            outputs = (loss,logits)
            return outputs
        else:
            return (logits,)
    
class CNNDiscourseCLS(CLSBase,RNNBase):
    def initialize_rnn(self,path = None,config = None,**kwargs):
        super().initialize_rnn(path,config,**kwargs)
        self.model = CNNDiscourseCLSModel(self.config.max_length,
                        self.tokenizer.vector_size,
                         self.config.num_labels,
                         self.config.onehot_embed,
                         self.config.embed_size,
                         self.config.kernel_size,
                        )

        self.model = self.model.to(self.device)
        
        if self.key_metric == 'validation loss' or self.key_metric == 'loss':
            if self.num_labels <= 2:
                self.set_attribute(key_metric = 'f1')
            else:
                self.set_attribute(key_metric = 'macro_f1')

    def align_config(self,*args,**kwargs):
        super().align_config(*args,**kwargs)
        if not hasattr(self.config, "kernel_size"):
            self.set_attribute(kernel_size = 3)

        if not hasattr(self.config, "num_labels"):
            self.set_attribute(num_labels = 2)

        self.set_attribute(model_name="cnn_cls")