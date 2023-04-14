from torch import nn
import torch
import torch.nn.functional as F
from .rnn_base import RNNBase
from ..tokenizers import Word2VecTokenizer,OnehotTokenizer
import math
from .cls_base import CLSBase


class TFIDFCLSModel(nn.Module):
    def __init__(self,  token_size,
                        num_classes, 
                        num_layers = 3,
                        embed_size = None  , 
                        kernel_size = 3,
                        **kwargs
                        ):
        super().__init__()

        if embed_size:
            self.proj_layer = nn.Linear(token_size, embed_size)
        else:
            self.proj_layer = nn.Identity()
            embed_size = token_size
            
        self.nn_stack = nn.ModuleList()

        #简单线性网络
        for i in range(num_layers):
            self.nn_stack.append(nn.Linear(embed_size,embed_size))
            self.nn_stack.append(nn.ReLU())

        self.fc = nn.Sequential(
            nn.Linear(embed_size,num_classes),
            )
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self,X,labels=None,**kwargs):
        X = self.proj_layer(X.float())

        #run network
        for module in self.nn_stack:
            X = module(X)

        #get logits
        logits = self.fc(X)

        if labels is not None:
            loss = self.loss_fn(logits,labels)
            outputs = (loss,logits)
            return outputs
        else:
            return (logits,)
    
class TFIDFCLS(CLSBase,RNNBase):
    def initialize_rnn(self,path = None,config = None,**kwargs):
        super().initialize_rnn(path,config,**kwargs)
        self.model = TFIDFCLSModel(self.tokenizer.vector_size,
                         self.config.num_labels,
                         self.config.num_layers,
                         self.config.embed_size,
                         self.config.kernel_size,
                        )

        self.model = self.model.to(self.device)
        
        if self.key_metric == 'validation loss' or self.key_metric == 'loss':
            if self.num_labels <= 2:
                self.set_attribute(key_metric = 'f1')
            else:
                self.set_attribute(key_metric = 'macro_f1')

    def initialize_config(self,*args,**kwargs):
        super().initialize_config(*args,**kwargs)
        if not hasattr(self.config, "kernel_size") or self.config.kernel_size is None:
            self.set_attribute(kernel_size = 3)

        if not hasattr(self.config, "num_labels") or self.config.num_labels is None:
            self.set_attribute(num_labels = 2)

        if not hasattr(self.config, "num_layers") or self.config.num_layers is None:
            self.set_attribute(num_layers = 2)

        if not hasattr(self.config, "embed_size") or self.config.embed_size is None:
            self.set_attribute(embed_size = 512)

        if not hasattr(self.config, "split") or self.config.split is None:
            self.set_attribute(split = "[。\n？！；]")

        if not hasattr(self.config, "vocab_path") or self.config.vocab_path is None:
            self.set_attribute(vocab_path = "default")

        self.set_attribute(
                model_name="tfidf_cls",
                token_method="tf-idf",
                truncation = False,
                padding = False,
                num_layers=1,
                )





        