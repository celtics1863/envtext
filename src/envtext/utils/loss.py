from torch import nn
import torch
from torch.nn import functional as F

#focal_loss func, L = -α(1-yi)**γ *ce_loss(xi,yi)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1  
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)

        self.gamma = gamma

    def forward(self, logits, labels):
        # assert preds.dim()==2 and labels.dim()==1
        logits = logits.view(-1,logits.size(-1)) #flatten

        if len(labels) >= 2:
            labels = labels.argmax(dim=-1)


        self.alpha = self.alpha.to(logits.device)
        preds_softmax = F.softmax(logits, dim=1) 
        preds_logsoft = torch.log(preds_softmax)
        
        #focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
        preds_softmax = preds_softmax.gather(1,labels.long().view(-1,1)) 
        preds_logsoft = preds_logsoft.gather(1,labels.long().view(-1,1))
        self.alpha = self.alpha.gather(0,labels.long().view(-1))
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) 

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
