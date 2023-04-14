
import numpy as np
import torch
from typing import *
from collections import Counter
import math


def viterbi_decode(self, emissions: torch.FloatTensor,
                    mask: torch.ByteTensor) -> List[List[int]]:
    # emissions: (seq_length, batch_size, num_tags)
    # mask: (seq_length, batch_size)
    assert emissions.dim() == 3 and mask.dim() == 2
    assert emissions.shape[:2] == mask.shape
    assert emissions.size(2) == self.num_tags
    assert mask[0].all()

    seq_length, batch_size = mask.shape

    # Start transition and first emission
    # shape: (batch_size, num_tags)
    score = self.start_transitions + emissions[0]
    history = []

    # score is a tensor of size (batch_size, num_tags) where for every batch,
    # value at column j stores the score of the best tag sequence so far that ends
    # with tag j
    # history saves where the best tags candidate transitioned from; this is used
    # when we trace back the best tag sequence

    # Viterbi algorithm recursive case: we compute the score of the best tag sequence
    # for every possible next tag
    for i in range(1, seq_length):
        # Broadcast viterbi score for every possible next tag
        # shape: (batch_size, num_tags, 1)
        broadcast_score = score.unsqueeze(2)

        # Broadcast emission score for every possible current tag
        # shape: (batch_size, 1, num_tags)
        broadcast_emission = emissions[i].unsqueeze(1)

        # Compute the score tensor of size (batch_size, num_tags, num_tags) where
        # for each sample, entry at row i and column j stores the score of the best
        # tag sequence so far that ends with transitioning from tag i to tag j and emitting
        # shape: (batch_size, num_tags, num_tags)
        next_score = broadcast_score + self.transitions + broadcast_emission

        # Find the maximum score over all possible current tag
        # shape: (batch_size, num_tags)
        next_score, indices = next_score.max(dim=1)

        # Set score to the next score if this timestep is valid (mask == 1)
        # and save the index that produces the next score
        # shape: (batch_size, num_tags)
        score = torch.where(mask[i].unsqueeze(1), next_score, score)
        history.append(indices)

    # End transition score
    # shape: (batch_size, num_tags)
    score += self.end_transitions

    # Now, compute the best path for each sample

    # shape: (batch_size,)
    seq_ends = mask.long().sum(dim=0) - 1
    best_tags_list = []

    for idx in range(batch_size):
        # Find the tag which maximizes the score at the last timestep; this is our best tag
        # for the last timestep
        _, best_last_tag = score[idx].max(dim=0)
        best_tags = [best_last_tag]

        # We trace back where the best last tag comes from, append that to our best tag
        # sequence, and trace it back again, and so on
        for hist in reversed(history[:seq_ends[idx]]):
            best_last_tag = hist[idx][best_tags[-1]]
            best_tags.append(best_last_tag)

        # Reverse the order because we start from the last timestep
        best_tags.reverse()
        best_tags_list.append(best_tags)

    return best_tags_list

def viterbi_decode_np(predictions,transition):
    '''
    predictions (np.ndarray): [B,L,F]
            B: batchsize
            L: length
            F: number of features
            取对数后的预测结果
        
    transition  (np.ndarray): [F,F]
            F: number of featurs
            取对数后的转移概率矩阵

    predictions中出现-100意味着停止解码
    '''
    predictions_labels = []
    
    for b in range(predictions.shape[0]):
        observation = predictions[b]
        start = 0
        # P(path->B) = P(path) * P(B) * P(A->B|path,B) #A is the last value of path
        #            = P(path) * P(B) * P(A->B|A,B) 
        dp = np.zeros_like(observation)
        prev_states = np.zeros_like(dp,dtype=np.int64)
        for i in range(len(observation)):
            if i == 0:
                dp[i] = observation[i]
            else:
                for t in range(len(transition)):
                    arr = [dp[i-1,c] + transition[c,t] + observation[i,t] for c in range(len(transition))]
                    dp[i,t] = np.max(arr)
                    prev_states[i,t] = np.argmax(arr)

        path = [np.argmax(dp[-1])] 
        for i in range(len(observation)-1,0,-1):
            path.append(prev_states[i,int(path[-1])])
        
        path.reverse()
        predictions_labels.append(path)
    
    return predictions_labels


def ner_decode(pred,ner_encoding = "BIO",return_method=None):
    '''
    pred (np.ndarray) or List[int]: [L]
        预测出的路径
    
    ner_encoding : 
        ner编码方式：
            IO
            BIO
            BIOE
            BIOES

            GP:
            SPAN:
            Pointer:


    return_method:
        返回格式：
        Pointer : 
            return np.ndarray(), shape: [N,3] : [label,start,end]
    '''
    labels = []
    locs = []
    end = 0
    for start,c in enumerate(pred):
        #BIO
        if (ner_encoding == 'BIO' and c > 0) or (ner_encoding is None): #start B
            class_id = (c-1) // 2
            
            if start < end:
                continue
                
            end = start
            while end < len(pred) and (pred[end] - 1) // 2 == class_id:
                if end == len(pred) - 1 or pred[end + 1] == 0 or (pred[end+1] - 1) // 2 != class_id : #stop I
                    labels.append(class_id)
                    locs.append([start,end])
                    end += 1
                    break
                else:
                    end += 1
                    
        #IO
        elif ner_encoding == 'IO' and c > 0:
            class_id = c-1
            if start < end:
                continue
            
            end = start + 1
            while end < len(pred) and pred[end] != c:
                end += 1
            
            labels.append(class_id)
            locs.append([start,end])

        #BIOE
        elif ner_encoding == 'BIOE' and (c % 3 == 1 or (c % 3 == 0 and c > 0)):
            class_id = (c-1)//3
            
            if start < end:
                continue
                
            end = start + 1                    
            while end < len(pred):
                if (pred[end] > 0 and pred[end] % 3 == 0) or end == len(pred) - 1: # E : break
                    labels.append(class_id)
                    locs.append([start,end])
                    end += 1
                    break
                elif pred[end] % 3 == 1 or pred[end] == 0: # O or B : break
                    labels.append(class_id)
                    locs.append([start,end])
                    end += 1
                    break
                else: # I : continue
                    end += 1

        #BIOES
        elif ner_encoding == 'BIOES' and (c % 4 == 1 or (c % 4 == 0 and c > 0)):
            class_id = (c-1)//4
            
            if start < end:
                continue
                
            end = start + 1
            if c % 4 == 0 and c > 0: # S: break
                labels.append(class_id)
                locs.append([start,end])
                continue
                
            while end < len(pred):
                if pred[end] % 4 == 3 or end == len(pred) - 1: # E : break
                    labels.append(class_id)
                    locs.append([start,end])
                    end += 1
                    break
                elif pred[end] % 4 == 0 or pred[end] % 4 == 1: # O or S or B : break
                    labels.append(class_id)
                    locs.append([start,end])
                    end += 1
                    break
                else: # I : continue
                    end += 1

    if return_method == "Pointer":
        return np.array([[l,start,end] for l,(start,end) in zip(labels,locs)])
    else:
        return labels,locs



def _tf(sents):
    tf = Counter()
    for s in sents:
        tf.update(s)
    return tf

def _idf(sents):
    df = Counter()
    for s in sents:
        df.update(set(s))

    idf = {k: math.log(len(sents)/ (v + 1)  ) for k,v in df.items()}
    return idf

def _tfidf(sents):
    tf = _tf(sents)
    idf = _idf(sents)

    tfidf = Counter({k: tf[k] * idf[k] for k in tf if k in idf})
    return tfidf


def TFIDF(list_of_words:List[List[str]]):
    return _tfidf(list_of_words)