import numpy as np
import math

eps = 1e-5 #避免除0

def rmse(eval_pred):
    '''
    均方根误差计算，用于回归问题
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    predictions, labels = eval_pred
    return math.sqrt(np.mean((predictions-labels)**2))

def mae(eval_pred):
    '''
    平均绝对误差计算，用于回归问题
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    predictions, labels = eval_pred
    return np.mean(np.abs(predictions-labels))

def r2(eval_pred):
    '''
    r2计算，用于回归问题
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    predictions, labels = eval_pred
    s1 = predictions.reshape(-1) - np.mean(predictions)
    s2 = labels.reshape(-1) - np.mean(labels)
    return sum(s1*s2)/math.sqrt(sum(s1*s1)*sum(s2*s2))

def accuracy(eval_pred):
    '''
    准确率计算，用于分类问题。
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    predictions, labels = eval_pred
    predictions_labels = np.argmax(predictions,axis=1)
    return sum(predictions_labels==labels)/len(labels)

def accuracy_for_binary_logits(eval_pred):
    '''
    准确率计算，用于binary logits输出的分类问题。
     eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    predictions, labels = eval_pred
    predictions_labels = predictions > 0.5
    return np.mean(predictions_labels==labels)

def accuracy_for_ner(eval_pred):
    '''
    准确率计算，用于序列标注问题。
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
           shape of predictions: (batchsize,sequence length,label numbers)
           shape of labels: (batchsize,sequence length)
           不计算labels中小于0的标签
    '''
    predictions, labels = eval_pred
    predictions_labels = np.argmax(predictions,axis=-1)
    
    g = lambda seq_logits,seq_label : sum([logits == label for logits,label in zip(seq_logits,seq_label) if label > 0])/(sum(seq_label > 0)+eps)
    return np.mean(list(map(g,predictions_labels,labels)))

def confusion_matrix(eval_pred):
    '''
    困惑矩阵计算，用于分类问题。
     eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    predictions, labels = eval_pred
    predictions_labels = np.argmax(predictions,axis=1)
    LABLES = set(labels).union(set(predictions_labels))
    LABLES = sorted(list(LABLES))
    matrix = np.zeros((len(LABLES),len(LABLES)))
    for i,j in zip(predictions_labels,labels):
        matrix[LABLES.index(i),LABLES.index(j)] += 1
    return matrix

def confusion_matrix_for_binary_logits(eval_pred):
    '''
    困惑矩阵计算，用于binary logits输出的分类问题。
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    predictions, labels = eval_pred
    predictions_labels = predictions > 0.5
    label_num = predictions.shape[1]
    matrix = np.zeros((label_num,2,2))
    for pred,label in zip(predictions_labels,labels):
        for idx,(i,j) in enumerate(zip(pred,label)):
            matrix[idx][i,j] += 1
    return matrix


def confusion_matrix_for_ner(eval_pred):
    '''
    困惑矩阵计算，用于序列标注问题。
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
           shape of predictions: (batchsize,sequence length,label numbers)
           shape of labels: (batchsize,sequence length)
           不计算labels中小于0的标签
    '''
    predictions, labels = eval_pred
    predictions_labels = np.argmax(predictions,axis=-1)
    #BIO 标注
    #多少种实体：
    emtity_num = (predictions.shape[-1] - 1) // 2
    
    # 标注了多少个实体：
    def get_labeled_entities(preds,labels):
        p = 0 #指针
        entities = [] #[实体类别，开始位置，结束位置]
        flag = False
        start,end = -1,-1
        while True:
            if p == len(labels) or p == len(preds) or labels[p] < 0:
                if flag and start > 0:
                    entities.append([preds[start]//2,start,p])
                break
            elif (not flag) and (preds[p] % 2 == 1):
                flag = True
                start = p
            elif flag and (preds[p] % 2 == 0):
                if preds[p] == 0 :
                    end = p
                    if start > 0 and end > 0:
                        entities.append([preds[start]//2,start,end])
                    flag = False
                elif preds[p]-1 == preds[start]:
                    pass
                else:
                    flag = False
                    start,end = -1,-1
                    pass
            else:
                flag = False
                start,end = -1,-1
                pass
            p += 1
        return entities
    
    seqs_preds_entities = map(get_labeled_entities,predictions_labels,labels)
    seqs_labels_entities = map(get_labeled_entities,labels,labels)
    
    def get_tp_fp_fn(preds_entities,labels_entities):
        tp,tn,fp,fn = [0]*emtity_num,[0]*emtity_num,[0]*emtity_num,[0]*emtity_num
        for entity in labels_entities:
            if entity in preds_entities:
                tp[entity[0]] += 1
            else:
                fn[entity[0]] += 1
        
        for entity in preds_entities:
            if entity not in labels_entities:
                fp[entity[0]] += 1
        
        return [tp,fn,fp,tn]
    
    matrix = np.sum(list(map(get_tp_fp_fn,seqs_preds_entities,seqs_labels_entities)),axis = 0)
    matrix = np.array(matrix).reshape(2,2,emtity_num).transpose(2,0,1)
    return matrix


def precision(eval_pred):
    '''
    精确率计算，用于2分类问题。
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    predictions, labels = eval_pred
    predictions_labels = np.argmax(predictions,axis=1)
    TP = sum((labels==1) == (predictions_labels==1))
    TN = sum((labels==0) == (predictions_labels==0))
    FP = sum((labels==0) == (predictions_labels==1))
    FN = sum((labels==1) == (predictions_labels==0))
    return TP/(TP+FP+eps)

def precision_for_binary_logits(eval_pred):
    '''
    精确率计算，用于binary logits输出的2分类问题。
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    predictions, labels = eval_pred
    predictions_labels = predictions > 0.5
    TP = sum((labels==1) == (predictions_labels==1))
    TN = sum((labels==0) == (predictions_labels==0))
    FP = sum((labels==0) == (predictions_labels==1))
    FN = sum((labels==1) == (predictions_labels==0))
    return TP/(TP+FP+eps)

def precision_for_ner(eval_pred):
    '''
    精确率计算，用于序列标注问题。
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
           shape of predictions: (batchsize,sequence length,label numbers)
           shape of labels: (batchsize,sequence length)
           不计算labels中小于0的标签
    '''
    predictions, labels = eval_pred
    predictions_labels = np.argmax(predictions,axis=-1)
    #BIO 标注
    # 标注了多少个实体：
    def get_labeled_entities(preds,labels):
        p = 0 #指针
        entities = []
        flag = False
        start,end = -1,-1
        while True:
            if p == len(labels) or p == len(preds) or labels[p] < 0:
                if flag and start > 0:
                    entities.append([start,p])
                break
            elif (not flag) and (preds[p] % 2 == 1):
                flag = True
                start = p
            elif flag and (preds[p] % 2 == 0):
                if preds[p] == 0 :
                    end = p
                    if start > 0 and end > 0:
                        entities.append([start,end])
                    flag = False
                    start,end = -1,-1
                elif preds[p]-1 == preds[start]:
                    pass
                else:
                    end = p
                    if start > 0 and end > 0:
                        entities.append([start,end])
                    flag = False
                    start,end = -1,-1
            else:
                flag = False
                start,end = -1,-1
            p += 1
        return entities
    
    seqs_preds_entities = map(get_labeled_entities,predictions_labels,labels)
    seqs_labels_entities = map(get_labeled_entities,labels,labels)
    
    def get_tp_fp_fn(preds_entities,labels_entities):
        tp,tn,fp,fn = 0,0,0,0
        for entity in labels_entities:
            if entity in preds_entities:
                tp += 1
            else:
                fn += 1
        
        for entity in preds_entities:
            if entity not in labels_entities:
                fp += 1
        
        return [tp,fn,fp]
    
    TP,FN,FP = np.sum(list(map(get_tp_fp_fn,seqs_preds_entities,seqs_labels_entities)),axis = 0)
    return TP/(TP+FP+eps)

def recall(eval_pred):
    '''
    召回率计算，用于2分类问题。
     eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    predictions, labels = eval_pred
    predictions_labels = np.argmax(predictions,axis=1)
    TP = sum((labels==1) == (predictions_labels==1))
    TN = sum((labels==0) == (predictions_labels==0))
    FP = sum((labels==0) == (predictions_labels==1))
    FN = sum((labels==1) == (predictions_labels==0))
    return TP/(TP+FN+eps)   

def recall_for_binary_logits(eval_pred):
    '''
    召回率计算，用于binary logits输出的分类问题。
     eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    predictions, labels = eval_pred
    predictions_labels = predictions > 0.5
    TP = sum((labels==1) == (predictions_labels==1))
    TN = sum((labels==0) == (predictions_labels==0))
    FP = sum((labels==0) == (predictions_labels==1))
    FN = sum((labels==1) == (predictions_labels==0))
    return TP/(TP+FN+eps) 

def recall_for_ner(eval_pred):
    '''
    召回率计算，用于序列标注问题。
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
           shape of predictions: (batchsize,sequence length,label numbers)
           shape of labels: (batchsize,sequence length)
           不计算labels中小于0的标签
    '''
    predictions, labels = eval_pred
    predictions_labels = np.argmax(predictions,axis=-1)
    #BIO 标注
    # 标注了多少个实体：
    def get_labeled_entities(preds,labels):
        p = 0 #指针
        entities = []
        flag = False
        start,end = -1,-1
        while True:
            if p == len(labels) or p == len(preds) or labels[p] < 0:
                if flag and start > 0:
                    entities.append([start,p])
                break
            elif (not flag) and (preds[p] % 2 == 1):
                flag = True
                start = p
            elif flag and (preds[p] % 2 == 0):
                if preds[p] == 0 :
                    end = p
                    if start > 0 and end > 0:
                        entities.append([start,end])
                    flag = False
                    start,end = -1,-1
                elif preds[p]-1 == preds[start]:
                    pass
                else:
                    end = p
                    if start > 0 and end > 0:
                        entities.append([start,end])
                    flag = False
                    start,end = -1,-1
            else:
                flag = False
                start,end = -1,-1
            p += 1
        return entities
    
    seqs_preds_entities = map(get_labeled_entities,predictions_labels,labels)
    seqs_labels_entities = map(get_labeled_entities,labels,labels)
    
    def get_tp_fp_fn(preds_entities,labels_entities):
        tp,tn,fp,fn = 0,0,0,0
        for entity in labels_entities:
            if entity in preds_entities:
                tp += 1
            else:
                fn += 1
        
        for entity in preds_entities:
            if entity not in labels_entities:
                fp += 1
        
        return [tp,fn,fp]
    
    TP,FN,FP = np.sum(list(map(get_tp_fp_fn,seqs_preds_entities,seqs_labels_entities)),axis = 0)
    
    return TP/(TP+FN+eps)


def f1(eval_pred):
    '''
    f1 计算，用于2分类问题
     eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    predictions, labels = eval_pred
    predictions_labels = np.argmax(predictions,axis=1)
    TP = sum((labels==1) == (predictions_labels==1))
    TN = sum((labels==0) == (predictions_labels==0))
    FP = sum((labels==0) == (predictions_labels==1))
    FN = sum((labels==1) == (predictions_labels==0))
    precision = TP/(TP+FP+eps)
    recall = TP/(TP+FN+eps)    
    return 2*precision*recall / (precision+recall+eps)


def f1_for_ner(eval_pred):
    '''
    f1-score计算，用于序列标注问题。
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
           shape of predictions: (batchsize,sequence length,label numbers)
           shape of labels: (batchsize,sequence length)
           不计算labels中小于0的标签
    '''
    precision = precision_for_ner(eval_pred)
    recall = recall_for_ner(eval_pred)
    return 2*precision*recall / (precision+recall+eps)

def micro_f1(eval_pred):
    '''
    micro_f1 计算，用于多分类问题，等效于accuracy
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    matrix = confusion_matrix(eval_pred)
    num_labels = len(matrix)
    TP,TN,FP,FN = \
        np.zeros(num_labels),np.zeros(num_labels),np.zeros(num_labels),np.zeros(num_labels)
    TP = matrix.diagonal()
    FP = matrix.sum(axis = 0)-TP
    FN = matrix.sum(axis = 1)-TP
    precision = TP.sum()/(TP.sum()+FP.sum()+eps)
    recall = TP.sum()/(TP.sum()+FN.sum()+eps)    
    return 2*precision*recall / (precision+recall+eps)


def micro_f1_for_binary_logits(eval_pred):
    '''
    micro_f1 计算，用于binary logits输出的多分类问题，等效于accuracy
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    matrix = confusion_matrix_for_binary_logits(eval_pred)
    matrix = matrix.sum(axis=0)
    TP,TN,FP,FN = \
        matrix[1,1],matrix[0,0],matrix[0,1],matrix[1,0]
    precision = TP/(TP+FP+eps)
    recall = TP/(TP+FN+eps)    
    return 2*precision*recall / (precision+recall+eps)


def micro_f1_for_ner(eval_pred):
    '''
    micro_f1 计算，用于namely entity recognition问题
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
           shape of predictions: (batchsize,sequence length,label numbers)
           shape of labels: (batchsize,sequence length)
           不计算labels中小于0的标签
    '''
    matrix = confusion_matrix_for_ner(eval_pred)
    matrix = matrix.sum(axis=0)
    TP,TN,FP,FN = \
        matrix[0,0],matrix[1,1],matrix[0,1],matrix[1,0]
    precision = TP/(TP+FP+eps)
    recall = TP/(TP+FN+eps)    
    return 2*precision*recall / (precision+recall+eps)


def macro_f1(eval_pred):
    '''
    macro_f1 计算，用于多分类问题，等效于accuracy
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    matrix = confusion_matrix(eval_pred)
    num_labels = len(matrix)
    TP,TN,FP,FN = \
        np.zeros(num_labels),np.zeros(num_labels),np.zeros(num_labels),np.zeros(num_labels)
    TP = matrix.diagonal()
    FP = matrix.sum(axis = 0)-TP
    FN = matrix.sum(axis = 1)-TP
    def fun(tp,fp,fn):
        precision = tp/(tp+fp+eps)
        recall = tp/(tp+fn+eps)    
        return 2*precision*recall / (precision+recall+eps)  
    return sum(map(fun,TP,FP,FN))/num_labels

def macro_f1_for_binary_logits(eval_pred):
    '''
    macro_f1 计算，用于binary logits输出的多分类问题，等效于accuracy
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    matrix = confusion_matrix_for_binary_logits(eval_pred)
    def fun(matrix):
        tp,fp,fn = \
            matrix[0,0],matrix[0,1],matrix[1,0]
        precision = tp/(tp+fp+eps)
        recall = tp/(tp+fn+eps)    
        return 2*precision*recall / (precision+recall+eps)  
    return sum(map(fun,matrix))/len(matrix)

def macro_f1_for_ner(eval_pred):
    '''
    macro_f1 计算，用于namely entity recognition问题
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
           shape of predictions: (batchsize,sequence length,label numbers)
           shape of labels: (batchsize,sequence length)
           不计算labels中小于0的标签
    '''
    matrix = confusion_matrix_for_ner(eval_pred)
    def fun(matrix):
        tp,fp,fn = \
            matrix[0,0],matrix[0,1],matrix[1,0]
        precision = tp/(tp+fp+eps)
        recall = tp/(tp+fn+eps)    
        return 2*precision*recall / (precision+recall+eps)  
    return sum(map(fun,matrix))/len(matrix)

#https://zhuanlan.zhihu.com/p/374269641
def find_topk(a, k, axis=-1, largest=True, sorted=True):
    if axis is None:
        axis_size = a.size
    else:
        axis_size = a.shape[axis]
    assert 1 <= k <= axis_size

    a = np.asanyarray(a)
    if largest:
        index_array = np.argpartition(a, axis_size-k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k)-1, axis=axis)
    else:
        index_array = np.argpartition(a, k-1, axis=axis)
        topk_indices = np.take(index_array, np.arange(k), axis=axis)
    topk_values = np.take_along_axis(a, topk_indices, axis=axis)
    if sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        if largest:
            sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
        sorted_topk_values = np.take_along_axis(
            topk_values, sorted_indices_in_topk, axis=axis)
        sorted_topk_indices = np.take_along_axis(
            topk_indices, sorted_indices_in_topk, axis=axis)
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices


def topk_accuracy(eval_pred,topk=3):
    '''
    计算topk_accuracy，用于多分类问题。模型输出所有可能类别的概率，如果label在概率topk大的预测中，则认为预测准确。
    '''
    predictions, labels = eval_pred
    preds,indices = find_topk(predictions,topk,axis=1)
    topk_acc = sum(map(lambda x,y: x in y , labels,indices))/len(labels)
    return topk_acc

def metrics_for_reg(eval_pred):
    '''
    用于回归问题的评价指标报告
    '''
    report = {
        'rmse':rmse(eval_pred),
        'mae':mae(eval_pred),
        'r2':r2(eval_pred)
        }
    return report
    
def metrics_for_cls(eval_pred):
    '''
    用于分类问题的评价指标报告
    '''
    predictions, labels = eval_pred
    
    if isinstance(labels,np.ndarray):
        if len(labels.shape) == 2: 
            #one-hot label
            LABLES = range(labels.shape[1])
            labels = np.nonzero(labels > 0.5)[1]
        else:
            #index label
            LABLES = sorted(list(set(labels)))
            
    elif isinstance(labels[0],int):
        #index label
        LABLES = sorted(list(set(labels)))
        labels = np.array(labels)
        
    elif isinstance(labels[0],str):
        #str label
        LABLES = sorted(list(set(labels)))
        labels = np.array([LABLES.index(l) for l in labels])
        
    elif isinstance(labels[0],list): 
        #one-hot label
        labels = np.nonzero(np.array(labels) > 0.5)[1]
        
    else:
        raise NotImplemented
    
    eval_pred = (predictions,labels)
    if len(LABLES) == 2:
        report = {
            'accuracy':accuracy(eval_pred),
            'f1':f1(eval_pred),
            'precision':precision(eval_pred),
            'recall':recall(eval_pred),
        }
    elif 3 < len(LABLES) <= 5:
        report = {
            'accuracy':accuracy(eval_pred),
            'top3 accuracy':topk_accuracy(eval_pred),
            'micro_f1':micro_f1(eval_pred),
            'macro_f1':macro_f1(eval_pred),
        }
    else:
        report = {
            'accuracy':accuracy(eval_pred),
            'top3 accuracy':topk_accuracy(eval_pred),
            'top5 accuracy':topk_accuracy(eval_pred),
            'micro_f1':micro_f1(eval_pred),
            'macro_f1':macro_f1(eval_pred),
        }
    return report

def metrics_for_cls_with_binary_logits(eval_pred):
    '''
    用于binary_logits输出的分类问题的评价指标报告
    '''
    predictions, labels = eval_pred
    NUM_LABLES = 1 if len(labels.shape) == 1 else labels.shape[1]
    predictions = predictions.astype(np.int)
    labels = labels.astype(np.int)
    eval_pred = (predictions,labels)
    if NUM_LABLES == 1:
        report = {
            'accuracy':accuracy_for_binary_logits(eval_pred),
            'f1':f1_for_binary_logits(eval_pred),
            'precision':precision_for_binary_logits(eval_pred),
            'recall':recall_for_binary_logits(eval_pred),
        }
    else:
        report = {
            'accuracy':accuracy_for_binary_logits(eval_pred),
            'micro_f1':micro_f1_for_binary_logits(eval_pred),
            'macro_f1':macro_f1_for_binary_logits(eval_pred),
        }
    return report

def metrics_for_ner(eval_pred):
    '''
    用于NER问题的评价指标报告
    '''
    predictions, labels = eval_pred
    # BIO 标注
    NUM_LABLES = predictions.shape[-1] // 2
    if NUM_LABLES == 1:
        report =  {
            'f1':f1_for_ner(eval_pred),
            'accuracy':accuracy_for_ner(eval_pred),
            'precision':precision_for_ner(eval_pred),
            'recall':recall_for_ner(eval_pred),
        }
    else:
        report =  {
            'f1':f1_for_ner(eval_pred),
            'accuracy':accuracy_for_ner(eval_pred),
            'precision':precision_for_ner(eval_pred),
            'recall':recall_for_ner(eval_pred),
            'micro_f1':micro_f1_for_ner(eval_pred),
            'macro_f1':macro_f1_for_ner(eval_pred),
        }
    return report
    