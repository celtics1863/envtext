import numpy as np
import math

eps = 1e-7 #避免除0

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

def get_labeled_entities(pred,label,ner_encoding = 'BIO'):
    # 标注了多少个实体：
    p = 0 #指针
    entities = [] #[实体类别，开始位置，结束位置]
    flag = False
    start,end = -1,-1
    #一个状态机
    for start,c in enumerate(pred): # len(pred) >= len(label)
        if c < 0 or (start < len(label) and label[start] < 0): #pred 或 label 出现 -100
            break
        #BIO
        if ner_encoding == 'BIO' and c > 0: #start B
            class_id = (c-1) // 2
            
            if start < end:
                continue
                
            end = start
            while end < len(pred) and end < len(label) \
                    and pred[end] >= 0 and label[end] >=0 \
                    and (pred[end] -1)//2 == class_id :
                end +=1
            entities.append([class_id,start,end])
            end += 1 #end = entity_loc_y + 1
        #IO
        elif ner_encoding == 'IO' and c > 0:
            class_id = c-1
            end = start
            while (end < len(pred) and pred[end] == c) and (end < len(label) and label[end]>=0):
                end += 1
            entities.append([class_id,start,end-1])

        #BIOE
        elif ner_encoding == 'BIOE' and c > 0 :
            class_id = (c-1)//3
            
            if start < end:
                continue
            
            end = start
            while end < len(pred):
                if end == len(pred) - 1 or end == len(label) - 1 or label[end+1] < 0:
                    entities.append([class_id,start,end])
                    break
                elif pred[end] % 3 == 2: #E stop
                    entities.append([class_id,start,end])
                    break
                elif pred[end] % 3 == 0 or pred[end] % 3 == 1: # B O I
                    entities.append([class_id,start,end-1])
                    break
                else:
                    end += 1
        #BIOES
        elif ner_encoding == 'BIOES' and c > 0 :
            class_id = (c-1)//4
            
            if start < end:
                continue
            
            end = start
            if c % 4 == 0: #S
                entities.append([class_id,start,start])

            while end < len(pred):
                if end == len(pred) - 1 or end == len(label) - 1 or label[end+1] < 0:
                    entities.append([class_id,start,end])
                    break
                elif pred[end] % 4 == 3: #E stop
                    entities.append([class_id,start,end])
                    break
                elif pred[end] % 4 == 0 or pred[end] % 4 == 1: # B O I
                    entities.append([class_id,start,end-1])
                    break
                else:
                    end += 1
                
    return entities    

def softmax( f ):
    return np.exp(f) / np.sum(np.exp(f),axis=-1)[...,None]

def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


def confusion_matrix_for_ner(eval_pred,ner_encoding = 'BIO',transition = None,return_all = False):
    '''
    困惑矩阵计算，用于序列标注问题。
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
           shape of predictions: (batchsize,sequence length,label numbers)
           shape of labels: (batchsize,sequence length)
           不计算labels中小于0的标签
    '''
    predictions_labels = None
    try:
        predictions, labels = eval_pred
        predictions_labels = np.argmax(predictions,axis=-1)
    except:
        predictions, predictions_labels, labels = eval_pred

    
    if transition is None and predictions_labels is None:
        predictions_labels = np.argmax(predictions,axis=-1)
    elif predictions_labels is None:
        #使用viterbi解码
        #取对数
        predictions_p = np.log(softmax(predictions)+eps)

        transition = np.log(transition + eps)

        predictions_labels = []
        
        for b in range(predictions.shape[0]):
            observation = predictions_p[b]
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


        predictions_labels = np.array(predictions_labels).astype("int").tolist()

        old = np.argmax(predictions,axis=-1)
        new = np.array(predictions_labels)


    #BIO 标注
    #多少种实体：
    if ner_encoding == 'BIO':
        entity_num = (predictions.shape[-1] - 1) // 2
    elif ner_encoding == 'IO':
        entity_num = (predictions.shape[-1] - 1) 
    elif ner_encoding == 'BIOE':
        entity_num = (predictions.shape[-1] - 1) // 3 
    elif ner_encoding == 'BIOES':
        entity_num = (predictions.shape[-1] - 1) // 4
    
    seqs_preds_entities = map(lambda x,y: get_labeled_entities(x,y,ner_encoding),predictions_labels,labels)
    seqs_labels_entities = map(lambda x,y: get_labeled_entities(x,y,ner_encoding),labels,labels)
    
    def get_tp_fp_fn(preds_entities,labels_entities):
        tp,tn,fp,fn = [0]*entity_num,[0]*entity_num,[0]*entity_num,[0]*entity_num
        for entity in labels_entities:
            if entity in preds_entities:
                tp[entity[0]] += 1
            else:
                fn[entity[0]] += 1
        
        for entity in preds_entities:
            if entity not in labels_entities:
                fp[entity[0]] += 1
        
        return [tp,fn,fp,tn]
    
    matrix_of_seqs = list(map(get_tp_fp_fn,seqs_preds_entities,seqs_labels_entities))
    matrix = np.sum(matrix_of_seqs,axis = 0)
    matrix = np.array(matrix).reshape(2,2,entity_num).transpose(2,0,1)

    if return_all:
        return matrix, matrix_of_seqs
    else:
        return matrix


def precision(eval_pred):
    '''
    精确率计算，用于2分类问题。
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    predictions, labels = eval_pred
    predictions_labels = np.argmax(predictions,axis=1)
    TP = sum(np.logical_and(labels==1,predictions_labels==1))
    TN = sum(np.logical_and(labels==0,predictions_labels==0))
    FP = sum(np.logical_and(labels==0,predictions_labels==1))
    FN = sum(np.logical_and(labels==1,predictions_labels==0))
    return TP/(TP+FP+eps)

def precision_for_binary_logits(eval_pred):
    '''
    精确率计算，用于binary logits输出的2分类问题。
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    predictions, labels = eval_pred
    predictions_labels = predictions > 0.5
    TP = sum(np.logical_and(labels==1,predictions_labels==1))
    TN = sum(np.logical_and(labels==0,predictions_labels==0))
    FP = sum(np.logical_and(labels==0,predictions_labels==1))
    FN = sum(np.logical_and(labels==1,predictions_labels==0))
    return TP/(TP+FP+eps)

def precision_for_ner(eval_pred,ner_encoding = 'BIO',transition = None,confusion_matrix = None):
    '''
    精确率计算，用于序列标注问题。
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
           shape of predictions: (batchsize,sequence length,label numbers)
           shape of labels: (batchsize,sequence length)
           不计算labels中小于0的标签
    '''

    if confusion_matrix is None:
        predictions, labels = eval_pred
        predictions_labels = np.argmax(predictions,axis=-1)
        confusion_matrix = confusion_matrix_for_ner(eval_pred,ner_encoding,transition)
    
    matrix = confusion_matrix.sum(axis=0)
    return matrix[0][0] / (matrix[0][0] + matrix[1][0] + eps)


def recall(eval_pred):
    '''
    召回率计算，用于2分类问题。
     eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    predictions, labels = eval_pred
    predictions_labels = np.argmax(predictions,axis=1)
    TP = sum(np.logical_and(labels==1,predictions_labels==1))
    TN = sum(np.logical_and(labels==0,predictions_labels==0))
    FP = sum(np.logical_and(labels==0,predictions_labels==1))
    FN = sum(np.logical_and(labels==1,predictions_labels==0))
    return TP/(TP+FN+eps)   

def recall_for_binary_logits(eval_pred):
    '''
    召回率计算，用于binary logits输出的分类问题。
     eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    predictions, labels = eval_pred
    predictions_labels = predictions > 0.5
    TP = sum(np.logical_and(labels==1,predictions_labels==1))
    TN = sum(np.logical_and(labels==0,predictions_labels==0))
    FP = sum(np.logical_and(labels==0,predictions_labels==1))
    FN = sum(np.logical_and(labels==1,predictions_labels==0))
    return TP/(TP+FN+eps) 

def recall_for_ner(eval_pred,ner_encoding = 'BIO',transition = None,confusion_matrix = None):
    '''
    召回率计算，用于序列标注问题。
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
           shape of predictions: (batchsize,sequence length,label numbers)
           shape of labels: (batchsize,sequence length)
           不计算labels中小于0的标签
    '''
    if confusion_matrix is None:
        confusion_matrix = confusion_matrix_for_ner(eval_pred,ner_encoding,transition)
    matrix = confusion_matrix.sum(axis=0)
    
    return matrix[0][0] / (matrix[0][0] + matrix[0][1] + eps)

def f1(eval_pred):
    '''
    f1 计算，用于2分类问题
     eval_pred (`tuple`):
       eval_pred = (predictions, labels)
    '''
    predictions, labels = eval_pred
    predictions_labels = np.argmax(predictions,axis=1)
    # TP = sum((labels==1) == (predictions_labels==1))
    TP = sum(np.logical_and(labels==1,predictions_labels==1))
    TN = sum(np.logical_and(labels==0,predictions_labels==0))
    FP = sum(np.logical_and(labels==0,predictions_labels==1))
    FN = sum(np.logical_and(labels==1,predictions_labels==0))

    precision = TP/(TP+FP+eps)
    recall = TP/(TP+FN+eps)    
    return 2*precision*recall / (precision+recall+eps)


def f1_for_ner(eval_pred,ner_encoding = 'BIO',transition = None,confusion_matrix = None):
    '''
    f1-score计算，用于序列标注问题。
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
           shape of predictions: (batchsize,sequence length,label numbers)
           shape of labels: (batchsize,sequence length)
           不计算labels中小于0的标签
    '''
    precision = precision_for_ner(eval_pred,ner_encoding,transition,confusion_matrix)
    recall = recall_for_ner(eval_pred,ner_encoding,transition,confusion_matrix)
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


def micro_f1_for_ner(eval_pred,ner_encoding = 'BIO',transition = None,confusion_matrix = None):
    '''
    micro_f1 计算，用于namely entity recognition问题
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
           shape of predictions: (batchsize,sequence length,label numbers)
           shape of labels: (batchsize,sequence length)
           不计算labels中小于0的标签
    '''
    if confusion_matrix is None:
        confusion_matrix = confusion_matrix_for_ner(eval_pred,ner_encoding,transition)
    matrix = confusion_matrix.sum(axis=0)
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

def macro_f1_for_ner(eval_pred,ner_encoding = 'BIO',transition = None,confusion_matrix = None):
    '''
    macro_f1 计算，用于namely entity recognition问题
    eval_pred (`tuple`):
       eval_pred = (predictions, labels)
           shape of predictions: (batchsize,sequence length,label numbers)
           shape of labels: (batchsize,sequence length)
           不计算labels中小于0的标签
    '''
    if confusion_matrix is None:
        confusion_matrix = confusion_matrix_for_ner(eval_pred,ner_encoding,transition)

    def fun(matrix):
        tp,fp,fn = \
            matrix[0,0],matrix[0,1],matrix[1,0]
        precision = tp/(tp+fp+eps)
        recall = tp/(tp+fn+eps)    
        return 2*precision*recall / (precision+recall+eps)  
    return sum(map(fun,confusion_matrix))/len(confusion_matrix)

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

def metrics_for_ner(eval_pred,ner_encoding = 'BIO',transition = None):
    '''
    用于NER问题的评价指标报告
    '''
    
    try:
        predictions, labels = eval_pred
    except:
        predictions, preds_seq , labels = eval_pred

    if ner_encoding == 'BIOES':
        NUM_LABLES = (predictions.shape[-1] -1) // 4
    elif ner_encoding == "BIOE":
        NUM_LABLES = (predictions.shape[-1] -1) // 3
    elif ner_encoding == 'IO':
        NUM_LABLES = predictions.shape[-1] - 1
    # BIO 标注
    elif ner_encoding == 'BIO':
        NUM_LABLES = (predictions.shape[-1] -1) // 2


    matrix,metric_for_seq = confusion_matrix_for_ner(eval_pred,ner_encoding,transition,return_all=True)

    if NUM_LABLES == 1:
        report =  {
            'f1':f1_for_ner(eval_pred,ner_encoding,transition,matrix),
            'precision':precision_for_ner(eval_pred,ner_encoding,transition,matrix),
            'recall':recall_for_ner(eval_pred,ner_encoding,transition,matrix),
            'confusion_matrix':matrix.tolist(),
            'metric_for_seq':metric_for_seq,
        }
    else:
        report =  {
            'f1':f1_for_ner(eval_pred,ner_encoding,transition,matrix),
            'precision':precision_for_ner(eval_pred,ner_encoding,transition,matrix),
            'recall':recall_for_ner(eval_pred,ner_encoding,transition,matrix),
            'micro_f1':micro_f1_for_ner(eval_pred,ner_encoding,transition,matrix),
            'macro_f1':macro_f1_for_ner(eval_pred,ner_encoding,transition,matrix),
            'confusion_matrix':matrix.tolist(),
            'metric_for_seq':metric_for_seq,
        }
    
    # print(matrix)
    return report
    