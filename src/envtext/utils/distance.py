eps = 1e-5

from ..files import Config
import jieba
# jieba.load_userdict(config.env_vocab)

def hamming_dist(a,b):
    if isinstance(a, str):
        a = jieba.lcut(a)
        b = jieba.lcut(b)

    a_in_b = sum([True for aa in a if aa in b])/(len(a) + eps)
    b_in_a = sum([True for bb in b if bb in a])/(len(b) + eps)

    weight_a = len(a) / (len(a) + len(b) + eps)
    weight_b = len(b) / (len(a) + len(b) + eps)

    return a_in_b * weight_a + b_in_a * weight_b


#https://blog.csdn.net/sinat_26811377/article/details/102652547
def edit_dist(a, b):
    """
    计算字符串 a 和 b 的编辑距离
    :param a
    :param b
    :return: dist
    """
    matrix = [[ i + j for j in range(len(a) + 1)] for i in range(len(b) + 1)]
 
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if(a[i-1] == b[j-1]):
                d = 0
            else:
                d = 1
            
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)
 
    return matrix[len(a)][len(b)] / (len(a) + len(b))
