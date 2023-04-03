from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score


import jieba
from ..files import Config
import re
from ..utils.metrics import metrics_for_cls
from ..data.utils import load_dataset

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dataset')
parser.add_argument('-s','--split',type=float,default=0.7)
parser.add_argument('-v','--jieba_vocab',type=str,default = '')

args = parser.parse_args()


if args.jieba_vocab:
    jieba.load_userdict(args.jieba_vocab)
else:
    jieba.load_userdict(Config.env_vocab)


datasets,config = load_dataset(args.dataset,task="cls",format="text2",split=1,label_inline=True)

print("*"*5,"数据集参数是")
print(config)

#制作数据集
corpus = [" ".join(jieba.lcut(re.sub("\s", "", t))) for t in texts]

# 导入文本
tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform(texts)

X = tfidf_matrix.toarray()
y = np.array(labels)

X_train,X_test,y_train,y_test = ts(X,y,test_size=1 - args.split)

model = SVC()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# print(metrics_for_cls((y_train,y_test)))
print("*"*5,"scores are")
# model.score(y_test,model.predict(X_test))
print("f1",f1_score(y_train, y_pred))
print("macro-f1",f1_score(y_train, y_pred,average="macro"))
print("micro-f1",f1_score(y_train, y_pred,average="micro"))




