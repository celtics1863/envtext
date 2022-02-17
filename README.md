# envText

**首款**中文环境领域文本分析工具。仍然在内测中，敬请期待。

特性：  
1. :one:支持中文环境领域大规模预训练模型**envBert**！

2. :two:支持中文环境领域大规模预训练**词向量**!

3. :three:支持中文环境领域专家过滤的**词表**!

4. :four: **一且设计均为领域专家研究服务**：
    - 为神经网络模型精简了接口，只保留了必要的batch_size, learning_rate等参数
    - 进一步优化huggingface transformers输入输出接口，支持20余种数据集格式
    - 一键使用模型，让领域专家精力集中在分析问题上
    

下一步计划：  
- [ ] 数据集支持：支持常用**标注工具**数据集  
    - [ ] 精灵标注助手  
    - [ ] Doccano  
    - [ ] universal data annotator
- [ ] **专题支持**  
    - [ ] 无监督实体/短语/固定搭配挖掘  
    - [ ] 气候变化文本分析工具箱  
    - [ ] 环境领域实体  
- [ ] 更新文档和案例  
        

如果您觉得本项目有用或是有帮助到您，麻烦您点击一下右上角的star :star:。您的支持是我们维护项目的最大动力:metal:！


# 使用方法

### 1. 安装

python环境配置

```bash
pip install envtext

#国内用户使用清华镜像加速
pip install envtext -i https://pypi.tuna.tsinghua.edu.cn/simple 
```
由于envtext库依赖于transformers和pytorch，故安装时间比较长，建议等待时喝一杯咖啡:coffee:。


### 2. 推理

目前支持的模型有：

| 任务名称 | Bert模型 | RNNs模型 | 其他模型 |
| ------ | ------ | ------ | ------ |
| 完型填空 | BertMLM  |  ------  |  ------  |
|  分类   | BertCLS  |  RNNCLS  |  ------  |
| 情感分析（回归） | BertSA  |  RNNSA  |  ------  |
|  多选   |BertMultiChoice | RNNMultiChoice | ----- |
| 实体识别 | BertNER  | RNNNER  | -----    |
| 词向量  |  -----  |  -----   | Word2Vec |

除文本生成任务外，基本支持大部分模型。

Bert 支持环境领域大规模预训练模型`envBert`，也支持其他huggingface transformer的Bert模型。

RNNs模型包括`LSTM`,`GRU`,`RNN`三种，可以选择使用环境领域预训练的词向量初始化，也可以使用Onehot编码初始化。


#### 2.1 使用Bert

由于bert模型较大，建议从huggingface transformer上预先下载模型权重，
或者从我们提供的百度网盘链接上下载权重，保存到本地，方便使用。

百度网盘链接：  
链接：[百度网盘 envBert 模型](https://pan.baidu.com/s/1KNE5JnUoulLgVK9yW5WtAw)
提取码：lfwm 

```python
#导入完形填空模型(masked language model)
from envtext.models import BertMLM
model = BertMLM('celtics1863/env-bert-chinese')

#进行预测
model('[MASK][MASK][MASK][MASK]是各国政府都关心的话题')


#导出结果
model.save_result('result.csv')
```
#### 2.2 使用RNN

目前RNN的初始化接口没有完全与Bert同步，后续有同步计划，尽请期待。
```python
from envtext.models import RNNCLS

model = RNNCLS()
model.load('本地pytorch_model.bin所在文件夹')

#进行预测
model('气候[Mask][Mask]是各国政府都关心的话题')

#导出结果
model.save_result('result.csv')
```

#### 2.3 使用word2vec

envtext自带长度为64的预训练词向量。
```python
from envtext.models import load_word2vec

model = load_word2vec()

model.most_similar('环境保护')
```

### 3. 训练


```python
#导入分类模型(classification)
from envtext.models import BertCLS
model = BertCLS('celtics1863/env-bert-chinese')

model.load_dataset('数据集位置',task = 'cls',format = '数据集格式')

#模型训练
model.train()
```

或者：

```python
#导入分类模型(classification)
from envtext.models import BertCLS
from envtext.data.utils import load_dataset

datasets,config = load_dataset('数据集位置',task = 'cls',format = '数据集格式')
model = BertCLS('celtics1863/env-bert-chinese',config)

#模型训练
model.train(datasets)
```


更详细的教程，请参见我们的案例 [jupyter notebooks]('jupyter_notebooks')


# LISENCE
Apache Lisence


