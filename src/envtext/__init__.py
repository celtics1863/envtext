'''
**首款**中文环境领域文本分析工具。

特性：  
1. 支持中文环境领域大规模预训练模型**envBert**！

2. 支持中文环境领域大规模预训练**词向量**!

3. 支持中文环境领域专家过滤的**词表**!

4. **一且设计均为领域专家研究服务**：
    - 为神经网络模型精简了接口，只保留了必要的batch_size, learning_rate等参数
    - 进一步优化huggingface transformers输入输出接口，支持20余种数据集格式
    - 一键使用模型，让领域专家精力集中在分析问题上  
    
    
快速使用：

使用Bert模型

```python
from envtext.models import BertMLM
model = BertMLM('celtics1863/env-bert-chinese')
model('[MASK][MASK][MASK][MASK]是各国政府都关心的话题')
model.save_result('result.csv')
```

使用word2vec模型：
```python
from envtext.models import load_word2vec
model = load_word2vec()
model.most_similar('环境保护')
```

'''
__version__ = '0.0.1'
__license__ = 'Apache Software License'