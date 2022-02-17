from transformers import BertForMaskedLM,BertConfig
from ..data import DataCollatorForZHWholeWordMask
from transformers import Trainer, TrainingArguments
import math
from datasets import Dataset
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model',default = 'hfl/chinese-bert-wwm-ext')
parser.add_argument('--train')
parser.add_argument('--valid')
parser.add_argument('-s','--save_path',default = 'whole_phrase_model')
parser.add_argument('-c','--checkpoint_path',default = 'env_phrase_checkpoint')
parser.add_argument('-v','--jieba_vocab',default = '../files/env_vocab.jieba.txt')
parser.add_argument('-l','--logging_dir',default = 'runs')


args = parser.parse_args()

from envText.models import BertMLM
model = BertMLM(args.model,tokenize_chinese_char = False)

#换成其他分词器，可以训练其他的模型
def pre_tokenizer(text):
    import jieba
    jieba.load_usrdict(args.jieba_vocab)
    return jieba.lcut(text)


print("*"*20,"正在读取数据","*"*20)

if not os.path.exists(args.train):
    assert 0,'请输入正确的训练数据集路径'
    
if not os.path.exists(args.valid):
    assert 0,'请输入正确的验证数据集路径'

def txt_generator(path):
    f = open(path, 'r',encoding = 'utf-8')
    for idx,line in enumerate(path):
        yield pre_tokenizer(line.strip())
        
print("*"*20,"tokenize 训练集","*"*20)
train_txt = list(map(txt_generator,args.train))
train_tokens = model.tokenizer(train_txt,max_length=512,padding='max_length',truncation=True,return_tensors='pt')['input_ids']
del train_txt

print("*"*20,"tokenize 验证集","*"*20)
valid_txt = list(map(txt_generator,args.train))
valid_tokens = model.tokenizer(valid_txt,max_length=512,padding='max_length',truncation=True,return_tensors='pt')['input_ids']
del valid_txt


print("*"*20,"正在整理为dataset","*"*20)
train_dataset = Dataset.from_dict(train_tokens)
valid_dataset = Dataset.from_dict(valid_tokens)


training_args = TrainingArguments(
    args.checkpoint_path,
    evaluation_strategy = "steps",
    eval_steps = 2000,
    num_train_epochs = 3,
    per_device_train_batch_size=26,
    per_device_eval_batch_size=26,
    gradient_accumulation_steps = 10,
    warmup_steps = 200,
    learning_rate= 2e-5, 
    weight_decay=0.01,
    push_to_hub=False,
    logging_steps = 100,
    logging_first_step = True,
    save_steps = 200,
    seed = random.randint(0,100),
    save_total_limit = 3,
    logging_dir = args.logging_dir
    )

data_collator = DataCollatorForZHWholeWordMask(tokenizer=model.tokenizer, mlm_probability=0.15)

print("*"*20,"开始训练","*"*20)
trainer = Trainer(
    model=model.model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model(args.save_path)
