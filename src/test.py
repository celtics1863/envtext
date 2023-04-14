from envtext import  BertCLS,BertNER,AlbertNER,Config,CNNCLS,RNNCLS,TFIDFTokenizer,TFIDFCLS,RNNMultiChoice

model = BertCLS(Config.bert.bert_mlm)

model.load_dataset("isClimate")
model.train(epoch=3,batch_size=4)

# model = BertNER("../../../2022Spring/EnvText/models/pos/",visualizer="pos")
# model = AlbertNER(Config.albert.pos_ner,visualizer="pos")

# model = AlbertNER("../../envpos/envpos/files/pretrained_models/pos-albert/",visualizer="pos")

# model("在全球气候大会上，气候变化是各国政府都关心的话题")

# model = CNNCLS(max_length=128)

# model = CNNCLS(max_length=128,num_labels=2,kernel_size=7)

# model = RNNCLS(max_length=128,num_labels=2,kernel_size=7)

# model.load_dataset("../../../2022Spring/EnvText/EnvCLUE数据集/CLS_PaperSubject.json",task="cls")

# print(model.data_config)

# model.train(learning_rate = 1e-3,epoch=20,batch_size=16)

# model("呵呵哈哈哈")

# model = TFIDFCLS(num_layers=1)
# # model("哈哈哈。呵呵哈哈哈")


model = RNNMultiChoice()

model.load_dataset()

# model.load_dataset("../../../2022Spring/EnvText/EnvCLUE数据集/CLS_PaperSubject.json",task="cls")
# model.train(learning_rate = 1e-3,epoch=20,batch_size=16)

# tokenizer = TFIDFTokenizer(vocab_path="default")
# print((Config.env_vocab))