from envtext import  BertNER,AlbertNER,Config
# model = BertNER("../../../2022Spring/EnvText/models/pos/",visualizer="pos")
# model = AlbertNER(Config.albert.pos_ner,visualizer="pos")

model = AlbertNER("../../envpos/envpos/files/pretrained_models/pos-albert/",visualizer="pos")

model("在全球气候大会上，气候变化是各国政府都关心的话题")