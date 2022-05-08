from tokenizers import ByteLevelBPETokenizer
import os
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=["data/train.csv", "data/valid.csv", "data/test.csv"],
vocab_size=52000, min_frequency=1, special_tokens=["<s>", "<pad>", "</s>", "<unk>"])
tokenizer.save_model("./Config")
# 训练完分词器 修改config.json中的vocab_size