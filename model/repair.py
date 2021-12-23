import re

import pandas as pd
from nlgeval import compute_metrics


def get_repair(input_text, pred_text):
    if("0x" in input_text and "0x" in pred_text):
        input_str = ""
        pred_str = ""
        for text in input_text.split():
            if(text.startswith("0x")):
                input_str = text
        for text in pred_text.split():
            if(text.startswith("0x")):
                pred_str = text
        pred_text = pred_text.replace(pred_str, input_str)
    elif("0x" in input_text and bool(re.search(r'\d', pred_text))):
        input_str = ""
        pred_str = ""
        for text in input_text.split():
            if(text.startswith("0x")):
                input_str = text
        for text in pred_text.split():
            if(text.isdigit()):
                pred_str = text
        pred_text = pred_text.replace(pred_str, input_str)
    elif ("[" in input_text and "[" in pred_text):
        input_str = ""
        pred_str = ""
        for text in input_text.split():
            if (text[0] == "[" and text[-1]=="]"):
                input_str = text
        for text in pred_text.split():
            if (text[0] == "[" and text[-2]=="]"):
                pred_str = text[:-1]
            if (text[0] == "[" and text[-1]=="]"):
                pred_str = text
        pred_text = pred_text.replace(pred_str, input_str)
    elif ("\'" in input_text and "\'" in pred_text):
        input_str = ""
        pred_str = ""
        for text in input_text.split():
            if (text[0] == "\'" and text[-1]=="\'"):
                input_str = text
        for text in pred_text.split():
            if (text[0] == "\'" and text[-2]=="\'"):
                pred_str = text[:-1]
            elif (text[0] == "\'" and text[-1]=="\'"):
                pred_str = text
            elif (text[0] == "\'"):
                pred_str = text
        pred_text = pred_text.replace(pred_str, input_str)
    elif (bool(re.search(r'\d', input_text)) and bool(re.search(r'\d', pred_text))):
        input_text_digit_list = []
        pred_text_digit_list = []
        for text in input_text.split():
            if (text.isdigit()):
                input_text_digit_list.append(text)
        for text in pred_text.split():
            if (text.isdigit()):
                pred_text_digit_list.append(text)
        if (len(input_text_digit_list) == len(pred_text_digit_list) == 1):
            pred_text = pred_text.replace(pred_text_digit_list[0], input_text_digit_list[0])
    return pred_text


df = pd.read_csv("I2S_Input.csv", header=None)
I2S_list = df[0].tolist()

df = pd.read_csv("Trans_I2S_adj.csv", header=None)
pred_list = df[0].tolist()
data_list = []
for i in range(len(I2S_list)):
    data = get_repair(I2S_list[i], pred_list[i])
    data_list.append(data)
df = pd.DataFrame(data_list)
df.to_csv("Trans_I2S_adj_repair.csv", index=False, header=None)
metrics_dict = compute_metrics(hypothesis="Trans_I2S_adj_repair.csv",
                               references=['Trans_I2S_true.csv'],no_skipthoughts=True, no_glove=True)