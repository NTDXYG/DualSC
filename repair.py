import re

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
