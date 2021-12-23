import pandas as pd

df = pd.read_csv('ShellCodeGen_references.csv', header=None)
true_list = df[0].tolist()

df = pd.read_csv("Trans_ShellCodeGen_adj.csv", header=None)
pred_list = df[0].tolist()
count = 0
for i in range(len(pred_list)):
    if (pred_list[i] == true_list[i]):
        count += 1
print(str(count / len(pred_list)))