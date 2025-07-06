import pandas as pd
import os

input_dir="./NAB"
dfs = []
for file in os.listdir(input_dir):
    df = pd.read_csv(os.path.join(input_dir,file))
    df = df.dropna()
    label = df["label"]
    dfs.append(label)

label = pd.concat(dfs).values
ratio = float(sum(label)/len(label))
print(len(label))
print(ratio)