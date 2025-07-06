import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler

src_dir = './WSD'
file_list = os.listdir(src_dir)
df_all = pd.DataFrame()
scaler = StandardScaler()
with tqdm(total=len(file_list)) as pbar:
    for file in file_list:
        file = os.path.join(src_dir,file)
        df = pd.read_csv(file)
        df = df.drop(["timestamp","label"],axis=1)
        df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        df_all = pd.concat([df_all,df_standardized])
        pbar.update(1)

for i in range(6):
    df_all[f'shift_{i}']=df_all["value"].shift(i*30)


df_all = df_all.reset_index(drop=True)
df_all = df_all.bfill()
df_all = df_all.fillna(0)
df_all = df_all[10000:12000]
sample_size = 200

num_colors = len(df_all.columns)
num_rows = int(np.ceil(len(df_all) / sample_size))
color_map = cm.get_cmap("tab20",num_colors)

print(num_colors)

fig, axes = plt.subplots(num_rows, 1, figsize=(20, 3*num_rows))

# 分段绘制
for row in range(num_rows):
    start_idx = row * 200
    end_idx = min((row + 1) * 200, len(df_all))
    df_segment = df_all.iloc[start_idx:end_idx]  # 提取当前段的数据
    
    # 绘制每个维度的曲线
    for i, col in enumerate(df_segment.columns):
        axes[row].plot(
            df_segment.index,
            df_segment[col],
            label=col,
            color=color_map(i)
        )
    axes[row].legend(loc="upper left", fontsize=10)  # 添加图例
    axes[row].grid(True, linestyle="--", alpha=0.6)

# 添加全局X/Y标签
fig.supxlabel("Time", fontsize=12)
fig.supylabel("Value", fontsize=12)

plt.tight_layout()
plt.savefig("overlap_series_WSD.pdf")