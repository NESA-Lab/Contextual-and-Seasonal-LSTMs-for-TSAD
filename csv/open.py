import pandas as pd  
import matplotlib.pyplot as plt  
import numpy as np
from tqdm import tqdm
import os
csv_folder = './Yahoo'
  
# 获取文件夹中的所有CSV文件  
csv_files = sorted([file for file in os.listdir(csv_folder) if file.endswith('.csv')])
  
# 遍历每个CSV文件并转换成图片  
for file in csv_files:
	data= pd.read_csv(os.path.join(csv_folder, file))
	data =data[20:]
	data=data[(len(data)*0//100):(len(data)*10//100)]
	datasize = 100
	duan=data.shape[0]//datasize
	random_start_indices = np.array([0])
	o_label=data['y'].values
	# label=data['delay_predict'].values
	b_label=data["best_predict"].values
	# p_label=data["predict"].values

	split_data =np.array_split(data['x'].values, duan)
	# split_data =np.array_split(data['mu_x_test'].values, duan)
	o_split_label=np.array_split(o_label, duan)
	b_split_label=np.array_split(b_label, duan)
	# p_split_label=np.array_split(p_label, duan)

	selected_data = np.array([split_data[i][random_start_indices[0]:random_start_indices[0] + datasize] for i in range(duan)]).reshape(duan, datasize)
	o_selected_label=np.array([o_split_label[i][random_start_indices[0]:random_start_indices[0] + datasize] for i in range(duan)]).reshape(duan, datasize)
	b_selected_label=np.array([b_split_label[i][random_start_indices[0]:random_start_indices[0] + datasize] for i in range(duan)]).reshape(duan, datasize)
	# p_selected_label=np.array([p_split_label[i][random_start_indices[0]:random_start_indices[0] + datasize] for i in range(duan)]).reshape(duan, datasize)
	
	#原始标签
	displayed_segments = 0
	for j in range(duan):
		if 1.0 in o_selected_label[j]:
			displayed_segments=displayed_segments+1

	fig, axs = plt.subplots(displayed_segments, 1, figsize=(10, displayed_segments))
	temp = 0
	for j in tqdm(range(duan),desc="Origin", unit="iteration"):
		if 1.0 in o_selected_label[j]:
			color = 'black'
			for i in range(datasize-1):
				if o_selected_label[j][i] != o_selected_label[j][i+1]:
					axs[temp].plot([i, i+1], [selected_data[j, i], selected_data[j, i+1]], color=color if o_selected_label[j][i] == 0.0 else 'red',linewidth=4)
				else:
					axs[temp].plot([i, i+1], [selected_data[j, i], selected_data[j, i+1]], color=color if o_selected_label[j][i] == 0.0 else 'red', linestyle='-', solid_capstyle='round',linewidth=4)
			axs[temp].set_xticks([])  # 隐藏 x 轴刻度
			axs[temp].tick_params(axis='y', labelsize=14, width=2)
			# axs[temp].set_yticks([])  # 隐藏 y 轴刻度
			axs[temp].spines['top'].set_visible(False)    # 隐藏顶部边框
			axs[temp].spines['right'].set_visible(False)  # 隐藏右侧边框
			# axs[temp].spines['left'].set_visible(False)   # 隐藏左侧边框
			# axs[temp].spines['bottom'].set_visible(False) # 隐藏底部边框			
			temp=temp+1
	plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3)
	plt.savefig(os.path.join(csv_folder, file.replace('.csv', '_cl-lstm_predict.png')))
	plt.close()

	#未预测成功标签
	fig, axs = plt.subplots(displayed_segments, 1, figsize=(10, displayed_segments))
	temp = 0
	for j in tqdm(range(duan), desc="loss", unit="iteration"):
		if 1.0 in o_selected_label[j]:
			for i in range(datasize - 1):
				# 颜色判断逻辑
				if o_selected_label[j][i] == 1.0 and b_selected_label[j][i] == 1.0:
					color = 'blue'
				elif o_selected_label[j][i] == 1.0 and b_selected_label[j][i] == 0.0:
					color = 'red'
				else:
					color = 'black'

				if o_selected_label[j][i] != o_selected_label[j][i + 1]:
					axs[temp].plot([i, i + 1], [selected_data[j, i], selected_data[j, i + 1]], color=color,linewidth=4)
				else:
					axs[temp].plot([i, i + 1], [selected_data[j, i], selected_data[j, i + 1]], color=color, linestyle='-', solid_capstyle='round',linewidth=4)
			axs[temp].set_xticks([])  # 隐藏 x 轴刻度
			axs[temp].tick_params(axis='y', labelsize=14, width=2)
			# axs[temp].set_yticks([])  # 隐藏 y 轴刻度
			axs[temp].spines['top'].set_visible(False)    # 隐藏顶部边框
			axs[temp].spines['right'].set_visible(False)  # 隐藏右侧边框
			# axs[temp].spines['left'].set_visible(False)   # 隐藏左侧边框
			# axs[temp].spines['bottom'].set_visible(False) # 隐藏底部边框			
			temp = temp + 1  # 这里的缩进也要正确
	plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, hspace=0.3)
	plt.savefig(os.path.join(csv_folder, file.replace('.csv', '_cl-lstm_loss.png')))
	plt.close()
print("finish")