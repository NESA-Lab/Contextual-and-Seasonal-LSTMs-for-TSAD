import torch
import torch.utils.data
import logging
import numpy as np
import pandas as pd
import os
import datapreprocess


class UniDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        use_label,
        window,
        data_dir,
        data_name,
        mode,
        sliding_window_size,
        data_pre_mode=0,
    ):
        self.window = window
        self.data_dir = data_dir
        self.data_name = data_name
        file_list = os.listdir(data_dir)
        value_all = []
        label_all = []
        missing_all = []
        self.len = 0
        self.sample_num = 0
        for file in file_list:
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)
            df_train = df[int(0 * len(df)): int(0.35 * len(df))]
            df_train = df_train.ffill()
            train_value = np.asarray(df_train["value"])
            train_label = np.asarray(df_train["label"])
            #只保留normal点,计算最大最小值
            train_value = train_value[np.where(train_label == 0)[0]]
            train_max = train_value.max()
            train_min = train_value.min()
            if mode == "train":
                df = df[int(0 * len(df)):int(0.35 * len(df))]
            elif mode == "valid":
                df = df[int(0.35 * len(df)):int(0.5 * len(df))]
            elif mode == "test":
                df = df[int(0.5 * len(df)):]
                # df = df[int(0 * len(df)):int(0.5*len(df))]
            timestamp, missing, (value, label) = datapreprocess.complete_timestamp(
                df["timestamp"], (df["value"], df["label"])
            )
            value = value.astype(float)
            missing2 = np.isnan(value)
            missing = np.logical_or(missing, missing2).astype(int)
            label = label.astype(float)
            label[np.where(missing == 1)[0]] = np.nan
            value[np.where(missing == 1)[0]] = np.nan
            df2 = pd.DataFrame()
            df2["timestamp"] = timestamp
            df2["value"] = value
            df2["label"] = label
            df2["missing"] = missing.astype(int)
            df2 = df2.ffill()
            df2 = df2.fillna(0)
            df2["label"] = df2["label"].astype(int)
            if data_pre_mode == 0:
                df2["value"], *_ = datapreprocess.standardize_kpi(df2["value"])
            else:
                v = np.asarray(df2["value"])
                v = 2 * (v - train_min) / (train_max - train_min) - 1
                df2["value"] = v
            timestamp, values, labels = (
                np.asarray(df2["timestamp"]),
                np.clip(np.asarray(df2["value"]), -40, 40),
                np.asarray(df2["label"]),
            )
            values[np.where(missing == 1)[0]] = 0

            #设为0，防止学习到错误模式
            if (mode == "train" or mode == "valid") and use_label == 1:
                # values[np.where(labels==1)]==0
                pass
            elif (mode == "train" or mode == "valid") and use_label == 0:#标签全部设置为0
                labels[:] = 0
            else:
                pass

            # 平滑数据
            if (mode == "train" or mode == "valid") and use_label == 0:
                values = np.convolve(
                    values,
                    np.ones((sliding_window_size,)) / sliding_window_size,
                    mode="valid",
                )
            timestamp = timestamp[sliding_window_size - 1 :]
            labels = labels[sliding_window_size - 1 :]
            missing = missing[sliding_window_size - 1 :]
            value_all.append(values)
            label_all.append(labels)
            missing_all.append(missing)
            self.sample_num += max(len(values) - window + 1, 0)
        self.samples, self.labels, self.miss_label = self.__getsamples(
            value_all, label_all, missing_all
        )

#按window大小切分数据集
    def __getsamples(self, values, labels, missing):
        #数据格式（shape）
        X = torch.zeros((self.sample_num, 1, self.window))
        Y = torch.zeros((self.sample_num, self.window))
        Z = torch.zeros((self.sample_num, self.window))
        i = 0
        for cnt in range(len(values)):
            v = values[cnt]
            l = labels[cnt]
            m = missing[cnt]
            for j in range(len(v) - self.window + 1):
                X[i, 0, :] = torch.from_numpy(v[j : j + self.window])
                Y[i, :] = torch.from_numpy(np.asarray(l[j : j + self.window]))
                Z[i, :] = torch.from_numpy(np.asarray(m[j : j + self.window]))
                i += 1   
        return (X, Y, Z)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        sample = [self.samples[idx, :, :], self.labels[idx, :], self.miss_label[idx, :]]
        return sample

# import torch
# import torch.utils.data
# import logging
# import numpy as np
# import pandas as pd
# import os
# import datapreprocess


# class UniDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         use_label,
#         window,
#         data_dir,
#         data_name,
#         mode,
#         sliding_window_size,
#         data_pre_mode=0,
#         test_sampling_interval=24,  # 参数不变，表示跳跃间隔
#     ):
#         self.window = window
#         self.data_dir = data_dir
#         self.data_name = data_name
#         self.mode = mode
#         self.test_sampling_interval = test_sampling_interval
#         file_list = os.listdir(data_dir)
#         value_all = []
#         label_all = []
#         missing_all = []
#         self.len = 0
#         self.sample_num = 0

#         for file in file_list:
#             file_path = os.path.join(data_dir, file)
#             df = pd.read_csv(file_path)

#             df_train_cal_stats = df[int(0 * len(df)): int(0.35 * len(df))].copy()
#             df_train_cal_stats = df_train_cal_stats.ffill()
#             train_value = np.asarray(df_train_cal_stats["value"])
#             train_label = np.asarray(df_train_cal_stats["label"])
#             train_value = train_value[np.where(train_label == 0)[0]]
#             train_max = train_value.max()
#             train_min = train_value.min()

#             if mode == "train":
#                 df = df[int(0 * len(df)):int(0.35 * len(df))]
#             elif mode == "valid":
#                 df = df[int(0.35 * len(df)):int(0.5 * len(df))]
#             elif mode == "test":
#                 df = df[int(0.5 * len(df)):]
            
#             timestamp, missing, (value, label) = datapreprocess.complete_timestamp(
#                 df["timestamp"], (df["value"], df["label"])
#             )
#             value = value.astype(float)
#             missing2 = np.isnan(value)
#             missing = np.logical_or(missing, missing2).astype(int)
#             label = label.astype(float)
#             label[np.where(missing == 1)[0]] = np.nan
#             value[np.where(missing == 1)[0]] = np.nan
            
#             df2 = pd.DataFrame()
#             df2["timestamp"] = timestamp
#             df2["value"] = value
#             df2["label"] = label
#             df2["missing"] = missing.astype(int)
#             df2 = df2.ffill()
#             df2 = df2.fillna(0)
#             df2["label"] = df2["label"].astype(int)

#             if data_pre_mode == 0:
#                 df2["value"], *_ = datapreprocess.standardize_kpi(df2["value"])
#             else:
#                 v = np.asarray(df2["value"])
#                 v = 2 * (v - train_min) / (train_max - train_min) - 1
#                 df2["value"] = v
            
#             timestamp, values, labels = (
#                 np.asarray(df2["timestamp"]),
#                 np.clip(np.asarray(df2["value"]), -40, 40),
#                 np.asarray(df2["label"]),
#             )
#             values[np.where(missing == 1)[0]] = 0

#             if (mode == "train" or mode == "valid") and use_label == 1:
#                 pass
#             elif (mode == "train" or mode == "valid") and use_label == 0:
#                 labels[:] = 0
#             else:
#                 pass

#             if (mode == "train" or mode == "valid") and use_label == 0:
#                 values = np.convolve(
#                     values,
#                     np.ones((sliding_window_size,)) / sliding_window_size,
#                     mode="valid",
#                 )
#                 timestamp = timestamp[sliding_window_size - 1 :]
#                 labels = labels[sliding_window_size - 1 :]
#                 missing = missing[sliding_window_size - 1 :]
#             else:
#                 timestamp = timestamp[sliding_window_size - 1 :]
#                 labels = labels[sliding_window_size - 1 :]
#                 missing = missing[sliding_window_size - 1 :]

#             value_all.append(values)
#             label_all.append(labels)
#             missing_all.append(missing)
            
#             # 重新计算 sample_num
#             # 在测试模式下，需要确保序列足够长，才能采样出 window 个点
#             # 最后一个点索引：(window - 1) * test_sampling_interval
#             # 那么需要总长度至少为 (window - 1) * test_sampling_interval + 1
#             if self.mode == "test":
#                 required_length = (self.window - 1) * self.test_sampling_interval + 1
#                 if len(values) >= required_length:
#                     # 每个可能的起始点都可以生成一个样本
#                     # 起始点可以从 0 到 len(values) - required_length
#                     self.sample_num += (len(values) - required_length + 1)
#                 else:
#                     self.sample_num += 0 # 不能生成任何样本
#             else:
#                 # 正常滑动窗口的样本数量
#                 self.sample_num += max(len(values) - window + 1, 0)
        
#         self.samples, self.labels, self.miss_label = self.__getsamples(
#             value_all, label_all, missing_all
#         )

#     def __getsamples(self, values_list, labels_list, missing_list):
#         # 预先分配内存
#         X = torch.zeros((self.sample_num, 1, self.window))
#         Y = torch.zeros((self.sample_num, self.window))
#         Z = torch.zeros((self.sample_num, self.window))
        
#         i = 0 # 实际生成的样本索引
        
#         for cnt in range(len(values_list)):
#             v = values_list[cnt]
#             l = labels_list[cnt]
#             m = missing_list[cnt]
            
#             if self.mode == "test":
#                 # 计算需要确保序列的最小长度
#                 required_length = (self.window - 1) * self.test_sampling_interval + 1

#                 # 遍历所有可能的样本起始点
#                 # 起始点可以从 0 到 len(v) - required_length
#                 for start_idx in range(len(v) - required_length + 1):
#                     # 构建跳跃式采样的索引
#                     indices = np.arange(start_idx, start_idx + self.window * self.test_sampling_interval, self.test_sampling_interval)
                    
#                     # 确保索引在有效范围内，尽管前面的range已经限制了
#                     # 但是为了代码健壮性，可以再检查一下，不过np.arange通常是安全的
                    
#                     X[i, 0, :] = torch.from_numpy(v[indices])
#                     Y[i, :] = torch.from_numpy(np.asarray(l[indices]))
#                     Z[i, :] = torch.from_numpy(np.asarray(m[indices]))
#                     i += 1
#             else:
#                 # 训练和验证模式下，仍是标准的滑动窗口
#                 for j in range(len(v) - self.window + 1):
#                     X[i, 0, :] = torch.from_numpy(v[j : j + self.window])
#                     Y[i, :] = torch.from_numpy(np.asarray(l[j : j + self.window]))
#                     Z[i, :] = torch.from_numpy(np.asarray(m[j : j + self.window]))
#                     i += 1
        
#         # 检查实际生成的样本数并调整
#         if i != self.sample_num:
#             logging.warning(f"Expected {self.sample_num} samples, but got {i} samples. Adjusting dataset size.")
#             self.sample_num = i
#             X = X[:self.sample_num]
#             Y = Y[:self.sample_num]
#             Z = Z[:self.sample_num]
            
#         return (X, Y, Z)

#     def __len__(self):
#         return self.sample_num

#     def __getitem__(self, idx):
#         sample = [self.samples[idx, :, :], self.labels[idx, :], self.miss_label[idx, :]]
#         return sample