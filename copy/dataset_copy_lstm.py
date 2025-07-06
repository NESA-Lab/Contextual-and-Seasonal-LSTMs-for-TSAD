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
        cycle,
        window,
        data_dir,
        data_name,
        mode,
        data_pre_mode=0,
    ):
        self.cycle = cycle
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
            df_origin = pd.read_csv(file_path)
            for num in range(cycle):
                df = df_origin.iloc[::cycle, :].reset_index(drop=True)
                df_train = df[: int(0.35 * len(df))]
                df_train = df_train.ffill()
                train_value = np.asarray(df_train["value"])
                train_label = np.asarray(df_train["label"])
                #只保留normal点,计算最大最小值
                train_value = train_value[np.where(train_label == 0)[0]]
                train_max = train_value.max()
                train_min = train_value.min()
                if mode == "train":
                    df = df[: int(0.35 * len(df))]
                elif mode == "valid":
                    df = df[int(0.35 * len(df)) : int(0.5 * len(df))]
                elif mode == "test":
                    df = df[int(0.5 * len(df)) :]
                # timestamp, missing, (value, label) = datapreprocess.complete_timestamp(
                #     df["timestamp"], (df["value"], df["label"])
                # )
                value = df["value"]
                missing = np.isnan(value)
                label = df["label"].astype(float)
                # label[np.where(missing == 1)[0]] = np.nan
                # value[np.where(missing == 1)[0]] = np.nan
                df2 = pd.DataFrame()
                df2["value"] = value
                df2["label"] = label
                df2["missing"] = missing.astype(float)
                df2["label"] = df2["label"].astype(float)
                df2 = df2.ffill()
                df2 = df2.fillna(0)

                if data_pre_mode == 0:
                    df2["value"], *_ = datapreprocess.standardize_kpi(df2["value"])
                else:
                    v = np.asarray(df2["value"])
                    v = 2 * (v - train_min) / (train_max - train_min) - 1
                    df2["value"] = v
                values, labels = (
                    np.asarray(df2["value"]),
                    np.asarray(df2["label"]),
                )
                values[np.where(missing == 1)[0]] = 0

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
        X = torch.zeros((self.sample_num, self.window))
        Y = torch.zeros((self.sample_num, self.window))
        Z = torch.zeros((self.sample_num, self.window))
        i = 0
        for cnt in range(len(values)):
            v = values[cnt]
            l = labels[cnt]
            m = missing[cnt]
            for j in range(len(v) - self.window + 1):
                X[i, :] = torch.from_numpy(v[j : j + self.window])
                Y[i, :] = torch.from_numpy(np.asarray(l[j : j + self.window]))
                Z[i, :] = torch.from_numpy(np.asarray(m[j : j + self.window]))
                i += 1   
        return (X, Y, Z)
    

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        sample = [self.samples[idx, :], self.labels[idx, :], self.miss_label[idx, :]]
        return sample
