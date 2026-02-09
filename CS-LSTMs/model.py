from pytorch_lightning import LightningModule
from sklearn import logger
from torch.utils.data import DataLoader, DistributedSampler
from cs_lstms import CSLSTMs
from dataset import UniDataset
import argparse
from torch import optim
from collections import OrderedDict
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from get_f1_score import best_f1, delay_f1, best_f1_without_pointadjust
from sklearn.metrics import roc_auc_score


class MyCSLSTMs(LightningModule):
    """Context and Seasonal LSTMs"""

    def __init__(self, hparams):
        super(MyCSLSTMs, self).__init__()
        self.save_hyperparameters()
        self.hp = hparams
        self.__build_model()
        # self.json_list=[]
        self.test_step_outputs = []  # For PyTorch Lightning 2.0+

    def __build_model(self):
        self.cslstms = CSLSTMs(self.hp)

    def forward(self, x, x_normal, mode, mask):
        x = x.view(-1, 1, self.hp.window)
        return self.cslstms.forward(x, x_normal, mode, mask)

    def loss(self, x, y_all, z_all, x_normal, mode="train"):
        mask = torch.logical_not(torch.logical_or(y_all, z_all))
        loss = self.forward(
            x,
            x_normal,
            "train",
            mask,
        )
        return loss

    def training_step(self, data_batch, batch_idx):
        x, y_all, z_all = data_batch
        loss_val = self.loss(x, y_all, z_all, x)
        if self.trainer.strategy == "dp":
            loss_val = loss_val.unsqueeze(0)
        self.log("val_loss_train", loss_val, on_step=True, on_epoch=False, logger=True)
        output = OrderedDict(
            {
                "loss": loss_val,
            }
        )
        return output

    def validation_step(self, data_batch, batch_idx):
        x, y_all, z_all = data_batch
        loss_val = self.loss(x, y_all, z_all, x)
        if self.trainer.strategy == "dp":
            loss_val = loss_val.unsqueeze(0)
        self.log("val_loss_valid", loss_val, on_step=True, on_epoch=True, logger=True)
        output = OrderedDict(
            {
                "loss": loss_val,
            }
        )
        return output
        

    def test_step(self, data_batch, batch_idx):
        x, y_all, z_all = data_batch
        y = (y_all[:, -1]).unsqueeze(1)
        mask = torch.logical_not(torch.logical_or(y_all, z_all))
        with torch.no_grad():
            recon_prob, recon_x = self.forward(x, x,"test", mask)
        recon_prob = recon_prob[:, :, -1]
        recon_x = recon_x[:,:,-1]
        x = x[:,:,-1]
        output = OrderedDict(
            {
                "y": y,
                "x": x,
                "recon_prob": recon_prob,
                "recon_x" : recon_x,
            }
        )
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_start(self):
        """Clear test_step_outputs at the start of test epoch"""
        self.test_step_outputs.clear()

    def on_test_epoch_end(self):
        """Process test outputs at the end of test epoch"""
        outputs = self.test_step_outputs

        if not outputs:
            print("Warning: No test outputs collected")
            return

        y = torch.cat(([x["y"] for x in outputs]), 0)
        x = torch.cat(([x["x"] for x in outputs]), 0)
        recon_prob = torch.cat(([x["recon_prob"] for x in outputs]), 0)
        recon_x = torch.cat(([x["recon_x"] for x in outputs]), 0)
        score = recon_prob.squeeze(1).cpu().numpy()
        label = y.squeeze(1).cpu().numpy()
        df = pd.DataFrame()
        df["y"] = y.cpu().numpy().reshape(-1)
        df["x"] = x.cpu().numpy().reshape(-1)
        df["recon_x"] = recon_x.cpu().numpy().reshape(-1) 
        np.save("./npy/score.npy", score)
        np.save("./npy/label.npy", label)
        if self.hp.data_dir == "./data/Yahoo":
            k = 3
        elif self.hp.data_dir == "./data/AIOPS":
            k = 7
        elif self.hp.data_dir == "./data/NAB" or self.hp.data_dir == "./data/new_NAB":
            k = 150
        else:
            k = 40

        if len(np.unique(label)) > 1:
            auc = roc_auc_score(label, score)
        else:
            auc = np.nan
            logger.warning(f"AUC skipped due to single class in label: unique={np.unique(label)}")

        delay_f1_score, delay_precison, delay_recall, delay_predict = delay_f1(
            score, label, k
        )
        best_f1_socre, best_precison, best_recall, best_predict, thres = best_f1(score, label)
        best_f1_socre_, best_precison_, best_recall_, best_predict_ = (
            best_f1_without_pointadjust(score, label)
        )
        df["delay_predict"] = delay_predict
        df["best_predict"] = best_predict
        df.to_csv(
            "./csv/result.csv",
            index=False,
        )

        file_name = self.hp.save_file
        with open(file_name, "a") as f:
            f.write(
                "Thres %f \nAuc %f \nbest f1 score %f %f %f \nDelay f1 score  %f %f %f\nBest f1 without pointadjust %f %f %f\n"
                % (
                    auc,
                    thres,
                    best_f1_socre,
                    best_precison,
                    best_recall,
                    delay_f1_score,
                    delay_precison,
                    delay_recall,
                    best_f1_socre_,
                    best_precison_,
                    best_recall_,
                )
            )


    def mydataloader(self, mode):
        dataset = UniDataset(
            self.hp.use_label,
            self.hp.window,
            self.hp.data_dir,
            self.hp.data_name,
            mode,
            self.hp.sliding_window_size,
            data_pre_mode=self.hp.data_pre_mode,
        )
        train_sampler = None
        batch_size = self.hp.batch_size
        try:
            if self.on_gpu:
                train_sampler = DistributedSampler(dataset, rank=self.trainer.proc_rank)
                batch_size = batch_size // self.trainer.world_size  # scale batch size
        except Exception as e:
            pass
        should_shuffle = train_sampler is None
        if mode == "valid" or mode == "test":
            should_shuffle = False
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=self.hp.num_workers,
        )
        return loader

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hp.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--data_name", default="Yahoo", type=str
        )
        parser.add_argument("--data_dir", default="./data/WSD_test/", type=str)
        parser.add_argument("--window", default = 768, type=int)
        parser.add_argument("--d_model", default = 256, type=int)
        parser.add_argument("--dropout_rate", default = 0.1, type=float)
        parser.add_argument("--gpu", default=2, type=int)
        parser.add_argument("--use_gpu", default=True, type=bool)
        parser.add_argument("--use_label", default = 1, type=int)
        parser.add_argument("--seasonal_window", default = 128,type=int)
        parser.add_argument("--cycle", default = 128, type=int)
        parser.add_argument("--contextual_window", default = 12, type=int)
        parser.add_argument("--step", default=4, type=int)
        parser.add_argument("--sliding_window_size", default=1, type=int)
        parser.add_argument("--batch_size", default=512, type=int)
        parser.add_argument("--learning_rate", default=0.001, type=float)

        parser.add_argument("--only_test", default=0, type=int)
        parser.add_argument("--max_epoch", default= 1, type=int)
        parser.add_argument("--num_workers", default=64, type=int)
        parser.add_argument("--save_file", default="./result/Score.txt", type=str)
        parser.add_argument("--data_pre_mode", default=0, type=int)
        parser.add_argument("--eval_all", default=0, type=int)
        parser.add_argument("--kernel_size", default=16, type=int)
        parser.add_argument(
            "--ckpt_path", default="/home/hwces/zlp/FCVAE/ckpt/Yahoo.ckpt", type=str
        )
        return parser