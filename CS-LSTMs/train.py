from ast import arg
import os
import logging
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from model import MyCSLSTMs
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import time
import json
from collections import defaultdict

SEED = 8
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

logger = TensorBoardLogger(name="logs", save_dir="./")


def main(hparams):
    print("loading model...")
    model = MyCSLSTMs(hparams)
    print("model built")
    early_stop = EarlyStopping(
        monitor="val_loss_valid_epoch", patience=3, verbose=True, mode="min"
    )
    checkpoint = ModelCheckpoint(
        dirpath="./ckpt/",
        filename="{}".format(hparams.data_name),
        monitor="val_loss_valid_epoch",
        mode="min",
    )
    trainer = Trainer(
        max_epochs=hparams.max_epoch,
        callbacks=[early_stop, checkpoint],
        logger=logger,
        accelerator="gpu",
        devices=[hparams.gpu],
        check_val_every_n_epoch=1,
        gradient_clip_algorithm="value",
        gradient_clip_val=2,
    )
    print("fit start")
    train_loader = model.mydataloader("train")
    val_loader = model.mydataloader("valid")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    start_time = time.time()
    trainer.test(model, dataloaders=model.mydataloader("test"))
    end_time = time.time()
    print("View tensorboard logs by running\ntensorboard --logdir %s" % os.getcwd())
    print("and going to http://localhost:6006 on your browser")
    print(end_time - start_time)


def compute_json_values_mean(json_list):
    """
    读取文件内包含的JSON数组，对每个字段求均值。
    假设所有JSON对象字段一致且值为数字。

    参数:
        json_file_path (str): JSON文件路径

    返回:
        dict: 每个字段的均值字典
    """

    if not json_list:
        return {}

    field_sums = defaultdict(float)
    field_counts = defaultdict(int)

    for json_obj in json_list:
        if isinstance(json_obj, dict):
            for key, value in json_obj.items():
                if isinstance(value, (int, float)):
                    field_sums[key] += value
                    field_counts[key] += 1
        else:
            print(f"Warning: Skipping non-dictionary item in list: {json_obj}")

    averages = {}
    for key in field_sums:
        if field_counts[key] > 0:
            averages[key] = field_sums[key] / field_counts[key]

    return averages



if __name__ == "__main__":
    parser = MyCSLSTMs.add_model_specific_args()
    hyperparams = parser.parse_args()
    print(f"RUNNING")
    if hyperparams.only_test == 1:
        model = MyCSLSTMs.load_from_checkpoint(checkpoint_path=hyperparams.ckpt_path)
        model.hp = hyperparams
        print(model.hp)
        trainer = Trainer(accelerator="gpu", devices=[hyperparams.gpu])
        trainer.test(model, dataloaders=model.mydataloader("test"))
    else:
        main(hyperparams)

