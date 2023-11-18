import numpy as np
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
import yaml

import reconstruction_deep_network
from reconstruction_deep_network.data_loader.custom_loader import CustomDataLoader
from reconstruction_deep_network.trainer.trainer import ModelTrainer

def parse_args():

    parser = ArgumentParser()
    parser.add_argument("--main_config_path", type = str, dest = "main_config_path")
    parser.add_argument("--num_workers", type = int, dest = "num_workers")
    parser.add_argument("--exp_name", dest = "exp_name", type = str)
    parser.add_argument("--batch_size", dest = "batch_size", type = int)
    parser.add_argument("--max_epochs", dest = "max_epochs", type = int)
    parser.add_argument("--learning_rate", dest = "learning_rate", type = float)
    parser.add_argument("--ckpt_path", dest = "ckpt_path", type = str)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    return args

def main(args):

    config_file_path = args.main_config_path
    with open(config_file_path, 'r') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)
    
    config["train"]["learning_rate"] = args.learning_rate
    config["train"]["max_epochs"] = args.max_epochs
    config["train"]["batch_size"] = args.batch_size

    train_dataset = CustomDataLoader(mode = "train")
    val_dataset = CustomDataLoader(mode = "val")

    train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size = config["train"]["batch_size"],
                    shuffle = True,
                    num_workers = args.num_workers,
                    drop_last = True)
    
    val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size = 1,
                    shuffle = False,
                    num_workers = args.num_workers,
                    drop_last = False)
    
    model_trainer = ModelTrainer()

    if args.ckpt_path is not None:
        model_trainer.load_state_dict(torch.load(args.ckpt_path, map_location='cpu')[
            'state_dict'], strict=False)

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor=["train_loss", "fid_score"],
                                          mode="min", save_last=1,
                                          filename='epoch={epoch}-loss={train_loss:.4f}-fid={fid_score:.2f}')
    

    logger = TensorBoardLogger(
        save_dir='logs/tb_logs', name=args.exp_name, default_hp_metric=False)
    
    training_pipeline = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        logger=logger)
    
    training_pipeline.fit(model_trainer, train_loader, val_loader)

if __name__ == "__main__":
    args = parse_args()
    main(args)

