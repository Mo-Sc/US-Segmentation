import os
import sys
from datetime import datetime
import shutil
import logging

import sys

# workaround to not have to change imports in MedSAM code
sys.path.append("MedSAM")

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import DiceLoss
from sklearn.model_selection import KFold

from MedSAM.segment_anything import sam_model_registry

from utils.train_utils import (
    plot_prediction,
    plot_losses,
    NpyDataset,
    InferenceNpyDataset,
    MedSAM,
)

# Load configuration
from config import train_config_CV as config


# Set seeds
torch.manual_seed(config["random_state"])
torch.cuda.empty_cache()

# Environment settings for training
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

# each run has unique id based on timestamp. Used to store model and logs
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = os.path.join(config["work_dir"], config["task_name"] + "-" + run_id)
os.makedirs(model_save_path, exist_ok=True)
device = torch.device(config["device"])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(model_save_path, "training.log")),
        logging.StreamHandler(),
    ],
)

import utils.mlflow as mlflow
from mlflow_config import config_mlflow

mf = mlflow.mlflow_start(config_mlflow)


def create_dataloaders(config, train_scans, val_scans):
    train_dataset = NpyDataset(config["npy_base_path"], train_scans, no_bbox=True)
    validation_dataset = InferenceNpyDataset(
        config["npy_base_path"], val_scans, validation=True
    )

    mf.log_param("n training images", len(train_dataset))
    mf.log_param("n validation images", len(validation_dataset))

    logging.info(f"Number of training samples: {len(train_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    return train_dataloader, validation_dataloader


def initialize_model(config, device):
    sam_model = sam_model_registry[config["model_type"]](
        checkpoint=config["checkpoint"]
    )
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    for param in medsam_model.prompt_encoder.parameters():
        param.requires_grad = False

    return medsam_model


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0
    for image, gt2D, boxes, _ in tqdm(dataloader):
        optimizer.zero_grad()
        boxes_np = boxes.detach().cpu().numpy()
        image, gt2D = image.to(device), gt2D.to(device)

        medsam_pred = model(image, boxes_np)
        loss = loss_fn(medsam_pred, gt2D)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def validate_epoch(
    model,
    dataloader,
    loss_fn,
    dice_metric,
    hausdorff_metric,
    device,
    config,
    epoch,
    model_save_path,
):
    model.eval()
    val_loss = 0

    spacing = [0.0390625, 0.0390625]  # in mm (40mm/1024px)

    with torch.no_grad():
        for image, gt2D, filenames in dataloader:
            image, gt2D = image.to(device), gt2D.to(device)
            boxes_np = np.array([[0, 0, image.shape[2], image.shape[3]]])
            boxes_np = torch.tensor(boxes_np).float().to(device)

            medsam_pred = model(image, boxes_np)
            loss = loss_fn(medsam_pred, gt2D)
            val_loss += loss.item()

            medsam_pred_bin = (torch.sigmoid(medsam_pred) > 0.5).float()
            dice_metric(medsam_pred_bin, gt2D)

            hausdorff_metric(medsam_pred_bin, gt2D, spacing=spacing)

            if config["plot_val_pred"]:
                pred_mask = medsam_pred[0].cpu().numpy()
                gt2D = gt2D[0].cpu().numpy()
                img = image[0].cpu().permute(1, 2, 0).numpy()
                boxes_np = boxes_np.detach().cpu().numpy()
                plot_prediction(
                    img, gt2D, pred_mask, boxes_np, filenames, model_save_path, epoch
                )

    val_loss /= len(dataloader)
    val_dice = dice_metric.aggregate().item()
    val_hausdorff = hausdorff_metric.aggregate().item()
    dice_metric.reset()
    hausdorff_metric.reset()

    return val_loss, val_dice, val_hausdorff  # Return the Hausdorff distance


def save_checkpoint(state, filename="checkpoint"):
    torch.save(state, filename + ".pth")
    torch.save(state["model"], filename + "_weights_only.pth")


def main():

    # copy training script and config to model save path
    shutil.copyfile(
        __file__,
        os.path.join(model_save_path, run_id + "_" + os.path.basename(__file__)),
    )

    shutil.copyfile(
        "config.py",
        os.path.join(model_save_path, run_id + "_" + os.path.basename("config.py")),
    )

    with mf.start_run(run_name=f"{run_id}_CV_{config['k_folds']}"):

        mf.log_params(config)
        mf.log_param("Checkpoint path", model_save_path)
        mf.log_param("Run ID", run_id)

        kfold = KFold(
            n_splits=config["k_folds"],
            random_state=config["random_state"],
            shuffle=True,
        )

        fold_dices = []
        fold_best_dices = []
        fold_hausdorffs = []

        scans = [
            filename.split(".")[0]
            for filename in os.listdir(os.path.join(config["npy_base_path"], "imgs"))
            if filename.endswith(".npy")
        ]

        for fold, (train_idx, test_idx) in enumerate(kfold.split(scans)):

            print(f"---Fold {fold}---")

            with mf.start_run(nested=True):

                model_save_path_fold = os.path.join(
                    config["work_dir"],
                    config["task_name"] + "-" + run_id,
                    f"fold_{fold}",
                )

                mf.log_param("Fold Checkpoint path", model_save_path_fold)
                mf.log_param("Fold", fold)

                os.makedirs(model_save_path_fold, exist_ok=True)

                train_scans = [scans[i] for i in train_idx]
                val_scans = [scans[i] for i in test_idx]

                # create dataloaders from custom npy dataset
                train_dataloader, val_dataloader = create_dataloaders(
                    config, train_scans, val_scans
                )

                # intialized model and optimizer. Can probably be moved outside of cv
                medsam_model = initialize_model(config, device)
                optimizer = torch.optim.AdamW(
                    medsam_model.parameters(),
                    lr=config["lr"],
                    weight_decay=config["weight_decay"],
                )

                loss_fn = lambda pred, target: DiceLoss(
                    sigmoid=True, squared_pred=True, reduction="mean"
                )(pred, target) + nn.BCEWithLogitsLoss(reduction="mean")(
                    pred, target.float()
                )
                dice_metric = DiceMetric(include_background=False)

                hausdorff_metric = HausdorffDistanceMetric(
                    include_background=False, percentile=95.0
                )

                train_losses = []
                val_losses = []
                best_loss = float("inf")
                best_val_dice = (0, 0)

                for epoch in range(config["num_epochs"]):
                    train_loss = train_epoch(
                        medsam_model, train_dataloader, optimizer, loss_fn, device
                    )
                    train_losses.append(train_loss)
                    mf.log_metric("train_loss", train_loss, step=epoch)

                    if epoch % config["val_step"] == 0:
                        val_loss, val_dice, val_hausdorff = validate_epoch(
                            medsam_model,
                            val_dataloader,
                            loss_fn,
                            dice_metric,
                            hausdorff_metric,
                            device,
                            config,
                            epoch,
                            model_save_path_fold,
                        )
                        val_losses.append(val_loss)
                        mf.log_metric("val_loss", val_loss, step=epoch)
                        mf.log_metric("val_dice", val_dice, step=epoch)
                        mf.log_metric("val_hausdorff", val_hausdorff, step=epoch)
                        logging.info(
                            f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Dice: {val_dice}, Val Hausdorff: {val_hausdorff}"
                        )

                        if val_loss < best_loss:
                            best_loss = val_loss
                            best_val_dice = (epoch, val_dice)
                            save_checkpoint(
                                {
                                    "model": medsam_model.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "epoch": epoch,
                                },
                                os.path.join(model_save_path_fold, "checkpoint_best"),
                            )

                    save_checkpoint(
                        {
                            "model": medsam_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                        },
                        os.path.join(model_save_path_fold, "checkpoint_latest"),
                    )

                    plot_losses(
                        epoch, train_losses, val_losses, config, model_save_path_fold
                    )

                fold_dices.append(val_dice)
                fold_best_dices.append(best_val_dice[1])
                fold_hausdorffs.append(val_hausdorff)

                logging.info(f"Fold {fold} last dice: {val_dice}")
                logging.info(
                    f"Fold {fold} best dice: {best_val_dice[1]} (epoch {best_val_dice[0]})"
                )
                logging.info(f"Fold {fold} Hausdorff distance: {val_hausdorff}")
                mf.log_metric("best_val_dice", best_val_dice[1], step=best_val_dice[0])

        logging.info(f"Mean fold dice: {np.mean(fold_dices)}")
        logging.info(f"Mean fold best dice: {np.mean(fold_best_dices)}")
        logging.info(f"Mean fold Hausdorff distance: {np.mean(fold_hausdorffs)}")
        mf.log_metric("fold_dice", np.mean(fold_dices), step=epoch)
        mf.log_metric("fold_best_dice", np.mean(fold_best_dices), step=epoch)
        mf.log_metric("fold_hausdorff", np.mean(fold_hausdorffs), step=epoch)


if __name__ == "__main__":
    main()
