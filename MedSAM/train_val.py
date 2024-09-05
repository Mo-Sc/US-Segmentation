import os
from datetime import datetime
import shutil
import logging


import sys

# workaround to not have to change imports in MedSAM code
sys.path.append("MedSAM")

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import monai
from sklearn.model_selection import train_test_split

# from MedSAM.train_one_gpu import MedSAM
from MedSAM.segment_anything import sam_model_registry

from utils.train_utils import (
    plot_prediction,
    plot_losses,
    NpyDataset,
    InferenceNpyDataset,
    MedSAM,
)


# Load configuration
from config import train_config as config

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

# setup mlflow tracking (requires mlflow server)
if config["use_mlflow"]:
    import utils.mlflow as mlflow
    from config import config_mlflow

    mf = mlflow.mlflow_start(config_mlflow)


def create_dataloaders(config, train_scans, val_scans):
    train_dataset = NpyDataset(config["npy_base_path"], train_scans, no_bbox=True)
    validation_dataset = InferenceNpyDataset(
        config["npy_base_path"], val_scans, validation=True
    )

    if config["use_mlflow"]:
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
    model, dataloader, loss_fn, dice_metric, device, config, epoch, model_save_path
):
    model.eval()
    val_loss = 0
    val_dice = 0

    with torch.no_grad():
        for image, gt2D, filenames in dataloader:
            image, gt2D = image.to(device), gt2D.to(device)
            boxes_np = np.array([[0, 0, image.shape[2], image.shape[3]]])
            boxes_np = torch.tensor(boxes_np).float().to(device)

            medsam_pred = model(image, boxes_np)
            loss = loss_fn(medsam_pred, gt2D)
            val_loss += loss.item()

            medsam_pred_bin = (torch.sigmoid(medsam_pred) > 0.5).float()
            dice_score = dice_metric(medsam_pred_bin, gt2D)
            val_dice += dice_score.item()

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
    dice_metric.reset()

    return val_loss, val_dice


def save_checkpoint(state, filename="checkpoint"):
    torch.save(state, filename + ".pth")
    torch.save(state["model"], filename + "_weights_only.pth")


def main():

    if config["use_mlflow"]:
        mf.log_param("Run ID", run_id)
        mf.log_param("Checkpoint path", model_save_path)
        mf.log_params(config)

    # copy training script and config to model save path
    shutil.copyfile(
        __file__,
        os.path.join(model_save_path, run_id + "_" + os.path.basename(__file__)),
    )

    shutil.copyfile(
        "config.py",
        os.path.join(model_save_path, run_id + "_" + os.path.basename("config.py")),
    )

    # intialized model and optimizer
    medsam_model = initialize_model(config, device)
    optimizer = torch.optim.AdamW(
        medsam_model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    loss_fn = lambda pred, target: monai.losses.DiceLoss(
        sigmoid=True, squared_pred=True, reduction="mean"
    )(pred, target) + nn.BCEWithLogitsLoss(reduction="mean")(pred, target.float())
    dice_metric = monai.metrics.DiceMetric(
        include_background=True, reduction="mean", get_not_nans=False
    )

    # load scan paths, split into training and validation
    scans = [
        filename.split(".")[0]
        for filename in os.listdir(os.path.join(config["npy_base_path"], "imgs"))
        if filename.endswith(".npy")
    ]
    train_scans, val_scans = train_test_split(
        scans, test_size=config["val_size"], random_state=config["random_state"]
    )

    # create dataloaders from custom npy dataset
    train_dataloader, val_dataloader = create_dataloaders(
        config, train_scans, val_scans
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
        if config["use_mlflow"]:
            mf.log_metric("train_loss", train_loss, step=epoch)

        if epoch % config["val_step"] == 0:
            val_loss, val_dice = validate_epoch(
                medsam_model,
                val_dataloader,
                loss_fn,
                dice_metric,
                device,
                config,
                epoch,
                model_save_path,
            )
            val_losses.append(val_loss)
            if config["use_mlflow"]:
                mf.log_metric("val_loss", val_loss, step=epoch)
                mf.log_metric("val_dice", val_dice, step=epoch)
            logging.info(
                f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Dice: {val_dice}"
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
                    os.path.join(model_save_path, f"checkpoint_best"),
                )

        save_checkpoint(
            {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            os.path.join(model_save_path, f"checkpoint_latest"),
        )

        plot_losses(epoch, train_losses, val_losses, config, model_save_path)
        logging.info(
            f"Best model updated at epoch {best_val_dice[0]} with val dice {best_val_dice[1]}"
        )
        if config["use_mlflow"]:
            mf.log_metric("best_val_dice", best_val_dice[1], step=best_val_dice[0])


if __name__ == "__main__":
    main()
