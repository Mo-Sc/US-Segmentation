import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

join = os.path.join


# Dataloaders adapted from MedSAM
class NpyDataset(Dataset):
    def __init__(self, data_root, scans, bbox_shift=20, no_bbox=True):
        """
        Dataset for training
        data_root: str, path to the data folder
        scans: list of str, scan names
        bbox_shift: int, maximum pertubation of bounding box coordinates. Only used if no_bbox is False.
        no_bbox: bool, if True, the whole image will be used as the bounding box
        """

        self.data_root = data_root
        self.scans = scans
        self.bbox_shift = bbox_shift
        self.no_bbox = no_bbox

        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")

        gt_path_files = []
        img_path_files = []

        # check if the gt and img files exist and add to filepath lists
        for scan in scans:
            gt_file_path = os.path.join(self.gt_path, f"{scan}.npy")
            img_file_path = os.path.join(self.img_path, f"{scan}.npy")
            if os.path.exists(gt_file_path) and os.path.exists(img_file_path):
                gt_path_files.append(gt_file_path)
                img_path_files.append(img_file_path)
            else:
                raise FileNotFoundError(
                    f"GT or IMG for {scan} not found in {self.data_root}"
                )

        # sort
        img_path_files.sort(key=lambda x: int(os.path.basename(x).split("_")[0]))
        gt_path_files.sort(key=lambda x: int(os.path.basename(x).split("_")[0]))

        print(f"#images: {len(img_path_files)}")

        self.img_path_files = img_path_files
        self.gt_path_files = gt_path_files

    def __len__(self):
        return len(self.img_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"

        H, W = gt2D.shape

        if self.no_bbox:
            # no bounding box. -> set the whole image as bounding box
            bboxes = np.array([0, 0, W, H])
        else:
            y_indices, x_indices = np.where(gt2D > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # add perturbation to bounding box coordinates
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            bboxes = np.array([x_min, y_min, x_max, y_max])

        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )


class InferenceNpyDataset(Dataset):
    def __init__(self, data_root, scans, preprocessed=True, validation=False):
        """
        Dataset for validation / inference
        data_root: str, path to the data folder
        scans: list of str, scan names
        preprocessed: bool, whether the data is preprocessed as required by MedSAM. If False, the data will be preprocessed on the fly.
        validation: bool. If True, the dataset will return the ground truth masks. If False, the dataset will return only the images.
        """

        self.data_root = data_root
        self.preprocessed = preprocessed
        self.validation = validation

        self.img_path = join(data_root, "imgs")
        img_path_files = []

        # check if the gt and img files exist and add to filepath lists
        for scan in scans:
            img_file_path = os.path.join(self.img_path, f"{scan}.npy")
            if os.path.exists(img_file_path):
                img_path_files.append(img_file_path)
            else:
                raise FileNotFoundError(f"IMG for {scan} not found in {self.data_root}")

        img_path_files.sort(key=lambda x: int(os.path.basename(x).split("_")[0]))
        self.img_path_files = img_path_files

        if validation:

            self.gt_path = join(data_root, "gts")
            gt_path_files = []
            for scan in scans:
                gt_file_path = os.path.join(self.gt_path, f"{scan}.npy")
                if os.path.exists(gt_file_path):
                    gt_path_files.append(gt_file_path)
                else:
                    raise FileNotFoundError(
                        f"GT or IMG for {scan} not found in {self.data_root}"
                    )

            gt_path_files.sort(key=lambda x: int(os.path.basename(x).split("_")[0]))
            self.gt_path_files = gt_path_files
        else:
            self.gt_path_files = None

        print(f"#images: {len(img_path_files)}")

    def __len__(self):
        return len(self.img_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.img_path_files[index])

        if not self.preprocessed:
            img_1024 = None  # TODO: implement.
        else:
            img_1024 = np.load(
                join(self.img_path, img_name), "r", allow_pickle=True
            )  # (1024, 1024, 3)

        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"

        if not self.validation:

            return (
                torch.tensor(img_1024).float(),
                img_name,
            )
        else:
            gt = np.load(
                join(self.gt_path, img_name), "r", allow_pickle=True
            )  # multiple labels [0, 1,4,5...], (256,256)
            label_ids = np.unique(gt)[1:]
            gt2D = np.uint8(
                gt == random.choice(label_ids.tolist())
            )  # only one label, (256, 256)
            assert (
                np.max(gt2D) == 1 and np.min(gt2D) == 0.0
            ), "ground truth should be 0, 1"

            return (
                torch.tensor(img_1024).float(),
                torch.tensor(gt2D[None, :, :]).long(),
                img_name,
            )


# MEdSAM model. Unchanged from original MedSAM codebase (MedSAM/train_one_gpu.py).
class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


def extract_model_weights(checkpoint_filepath):
    """
    Workaround, because MedSAM saves models including optimizer, but wants to load file that only contains the model weights
    Extracts the model weights from a .pth file and saves them to a new .pth file.
    """

    checkpoint = torch.load(checkpoint_filepath, map_location="cpu")

    assert (
        "model" in checkpoint.keys()
    ), "The .pth file does not contain the model weights"

    weights = checkpoint["model"]

    out_filepath = os.path.join(
        os.path.dirname(checkpoint_filepath),
        f"{os.path.basename(checkpoint_filepath)[:-4]}_weights.pth",
    )
    torch.save(weights, out_filepath)
    print(f"Weights are saved to {out_filepath}")

    return out_filepath


# --- Visualization functions ---


def plot_losses(epoch, train_losses, val_losses, config, model_save_path):
    plt.plot(range(epoch + 1), train_losses, label="Train Loss")
    if len(val_losses) > 0:
        plt.plot(
            range(0, epoch + 1, config["val_step"]),
            val_losses,
            label="Validation Loss",
        )
    plt.title("Dice + Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(model_save_path, config["task_name"] + "_loss.png"))
    plt.close()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


def plot_prediction(img, gt2D, pred_mask, boxes_np, filenames, model_save_path, step):

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))

    # Subplot 1: Image with ground truth mask
    axs[0].imshow(img)
    show_mask(gt2D[0], axs[0], random_color=True)
    show_box(boxes_np[0], axs[0])
    axs[0].axis("off")
    axs[0].set_title("Ground Truth")

    # Subplot 2: Image with predicted mask
    axs[1].imshow(img)
    show_mask(pred_mask[0] > 0, axs[1], random_color=True)
    show_box(boxes_np[0], axs[1])
    axs[1].axis("off")
    axs[1].set_title("Prediction")

    # Subplot 3: Image with both ground truth and predicted masks
    axs[2].imshow(img)
    show_mask(gt2D[0], axs[2], random_color=True)
    show_mask(pred_mask[0] > 0, axs[2], random_color=False)
    show_box(boxes_np[0], axs[2])
    axs[2].axis("off")
    axs[2].set_title("Ground Truth & Prediction")

    # Save the figure
    outpath = join(model_save_path, "plots")
    os.makedirs(outpath, exist_ok=True)
    plt.savefig(
        join(
            outpath,
            f"{step}_prediction_comparison_{filenames[0][:-4]}.png",
        )
    )
    plt.close()
