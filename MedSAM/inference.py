import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

# workaround to not have to change imports in MedSAM code
sys.path.append("MedSAM")

from MedSAM.segment_anything import sam_model_registry
from utils.train_utils import MedSAM, extract_model_weights
from utils.pre_post_process import preprocess_image

from config import inference_config as config


def load_hdf5(file_path):
    from dataset_tools.hdf5.hdf_dataset import load_image

    us_tags = ["ultrasounds", "ultrasound", "0"]
    image, _, _ = load_image(file_path, tags=us_tags)
    frame = image[40, 0, :, :]
    return np.rot90(frame, 1)


def load_model(checkpoint_path, model_type, device):

    # weights_filepath = extract_model_weights(checkpoint_path)
    weights_filepath = checkpoint_path

    sam_model = sam_model_registry[model_type](checkpoint=weights_filepath)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    return medsam_model


def postprocess_mask(mask):
    mask = mask.squeeze().cpu().numpy()  # Remove batch and channel dimensions
    mask = (mask > 0.5).astype(np.uint8)  # Binarize the mask
    return mask


def plot_prediction(image, mask, outfile):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image[0, 0, :, :], cmap="gray")
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(image[0, 0, :, :], cmap="gray")
    plt.imshow(mask, alpha=0.5, cmap="jet")
    plt.title("Segmentation Mask")
    plt.axis("off")
    plt.savefig(outfile)
    plt.close()


def main():

    os.makedirs(config["output_path"], exist_ok=True)

    medsam_model = load_model(
        config["checkpoint"], config["model_type"], config["device"]
    )

    # in_files = [
    #     file for file in os.listdir(config["input_path"]) if file.endswith(".hdf5")
    # ]
    in_files = [
        file for file in os.listdir(config["input_path"]) if file.endswith(".npy")
    ]

    pbar = tqdm(total=len(in_files))

    for npy_file in in_files:

        pbar.set_description(f"Seg Inference: {npy_file}")

        image = np.load(os.path.join(config["input_path"], npy_file))
        # image = load_hdf5(os.path.join(config["input_path"], npy_file))

        if config["preprocess"]:
            image = np.squeeze(image)
            image = preprocess_image(image, image_size=config["img_size"])

        # Some sanity checks
        assert (
            len(image.shape) == 3
        ), f"image data is not three channels: img shape: {image.shape}"
        assert image.shape[0:2] == (
            config["img_size"],
            config["img_size"],
        ), f"image shape should be {config['img_size']}x{config['img_size']}"

        assert (
            np.max(image) <= 1.0 and np.min(image) >= 0.0
        ), "image should be normalized to [0, 1]"

        # Convert to tensor and add batch dimension, send to gpu
        image = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(config["device"])
        )

        # Define a dummy bounding box box (whole image)
        bbox = np.array([config["bbox"]])

        with torch.no_grad():
            mask = medsam_model(image, bbox)
            mask = torch.sigmoid(mask)  # Apply sigmoid to get probabilities
            mask = postprocess_mask(mask)  # Postprocess to get binary mask

        outfile_path = os.path.join(config["output_path"], npy_file[:-4])
        np.save(outfile_path + ".npy", mask)

        if config["plot_pred"]:
            plot_prediction(image.cpu().numpy(), mask, outfile_path + ".png")

        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    main()
