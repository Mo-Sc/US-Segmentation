from typing import Tuple
import cv2
import numpy as np
from skimage import transform


def preprocess_image(image_data, do_intensity_cutoff=True, image_size=1024):
    """
    Preprocessing function to convert images into format required by MedSAM.
    Adapted directly from the pre_grey_rgb.py script in the MedSAM repository.
    """

    # Ensure the image has the correct shape (C, H, W)
    if np.max(image_data) > 255.0:
        image_data = np.uint8(
            (image_data - image_data.min())
            / (np.max(image_data) - np.min(image_data))
            * 255.0
        )

    if len(image_data.shape) == 2:
        image_data = np.repeat(np.expand_dims(image_data, -1), 3, -1)
    assert (
        len(image_data.shape) == 3
    ), "image data is not three channels: img shape:" + str(image_data.shape)
    # convert three channel to one channel
    if image_data.shape[-1] > 3:
        image_data = image_data[:, :, :3]

    # image preprocess start
    if do_intensity_cutoff:
        lower_bound, upper_bound = np.percentile(
            image_data[image_data > 0], 0.5
        ), np.percentile(image_data[image_data > 0], 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
        image_data_pre[image_data == 0] = 0
        image_data_pre = np.uint8(image_data_pre)
    else:
        image_data_pre = image_data.copy()

    resize_img = transform.resize(
        image_data_pre,
        (image_size, image_size),
        order=3,
        mode="constant",
        preserve_range=True,
        anti_aliasing=True,
    )
    resize_img01 = resize_img / 255.0

    return resize_img01


# --- Post-processing functions ---


def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def keep_largest_region(mask: np.ndarray, mode: str) -> Tuple[np.ndarray, bool]:
    """
    Keeps only the largest connected region in the mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)

    if n_labels <= 1:
        # Only the background is present, nothing to keep
        return mask, False

    # Find the label of the largest component (excluding the background)
    largest_label = (
        np.argmax(stats[1:, -1]) + 1
    )  # +1 to account for background label at index 0

    # Create a mask with only the largest component
    mask = regions == largest_label

    return mask, True
