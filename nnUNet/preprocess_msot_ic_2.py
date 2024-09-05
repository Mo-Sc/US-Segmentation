import os
import numpy as np
import nrrd

# Ensure nnUNet_raw environment variable is set
nnUNet_raw = os.getenv("nnUNet_raw")
if not nnUNet_raw:
    raise EnvironmentError("nnUNet_raw environment variable is not set.")

# Define paths
src_dataset_path = "../Datasets/msot_ic_2_us_segmentation"
dst_dataset_name = "Dataset044-msot-ic-2-us-segmentation"
dst_dataset_path = os.path.join(nnUNet_raw, dst_dataset_name)

# Define source and destination folders
src_images_folder = os.path.join(src_dataset_path, "images")
src_labels_folder = os.path.join(src_dataset_path, "labels")
dst_images_folder = os.path.join(dst_dataset_path, "imagesTr")
dst_labels_folder = os.path.join(dst_dataset_path, "labelsTr")

# Create destination folders
os.makedirs(dst_images_folder, exist_ok=True)
os.makedirs(dst_labels_folder, exist_ok=True)


# Helper function to convert and save npy to nrrd
def convert_and_save_npy_to_nrrd(src_file, dst_file):
    data = np.load(src_file)
    nrrd.write(dst_file, data)


# Counters
converted_images_count = 0

# Process images
for src_image_file in os.listdir(src_images_folder):
    if src_image_file.endswith(".npy"):
        study_number, scan_number = map(
            int, src_image_file.replace(".npy", "").split("_")
        )
        dst_image_file = f"msot-ic-2-us-segmentation_{str(study_number).zfill(3)}{str(scan_number).zfill(3)}_0000.nrrd"

        # Check for corresponding label file
        src_label_file = f"{study_number}_{scan_number}.npy"
        if src_label_file in os.listdir(src_labels_folder):
            # Convert and save image
            convert_and_save_npy_to_nrrd(
                os.path.join(src_images_folder, src_image_file),
                os.path.join(dst_images_folder, dst_image_file),
            )
            converted_images_count += 1

            # Convert and save label
            dst_label_file = f"msot-ic-2-us-segmentation_{str(study_number).zfill(3)}{str(scan_number).zfill(3)}.nrrd"
            convert_and_save_npy_to_nrrd(
                os.path.join(src_labels_folder, src_label_file),
                os.path.join(dst_labels_folder, dst_label_file),
            )
        else:
            print(f"Warning: No corresponding label found for image {src_image_file}")

print(f"Number of converted images: {converted_images_count}")
