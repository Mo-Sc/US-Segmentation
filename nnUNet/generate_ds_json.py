import os
from nnUNet.nnunetv2.dataset_conversion.generate_dataset_json import (
    generate_dataset_json,
)

nnUNet_raw = os.getenv("nnUNet_raw")
if not nnUNet_raw:
    raise EnvironmentError("nnUNet_raw environment variable is not set.")

channel_names = {0: "US"}

labels = {
    "background": 0,
    "Muskel1": 1,
}

num_train = 97

dataset_name = "Dataset044-msot-ic-2-us-segmentation"

output_dir = os.path.join(nnUNet_raw, dataset_name)

generate_dataset_json(
    output_dir, channel_names, labels, num_train, ".nrrd", dataset_name=dataset_name
)
