import os

import numpy as np
from tqdm import tqdm
from skimage import transform

from pre_post_process import preprocess_image


# def extract_scans(hdf5_path, hdf5_tags):
#     msot_data = MSOTData(hdf5_path)
#     try:
#         annotated_frame = msot_data.get_iannotations().get_annotated_frames(0)[0]
#     except:
#         annotated_frame = 12
#     scan = msot_data.load_img(tags=hdf5_tags, flipud=False)[0]
#     return scan[annotated_frame]


def main():

    raw_data_dir = "../../Datasets/msot_ic_2_us_segmentation"

    # where to save the preprocessed data
    preprocessed_data_dir = "../data/msot_ic_2_us_segmentation_preprocessed"
    img_size = 1024

    for orig_dir_name, new_dir_name in zip(["images", "labels"], ["imgs", "gts"]):

        input_dir = os.path.join(raw_data_dir, orig_dir_name)
        output_dir = os.path.join(preprocessed_data_dir, new_dir_name)

        os.makedirs(os.path.join(output_dir), exist_ok=True)

        input_files = [file for file in os.listdir(input_dir) if file.endswith(".npy")]

        print(
            f"Processing {orig_dir_name}. Saving to {output_dir}. Total files: {len(input_files)}"
        )

        for npy_file in tqdm(input_files):

            img = np.load(os.path.join(input_dir, npy_file))
            # img = np.flipud(img)  # if necessary. Vorlaufstrecke should be at the top

            # for masks, only resize
            if orig_dir_name == "labels":
                res_img = transform.resize(
                    np.uint8(img),
                    (img_size, img_size),
                    order=0,
                    mode="constant",
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(np.uint8)
            else:
                res_img = preprocess_image(img, image_size=img_size)

            np.save(os.path.join(output_dir, npy_file), res_img)

            tqdm.write(f"Processed {npy_file}")


if __name__ == "__main__":
    main()
