# MedSAM / SAM for US Segmentation

## Installation

- install the packages in requirements.txt
- clone the original [MedSAM repo](https://github.com/bowang-lab/MedSAM) into this folder
```
git clone https://github.com/bowang-lab/MedSAM.git
```

# Checkpoints

Download the [MedSAM checkpoint](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=drive_link) (medsam_vit_b.pth) and put it into the checkpoints folder. Adapt the checkpoint path in config.py if necessary.  

If you want to use the original SAM, you can simply set the checkpoint path to sam_vit_b_01ec64.pth, and the checkpoint will be downloaded automatically from [facebook AI](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## Usage

### Training

Preprocess the dataset for MedSAM using the preprocess.py script in the utils folder:
```
python preprocess_msot_ic_2.py
```

Put the preprocessed dataset into the ../data folder

Training parameters can be set in the config.py file. The training can be done using single train-test split or using cross validation. The training can be monitored using mlflow by setting use_mlfow (requires mlflow server + mlflow package installed). Currently the CV script only works with mlflow (but can easily be adapted).

If you want to plot the predictions on the validation set, set plot_val_pred to True

**Sinlge split**

specify training parameters in train_config dict in config.py. As checkpoint, you can either use the MedSAM checkpoint or SAM checkpoint.
```
python train_val.py
```

Note: Doenst calculate HSD95 metric

**CV**

specify training parameters in train_config_CV dict in config.py. As checkpoint, you can either use the MedSAM checkpoint or SAM checkpoint.
```
python train_val_CV.py
```

Note: mlflow implementation. To run without mlflow, remove the corresponding code.

### Inference

Use one of your pretrained models from the temp/{run_id}/checkpoint_best_weights_only.pth folder. Alternatively, download one of my pretrained models [MedSAM-ViT-B-20240719-1116](https://faubox.rrze.uni-erlangen.de/getlink/fi51JCKAMGtmUdubCLFtbx/checkpoint_best_weights_only.pth) (after access request).

If you want to use the base SAM (sam_vit_b_01ec64.pth) or base MedSAM (medsam_vit_b.pth) weights for inference, you might have to convert, see [here](checkpoints/README.md).

Put the pretrained model into the checkpoints folder or adapt the inference_config in the config file.

Adapt the input_path and output_path in inference_config to point to your data. Note that the data has to be preprocessed prior to inference as well. You can either do that by setting preprocess=True in the inference_config or by manually preprocessing your dataset using the preprocess_msot_ic_2.py script.


Run the inference script:
```
python inference.py
```


## TODO
- hyperparemter tuning
- augmentations


