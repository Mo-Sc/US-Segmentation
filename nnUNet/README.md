# nnUNet for US Segmentation

## Installation

Clone and install the original repo into this folder
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e nnUNet
```


nnUNet requires setting some environment variables to know where to store data. You can set them using the env_vars.sh script:
```
source ./env_vars.sh
```

## Dataset Preparation

nnUNet requires the data to be in a speficic format (see https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md). To convert the npy dataset into the required format, you can use the preprocess_msot_ic_2.py script (make sure msot_ic_2_us_segmentation is loacated and extracted in the Datasets folder):

```
pip install pynrrd

python preprocess_msot_ic_2.py
```

Now we have to create a dataset.json file. It contains some information about the dataset, like channels, label names, etc. You can use the generate_ds_json.py script for that:

```
python generate_ds_json.py
```

The dataset.json should be located in the nnUNet_raw/Dataset044-msot-ic-2-us-segmentation folder.

When processing a new dataset for the first time, nnUNet will extract a dataset fingerprint (44 is the dataset id in our case):
```
nnUNetv2_plan_and_preprocess -d 44 --verify_dataset_integrity
```

You can look at the output in the nnUNet_preprocessed folder. For more information, see https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md


## Training

nnU-Net trains all configurations in a 5-fold cross-validation over the training cases:
```
nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD --npz
```

So we run the training script 5 times:

For FOLD in [0, 1, 2, 3, 4]:
```
nnUNetv2_train 44 2d FOLD --npz
```
with 44 being our dataset id, 2d to specify that we only want to train the 2d model, fold being the fold and --npz to save the softmax outputs during the final validation (required for later step)

Once all folds are trained, we can tell nnU-Net to automatically find the best combination. It will also determine the postprocessing that should be used.
```
nnUNetv2_find_best_configuration 44 -c 2d
```
with 44 being our dataset id and specifying that we only want to evaluate the 2d configurations

Once completed, the command will create two files in the nnUNet_results/44 folder:
- inference_instructions.txt what commands you need to run to make predictions using ideal configuration
- inference_information.json can be inspected to see the performance of all configurations and ensembles, as well as the effect of the postprocessing plus some debug information.

The calculated metrics, including for individual samples, can be found in the nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/summary.json file. 

## Inference

The inference dataset must follow the same conventions as the training dataset format (also see https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format_inference.md).

Then you can simply run inference using the command given by nnUNetv2_find_best_configuration in nference_instructions.txt.

To run it manually: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md#run-inference


## Hausdorff Distance

For the evaluation of the US segmentation, I also wanted to include a boundary based metric to focus on the accuracy of object boundaries. I included the 95% percentile Hausdorff distance (HD95, a bit more robust against outliers than pure HD) into nnUnet. The easiest way to use it is by replacing the nnUNet/nnunetv2/evaluation/evaluate_predictions.py file by my evaluate_predictions_custom.py, which includes calculation of HD95. The result will be included in the summary.json file after running nnUNetv2_find_best_configuration. To quickly copy (and rename) the file, run:
```
cp evaluate_predictions_custom.py nnUNet/nnunetv2/evaluation/evaluate_predictions.py
```

