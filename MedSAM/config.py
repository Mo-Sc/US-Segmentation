train_config = {
    "random_state": 2023,
    "npy_base_path": "data/msot_ic_2_us_segmentation_preprocessed",
    "task_name": "MedSAM-ViT-B",
    "model_type": "vit_b",
    "checkpoint": "checkpoints/sam_vit_b_01ec64.pth",
    # "checkpoint": "checkpoints/medsam_vit_b.pth", # will auto-download
    "work_dir": "temp/",
    "val_size": 0.2,
    "num_epochs": 10,
    "batch_size": 4,
    "num_workers": 0,
    "weight_decay": 0.01,
    "lr": 0.0001,
    "val_step": 1,
    "device": "cuda:0",
    "use_mlflow": False,
    "plot_val_pred": True,
}

# config for cross validation
train_config_CV = {
    "random_state": 1809,
    "npy_base_path": "data/msot_ic_2_us_segmentation_preprocessed",
    "task_name": "MedSAM-ViT-B",
    "model_type": "vit_b",
    # "checkpoint": "checkpoints/sam_vit_b_01ec64.pth",
    "checkpoint": "checkpoints/medsam_vit_b.pth",
    "work_dir": "temp/",
    "k_folds": 5,
    "num_epochs": 50,
    "batch_size": 4,
    "num_workers": 0,
    "weight_decay": 0.01,
    "lr": 0.0001,
    "val_step": 1,
    "device": "cuda:0",
    "plot_val_pred": False,
}

inference_config = {
    "input_path": "data/testdata",
    "output_path": "data/msot_1_tests",
    "device": "cuda:0",
    "model_type": "vit_b",
    "checkpoint": "checkpoints/checkpoint_best_weights_only.pth",
    "img_size": 1024,
    "plot_pred": True,
    "preprocess": True,
    "bbox": [0, 0, 1024, 1024],  # whole image
}
