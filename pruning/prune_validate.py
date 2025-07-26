"""
Fine-Tune Pruned Models Script
------------------------------

Loads a previously pruned YOLO or RT-DETR model, validates it, fine-tunes it on the specified dataset,
and evaluates its performance after fine-tuning.

Command-line arguments:
    --folder_name  : Name of the folder where pruning results and pruned model are stored.
    --model_type   : Type of model to fine-tune ('yolo' or 'rtdetr').
    --dset_type    : Dataset type ('ccrop', 'tomatod', or 'camocrops').

Sample usage:
    python prune_validate.py --folder_name "YOLO_N_CC_HEAD_50" --model_type "yolo" --dset_type "ccrop"

Outputs:
    - Fine-tuned model weights in the results folder.
    - Validation results with mAP scores and true positive rates.
    - Updated CSV log file with fine-tuning performance metrics.
    - Visualization outputs for model predictions.

"""

import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO, RTDETR
import argparse
import os
import prune_helpers as ph
import csv


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Fine-tune pruned YOLO/RT-DETR models.")
    parser.add_argument("--folder_name", type=str, required=True, help="Folder for pruning results")
    parser.add_argument("--model_type", type=str, choices=['yolo', 'rtdetr'], required=True, help="Type of model to fine-tune")
    parser.add_argument("--dset_type", type=str, choices=['ccrop', 'tomatod', 'camocrops'], required=True, help="Dataset type for fine-tuning")
    return parser.parse_args()


def get_dataset_paths(dset_type: str):
    """
    Get dataset paths based on dataset type.

    Args:
        dset_type (str): Dataset type ('ccrop', 'tomatod', or 'camocrops').

    Returns:
        tuple: (data_yaml, test_set, test_imgs)
    """
    if dset_type == 'tomatod':
        data = "tomatOD_dataset.yaml"  # Replace with actual path to dataset YAML
        test_set = "tomatOD/test/labels"  # Replace with actual path to test labels folder
        test_imgs = "tomatOD/test/images"  # Replace with actual path to test images folder
    elif dset_type == 'ccrop':
        data = "ccrop_dataset.yaml"  # Replace with actual path to dataset YAML
        test_set = "CCROP_test.txt"  # Replace with actual path to test set text file
        test_imgs = None
    elif dset_type == 'camocrops':
        data = "camocrops_dataset.yaml"  # Replace with actual path to dataset YAML
        test_set = "camocrops/labels/test"  # Replace with actual path to test labels folder
        test_imgs = "camocrops/images/test"  # Replace with actual path to test images folder

    return data, test_set, test_imgs


def load_pruned_model(model_type: str, model_path: str):
    """
    Load pruned YOLO or RT-DETR model.

    Args:
        model_type (str): Either 'yolo' or 'rtdetr'.
        model_path (str): Path to the pruned model weights.

    Returns:
        Model: Loaded pruned model.
    """
    if model_type == 'yolo':
        return YOLO(model_path)
    else:
        return RTDETR(model_path)


def get_training_params(model_type: str, dset_type: str):
    """
    Get training hyperparameters based on model and dataset type.

    Args:
        model_type (str): Either 'yolo' or 'rtdetr'.
        dset_type (str): Dataset type ('ccrop', 'tomatod', or 'camocrops').

    Returns:
        dict: Training parameters.
    """
    batch = 16 if model_type == 'rtdetr' or (dset_type == 'tomatod' and model_type == 'yolo') else 32
    
    if model_type == 'rtdetr':
        epochs, save_period = 90, 15
    elif model_type == 'yolo':
        if dset_type == 'tomatod':
            epochs, save_period = 30, 5
        elif dset_type == 'ccrop':
            epochs, save_period = 300, 50
        elif dset_type == 'camocrops':
            epochs, save_period = 250, 50
    
    return {
        'epochs': epochs,
        'batch': batch,
        'save_period': save_period,
        'imgsz': 320,
        'lr0': 0.01,
        'lrf': 0.0001,
        'momentum': 0.9,
        'optimizer': 'SGD',
        'iou_thresh': 0.5
    }


def write_results_csv(results_path: str, avg_time: float, std_time: float, ft_results, tp_percent: float):
    """
    Write fine-tuning results to CSV.

    Args:
        results_path (str): Path to the CSV file.
        avg_time (float): Average inference time.
        std_time (float): Standard deviation of inference time.
        ft_results: Fine-tuning validation results.
        tp_percent (float): True positive percentage.
    """
    with open(results_path, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Inference_Time_ms', '', f"{avg_time:.2f} +/- {std_time:.2f}"])
        
        # Header
        writer.writerow(['Metric', 'Fine_Tuned_Value'])
        
        # Per-class mAP50
        for idx, ft_ap in enumerate(ft_results.box.ap50):
            writer.writerow([f'Class_{idx}_mAP50', f"{ft_ap:.4f}"])
        
        # Overall mAP50
        writer.writerow(['Overall_mAP50', f"{ft_results.box.map50:.4f}"])
        
        # TP Percentage
        writer.writerow(['True_Positives_Percent', '', f"{tp_percent:.2f}"])

def main():
    """Main entry point for fine-tuning script."""
    args = parse_args()

    # Get dataset paths
    data, test_set, test_imgs = get_dataset_paths(args.dset_type)
    
    # Set up paths
    proj_path = f"pruned_results/{args.folder_name}"
    os.makedirs(proj_path, exist_ok=True)
    pruned_model_path = f"{proj_path}/pruned_model.pt"
    results_file = f"{proj_path}/{args.folder_name}_prune_log.csv"
    visualization_folder = f"{proj_path}/visualizations"

    # Load pruned model
    pruned_model = load_pruned_model(args.model_type, pruned_model_path)

    # Get training parameters
    params = get_training_params(args.model_type, args.dset_type)

    # Validate pruned model before fine-tuning
    print("Validating pruned model before fine-tuning...")
    pruned_results = pruned_model.val(
        data=data, imgsz=params['imgsz'], batch=params['batch'], project=proj_path,
        save_json=True, name="pruned_val", split="test"
    )
    print("Pruned Results:")
    print(f"mAPs: {pruned_results.box.ap50}")
    print(f"mAP50: {pruned_results.box.map50:.4f}")

    # Fine-tune the pruned model
    print("\nFine-tuning the pruned model...")
    pruned_model.train(
        data=data, epochs=params['epochs'], batch=params['batch'], imgsz=params['imgsz'],
        lr0=params['lr0'], lrf=params['lrf'], momentum=params['momentum'], 
        optimizer=params['optimizer'], project=proj_path, name="ft_pruned", 
        save_period=params['save_period'], patience=300
    )

    # Validate fine-tuned model
    print("\nValidating fine-tuned model...")
    ft_results = pruned_model.val(
        data=data, imgsz=params['imgsz'], batch=params['batch'], project=proj_path,
        save_json=True, name="ft_val", split="test"
    )

    # Compute True Positive percentage
    # If the dataset is CCROP and containing text files to the paths of the images, use the specific function for it. Otherwise, use the generic dataset function.
    if args.dset_type == 'ccrop':
        tp_percent = ph.ccrop_tp(test_set, pruned_model, params['iou_thresh'], visualization_folder)
    else:
        tp_percent = ph.dset_tp(test_imgs, test_set, pruned_model, params['iou_thresh'], visualization_folder)

    # Display results
    pruned_model.info(verbose=True)
    print("\nFine-Tuned Results Summary:")
    print("=" * 50)
    print(f"mAPs: {ft_results.box.ap50}")
    print(f"mAP50: {ft_results.box.map50:.4f}")
    print(f"True Positives: {tp_percent:.2f}%")

    # Benchmark inference time
    avg_time, std_time = ph.benchmark_inference(pruned_model.model)
    print(f"Inference Time: {avg_time:.2f} +/- {std_time:.2f} ms")

    # Save results to CSV
    write_results_csv(results_file, avg_time, std_time, ft_results, tp_percent)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
