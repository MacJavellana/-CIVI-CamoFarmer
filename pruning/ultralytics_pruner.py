"""
Pruning Script for YOLO and RT-DETR Models
------------------------------------------

Performs L1 unstructured pruning on YOLO or RT-DETR models.
It allows pruning specific groups of layers (backbone, neck, head, or all convolutional layers)
and saves the pruned model along with pruning statistics.

Command-line arguments:
    --folder_name     : Folder name for storing pruning results.
    --bmodel_path     : Path to the baseline model weights.
    --model_type      : Type of model to prune ('yolo' or 'rtdetr').
    --prune_group     : Layer group to prune ('backbone', 'neck', 'head', 'all').
    --prune_ratio     : L1 unstructured pruning ratio (float).

Sample usage:
    python ultralytics_pruner.py --folder_name "YOLO_N_CC_HEAD_50" --bmodel_path "runs/detect/ccrop_300n_coare/weights/best.pt" --model_type "yolo" --prune_group "head" --prune_ratio .50

Outputs:
    - Pruned model (.pt file) in the results folder.
    - CSV log file containing pruning statistics.
"""

import os
import csv
import argparse
import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO, RTDETR
import prune_helpers as ph


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Prune YOLO/RT-DETR models with L1 unstructured pruning.")
    parser.add_argument("--folder_name", type=str, required=True, help="Folder for pruning results")
    parser.add_argument("--bmodel_path", type=str, required=True, help="Path to baseline model weights")
    parser.add_argument("--model_type", type=str, choices=['yolo', 'rtdetr'], required=True, help="Type of model to prune")
    parser.add_argument("--prune_group", type=str, choices=['backbone', 'neck', 'head', 'all'],
                        default='all', help="Group of layers to prune (default: all)")
    parser.add_argument("--prune_ratio", type=float, required=True, help="L1 Unstructured Pruning Ratio")
    return parser.parse_args()


def load_model(model_type: str, model_path: str):
    """
    Load YOLO or RT-DETR model.

    Args:
        model_type (str): Either 'yolo' or 'rtdetr'.
        model_path (str): Path to the model weights.

    Returns:
        tuple: (base_model, inner_model, group_slices)
    """
    if model_type == 'yolo':
        base_model = YOLO(model_path)
        group_slices = {
            'backbone': slice(0, 10),
            'neck': slice(10, 22),
            'head': slice(22, None),
        }
    else:
        base_model = RTDETR(model_path)
        group_slices = {
            'backbone': slice(0, 10),
            'neck': slice(10, 28),
            'head': slice(28, None),
        }
    return base_model, base_model.model, group_slices


def prune_model(model, prune_group: str, prune_ratio: float, group_slices: dict):
    """
    Apply L1 unstructured pruning to a model.

    Args:
        model (torch.nn.Module): The model to prune.
        prune_group (str): The group of layers ('backbone', 'neck', 'head', or 'all').
        prune_ratio (float): The proportion of weights to prune.
        group_slices (dict): Slice indices for each group.

    Returns:
        torch.nn.Module: Pruned model.
    """
    if prune_group == 'all':
        print("\nPruning all layers:")
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name="weight", amount=prune_ratio)
                prune.remove(module, "weight")
    else:
        group_slice = group_slices[prune_group]
        print(f"\nPruning {prune_group} layers:")
        for i, layer in enumerate(model.model[group_slice]):
            print(f"Layer {i + group_slice.start}: {layer.__class__.__name__}")
            for _, module in layer.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module, name="weight", amount=prune_ratio)
                    prune.remove(module, "weight")
    return model


def write_results_csv(results_path: str, data: list):
    """
    Write pruning results to CSV.

    Args:
        results_path (str): Path to the CSV file.
        data (list): List of data rows.
    """
    with open(results_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Model_Type', 'Prune_Group', 'Prune_Ratio',
            'Total_Params_Before', 'Nonzero_Params_Before',
            'Total_Params_After', 'Nonzero_Params_After',
            'Sparsity_Percent',
            'Inference_Time_Before_ms', 'Inference_Time_Std_ms'
        ])
        writer.writerow(data)


def main():
    """Main entry point for pruning script."""
    args = parse_args()

    # Set up paths
    proj_path = f"pruned_results/{args.folder_name}"
    os.makedirs(proj_path, exist_ok=True)
    pruned_model_path = f"{proj_path}/pruned_model.pt"
    results_file = f"{proj_path}/{args.folder_name}_prune_log.csv"

    # Load model
    base_model, model, group_slices = load_model(args.model_type, args.bmodel_path)

    # Baseline stats
    print("Baseline Model Stats:")
    total_before, nonzero_before = ph.count_parameters(model)
    base_model.info(verbose=True)
    avg_time_before, std_time_before = ph.benchmark_inference(model)

    # Prune model
    pruned_model = prune_model(model, args.prune_group, args.prune_ratio, group_slices)

    # Pruned stats
    total_after, nonzero_after = ph.count_parameters(pruned_model)
    sparsity = (total_after - nonzero_after) / total_after * 100
    params_pruned = nonzero_before - nonzero_after

    print("\nPruning Results Summary:")
    print("=" * 50)
    print(f"Total Params (Before):   {total_before:,}")
    print(f"Nonzero Params (Before): {nonzero_before:,}")
    print(f"Total Params (After):    {total_after:,}")
    print(f"Nonzero Params (After):  {nonzero_after:,}")
    print(f"Parameters Pruned:       {params_pruned:,}")
    print(f"Sparsity Achieved:       {sparsity:.2f}%")
    print(f"Inference Time (Before): {avg_time_before:.2f} +/- {std_time_before:.2f} ms")

    # Write results
    write_results_csv(results_file, [
        args.model_type, args.prune_group, args.prune_ratio,
        total_before, nonzero_before,
        total_after, nonzero_after,
        f"{sparsity:.2f}",
        f"{avg_time_before:.2f}", f"{std_time_before:.2f}"
    ])

    # Save pruned model
    base_model.model = pruned_model
    base_model.save(pruned_model_path)
    print(f"\nPruned model saved to: {pruned_model_path}")


if __name__ == "__main__":
    main()
