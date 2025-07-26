"""
Evaluation and Benchmarking Helpers for Object Detection Models
---------------------------------------------------------------

This module provides utility functions for:
    - Counting parameters and sparsity
    - Benchmarking inference speed
    - Computing IoU between bounding boxes
    - Evaluating true positive rates on datasets with visualizations

Functions:
    count_parameters(model): Count total and non-zero parameters.
    benchmark_inference(model, iterations, input_size): Measure average inference time.
    compute_iou(box1, box2): Compute Intersection over Union (IoU) for two boxes.
    ccrop_tp(txt_file, model, iou_threshold, viz_folder): Evaluate model on file list (CCROP). Separate function for CCROP dataset created because of the specific format of the dataset.
    dset_tp(image_dir, label_dir, model, iou_threshold, viz_folder): Evaluate model on other related datasets.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import time
import os
import cv2


def count_parameters(model):
    """
    Count total and non-zero parameters of a model.

    Args:
        model (torch.nn.Module): The model to evaluate.

    Returns:
        tuple: (total_params, non_zero_params)
    """
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(torch.count_nonzero(p).item() for p in model.parameters())
    return total, nonzero


def benchmark_inference(model, iterations=100, input_size=(1, 3, 320, 320)):
    """
    Benchmark average inference time of a model.

    Args:
        model (torch.nn.Module): The model to benchmark.
        iterations (int): Number of iterations for timing.
        input_size (tuple): Shape of the dummy input (B, C, H, W).

    Returns:
        tuple: (average_time_ms, std_time_ms)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Actual benchmark
    times = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    std_time = (sum([(t - avg_time) ** 2 for t in times]) / len(times)) ** 0.5
    return avg_time * 1000, std_time * 1000


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) for two bounding boxes.

    Args:
        box1 (list or tuple): [x1, y1, x2, y2] of box 1.
        box2 (list or tuple): [x1, y1, x2, y2] of box 2.

    Returns:
        float: IoU value (0.0 - 1.0).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def ccrop_tp(txt_file, model, iou_threshold, viz_folder="visualizations"):
    """
    Evaluate true positives on the CCROP dataset using a list of image paths in a text file.

    Args:
        txt_file (str): Path to the .txt file listing image paths.
        model: Inference model (must support `model(image_path)` call).
        iou_threshold (float): IoU threshold for true positive classification.
        viz_folder (str): Directory to save visualization images.

    Returns:
        float: True positive percentage relative to ground truth labels.
    """
    with open(txt_file, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip().endswith(('.jpg', '.png'))]

    total_ground_truths = total_detections = 0
    true_positives = false_positives = false_negatives = 0

    for image_path in image_paths:
        results = model(image_path)
        r = results[0]

        # Label file
        label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
        if not os.path.exists(label_path):
            print(f"[WARNING] Label file not found for: {image_path}")
            continue

        with open(label_path, 'r') as f:
            gt_lines = f.readlines()

        img = r.orig_img
        h_img, w_img = img.shape[:2]
        gt_data = []

        for line in gt_lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, bw, bh = map(float, parts)
            x1 = int((cx - bw / 2) * w_img)
            y1 = int((cy - bh / 2) * h_img)
            x2 = int((cx + bw / 2) * w_img)
            y2 = int((cy + bh / 2) * h_img)
            gt_data.append({'class': int(cls), 'box': [x1, y1, x2, y2]})

        total_ground_truths += len(gt_data)
        pred_boxes = r.boxes.xyxy.cpu().numpy()
        pred_classes = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, 'cls') else [None] * len(pred_boxes)
        total_detections += len(pred_boxes)

        gt_matched = [False] * len(gt_data)
        pred_matched = [False] * len(pred_boxes)

        for i, (pred_box, pred_cls) in enumerate(zip(pred_boxes, pred_classes)):
            best_iou = 0
            best_match_idx = -1
            for j, gt in enumerate(gt_data):
                if gt_matched[j]:
                    continue
                iou = compute_iou(pred_box, gt['box'])
                if iou > best_iou and pred_cls is not None and int(pred_cls) == gt['class']:
                    best_iou = iou
                    best_match_idx = j
            if best_iou >= iou_threshold and best_match_idx != -1:
                true_positives += 1
                pred_matched[i] = True
                gt_matched[best_match_idx] = True
            else:
                false_positives += 1

        false_negatives += sum(1 for matched in gt_matched if not matched)

        # Visualization
        vis_img = img.copy()
        for i, box in enumerate(pred_boxes):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if pred_matched[i] else (0, 0, 255)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        for i, gt in enumerate(gt_data):
            x1, y1, x2, y2 = gt['box']
            color = (0, 255, 0) if gt_matched[i] else (255, 0, 0)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        img_name = os.path.basename(image_path)
        os.makedirs(viz_folder, exist_ok=True)
        cv2.imwrite(os.path.join(viz_folder, f"vis_{img_name}"), vis_img)

    print(f"True Positive Percentage: {(true_positives / total_ground_truths) * 100:.2f}%")
    return (true_positives / total_ground_truths) * 100 if total_ground_truths > 0 else 0


def dset_tp(image_dir, label_dir, model, iou_threshold, viz_folder="visualizations"):
    """
    Evaluate true positives on a non-CCROP dataset directory.

    Args:
        image_dir (str): Directory containing images.
        label_dir (str): Directory containing label .txt files.
        model: Inference model (must support `model(image_path)` call).
        iou_threshold (float): IoU threshold for true positive classification.
        viz_folder (str): Directory to save visualization images.

    Returns:
        float: True positive percentage relative to ground truth labels.
    """
    image_paths = [
        os.path.join(image_dir, fname)
        for fname in os.listdir(image_dir)
        if fname.lower().endswith(('.jpg', '.png'))
    ]
    image_paths.sort()

    total_ground_truths = total_detections = 0
    true_positives = false_positives = false_negatives = 0

    for image_path in image_paths:
        results = model(image_path)
        r = results[0]

        label_file = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        if not os.path.exists(label_path):
            continue

        with open(label_path, 'r') as f:
            gt_lines = f.readlines()

        img = r.orig_img
        h_img, w_img = img.shape[:2]
        gt_data = []
        for line in gt_lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, bw, bh = map(float, parts)
            x1 = int((cx - bw / 2) * w_img)
            y1 = int((cy - bh / 2) * h_img)
            x2 = int((cx + bw / 2) * w_img)
            y2 = int((cy + bh / 2) * h_img)
            gt_data.append({'class': int(cls), 'box': [x1, y1, x2, y2]})

        total_ground_truths += len(gt_data)
        pred_boxes = r.boxes.xyxy.cpu().numpy()
        pred_classes = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, 'cls') else [None] * len(pred_boxes)
        total_detections += len(pred_boxes)

        gt_matched = [False] * len(gt_data)
        pred_matched = [False] * len(pred_boxes)

        for i, (pred_box, pred_cls) in enumerate(zip(pred_boxes, pred_classes)):
            best_iou = 0
            best_match_idx = -1
            for j, gt in enumerate(gt_data):
                if gt_matched[j]:
                    continue
                iou = compute_iou(pred_box, gt['box'])
                if iou > best_iou and pred_cls is not None and int(pred_cls) == gt['class']:
                    best_iou = iou
                    best_match_idx = j
            if best_iou >= iou_threshold and best_match_idx != -1:
                true_positives += 1
                pred_matched[i] = True
                gt_matched[best_match_idx] = True
            else:
                false_positives += 1

        false_negatives += sum(1 for matched in gt_matched if not matched)

        # Visualization
        vis_img = img.copy()
        for i, box in enumerate(pred_boxes):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if pred_matched[i] else (0, 0, 255)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        for i, gt in enumerate(gt_data):
            x1, y1, x2, y2 = gt['box']
            color = (0, 255, 0) if gt_matched[i] else (255, 0, 0)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        os.makedirs(viz_folder, exist_ok=True)
        img_name = os.path.basename(image_path)
        cv2.imwrite(os.path.join(viz_folder, f"vis_{img_name}"), vis_img)

    print(f"True Positive Percentage: {(true_positives / total_ground_truths) * 100:.2f}%")
    return (true_positives / total_ground_truths) * 100 if total_ground_truths > 0 else 0
