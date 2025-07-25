import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from ultralytics import RTDETR

# Global configuration variables
TOTAL_EPOCHS = 90  # Total number of epochs to train
SAVE_PERIOD = 15   # Save checkpoint every N epochs
VALIDATION_EPOCHS = list(range(SAVE_PERIOD, TOTAL_EPOCHS + 1, SAVE_PERIOD))  # Epochs to validate
DEVICE = 3  # Default GPU device

# Training configuration
TRAIN_CONFIG = {
    'optimizer': 'SGD',
    'momentum': 0.9,
    'lr0': 0.01,
    'lrf': 0.001,
    'batch': 32,
    'val': False,  # Skip validation during training
    'imgsz': 320,
    'weight_decay': 0.0001,
    'save_period': SAVE_PERIOD,
    'device': DEVICE
}

# Dataset configurations
DATASET_CONFIGS = {
    'TomatOD': {
        'yaml': 'tomatod.yaml',
        'project': 'runs/detect/TomatOD_default_results'
    },
    'CamoCrops': {
        'yaml': 'camocrops.yaml',
        'project': 'runs/detect/CamoCrops_default_results'
    },
    'CCrop': {
        'yaml': 'ccrop.yaml',
        'project': 'runs/detect/CCrop_default_results'
    }
}


def train_with_default_rtdetr(dataset_name):
    base_path = DATASET_CONFIGS[dataset_name]['project']
    os.makedirs(base_path, exist_ok=True)
    
    print(f"Starting training for {dataset_name} with default RT-DETR for {TOTAL_EPOCHS} epochs")
    
    # Initialize model with default RT-DETR (no custom backbone)
    model = RTDETR('rtdetr-l.pt')
    
    # Create a training folder name
    training_folder_name = f'rtdetr_default_{TOTAL_EPOCHS}epochs'
    training_folder_path = os.path.join(base_path, training_folder_name)
    
    # Configure training
    config = TRAIN_CONFIG.copy()
    config.update({
        'name': training_folder_name,
        'data': DATASET_CONFIGS[dataset_name]['yaml'],
        'epochs': TOTAL_EPOCHS,
        'project': base_path
    })
    
    # Run training
    results = model.train(**config)
    
    # After training is complete, run validation
    print(f"Training completed for {dataset_name} with default RT-DETR. Running validation...")
    
    # Run validation on the final model
    metrics = model.val(
        data=DATASET_CONFIGS[dataset_name]['yaml'],
        split='test',
        conf=0.01,
        iou=0.5,
        max_det=50,
        project=base_path,
        name=f'val_default_{TOTAL_EPOCHS}epochs',
        save_json=True
    )
    
    # Save validation results
    save_results(metrics, 'default', TOTAL_EPOCHS, base_path)
    
    # Also validate each saved checkpoint
    for epoch in VALIDATION_EPOCHS:
        checkpoint_path = os.path.join(training_folder_path, 'weights', f'epoch{epoch}.pt')
        if os.path.exists(checkpoint_path):
            print(f"Validating {dataset_name} with default RT-DETR checkpoint from epoch {epoch}...")
            
            checkpoint_model = RTDETR(checkpoint_path)
            checkpoint_metrics = checkpoint_model.val(
                data=DATASET_CONFIGS[dataset_name]['yaml'],
                split='test',
                conf=0.01,
                iou=0.5,
                max_det=50,
                project=base_path,
                name=f'val_default_{epoch}epochs',
                save_json=True
            )
            save_results(checkpoint_metrics, 'default', epoch, base_path)
    
    torch.cuda.empty_cache()
    print(f"Completed training and validation for {dataset_name} with default RT-DETR")

def save_results(metrics, model_name, epochs, base_path):
    # Define the CSV file path
    results_file = os.path.join(base_path, 'training_results.csv')
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(results_file)
    
    # Open file in append mode
    with open(results_file, 'a') as f:
        # Write headers if file doesn't exist
        if not file_exists:
            # Create header row
            headers = ['epoch', 'model', 'mAP50', 'mAP50-95']
            
            # Add class-specific headers
            for i in range(len(metrics.box.ap50)):
                headers.append(f'class_{i}_mAP50')
                headers.append(f'class_{i}_mAP50-95')
            
            # Write headers
            f.write(','.join(headers) + '\n')
        
        # Create data row
        row_data = [str(epochs), model_name, f"{metrics.box.map50:.6f}", f"{metrics.box.map:.6f}"]
        
        # Add class-specific metrics
        for i, (ap50, ap) in enumerate(zip(metrics.box.ap50, metrics.box.ap)):
            row_data.append(f"{ap50:.6f}")
            row_data.append(f"{ap:.6f}")
        
        # Write data row
        f.write(','.join(row_data) + '\n')

def run_all_experiments():
    for dataset_name in DATASET_CONFIGS.keys():
        print(f"Starting experiments for {dataset_name} dataset")
        try:
            train_with_default_rtdetr(dataset_name)
        except Exception as e:
            print(f"Error training {dataset_name} with default RT-DETR: {str(e)}")
            continue
        print(f"Completed experiments for {dataset_name} dataset")

if __name__ == "__main__":
    run_all_experiments()