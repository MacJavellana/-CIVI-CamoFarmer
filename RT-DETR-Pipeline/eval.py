import os
import argparse
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
from ultralytics import RTDETR

# Dataset configurations
DATASET_CONFIGS = {
    'TomatOD': {
        'yaml': 'tomatod.yaml',
        'project': 'runs/TomatOD'
    },
    'CCrop': {
        'yaml': 'ccrop.yaml',
        'project': 'runs/CCrop'
    },
    'CamoCrops': {
        'yaml': 'camocrops.yaml',
        'project': 'runs/CamoCrops'
    }
}

def write_results(name, metrics):
    with open(name + '/ap50.txt', 'a') as file:
        for ap50 in metrics.box.ap50:
            file.write(str(ap50) + '\n')
        file.write('\n')
        file.write(str(metrics.box.map50))

    with open(name + '/maps.txt', 'a') as file:
        for maps in metrics.box.maps:
            file.write(str(maps) + '\n')
        file.write('\n')
        file.write(str(metrics.box.map))

def evaluate_model(model, dataset_config, split, imgsz, batch, name, base_path):
    metrics = model.val(
        data=dataset_config['yaml'],
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=0,
        project=base_path,
        name=name + f"_{split}"
    )
    write_results(os.path.join(base_path, name + f"_{split}"), metrics)
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=DATASET_CONFIGS.keys(),
                       help='Dataset to use: TomatOD, CCrop, or CamoCrops')
    parser.add_argument('--start_epoch', type=int, default=1, help='Start epoch for evaluation')
    parser.add_argument('--end_epoch', type=int, default=5, help='End epoch for evaluation')
    parser.add_argument('--interval', type=int, default=1, help='Epoch interval for evaluation')
    args = parser.parse_args()

    # Settings
    imgsz = 300
    batch = 1
    
    # Get dataset specific paths
    dataset_config = DATASET_CONFIGS[args.dataset]
    base_path = dataset_config['project']
    weights_path = f"{base_path}/weights/"
    
    # Create list of weight files for epochs
    epoch_weights = [f'epoch{e}.pt' for e in range(args.start_epoch, args.end_epoch + 1, args.interval)]
    
    # Dictionary to store validation results
    val_results = {}
    
    print("Starting Validation Phase...")
    # First phase: Validation across all epochs
    for weight_file in epoch_weights:
        if not os.path.exists(os.path.join(weights_path, weight_file)):
            print(f"Skipping {weight_file} - file not found")
            continue
            
        model = RTDETR(os.path.join(weights_path, weight_file))
        name = f"{args.dataset}_{weight_file.split('.')[0]}"
        
        print(f"\nValidating {weight_file}")
        metrics = evaluate_model(model, dataset_config, 'val', imgsz, batch, name, base_path)
        val_results[weight_file] = metrics.box.map50  # Store mAP50 for comparison
    
    # Find best performing model from validation
    best_weight = max(val_results.items(), key=lambda x: x[1])[0]
    print(f"\nBest performing model from validation: {best_weight}")
    print(f"Validation mAP50: {val_results[best_weight]}")
    
    print("\nStarting Test Phase...")
    # Second phase: Test using best model
    best_model = RTDETR(os.path.join(weights_path, best_weight))
    name = f"{args.dataset}_{best_weight.split('.')[0]}"
    
    print(f"Testing with best model: {best_weight}")
    test_metrics = evaluate_model(best_model, dataset_config, 'test', imgsz, batch, name, base_path)
    
    print("\nEvaluation Complete!")
    print(f"Best model: {best_weight}")
    print(f"Final Test mAP50: {test_metrics.box.map50}")
