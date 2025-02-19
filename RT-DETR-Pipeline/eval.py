import os
import argparse
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
from ultralytics import RTDETR

# Dataset configurations
DATASET_CONFIGS = {
    'TomatOD': {
        'yaml': 'tomatod.yaml',
        'yaml_combined': 'tomatod_combined.yaml',
        'project': 'runs/detect/TomatOD'
    },
    'CCrop': {
        'yaml': 'ccrop.yaml',
        'yaml_combined': 'ccrop_combined.yaml',
        'project': 'runs/detect/CCrop'
    },
    'CamoCrops': {
        'yaml': 'camocrops.yaml',
        'yaml_combined': 'camocrops_combined.yaml',
        'project': 'runs/detect/CamoCrops'
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

def evaluate_final_model(dataset_config, base_path, imgsz, batch):
    final_folder = os.path.join(base_path, 'final', 'weights')

    if not os.path.exists(final_folder):
        print(f"Final training folder not found at {final_folder}")
        return None
    
    weights_files = [f for f in os.listdir(final_folder) if f.endswith('.pt')]
    if not weights_files:
        print("No weight files found in final folder!")
        return None
    
    latest_weights = max(weights_files, key=lambda x: os.path.getmtime(os.path.join(final_folder, x)))
    final_model_path = os.path.join(final_folder, latest_weights)
    
    print(f"Using model: {final_model_path}")
    model = RTDETR(final_model_path)
    name = "final_evaluation"
    
    print("\nEvaluating Final Model...")
    metrics = model.val(
        data=dataset_config['yaml_combined'],
        split='test',
        imgsz=imgsz,
        batch=batch,
        device=0,
        project=base_path,
        name=name + "_test"
    )
    write_results(os.path.join(base_path, name + "_test"), metrics)
    return metrics



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=DATASET_CONFIGS.keys(),
                       help='Dataset to use: TomatOD, CCrop, or CamoCrops')
    args = parser.parse_args()

    # Settings
    imgsz = 320  # Match the training image size
    batch = 1
    
    # Get dataset specific paths
    dataset_config = DATASET_CONFIGS[args.dataset]
    base_path = dataset_config['project']
    
    # Evaluate Final Model
    test_metrics = evaluate_final_model(dataset_config, base_path, imgsz, batch)
    
    if test_metrics:
        print("\nFinal Evaluation Results:")
        print(f"Test mAP50: {test_metrics.box.map50}")
