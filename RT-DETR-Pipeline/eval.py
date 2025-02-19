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

def write_results(name, metrics, model):
    os.makedirs(name, exist_ok=True)
    
    with open(os.path.join(name, 'results.csv'), 'w') as file:
        file.write("Metric,Value\n")
        
        # Model Information
        file.write(f"Model,{model.model.name}\n")
        file.write(f"Version,{model.model.version}\n")
        file.write(f"Parameters,{model.model.num_params}\n")
        file.write(f"GFLOPs,{model.model.flops}\n")
        
        # Training Configuration
        file.write(f"Optimizer,{model.trainer.args.optimizer}\n")
        file.write(f"Batch Size,{model.trainer.args.batch}\n")
        file.write(f"Image Size,{model.trainer.args.imgsz}\n")
        file.write(f"Initial Learning Rate,{model.trainer.args.lr0}\n")
        file.write(f"Final Learning Rate,{model.trainer.args.lrf}\n")
        file.write(f"Total Epochs,{model.trainer.epochs}\n")
        
        # Performance Metrics
        file.write(f"mAP50,{metrics.box.map50:.6f}\n")
        file.write(f"mAP50-95,{metrics.box.map:.6f}\n")
        
        # Per-Class Metrics
        for i, (ap50, ap) in enumerate(zip(metrics.box.ap50, metrics.box.ap)):
            file.write(f"Class_{i}_AP50,{ap50:.6f}\n")
            file.write(f"Class_{i}_AP,{ap:.6f}\n")
        
        # Additional Metrics
        file.write(f"Precision,{metrics.box.p:.6f}\n")
        file.write(f"Recall,{metrics.box.r:.6f}\n")



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
