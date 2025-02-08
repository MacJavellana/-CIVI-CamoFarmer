import os
import argparse
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
torch.cuda.empty_cache()  # Clear GPU memory
torch.backends.cudnn.benchmark = True  # Optimize CUDA operations

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

# Hyperparameter configurations (for future use)
HYPERPARAMETERS = {
    'batch_sizes': [16, 32],
    'img_sizes': [320, 416, 640],
    'optimizers': ['SGD', 'Adam', 'AdamW', 'RMSProp'],
    'learning_rates': [0.0001, 0.001, 0.01]
}

def train_model(model, config):
    return model.train(
        data=config['data'],
        epochs=config['epochs'],
        batch=config['batch'],
        workers=config['workers'],
        imgsz=config['imgsz'],
        device=config['device'],
        cache=config['cache'],
        amp=config['amp'],
        val=config['val'],
        save_period=config['save_period'],
        optimizer=config['optimizer'],
        lr0=config['lr'],
        project=config['project'],
        name=config['name'],
        seed=config['seed']
    )

def final_training(config):
    print("\nStarting final training...")
    config['data'] = DATASET_CONFIGS[args.dataset]['yaml_combined']
    model = RTDETR(config['model'] + '.pt')
    return train_model(model, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=DATASET_CONFIGS.keys(),
                       help='Dataset to use: TomatOD, CCrop, or CamoCrops')
    args = parser.parse_args()

    # Base configuration
    base_config = {
        'model': 'rtdetr-l',
        'data': DATASET_CONFIGS[args.dataset]['yaml'],
        'epochs': 100,
        'batch': 16,
        'workers': 2,
        'imgsz': 320,
        'device': 0,
        'cache': True,
        'amp': False,
        'val': True,
        'save_period': 1,
        'optimizer': 'AdamW',
        'lr': 0.001,
        'project': DATASET_CONFIGS[args.dataset]['project'],
        'name': 'final',
        'seed': 42
    }

    """
    # Phase 1: Hyperparameter Tuning (Commented out for future use)
    print("Starting hyperparameter tuning...")
    best_config, best_map50 = hyperparameter_tuning(base_config)
    
    print("\nBest Configuration Found:")
    print(f"Batch Size: {best_config['batch_size']}")
    print(f"Image Size: {best_config['img_size']}")
    print(f"Optimizer: {best_config['optimizer']}")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"Best mAP50: {best_map50}")
    """

    # Phase 2: Final Training
    train_results = final_training(base_config)
