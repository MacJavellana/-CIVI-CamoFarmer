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

# Hyperparameter configurations
HYPERPARAMETERS = {
    'batch_sizes': [16, 32],
    'img_sizes': [320, 416, 640],
    'optimizers': ['SGD', 'Adam', 'AdamW', 'RMSProp'],
    'learning_rates': [0.001, 0.01, 0.1]
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
        name=config['name']
    )

def hyperparameter_tuning(base_config):
    best_map50 = 0
    best_config = {}
    
    for batch_size in HYPERPARAMETERS['batch_sizes']:
        for img_size in HYPERPARAMETERS['img_sizes']:
            for optimizer in HYPERPARAMETERS['optimizers']:
                for lr in HYPERPARAMETERS['learning_rates']:
                    config = base_config.copy()
                    config.update({
                        'batch': batch_size,
                        'imgsz': img_size,
                        'optimizer': optimizer,
                        'lr': lr,
                        'name': f'tune_b{batch_size}_i{img_size}_{optimizer}_lr{lr}',
                        'val': True
                    })
                    
                    print(f"\nTrying configuration: batch={batch_size}, img_size={img_size}, optimizer={optimizer}, lr={lr}")
                    
                    model = RTDETR(config['model'] + '.pt')
                    results = train_model(model, config)
                    
                    if results.box.map50 > best_map50:
                        best_map50 = results.box.map50
                        best_config = {
                            'batch_size': batch_size,
                            'img_size': img_size,
                            'optimizer': optimizer,
                            'learning_rate': lr
                        }
                        print(f"New best configuration found! mAP50: {best_map50}")
    
    return best_config, best_map50

def final_training(config):
    print("\nStarting final training with best parameters...")
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
        'epochs': 1,
        'batch': 32,
        'workers': 2,
        'imgsz': 320,
        'device': 0,
        'cache': True,
        'amp': False,
        'val': True,  # Always use validation for tuning
        'save_period': 1,
        'optimizer': 'SGD',
        'lr': 0.01,
        'project': DATASET_CONFIGS[args.dataset]['project'],
        'name': 'tune'
    }

    # Phase 1: Hyperparameter Tuning
    print("Starting hyperparameter tuning...")
    best_config, best_map50 = hyperparameter_tuning(base_config)
    
    print("\nBest Configuration Found:")
    print(f"Batch Size: {best_config['batch_size']}")
    print(f"Image Size: {best_config['img_size']}")
    print(f"Optimizer: {best_config['optimizer']}")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"Best mAP50: {best_map50}")
    
    # Update config with best parameters
    base_config.update({
        'batch': best_config['batch_size'],
        'imgsz': best_config['img_size'],
        'optimizer': best_config['optimizer'],
        'lr': best_config['learning_rate'],
        'name': 'final'
    })

    # Phase 2: Final Training
    train_results = final_training(base_config)
