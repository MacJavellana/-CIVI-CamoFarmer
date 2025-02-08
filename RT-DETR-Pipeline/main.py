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
    'img_sizes': [320, 640],
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

def get_completed_configs():
    completed = set()
    if os.path.exists(base_config['project']):
        for folder in os.listdir(base_config['project']):
            if folder.startswith('tune_b'):
                parts = folder.split('_')
                batch = parts[1][1:]  # Remove 'b'
                img = parts[2][1:]    # Remove 'i'
                opt = parts[3]
                lr = parts[4][2:]     # Remove 'lr'
                completed.add((int(batch), int(img), opt, float(lr)))
    return completed

def save_checkpoint(best_config, best_map50):
    checkpoint = {
        'best_config': best_config,
        'best_map50': best_map50
    }
    checkpoint_path = os.path.join(base_config['project'], 'tuning_checkpoint.pt')
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint():
    checkpoint_path = os.path.join(base_config['project'], 'tuning_checkpoint.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        return checkpoint['best_config'], checkpoint['best_map50']
    return None, 0

def hyperparameter_tuning(base_config):
    best_map50 = 0
    best_config = {}
    completed_configs = get_completed_configs()
    
    total_combinations = (len(HYPERPARAMETERS['batch_sizes']) * 
                         len(HYPERPARAMETERS['img_sizes']) * 
                         len(HYPERPARAMETERS['optimizers']) * 
                         len(HYPERPARAMETERS['learning_rates']))
    
    print(f"Total configurations: {total_combinations}")
    print(f"Already completed: {len(completed_configs)}")
    
    for batch_size in HYPERPARAMETERS['batch_sizes']:
        for img_size in HYPERPARAMETERS['img_sizes']:
            for optimizer in HYPERPARAMETERS['optimizers']:
                for lr in HYPERPARAMETERS['learning_rates']:
                    if (batch_size, img_size, optimizer, lr) in completed_configs:
                        print(f"Skipping completed config: batch={batch_size}, img_size={img_size}, optimizer={optimizer}, lr={lr}")
                        continue
                        
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
                    
                    try:
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
                            
                        save_checkpoint(best_config, best_map50)
                        
                    except KeyboardInterrupt:
                        print("\nTraining interrupted! Progress saved.")
                        return best_config, best_map50
                    except Exception as e:
                        print(f"Error with configuration: {e}")
                        continue
    
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

    """
    # Base configuration
    base_config = {
        'model': 'rtdetr-l',
        'data': DATASET_CONFIGS[args.dataset]['yaml'],
        'epochs': 50,
        'batch': 32,
        'workers': 2,
        'imgsz': 320,
        'device': 0,
        'cache': True,
        'amp': False,
        'val': True,
        'save_period': 1,
        'optimizer': 'SGD',
        'lr': 0.01,
        'project': DATASET_CONFIGS[args.dataset]['project'],
        'name': 'tune'
    }

    # Load previous progress if exists
    best_config, best_map50 = load_checkpoint()
    if best_config:
        print(f"Resuming from previous checkpoint with mAP50: {best_map50}")
    
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
    """

    # best hyperparameter as of 02/08/2025
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

    # Phase 2: Final Training
    train_results = final_training(base_config)
