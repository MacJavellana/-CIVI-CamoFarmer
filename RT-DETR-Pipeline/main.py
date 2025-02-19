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
    'img_sizes': [320,640],
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
                        torch.cuda.empty_cache()
                        
                    except KeyboardInterrupt:
                        print("\nTraining interrupted! Progress saved.")
                        return best_config, best_map50
                    except Exception as e:
                        print(f"Error with configuration: {e}")
                        continue

                    
    
    return best_config, best_map50

def find_best_configuration(project_dir):
    best_map50 = 0
    best_config = None
    
    if not os.path.exists(project_dir):
        print(f"Project directory {project_dir} not found!")
        return None, 0
    
    # Scan through all tune directories
    for folder in os.listdir(project_dir):
        if folder.startswith('tune_b'):
            try:
                # Parse configuration from folder name
                parts = folder.split('_')
                batch_size = int(parts[1][1:])  # Remove 'b'
                img_size = int(parts[2][1:])    # Remove 'i'
                optimizer = parts[3]
                lr = float(parts[4][2:])        # Remove 'lr'
                
                # Look for results.csv in the folder
                results_path = os.path.join(project_dir, folder, 'results.csv')
                if os.path.exists(results_path):
                    import pandas as pd
                    results = pd.read_csv(results_path)
                    if not results.empty:
                        map50 = results['metrics/mAP50(B)'].max()
                        
                        if map50 > best_map50:
                            best_map50 = map50
                            best_config = {
                                'batch_size': batch_size,
                                'img_size': img_size,
                                'optimizer': optimizer,
                                'learning_rate': lr
                            }
                            print(f"Found better configuration with mAP50: {map50}")
                            print(f"Configuration: {best_config}")
            except Exception as e:
                print(f"Error processing folder {folder}: {e}")
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

    # Base configuration
    base_config = {
        'model': 'rtdetr-l',
        'data': DATASET_CONFIGS[args.dataset]['yaml'],
        'epochs': 10,
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

    # Calculate total expected configurations
    total_expected_configs = (
        len(HYPERPARAMETERS['batch_sizes']) *
        len(HYPERPARAMETERS['img_sizes']) *
        len(HYPERPARAMETERS['optimizers']) *
        len(HYPERPARAMETERS['learning_rates'])
    )

    # Get completed configurations
    completed_configs = get_completed_configs()
    print(f"Total expected configurations: {total_expected_configs}")
    print(f"Completed configurations: {len(completed_configs)}")

    if len(completed_configs) < total_expected_configs:
        # Need to run more hyperparameter tuning
        print("Continuing hyperparameter tuning...")
        best_config, best_map50 = hyperparameter_tuning(base_config)
    
    # All configurations completed, find the best one
    print("All hyperparameter combinations tested. Finding best configuration...")
    best_config, best_map50 = find_best_configuration(base_config['project'])

    print("\nBest Configuration Found:")
    print(f"Batch Size: {best_config['batch_size']}")
    print(f"Image Size: {best_config['img_size']}")
    print(f"Optimizer: {best_config['optimizer']}")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"Best mAP50: {best_map50}")

    


    rtdetr_models = ['rtdetr-l', 'rtdetr-x']

    
    for model_variant in rtdetr_models:
        print(f"\nStarting training with {model_variant}")
        base_config['model'] = model_variant
        
        # Training loop for different epoch amounts
        for epochs in range(15, 91, 15):  # Will run for 15, 30, 45, 60, 75, 90 epochs
            base_config.update({
                'batch': best_config['batch_size'],
                'imgsz': best_config['img_size'],
                'optimizer': best_config['optimizer'],
                'lr': best_config['learning_rate'],
                'name': f'final_{model_variant}_{epochs}epochs',  # Unique name for each model and epoch
                'epochs': epochs,
                'val': False,
                'data': DATASET_CONFIGS[args.dataset]['yaml_combined']
            })

            print(f"\nStarting training for {model_variant} with {epochs} epochs...")
            train_results = final_training(base_config)
            
            # Clear GPU memory after each training run
            torch.cuda.empty_cache()
            print(f"Completed {model_variant} {epochs} epochs training, memory cleared!")
