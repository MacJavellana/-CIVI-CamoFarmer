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
    'batch_sizes': [16],
    'img_sizes': [320, 400],
    'optimizers': ['SGD', 'Adam', 'AdamW', 'RMSProp'],
    'learning_rates': [0.001, 0.01, 0.1]
}

def create_epoch_logger(output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("Training Results Log\n")
        f.write("===================\n\n")
    
    def log_epoch(trainer):
        epoch = trainer.epoch
        metrics = trainer.metrics
        with open(output_file, 'a') as f:
            f.write(f"Epoch {epoch}:\n")
            f.write(f"Loss: {metrics.get('train/box_loss', 'N/A'):.6f}\n")
            f.write(f"mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.6f}\n")
            f.write("-----------------\n")
    return log_epoch

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
        callbacks=config.get('callbacks', None)
    )

def get_completed_configs():
    completed = set()
    if os.path.exists(base_config['project']):
        for folder in os.listdir(base_config['project']):
            if folder.startswith('tune_b'):
                parts = folder.split('_')
                batch = parts[1][1:]
                img = parts[2][1:]
                opt = parts[3]
                lr = parts[4][2:]
                completed.add((int(batch), int(img), opt, float(lr)))
    return completed

def find_best_configuration(project_dir):
    best_map50 = 0
    best_config = None
    
    if not os.path.exists(project_dir):
        print(f"Project directory {project_dir} not found!")
        return None, 0
    
    for folder in os.listdir(project_dir):
        if folder.startswith('tune_b'):
            try:
                parts = folder.split('_')
                batch_size = int(parts[1][1:])
                img_size = int(parts[2][1:])
                optimizer = parts[3]
                lr = float(parts[4][2:])
                
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
                        
                        torch.cuda.empty_cache()
                        
                    except KeyboardInterrupt:
                        print("\nTraining interrupted! Progress saved.")
                        return best_config, best_map50
                    except Exception as e:
                        print(f"Error with configuration: {e}")
                        continue
    
    return best_config, best_map50

def final_training(config):
    print("\nStarting final training with best parameters...")
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
        'epochs': 3,
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
        print("Continuing hyperparameter tuning...")
        best_config, best_map50 = hyperparameter_tuning(base_config)
    
    print("All hyperparameter combinations tested. Finding best configuration...")
    best_config, best_map50 = find_best_configuration(base_config['project'])

    print("\nBest Configuration Found:")
    print(f"Batch Size: {best_config['batch_size']}")
    print(f"Image Size: {best_config['img_size']}")
    print(f"Optimizer: {best_config['optimizer']}")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"Best mAP50: {best_map50}")

    # Update config with best parameters for final training
    output_file = os.path.join(base_config['project'], 'final', 'epoch_results.txt')
    base_config.update({
        'batch': best_config['batch_size'],
        'imgsz': best_config['img_size'],
        'optimizer': best_config['optimizer'],
        'lr': best_config['learning_rate'],
        'name': 'final',
        'epochs': 100,
        'val': False,
        'data': DATASET_CONFIGS[args.dataset]['yaml_combined'],
        'callbacks': {'on_train_epoch_end': create_epoch_logger(output_file)}
    })

    # Phase 2: Final Training
    print("\nStarting final training with best parameters...")
    train_results = final_training(base_config)
