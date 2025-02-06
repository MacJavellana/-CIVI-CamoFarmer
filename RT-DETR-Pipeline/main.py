import os
import argparse
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
from ultralytics import RTDETR

# Dataset configurations
DATASET_CONFIGS = {
    'TomatOD': {
        'yaml': 'tomatod.yaml',
        'project': 'runs/detect/TomatOD'
    },
    'CCrop': {
        'yaml': 'ccrop.yaml',
        'project': 'runs/detect/CCrop'
    },
    'CamoCrops': {
        'yaml': 'camocrops.yaml',
        'project': 'runs/detect/CamoCrops'
    }
}

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=DATASET_CONFIGS.keys(),
                       help='Dataset to use: TomatOD, CCrop, or CamoCrops')
    args = parser.parse_args()

    # SETTINGS
    MODEL = 'rtdetr-l'
    DATA = DATASET_CONFIGS[args.dataset]['yaml']
    EPOCHS = 1
    
    BATCH_SIZE = 32
    IMG_SIZE = 320
    WORKERS = 2
    VAL = False
    SAVED_PERIOD = 1
    OPTIMIZER = 'SGD'

    # Initialize model
    model = RTDETR(MODEL + '.pt')

    # Train with optimized settings
    train_results = model.train(
        data=DATA,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        workers=WORKERS,
        imgsz=IMG_SIZE,
        device=0,
        cache=True,
        amp=False,
        val=VAL,
        save_period=SAVED_PERIOD,
        optimizer=OPTIMIZER,
        project=DATASET_CONFIGS[args.dataset]['project']
    )