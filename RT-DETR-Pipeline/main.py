import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Rest of your imports
import torch
from ultralytics import RTDETR

if __name__ == '__main__':
    # SETTINGS
    MODEL = 'rtdetr-l'
    DATA = 'camocrops.yaml'
    EPOCHS = 250
    
    BATCH_SIZE = 32
    IMG_SIZE = 320
    WORKERS =2
    VAL = False
    SAVED_PERIOD = 5
    # Initialize model
    OPTIMIZER = 'SGD'  # add more optimizers here later
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
        amp=False,  # Disable AMP as suggested by the warning
        val=VAL,
        save_period=SAVED_PERIOD,
        optimizer=OPTIMIZER
    )