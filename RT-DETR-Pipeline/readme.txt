how RT-DETR pipeline works
main.py Pipeline:
    Phase 1: Hyperparameter Tuning
        Tests different combinations of batch sizes, image sizes, optimizers, and learning rates
        Uses regular dataset split (train/val/test)
        Validates each configuration and tracks best performing parameters
        Saves all models in runs/detect/[dataset]/weights/
    Phase 2: Final Training
        Uses the best parameters found during tuning
        Trains on combined dataset (train+val) using [dataset]_combined.yaml
        Saves final model as 'final.pt' in runs/detect/[dataset]/weights/

eval.py Pipeline:
    Loads the final trained model (final.pt)
    Uses the combined dataset configuration
    Evaluates only on test set
    Generates performance metrics
    Saves results in runs/detect/[dataset]/final_evaluation_test/

    