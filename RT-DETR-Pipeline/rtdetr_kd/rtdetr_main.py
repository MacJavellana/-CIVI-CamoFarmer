from ultralytics.models.rtdetr.distill_model import RTDETRDistillation
from ultralytics.utils import LOGGER
import torch
import pandas as pd
from pathlib import Path
import time

# ==================== GLOBAL VARIABLES ====================

# Model configurations to test
MODEL_CONFIGS = [
    "efficientnet_b0.yaml",
    "mobilenet_v2.yaml", 
    "shufflenet_v2_x1_0.yaml"
]



# Dataset configurations with corresponding teacher weights
DATASET_CONFIGS = [
    {
        "data": "tomatod.yaml",
        "name": "tomatod",
        "teacher_weights": "pt_files/rtdetr_tomatod_best_90.pt"
    },
    {
        "data": "camocrops.yaml", 
        "name": "camocrops",
        "teacher_weights": "pt_files/rtdetr_camocrops_best_90.pt"
    },
    {
        "data": "ccrop.yaml",
        "name": "ccrop", 
        "teacher_weights": "pt_files/rtdetr_ccrop_best_90.pt"
    }
]


# Training hyperparameters
TRAINING_PARAMS = {
    'epochs': 90,
    'imgsz': 320,
    'batch': 16,
    'lr0': 0.01,
    'lrf': 0.0001,
    'optimizer': 'SGD',
    'device': 0 if torch.cuda.is_available() else 'cpu',
    'project': 'RTDETR_KD',
    'save_period': 15,  # Save every 5 epochs
    'val': True,
    'plots': True,
    'patience': 50,
    'save': True,
    'exist_ok': True,
    # Fixed distillation hyperparameters
    'distill_weight': 0.1,
    'temperature': 2.0,
}

# ==================== HELPER FUNCTIONS ====================

def get_model_name(yaml_file):
    """Extract model name from yaml filename."""
    return yaml_file.replace('.yaml', '')

def get_epoch_weights(weights_dir, save_period, total_epochs):
    """Get all epoch weight files that should exist based on save_period."""
    weights_path = Path(weights_dir)
    epoch_weights = []
    
    # Check for epoch weights (every save_period epochs)
    for epoch in range(save_period, total_epochs + 1, save_period):
        epoch_file = weights_path / f"epoch{epoch}.pt"
        if epoch_file.exists():
            epoch_weights.append((epoch, str(epoch_file)))
    
    # Also check for best.pt and last.pt
    best_file = weights_path / "best.pt"
    last_file = weights_path / "last.pt"
    
    if best_file.exists():
        epoch_weights.append(("best", str(best_file)))
    if last_file.exists():
        epoch_weights.append(("last", str(last_file)))
    
    return epoch_weights

def test_model_weights(model_config, weight_path, epoch_identifier, dataset_config):
    """Test a specific model weight file and return metrics."""
    try:
        model_name = get_model_name(model_config)
        dataset_name = dataset_config["name"]
        
        # Load model with trained weights
        model = RTDETRDistillation(weight_path)
        
        # Run validation/testing on the same dataset it was trained on
        metrics = model.val(data=dataset_config["data"], split='test')
        
        # Extract metrics
        final_map = metrics.box.map if hasattr(metrics, 'box') and hasattr(metrics.box, 'map') else 0.0
        final_map50 = metrics.box.map50 if hasattr(metrics, 'box') and hasattr(metrics.box, 'map50') else 0.0
        final_map75 = metrics.box.map75 if hasattr(metrics, 'box') and hasattr(metrics.box, 'map75') else 0.0
        
        # Create result dictionary
        result = {
            'dataset': dataset_name,
            'backbone': model_name,
            'epoch': epoch_identifier,
            'distill_weight': TRAINING_PARAMS['distill_weight'],
            'temperature': TRAINING_PARAMS['temperature'],
            'batch_size': TRAINING_PARAMS['batch'],
            'learning_rate': TRAINING_PARAMS['lr0'],
            'lrf': TRAINING_PARAMS['lrf'],
            'optimizer': TRAINING_PARAMS['optimizer'],
            'teacher_weights': dataset_config["teacher_weights"],
            'mAP': final_map,
            'mAP50': final_map50,
            'mAP75': final_map75,
            'weight_path': weight_path,
            'status': 'success'
        }
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return result
        
    except Exception as e:
        return {
            'dataset': dataset_config["name"],
            'backbone': get_model_name(model_config),
            'epoch': epoch_identifier,
            'distill_weight': TRAINING_PARAMS['distill_weight'],
            'temperature': TRAINING_PARAMS['temperature'],
            'batch_size': TRAINING_PARAMS['batch'],
            'learning_rate': TRAINING_PARAMS['lr0'],
            'lrf': TRAINING_PARAMS['lrf'],
            'optimizer': TRAINING_PARAMS['optimizer'],
            'teacher_weights': dataset_config["teacher_weights"],
            'mAP': 0.0,
            'mAP50': 0.0,
            'mAP75': 0.0,
            'weight_path': weight_path,
            'status': f'failed: {str(e)[:100]}'
        }

def save_test_results(results, dataset_name, model_name, epoch):
    """Save test results to organized folder structure using absolute paths."""
    
    # Create folder structure using absolute path: RTDETR_KD/dataset/backbone/train_results/
    result_dir = Path.cwd() / "RTDETR_KD" / dataset_name / model_name / "train_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Save/append to single CSV file
    csv_path = result_dir / "training_results.csv"
    df = pd.DataFrame([results])
    
    # Append to existing file or create new one
    if csv_path.exists():
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    
    return csv_path

def check_teacher_weights():
    """Check if all teacher weight files exist."""
    missing_files = []
    for dataset_config in DATASET_CONFIGS:
        teacher_path = Path(dataset_config["teacher_weights"])
        if not teacher_path.exists():
            missing_files.append(str(teacher_path))
    
    if missing_files:
        for file in missing_files:
            print(f"Missing teacher weight file: {file}")
        return False
    else:
        return True

def train_model_with_fixed_hyperparameters(model_config, dataset_config):
    """Train model with fixed hyperparameters."""
    
    model_name = get_model_name(model_config)
    dataset_name = dataset_config["name"]
    
    print(f"Training {model_name} on {dataset_name} with fixed hyperparameters...")
    
    try:
        # Initialize student model
        student_model = RTDETRDistillation(model_config)
        
        # Create training arguments with fixed hyperparameters
        train_args = TRAINING_PARAMS.copy()
        train_args.update({
            'data': dataset_config["data"],
            'name': f'{dataset_name}/{model_name}/train_results',
            'teacher_weights': dataset_config["teacher_weights"],
        })
        
        # Start training
        start_time = time.time()
        training_results = student_model.train(**train_args)
        training_time = time.time() - start_time
        
        # ============ TESTING PHASE ============
        
        # Get weights directory using absolute path
        weights_dir = Path.cwd() / train_args['project'] / train_args['name'] / 'weights'
        
        # Get all epoch weights to test
        epoch_weights = get_epoch_weights(weights_dir, TRAINING_PARAMS['save_period'], TRAINING_PARAMS['epochs'])
        
        final_results = []
        
        # Test each epoch weight on the same dataset
        for epoch_id, weight_path in epoch_weights:
            
            # Test the model on the same dataset it was trained on
            test_result = test_model_weights(
                model_config, weight_path, epoch_id, dataset_config
            )
            
            # Add training info to test result
            test_result['training_time_minutes'] = training_time / 60
            
            # Save individual test result in organized structure
            save_test_results(test_result, dataset_name, model_name, epoch_id)
            
            # Add to results
            final_results.append(test_result)
        
        # Clean up training model
        del student_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"? Training completed for {model_name} on {dataset_name}")
        return final_results
        
    except Exception as e:
        print(f"? Training failed for {model_name} on {dataset_name}: {str(e)}")
        
        # Clean up on failure
        try:
            del student_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except:
            pass
        
        return []

# ==================== MAIN TRAINING AND TESTING PIPELINE ====================

def train_and_test_all_combinations():
    """Train all model configurations with fixed hyperparameters."""
    
    # Check teacher weights first
    if not check_teacher_weights():
        raise FileNotFoundError("Missing teacher weight files. Please check the paths.")
    
    all_results = []
    total_combinations = len(MODEL_CONFIGS) * len(DATASET_CONFIGS)
    current_combination = 0
    
    print(f"?? STARTING TRAINING WITH FIXED HYPERPARAMETERS")
    print(f"?? Total model-dataset combinations: {total_combinations}")
    print(f"?? Fixed hyperparameters:")
    print(f"   - Distill weight: {TRAINING_PARAMS['distill_weight']}")
    print(f"   - Temperature: {TRAINING_PARAMS['temperature']}")
    print(f"   - Batch size: {TRAINING_PARAMS['batch']}")
    print(f"   - Learning rate: {TRAINING_PARAMS['lr0']}")
    print(f"   - LRF: {TRAINING_PARAMS['lrf']}")
    print(f"   - Optimizer: {TRAINING_PARAMS['optimizer']}")
    print(f"   - Image size: {TRAINING_PARAMS['imgsz']}")
    print(f"   - Epochs: {TRAINING_PARAMS['epochs']}")
    print("=" * 100)
    
    # Iterate through all model-dataset combinations
    for model_config in MODEL_CONFIGS:
        for dataset_config in DATASET_CONFIGS:
            current_combination += 1
            model_name = get_model_name(model_config)
            dataset_name = dataset_config["name"]
            
            print(f"\n?? COMBINATION {current_combination}/{total_combinations}")
            print(f"?? Model: {model_name}")
            print(f"?? Dataset: {dataset_name}")
            print(f"?? Teacher Weights: {dataset_config['teacher_weights']}")
            print("=" * 100)
            
            # Train with fixed hyperparameters
            final_results = train_model_with_fixed_hyperparameters(model_config, dataset_config)
            
            if final_results:
                all_results.extend(final_results)
            else:
                # Add failed result
                failed_result = {
                    'dataset': dataset_name,
                    'backbone': model_name,
                    'epoch': 'training_failed',
                    'distill_weight': TRAINING_PARAMS['distill_weight'],
                    'temperature': TRAINING_PARAMS['temperature'],
                    'batch_size': TRAINING_PARAMS['batch'],
                    'learning_rate': TRAINING_PARAMS['lr0'],
                    'lrf': TRAINING_PARAMS['lrf'],
                    'optimizer': TRAINING_PARAMS['optimizer'],
                    'teacher_weights': dataset_config["teacher_weights"],
                    'mAP': 0.0,
                    'mAP50': 0.0,
                    'mAP75': 0.0,
                    'weight_path': 'N/A',
                    'training_time_minutes': 0.0,
                    'status': 'training_failed'
                }
                all_results.append(failed_result)
    
    return all_results

def create_final_summary(all_results):
    """Create a final summary of all training results in single CSV files."""
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save complete results using absolute path
    summary_dir = Path.cwd() / "RTDETR_KD" / "final_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Single comprehensive CSV with all results
    complete_results_path = summary_dir / "all_training_results.csv"
    df.to_csv(complete_results_path, index=False)
    
    # Filter successful results
    successful_df = df[df['status'] == 'success'].copy()
    
    if len(successful_df) == 0:
        return
    
    # Save only successful results
    successful_results_path = summary_dir / "successful_results_only.csv"
    successful_df.to_csv(successful_results_path, index=False)
    
    # Save best results summary (one row per model-dataset combination)
    best_per_combo = successful_df.loc[successful_df.groupby(['backbone', 'dataset'])['mAP'].idxmax()][
        ['backbone', 'dataset', 'epoch', 'distill_weight', 'temperature', 'batch_size', 'learning_rate', 'lrf', 'optimizer', 'mAP', 'mAP50', 'mAP75', 'training_time_minutes']
    ]
    best_per_combo.to_csv(summary_dir / "best_results_summary.csv", index=False)

    print(f"?? Results saved:")
    print(f"   - all_training_results.csv (complete dataset)")
    print(f"   - successful_results_only.csv (filtered)")
    print(f"   - best_results_summary.csv (best per model-dataset combo)")

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("?? Starting RT-DETR Knowledge Distillation Training with Fixed Hyperparameters")
    
    try:
        # Run training pipeline
        all_results = train_and_test_all_combinations()
        
        # Create final summary
        create_final_summary(all_results)
        
        print("?? Training pipeline completed!")
        print(f"?? Total experiments: {len(all_results)}")
        print("?? Results saved in: RTDETR_KD/")
        
    except Exception as e:
        print(f"? Process failed: {str(e)}")
        import traceback
        traceback.print_exc()