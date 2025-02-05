import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
from ultralytics import RTDETR

def write_results(name, metrics):
    with open('runs/detect/' + name + '/ap50.txt', 'a') as file:
        for ap50 in metrics.box.ap50:
            file.write(str(ap50) + '\n')
        file.write('\n')
        file.write(str(metrics.box.map50))

    with open('runs/detect/' + name + '/maps.txt', 'a') as file:
        for maps in metrics.box.maps:
            file.write(str(maps) + '\n')
        file.write('\n')
        file.write(str(metrics.box.map))

if __name__ == '__main__':
    version = 'train'  # Matches your training output folder
    
    # Evaluation settings
    start_e = 1
    end_e = 250  # Since you trained for 1 epoch
    interval = 5
    
    # Match your training settings
    imgsz = 300  # Same as training
    batch = 1    # Same as training
    
    path = 'runs/detect/' + version + '/weights/'
    
    # Evaluate best.pt and last.pt
    weight_files = ['best.pt', 'last.pt']
    
    for weight_file in weight_files:
        model = RTDETR(path + weight_file)
        name = f"{version}_{weight_file.split('.')[0]}"
        
        # Validation metrics
        val_metrics = model.val(
            data='camocrops.yaml',
            imgsz=imgsz,
            batch=batch,
            device=0,
            save_json=True,
            conf=0.01,
            iou=0.5,
            max_det=50,
            name=name + "_val"
        )
        write_results(name + "_val", val_metrics)
        
        # Test metrics
        test_metrics = model.val(
            data='camocrops.yaml',
            split='test',
            imgsz=imgsz,
            batch=batch,
            device=0,
            name=name + "_test"
        )
        write_results(name + "_test", test_metrics)
