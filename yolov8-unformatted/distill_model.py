# distill_model.py

from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.detect.val     import DetectionValidator
from ultralytics.nn.tasks                   import DetectionModel
from distill_train                          import YOLOv8DistillationTrainer

class YOLOv8Distillation(YOLO):
    """
    YOLOv8 with feature-based knowledge distillation support.

    .train(...) will use our custom
    YOLOv8DistillationTrainer and the correct detect
    validator/predictor (with get_dataloader implemented).
    """
    def __init__(self, model='yolov8n.yaml'):
        super().__init__(model=model)

    @property
    def task_map(self):
        return {
            'detect': {
                'predictor': DetectionPredictor,
                'validator': DetectionValidator,
                'trainer':   YOLOv8DistillationTrainer,
                'model':     DetectionModel,
            }
        }
