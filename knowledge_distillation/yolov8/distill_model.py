"""YOLOv8 Knowledge Distillation Model.

Custom YOLOv8 class with feature-based knowledge distillation for model compression.

Classes:
    YOLOv8Distillation: Extended YOLO class with distillation trainer.
----
"""

from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.detect.val     import DetectionValidator
from ultralytics.nn.tasks                   import DetectionModel
from .distill_train                          import YOLOv8DistillationTrainer

class YOLOv8Distillation(YOLO):
    """YOLOv8 with Knowledge Distillation.
    
    Extended YOLO class supporting teacher-student distillation training.

    Example:
        >>> model = YOLOv8Distillation('yolov8n.yaml')
        >>> model.train(data='coco.yaml', teacher_model='yolov8l.pt', epochs=100)
    """
    
    def __init__(self, model='yolov8n.yaml'):
        """Initialize Distillation Model.

        Args:
            model (str): Model config path or architecture name. Default: 'yolov8n.yaml'.
        """
        super().__init__(model=model)

    @property
    def task_map(self):
        """Define Task Mapping for Distillation.

        Returns:
            dict: Task components with custom distillation trainer.
        """
        return {
            'detect': {
                'predictor': DetectionPredictor,
                'validator': DetectionValidator,
                'trainer':   YOLOv8DistillationTrainer,
                'model':     DetectionModel,
            }
        }
