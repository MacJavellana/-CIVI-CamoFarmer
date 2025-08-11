"""RT-DETR Knowledge Distillation Model.

Custom RT-DETR class with knowledge distillation for lightweight object detection.

Classes:
    RTDETRDistillation: Extended RT-DETR class with distillation trainer.
----
"""

from ultralytics.models.rtdetr.model import RTDETR
from ultralytics.models.rtdetr.predict import RTDETRPredictor
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.nn.tasks import RTDETRDetectionModel
from .distill_train import RTDETRDistillationTrainer


class RTDETRDistillation(RTDETR):
    """RT-DETR with Knowledge Distillation.
    
    Extended RT-DETR class supporting teacher-student distillation training.

    Example:
        >>> student = RTDETRDistillation("efficientnet_b0.yaml")
        >>> student.train(data="dataset.yaml", teacher_weights="teacher.pt", epochs=100)
    """
    
    def __init__(self, model="rtdetr-l.pt"):
        """Initialize RT-DETR Distillation Model.

        Args:
            model (str): Model file path or configuration. Default: "rtdetr-l.pt".
        """
        super().__init__(model=model)
    
    @property
    def task_map(self):
        """Define Task Mapping for RT-DETR Distillation.

        Returns:
            dict: Task components with custom distillation trainer.
        """
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": RTDETRDistillationTrainer,
                "model": RTDETRDetectionModel,
            }
        }
