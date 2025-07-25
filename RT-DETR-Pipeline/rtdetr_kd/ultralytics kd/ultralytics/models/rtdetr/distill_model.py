from ultralytics.models.rtdetr.model import RTDETR
from ultralytics.models.rtdetr.predict import RTDETRPredictor
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.nn.tasks import RTDETRDetectionModel
from .distill_train import RTDETRDistillationTrainer


class RTDETRDistillation(RTDETR):
    """
    RT-DETR model with knowledge distillation support.
    
    This class extends the standard RT-DETR model to support knowledge distillation
    training with a teacher-student setup.
    
    Examples:
        >>> # Initialize student model with lightweight backbone
        >>> student = RTDETRDistillation("efficientnet_b0.yaml")
        >>> # Train with distillation
        >>> results = student.train(
        ...     data="custom_dataset.yaml",
        ...     teacher_weights="teacher_model.pt",
        ...     distill_weight=0.5,
        ...     temperature=4.0,
        ...     epochs=100
        ... )
    """
    
    def __init__(self, model="rtdetr-l.pt"):
        """
        Initialize the RT-DETR distillation model.
        
        Args:
            model (str): Path to the model file or model configuration.
        """
        super().__init__(model=model)
    
    @property
    def task_map(self):
        """Return task map with distillation trainer."""
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": RTDETRDistillationTrainer,
                "model": RTDETRDetectionModel,
            }
        }
